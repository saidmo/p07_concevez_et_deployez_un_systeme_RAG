"""
main.py — Couche API FastAPI
────────────────────────────────────────────────────────────────────────────
Enveloppe HTTP autour de la chaîne RAG (scripts/rag_chain.py).
Ne contient AUCUNE logique métier : validation des entrées/sorties,
gestion des erreurs HTTP, cycle de vie de l'application, orchestration
de la reconstruction de l'index.

Endpoints :
    GET  /health          → état de l'application et de la chaîne
    POST /ask             → question → réponse RAG + sources + timings
    POST /rebuild         → reconstruction de l'index FAISS (asynchrone,
                            protégé par le header X-API-Key)
    GET  /rebuild/status  → suivi de la reconstruction en cours / passée

Lancement (géré par le Dockerfile) :
    uvicorn main:app --host 0.0.0.0 --port 8000
────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from scripts.rag_chain import get_chain, reset_chain
from scripts.build_index import run_build
from scripts.clean_data import clean_pipeline, save_clean

log = logging.getLogger(__name__)

# ── Configuration (alignée sur docker-compose.yml, fallback exécution locale) ─

API_KEY         = os.getenv("API_KEY", "")
RAW_FILE        = Path(os.getenv("RAW_DATA_PATH", "data/raw")) / "evenements_idf.json"
CLEAN_DATA_PATH = Path(os.getenv("CLEAN_DATA_PATH", "data/processed/evenements_idf_clean.json"))
INDEX_DIR       = Path(os.getenv("FAISS_INDEX_PATH", "data/faiss_index"))


# ── Cycle de vie : charger la chaîne AU DÉMARRAGE ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise la chaîne RAG au démarrage du serveur (chargements lourds :
    SBERT ~1 Go, index FAISS, événements) plutôt qu'à la première requête.
    Si l'index n'existe pas encore (build_index.py non exécuté),
    l'application démarre quand même — /ask renverra une 503 explicite
    et POST /rebuild permettra de construire l'index.
    """
    try:
        get_chain()
        app.state.chain_ready = True
        log.info("Chaîne RAG initialisée — API prête")
    except Exception as exc:
        app.state.chain_ready = False
        app.state.chain_error = str(exc)
        log.error(f"Chaîne RAG indisponible : {exc}")
    yield


app = FastAPI(
    title="RAG p07 — Événements culturels Île-de-France",
    description="Assistant conversationnel basé sur les données OpenAgenda "
                "(SBERT + FAISS + API Mistral)",
    version="1.1.0",
    lifespan=lifespan,
)


# ── Sécurité : header X-API-Key pour les endpoints sensibles ─────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(provided_key: str = Security(api_key_header)) -> None:
    """
    Protège les endpoints sensibles (/rebuild).
    - API_KEY absente de l'environnement → endpoint désactivé (503),
      on refuse d'exposer une reconstruction non protégée.
    - Header absent ou invalide → 401 / 403.
    """
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Endpoint désactivé : la variable d'environnement API_KEY "
                   "n'est pas configurée (voir .env).",
        )
    if not provided_key:
        raise HTTPException(status_code=401, detail="Header X-API-Key manquant.")
    if provided_key != API_KEY:
        raise HTTPException(status_code=403, detail="Clé API invalide.")


# ── État de la reconstruction (partagé entre threads) ────────────────────────

_rebuild_lock = threading.Lock()      # une seule reconstruction à la fois
_rebuild_state: dict = {
    "status"     : "idle",            # idle | running | done | failed
    "params"     : None,
    "started_at" : None,
    "finished_at": None,
    "stats"      : None,              # {"nb_events", "nb_chunks", "duration_s"}
    "error"      : None,
}


def _execute_rebuild(limit: int | None, clean_first: bool) -> None:
    """
    Corps de la reconstruction — exécuté dans un thread d'arrière-plan.

    Étapes :
      1. Décharger la chaîne RAG (libère SBERT + index + événements —
         évite d'avoir DEUX instances de SBERT en RAM, celle de la
         chaîne et celle de la vectorisation, sur une VM contrainte).
         Pendant la reconstruction, /ask répond 503 « reconstruction en cours ».
      2. Nettoyage des données brutes si demandé (clean_first=True)
         ou si le fichier nettoyé n'existe pas encore.
      3. Chunking + vectorisation SBERT + index FAISS + manifest (run_build).
      4. Recharger la chaîne sur le nouvel index.
    """
    global _rebuild_state
    try:
        # 1. Libérer la mémoire de la chaîne actuelle
        app.state.chain_ready = False
        app.state.chain_error = "Reconstruction de l'index en cours (voir /rebuild/status)"
        reset_chain()

        # 2. Nettoyage si nécessaire
        if clean_first or not CLEAN_DATA_PATH.exists():
            log.info(f"Nettoyage des données brutes : {RAW_FILE}")
            cleaned, _stats = clean_pipeline(RAW_FILE)
            save_clean(cleaned, CLEAN_DATA_PATH)

        # 3. Reconstruction de l'index
        stats = run_build(
            input_path=CLEAN_DATA_PATH,
            output_dir=INDEX_DIR,
            limit=limit,
        )

        _rebuild_state.update(
            status="done",
            finished_at=datetime.now(timezone.utc).isoformat(),
            stats=stats,
        )
        log.info(f"Reconstruction terminée : {stats}")

    except Exception as exc:
        log.exception("Échec de la reconstruction de l'index")
        _rebuild_state.update(
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )

    finally:
        # 4. (Re)charger la chaîne — sur le nouvel index si succès, sur
        # l'ancien si la reconstruction a échoué avant la sauvegarde
        try:
            get_chain()
            app.state.chain_ready = True
            log.info("Chaîne RAG rechargée")
        except Exception as exc:
            app.state.chain_ready = False
            app.state.chain_error = str(exc)
            log.error(f"Chaîne RAG indisponible après reconstruction : {exc}")
        _rebuild_lock.release()


# ── Schémas Pydantic (contrat d'API) ─────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500,
        description="Question en langage naturel",
        examples=["Un concert gratuit à Paris ce week-end ?"],
    )


class Source(BaseModel):
    uid: str
    title: str
    city: str
    date_label: str
    price: str
    url: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    contexts: list[str]      # texte des événements fournis au LLM (transparence + Ragas)
    timings: dict[str, float]


class RebuildRequest(BaseModel):
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Limiter le nombre d'événements indexés (test rapide). "
                    "None = corpus complet (~20-40 min sur CPU).",
        examples=[200],
    )
    clean_first: bool = Field(
        default=False,
        description="Relancer le nettoyage des données brutes avant "
                    "l'indexation (utile après une nouvelle collecte). "
                    "Automatique si le fichier nettoyé n'existe pas.",
    )


class RebuildAccepted(BaseModel):
    message: str
    status_url: str = "/rebuild/status"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """État de l'application — utilisé par le healthcheck Docker."""
    ready = getattr(app.state, "chain_ready", False)
    body = {"status": "ok" if ready else "degraded", "chain_ready": ready}
    if not ready:
        body["detail"] = getattr(app.state, "chain_error", "chaîne non initialisée")
    return body


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Pose une question à l'assistant.
    Génération via l'API Mistral : quelques secondes par question.
    """
    if not getattr(app.state, "chain_ready", False):
        raise HTTPException(
            status_code=503,
            detail=getattr(
                app.state, "chain_error",
                "Chaîne RAG indisponible — l'index FAISS a-t-il été "
                "construit ? (python scripts/build_index.py ou POST /rebuild)",
            ),
        )
    try:
        return get_chain().ask(request.question)
    except Exception as exc:
        log.exception("Erreur pendant le traitement de la question")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/rebuild",
    response_model=RebuildAccepted,
    status_code=202,
    dependencies=[Depends(verify_api_key)],
)
def rebuild(request: RebuildRequest):
    """
    Reconstruit l'index vectoriel FAISS à partir des données.

    Opération LONGUE (~20-40 min sur CPU pour le corpus complet) :
    elle est lancée en arrière-plan et cette route répond immédiatement
    202 Accepted. Suivre l'avancement via GET /rebuild/status.

    Pendant la reconstruction, /ask répond 503 (la chaîne est déchargée
    pour libérer la mémoire nécessaire à la vectorisation).

    Protégé par le header X-API-Key.
    """
    if not _rebuild_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Une reconstruction est déjà en cours (voir /rebuild/status).",
        )

    _rebuild_state.update(
        status="running",
        params={"limit": request.limit, "clean_first": request.clean_first},
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
        stats=None,
        error=None,
    )

    # Le verrou est libéré par _execute_rebuild (bloc finally)
    threading.Thread(
        target=_execute_rebuild,
        args=(request.limit, request.clean_first),
        daemon=True,
        name="rebuild-index",
    ).start()

    return RebuildAccepted(
        message="Reconstruction lancée en arrière-plan "
                "(~20-40 min sur CPU pour le corpus complet).",
    )


@app.get("/rebuild/status")
def rebuild_status():
    """
    État de la dernière reconstruction :
    idle (jamais lancée), running, done (avec stats) ou failed (avec erreur).
    """
    return _rebuild_state
