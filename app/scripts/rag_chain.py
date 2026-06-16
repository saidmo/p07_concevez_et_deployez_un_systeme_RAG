"""
rag_chain.py
────────────────────────────────────────────────────────────────────────────
Étape 4 — Orchestration LangChain : la chaîne RAG complète

Rôle : assembler retriever (SBERT + FAISS), small-to-big, prompt et
Mistral (API plateforme mistral.ai) en un pipeline unique. Ne connaît RIEN de FastAPI —
main.py enveloppera la fonction ask() dans un endpoint HTTP.

Flux par question :
    question
      → retriever FAISS (k chunks)
      → déduplication par uid (un chunk = un pointeur vers son événement)
      → small-to-big : lookup uid → full_text COMPLET de chaque événement
      → prompt (instructions + date du jour + contexte + question)
      → l'API Mistral génère
      → {answer, sources, contexts}

Prérequis : l'index FAISS doit avoir été construit (build_index.py).

Usage en CLI (test rapide sans FastAPI) :
    python app/scripts/rag_chain.py "un concert gratuit à Paris ce week-end ?"
────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()   # charge .env en exécution locale (MISTRAL_API_KEY...) ;
                # sans effet dans Docker où l'env vient de docker-compose.yml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SBERT_MODEL     = os.getenv("SBERT_MODEL", "paraphrase-multilingual-mpnet-base-v2")
MISTRAL_MODEL   = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
INDEX_DIR       = Path(os.getenv("FAISS_INDEX_PATH", "data/faiss_index"))
CLEAN_DATA_PATH = Path(os.getenv("CLEAN_DATA_PATH", "data/processed/evenements_idf_clean.json"))

K_CHUNKS        = 10     # chunks récupérés par FAISS (large, avant déduplication)
K_EVENTS        = 4      # événements distincts conservés après déduplication
MAX_EVENT_CHARS = 3000   # plafond par full_text (un événement verbeux ne doit
                         # pas dévorer le budget de contexte des autres)

# ── Prompt système ────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """Tu es un assistant spécialisé dans les événements culturels en Île-de-France.

Nous sommes le {today}.

Réponds à la question de l'utilisateur en t'appuyant UNIQUEMENT sur les événements ci-dessous. Règles :
- Réponds en français, de manière concise et naturelle.
- Ne recommande que des événements qui correspondent réellement aux critères de la question (lieu, date, tarif, public...).
- Si aucun événement fourni ne correspond, dis-le clairement — n'invente jamais d'événement.
- Mentionne pour chaque recommandation : le titre, le lieu, la date et le tarif si disponibles.

ÉVÉNEMENTS DISPONIBLES :
{context}

QUESTION : {question}

RÉPONSE :"""


# ── La chaîne RAG ─────────────────────────────────────────────────────────────

class RagChain:
    """
    Chaîne RAG complète. À instancier UNE FOIS au démarrage de
    l'application (chargements lourds : SBERT ~1 Go, index FAISS,
    dictionnaire des événements), puis ask() à chaque question.
    """

    def __init__(
        self,
        index_dir: Path = INDEX_DIR,
        clean_data_path: Path = CLEAN_DATA_PATH,
        k_chunks: int = K_CHUNKS,
        k_events: int = K_EVENTS,
    ):
        self.k_chunks = k_chunks
        self.k_events = k_events

        # 1. Vérification de cohérence index ↔ modèle (manifest.json)
        self._check_manifest(index_dir)

        # 2. Modèle d'embedding — STRICTEMENT le même que build_index.py
        log.info(f"Chargement du modèle d'embedding : {SBERT_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=SBERT_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 3. Index FAISS
        log.info(f"Chargement de l'index FAISS : {index_dir}")
        self.vectorstore = FAISS.load_local(
            str(index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,   # index.pkl produit par nous
        )

        # 4. Dictionnaire uid → événement complet (pour le small-to-big)
        log.info(f"Chargement des événements : {clean_data_path}")
        with open(clean_data_path, encoding="utf-8") as f:
            data = json.load(f)
        events = data["results"] if isinstance(data, dict) else data
        self.events_by_uid = {e["uid"]: e for e in events}
        log.info(f"{len(self.events_by_uid)} événements en mémoire")

        # 5. LLM Mistral via l'API de la plateforme mistral.ai
        if not os.getenv("MISTRAL_API_KEY"):
            raise RuntimeError(
                "MISTRAL_API_KEY absente de l'environnement : ajoutez-la "
                "dans le fichier .env (jamais dans le code ni dans Git). "
                "Clé à créer sur https://console.mistral.ai"
            )
        log.info(f"Connexion à l'API Mistral : {MISTRAL_MODEL}")
        self.llm = ChatMistralAI(
            model=MISTRAL_MODEL,
            temperature=0.1,     # factuel : on veut des recommandations fidèles
            max_retries=2,       # tolère les erreurs transitoires (429 rate limit)
            timeout=60,
        )

        # ChatMistralAI renvoie un AIMessage : StrOutputParser en extrait
        # le texte, pour que ask() continue de retourner un str
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.parser = StrOutputParser()
        log.info("Chaîne RAG prête")

    # ── Étapes internes ──────────────────────────────────────────────────────

    def _check_manifest(self, index_dir: Path) -> None:
        """Refuse de démarrer si l'index a été construit avec un autre modèle."""
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists():
            log.warning("manifest.json absent — impossible de vérifier la "
                        "cohérence index ↔ modèle d'embedding")
            return
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        index_model = manifest.get("embedding_model", "")
        if index_model != SBERT_MODEL:
            raise RuntimeError(
                f"Incohérence d'embedding : l'index a été construit avec "
                f"'{index_model}' mais l'application utilise '{SBERT_MODEL}'. "
                f"Reconstruisez l'index (build_index.py) ou alignez SBERT_MODEL."
            )

    def retrieve(self, question: str) -> list[dict]:
        """
        R du RAG : recherche FAISS large, déduplication par uid,
        small-to-big (récupération des événements complets).
        Retourne une liste de dicts {event, score} triée par pertinence.
        """
        results = self.vectorstore.similarity_search_with_score(
            question, k=self.k_chunks
        )

        # Déduplication : un événement = son meilleur score (le plus petit en L2)
        best_by_uid: dict[str, float] = {}
        for doc, score in results:
            uid = doc.metadata.get("uid", "")
            if uid and (uid not in best_by_uid or score < best_by_uid[uid]):
                best_by_uid[uid] = score

        # Top k_events événements distincts, triés par score croissant
        top_uids = sorted(best_by_uid, key=best_by_uid.get)[: self.k_events]

        # Small-to-big : lookup uid → événement complet
        retrieved = []
        for uid in top_uids:
            event = self.events_by_uid.get(uid)
            if event:
                retrieved.append({"event": event, "score": best_by_uid[uid]})
        return retrieved

    def build_context(self, retrieved: list[dict]) -> str:
        """
        A du RAG : assemble le contexte injecté dans le prompt.
        full_text complet de chaque événement, plafonné à MAX_EVENT_CHARS.
        """
        blocks = []
        for i, item in enumerate(retrieved, 1):
            text = item["event"]["full_text"]
            if len(text) > MAX_EVENT_CHARS:
                text = text[:MAX_EVENT_CHARS] + " [...]"
            blocks.append(f"--- Événement {i} ---\n{text}")
        return "\n\n".join(blocks) if blocks else "(aucun événement trouvé)"

    # ── API publique ─────────────────────────────────────────────────────────

    def ask(self, question: str) -> dict:
        """
        Pipeline complet : R → A → G.
        Retourne {"answer", "sources", "timings"}.
        """
        timings = {}

        # R — Retrieval
        t0 = time.perf_counter()
        retrieved = self.retrieve(question)
        timings["retrieval_s"] = round(time.perf_counter() - t0, 3)

        # A — Augmentation
        context = self.build_context(retrieved)
        today   = datetime.now().strftime("%A %d %B %Y")

        # G — Generation
        t0 = time.perf_counter()
        chain  = self.prompt | self.llm | self.parser
        answer = chain.invoke({
            "context" : context,
            "question": question,
            "today"   : today,
        })
        timings["generation_s"] = round(time.perf_counter() - t0, 2)

        # Sources : métadonnées des événements utilisés (pour l'API / l'UI)
        sources = [
            {
                "uid"       : item["event"]["uid"],
                "title"     : item["event"]["title"],
                "city"      : item["event"]["city"],
                "date_label": item["event"]["date_label"],
                "price"     : item["event"]["price"],
                "url"       : item["event"]["url"],
                "score"     : round(item["score"], 4),
            }
            for item in retrieved
        ]

        # Contexts : texte brut de chaque événement injecté dans le prompt.
        # Exposé pour la transparence et pour l'évaluation Ragas
        # (métriques faithfulness / context_precision), qui a besoin de la
        # liste des passages fournis au LLM — un par événement récupéré.
        contexts = [
            item["event"]["full_text"][:MAX_EVENT_CHARS]
            for item in retrieved
        ]

        return {
            "answer"  : answer.strip(),
            "sources" : sources,
            "contexts": contexts,
            "timings" : timings,
        }


# ── Singleton pour FastAPI ────────────────────────────────────────────────────

_chain: RagChain | None = None


def get_chain() -> RagChain:
    """
    Retourne l'instance unique de la chaîne (créée au premier appel).
    main.py l'utilisera comme dépendance — les chargements lourds
    ne sont faits qu'une seule fois.
    """
    global _chain
    if _chain is None:
        _chain = RagChain()
    return _chain


def reset_chain() -> None:
    """
    Décharge la chaîne (SBERT, index FAISS, événements en mémoire).
    Utilisé par l'endpoint /rebuild :
      1. avant la reconstruction — pour libérer la RAM (la vectorisation
         charge sa propre instance de SBERT, on évite d'en avoir deux) ;
      2. après — le prochain get_chain() rechargera le NOUVEL index.
    """
    global _chain
    _chain = None


# ── CLI de test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    question = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "un concert de musique gratuit à Paris ce week-end ?"
    )

    chain = get_chain()
    log.info(f"Question : {question}")
    result = chain.ask(question)

    print("\n══ RÉPONSE ══════════════════════════════════════════")
    print(result["answer"])
    print("\n══ SOURCES ══════════════════════════════════════════")
    for src in result["sources"]:
        print(f"  [{src['score']}] {src['title']} — {src['city']} — "
              f"{src['date_label']} — {src['price']}")
    print(f"\nRetrieval : {result['timings']['retrieval_s']}s | "
          f"Génération : {result['timings']['generation_s']}s")
