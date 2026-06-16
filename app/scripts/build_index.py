"""
build_index.py
────────────────────────────────────────────────────────────────────────────
Étape 3 — Chunking, vectorisation (SBERT) et indexation FAISS

Entrée  : data/processed/evenements_idf_clean.json   (produit par clean_data.py)
Sortie  : data/faiss_index/
            ├── index.faiss      (vecteurs)
            ├── index.pkl        (mapping vecteurs → chunks + métadonnées)
            └── manifest.json    (modèle utilisé, paramètres, traçabilité)

Choix techniques:
  - Modèle   : paraphrase-multilingual-mpnet-base-v2 (SBERT)
               → multilingue (fr inclus), 768 dims, symétrique (pas de préfixes)
  - chunk_size = 450 caractères, overlap = 80
               → le modèle tronque silencieusement au-delà de 128 tokens
                 (~450-500 caractères de français) ; un chunk plus grand
                 perdrait l'information de fin (lieu, tarif) dans le vecteur
  - normalize_embeddings = True
               → vecteurs de norme 1 : la distance L2 de FAISS devient
                 équivalente au classement par similarité cosinus
  - IndexFlatL2 (défaut LangChain) : recherche exacte, adapté à ~25K vecteurs

Usage (depuis la racine du projet) :
    python app/scripts/build_index.py
    python app/scripts/build_index.py --limit 200        # test rapide
    python app/scripts/build_index.py --chunk-size 450 --overlap 80
────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import time
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Configuration ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Variables d'environnement (définies dans docker-compose.yml) avec
# fallback sur les chemins relatifs pour l'exécution locale en venv
SBERT_MODEL  = os.getenv("SBERT_MODEL", "paraphrase-multilingual-mpnet-base-v2")
INPUT_PATH   = Path(os.getenv("CLEAN_DATA_PATH", "data/processed/evenements_idf_clean.json"))
INDEX_DIR    = Path(os.getenv("FAISS_INDEX_PATH", "data/faiss_index"))

DEFAULT_CHUNK_SIZE    = 450   # ≈ limite des 128 tokens du modèle SBERT choisi
DEFAULT_CHUNK_OVERLAP = 80


# ── Chargement ────────────────────────────────────────────────────────────────

def load_clean_events(input_path: Path) -> list[dict]:
    """Charge les événements nettoyés produits par clean_data.py."""
    log.info(f"Lecture de {input_path}")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    events = data["results"] if isinstance(data, dict) else data
    log.info(f"{len(events)} événements nettoyés chargés")
    return events


# ── Construction des Documents LangChain ──────────────────────────────────────

def event_to_document(event: dict) -> Document:
    """
    Convertit un événement nettoyé en Document LangChain.
    Le page_content (full_text) sera chunké puis vectorisé ;
    les metadata voyagent avec chaque chunk et permettent
    le filtrage et la citation des sources côté retriever.
    """
    return Document(
        page_content=event["full_text"],
        metadata={
            "uid"       : event.get("uid", ""),
            "title"     : event.get("title", ""),
            "city"      : event.get("city", ""),
            "department": event.get("department", ""),
            "date_begin": event.get("date_begin", ""),
            "date_end"  : event.get("date_end", ""),
            "date_label": event.get("date_label", ""),
            "price"     : event.get("price", ""),
            "attendance": event.get("attendance", ""),
            "url"       : event.get("url", ""),
        },
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """
    Découpe les documents en chunks.
    Sur ce corpus (médiane full_text : 859 car.), ~84 % des événements
    dépassent 450 caractères et seront découpés en 2+ chunks
    (~64 000 chunks attendus, ratio ~3,3 chunks/événement).
    Chaque chunk hérite des métadonnées de son événement parent.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Ordre de découpe : paragraphes, lignes, phrases, mots
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    nb_docs   = len(documents)
    nb_chunks = len(chunks)
    log.info(f"Chunking : {nb_docs} documents → {nb_chunks} chunks "
             f"(ratio {nb_chunks / nb_docs:.2f})")
    return chunks


# ── Embeddings + Index FAISS ──────────────────────────────────────────────────

def get_embeddings(model_name: str = SBERT_MODEL) -> HuggingFaceEmbeddings:
    """
    Initialise le modèle d'embedding SBERT.
    IMPORTANT : ce même objet (même modèle, mêmes paramètres) devra être
    utilisé au chargement de l'index pour vectoriser les questions —
    les deux espaces vectoriels doivent être identiques.
    """
    log.info(f"Chargement du modèle d'embedding : {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,   # norme 1 → L2 ≡ cosinus
            "batch_size": 64,
        },
    )


def build_faiss_index(chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Vectorise tous les chunks et construit l'index FAISS.
    Étape la plus longue du pipeline sur CPU (~20-40 min pour 25K chunks).
    """
    log.info(f"Vectorisation de {len(chunks)} chunks (CPU, patience...)")
    t0 = time.perf_counter()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    elapsed = time.perf_counter() - t0
    log.info(f"Index construit en {elapsed / 60:.1f} min "
             f"({len(chunks) / elapsed:.0f} chunks/s)")
    return vectorstore


def save_index(
    vectorstore: FAISS,
    index_dir: Path,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    nb_events: int,
    nb_chunks: int,
) -> None:
    """
    Sauvegarde l'index FAISS + un manifest de traçabilité.
    Le manifest permet de vérifier au chargement que le modèle
    d'embedding de l'application correspond à celui de l'index.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))

    manifest = {
        "embedding_model": model_name,
        "normalize_embeddings": True,
        "chunk_size"     : chunk_size,
        "chunk_overlap"  : chunk_overlap,
        "nb_events"      : nb_events,
        "nb_chunks"      : nb_chunks,
        "built_at"       : datetime.now(timezone.utc).isoformat(),
    }
    with open(index_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log.info(f"Index sauvegardé dans {index_dir}/")
    for file in sorted(index_dir.iterdir()):
        size_mb = file.stat().st_size / 1024 / 1024
        log.info(f"  {file.name:15} {size_mb:8.1f} Mo")


# ── Validation rapide ─────────────────────────────────────────────────────────

def smoke_test(vectorstore: FAISS, k: int = 3) -> None:
    """Recherche de validation sur une question type."""
    question = "un concert de musique gratuit à Paris"
    log.info(f"── Smoke test : « {question} » ──")
    results = vectorstore.similarity_search_with_score(question, k=k)
    for i, (doc, score) in enumerate(results, 1):
        title = doc.metadata.get("title", "?")
        city  = doc.metadata.get("city", "?")
        price = doc.metadata.get("price", "?")
        log.info(f"  {i}. [{score:.3f}] {title} — {city} — {price}")


# ── Pipeline complet (appelable depuis la CLI ou l'endpoint /rebuild) ────────

def run_build(
    input_path: Path = INPUT_PATH,
    output_dir: Path = INDEX_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    limit: int | None = None,
) -> dict:
    """
    Pipeline complet de construction de l'index :
    chargement → Documents → chunking → embeddings → FAISS → sauvegarde.

    Fonction PURE vis-à-vis de la CLI : aucune lecture d'argv, tous les
    paramètres sont explicites. C'est elle que l'endpoint POST /rebuild
    de l'API appelle (main.py), dans un thread d'arrière-plan.

    Retourne un dict de statistiques (repris dans /rebuild/status) :
        {"nb_events", "nb_chunks", "duration_s"}
    """
    log.info("═══ Construction de l'index FAISS ═══")
    t0 = time.perf_counter()

    events = load_clean_events(input_path)
    if limit:
        events = events[:limit]
        log.info(f"Mode test : limité à {len(events)} événements")

    documents  = [event_to_document(e) for e in events]
    chunks     = chunk_documents(documents, chunk_size, chunk_overlap)
    embeddings = get_embeddings()
    vectorstore = build_faiss_index(chunks, embeddings)

    save_index(
        vectorstore, output_dir,
        model_name=SBERT_MODEL,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        nb_events=len(events),
        nb_chunks=len(chunks),
    )

    smoke_test(vectorstore)
    duration = round(time.perf_counter() - t0, 1)
    log.info("═══ Index prêt ═══")

    return {
        "nb_events" : len(events),
        "nb_chunks" : len(chunks),
        "duration_s": duration,
    }


# ── Point d'entrée CLI ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Construction de l'index FAISS")
    parser.add_argument("--input",      type=Path, default=INPUT_PATH)
    parser.add_argument("--output",     type=Path, default=INDEX_DIR)
    parser.add_argument("--chunk-size", type=int,  default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap",    type=int,  default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--limit",      type=int,  default=None,
                        help="Limiter le nombre d'événements (test rapide)")
    args = parser.parse_args()

    run_build(
        input_path=args.input,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
