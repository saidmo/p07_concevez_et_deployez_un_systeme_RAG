"""
evaluate_rag.py
────────────────────────────────────────────────────────────────────────────
Étape 5 (volet évaluation) — Évaluation automatique de la qualité du RAG

Interroge l'API (POST /ask) pour chaque cas du jeu de test annoté, puis
calcule des métriques Ragas en comparant les réponses générées aux réponses
de référence humaines.

Architecture du test (choix assumés) :
  - Le système est atteint VIA L'API HTTP (l'app doit tourner), et non en
    important RagChain : on évalue exactement ce que les équipes métier
    utiliseront, endpoint compris.
  - Ragas est câblé sur les ressources DU PROJET, pas sur OpenAI (son
    défaut) : LLM juge = API Mistral (ChatMistralAI), embeddings = SBERT
    local. Aucune dépendance ni coût externe au projet.

Métriques (alignées sur la mission : pertinence, fidélité, couverture) :
  - faithfulness        : la réponse est-elle fidèle au contexte fourni ?
                          (mesure directe de l'anti-hallucination)
  - answer_relevancy    : la réponse répond-elle vraiment à la question ?
  - answer_correctness  : proximité avec la réponse de référence (ground truth)
  - context_precision   : les passages pertinents sont-ils bien classés en tête ?

⚠️  RATE LIMIT — Ragas multiplie les appels au LLM juge (décomposition en
    affirmations, questions inverses...) : compter ~5 à 15 requêtes Mistral
    PAR cas. Sur le palier gratuit (~1 req/s), l'évaluation des 21 cas peut
    durer plusieurs minutes. Le script throttle les appels /ask et s'appuie
    sur le max_retries de ChatMistralAI pour absorber les 429 côté Ragas.
    → Tester d'abord avec --limit 3.

Usage :
    # 1. API lancée dans un terminal :
    #    uvicorn main:app --app-dir app   (ou docker compose up)
    # 2. Évaluation (depuis la racine du projet, venv actif) :
    python eval/evaluate_rag.py --limit 3          # rodage rapide
    python eval/evaluate_rag.py                     # les 21 cas
    python eval/evaluate_rag.py --no-ragas          # métriques rapides seules

Sorties :
    - tableau récapitulatif en console
    - eval/resultats_evaluation.json  (détail par cas + agrégats)
────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate_rag")

# ── Configuration ─────────────────────────────────────────────────────────────

ROOT          = Path(__file__).resolve().parent.parent
JEU_DE_TEST   = ROOT / "eval" / "jeu_de_test_annote.json"
SORTIE_JSON   = ROOT / "eval" / "resultats_evaluation.json"

API_BASE      = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY       = os.getenv("API_KEY", "")
ASK_TIMEOUT   = 90       # s — marge large par requête /ask
THROTTLE_S    = 1.1      # s entre deux appels /ask (palier gratuit Mistral ~1 req/s)

SBERT_MODEL   = os.getenv("SBERT_MODEL", "paraphrase-multilingual-mpnet-base-v2")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")


# ── 1. Collecte des réponses du système (via l'API HTTP) ─────────────────────

def interroger_api(question: str) -> dict | None:
    """
    Appelle POST /ask. Retourne le corps JSON, ou None si la chaîne est
    indisponible (503) ou en cas d'erreur réseau.
    """
    try:
        resp = requests.post(
            f"{API_BASE}/ask",
            json={"question": question},
            timeout=ASK_TIMEOUT,
        )
    except requests.RequestException as exc:
        log.error(f"  ↳ échec réseau : {exc}")
        return None

    if resp.status_code == 503:
        log.error("  ↳ 503 : chaîne RAG indisponible. L'index est-il construit ?")
        return None
    if resp.status_code != 200:
        log.error(f"  ↳ HTTP {resp.status_code} : {resp.text[:200]}")
        return None
    return resp.json()


def collecter_reponses(cas: list[dict]) -> list[dict]:
    """
    Pour chaque cas, interroge l'API et assemble les données nécessaires
    à l'évaluation (métriques maison + Ragas).
    """
    enregistrements = []
    for i, c in enumerate(cas, 1):
        log.info(f"[{i}/{len(cas)}] {c['id']} — {c['question'][:60]}")
        reponse = interroger_api(c["question"])

        if reponse is None:
            enregistrements.append({**_vide(c), "erreur": "api_indisponible"})
        else:
            sources_uids = [s["uid"] for s in reponse.get("sources", [])]
            enregistrements.append({
                "id"                  : c["id"],
                "categorie"           : c["categorie"],
                "comportement_attendu": c["comportement_attendu"],
                "question"            : c["question"],
                "answer"              : reponse.get("answer", ""),
                "contexts"            : reponse.get("contexts", []),
                "ground_truth"        : c["reponse_reference"],
                "uids_attendus"       : c["uids_attendus"],
                "uids_obtenus"        : sources_uids,
                "mots_cles_attendus"  : c.get("mots_cles_attendus", []),
                "timings"             : reponse.get("timings", {}),
                "erreur"              : None,
            })
        time.sleep(THROTTLE_S)    # ménager le rate limit (génération Mistral)
    return enregistrements


def _vide(c: dict) -> dict:
    return {
        "id": c["id"], "categorie": c["categorie"],
        "comportement_attendu": c["comportement_attendu"],
        "question": c["question"], "answer": "", "contexts": [],
        "ground_truth": c["reponse_reference"], "uids_attendus": c["uids_attendus"],
        "uids_obtenus": [], "mots_cles_attendus": c.get("mots_cles_attendus", []),
        "timings": {},
    }


# ── 2. Métriques maison (rapides, sans appel LLM supplémentaire) ──────────────

def metriques_maison(enr: list[dict]) -> dict:
    """
    Trois indicateurs calculés sans coût API additionnel :

      - retrieval_hit_rate : sur les cas « repondre », au moins un uid attendu
        figure-t-il dans les sources ? (le retrieval a-t-il trouvé la bonne
        fiche ?). Binaire par cas, moyenné.
      - couverture_mots_cles : proportion des mots-clés attendus présents dans
        la réponse (insensible à la casse).
      - taux_abstention_correct : sur les cas « aucun_resultat », le système
        a-t-il bien refusé/abstenu ? Détecté par marqueurs lexicaux.
    """
    cas_repondre = [e for e in enr if e["comportement_attendu"] == "repondre" and e["erreur"] is None]
    cas_negatifs = [e for e in enr if e["comportement_attendu"] == "aucun_resultat" and e["erreur"] is None]

    # Retrieval hit-rate (cas attendus avec au moins un uid de référence)
    pertinents = [e for e in cas_repondre if e["uids_attendus"]]
    hits = sum(
        1 for e in pertinents
        if set(e["uids_attendus"]) & set(e["uids_obtenus"])
    )
    retrieval_hit_rate = round(hits / len(pertinents), 3) if pertinents else None

    # Couverture des mots-clés
    couvertures = []
    for e in cas_repondre:
        attendus = e["mots_cles_attendus"]
        if not attendus:
            continue
        ans = e["answer"].lower()
        presents = sum(1 for m in attendus if m.lower() in ans)
        couvertures.append(presents / len(attendus))
    couverture_mots_cles = round(sum(couvertures) / len(couvertures), 3) if couvertures else None

    # Abstention correcte sur les cas négatifs
    marqueurs = ("aucun", "ne figure", "ne dispose", "ne peux pas",
                 "pas d'", "pas de", "n'ai pas", "ne correspond", "désolé")
    abstentions = sum(
        1 for e in cas_negatifs
        if any(m in e["answer"].lower() for m in marqueurs)
    )
    taux_abstention = round(abstentions / len(cas_negatifs), 3) if cas_negatifs else None

    return {
        "retrieval_hit_rate"      : retrieval_hit_rate,
        "couverture_mots_cles"    : couverture_mots_cles,
        "taux_abstention_correct" : taux_abstention,
        "n_cas_repondre"          : len(cas_repondre),
        "n_cas_negatifs"          : len(cas_negatifs),
    }


# ── 3. Métriques Ragas (LLM juge = Mistral, embeddings = SBERT) ───────────────

def metriques_ragas(enr: list[dict]) -> dict | None:
    """
    Calcule les métriques Ragas sur les cas « repondre » (les cas négatifs,
    sans ground truth factuel ni contexte pertinent, fausseraient les scores
    et sont évalués par taux_abstention_correct côté maison).

    Ragas est explicitement configuré sur les ressources du projet :
    LLM juge = ChatMistralAI, embeddings = HuggingFaceEmbeddings (SBERT).
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy,
            answer_correctness, context_precision,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_mistralai import ChatMistralAI
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        log.error(f"Dépendances Ragas manquantes ({exc}). "
                  f"Installez : pip install ragas datasets")
        return None

    if not os.getenv("MISTRAL_API_KEY"):
        log.error("MISTRAL_API_KEY absente : Ragas ne peut pas juger. Ignoré.")
        return None

    cas = [e for e in enr if e["comportement_attendu"] == "repondre"
           and e["erreur"] is None and e["contexts"]]
    if not cas:
        log.warning("Aucun cas exploitable pour Ragas.")
        return None

    log.info(f"Ragas : évaluation de {len(cas)} cas "
             f"(LLM juge = {MISTRAL_MODEL}, embeddings = SBERT local)…")
    log.info("⚠️  Plusieurs appels API Mistral par cas — soyez patient "
             "(rate limit du palier gratuit).")

    dataset = Dataset.from_dict({
        "question"     : [e["question"]     for e in cas],
        "answer"       : [e["answer"]       for e in cas],
        "contexts"     : [e["contexts"]     for e in cas],
        "ground_truth" : [e["ground_truth"] for e in cas],
    })

    juge = LangchainLLMWrapper(ChatMistralAI(
        model=MISTRAL_MODEL, temperature=0.0, max_retries=5, timeout=120,
    ))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=SBERT_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ))

    try:
        resultat = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy,
                     answer_correctness, context_precision],
            llm=juge,
            embeddings=embeddings,
            raise_exceptions=False,    # un cas en échec ne fait pas tout planter
        )
    except Exception as exc:
        log.error(f"Échec de l'évaluation Ragas : {exc}")
        return None

    df = resultat.to_pandas()
    # Moyennes par métrique (NaN ignorés)
    agregats = {}
    for col in ("faithfulness", "answer_relevancy",
                "answer_correctness", "context_precision"):
        if col in df.columns:
            valeurs = df[col].dropna()
            agregats[col] = round(float(valeurs.mean()), 3) if len(valeurs) else None

    # Détail par cas (pour le rapport / l'annexe)
    detail = []
    for e, (_, row) in zip(cas, df.iterrows()):
        detail.append({
            "id": e["id"],
            **{col: (round(float(row[col]), 3) if col in df.columns
                     and row[col] == row[col] else None)   # row[col]==row[col] : exclut NaN
               for col in ("faithfulness", "answer_relevancy",
                           "answer_correctness", "context_precision")},
        })

    return {"agregats": agregats, "detail_par_cas": detail, "n_cas": len(cas)}


# ── 4. Restitution ────────────────────────────────────────────────────────────

def afficher_tableau(maison: dict, ragas: dict | None) -> None:
    print("\n" + "═" * 60)
    print("  ÉVALUATION DU SYSTÈME RAG — SYNTHÈSE")
    print("═" * 60)

    print("\n  Métriques maison")
    print("  ────────────────")
    _ligne("Retrieval hit-rate", maison["retrieval_hit_rate"],
           f"(au moins 1 uid attendu trouvé / {maison['n_cas_repondre']} cas)")
    _ligne("Couverture mots-clés", maison["couverture_mots_cles"],
           "(termes attendus présents dans la réponse)")
    _ligne("Abstention correcte", maison["taux_abstention_correct"],
           f"(refus sur les {maison['n_cas_negatifs']} cas sans réponse attendue)")

    if ragas and ragas["agregats"]:
        print(f"\n  Métriques Ragas  (sur {ragas['n_cas']} cas « répondre »)")
        print("  ───────────────")
        libelles = {
            "faithfulness"      : "Fidélité au contexte",
            "answer_relevancy"  : "Pertinence de la réponse",
            "answer_correctness": "Exactitude vs référence",
            "context_precision" : "Précision du contexte",
        }
        for col, lib in libelles.items():
            _ligne(lib, ragas["agregats"].get(col), "")
    else:
        print("\n  Métriques Ragas : non calculées "
              "(--no-ragas, dépendance manquante ou clé absente)")
    print("═" * 60 + "\n")


def _ligne(label: str, valeur, suffixe: str) -> None:
    txt = f"{valeur:.1%}" if isinstance(valeur, float) else "n/a"
    print(f"  {label:.<28} {txt:>7}  {suffixe}")


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Évaluation automatique du RAG (Ragas)")
    parser.add_argument("--limit", type=int, default=None,
                        help="N'évaluer que les N premiers cas (rodage rapide)")
    parser.add_argument("--no-ragas", action="store_true",
                        help="Sauter Ragas : seules les métriques maison rapides")
    parser.add_argument("--jeu", type=Path, default=JEU_DE_TEST,
                        help="Chemin du jeu de test annoté")
    args = parser.parse_args()

    # Vérification préalable : l'API répond-elle ?
    try:
        h = requests.get(f"{API_BASE}/health", timeout=5).json()
        if not h.get("chain_ready"):
            log.error(f"L'API est en mode dégradé ({h}). Construisez l'index "
                      f"(build_index.py ou POST /rebuild) avant d'évaluer.")
            sys.exit(1)
    except requests.RequestException:
        log.error(f"API injoignable sur {API_BASE}. Lancez l'app "
                  f"(uvicorn main:app --app-dir app) puis réessayez.")
        sys.exit(1)

    cas = json.loads(args.jeu.read_text(encoding="utf-8"))["cas"]
    if args.limit:
        cas = cas[: args.limit]
    log.info(f"Jeu de test : {len(cas)} cas à évaluer")

    enregistrements = collecter_reponses(cas)
    maison = metriques_maison(enregistrements)
    ragas  = None if args.no_ragas else metriques_ragas(enregistrements)

    afficher_tableau(maison, ragas)

    sortie = {
        "execute_le"   : datetime.now(timezone.utc).isoformat(),
        "api_base"     : API_BASE,
        "modele_juge"  : MISTRAL_MODEL,
        "n_cas_evalues": len(cas),
        "metriques_maison": maison,
        "metriques_ragas" : ragas,
        "detail_par_cas"  : [
            {k: e[k] for k in ("id", "categorie", "comportement_attendu",
                               "question", "answer", "uids_attendus",
                               "uids_obtenus", "timings", "erreur")}
            for e in enregistrements
        ],
    }
    SORTIE_JSON.write_text(
        json.dumps(sortie, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(f"Résultats détaillés écrits dans {SORTIE_JSON}")


if __name__ == "__main__":
    main()
