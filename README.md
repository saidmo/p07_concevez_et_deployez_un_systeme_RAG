# Assistant RAG — Événements culturels en Île-de-France

POC d'un assistant conversationnel qui répond en langage naturel à des
questions sur les événements culturels d'Île-de-France, à partir des données
**OpenAgenda**. Mission Puls-Events (OpenClassrooms — projet 7).

Le système combine recherche vectorielle (embeddings **SBERT** + index
**FAISS**) et génération de réponse par un LLM (**API Mistral**), orchestrées
par **LangChain** et exposées via une API **FastAPI**.

```
Question  →  embedding SBERT  →  recherche FAISS  →  événements pertinents
                                                          │
                          réponse  ←  LLM Mistral  ←  contexte (small-to-big)
```

---

## Sommaire

- [Architecture](#architecture)
- [Pile technique et choix](#pile-technique-et-choix)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Pipeline de données et construction de l'index](#pipeline-de-données-et-construction-de-lindex)
- [Lancer l'API](#lancer-lapi)
- [Endpoints](#endpoints)
- [Tests](#tests)
- [Évaluation](#évaluation)
- [Exécution avec Docker](#exécution-avec-docker)
- [Variables d'environnement](#variables-denvironnement)

---

## Architecture

Le flux complet, de la donnée brute à la réponse :

1. **Collecte** (`collect_data.py`) — export des événements via l'API
   OpenDataSoft (hub Huwise), filtrés par région et période.
2. **Nettoyage** (`clean_data.py`) — normalisation, suppression du HTML,
   désérialisation des champs JSON-string, exclusion des événements annulés
   ou vides.
3. **Indexation** (`build_index.py`) — découpage en chunks, vectorisation
   SBERT, construction de l'index FAISS et d'un `manifest.json` de traçabilité.
4. **Chaîne RAG** (`rag_chain.py`) — recherche vectorielle, déduplication,
   pattern *small-to-big*, prompt anti-hallucination, génération Mistral.
5. **API** (`main.py`) — exposition HTTP : `/ask`, `/rebuild`, `/health`.

Le *retrieval* applique un schéma **small-to-big** : la recherche se fait sur
des petits chunks (précision), mais ce sont les **événements complets** qui
sont fournis au LLM (contexte riche). Concrètement : FAISS remonte les 10
meilleurs chunks, ceux-ci sont dédupliqués par identifiant d'événement
(`uid`), et les 4 événements distincts les plus pertinents sont reconstitués
en entier pour le prompt.

---

## Pile technique et choix

| Composant | Choix | Pourquoi |
|---|---|---|
| Embeddings | SBERT `paraphrase-multilingual-mpnet-base-v2` (HuggingFace, local CPU) | multilingue/français, 768 dimensions, exécuté localement (pas de clé, pas de coût, données non envoyées à un tiers pour l'indexation) |
| Chunking | 450 caractères, overlap 80 | le modèle tronque silencieusement au-delà de ~128 tokens (~450-500 caractères de français) ; un chunk plus long perdrait sa fin (lieu, tarif) dans le vecteur |
| Index | FAISS `IndexFlatL2`, embeddings normalisés | norme 1 → la distance L2 équivaut au cosinus ; recherche exacte adaptée à cette volumétrie |
| LLM | **API Mistral** (`mistral-small-latest`), température 0.1 | conforme à la mission ; pas de modèle lourd à héberger, génération en quelques secondes |
| Orchestration | LangChain | chaîne `prompt │ llm │ parser`, intégrations FAISS et Mistral |
| API | FastAPI + Pydantic | séparation stricte couche HTTP / logique métier, doc Swagger automatique |
| Conteneur | Docker Compose, CPU only | un seul service applicatif ; le LLM est désormais distant (API) |

> **Note d'architecture (hybride local/distant).** Les *embeddings* tournent
> en local (gratuits, pas de réindexation des dizaines de milliers de chunks à
> chaque appel, aucune donnée envoyée à un tiers pour l'indexation) ; seule la
> *génération* est déléguée à l'API Mistral. Une version antérieure du POC
> utilisait Mistral 7B en local via Ollama : abandonnée car plus lourde (~4 Go
> de modèle, génération de 30-60 s sur CPU) et non conforme à l'énoncé, qui
> demande la plateforme Mistral.

---

## Structure du dépôt

```
p07_concevez_et_deployez_un_systeme_RAG/
├── README.md
├── docker-compose.yml          # service applicatif (FastAPI), volumes data + cache SBERT
├── .env                        # secrets et configuration (NON versionné)
├── .gitignore
│
├── app/
│   ├── Dockerfile              # python:3.11-slim
│   ├── requirements.txt        # dépendances de l'application
│   ├── main.py                 # API FastAPI : /health, /ask, /rebuild, /rebuild/status
│   └── scripts/
│       ├── collect_data.py     # 1. collecte OpenAgenda (OpenDataSoft / Huwise)
│       ├── clean_data.py       # 2. nettoyage et normalisation
│       ├── build_index.py      # 3. chunking + embeddings + index FAISS (+ run_build())
│       └── rag_chain.py        # 4. chaîne RAG (retrieval + génération Mistral)
│
├── tests/                      # tests unitaires (pytest)
│   ├── conftest.py             # ajoute app/ au sys.path
│   ├── test_clean_data.py      # nettoyage
│   ├── test_index.py           # event_to_document, chunking
│   ├── test_rag_chain.py       # déduplication, small-to-big, manifest
│   └── test_api.py             # endpoints, codes HTTP, /rebuild
│
├── eval/                       # évaluation de la qualité du système
│   ├── jeu_de_test_annote.json # 21 cas annotés (questions / réponses de référence)
│   ├── evaluate_rag.py         # évaluation automatique (Ragas + métriques maison)
│   └── requirements-eval.txt   # dépendances d'évaluation (venv dédié)
│
└── data/                       # NON versionné (voir .gitignore)
    ├── raw/                    # export OpenAgenda brut
    ├── processed/              # données nettoyées
    └── faiss_index/            # index vectoriel + manifest.json
```

---

## Installation

**Prérequis :** Python ≥ 3.11 (testé sur 3.13), et une clé API Mistral
(à créer sur <https://console.mistral.ai>).

```bash
# 1. Cloner puis créer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate        # Linux / macOS

# 2. Installer les dépendances de l'application
pip install -r app/requirements.txt

# 3. Configurer les secrets
#    Créer un fichier .env à la racine (voir « Variables d'environnement »)
```

> Les dépendances sont épinglées pour Python 3.11 **et** 3.13 (`torch==2.6.0+cpu`,
> `faiss-cpu==1.9.0.post1`). Tester l'installation sur une machine propre fait
> partie des bonnes pratiques de reproductibilité du projet.

---

## Pipeline de données et construction de l'index

Les trois étapes s'exécutent depuis la **racine** du projet, venv activé.

```bash
# 1. Collecte (≈ 20 000 événements, ~90 Mo de JSON brut)
python app/scripts/collect_data.py
#    options : --region "Île-de-France" --days-history 365 --days-future 365

# 2. Nettoyage
python app/scripts/clean_data.py --input data/raw/evenements_idf.json \
                                 --output data/processed/evenements_idf_clean.json

# 3. Construction de l'index FAISS
python app/scripts/build_index.py            # corpus complet
python app/scripts/build_index.py --limit 500  # sous-ensemble (test rapide)
```

`build_index.py` écrit l'index **et** un `manifest.json` qui mémorise le modèle
d'embedding utilisé. Au démarrage, la chaîne RAG **refuse de fonctionner** si
l'index a été construit avec un modèle différent de celui configuré — garde-fou
contre les incohérences silencieuses.

L'index peut aussi être (re)construit à chaud via l'API : voir l'endpoint
[`/rebuild`](#endpoints).

---

## Lancer l'API

```bash
uvicorn main:app --app-dir app --reload
```

- Documentation interactive Swagger : <http://localhost:8000/docs>
- Vérification d'état : <http://localhost:8000/health>

Si l'index n'a pas encore été construit, l'API démarre en **mode dégradé** :
`/health` renvoie `degraded` et `/ask` répond `503`. Construire l'index
(ci-dessus) puis relancer, ou appeler `/rebuild`.

---

## Endpoints

### `GET /health`

État de l'application (utilisé par le *healthcheck* Docker).

```json
{ "status": "ok", "chain_ready": true }
```

### `POST /ask`

Pose une question à l'assistant.

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d "{\"question\": \"Un concert de jazz gratuit à Paris en juillet 2026 ?\"}"
```

Réponse :

```json
{
  "answer": "…",
  "sources": [
    { "uid": "…", "title": "…", "city": "Paris",
      "date_label": "…", "price": "Gratuit", "url": "…", "score": 0.41 }
  ],
  "contexts": ["…texte des événements fournis au LLM…"],
  "timings": { "retrieval_s": 0.05, "generation_s": 1.8 }
}
```

Le champ `sources` rend chaque recommandation **traçable** jusqu'à la fiche
OpenAgenda. Le champ `contexts` (texte effectivement injecté dans le prompt)
sert la transparence et l'évaluation.

### `POST /rebuild`

Reconstruit l'index FAISS. Opération **longue** : elle est lancée en
arrière-plan et la route répond immédiatement `202 Accepted`. **Protégée par
le header `X-API-Key`.**

```bash
curl -X POST http://localhost:8000/rebuild \
     -H "X-API-Key: <votre_API_KEY>" \
     -H "Content-Type: application/json" \
     -d "{\"limit\": 200}"          # corps optionnel : limit, clean_first
```

Pendant la reconstruction, la chaîne est déchargée (libération mémoire) et
`/ask` répond `503`. Une fois terminée, la chaîne est rechargée
automatiquement sur le nouvel index.

### `GET /rebuild/status`

Suivi de la dernière reconstruction : `idle`, `running`, `done` (avec
statistiques) ou `failed` (avec l'erreur).

---

## Tests

Tests unitaires (chaîne RAG, API et modèles mockés — aucun appel réseau ni
modèle chargé) :

```bash
python -m pytest tests/ -v
```

Couverture : nettoyage des données, transformation événement → document et
chunking, déduplication / small-to-big / vérification du manifest, et les
endpoints de l'API (codes `200` / `422` / `503` / `500`, sécurité et cycle de
vie de `/rebuild`).

---

## Évaluation

La qualité du système est mesurée sur un **jeu de test annoté de 21 cas**
(`eval/jeu_de_test_annote.json`), ancré dans le corpus réel : 17 cas attendant
une réponse (recherche thématique, contraintes de tarif / date / public,
événements nommés…) et 4 cas négatifs (hors périmètre géographique, hors
période, hors domaine, critères impossibles) qui testent la capacité du
système à **ne pas halluciner**.

`evaluate_rag.py` interroge l'API (`POST /ask`) pour chaque cas et calcule :

- **Métriques Ragas** (LLM juge = API Mistral, embeddings = SBERT local) :
  `faithfulness`, `answer_relevancy`, `answer_correctness`, `context_precision`.
- **Métriques maison** (sans coût API) : taux de *retrieval* (au moins un
  `uid` attendu dans les sources), couverture des mots-clés, et taux
  d'abstention correcte sur les cas négatifs.

L'évaluation s'installe dans un **venv dédié** (Ragas impose ses propres
contraintes de versions) :

```bash
python -m venv .venv-eval
.venv-eval\Scripts\activate
pip install -r eval/requirements-eval.txt
```

Puis, l'API tournant dans un autre terminal (venv de l'app) :

```bash
python eval/evaluate_rag.py --limit 3      # rodage rapide
python eval/evaluate_rag.py                # les 21 cas
python eval/evaluate_rag.py --no-ragas     # métriques maison seules
```

> **Rate limit.** Ragas multiplie les appels au LLM juge (plusieurs requêtes
> par cas). Sur le palier gratuit Mistral (~1 requête/seconde), l'évaluation
> complète prend plusieurs minutes ; le script throttle les appels. Commencer
> par `--limit 3`.

Résultats écrits dans `eval/resultats_evaluation.json` (agrégats + détail par
cas) en plus du tableau console.

---

## Exécution avec Docker

```bash
docker compose up -d --build
```

Le service applicatif est construit depuis `app/Dockerfile`
(`python:3.11-slim`). Les volumes persistent les données et l'index
(`./data`) ainsi que le cache du modèle SBERT (`hf_cache`, ~1 Go, téléchargé
une seule fois). La clé `MISTRAL_API_KEY` est transmise depuis le `.env`.

L'index doit exister dans `./data/faiss_index` (monté dans le conteneur), ou
être (re)construit via `POST /rebuild` une fois le service démarré.

---

## Variables d'environnement

À placer dans un fichier `.env` à la racine — **jamais versionné** (présent
dans `.gitignore`).

| Variable | Rôle | Exemple |
|---|---|---|
| `MISTRAL_API_KEY` | Clé de l'API Mistral (génération) | *(secret)* |
| `MISTRAL_MODEL` | Modèle de génération | `mistral-small-latest` |
| `API_KEY` | Protège l'endpoint `/rebuild` (header `X-API-Key`) | *(secret)* |
| `SBERT_MODEL` | Modèle d'embedding | `paraphrase-multilingual-mpnet-base-v2` |
| `FAISS_INDEX_PATH` | Emplacement de l'index | `data/faiss_index` |
| `CLEAN_DATA_PATH` | Données nettoyées | `data/processed/evenements_idf_clean.json` |

`.env` minimal pour démarrer :

```dotenv
MISTRAL_API_KEY=votre_cle_ici
MISTRAL_MODEL=mistral-small-latest
API_KEY=changez_cette_valeur
```

---

*Projet réalisé dans le cadre du parcours AI Engineer — OpenClassrooms (projet 7).*
