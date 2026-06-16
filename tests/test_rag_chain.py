"""
tests/test_rag_chain.py
────────────────────────────────────────────────────────────────────────────
Tests unitaires de scripts/rag_chain.py — SANS modèle ni LLM réels

Stratégie : RagChain.__new__ permet d'instancier la classe sans passer
par __init__ (qui charge SBERT, FAISS, le client Mistral). On injecte des fakes
à la place du vectorstore et du LLM, puis on teste la LOGIQUE :
  - retrieve()        : déduplication par uid, ordre des scores,
                        small-to-big (lookup uid → événement complet)
  - build_context()   : assemblage, plafonnement à MAX_EVENT_CHARS,
                        cas "aucun résultat"
  - _check_manifest() : cohérence index ↔ modèle, refus si mismatch
  - ask()             : structure de la réponse {answer, sources, timings}

Lancer :  python -m pytest tests/test_rag_chain.py -v
────────────────────────────────────────────────────────────────────────────
"""

import json

import pytest

from scripts.rag_chain import RagChain, MAX_EVENT_CHARS, SBERT_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Fakes (remplacent les composants lourds)
# ─────────────────────────────────────────────────────────────────────────────

class FakeDoc:
    """Imite un Document LangChain : seul .metadata est utilisé par retrieve()."""
    def __init__(self, uid: str):
        self.metadata = {"uid": uid}


class FakeVectorstore:
    """Imite FAISS : retourne une liste prédéfinie de (doc, score)."""
    def __init__(self, results: list[tuple[str, float]]):
        self._results = [(FakeDoc(uid), score) for uid, score in results]

    def similarity_search_with_score(self, question: str, k: int):
        return self._results[:k]


class FakeRunnable:
    """Imite la chaîne LCEL prompt | llm : invoke() rend une réponse fixe."""
    def __init__(self, response: str):
        self.response = response
        self.last_inputs = None

    def __or__(self, other):          # prompt | llm → se retourne lui-même
        return self

    def invoke(self, inputs: dict):
        self.last_inputs = inputs     # mémorisé pour inspection dans les tests
        return self.response


def make_event(uid: str, full_text: str = None) -> dict:
    return {
        "uid"       : uid,
        "title"     : f"Événement {uid}",
        "city"      : "Paris",
        "date_label": "samedi 13 juin",
        "price"     : "Gratuit",
        "url"       : f"https://example.org/{uid}",
        "full_text" : full_text or f"Titre : Événement {uid}\nLieu : Paris",
    }


@pytest.fixture
def chain():
    """
    RagChain SANS __init__ : pas de SBERT, pas de FAISS, pas d'appel API Mistral.
    Le vectorstore simule 10 chunks issus de 4 événements (A dupliqué 4×).
    """
    c = RagChain.__new__(RagChain)
    c.k_chunks = 10
    c.k_events = 4
    c.vectorstore = FakeVectorstore([
        ("evt-A", 0.42), ("evt-B", 0.45), ("evt-A", 0.48), ("evt-A", 0.51),
        ("evt-C", 0.55), ("evt-B", 0.58), ("evt-A", 0.60), ("evt-C", 0.61),
        ("evt-D", 0.70), ("evt-B", 0.72),
    ])
    c.events_by_uid = {
        uid: make_event(uid) for uid in ["evt-A", "evt-B", "evt-C", "evt-D"]
    }
    c.prompt = FakeRunnable("Je vous recommande l'Événement evt-A.")
    c.llm    = None    # FakeRunnable.__or__ absorbe le pipe
    return c


# ─────────────────────────────────────────────────────────────────────────────
# retrieve : déduplication + small-to-big
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieve:

    def test_deduplication_par_uid(self, chain):
        retrieved = chain.retrieve("un concert ?")
        uids = [item["event"]["uid"] for item in retrieved]
        assert len(uids) == len(set(uids)), "doublons dans le top-k événements"

    def test_ordre_par_meilleur_score(self, chain):
        retrieved = chain.retrieve("un concert ?")
        uids   = [item["event"]["uid"] for item in retrieved]
        scores = [item["score"] for item in retrieved]
        assert uids == ["evt-A", "evt-B", "evt-C", "evt-D"]
        assert scores == sorted(scores), "scores non croissants"

    def test_meilleur_score_conserve_par_evenement(self, chain):
        """evt-A apparaît 4 fois (0.42, 0.48, 0.51, 0.60) → garder 0.42."""
        retrieved = chain.retrieve("un concert ?")
        score_a = next(i["score"] for i in retrieved
                       if i["event"]["uid"] == "evt-A")
        assert score_a == 0.42

    def test_small_to_big_evenement_complet(self, chain):
        """Le retrieve doit rendre l'événement COMPLET, pas le chunk."""
        retrieved = chain.retrieve("un concert ?")
        assert retrieved[0]["event"]["full_text"].startswith("Titre :")
        assert "title" in retrieved[0]["event"]

    def test_limite_k_events(self, chain):
        chain.k_events = 2
        retrieved = chain.retrieve("un concert ?")
        assert len(retrieved) == 2
        assert [i["event"]["uid"] for i in retrieved] == ["evt-A", "evt-B"]

    def test_uid_inconnu_ignore(self, chain):
        """Un uid présent dans l'index mais absent du JSON est ignoré
        proprement (index et données désynchronisés)."""
        del chain.events_by_uid["evt-C"]
        retrieved = chain.retrieve("un concert ?")
        uids = [i["event"]["uid"] for i in retrieved]
        assert "evt-C" not in uids
        assert len(retrieved) == 3


# ─────────────────────────────────────────────────────────────────────────────
# build_context : assemblage + plafonnement
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildContext:

    def test_assemblage_numerote(self, chain):
        retrieved = chain.retrieve("un concert ?")
        context = chain.build_context(retrieved)
        assert "--- Événement 1 ---" in context
        assert "--- Événement 4 ---" in context

    def test_plafonnement_texte_verbeux(self, chain):
        long_event = make_event("evt-X", full_text="x" * (MAX_EVENT_CHARS + 500))
        context = chain.build_context([{"event": long_event, "score": 0.5}])
        assert "[...]" in context
        # plafond + marqueur + en-tête, mais bien en dessous du texte original
        assert len(context) < MAX_EVENT_CHARS + 100

    def test_texte_court_non_tronque(self, chain):
        event = make_event("evt-Y")
        context = chain.build_context([{"event": event, "score": 0.5}])
        assert "[...]" not in context

    def test_aucun_resultat(self, chain):
        assert "aucun événement" in chain.build_context([])


# ─────────────────────────────────────────────────────────────────────────────
# _check_manifest : cohérence index ↔ modèle
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckManifest:

    def _write_manifest(self, tmp_path, model_name: str):
        (tmp_path / "manifest.json").write_text(
            json.dumps({"embedding_model": model_name}), encoding="utf-8"
        )

    def test_modele_coherent_ok(self, chain, tmp_path):
        self._write_manifest(tmp_path, SBERT_MODEL)
        chain._check_manifest(tmp_path)   # ne doit pas lever

    def test_modele_different_refuse(self, chain, tmp_path):
        self._write_manifest(tmp_path, "un-autre-modele")
        with pytest.raises(RuntimeError, match="Incohérence"):
            chain._check_manifest(tmp_path)

    def test_manifest_absent_tolere(self, chain, tmp_path):
        chain._check_manifest(tmp_path)   # warning seulement, pas d'exception


# ─────────────────────────────────────────────────────────────────────────────
# ask : pipeline complet (LLM mocké)
# ─────────────────────────────────────────────────────────────────────────────

class TestAsk:

    def test_structure_de_la_reponse(self, chain):
        result = chain.ask("un concert gratuit à Paris ?")
        assert set(result.keys()) == {"answer", "sources", "timings"}
        assert result["answer"] == "Je vous recommande l'Événement evt-A."
        assert "retrieval_s" in result["timings"]
        assert "generation_s" in result["timings"]

    def test_sources_completes(self, chain):
        result = chain.ask("un concert ?")
        assert len(result["sources"]) == 4
        src = result["sources"][0]
        for cle in ["uid", "title", "city", "date_label", "price", "url", "score"]:
            assert cle in src, f"Clé manquante dans source : {cle}"
        assert src["uid"] == "evt-A"

    def test_prompt_recoit_contexte_question_et_date(self, chain):
        chain.ask("un concert ce week-end ?")
        inputs = chain.prompt.last_inputs
        assert "un concert ce week-end ?" == inputs["question"]
        assert "--- Événement 1 ---" in inputs["context"]
        assert inputs["today"], "la date du jour doit être injectée"
