"""
tests/test_api.py
────────────────────────────────────────────────────────────────────────────
Tests de la couche API FastAPI (main.py) — chaîne RAG mockée

Couverture :
  - GET  /health         : mode prêt et mode dégradé
  - POST /ask            : réponse nominale, validation Pydantic (422),
                           chaîne indisponible (503), erreur interne (500)
  - POST /rebuild        : protection X-API-Key (503/401/403), lancement
                           nominal (202), reconstruction concurrente (409),
                           rechargement de la chaîne, gestion d'échec
  - GET  /rebuild/status : suivi de l'état

La chaîne RAG et la reconstruction sont remplacées par des mocks —
aucun modèle chargé, les tests s'exécutent en quelques millisecondes.

Lancer :  python -m pytest tests/test_api.py -v
────────────────────────────────────────────────────────────────────────────
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main


FAKE_RESULT = {
    "answer" : "Je vous recommande le concert de jazz au Parc de la Villette.",
    "sources": [{
        "uid": "evt-1", "title": "Concert de jazz", "city": "Paris",
        "date_label": "samedi 13 juin", "price": "Gratuit",
        "url": "https://example.org/evt-1", "score": 0.42,
    }],
    "contexts": ["Concert de jazz au Parc de la Villette, Paris, samedi 13 juin, gratuit."],
    "timings": {"retrieval_s": 0.05, "generation_s": 1.8},
}


@pytest.fixture
def client_pret():
    """Client avec chaîne RAG disponible (mockée)."""
    mock_chain = MagicMock()
    mock_chain.ask.return_value = FAKE_RESULT
    with patch("main.get_chain", return_value=mock_chain):
        with TestClient(main.app) as client:       # déclenche le lifespan
            yield client, mock_chain


@pytest.fixture
def client_degrade():
    """Client avec chaîne RAG indisponible (index absent)."""
    with patch("main.get_chain", side_effect=RuntimeError("index introuvable")):
        with TestClient(main.app) as client:
            yield client


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:

    def test_chaine_prete(self, client_pret):
        client, _ = client_pret
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["chain_ready"] is True

    def test_chaine_degradee(self, client_degrade):
        response = client_degrade.get("/health")
        assert response.status_code == 200          # l'app répond quand même
        body = response.json()
        assert body["status"] == "degraded"
        assert body["chain_ready"] is False
        assert "index introuvable" in body["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# /ask
# ─────────────────────────────────────────────────────────────────────────────

class TestAsk:

    def test_reponse_nominale(self, client_pret):
        client, mock_chain = client_pret
        response = client.post("/ask",
                               json={"question": "un concert gratuit à Paris ?"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == FAKE_RESULT["answer"]
        assert body["sources"][0]["uid"] == "evt-1"
        assert body["contexts"] == FAKE_RESULT["contexts"]
        mock_chain.ask.assert_called_once_with("un concert gratuit à Paris ?")

    def test_question_trop_courte_422(self, client_pret):
        client, _ = client_pret
        response = client.post("/ask", json={"question": "ab"})
        assert response.status_code == 422

    def test_question_trop_longue_422(self, client_pret):
        client, _ = client_pret
        response = client.post("/ask", json={"question": "x" * 501})
        assert response.status_code == 422

    def test_question_manquante_422(self, client_pret):
        client, _ = client_pret
        response = client.post("/ask", json={})
        assert response.status_code == 422

    def test_chaine_indisponible_503(self, client_degrade):
        response = client_degrade.post("/ask",
                                       json={"question": "un concert à Paris ?"})
        assert response.status_code == 503
        assert "index introuvable" in response.json()["detail"]

    def test_erreur_interne_500(self, client_pret):
        client, mock_chain = client_pret
        mock_chain.ask.side_effect = ConnectionError("API Mistral injoignable")
        response = client.post("/ask",
                               json={"question": "un concert à Paris ?"})
        assert response.status_code == 500
        assert "Mistral" in response.json()["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# /rebuild et /rebuild/status
# ─────────────────────────────────────────────────────────────────────────────

class FakeThread:
    """
    Remplace threading.Thread : exécute la cible IMMÉDIATEMENT et de façon
    synchrone au .start(). Les fonctions lourdes (run_build, clean_pipeline,
    get_chain) étant mockées par ailleurs, le test reste instantané et
    l'état final (/rebuild/status) est observable juste après le POST.
    """

    def __init__(self, target=None, args=(), kwargs=None, **_ignored):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


@pytest.fixture
def etat_rebuild_propre():
    """Réinitialise l'état module-level entre les tests (et libère le verrou)."""
    main._rebuild_state.update(
        status="idle", params=None, started_at=None,
        finished_at=None, stats=None, error=None,
    )
    if main._rebuild_lock.locked():
        main._rebuild_lock.release()
    yield
    if main._rebuild_lock.locked():
        main._rebuild_lock.release()


@pytest.fixture
def client_rebuild(etat_rebuild_propre):
    """
    Client avec : API_KEY configurée, thread synchrone, et toutes les
    fonctions lourdes mockées (reconstruction + rechargement de chaîne).
    """
    fake_stats = {"nb_events": 200, "nb_chunks": 652, "duration_s": 12.3}
    mock_chain = MagicMock()
    mock_chain.ask.return_value = FAKE_RESULT

    with patch.object(main, "API_KEY", "cle-de-test"), \
         patch.object(main, "threading") as mock_threading, \
         patch.object(main, "run_build", return_value=fake_stats) as mock_build, \
         patch.object(main, "clean_pipeline", return_value=([], {})) as mock_clean, \
         patch.object(main, "save_clean") as mock_save, \
         patch.object(main, "reset_chain"), \
         patch("main.get_chain", return_value=mock_chain):
        mock_threading.Thread = FakeThread
        with TestClient(main.app) as client:
            yield client, mock_build, mock_clean, mock_save


HEADERS_OK = {"X-API-Key": "cle-de-test"}


class TestRebuildSecurite:

    def test_api_key_non_configuree_503(self, client_pret, etat_rebuild_propre):
        client, _ = client_pret
        with patch.object(main, "API_KEY", ""):
            response = client.post("/rebuild", json={})
        assert response.status_code == 503
        assert "API_KEY" in response.json()["detail"]

    def test_header_manquant_401(self, client_pret, etat_rebuild_propre):
        client, _ = client_pret
        with patch.object(main, "API_KEY", "cle-de-test"):
            response = client.post("/rebuild", json={})
        assert response.status_code == 401

    def test_cle_invalide_403(self, client_pret, etat_rebuild_propre):
        client, _ = client_pret
        with patch.object(main, "API_KEY", "cle-de-test"):
            response = client.post("/rebuild", json={},
                                   headers={"X-API-Key": "mauvaise-cle"})
        assert response.status_code == 403


class TestRebuild:

    def test_lancement_nominal_202(self, client_rebuild):
        client, mock_build, _, _ = client_rebuild
        response = client.post("/rebuild", json={}, headers=HEADERS_OK)
        assert response.status_code == 202
        assert response.json()["status_url"] == "/rebuild/status"
        mock_build.assert_called_once()

    def test_status_done_avec_stats(self, client_rebuild):
        client, _, _, _ = client_rebuild
        client.post("/rebuild", json={}, headers=HEADERS_OK)
        body = client.get("/rebuild/status").json()
        assert body["status"] == "done"
        assert body["stats"]["nb_events"] == 200
        assert body["started_at"] is not None
        assert body["finished_at"] is not None
        assert body["error"] is None

    def test_limit_transmis_a_run_build(self, client_rebuild):
        client, mock_build, _, _ = client_rebuild
        client.post("/rebuild", json={"limit": 200}, headers=HEADERS_OK)
        assert mock_build.call_args.kwargs["limit"] == 200

    def test_limit_invalide_422(self, client_rebuild):
        client, _, _, _ = client_rebuild
        response = client.post("/rebuild", json={"limit": 0}, headers=HEADERS_OK)
        assert response.status_code == 422

    def test_clean_first_declenche_nettoyage(self, client_rebuild):
        client, _, mock_clean, mock_save = client_rebuild
        client.post("/rebuild", json={"clean_first": True}, headers=HEADERS_OK)
        mock_clean.assert_called_once()
        mock_save.assert_called_once()

    def test_sans_clean_first_fichier_present(self, client_rebuild, tmp_path):
        """Si le fichier nettoyé existe et clean_first=False → pas de nettoyage."""
        client, _, mock_clean, _ = client_rebuild
        fichier = tmp_path / "clean.json"
        fichier.write_text("[]")
        with patch.object(main, "CLEAN_DATA_PATH", fichier):
            client.post("/rebuild", json={}, headers=HEADERS_OK)
        mock_clean.assert_not_called()

    def test_chaine_rechargee_apres_rebuild(self, client_rebuild):
        """Après une reconstruction réussie, /ask refonctionne (chaîne rechargée)."""
        client, _, _, _ = client_rebuild
        client.post("/rebuild", json={}, headers=HEADERS_OK)
        response = client.post("/ask", json={"question": "un concert à Paris ?"})
        assert response.status_code == 200

    def test_reconstruction_deja_en_cours_409(self, client_rebuild):
        client, _, _, _ = client_rebuild
        main._rebuild_lock.acquire()                # simule un build en cours
        response = client.post("/rebuild", json={}, headers=HEADERS_OK)
        assert response.status_code == 409

    def test_echec_build_status_failed(self, client_rebuild):
        client, mock_build, _, _ = client_rebuild
        mock_build.side_effect = MemoryError("RAM insuffisante")
        response = client.post("/rebuild", json={}, headers=HEADERS_OK)
        assert response.status_code == 202          # le lancement, lui, a réussi
        body = client.get("/rebuild/status").json()
        assert body["status"] == "failed"
        assert "RAM" in body["error"]

    def test_status_initial_idle(self, client_rebuild):
        client, _, _, _ = client_rebuild
        body = client.get("/rebuild/status").json()
        assert body["status"] == "idle"
