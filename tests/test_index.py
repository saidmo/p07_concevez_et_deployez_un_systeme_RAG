"""
tests/test_index.py
────────────────────────────────────────────────────────────────────────────
Tests unitaires de scripts/build_index.py (hors vectorisation réelle)

Couverture :
  - event_to_document : page_content = full_text, métadonnées complètes
  - chunk_documents   : texte court → 1 chunk, texte long → plusieurs,
                        overlap présent, héritage des métadonnées
  - load_clean_events : formats dict et liste

La vectorisation SBERT et FAISS ne sont PAS testées ici (trop lourdes
pour des tests unitaires) — elles sont validées par le smoke test
intégré à build_index.py.

Lancer :  python -m pytest tests/test_index.py -v
────────────────────────────────────────────────────────────────────────────
"""

import json

import pytest

from scripts.build_index import (
    event_to_document,
    chunk_documents,
    load_clean_events,
)


@pytest.fixture
def event_court():
    return {
        "uid": "evt-1", "title": "Concert", "city": "Paris",
        "department": "Paris", "date_begin": "2026-06-13", "date_end": "",
        "date_label": "samedi 13 juin", "price": "Gratuit",
        "attendance": "Sur place", "url": "https://example.org/evt-1",
        "full_text": "Titre : Concert\nLieu : Paris\nTarif : Gratuit",
    }


@pytest.fixture
def event_long(event_court):
    event = dict(event_court)
    event["uid"] = "evt-2"
    # full_text de ~1800 caractères → doit produire plusieurs chunks de 450
    event["full_text"] = "Titre : Grand festival\n" + \
        "Une phrase descriptive qui se répète pour simuler un long texte. " * 28
    return event


class TestEventToDocument:

    def test_page_content_est_full_text(self, event_court):
        doc = event_to_document(event_court)
        assert doc.page_content == event_court["full_text"]

    def test_metadonnees_completes(self, event_court):
        meta = event_to_document(event_court).metadata
        for cle in ["uid", "title", "city", "department", "date_begin",
                    "date_label", "price", "attendance", "url"]:
            assert cle in meta, f"Métadonnée manquante : {cle}"
        assert meta["uid"] == "evt-1"
        assert meta["price"] == "Gratuit"


class TestChunkDocuments:

    def test_texte_court_un_seul_chunk(self, event_court):
        docs = [event_to_document(event_court)]
        chunks = chunk_documents(docs, chunk_size=450, chunk_overlap=80)
        assert len(chunks) == 1

    def test_texte_long_plusieurs_chunks(self, event_long):
        docs = [event_to_document(event_long)]
        chunks = chunk_documents(docs, chunk_size=450, chunk_overlap=80)
        assert len(chunks) >= 3
        # Aucun chunk ne dépasse la taille maximale
        assert all(len(c.page_content) <= 450 for c in chunks)

    def test_chunks_heritent_des_metadonnees(self, event_long):
        docs = [event_to_document(event_long)]
        chunks = chunk_documents(docs, chunk_size=450, chunk_overlap=80)
        # TOUS les chunks portent l'uid de l'événement parent
        assert all(c.metadata["uid"] == "evt-2" for c in chunks)

    def test_overlap_entre_chunks_consecutifs(self, event_long):
        docs = [event_to_document(event_long)]
        chunks = chunk_documents(docs, chunk_size=450, chunk_overlap=80)
        # La fin du chunk N doit partager du texte avec le début du chunk N+1
        fin_premier  = chunks[0].page_content[-40:]
        assert fin_premier in chunks[0].page_content
        debut_second = chunks[1].page_content
        # Au moins un fragment de 20 caractères en commun
        assert any(fin_premier[i:i+20] in debut_second
                   for i in range(len(fin_premier) - 20))

    def test_melange_courts_et_longs(self, event_court, event_long):
        docs = [event_to_document(event_court), event_to_document(event_long)]
        chunks = chunk_documents(docs, chunk_size=450, chunk_overlap=80)
        uids = {c.metadata["uid"] for c in chunks}
        assert uids == {"evt-1", "evt-2"}
        assert len(chunks) > 2


class TestLoadCleanEvents:

    def test_format_dict_results(self, tmp_path):
        path = tmp_path / "clean.json"
        path.write_text(json.dumps({"total": 1, "results": [{"uid": "a"}]}),
                        encoding="utf-8")
        assert len(load_clean_events(path)) == 1

    def test_format_liste(self, tmp_path):
        path = tmp_path / "clean.json"
        path.write_text(json.dumps([{"uid": "a"}, {"uid": "b"}]),
                        encoding="utf-8")
        assert len(load_clean_events(path)) == 2
