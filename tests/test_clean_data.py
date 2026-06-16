"""
tests/test_clean_data.py
────────────────────────────────────────────────────────────────────────────
Tests unitaires de scripts/clean_data.py

Couverture :
  - strip_html          : suppression HTML, espaces, entrées vides
  - parse_json_field    : string JSON, déjà parsé, invalide, None
  - get_label_fr        : champs structurés OpenAgenda (status, attendancemode)
  - clean_title         : préfixes "Annulé | ", casse, titre sain
  - is_cancelled        : via titre, via status JSON, événement sain
  - extract_city        : location_city direct, extraction depuis l'adresse
  - normalize_price     : gratuit, prix unique, fourchette, virgules, inscription
  - normalize_age_max   : valeurs aberrantes >= 110, valeurs saines, None
  - clean_event         : structure de sortie, exclusions (annulé, vide)
  - load_raw            : format liste ET format {"results": [...]}

Lancer :  python -m pytest tests/test_clean_data.py -v
────────────────────────────────────────────────────────────────────────────
"""

import json

import pytest

from scripts.clean_data import (
    strip_html,
    parse_json_field,
    get_label_fr,
    clean_title,
    is_cancelled,
    extract_city,
    normalize_price,
    normalize_age_max,
    clean_event,
    load_raw,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_event():
    """Événement brut représentatif du format réel evenements_idf.json."""
    return {
        "uid"                : 12345678,
        "slug"               : "concert-jazz-villette",
        "canonicalurl"       : "https://openagenda.com/events/concert-jazz",
        "title_fr"           : "Concert de jazz au Parc",
        "description_fr"     : "Un concert en plein air.",
        "longdescription_fr" : "<p>Le <b>quartet</b> se produit&nbsp;en plein air.</p>",
        "location_city"      : "Paris",
        "location_region"    : "Île-de-France",
        "location_department": "Paris",
        "location_name"      : "Parc de la Villette",
        "location_address"   : "211 avenue Jean Jaurès, 75019 Paris",
        "location_postalcode": "75019",
        "daterange_fr"       : "samedi 13 juin",
        "firstdate_begin"    : "2026-06-13T18:00:00+00:00",
        "lastdate_end"       : "2026-06-13T22:00:00+00:00",
        "conditions_fr"      : "Entrée gratuite",
        "status"             : '{"id": 1, "label": {"fr": "Programmé", "en": "Scheduled"}}',
        "attendancemode"     : '{"id": 1, "label": {"fr": "Sur place", "en": "Offline"}}',
        "age_min"            : None,
        "age_max"            : None,
        "keywords_fr"        : "jazz, musique, plein air",
    }


# ─────────────────────────────────────────────────────────────────────────────
# strip_html
# ─────────────────────────────────────────────────────────────────────────────

class TestStripHtml:

    def test_supprime_les_balises(self):
        assert strip_html("<p>Bonjour <b>monde</b></p>") == "Bonjour monde"

    def test_normalise_les_espaces(self):
        assert strip_html("<p>a</p>\n\n<p>b   c</p>") == "a b c"

    def test_entree_vide(self):
        assert strip_html("") == ""
        assert strip_html(None) == ""

    def test_texte_sans_html_inchange(self):
        assert strip_html("Texte simple.") == "Texte simple."

    def test_entites_html_decodees(self):
        # &nbsp; et &amp; doivent devenir des caractères normaux
        result = strip_html("Jazz&nbsp;&amp;&nbsp;Blues")
        assert "&" in result and "nbsp" not in result


# ─────────────────────────────────────────────────────────────────────────────
# parse_json_field / get_label_fr
# ─────────────────────────────────────────────────────────────────────────────

class TestParseJsonField:

    def test_string_json(self):
        assert parse_json_field('{"a": 1}') == {"a": 1}

    def test_deja_parse(self):
        assert parse_json_field({"a": 1}) == {"a": 1}
        assert parse_json_field([1, 2]) == [1, 2]

    def test_string_non_json(self):
        assert parse_json_field("pas du json") == "pas du json"

    def test_none(self):
        assert parse_json_field(None) is None


class TestGetLabelFr:

    def test_status_openagenda(self):
        status = '{"id": 6, "label": {"fr": "Annulé", "en": "Cancelled"}}'
        assert get_label_fr(status) == "Annulé"

    def test_attendancemode(self):
        mode = '{"id": 2, "label": {"fr": "En ligne", "en": "Online"}}'
        assert get_label_fr(mode) == "En ligne"

    def test_champ_absent_ou_invalide(self):
        assert get_label_fr(None) == ""
        assert get_label_fr("texte brut") == ""
        assert get_label_fr('{"sans": "label"}') == ""


# ─────────────────────────────────────────────────────────────────────────────
# clean_title / is_cancelled
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanTitle:

    @pytest.mark.parametrize("brut,attendu", [
        ("Annulé | Concert de jazz",   "Concert de jazz"),
        ("ANNULÉ - Exposition photo",  "Exposition photo"),
        ("Annulée : Visite guidée",    "Visite guidée"),
        ("Concert de jazz",            "Concert de jazz"),   # titre sain inchangé
        ("",                           ""),
        (None,                         ""),
    ])
    def test_prefixes(self, brut, attendu):
        assert clean_title(brut) == attendu


class TestIsCancelled:

    def test_via_titre(self):
        assert is_cancelled({"title_fr": "Annulé | Concert"}) is True

    def test_via_status_json(self):
        event = {
            "title_fr": "Concert de jazz",
            "status"  : '{"id": 6, "label": {"fr": "Annulé", "en": "Cancelled"}}',
        }
        assert is_cancelled(event) is True

    def test_evenement_programme(self, raw_event):
        assert is_cancelled(raw_event) is False

    def test_status_reporte_non_exclu(self):
        """Un événement 'Reporté' n'est PAS annulé — il doit être conservé."""
        event = {
            "title_fr": "Concert",
            "status"  : '{"id": 4, "label": {"fr": "Reporté", "en": "Postponed"}}',
        }
        assert is_cancelled(event) is False


# ─────────────────────────────────────────────────────────────────────────────
# extract_city
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractCity:

    def test_location_city_prioritaire(self, raw_event):
        assert extract_city(raw_event) == "Paris"

    def test_extraction_depuis_adresse(self):
        event = {"location_city": None,
                 "location_address": "12 rue de la Paix, 75002 Paris"}
        assert extract_city(event) == "Paris"

    def test_ville_composee(self):
        event = {"location_city": None,
                 "location_address": "3 allée des Tilleuls, 94100 Saint-Maur-Des-Fossés"}
        assert extract_city(event) == "Saint-Maur-Des-Fossés"

    def test_aucune_info(self):
        assert extract_city({"location_city": None, "location_address": None}) == ""


# ─────────────────────────────────────────────────────────────────────────────
# normalize_price
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizePrice:

    @pytest.mark.parametrize("brut,attendu", [
        ("Entrée gratuite",                  "Gratuit"),
        ("ENTRÉE GRATUITE",                  "Gratuit"),
        ("Entrée Libre",                     "Gratuit"),
        ("12€",                              "12€"),
        ("10€ en prévente | 15€ sur place",  "De 10€ à 15€"),
        ("Sur inscription",                  "Sur inscription"),
        ("",                                 ""),
        (None,                               ""),
    ])
    def test_normalisation(self, brut, attendu):
        assert normalize_price(brut) == attendu

    def test_prix_avec_virgule(self):
        """8,50€ doit être comparé numériquement, pas alphabétiquement."""
        assert normalize_price("8,50€ à 12€") == "De 8,50€ à 12€"

    def test_gratuit_prioritaire_sur_inscription(self):
        """'Gratuit sur inscription' → c'est gratuit, info la plus utile."""
        assert normalize_price("Gratuit sur inscription") == "Gratuit"


# ─────────────────────────────────────────────────────────────────────────────
# normalize_age_max
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeAgeMax:

    @pytest.mark.parametrize("brut,attendu", [
        (12,    12),
        (99,    99),
        (110,   None),   # plafond : aberrant
        (121,   None),   # max observé dans les données réelles
        (None,  None),
        ("abc", None),
    ])
    def test_plafond(self, brut, attendu):
        assert normalize_age_max(brut) == attendu


# ─────────────────────────────────────────────────────────────────────────────
# clean_event (intégration des fonctions unitaires)
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanEvent:

    def test_structure_de_sortie(self, raw_event):
        result = clean_event(raw_event)
        assert result is not None
        champs = ["uid", "title", "description", "full_text", "city",
                  "region", "date_begin", "price", "attendance", "status", "url"]
        for champ in champs:
            assert champ in result, f"Champ manquant : {champ}"

    def test_valeurs_nettoyees(self, raw_event):
        result = clean_event(raw_event)
        assert result["uid"]        == "12345678"        # converti en string
        assert result["city"]       == "Paris"
        assert result["price"]      == "Gratuit"
        assert result["attendance"] == "Sur place"
        assert result["status"]     == "Programmé"
        assert "<p>" not in result["full_text"]          # HTML nettoyé

    def test_full_text_contient_les_sections(self, raw_event):
        full_text = clean_event(raw_event)["full_text"]
        for section in ["Titre :", "Description :", "Lieu :", "Date :", "Tarif :"]:
            assert section in full_text, f"Section manquante : {section}"

    def test_evenement_annule_exclu(self, raw_event):
        raw_event["status"] = '{"id": 6, "label": {"fr": "Annulé", "en": "Cancelled"}}'
        assert clean_event(raw_event) is None

    def test_evenement_vide_exclu(self):
        assert clean_event({"uid": 1, "title_fr": None,
                            "description_fr": None,
                            "longdescription_fr": None}) is None


# ─────────────────────────────────────────────────────────────────────────────
# load_raw (robustesse aux deux formats)
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadRaw:

    def test_format_liste(self, tmp_path):
        path = tmp_path / "raw.json"
        path.write_text(json.dumps([{"uid": 1}, {"uid": 2}]), encoding="utf-8")
        assert len(load_raw(path)) == 2

    def test_format_results(self, tmp_path):
        path = tmp_path / "raw.json"
        path.write_text(json.dumps({"results": [{"uid": 1}]}), encoding="utf-8")
        assert len(load_raw(path)) == 1

    def test_format_inconnu_leve_erreur(self, tmp_path):
        path = tmp_path / "raw.json"
        path.write_text(json.dumps({"autre": "format"}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_raw(path)
