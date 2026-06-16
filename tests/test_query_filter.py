"""
tests/test_query_filter.py
────────────────────────────────────────────────────────────────────────────
Tests du module d'extraction des contraintes (recherche hybride).

Couverture :
  - extraction de dates : jour explicite, début/mi/fin de mois, mois entier,
    année entière, relatif (aujourd'hui/demain/week-end), absence de date
  - extraction de lieu : départements (noms, variantes, codes), villes connues
  - gratuité : adjectif vs question fermée sur le tarif (piège ManiFeste)
  - event_matches : date dans/hors fenêtre, lieu, gratuité, normalisation
    des départements (codes postaux, variantes orthographiques)

100 % déterministe, aucune dépendance lourde (pas de FAISS ni de modèle).

Lancer :  python -m pytest tests/test_query_filter.py -v
────────────────────────────────────────────────────────────────────────────
"""

from datetime import date

import pytest

from scripts.query_filter import (
    parse_constraints, event_matches, Constraints,
)

TODAY = date(2026, 6, 16)   # date fixe pour rendre les tests relatifs stables
VILLES = {"Paris", "Versailles", "Montreuil", "Issy-les-Moulineaux",
          "Nanterre", "Aubervilliers", "Fontainebleau"}


def constr(q):
    return parse_constraints(q, known_cities=VILLES, today=TODAY)


# ── Extraction de dates ───────────────────────────────────────────────────────

class TestDates:

    def test_jour_explicite(self):
        c = constr("un concert le 21 juin 2026")
        assert c.date_start == date(2026, 6, 21)
        assert c.date_end   == date(2026, 6, 21)

    def test_jour_avec_1er(self):
        c = constr("que faire le 1er juillet 2026 ?")
        assert c.date_start == date(2026, 7, 1)
        assert c.date_end   == date(2026, 7, 1)

    def test_debut_de_mois(self):
        c = constr("des concerts début juillet 2026")
        assert c.date_start == date(2026, 7, 1)
        assert c.date_end   == date(2026, 7, 10)

    def test_fin_de_mois(self):
        c = constr("une expo fin mars 2026")
        assert c.date_start == date(2026, 3, 21)
        assert c.date_end   == date(2026, 3, 31)

    def test_mois_entier(self):
        c = constr("un spectacle en juillet 2026")
        assert c.date_start == date(2026, 7, 1)
        assert c.date_end   == date(2026, 7, 31)

    def test_fevrier_annee_bissextile(self):
        c = constr("un atelier en février 2028")
        assert c.date_end == date(2028, 2, 29)   # 2028 bissextile

    def test_annee_entiere(self):
        c = constr("la Fête de la musique en 2026")
        assert c.date_start == date(2026, 1, 1)
        assert c.date_end   == date(2026, 12, 31)

    def test_demain(self):
        c = constr("un concert demain")
        assert c.date_start == date(2026, 6, 17)
        assert c.date_end   == date(2026, 6, 17)

    def test_ce_week_end(self):
        c = constr("que faire ce week-end ?")
        assert c.date_start == date(2026, 6, 20)   # samedi
        assert c.date_end   == date(2026, 6, 21)   # dimanche

    def test_aucune_date(self):
        c = constr("des spectacles de marionnettes")
        assert c.date_start is None and c.date_end is None

    def test_priorite_jour_sur_mois(self):
        # « 21 juin 2026 » doit donner le jour, pas le mois entier
        c = constr("le 21 juin 2026")
        assert c.date_start == c.date_end == date(2026, 6, 21)


# ── Extraction de lieu ────────────────────────────────────────────────────────

class TestLieu:

    def test_departement_nom(self):
        c = constr("des concerts dans les Yvelines")
        assert "Yvelines" in c.departments

    def test_departement_variante(self):
        c = constr("une expo en Seine-St-Denis")
        assert "Seine-Saint-Denis" in c.departments

    def test_ville_connue(self):
        c = constr("un spectacle à Versailles")
        assert "versailles" in c.cities

    def test_ville_inconnue_ignoree(self):
        # Marseille n'est pas dans le corpus → aucune contrainte de lieu
        c = constr("des concerts à Marseille")
        assert not c.cities and not c.departments

    def test_paris_ville_et_departement(self):
        c = constr("un concert à Paris")
        # Paris peut être capté comme ville et/ou département : au moins l'un
        assert ("paris" in c.cities) or ("Paris" in c.departments)


# ── Gratuité ──────────────────────────────────────────────────────────────────

class TestGratuite:

    def test_adjectif_gratuit(self):
        assert constr("un concert gratuit à Paris").free is True

    def test_gratuitement(self):
        assert constr("que faire gratuitement ce week-end ?").free is True

    def test_question_fermee_pas_de_filtre(self):
        # Piège ManiFeste : « est-il en accès libre ? » ne doit PAS filtrer
        assert constr("le festival ManiFeste-2026 est-il en accès libre ?").free is False

    def test_question_est_il_gratuit(self):
        assert constr("ce concert est-il gratuit ?").free is False

    def test_pas_de_gratuite(self):
        assert constr("un concert de jazz à Paris").free is False


# ── event_matches ─────────────────────────────────────────────────────────────

def meta(date_begin="2026-07-01T19:30:00+02:00", city="Paris",
         department="Paris", price="Gratuit"):
    return {"date_begin": date_begin, "city": city,
            "department": department, "price": price}


class TestEventMatches:

    def test_contraintes_vides_tout_passe(self):
        assert event_matches(meta(), Constraints()) is True

    def test_date_dans_fenetre(self):
        c = Constraints(date_start=date(2026, 7, 1), date_end=date(2026, 7, 31))
        assert event_matches(meta(date_begin="2026-07-15T20:00:00+02:00"), c)

    def test_date_hors_fenetre(self):
        c = Constraints(date_start=date(2026, 7, 1), date_end=date(2026, 7, 31))
        assert not event_matches(meta(date_begin="2025-07-15T20:00:00+02:00"), c)

    def test_date_invalide_rejetee(self):
        c = Constraints(date_start=date(2026, 7, 1), date_end=date(2026, 7, 31))
        assert not event_matches(meta(date_begin=""), c)

    def test_ville_match(self):
        c = Constraints(cities={"versailles"})
        assert event_matches(meta(city="Versailles", department="Yvelines"), c)

    def test_ville_non_match(self):
        c = Constraints(cities={"versailles"})
        assert not event_matches(meta(city="Paris", department="Paris"), c)

    def test_departement_par_code_postal(self):
        # department = « 75012 » doit être reconnu comme Paris
        c = Constraints(departments={"Paris"})
        assert event_matches(meta(city="Paris", department="75012"), c)

    def test_departement_variante_orthographe(self):
        c = Constraints(departments={"Seine-Saint-Denis"})
        assert event_matches(meta(city="Aubervilliers", department="Seine-St-Denis"), c)

    def test_ville_ou_departement_suffit(self):
        # ville demandée + département demandé : l'un des deux suffit
        c = Constraints(cities={"nanterre"}, departments={"Paris"})
        assert event_matches(meta(city="Nanterre", department="Hauts-de-Seine"), c)

    def test_gratuit_match(self):
        c = Constraints(free=True)
        assert event_matches(meta(price="Gratuit"), c)

    def test_gratuit_non_match(self):
        c = Constraints(free=True)
        assert not event_matches(meta(price="De 10€ à 19€"), c)
        assert not event_matches(meta(price="Sur inscription"), c)

    def test_combinaison_et(self):
        # date ET lieu ET gratuit : tout doit matcher
        c = Constraints(date_start=date(2026, 6, 21), date_end=date(2026, 6, 21),
                        cities={"paris"}, free=True)
        ok  = meta(date_begin="2026-06-21T16:30:00+02:00", city="Paris", price="Gratuit")
        ko  = meta(date_begin="2026-06-21T16:30:00+02:00", city="Paris", price="10€")
        assert event_matches(ok, c)
        assert not event_matches(ko, c)
