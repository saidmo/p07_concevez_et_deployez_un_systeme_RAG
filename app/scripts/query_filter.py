"""
query_filter.py — Extraction déterministe des contraintes d'une question

Le retrieval vectoriel pur ne sait pas honorer les contraintes « dures » d'une
question (date, ville, département, gratuité) : il rapproche sémantiquement
« concert de jazz à Paris en juillet 2026 » de tous les événements de jazz,
sans distinguer 2025 de 2026 ni Paris de Louvres.

Ce module extrait ces contraintes par des règles déterministes (regex +
gazetteer des départements franciliens), pour permettre un filtrage des
candidats AVANT de les passer au LLM (recherche hybride : sémantique + filtre
sur métadonnées).

Choix assumés :
  - Déterministe (pas d'appel LLM) : prévisible, gratuit, testable unitairement.
  - Dégradation gracieuse : aucune contrainte détectée → Constraints "vide",
    le retrieval se comporte comme avant (sémantique pur).
  - On ne gère que des dates ABSOLUES + quelques relatives simples : le jeu de
    test est volontairement en dates absolues (le LLM, lui, reçoit la date du
    jour pour le reste).

Fonctions publiques :
  - parse_constraints(question, known_departments, today) -> Constraints
  - event_matches(metadata, constraints) -> bool
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta


# ── Gazetteer des départements d'Île-de-France ───────────────────────────────
# Forme canonique → variantes/orthographes rencontrées dans les données et les
# questions. La normalisation (sans accents, minuscules) est appliquée des deux
# côtés avant comparaison.

DEPARTEMENTS_IDF = {
    "Paris"             : ["paris", "75"],
    "Seine-et-Marne"    : ["seine-et-marne", "seine et marne", "77"],
    "Yvelines"          : ["yvelines", "78"],
    "Essonne"           : ["essonne", "91"],
    "Hauts-de-Seine"    : ["hauts-de-seine", "hauts de seine", "92"],
    "Seine-Saint-Denis" : ["seine-saint-denis", "seine saint denis",
                            "seine-st-denis", "seine-st.-denis", "93"],
    "Val-de-Marne"      : ["val-de-marne", "val de marne", "94"],
    "Val-d'Oise"        : ["val-d'oise", "val d'oise", "val-doise", "95"],
}

# Préfixe de code postal → département (pour les valeurs « 75012 » etc.)
CP_PREFIXE_DEPT = {
    "75": "Paris", "77": "Seine-et-Marne", "78": "Yvelines",
    "91": "Essonne", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne", "95": "Val-d'Oise",
}

MOIS_FR = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
}


def _norm(texte: str) -> str:
    """Minuscule, sans accents, espaces normalisés — pour comparer ville/dept."""
    if not texte:
        return ""
    texte = unicodedata.normalize("NFD", texte)
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", texte.lower()).strip()


@dataclass
class Constraints:
    """Contraintes extraites d'une question. Tous les champs sont optionnels."""
    date_start: date | None = None          # borne inférieure (incluse)
    date_end:   date | None = None          # borne supérieure (incluse)
    departments: set[str] = field(default_factory=set)   # formes canoniques
    cities:      set[str] = field(default_factory=set)    # normalisées (_norm)
    free: bool = False                       # « gratuit » demandé explicitement

    def is_empty(self) -> bool:
        return (self.date_start is None and self.date_end is None
                and not self.departments and not self.cities and not self.free)


# ── Extraction des dates ──────────────────────────────────────────────────────

def _extract_dates(q: str, today: date) -> tuple[date | None, date | None]:
    """
    Détecte, par ordre de priorité décroissante :
      1. jour explicite       « 21 juin 2026 », « le 1er juillet 2026 »
      2. début/mi/fin de mois « début juillet 2026 »
      3. mois entier          « juillet 2026 », « en juin 2026 »
      4. année entière        « en 2026 »
      5. relatif simple       « aujourd'hui », « demain », « ce week-end »
    Retourne (date_start, date_end) inclusives, ou (None, None).
    """
    ql = _norm(q)
    mois_pat = "|".join(MOIS_FR)

    # 1. Jour explicite : (le) (1er|21) juin 2026
    m = re.search(rf"\b(\d{{1,2}})\s*(?:er)?\s+({mois_pat})\s+(\d{{4}})\b", ql)
    if m:
        jour, mois, annee = int(m.group(1)), MOIS_FR[m.group(2)], int(m.group(3))
        try:
            d = date(annee, mois, jour)
            return d, d
        except ValueError:
            pass

    # 2. Début / mi / fin de mois
    m = re.search(rf"\b(début|debut|mi|milieu|fin)\s+(?:de\s+|du\s+)?({mois_pat})\s+(\d{{4}})\b", ql)
    if m:
        portion, mois, annee = m.group(1), MOIS_FR[m.group(2)], int(m.group(3))
        dernier = _dernier_jour(annee, mois)
        if portion in ("début", "debut"):
            return date(annee, mois, 1), date(annee, mois, 10)
        if portion in ("mi", "milieu"):
            return date(annee, mois, 11), date(annee, mois, 20)
        return date(annee, mois, 21), date(annee, mois, dernier)   # fin

    # 3. Mois entier : (en) juillet 2026
    m = re.search(rf"\b(?:en\s+)?({mois_pat})\s+(\d{{4}})\b", ql)
    if m:
        mois, annee = MOIS_FR[m.group(1)], int(m.group(2))
        return date(annee, mois, 1), date(annee, mois, _dernier_jour(annee, mois))

    # 4. Année entière : (en) 2026
    m = re.search(r"\b(?:en\s+|année\s+|annee\s+)?(20\d{2})\b", ql)
    if m:
        annee = int(m.group(1))
        return date(annee, 1, 1), date(annee, 12, 31)

    # 5. Relatif simple
    if "aujourd'hui" in ql or "aujourdhui" in ql:
        return today, today
    if "demain" in ql:
        d = today + timedelta(days=1)
        return d, d
    if "ce week-end" in ql or "ce weekend" in ql or "ce week end" in ql:
        # samedi et dimanche de la semaine courante (lundi=0 … dimanche=6)
        samedi = today + timedelta(days=(5 - today.weekday()) % 7)
        return samedi, samedi + timedelta(days=1)
    if "cette semaine" in ql:
        lundi = today - timedelta(days=today.weekday())
        return lundi, lundi + timedelta(days=6)

    return None, None


def _dernier_jour(annee: int, mois: int) -> int:
    if mois == 12:
        suivant = date(annee + 1, 1, 1)
    else:
        suivant = date(annee, mois + 1, 1)
    return (suivant - timedelta(days=1)).day


# ── Extraction lieu / gratuité ────────────────────────────────────────────────

def _extract_departments(q: str) -> set[str]:
    ql = _norm(q)
    trouves = set()
    for canon, variantes in DEPARTEMENTS_IDF.items():
        for v in variantes:
            # Codes (75…95) : éviter les faux positifs (ex. « 2026 ») via \b
            if v.isdigit():
                if re.search(rf"\b{v}\b", ql):
                    trouves.add(canon)
            elif _norm(v) in ql:
                trouves.add(canon)
    return trouves


def _extract_cities(q: str, known_cities_norm: dict[str, str]) -> set[str]:
    """
    Repère les villes connues (issues des données) citées dans la question.
    known_cities_norm : { ville_normalisée : ville_normalisée } — on matche
    sur les villes d'au moins 4 caractères pour éviter le bruit, par mot entier.
    """
    ql = _norm(q)
    trouves = set()
    for ville_norm in known_cities_norm:
        if len(ville_norm) >= 4 and re.search(rf"\b{re.escape(ville_norm)}\b", ql):
            trouves.add(ville_norm)
    return trouves


def _wants_free(q: str) -> bool:
    """
    Vrai si la question demande de FILTRER sur les événements gratuits
    (« concert gratuit », « que faire gratuitement »).

    Faux pour une question fermée SUR le tarif d'un sujet nommé
    (« le festival X est-il gratuit / en accès libre ? ») : là, l'utilisateur
    interroge un événement précis, il ne faut pas exclure les autres tarifs.
    """
    ql = _norm(q)
    a_le_mot = bool(re.search(r"\bgratuit", ql))     # gratuit, gratuite, gratuitement
    question_sur_tarif = bool(re.search(
        r"est-(il|elle|ce)\b|acces libre|en acces", ql
    ))
    return a_le_mot and not question_sur_tarif


# ── API publique ──────────────────────────────────────────────────────────────

def parse_constraints(
    question: str,
    known_cities: set[str] | None = None,
    today: date | None = None,
) -> Constraints:
    """
    Extrait les contraintes dures d'une question.
    known_cities : ensemble des villes du corpus (pour le matching ville) ;
    si None, seules les contraintes date/département/gratuité sont extraites.
    """
    today = today or date.today()
    known_cities_norm = {_norm(c): _norm(c) for c in (known_cities or set())}

    date_start, date_end = _extract_dates(question, today)
    departments = _extract_departments(question)
    cities = _extract_cities(question, known_cities_norm)
    # Une ville détectée implique son département via les données (fait côté
    # event_matches : on garde ville ET département comme contraintes OR-compatibles)

    return Constraints(
        date_start=date_start,
        date_end=date_end,
        departments=departments,
        cities=cities,
        free=_wants_free(question),
    )


def _date_begin_to_date(date_begin: str) -> date | None:
    """'2026-07-01T19:30:00+02:00' → date(2026, 7, 1). Tolérant."""
    if not date_begin or len(date_begin) < 10:
        return None
    try:
        return date.fromisoformat(date_begin[:10])
    except ValueError:
        return None


def _canon_department(valeur: str) -> str:
    """Normalise une valeur de département du corpus vers sa forme canonique."""
    v = _norm(valeur)
    if v.isdigit() and len(v) == 5:           # code postal → préfixe
        return CP_PREFIXE_DEPT.get(v[:2], valeur)
    for canon, variantes in DEPARTEMENTS_IDF.items():
        if v == _norm(canon) or v in (_norm(x) for x in variantes):
            return canon
    return valeur


def event_matches(metadata: dict, c: Constraints) -> bool:
    """
    Un événement (via ses métadonnées d'index) satisfait-il les contraintes ?
    Logique ET entre les types de contrainte, OU à l'intérieur d'un type
    (plusieurs départements/villes acceptés). Lieu : ville OU département
    suffit (une ville détectée et un département détecté sont compatibles).
    """
    # Date : l'événement doit commencer dans la fenêtre [start, end]
    if c.date_start or c.date_end:
        d = _date_begin_to_date(metadata.get("date_begin", ""))
        if d is None:
            return False
        if c.date_start and d < c.date_start:
            return False
        if c.date_end and d > c.date_end:
            return False

    # Lieu : si ville(s) ET/OU département(s) demandés, au moins un doit matcher
    if c.cities or c.departments:
        ville_ev = _norm(metadata.get("city", ""))
        dept_ev  = _canon_department(metadata.get("department", ""))
        ok_ville = ville_ev in c.cities if c.cities else False
        ok_dept  = dept_ev in c.departments if c.departments else False
        if not (ok_ville or ok_dept):
            return False

    # Gratuité : tarif normalisé « Gratuit »
    if c.free:
        if _norm(metadata.get("price", "")) != "gratuit":
            return False

    return True
