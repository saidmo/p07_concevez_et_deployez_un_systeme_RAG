"""
clean_data.py
────────────────────────────────────────────────────────────────────────────
Étape 2 — Nettoyage et normalisation des événements bruts

Entrée  : data/raw/evenements_idf.json   (liste JSON produite par collect_data.py)
Sortie  : data/processed/evenements_idf_clean.json

Anomalies traitées (validées sur les 20 126 événements réels) :
  1. HTML dans longdescription_fr        → BeautifulSoup strip_tags (18 611 cas)
  2. Champs JSON sérialisés en string    → json.loads (status, attendancemode, timings)
  3. Titres avec préfixes parasites      → regex ("Annulé | ", "ANNULÉ - ", etc. — 25 cas)
  4. location_city null                  → reconstruite depuis location_address (21 cas)
  5. Événements annulés                  → exclus via status JSON (42) + titre (25)
  6. Doublons sur uid                    → dédoublonnage (0 dans ce jeu, sécurité)
  7. Sans titre ni description           → exclus (342 cas)
  8. conditions_fr normalisé             → "Gratuit" / prix extrait / "Sur inscription"
  9. age_max aberrant (>= 110)           → mis à None (56 cas, max observé : 121)

Usage (depuis la racine du projet) :
    python app/scripts/clean_data.py
    python app/scripts/clean_data.py --input data/raw/evenements_idf.json
────────────────────────────────────────────────────────────────────────────
"""

import json
import re
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_PATH      = Path("data/raw/evenements_idf.json")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH   = PROCESSED_DIR / "evenements_idf_clean.json"

AGE_MAX_PLAFOND = 110   # au-delà → valeur aberrante (max observé : 121)

# Regex : titres d'événements annulés ("Annulé | ", "ANNULÉE - ", "annulé : ", ...)
CANCELLED_RE = re.compile(r"^\s*annul[ée]{1,2}s?\s*[|:\-–—]?\s*", re.IGNORECASE)

# Regex : extraction de prix dans conditions_fr
PRICE_RE = re.compile(r"(\d+[\.,]?\d*)\s*€")

# Regex : ville après un code postal dans location_address ("33 rue X, 75018 Paris")
CITY_FROM_ADDRESS_RE = re.compile(r"\b\d{5}\s+([A-ZÀ-Ÿa-zà-ÿ'’\s\-]+)$")


# ── Fonctions de nettoyage unitaires ─────────────────────────────────────────

def strip_html(html: str | None) -> str:
    """Supprime les balises HTML et normalise les espaces."""
    if not html:
        return ""
    text = BeautifulSoup(html, "html.parser").get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def parse_json_field(value: Any) -> Any:
    """
    Désérialise un champ qui peut être une string JSON ou déjà parsé.
    Dans evenements_idf.json : status, attendancemode et timings sont
    TOUJOURS des strings JSON.
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def get_label_fr(value: Any) -> str:
    """
    Extrait le label français d'un champ structuré OpenAgenda
    de la forme {"id": 1, "label": {"fr": "Programmé", "en": "Scheduled"}}.
    """
    obj = parse_json_field(value)
    if isinstance(obj, dict):
        label = obj.get("label", {})
        if isinstance(label, dict):
            return label.get("fr", "") or ""
    return ""


def clean_title(title: str | None) -> str:
    """Supprime les préfixes parasites du titre."""
    if not title:
        return ""
    return CANCELLED_RE.sub("", title).strip()


def is_cancelled(event: dict) -> bool:
    """
    Détecte un événement annulé :
      - via le champ status (string JSON → label.fr contient 'annul')
      - via le titre commençant par 'Annulé'
    """
    title = event.get("title_fr") or ""
    if CANCELLED_RE.match(title):
        return True
    if "annul" in get_label_fr(event.get("status")).lower():
        return True
    return False


def extract_city(event: dict) -> str:
    """
    Récupère la ville depuis location_city en priorité,
    sinon extraction depuis location_address (pattern : '75018 Paris').
    """
    city = event.get("location_city")
    if city:
        return city
    address = event.get("location_address") or ""
    match = CITY_FROM_ADDRESS_RE.search(address)
    if match:
        return match.group(1).strip().title()
    return ""


def normalize_price(conditions: str | None) -> str:
    """Normalise les informations tarifaires."""
    if not conditions:
        return ""
    low = conditions.lower().strip()
    if any(w in low for w in ("gratuit", "libre", "free")):
        return "Gratuit"
    prices = PRICE_RE.findall(conditions)
    if prices:
        if len(prices) == 1:
            return f"{prices[0]}€"
        as_float = lambda p: float(p.replace(",", "."))
        return f"De {min(prices, key=as_float)}€ à {max(prices, key=as_float)}€"
    if any(w in low for w in ("inscription", "réservation", "reservation", "registration")):
        return "Sur inscription"
    return conditions.strip()


def normalize_age_max(age_max: Any) -> int | None:
    """Met à None les valeurs aberrantes (>= 110)."""
    if age_max is None:
        return None
    try:
        age = int(age_max)
    except (ValueError, TypeError):
        return None
    return age if age < AGE_MAX_PLAFOND else None


def build_text_content(event: dict) -> str:
    """
    Construit le champ texte principal destiné au chunking + vectorisation.
    Concatène : titre, description courte, description longue nettoyée,
    lieu, date, tarif, mode de participation, public.
    """
    parts = []

    title = clean_title(event.get("title_fr"))
    if title:
        parts.append(f"Titre : {title}")

    desc_short = (event.get("description_fr") or "").strip()
    if desc_short:
        parts.append(f"Description : {desc_short}")

    desc_long = strip_html(event.get("longdescription_fr"))
    if desc_long and desc_long != desc_short:
        parts.append(f"Détails : {desc_long}")

    city    = extract_city(event)
    venue   = event.get("location_name") or ""
    address = event.get("location_address") or ""
    region  = event.get("location_region") or ""
    location_parts = [p for p in (venue, address, city, region) if p]
    # Dédoublonner (la ville apparaît souvent déjà dans l'adresse)
    seen, loc = set(), []
    for p in location_parts:
        if p.lower() not in seen:
            seen.add(p.lower())
            loc.append(p)
    if loc:
        parts.append(f"Lieu : {', '.join(loc)}")

    date_str = event.get("daterange_fr") or ""
    if date_str:
        parts.append(f"Date : {date_str}")

    price = normalize_price(event.get("conditions_fr"))
    if price:
        parts.append(f"Tarif : {price}")

    attendance = get_label_fr(event.get("attendancemode"))
    if attendance:
        parts.append(f"Mode : {attendance}")

    age_min = event.get("age_min")
    age_max = normalize_age_max(event.get("age_max"))
    if age_min and age_max:
        parts.append(f"Public : {age_min} à {age_max} ans")
    elif age_min:
        parts.append(f"Public : dès {age_min} ans")

    return "\n".join(parts)


def clean_event(event: dict) -> dict | None:
    """
    Transforme un événement brut en document propre.
    Retourne None si l'événement doit être exclu (annulé ou vide).
    """
    if is_cancelled(event):
        return None

    title    = event.get("title_fr") or ""
    desc     = event.get("description_fr") or ""
    longdesc = event.get("longdescription_fr") or ""
    if not title and not desc and not longdesc:
        return None

    return {
        # Identifiants
        "uid"        : str(event.get("uid", "")),
        "slug"       : event.get("slug") or "",
        "url"        : event.get("canonicalurl") or "",

        # Contenu textuel
        "title"      : clean_title(title),
        "description": desc.strip(),
        "full_text"  : build_text_content(event),

        # Localisation (métadonnées pour le filtrage côté retriever)
        "city"       : extract_city(event),
        "department" : event.get("location_department") or "",
        "region"     : event.get("location_region") or "",
        "venue"      : event.get("location_name") or "",
        "address"    : event.get("location_address") or "",
        "postalcode" : event.get("location_postalcode") or "",

        # Dates
        "date_label" : event.get("daterange_fr") or "",
        "date_begin" : event.get("firstdate_begin") or "",
        "date_end"   : event.get("lastdate_end") or "",

        # Métadonnées utiles
        "price"      : normalize_price(event.get("conditions_fr")),
        "attendance" : get_label_fr(event.get("attendancemode")),
        "status"     : get_label_fr(event.get("status")),
        "age_min"    : event.get("age_min"),
        "age_max"    : normalize_age_max(event.get("age_max")),
        "image_url"  : event.get("thumbnail") or event.get("image") or "",
        "source"     : event.get("originagenda_title") or "",
        "keywords"   : event.get("keywords_fr") or "",
    }


# ── Pipeline principal ────────────────────────────────────────────────────────

def load_raw(raw_path: Path) -> list[dict]:
    """
    Charge le fichier brut. Gère les deux formats possibles :
      - liste directe [...]                  (format collect_data.py actuel)
      - dict {"results": [...]}              (ancien format API paginée)
    """
    log.info(f"Lecture de {raw_path}")
    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "results" in raw:
        return raw["results"]
    raise ValueError(f"Format inattendu dans {raw_path} : {type(raw).__name__}")


def clean_pipeline(raw_path: Path = RAW_PATH) -> tuple[list[dict], dict]:
    """Nettoie l'ensemble des événements. Retourne (events_propres, stats)."""
    events = load_raw(raw_path)
    log.info(f"{len(events)} événements bruts chargés")

    stats = {"total": len(events), "cancelled": 0, "empty": 0,
             "duplicates": 0, "ok": 0}
    seen_uids: set[str] = set()
    cleaned: list[dict] = []

    for event in tqdm(events, desc="Nettoyage", unit="evt"):
        uid = str(event.get("uid", ""))

        if uid and uid in seen_uids:
            stats["duplicates"] += 1
            continue
        seen_uids.add(uid)

        if is_cancelled(event):
            stats["cancelled"] += 1
            continue

        result = clean_event(event)
        if result is None:
            stats["empty"] += 1
        else:
            cleaned.append(result)
            stats["ok"] += 1

    log.info("── Rapport de nettoyage ──────────────────────")
    log.info(f"  Total brut        : {stats['total']}")
    log.info(f"  Conservés         : {stats['ok']}")
    log.info(f"  Annulés exclus    : {stats['cancelled']}")
    log.info(f"  Vides exclus      : {stats['empty']}")
    log.info(f"  Doublons exclus   : {stats['duplicates']}")
    if stats["total"]:
        log.info(f"  Taux conservation : {stats['ok'] / stats['total'] * 100:.1f}%")

    return cleaned, stats


def save_clean(events: list[dict], output_path: Path = OUTPUT_PATH) -> Path:
    """Sauvegarde les événements nettoyés avec métadonnées d'exécution."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "cleaned_at": datetime.now(timezone.utc).isoformat(),
        "total"     : len(events),
        "results"   : events,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    size_mb = output_path.stat().st_size / 1024 / 1024
    log.info(f"Sauvegardé : {output_path}  ({size_mb:.1f} Mo)")
    return output_path


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nettoyage des événements OpenAgenda")
    parser.add_argument("--input",  type=Path, default=RAW_PATH,    help="Fichier brut en entrée")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Fichier nettoyé en sortie")
    args = parser.parse_args()

    log.info("═══ Démarrage du nettoyage ═══")
    cleaned, _ = clean_pipeline(args.input)
    path = save_clean(cleaned, args.output)
    log.info(f"═══ Nettoyage terminé → {path} ═══")


if __name__ == "__main__":
    main()
