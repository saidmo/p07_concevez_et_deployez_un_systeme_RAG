"""
collect_data.py — Collecte des événements OpenAgenda via OpenDataSoft

Source  : https://public.opendatasoft.com/explore/dataset/evenements-publics-openagenda
API     : Explore API v2.1 (Huwise / OpenDataSoft)
Domaine : hub.huwise.com

Usage :
    python scripts/collect_data.py
    python scripts/collect_data.py --region "Île-de-France" --output data/raw/evenements_idf.json
    python scripts/collect_data.py --days-history 365 --days-future 365

Résultat :
    Fichier JSON contenant tous les événements de la région sur la période définie.
    Par défaut : 1 an d'historique + 1 an à venir, Île-de-France.
"""

import json
import time
import argparse
import requests
from datetime import datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://hub.huwise.com/api/explore/v2.1/catalog/datasets"
DATASET  = "evenements-publics-openagenda"

DEFAULT_REGION      = "Île-de-France"
DEFAULT_DAYS_HISTORY = 365
DEFAULT_DAYS_FUTURE  = 365
DEFAULT_OUTPUT      = "data/raw/evenements_idf.json"


# ---------------------------------------------------------------------------
# Collecte
# ---------------------------------------------------------------------------

def collect_events(
    region: str,
    date_from: datetime,
    date_to: datetime,
    output_path: str,
) -> list[dict]:
    """
    Récupère tous les événements d'une région sur une période donnée
    via l'endpoint /exports/json (sans limite de records).

    Args:
        region      : Nom de la région (ex: "Île-de-France").
        date_from   : Date de début de la période.
        date_to     : Date de fin de la période.
        output_path : Chemin du fichier JSON de sortie.

    Returns:
        Liste de tous les événements collectés.
    """
    date_from_str = date_from.strftime("%Y-%m-%dT%H:%M:%S")
    date_to_str   = date_to.strftime("%Y-%m-%dT%H:%M:%S")

    where_clause = (
        f'location_region="{region}" '
        f'AND firstdate_begin >= "{date_from_str}" '
        f'AND firstdate_begin <= "{date_to_str}"'
    )

    params = {
        "where"   : where_clause,
        "order_by": "firstdate_begin ASC",
        "timezone": "Europe/Paris",
        "lang"    : "fr",
        "limit"   : -1,   # -1 = tout récupérer (spécifique à /exports)
    }

    print("=" * 60)
    print("Collecte des événements OpenAgenda")
    print("=" * 60)
    print(f"Région   : {region}")
    print(f"Période  : {date_from.date()} → {date_to.date()}")
    print(f"Sortie   : {output_path}")
    print(f"Filtre   : {where_clause}")
    print()
    print("Téléchargement en cours (export JSON sans limite)...")

    url = f"{BASE_URL}/{DATASET}/exports/json"
    start_time = time.time()

    response = requests.get(url, params=params, timeout=180, stream=True)

    if not response.ok:
        print(f"\nErreur {response.status_code} :")
        print(f"  URL    : {response.url}")
        print(f"  Détail : {response.text}")
        response.raise_for_status()

    # Lecture streaming avec progression
    chunks = []
    total_bytes = 0
    for chunk in response.iter_content(chunk_size=1024 * 64):
        if chunk:
            chunks.append(chunk)
            total_bytes += len(chunk)
            print(f"\r  Reçu : {total_bytes / 1024 / 1024:.1f} Mo", end="", flush=True)

    print()

    # Décodage et parsing JSON
    raw = b"".join(chunks).decode("utf-8")
    events = json.loads(raw)

    elapsed = time.time() - start_time
    print(f"\n✓ {len(events)} événements collectés en {elapsed:.1f}s")

    # Création du dossier de sortie si nécessaire
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarde
    output.write_text(
        json.dumps(events, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"✓ Fichier sauvegardé : {output.resolve()} ({size_mb:.1f} Mo)")

    # Statistiques rapides
    _print_stats(events)

    return events


def _print_stats(events: list[dict]) -> None:
    """Affiche quelques statistiques sur les événements collectés."""
    if not events:
        return

    cities = {}
    null_desc = 0
    null_long_desc = 0

    for e in events:
        city = e.get("location_city") or "Inconnue"
        cities[city] = cities.get(city, 0) + 1
        if not e.get("description_fr"):
            null_desc += 1
        if not e.get("longdescription_fr"):
            null_long_desc += 1

    top_cities = sorted(cities.items(), key=lambda x: x[1], reverse=True)[:5]

    print()
    print("── Statistiques ──────────────────────────────────────")
    print(f"  Total événements        : {len(events)}")
    print(f"  Sans description courte : {null_desc} ({null_desc/len(events)*100:.1f}%)")
    print(f"  Sans description longue : {null_long_desc} ({null_long_desc/len(events)*100:.1f}%)")
    print(f"  Top 5 villes            :")
    for city, count in top_cities:
        print(f"    {city:<30} {count:>5} événements")
    print("──────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collecte les événements OpenAgenda pour une région et une période données."
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"Région à collecter (défaut: '{DEFAULT_REGION}')"
    )
    parser.add_argument(
        "--days-history",
        type=int,
        default=DEFAULT_DAYS_HISTORY,
        help=f"Nombre de jours d'historique (défaut: {DEFAULT_DAYS_HISTORY})"
    )
    parser.add_argument(
        "--days-future",
        type=int,
        default=DEFAULT_DAYS_FUTURE,
        help=f"Nombre de jours à venir (défaut: {DEFAULT_DAYS_FUTURE})"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Chemin du fichier JSON de sortie (défaut: '{DEFAULT_OUTPUT}')"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    today     = datetime.now()
    date_from = today - timedelta(days=args.days_history)
    date_to   = today + timedelta(days=args.days_future)

    collect_events(
        region      = args.region,
        date_from   = date_from,
        date_to     = date_to,
        output_path = args.output,
    )
