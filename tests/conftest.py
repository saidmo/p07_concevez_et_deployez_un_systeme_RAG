"""
conftest.py — configuration partagée pytest
Rend le dossier app/ importable depuis tests/ :
    from scripts.clean_data import ...
Lancer depuis la racine du projet :
    python -m pytest tests/ -v
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))
