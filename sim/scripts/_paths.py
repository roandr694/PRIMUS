"""Poe control/ (controlador compartilhado sim <-> robo real) no sys.path dos scripts da simulacao."""
import sys
from pathlib import Path

CONTROL = Path(__file__).resolve().parents[2] / "control"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))
