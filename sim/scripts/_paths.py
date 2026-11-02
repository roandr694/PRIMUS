"""Puts control/ (the controller shared by the simulation and the real robot) on sys.path."""
import sys
from pathlib import Path

CONTROL = str(Path(__file__).resolve().parents[2] / "control")
if CONTROL not in sys.path:
    sys.path.insert(0, CONTROL)
