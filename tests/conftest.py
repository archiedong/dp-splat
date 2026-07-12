"""Test config: float64 everywhere (brief §10.3 — the Beta/digamma path and exact
monotonicity checks need it) and src-layout import path."""

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for oracle_numpy
