"""Ensure the repository root and src/ are on sys.path for local imports.

Usage (put at the top of scripts):
    import _path  # noqa: F401
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)
