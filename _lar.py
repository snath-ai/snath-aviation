"""
_lar.py — connect Snath Aviation to the PUBLIC, genesis-anchored Lár-JEPA engine.
"""
import os
import sys

_CANDIDATES = [
    os.environ.get("SNATH_AVIATION_LARJEPA"),
    os.path.expanduser("~/Desktop/Lar_Main/lar_jepa"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lar_jepa")),
]

_LAR_JEPA = None
for _p in _CANDIDATES:
    if _p and os.path.isfile(os.path.join(_p, "core", "interfaces.py")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        _LAR_JEPA = _p
        break

if not _LAR_JEPA:
    raise RuntimeError("Could not locate the Lár-JEPA core (interfaces.py).")
    
# Connect to the main 'lar' repository for EU AI Act compliance primitives
_LAR_MAIN = os.path.expanduser("~/Desktop/Lar_Main")
if os.path.isdir(os.path.join(_LAR_MAIN, "lar")):
    if _LAR_MAIN not in sys.path:
        sys.path.insert(0, _LAR_MAIN)
