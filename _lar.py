"""
_lar.py — locate the Lár-JEPA engine for Snath Aviation.

Resolution order:
  1. $SNATH_AVIATION_LARJEPA   — set this env var to point to your lar_jepa/ directory
  2. ../lar_jepa               — sibling directory (clone Lar-JEPA next to this repo)
  3. ~/lar_jepa                — home directory install

To install the engine:
  git clone https://github.com/snath-ai/Lar-JEPA.git
  export SNATH_AVIATION_LARJEPA=/path/to/Lar-JEPA/lar_jepa
"""
import os
import sys

_CANDIDATES = [
    os.environ.get("SNATH_AVIATION_LARJEPA"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Lar-JEPA", "lar_jepa")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lar_jepa")),
    os.path.expanduser("~/lar_jepa"),
]

_LAR_JEPA = None
for _p in _CANDIDATES:
    if _p and os.path.isfile(os.path.join(_p, "core", "interfaces.py")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        _LAR_JEPA = _p
        break

if not _LAR_JEPA:
    raise RuntimeError(
        "Lár-JEPA engine not found.\n"
        "Clone it and set the env var:\n"
        "  git clone https://github.com/snath-ai/Lar-JEPA.git\n"
        "  export SNATH_AVIATION_LARJEPA=/path/to/Lar-JEPA/lar_jepa"
    )

# Optional: Lár main package (compliance primitives).
# Only required for aviation_full_stack.py — not needed for core routing demos.
_LAR_MAIN_CANDIDATES = [
    os.environ.get("SNATH_LAR_MAIN"),
    os.path.abspath(os.path.join(os.path.dirname(_LAR_JEPA), "..")),
]
for _m in _LAR_MAIN_CANDIDATES:
    if _m and os.path.isdir(os.path.join(_m, "lar")):
        if _m not in sys.path:
            sys.path.insert(0, _m)
        break
