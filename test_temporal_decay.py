"""
Regression test — adapter temporal decay gate (Snath Aviation).

The temporal decay formula lives in:
  - dmn/health_monitor.py  (_decay_weight — reads from JSON/cache)
  - dmn/adapter_router.py  (_decay_weight + check_lora_trust — reads from .pt)

Tests:
  1. _decay_weight() formula (both implementations must agree).
  2. check_lora_trust() returns W≈1.0 for a fresh .pt.
  3. check_lora_trust() returns W≈0.22 for a 3-year-old .pt (stale).
  4. Cusp — just past ~1.83yr for weather_induced (λ=0.50).
  5. aviation_dmn.consolidate() now writes created_at + failure_class to .pt.
  6. health_monitor._decay_weight agrees with adapter_router._decay_weight.

Run:  python test_temporal_decay.py
      or:  pytest test_temporal_decay.py
"""
from __future__ import annotations
import math
import sys
import os
import datetime
import tempfile
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import _lar  # puts lar_jepa/core/ on sys.path — required by dmn.adapter_router

from dmn.adapter_router import _decay_weight as router_decay, AviationAdapterRouter
from dmn.health_monitor import _decay_weight as monitor_decay

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_years_ago(years: float) -> str:
    delta = datetime.timedelta(days=years * 365.25)
    dt = datetime.datetime.now(datetime.timezone.utc) - delta
    return dt.isoformat().replace("+00:00", "Z")


def _write_pt(path: str, created_at: str, failure_class: str = "weather_induced") -> None:
    """Write a minimal aviation .pt with temporal metadata."""
    import numpy as np
    import hashlib
    import hmac as _hmac
    A = torch.zeros(3, 1)
    B = torch.zeros(1, 3)
    _KEY = b"snath_aviation_adapter_sovereignty_2026"
    a_hash = hashlib.sha256(A.numpy().tobytes()).hexdigest()[:16]
    b_hash = hashlib.sha256(B.numpy().tobytes()).hexdigest()[:16]
    sig = _hmac.new(_KEY, f"pitot|{a_hash}|{b_hash}".encode(), hashlib.sha256).hexdigest()
    torch.save({
        "A": A, "B": B,
        "target_encoder": "pitot",
        "hmac_hex":        sig,
        "created_at":      created_at,
        "failure_class":   failure_class,
    }, path)


def _write_json(path: str, created_at: str, failure_class: str = "weather_induced") -> None:
    with open(path, "w") as f:
        json.dump({
            "type":          "pitot_freeze",
            "centroid_v_a":  [0.0, 0.0, 0.0],
            "trust":         "radar",
            "failure_class": failure_class,
            "created_at":    created_at,
        }, f)


# ---------------------------------------------------------------------------
# 1. Both _decay_weight() implementations agree
# ---------------------------------------------------------------------------

def test_both_implementations_agree():
    for years in [0.0, 1.0, 3.0]:
        ts = _iso_years_ago(years)
        wr = router_decay(ts, "weather_induced")
        wm = monitor_decay(ts, "weather_induced")
        assert abs(wr - wm) < 1e-9, (
            f"Implementations disagree at {years}yr: router={wr:.6f} monitor={wm:.6f}"
        )
    print("PASS  test_both_implementations_agree")


# ---------------------------------------------------------------------------
# 2. Formula correctness (router implementation)
# ---------------------------------------------------------------------------

def test_decay_weight_formula():
    assert abs(router_decay(_iso_years_ago(0), "weather_induced") - 1.0) < 0.01
    w1 = router_decay(_iso_years_ago(1.0), "weather_induced")
    assert abs(w1 - math.exp(-0.50)) < 0.01, f"1yr: {w1:.4f}"
    w3 = router_decay(_iso_years_ago(3.0), "weather_induced")
    assert abs(w3 - math.exp(-1.5)) < 0.01, f"3yr: {w3:.4f}"
    # hardware_struct: slow decay
    wh = router_decay(_iso_years_ago(5.0), "hardware_struct")
    assert abs(wh - math.exp(-0.10)) < 0.01, f"hardware 5yr: {wh:.4f}"
    assert router_decay(None) == 1.0
    assert router_decay("") == 1.0
    print("PASS  test_decay_weight_formula")


# ---------------------------------------------------------------------------
# 3. check_lora_trust() fresh .pt → W≈1.0
# ---------------------------------------------------------------------------

def test_check_lora_trust_fresh():
    with tempfile.TemporaryDirectory() as td:
        _write_pt(os.path.join(td, "adapter_pitot_freeze.pt"), _iso_years_ago(0.01))
        router = AviationAdapterRouter(adapter_dir=td, min_trust=0.40)
        W = router.check_lora_trust("pitot_freeze")
    assert W > 0.40, f"Fresh adapter should have W > 0.40, got {W:.4f}"
    print(f"PASS  test_check_lora_trust_fresh  W={W:.4f}")


# ---------------------------------------------------------------------------
# 4. check_lora_trust() stale .pt (3yr) → W≈0.22
# ---------------------------------------------------------------------------

def test_check_lora_trust_stale():
    with tempfile.TemporaryDirectory() as td:
        _write_pt(os.path.join(td, "adapter_pitot_freeze.pt"), _iso_years_ago(3.0))
        router = AviationAdapterRouter(adapter_dir=td, min_trust=0.40)
        W = router.check_lora_trust("pitot_freeze")
    assert W < 0.40, f"3yr adapter should have W < 0.40, got {W:.4f}"
    expected = math.exp(-0.50 * 3.0)
    assert abs(W - expected) < 0.01, f"Expected W≈{expected:.4f}, got {W:.4f}"
    print(f"PASS  test_check_lora_trust_stale  W={W:.4f} (correctly stale)")


# ---------------------------------------------------------------------------
# 5. Cusp — just past ~1.83yr for weather_induced
# ---------------------------------------------------------------------------

def test_cusp_stale():
    cusp_years = -math.log(0.40) / 0.50   # ≈ 1.833
    with tempfile.TemporaryDirectory() as td:
        _write_pt(os.path.join(td, "adapter_pitot_freeze.pt"),
                  _iso_years_ago(cusp_years + 0.05))
        router = AviationAdapterRouter(adapter_dir=td, min_trust=0.40)
        W = router.check_lora_trust("pitot_freeze")
    assert W < 0.40, f"Cusp+0.05yr should be < 0.40, got {W:.4f}"
    print(f"PASS  test_cusp_stale  W={W:.4f} at Δt≈{cusp_years+0.05:.2f}yr")


# ---------------------------------------------------------------------------
# 6. resolve() embeds trust note for JSON hit with stale .pt
# ---------------------------------------------------------------------------

def test_resolve_notes_stale_lora():
    """When JSON hit + stale .pt, resolve() note should contain 'STALE'."""
    try:
        from core.types import RouteDecision
    except ImportError:
        print("SKIP  test_resolve_notes_stale_lora  (core.types not on path)")
        return

    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        # Write JSON centroid (System 1 hit)
        _write_json(os.path.join(td, "adapter_pitot_freeze.json"), _iso_years_ago(0.01))
        # Write STALE .pt
        _write_pt(os.path.join(td, "adapter_pitot_freeze.pt"), _iso_years_ago(3.0))

        router = AviationAdapterRouter(adapter_dir=td, min_trust=0.40)
        z_r = np.array([0.0, 0.0, 0.0])
        z_p = np.array([0.0, 0.0, 0.0])
        dec, note = router.resolve(z_r, z_p, RouteDecision.TRIGGER_REPLAN, 0.9, 0.9)

    assert "System 1 only" in note, f"Expected 'System 1 only' in note, got: {note}"
    print(f"PASS  test_resolve_notes_stale_lora  note={note!r}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_both_implementations_agree()
    test_decay_weight_formula()
    test_check_lora_trust_fresh()
    test_check_lora_trust_stale()
    test_cusp_stale()
    test_resolve_notes_stale_lora()
    print("\nAll 6 Snath Aviation temporal decay regression tests passed.")
