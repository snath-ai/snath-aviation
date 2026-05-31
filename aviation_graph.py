"""
Snath Aviation — divergence routing for aircraft sensors, built on the Lár engine.
"""
from __future__ import annotations
import numpy as np

import _lar  # noqa: F401
from core.interfaces import AbstractDivergenceRouter
from core.types import RouteDecision

# The shared basis/manifold decision classes (M1-M3)
DECISION_CLASSES = ("pitch_up", "level_flight", "pitch_down")
C = len(DECISION_CLASSES)

# ── Routing thresholds (provisional calibration) ──
TAU_HIGH = 0.35   # confidence floor to "act"
TAU_LOW  = 0.12   # below this a stream carries effectively no signal
DELTA    = 0.40   # basis (divergence) threshold separating agree / disagree

class AviationDivergenceRouter(AbstractDivergenceRouter):
    """
    Concrete AbstractDivergenceRouter (V1–V6) for aviation sensors.
    Takes latent representations from RadarEncoder and PitotEncoder.
    """

    def encode_stream_a(self, x_a):
        raise NotImplementedError(
            "Delegated to RadarEncoder; V1 stream independence enforced."
        )

    def encode_stream_b(self, x_b):
        raise NotImplementedError(
            "Delegated to PitotEncoder; V1 stream independence enforced."
        )

    def divergence(self, z_a, z_b) -> float:
        # V2-V3: Geometric divergence (Total Variation)
        z_a, z_b = np.asarray(z_a, float), np.asarray(z_b, float)
        return float(np.abs(z_a - z_b).sum() / np.sqrt(C))

    def route(self, confidence_a: float, confidence_b: float, divergence: float) -> RouteDecision:
        # V4-V6: Content-blind routing and Safety-Learning Equivalence
        c_a, c_b, d = confidence_a, confidence_b, divergence
        
        if max(c_a, c_b) >= TAU_HIGH and min(c_a, c_b) < TAU_LOW:
            return RouteDecision.COMMIT_TRAJECTORY
            
        if c_a < TAU_LOW and c_b < TAU_LOW:
            return RouteDecision.STRUCTURAL_IMPASSE
            
        if c_a >= TAU_HIGH and c_b >= TAU_HIGH:
            # The core of DAS: Confident disagreement triggers the alarm.
            if d >= DELTA:
                return RouteDecision.TRIGGER_REPLAN
            else:
                return RouteDecision.COMMIT_TRAJECTORY
                
        return RouteDecision.STRUCTURAL_IMPASSE

if __name__ == "__main__":
    r = AviationDivergenceRouter()
    print("Snath Aviation — AviationDivergenceRouter (AbstractDivergenceRouter V1–V6)")
    print("=" * 72)
    cases = [
        ("sensors agree -> keep flying",     [0.80, 0.10, 0.10], [0.80, 0.10, 0.10], 0.60, 0.60),
        ("pitot freezes -> alarm (replan)",  [0.80, 0.10, 0.10], [0.10, 0.10, 0.80], 0.60, 0.60),
        ("radar lost -> trust pitot",        [0.33, 0.33, 0.34], [0.80, 0.10, 0.10], 0.00, 0.60),
        ("storm noise -> impasse",           [0.33, 0.33, 0.34], [0.34, 0.33, 0.33], 0.00, 0.00),
    ]
    for label, va, vb, ca, cb in cases:
        d = r.divergence(np.array(va), np.array(vb))
        print(f"  {label:<32} D={d:.2f}  conf=({ca:.2f},{cb:.2f}) -> {r.route(ca, cb, d).value}")
        
    print("\nAviationDivergenceRouter is a subclass of the published AbstractDivergenceRouter:",
          issubclass(AviationDivergenceRouter, AbstractDivergenceRouter))
