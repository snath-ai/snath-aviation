"""
AviationAdapterRouter — Two-Component Adapter System
=====================================================
Implements the identification / correction trust asymmetry described in
"Architecture Is All You Need" (Sajeev 2026), §3.4 Remark (Temporal Decay
and Synaptic Depression):

  System 1 — Identification (trust-invariant)
  --------------------------------------------
  Centroid matching on the divergence vector fingerprint stored in each JSON
  sidecar.  The geometric signature of a failure class (e.g. pitot_freeze,
  gps_spoof) is durable — the underlying physics does not change with time.
  System 1 therefore fires regardless of how old the paired .pt adapter is.
  A centroid hit correctly names the fault class and overrides the base
  routing decision even when System 2 is fully stale.

  System 2 — Correction (perishable)
  ------------------------------------
  LoRA weights (.pt) encode learned routing corrections derived from a specific
  aircraft generation, altitude envelope, and sensor model.  These are
  perishable: a delta trained on one sensor variant may be wrong in sign for a
  successor variant several years later.  System 2 is therefore gated by the
  temporal trust score W = exp(-λ · Δt); adapters with W < min_trust are
  refused, and routing proceeds on System 1 logic alone.

  Degradation path
  ----------------
  When System 2 is refused the system still identifies the fault correctly
  (System 1 fires) and routes to COMMIT_TRAJECTORY; the audit note records
  both the identification event and the stale-adapter refusal.  This is the
  intended degradation path — identify correctly, correct conservatively —
  not a failure mode.

Derivative Works note
---------------------
This file is a Derivative Work of AbstractDivergenceRouter (V1–V6) and
JEPA_DMN_Consolidation_Node (Apache 2.0, github.com/snath-ai/Lar-JEPA).
The adapter routing pattern is proprietary, developed on personal hardware
outside employment.
"""
import os
import json
import math
import datetime
import glob
import numpy as np
from core.types import RouteDecision

# ── Temporal decay (mirrors Snath Basis / Snath Locus) ───────────────────────
# W = exp(-λ · Δt), Δt = fractional years since the adapter was trained.
_LAMBDA: dict = {
    "weather_induced": 0.50,   # ice / turbulence — highly seasonal, fast decay
    "hardware_struct": 0.02,   # manufacturing defect — slow decay
    "default":         0.10,
}


def _decay_weight(created_at_iso: str | None, failure_class: str = "default") -> float:
    """W = exp(-λ · Δt). Returns 1.0 if no timestamp (treat as current)."""
    if not created_at_iso:
        return 1.0
    try:
        created = datetime.datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
        now = datetime.datetime.now(datetime.timezone.utc)
        delta_years = (now - created).total_seconds() / (365.25 * 24 * 3600)
        lam = _LAMBDA.get(failure_class, _LAMBDA["default"])
        return math.exp(-lam * max(0.0, delta_years))
    except Exception:
        return 1.0


class AviationAdapterRouter:
    def __init__(self, adapter_dir: str = "models/adapters",
                 min_trust: float = 0.40):
        self.adapter_dir = adapter_dir
        self.min_trust = min_trust
        self.adapters = []
        if os.path.exists(adapter_dir):
            for f in os.listdir(adapter_dir):
                if f.endswith(".json"):
                    with open(os.path.join(adapter_dir, f), "r") as fp:
                        self.adapters.append(json.load(fp))

    def check_lora_trust(self, adapter_type: str) -> float:
        """
        Return temporal trust W for the .pt adapter of `adapter_type`.
        Reads created_at + failure_class from the .pt file metadata.
        Returns 1.0 if the file is missing or lacks temporal fields.
        """
        pt_path = os.path.join(self.adapter_dir, f"adapter_{adapter_type}.pt")
        if not os.path.exists(pt_path):
            return 1.0
        try:
            import torch as _torch
            meta = _torch.load(pt_path, map_location="cpu", weights_only=False)
            return _decay_weight(meta.get("created_at"), meta.get("failure_class", "default"))
        except Exception:
            return 1.0

    def resolve(self, z_radar: np.ndarray, z_pitot: np.ndarray, base_decision: RouteDecision, c_radar: float, c_pitot: float) -> tuple[RouteDecision, str]:
        if base_decision != RouteDecision.TRIGGER_REPLAN:
            return base_decision, "Base decision was safe."

        for adapter in self.adapters:
            if adapter["type"] == "pitot_freeze":
                # System 1 — Identification (trust-invariant).
                # The centroid match on z_radar has NO trust gate: the geometric
                # fingerprint of a pitot-freeze failure is durable regardless of
                # how old the paired .pt adapter is.  A hit here correctly names
                # the fault class even when System 2 is fully stale.
                dist = np.sum(np.abs(z_radar - np.array(adapter["centroid_v_a"])))
                if dist < 0.2:
                    # System 2 — Correction (perishable).
                    # Check the .pt temporal trust independently of the centroid
                    # match.  A stale adapter (W < min_trust) is refused; routing
                    # proceeds on System 1 logic alone.  The route decision
                    # (COMMIT_TRAJECTORY) is determined by System 1 either way —
                    # System 2 only sharpens the correction, it does not change
                    # the identification.
                    W = self.check_lora_trust("pitot_freeze")
                    lora_note = (f"LoRA W={W:.2f} — System 2 ready" if W >= self.min_trust
                                 else f"STALE LoRA W={W:.2f} < {self.min_trust} — System 1 only")
                    return RouteDecision.COMMIT_TRAJECTORY, (
                        f"System 1 Cache Hit (Pitot Freeze detected). "
                        f"Overriding base decision. | [System 2] {lora_note}"
                    )
            elif adapter["type"] == "gps_spoof":
                # System 1 — Identification (trust-invariant).
                # Same asymmetry applies: centroid match on z_pitot fires
                # regardless of .pt age; trust gate is applied to System 2 only.
                dist = np.sum(np.abs(z_pitot - np.array(adapter["centroid_v_b"])))
                if dist < 0.2:
                    W = self.check_lora_trust("gps_spoof")
                    lora_note = (f"LoRA W={W:.2f} — System 2 ready" if W >= self.min_trust
                                 else f"STALE LoRA W={W:.2f} < {self.min_trust} — System 1 only")
                    return RouteDecision.COMMIT_TRAJECTORY, (
                        f"System 1 Cache Hit (GPS Spoof detected). "
                        f"Overriding base decision. | [System 2] {lora_note}"
                    )

        return RouteDecision.TRIGGER_REPLAN, "System 1 Cache Miss. Falling back to System 2."
