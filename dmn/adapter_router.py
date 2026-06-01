"""
System 1 JSON Cache Router
Reads JSON centroid adapters and applies instant spatial overrides.
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
                dist = np.sum(np.abs(z_radar - np.array(adapter["centroid_v_a"])))
                if dist < 0.2:
                    W = self.check_lora_trust("pitot_freeze")
                    lora_note = (f"trust W={W:.2f}" if W >= self.min_trust
                                 else f"STALE LoRA W={W:.2f} < {self.min_trust} — skip System 2")
                    return RouteDecision.COMMIT_TRAJECTORY, (
                        f"System 1 Cache Hit (Pitot Freeze detected). "
                        f"Overriding base decision. | [System 2] {lora_note}"
                    )
            elif adapter["type"] == "gps_spoof":
                dist = np.sum(np.abs(z_pitot - np.array(adapter["centroid_v_b"])))
                if dist < 0.2:
                    W = self.check_lora_trust("gps_spoof")
                    lora_note = (f"trust W={W:.2f}" if W >= self.min_trust
                                 else f"STALE LoRA W={W:.2f} < {self.min_trust} — skip System 2")
                    return RouteDecision.COMMIT_TRAJECTORY, (
                        f"System 1 Cache Hit (GPS Spoof detected). "
                        f"Overriding base decision. | [System 2] {lora_note}"
                    )

        return RouteDecision.TRIGGER_REPLAN, "System 1 Cache Miss. Falling back to System 2."
