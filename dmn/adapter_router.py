"""
System 1 JSON Cache Router
Reads JSON centroid adapters and applies instant spatial overrides.
"""
import os
import json
import numpy as np
from core.types import RouteDecision

class AviationAdapterRouter:
    def __init__(self, adapter_dir: str = "models/adapters"):
        self.adapters = []
        if os.path.exists(adapter_dir):
            for f in os.listdir(adapter_dir):
                if f.endswith(".json"):
                    with open(os.path.join(adapter_dir, f), "r") as fp:
                        self.adapters.append(json.load(fp))

    def resolve(self, z_radar: np.ndarray, z_pitot: np.ndarray, base_decision: RouteDecision, c_radar: float, c_pitot: float) -> tuple[RouteDecision, str]:
        if base_decision != RouteDecision.TRIGGER_REPLAN:
            return base_decision, "Base decision was safe."

        for adapter in self.adapters:
            if adapter["type"] == "pitot_freeze":
                # L1 distance spatial check
                dist = np.sum(np.abs(z_radar - np.array(adapter["centroid_v_a"])))
                if dist < 0.2:
                    return RouteDecision.COMMIT_TRAJECTORY, "System 1 Cache Hit (Pitot Freeze detected). Overriding base decision."
            elif adapter["type"] == "gps_spoof":
                dist = np.sum(np.abs(z_pitot - np.array(adapter["centroid_v_b"])))
                if dist < 0.2:
                    return RouteDecision.COMMIT_TRAJECTORY, "System 1 Cache Hit (GPS Spoof detected). Overriding base decision."
                    
        return RouteDecision.TRIGGER_REPLAN, "System 1 Cache Miss. Falling back to System 2."
