"""
Snath Aviation — DMN demo.

Seeds a synthetic history of aviation anomalies (Pitot Freezes), consolidates
them into a mock LoRA adapter, and then shows the system RESOLVE a brand-new
anomaly from memory automatically.

Run: python demo_dmn.py
"""
import os
import sys
import numpy as np

# Ensure DMN directory is discoverable
sys.path.insert(0, os.path.dirname(__file__))

from dhard import DHardQueue, DHardEvent
from dmn.aviation_dmn import AviationDMN
from dmn.adapter_router import AviationAdapterRouter
from aviation_graph import AviationDivergenceRouter

def seed_history(path: str, n_events: int = 5):
    """Simulate a history of resolved Pitot Freeze events."""
    q = DHardQueue(path)
    q.clear()
    
    for i in range(n_events):
        # Radar (Stream A) says level_flight (idx 1). Pitot (Stream B) says stall/pitch_down (idx 2)
        v_a = [0.1, 0.8, 0.1]
        v_b = [0.1, 0.1, 0.8]
        # Ground truth reveals Radar was right, Pitot froze.
        ev = DHardEvent(
            asof="2026-05-31", scenario_id=f"flight_10{i}", decision="TRIGGER_REPLAN",
            basis=0.81, conf_a=0.8, conf_b=0.8, v_a=v_a, v_b=v_b,
            realised_outcome="level_flight", realised_class="level_flight", winner="radar"
        ).sign()
        
        with open(path, "a") as f:
            import json
            from dataclasses import asdict
            f.write(json.dumps(asdict(ev)) + "\n")
            
    return n_events

if __name__ == "__main__":
    DEMO_PATH = "d_hard_demo.jsonl"
    n = seed_history(DEMO_PATH)
    print(f"Seeded {n} resolved aviation anomalies (synthetic history)\n" + "=" * 60)

    # 1. Run the DMN (Sleep Cycle)
    dmn = AviationDMN(queue_path=DEMO_PATH, adapter_dir="models/adapters")
    print("Consolidating D_hard -> LoRA Adapter library:")
    dmn.consolidate()

    print("\n" + "=" * 60)
    print("Resolving a NEW anomaly from memory (two-pass)")
    print("-" * 60)
    
    router = AviationDivergenceRouter()
    arouter = AviationAdapterRouter(adapter_dir="models/adapters")

    # A brand new flight where the pitot tube freezes again!
    v_a = np.array([0.05, 0.85, 0.10]); c_a = 0.85  # Radar confident in level flight
    v_b = np.array([0.10, 0.05, 0.85]); c_b = 0.85  # Pitot confident in stall
    
    d = router.divergence(v_a, v_b)
    base_dec = router.route(c_a, c_b, d)
    print(f"  Base Router (V1-V6):  D={d:.2f}  -> {base_dec.value}")
    
    final_dec, note = arouter.resolve(v_a, v_b, base_dec, c_a, c_b)
    print(f"  DMN-Resolved:         -> {final_dec.value}")
    print(f"  Why: {note}")
