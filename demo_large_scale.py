"""
Large Scale Kahneman Hybrid Architecture Validation (N=2000)

Validates the architecture mathematically at scale.
Phase 1: Generates 1,000 synthetic flights, triggering D_hard events.
Phase 2: DMN Sleep Cycle trains JSON & PyTorch LoRAs on both Pitot and GPS failures.
Phase 3: Validates 1,000 new flights against the Tiered Defense Cascade.
"""

import os
import sys
import _lar
import json
import torch
import random
import numpy as np
from dataclasses import asdict
from core.interfaces import AbstractModalEncoder
from core.types import RouteDecision
from aviation_graph import AviationDivergenceRouter
from dhard import DHardEvent
from dmn.aviation_dmn import AviationDMN
from dmn.adapter_router import AviationAdapterRouter
from demo_real_world import RadarEncoder, PitotEncoder

def generate_flight(anomaly_type=None):
    base_v = random.uniform(200.0, 260.0)
    base_a = random.uniform(9000.0, 11000.0)
    
    if anomaly_type == "pitot_freeze":
        return {"radar_v": base_v, "radar_a": base_a, "pitot_v": 0.0, "pitot_a": base_a}
    elif anomaly_type == "gps_spoof":
        return {"radar_v": 50.0, "radar_a": 1000.0, "pitot_v": base_v, "pitot_a": base_a}
    else:
        return {"radar_v": base_v, "radar_a": base_a, "pitot_v": base_v, "pitot_a": base_a}

def main():
    print("="*60)
    print("LARGE-SCALE VALIDATION: LÁR-JEPA HYBRID CASCADE")
    print("="*60)
    
    # Init Encoders and Router
    radar = RadarEncoder()
    pitot = PitotEncoder()
    router = AviationDivergenceRouter()
    
    d_hard_file = "d_hard_large.jsonl"
    if os.path.exists(d_hard_file):
        os.remove(d_hard_file)
        
    print("\n[PHASE 1] Raw Fleet Ingestion (N=1000)")
    anomaly_types = (["none"] * 500) + (["pitot_freeze"] * 250) + (["gps_spoof"] * 250)
    random.shuffle(anomaly_types)
    
    d_hard_count = 0
    with open(d_hard_file, "w") as f:
        for i, atype in enumerate(anomaly_types):
            flight = generate_flight(atype)
            
            # For encoding, the encoders currently expect dicts with 'velocity' and 'altitude'
            # Let's wrap them
            radar_tele = {'velocity': flight['radar_v'], 'altitude': flight['radar_a']}
            pitot_tele = {'velocity': flight['pitot_v'], 'altitude': flight['pitot_a']}
            
            z_r = radar.encode(radar_tele)
            z_p = pitot.encode(pitot_tele)
            div = router.divergence(z_r, z_p)
            dec = router.route(0.98, 0.90, div)
            
            if dec == RouteDecision.TRIGGER_REPLAN:
                d_hard_count += 1
                winner = "radar" if atype == "pitot_freeze" else "pitot"
                ev = DHardEvent(
                    asof="2026-05-31", scenario_id=f"sim_{i}", decision="TRIGGER_REPLAN",
                    basis=0.90, conf_a=0.98, conf_b=0.90, v_a=z_r.tolist(), v_b=z_p.tolist(),
                    realised_outcome="sim", realised_class="sim", winner=winner
                ).sign()
                f.write(json.dumps(asdict(ev)) + "\n")
                
    print(f"    Processed 1000 flights. Detected {d_hard_count} severe anomalies.")
    
    print("\n[PHASE 2] DMN Sleep Cycle")
    dmn = AviationDMN(queue_path=d_hard_file, adapter_dir="models/adapters_large")
    dmn.consolidate()
    
    print("\n[PHASE 3] Holdout Fleet Validation (N=1000)")
    # New flights
    holdout_types = (["none"] * 500) + (["pitot_freeze"] * 250) + (["gps_spoof"] * 250)
    
    sys1_router = AviationAdapterRouter(adapter_dir="models/adapters_large")
    
    # Initialize metrics
    metrics = {
        "sys1_hits": 0,
        "base_div_sum": 0.0,
        "sys2_div_sum": 0.0,
        "anomalies_evaluated": 0
    }
    
    for i, atype in enumerate(holdout_types):
        flight = generate_flight(atype)
        radar_tele = {'velocity': flight['radar_v'], 'altitude': flight['radar_a']}
        pitot_tele = {'velocity': flight['pitot_v'], 'altitude': flight['pitot_a']}
        
        # Ensure base encoders are raw (no LoRA) for raw divergence check
        radar.lora_A = None
        pitot.lora_A = None
        
        z_r_raw = radar.encode(radar_tele)
        z_p_raw = pitot.encode(pitot_tele)
        div_raw = router.divergence(z_r_raw, z_p_raw)
        dec_raw = router.route(0.98, 0.90, div_raw)
        
        if dec_raw == RouteDecision.TRIGGER_REPLAN:
            metrics["anomalies_evaluated"] += 1
            metrics["base_div_sum"] += div_raw
            
            # System 1 Check
            sys1_dec, sys1_note = sys1_router.resolve(z_r_raw, z_p_raw, dec_raw, 0.98, 0.90)
            if sys1_dec == RouteDecision.COMMIT_TRAJECTORY:
                metrics["sys1_hits"] += 1
                
            # System 2 Execution (Dynamic Loading)
            if "Pitot Freeze" in sys1_note:
                radar.lora_A = None
                pitot.load_lora("models/adapters_large/adapter_pitot_freeze.pt")
            elif "GPS Spoof" in sys1_note:
                pitot.lora_A = None
                radar.load_lora("models/adapters_large/adapter_gps_spoof.pt")
                
            z_r_sys2 = radar.encode(radar_tele)
            z_p_sys2 = pitot.encode(pitot_tele)
            div_sys2 = router.divergence(z_r_sys2, z_p_sys2)
            metrics["sys2_div_sum"] += div_sys2
            
            # Reset LoRAs for next flight
            radar.lora_A = None
            pitot.lora_A = None
            
        else:
            pass

    print("\n============================================================")
    print("VERIFICATION RESULTS")
    print("============================================================")
    print(f"Total Anomalies Tested : {metrics['anomalies_evaluated']}")
    print(f"System 1 Hit Rate      : {(metrics['sys1_hits']/metrics['anomalies_evaluated'])*100:.1f}%")
    
    avg_base_div = metrics['base_div_sum'] / metrics['anomalies_evaluated']
    avg_sys2_div = metrics['sys2_div_sum'] / metrics['anomalies_evaluated']
    
    print(f"Average Raw Divergence (L1)    : {avg_base_div:.2f} (Alarm Threshold > 0.5)")
    print(f"Average System 2 Divergence    : {avg_sys2_div:.4f} (Safe Threshold)")
    print(f"System 2 Mathematical Reduction: {((avg_base_div - avg_sys2_div)/avg_base_div)*100:.1f}%")
    print("============================================================")
    
if __name__ == "__main__":
    main()
