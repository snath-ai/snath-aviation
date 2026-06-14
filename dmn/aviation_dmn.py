"""
Snath Aviation — Default Mode Network (DMN).

Simulates the sleep/consolidation cycle. Reads D_hard, clusters anomalies,
and generates both System 1 JSON Centroids and System 2 PyTorch LoRA matrices.
"""
import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import datetime
import hmac
import hashlib
from dhard import DHardQueue
from .sigreg import SIGRegLoss

try:
    from brain.abstract_dmn import AbstractDMN
except ImportError:
    from abc import ABC as AbstractDMN

_ADAPTER_HMAC_KEY = b"snath_aviation_adapter_sovereignty_2026"

class AviationDMN(AbstractDMN):
    def __init__(self, queue_path: str = "d_hard.jsonl", adapter_dir: str = "models/adapters"):
        self.q = DHardQueue(queue_path)
        self.adapter_dir = adapter_dir
        os.makedirs(self.adapter_dir, exist_ok=True)

    def consolidate(self, lambda_iso: float = 0.0):
        events = [e for e in self.q.all() if e.winner is not None]
        if not events:
            print("No resolved anomalies to consolidate.")
            return []
            
        # Cluster 1: Radar won, meaning Pitot tube failed.
        radar_wins = [e for e in events if e.winner == "radar"]
        
        if len(radar_wins) > 0:
            print(f"🧠 DMN Sleep Cycle: Processing {len(radar_wins)} 'Pitot Freeze' anomalies...")

            # ---------------------------------------------------------
            # 1. SYSTEM 1: Fast JSON Centroid Generation (HMAC-signed)
            # Both centroids stored so adapter_router can use cosine
            # similarity on the delta vector (matching Basis/Robotics).
            # ---------------------------------------------------------
            avg_v_a = np.mean([e.v_a for e in radar_wins], axis=0).tolist()
            avg_v_b = np.mean([e.v_b for e in radar_wins], axis=0).tolist()
            created_at = datetime.datetime.utcnow().isoformat() + "Z"
            json_immutable = {
                "type": "pitot_freeze",
                "centroid_v_a": avg_v_a,
                "centroid_v_b": avg_v_b,
                "trust": "radar",
                "winner": "radar",
                "win_rate": round(len(radar_wins) / max(len(events), 1), 3),
                "n_events": len(radar_wins),
                "failure_class": "weather_induced",
            }
            json_sig = hmac.new(
                _ADAPTER_HMAC_KEY,
                json.dumps(json_immutable, sort_keys=True).encode(),
                hashlib.sha256,
            ).hexdigest()
            json_path = os.path.join(self.adapter_dir, "adapter_pitot_freeze.json")
            with open(json_path, "w") as f:
                json.dump({**json_immutable, "created_at": created_at, "sig": json_sig}, f, indent=2)
            print(f"    [SYSTEM 1] Generated Fast JSON Centroid Cache at {json_path}")
            
            # ---------------------------------------------------------
            # 2. SYSTEM 2: Deep PyTorch LoRA Backpropagation
            # ---------------------------------------------------------
            A = nn.Parameter(torch.randn(3, 1) * 0.01)
            B = nn.Parameter(torch.randn(1, 3) * 0.01)
            optimizer = optim.AdamW([A, B], lr=0.1)
            sigreg = SIGRegLoss(lambda_iso=lambda_iso)

            target_safe = torch.tensor([e.v_a for e in radar_wins], dtype=torch.float32)  # Radar
            faulty_input = torch.tensor([e.v_b for e in radar_wins], dtype=torch.float32) # Pitot

            for epoch in range(100):
                optimizer.zero_grad()
                adapted_latent = faulty_input + torch.matmul(torch.matmul(faulty_input, A), B)
                loss = torch.nn.functional.l1_loss(adapted_latent, target_safe)
                # SIGReg: penalise anisotropy in the adapted pitot latent space so that
                # Δ = softmax(z_radar) − softmax(z_pitot) is an informative routing signal.
                # No-op when lambda_iso=0.0 (default until AIA Experiment 3).
                loss = loss + sigreg(adapted_latent)
                loss.backward()
                optimizer.step()
                
            pt_path = os.path.join(self.adapter_dir, "adapter_pitot_freeze.pt")
            
            # LoRA Sovereignty: HMAC Collision-Resistant Verification
            a_hash = hashlib.sha256(A.detach().numpy().tobytes()).hexdigest()[:16]
            b_hash = hashlib.sha256(B.detach().numpy().tobytes()).hexdigest()[:16]
            payload_str = f"pitot|{a_hash}|{b_hash}"
            signature = hmac.new(_ADAPTER_HMAC_KEY, payload_str.encode(), hashlib.sha256).hexdigest()
            
            torch.save({
                "A": A.detach(),
                "B": B.detach(),
                "target_encoder": "pitot",
                "hmac_hex":       signature,
                "created_at":     datetime.datetime.utcnow().isoformat() + "Z",
                "failure_class":  "weather_induced",   # temporal decay λ=0.50
            }, pt_path)
            print(f"    [SYSTEM 2] Trained PyTorch LoRA (Loss: {loss.item():.4f}) at {pt_path}")
            
        # Cluster 2: Pitot won, meaning Radar failed (e.g., GPS Spoof).
        pitot_wins = [e for e in events if e.winner == "pitot"]
        if len(pitot_wins) > 0:
            print(f"🧠 DMN Sleep Cycle: Processing {len(pitot_wins)} 'GPS Spoof' anomalies...")
            
            avg_v_a_spoof = np.mean([e.v_a for e in pitot_wins], axis=0).tolist()
            avg_v_b = np.mean([e.v_b for e in pitot_wins], axis=0).tolist()
            created_at_spoof = datetime.datetime.utcnow().isoformat() + "Z"
            json_immutable_spoof = {
                "type": "gps_spoof",
                "centroid_v_a": avg_v_a_spoof,
                "centroid_v_b": avg_v_b,
                "trust": "pitot",
                "winner": "pitot",
                "win_rate": round(len(pitot_wins) / max(len(events), 1), 3),
                "n_events": len(pitot_wins),
                "failure_class": "weather_induced",
            }
            json_sig_spoof = hmac.new(
                _ADAPTER_HMAC_KEY,
                json.dumps(json_immutable_spoof, sort_keys=True).encode(),
                hashlib.sha256,
            ).hexdigest()
            json_path = os.path.join(self.adapter_dir, "adapter_gps_spoof.json")
            with open(json_path, "w") as f:
                json.dump({**json_immutable_spoof, "created_at": created_at_spoof, "sig": json_sig_spoof}, f, indent=2)
            print(f"    [SYSTEM 1] Generated Fast JSON Centroid Cache at {json_path}")
            
            # Train LoRA for Radar Encoder
            A = nn.Parameter(torch.randn(3, 1) * 0.01)
            B = nn.Parameter(torch.randn(1, 3) * 0.01)
            optimizer = optim.AdamW([A, B], lr=0.1)
            sigreg = SIGRegLoss(lambda_iso=lambda_iso)

            target_safe = torch.tensor([e.v_b for e in pitot_wins], dtype=torch.float32)  # Pitot
            faulty_input = torch.tensor([e.v_a for e in pitot_wins], dtype=torch.float32) # Radar

            for epoch in range(100):
                optimizer.zero_grad()
                adapted_latent = faulty_input + torch.matmul(torch.matmul(faulty_input, A), B)
                loss = torch.nn.functional.l1_loss(adapted_latent, target_safe)
                # SIGReg: penalise anisotropy in the adapted radar latent space.
                # No-op when lambda_iso=0.0 (default until AIA Experiment 3).
                loss = loss + sigreg(adapted_latent)
                loss.backward()
                optimizer.step()
                
            pt_path = os.path.join(self.adapter_dir, "adapter_gps_spoof.pt")
            
            # LoRA Sovereignty: HMAC Collision-Resistant Verification
            a_hash = hashlib.sha256(A.detach().numpy().tobytes()).hexdigest()[:16]
            b_hash = hashlib.sha256(B.detach().numpy().tobytes()).hexdigest()[:16]
            payload_str = f"radar|{a_hash}|{b_hash}"
            signature = hmac.new(_ADAPTER_HMAC_KEY, payload_str.encode(), hashlib.sha256).hexdigest()
            
            torch.save({
                "A": A.detach(),
                "B": B.detach(),
                "target_encoder": "radar",
                "hmac_hex":       signature,
                "created_at":     datetime.datetime.utcnow().isoformat() + "Z",
                "failure_class":  "weather_induced",   # temporal decay λ=0.50
            }, pt_path)
            print(f"    [SYSTEM 2] Trained PyTorch LoRA (Loss: {loss.item():.4f}) at {pt_path}")

        return [
            {"path": os.path.join(self.adapter_dir, f)}
            for f in os.listdir(self.adapter_dir)
            if f.endswith(".json")
        ]

    def ingest(self, event) -> None:
        self.q.push(event)

    def recall(self, query, **kwargs) -> dict:
        for fname in os.listdir(self.adapter_dir):
            if fname.endswith(".json"):
                path = os.path.join(self.adapter_dir, fname)
                try:
                    data = json.load(open(path))
                    if data.get("type") == query or data.get("failure_class") == query:
                        return data
                except Exception:
                    pass
        return {}

    def stats(self) -> dict:
        return self.q.stats()
