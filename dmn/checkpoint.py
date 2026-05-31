"""
checkpoint.py — Flight Data Recorder (FDR) Checkpoint for Snath Aviation
========================================================================

A lightweight, non-disruptive checkpoint that serializes the full perception
and routing state to a timestamped, HMAC-signed JSON file immediately before
a critical routing decision.

This is the Snath Aviation equivalent of the EMA Annex 22 "replay" checkpoint
found in Snath Locus. It acts as an immutable Flight Data Recorder (FDR) audit
artefact. An NTSB investigator can load any checkpoint file and replay the exact
mathematical state of the autonomous system at the moment of an anomaly.

FDR Compliance
--------------
Checkpoint bundles include:
  - Full flight state at the moment of routing (all sensor telemetry,
    latent vectors, confidences, divergence values, and adapter states).
  - HMAC-SHA256 signature — tamper-evident cryptographic seal.
  - ISO-8601 timestamp (UTC) and flight_id (callsign).

Tensor Round-trip Guarantee
---------------------------
Every torch.Tensor in the state is serialized as:
    {"__tensor__": true, "shape": [...], "dtype": "torch.float32", "data": [...]}
and restored identically on load. This is lossless for float32.
"""

import os
import sys
import json
import hmac
import hashlib
import datetime
from typing import Dict, Any

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ===========================================================================
# Tensor serialisation helpers
# ===========================================================================

def _serialise_value(v):
    """Recursively convert tensors to JSON-safe dicts."""
    if _TORCH_AVAILABLE and isinstance(v, torch.Tensor):
        return {
            "__tensor__": True,
            "shape":      list(v.shape),
            "dtype":      str(v.dtype),
            "data":       v.detach().cpu().tolist(),
        }
    if isinstance(v, dict):
        return {k: _serialise_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        serialised = [_serialise_value(i) for i in v]
        return serialised if isinstance(v, list) else tuple(serialised)
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    import numpy as np
    if isinstance(v, np.ndarray):
        return _serialise_value(torch.from_numpy(v))
    return str(v)


def _deserialise_value(v):
    """Recursively convert tensor-proxy dicts back to torch.Tensors."""
    if isinstance(v, dict):
        if v.get("__tensor__") is True and _TORCH_AVAILABLE:
            dtype = getattr(torch, v["dtype"].replace("torch.", ""), torch.float32)
            return torch.tensor(v["data"], dtype=dtype).reshape(v["shape"])
        return {k: _deserialise_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_deserialise_value(i) for i in v]
    return v


# ===========================================================================
# AviationCheckpoint
# ===========================================================================

class AviationCheckpoint:
    """
    Flight Data Recorder (FDR) Checkpoint.

    Writes a timestamped, HMAC-signed snapshot of full flight state to disk.
    This does not halt execution, it simply acts as an immutable audit log.
    """

    def __init__(
        self,
        checkpoint_dir: str = "flight_data_recorder",
        hmac_secret: str = "snath_aviation_fdr_2026_compliance",
        verbose: bool = True,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.hmac_secret    = hmac_secret
        self.verbose        = verbose

    def record(self, flight_id: str, state: Dict[str, Any]) -> str:
        """
        Record the current state to the FDR.
        Returns the path to the written checkpoint file.
        """
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"fdr_{flight_id}_{ts}.json"
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filepath = os.path.join(self.checkpoint_dir, filename)

        # 1. Serialise full state (tensors → JSON-safe dicts)
        serial_state = _serialise_value(state)

        # 2. Sign payload
        payload = json.dumps(serial_state, sort_keys=True)
        signature = hmac.new(
            self.hmac_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        # 3. Bundle with FDR compliance metadata
        bundle = {
            "signature": signature,
            "meta": {
                "schema_version":  "1.0",
                "fdr_compliant":   True,
                "recorded_at":     ts,
                "flight_id":       flight_id,
                "state_keys":      list(serial_state.keys()),
                "hmac_algorithm":  "HMAC-SHA256",
            },
            "state": serial_state,
        }

        # 4. Write atomically via temp file → rename (prevents partial reads)
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
        os.replace(tmp_path, filepath)

        if self.verbose:
            size = os.path.getsize(filepath)
            print(
                f"\n📋 [FDR Checkpoint] Flight state cryptographically sealed\n"
                f"   File       : {filepath}\n"
                f"   State keys : {len(serial_state)}\n"
                f"   Size       : {size:,} bytes\n"
                f"   Signature  : {signature[:16]}…"
            )

        return filepath

    @classmethod
    def load_and_verify(cls, checkpoint_file: str, hmac_secret: str = "snath_aviation_fdr_2026_compliance") -> dict:
        """
        Load a checkpoint file, verify HMAC, restore tensor objects.
        """
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"FDR Checkpoint not found: '{checkpoint_file}'")

        with open(checkpoint_file, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        saved_sig    = bundle.get("signature", "")
        serial_state = bundle.get("state", {})
        meta         = bundle.get("meta", {})

        payload      = json.dumps(serial_state, sort_keys=True)
        expected_sig = hmac.new(
            hmac_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(expected_sig, saved_sig):
            raise ValueError(
                f"🚨 FDR HMAC VERIFICATION FAILED — checkpoint was tampered with:\n"
                f"  {checkpoint_file}\n"
                "Investigation halted. Data integrity cannot be guaranteed."
            )

        print(f"\n✅ FDR Checkpoint HMAC verified. State is pristine.")
        print(f"   File            : {checkpoint_file}")
        print(f"   Recorded at     : {meta.get('recorded_at', 'unknown')} UTC")
        print(f"   Flight ID       : {meta.get('flight_id', 'unknown')}")
        print(f"   State keys      : {meta.get('state_keys', [])}")

        return _deserialise_value(serial_state)
