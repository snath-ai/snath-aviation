"""
AviationAdapterRouter — Two-Component Adapter System
=====================================================
Updated to match Snath Basis / Robotics pattern:
  - System 1: cosine similarity ≥ τ_sim on delta vector (was L1 < 0.2)
  - JSON centroid HMAC verified at load time (was unverified)
  - .pt HMAC verified before injection (was unverified)

Implements the identification / correction trust asymmetry described in
"Architecture Is All You Need" (Sajeev 2026), §3.4 Remark (Temporal Decay
and Synaptic Depression):

  System 1 — Identification (trust-invariant)
  Cosine similarity on the divergence vector fingerprint Δ = z_radar − z_pitot.
  The geometric signature of a failure class is durable across years.
  Fires regardless of adapter age.

  System 2 — Correction (perishable)
  LoRA weights gated by W = exp(−λ · Δt) ≥ min_trust.
  A delta trained on one sensor variant may be wrong for a later variant.

  Degradation path
  When System 2 is refused, System 1 still identifies the fault and
  returns COMMIT_TRAJECTORY. Identify correctly, correct conservatively.

Temporal decay constants:
  weather_induced   λ=0.50  — ice / turbulence, fast decay
  hardware_struct   λ=0.02  — manufacturing defect, slow decay

Derivative Works note
---------------------
Derivative Work of AbstractDivergenceRouter (V1–V6) and
JEPA_DMN_Consolidation_Node, Apache 2.0, github.com/snath-ai/Lar-JEPA.
"""
import os
import json
import math
import hmac as _hmac
import hashlib
import datetime
import glob
import numpy as np
from pathlib import Path
from typing import Optional

from core.types import RouteDecision

try:
    from brain.abstract_adapter_router import AbstractAdapterRouter
except ImportError:
    from abc import ABC as AbstractAdapterRouter

_ADAPTER_HMAC_KEY = b"snath_aviation_adapter_sovereignty_2026"

# ── Temporal decay ────────────────────────────────────────────────────────────
_LAMBDA: dict = {
    "weather_induced": 0.50,
    "hardware_struct": 0.02,
    "default":         0.10,
}


def _decay_weight(created_at_iso: str | None, failure_class: str = "default") -> float:
    """W = exp(-λ · Δt). Returns 1.0 if no timestamp."""
    if not created_at_iso:
        return 1.0
    try:
        created     = datetime.datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
        now         = datetime.datetime.now(datetime.timezone.utc)
        delta_years = (now - created).total_seconds() / (365.25 * 24 * 3600)
        lam         = _LAMBDA.get(failure_class, _LAMBDA["default"])
        return math.exp(-lam * max(0.0, delta_years))
    except Exception:
        return 1.0


def _cos(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def _verify_json(adapter: dict) -> bool:
    """Verify HMAC of JSON centroid adapter. Returns False for unsigned adapters."""
    stored_sig = adapter.get("sig", "")
    if not stored_sig:
        return False
    immutable_keys = [
        "type", "centroid_v_a", "centroid_v_b", "trust",
        "winner", "win_rate", "n_events", "failure_class",
    ]
    payload = json.dumps(
        {k: adapter[k] for k in immutable_keys if k in adapter},
        sort_keys=True,
    ).encode()
    expected = _hmac.new(_ADAPTER_HMAC_KEY, payload, hashlib.sha256).hexdigest()
    return _hmac.compare_digest(stored_sig, expected)


def _verify_pt(meta: dict) -> bool:
    """Verify HMAC of .pt adapter payload before injection."""
    stored_sig = meta.get("hmac_hex", "")
    if not stored_sig:
        return False
    try:
        A = meta["A"]
        B = meta["B"]
        a_hash = hashlib.sha256(A.numpy().tobytes()).hexdigest()[:16]
        b_hash = hashlib.sha256(B.numpy().tobytes()).hexdigest()[:16]
        target = meta.get("target_encoder", "")
        payload_str = f"{target}|{a_hash}|{b_hash}"
        expected = _hmac.new(
            _ADAPTER_HMAC_KEY, payload_str.encode(), hashlib.sha256
        ).hexdigest()
        return _hmac.compare_digest(stored_sig, expected)
    except Exception:
        return False


class AviationAdapterRouter(AbstractAdapterRouter):
    """
    Two-pass adapter router for Snath Aviation.

    Args:
        adapter_dir: Directory with .json centroid caches and .pt LoRA files.
        tau_sim:     Cosine similarity threshold for System 1 centroid match.
        min_trust:   Temporal trust floor W for System 2 LoRA injection.
        verbose:     Print debug info on HMAC failures.
    """

    def __init__(
        self,
        adapter_dir: str   = "models/adapters",
        tau_sim:     float = 0.90,
        min_trust:   float = 0.40,
        verbose:     bool  = False,
    ):
        self.adapter_dir = Path(adapter_dir)
        self.tau_sim     = tau_sim
        self.min_trust   = min_trust
        self.verbose     = verbose
        self._adapters:  list[dict] = []
        self._load_all()

    def _load_all(self) -> None:
        """Load and HMAC-verify all JSON centroid adapters."""
        self._adapters = []
        for fp in glob.glob(str(self.adapter_dir / "*.json")):
            try:
                data = json.loads(Path(fp).read_text())
                if _verify_json(data):
                    self._adapters.append(data)
                elif self.verbose:
                    print(f"[AviationAdapterRouter] HMAC FAIL — skipped {fp}")
            except Exception as e:
                if self.verbose:
                    print(f"[AviationAdapterRouter] load error {fp}: {e}")

    def refresh(self) -> None:
        self._load_all()

    def available(self) -> list[str]:
        return [a.get("type", "?") for a in self._adapters]

    def check_lora_trust(self, adapter_type: str) -> float:
        """
        Return temporal trust W for the .pt adapter of adapter_type.
        Reads created_at + failure_class from the .pt file metadata.
        Returns 1.0 if the file is missing or lacks temporal fields.
        """
        pt_path = self.adapter_dir / f"adapter_{adapter_type}.pt"
        if not pt_path.exists():
            return 1.0
        try:
            import torch as _torch
            meta = _torch.load(str(pt_path), map_location="cpu", weights_only=False)
            return _decay_weight(meta.get("created_at"), meta.get("failure_class", "default"))
        except Exception:
            return 1.0

    def _nearest(self, delta: np.ndarray) -> Optional[dict]:
        """
        System 1 — trust-invariant centroid match via cosine similarity.

        Matches Δ = z_radar − z_pitot against stored centroid delta
        (centroid_v_a − centroid_v_b) for each adapter.

        No temporal gate — failure signatures are durable.
        """
        best, best_s = None, self.tau_sim
        for adapter in self._adapters:
            c_a = adapter.get("centroid_v_a", [])
            c_b = adapter.get("centroid_v_b", [])
            if not c_a or not c_b:
                continue
            centroid_delta = np.array(c_a) - np.array(c_b)
            n = min(len(delta), len(centroid_delta))
            s = _cos(delta[:n], centroid_delta[:n])
            if s >= best_s:
                best, best_s = adapter, s
        return best

    def resolve(
        self,
        z_radar:       np.ndarray,
        z_pitot:       np.ndarray,
        base_decision: RouteDecision,
        c_radar:       float,
        c_pitot:       float,
        enc_radar      = None,
        enc_pitot      = None,
    ) -> tuple[RouteDecision, str]:
        """System 1 + System 2 combined inference."""
        if base_decision != RouteDecision.TRIGGER_REPLAN:
            return base_decision, "Base decision was safe."

        z_a   = np.asarray(z_radar, float)
        z_b   = np.asarray(z_pitot, float)
        delta = z_a - z_b

        # ── SYSTEM 1: Fast centroid match (trust-invariant) ───────────────────
        match = self._nearest(delta)
        if match is None:
            return base_decision, "System 1 Cache Miss — no matching pattern."

        adapter_type  = match.get("type",         "unknown")
        winner        = match.get("winner",        "unknown")
        win_rate      = match.get("win_rate",      0.0)
        n_events      = match.get("n_events",      0)
        failure_class = match.get("failure_class", "default")

        decision = RouteDecision.COMMIT_TRAJECTORY
        note = (f"[System 1] pattern [{adapter_type}] n={n_events} "
                f"winner={winner} win_rate={win_rate:.0%}")

        # ── SYSTEM 2: LoRA injection (perishable, trust-gated) ───────────────
        # target_encoder is the FAULTY encoder (the loser), not the winner.
        # Read it from the .pt rather than inferring from winner, so the
        # injection always matches what the DMN trained.
        pt_path = self.adapter_dir / f"adapter_{adapter_type}.pt"

        if pt_path.exists():
            try:
                import torch as _torch
                meta = _torch.load(str(pt_path), map_location="cpu", weights_only=False)

                if not _verify_pt(meta):
                    note += " | [System 2] .pt HMAC FAIL — refused"
                else:
                    W = _decay_weight(meta.get("created_at"), failure_class)
                    target_enc_name = meta.get("target_encoder", "")
                    target_enc_obj  = enc_radar if target_enc_name == "radar" else enc_pitot
                    if W >= self.min_trust:
                        if target_enc_obj is not None and hasattr(target_enc_obj, "load_lora"):
                            target_enc_obj.load_lora(str(pt_path))
                            note += (f" | [System 2] LoRA → '{target_enc_name}' encoder "
                                     f"W={W:.2f} (injected)")
                        else:
                            note += f" | [System 2] W={W:.2f} — System 2 ready"
                    else:
                        note += (f" | [System 2] STALE adapter refused "
                                 f"(W={W:.2f} < {self.min_trust}) — System 1 only")
            except Exception as exc:
                note += f" | [System 2] load error: {exc}"

        return decision, note
