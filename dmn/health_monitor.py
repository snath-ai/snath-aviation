"""
AviationHealthMonitor — Automatic LoRA Detachment + Temporal Decay System.

Watches the raw divergence between all encoder pairs on every telemetry tick.
When a previously-failed sensor recovers, it silently detaches the LoRA adapter.
When a new spike occurs, it re-classifies via System 1 and checks temporal decay
before re-arming — preventing stale adapters from being loaded.

Temporal Decay (from TemporalNode, Snath Locus):
    W = exp(-λ · Δt)    where Δt = years since adapter was trained

    Failure class  λ      Rationale
    ─────────────────────────────────────────────────────────────────
    weather_induced  0.50  Ice, turbulence, atmospheric — highly seasonal.
                           A summer Pitot freeze pattern is stale by winter.
    hardware_struct  0.02  Bent sensor, manufacturing defect — intrinsic to
                           the aircraft, doesn't age with seasons.
    default          0.10  General DMN-derived adapters.

If W drops below `min_trust_weight` (default 0.40) at re-arm time, the monitor
refuses to load the stale adapter and instead flags the event for a fresh DMN
consolidation cycle — exactly as TemporalNode pushes borderline viability scores
toward REPLAN when evidence is stale.

Revival Signaling:
    Mirrors TemporalNode Part B. If a previously-detached adapter's failure type
    re-appears with raw_div >= safe_threshold, AND the cached adapter is still
    within trust weight, the monitor flags a REVIVAL signal before re-arming,
    giving the maintenance system visibility into recurring failure patterns.
"""
from __future__ import annotations
from dataclasses import dataclass
import datetime
import math
import json
import glob
import os
import numpy as np

# ── Temporal decay λ table (mirrors Snath Locus TemporalNode) ────────────────
_LAMBDA: dict[str, float] = {
    "weather_induced":  0.50,   # ice / turbulence — fast decay
    "hardware_struct":  0.02,   # manufacturing defect — slow decay
    "default":          0.10,   # general adapter
}

def _decay_weight(created_at_iso: str | None, failure_class: str = "default") -> float:
    """
    W = exp(-λ · Δt).  Δt = fractional years since the adapter was trained.
    Returns 1.0 if no timestamp available (treat as current).
    """
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


@dataclass
class SensorHealthRecord:
    encoder_name: str
    lora_loaded: bool
    active_adapter_type: str    # e.g. "pitot_freeze"
    active_adapter_weight: float  # temporal trust W ∈ [0,1] at time of load
    consecutive_safe_ticks: int
    last_divergence: float
    revival_flagged: bool       # True if revival signal fired on last re-arm


class AviationHealthMonitor:
    """
    Continuous sensor health monitor with automatic LoRA lifecycle + temporal decay.

    On every call to .tick():
      1. Strip LoRA temporarily, measure RAW divergence.
      2. If LoRA loaded and raw_div safe for recovery_window ticks → DETACH.
      3. If no LoRA and raw_div spikes → System 1 re-classifies → compute
         temporal decay weight W for the matching cached adapter.
         - W >= min_trust_weight → RE-ARM (safe, adapter is fresh enough).
         - W <  min_trust_weight → STALE (refuse re-arm, flag for DMN refresh).
      4. If a previously detached adapter type re-appears → REVIVAL SIGNAL.

    Parameters
    ----------
    router            : AviationDivergenceRouter instance
    adapter_router    : AviationAdapterRouter instance (System 1 spatial index)
    adapter_dir       : path to the adapter directory (for reading .json metadata)
    safe_threshold    : L1 divergence below which sensor is healthy (default 0.5)
    recovery_window   : consecutive safe ticks before detachment (default 10)
    min_trust_weight  : minimum temporal W to allow re-arm (default 0.40)
    verbose           : print lifecycle events
    """

    def __init__(self, router, adapter_router,
                 adapter_dir: str = "models/adapters_live",
                 safe_threshold: float = 0.5,
                 recovery_window: int = 10,
                 min_trust_weight: float = 0.40,
                 verbose: bool = True):
        self.router           = router
        self.adapter_router   = adapter_router
        self.adapter_dir      = adapter_dir
        self.safe_threshold   = safe_threshold
        self.recovery_window  = recovery_window
        self.min_trust_weight = min_trust_weight
        self.verbose          = verbose
        self._records: dict[str, SensorHealthRecord] = {}
        # Typed cache: (encoder_name, adapter_type) -> (lora_A, lora_B, created_at, failure_class)
        self._lora_cache: dict[tuple[str, str], tuple] = {}
        # Revival memory: tracks adapter types that have been detached this session
        self._detached_types: set[str] = set()

    # ── Adapter metadata ──────────────────────────────────────────────────────
    def _load_adapter_meta(self, adapter_type: str) -> tuple[str | None, str]:
        """
        Read created_at and failure_class from the matching .json file.
        Returns (created_at_iso, failure_class).
        """
        pattern = os.path.join(self.adapter_dir, f"adapter_{adapter_type}.json")
        matches = glob.glob(pattern)
        if not matches:
            return None, "default"
        try:
            with open(matches[0]) as f:
                meta = json.load(f)
            return meta.get("created_at"), meta.get("failure_class", "default")
        except Exception:
            return None, "default"

    # ── Registration ──────────────────────────────────────────────────────────
    def register(self, encoder) -> None:
        name = encoder.modality
        self._records[name] = SensorHealthRecord(
            encoder_name=name,
            lora_loaded=False,
            active_adapter_type="none",
            active_adapter_weight=0.0,
            consecutive_safe_ticks=0,
            last_divergence=0.0,
            revival_flagged=False,
        )

    def notify_lora_loaded(self, encoder, adapter_type: str) -> None:
        """
        Call immediately after load_lora() so the monitor tracks it.
        Reads adapter metadata from the .json file and computes W.
        """
        name = encoder.modality
        if name not in self._records:
            return
        created_at, failure_class = self._load_adapter_meta(adapter_type)
        W = _decay_weight(created_at, failure_class)
        self._records[name].lora_loaded = True
        self._records[name].active_adapter_type = adapter_type
        self._records[name].active_adapter_weight = W
        self._records[name].consecutive_safe_ticks = 0
        self._records[name].revival_flagged = False
        # Typed cache — store with full provenance
        self._lora_cache[(name, adapter_type)] = (
            encoder.lora_A, encoder.lora_B, created_at, failure_class
        )
        if self.verbose:
            print(f"  📋 [HealthMonitor] Adapter '{adapter_type}' loaded "
                  f"(W={W:.3f}, class='{failure_class}', created={created_at or 'unknown'})")

    # ── Main tick ─────────────────────────────────────────────────────────────
    def tick(self, encoder_a, encoder_b, telemetry_a: dict, telemetry_b: dict) -> dict:
        name_b = encoder_b.modality
        rec = self._records.get(name_b)
        if rec is None:
            raise ValueError(f"Encoder '{name_b}' not registered.")

        # Step 1: Raw divergence (strip LoRA for honest measurement)
        saved_A, saved_B = encoder_b.lora_A, encoder_b.lora_B
        encoder_b.lora_A = None
        encoder_b.lora_B = None
        z_a     = encoder_a.encode(telemetry_a)
        z_b_raw = encoder_b.encode(telemetry_b)
        raw_div = self.router.divergence(z_a, z_b_raw)
        encoder_b.lora_A = saved_A
        encoder_b.lora_B = saved_B

        rec.last_divergence = raw_div
        rec.revival_flagged = False
        action_taken = "none"

        # Step 2: Health state machine
        if rec.lora_loaded:
            if raw_div < self.safe_threshold:
                rec.consecutive_safe_ticks += 1
                if rec.consecutive_safe_ticks >= self.recovery_window:
                    # DETACH
                    prev_type = rec.active_adapter_type
                    encoder_b.lora_A = None
                    encoder_b.lora_B = None
                    rec.lora_loaded = False
                    rec.active_adapter_type = "none"
                    rec.active_adapter_weight = 0.0
                    rec.consecutive_safe_ticks = 0
                    self._detached_types.add(prev_type)
                    action_taken = "LORA_DETACHED"
                    if self.verbose:
                        print(f"  🟢 [HealthMonitor] '{name_b}' RECOVERED after "
                              f"{self.recovery_window} safe ticks "
                              f"(raw_div={raw_div:.3f}). LoRA '{prev_type}' detached.")
                else:
                    action_taken = f"recovering ({rec.consecutive_safe_ticks}/{self.recovery_window})"
            else:
                rec.consecutive_safe_ticks = 0
                action_taken = "lora_active"

        else:
            if raw_div >= self.safe_threshold:
                # Re-classify via System 1 (never blindly re-arm)
                from core.types import RouteDecision
                sys1_dec, sys1_note = self.adapter_router.resolve(
                    z_a, z_b_raw, RouteDecision.TRIGGER_REPLAN,
                    encoder_a.get_confidence(z_a),
                    encoder_b.get_confidence(z_b_raw)
                )

                adapter_type = None
                if "Pitot Freeze" in sys1_note:   adapter_type = "pitot_freeze"
                elif "GPS Spoof" in sys1_note:    adapter_type = "gps_spoof"

                if adapter_type is not None:
                    cached = self._lora_cache.get((name_b, adapter_type))

                    # ── REVIVAL SIGNAL ────────────────────────────────────────
                    if adapter_type in self._detached_types:
                        rec.revival_flagged = True
                        if self.verbose:
                            print(f"  🔁 [HealthMonitor] REVIVAL SIGNAL — "
                                  f"'{adapter_type}' has re-appeared on '{name_b}'. "
                                  f"Recurring failure pattern detected.")

                    if cached is not None:
                        lora_A, lora_B, created_at, failure_class = cached
                        W = _decay_weight(created_at, failure_class)

                        if W >= self.min_trust_weight:
                            # ── RE-ARM ────────────────────────────────────────
                            encoder_b.lora_A = lora_A
                            encoder_b.lora_B = lora_B
                            rec.lora_loaded = True
                            rec.active_adapter_type = adapter_type
                            rec.active_adapter_weight = W
                            rec.consecutive_safe_ticks = 0
                            action_taken = f"LORA_REARMED (W={W:.3f})"
                            if self.verbose:
                                print(f"  🔴 [HealthMonitor] '{name_b}' DIVERGED "
                                      f"(raw_div={raw_div:.3f}). Re-classified as "
                                      f"'{adapter_type}' (W={W:.3f} ≥ {self.min_trust_weight}). "
                                      f"LoRA re-armed safely.")
                        else:
                            # ── STALE — refuse re-arm ─────────────────────────
                            action_taken = f"STALE_ADAPTER (W={W:.3f} < {self.min_trust_weight})"
                            if self.verbose:
                                print(f"  🟡 [HealthMonitor] '{name_b}' DIVERGED but "
                                      f"cached adapter '{adapter_type}' is STALE "
                                      f"(W={W:.3f} < {self.min_trust_weight}). "
                                      f"Triggering DMN consolidation cycle.")
                    else:
                        action_taken = "DIVERGED_NO_CACHE (DMN needed)"
                        if self.verbose:
                            print(f"  🟠 [HealthMonitor] '{name_b}' DIVERGED. "
                                  f"No cached LoRA for '{adapter_type}'. "
                                  f"Flagging for DMN sleep cycle.")
                else:
                    action_taken = "DIVERGED_UNCLASSIFIED"
                    if self.verbose:
                        print(f"  🟠 [HealthMonitor] '{name_b}' DIVERGED "
                              f"(raw_div={raw_div:.3f}). System 1 miss — "
                              f"logging to D_hard for overnight consolidation.")

        return {
            "encoder":                name_b,
            "raw_divergence":         round(raw_div, 4),
            "lora_loaded":            rec.lora_loaded,
            "active_adapter_type":    rec.active_adapter_type,
            "active_adapter_weight":  round(rec.active_adapter_weight, 4),
            "consecutive_safe_ticks": rec.consecutive_safe_ticks,
            "revival_flagged":        rec.revival_flagged,
            "action_taken":           action_taken,
        }

    def status(self) -> dict:
        return {name: {
            "lora_loaded":           r.lora_loaded,
            "active_adapter_type":   r.active_adapter_type,
            "active_adapter_weight": r.active_adapter_weight,
            "consecutive_safe_ticks": r.consecutive_safe_ticks,
            "last_divergence":       r.last_divergence,
            "revival_flagged":       r.revival_flagged,
        } for name, r in self._records.items()}

    def temporal_audit(self) -> None:
        """Print the trust weight W for every cached adapter."""
        print("\n  📊 [TemporalAudit] Adapter trust weights:")
        for (enc, atype), (_, _, created_at, failure_class) in self._lora_cache.items():
            W = _decay_weight(created_at, failure_class)
            bar = "█" * int(W * 20)
            status = "✅ TRUSTED" if W >= self.min_trust_weight else "⚠️  STALE"
            print(f"    ({enc}, {atype})  W={W:.4f}  [{bar:<20}]  {status}")
            print(f"      created={created_at or 'unknown'}  λ-class='{failure_class}'")
