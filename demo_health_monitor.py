"""
demo_health_monitor.py — LoRA Lifecycle + Temporal Decay + Revival Signaling

Tests the full pipeline:
  Phase 1 (Ticks  1- 5): Normal cruise.
  Phase 2 (Ticks  6-20): Pitot freeze. System 1+2 fires. Adapter loaded.
  Phase 3 (Ticks 21-25): Ice clears. Recovery window.
  Phase 4 (Tick     26): LoRA detached. Revival memory records 'pitot_freeze'.
  Phase 5 (Ticks 26-35): Ice freezes AGAIN. System 1 re-classifies.
                          Temporal decay W computed. If fresh → re-armed.
                          Revival signal fired (recurring pattern).
  Phase 6 (Ticks 36-45): Clean flight. Temporal audit printed.

Run: python demo_health_monitor.py
"""
import numpy as np
import _lar  # noqa

from demo_real_world import RadarEncoder, PitotEncoder, AviationDivergenceRouter
from dmn.adapter_router import AviationAdapterRouter
from dmn.health_monitor import AviationHealthMonitor
from core.types import RouteDecision


def healthy_tel(velocity=247.0, altitude=11582.0):
    return {"velocity": velocity, "altitude": altitude}

def frozen_tel(altitude=11582.0):
    return {"velocity": 0.0, "altitude": altitude}


def main():
    print("=" * 72)
    print("Snath Aviation — LoRA Lifecycle + Temporal Decay + Revival Signaling")
    print("=" * 72)

    radar   = RadarEncoder()
    pitot   = PitotEncoder()
    router  = AviationDivergenceRouter()
    arouter = AviationAdapterRouter(adapter_dir="models/adapters_live")
    monitor = AviationHealthMonitor(
        router=router,
        adapter_router=arouter,
        adapter_dir="models/adapters_live",
        safe_threshold=0.5,
        recovery_window=5,
        min_trust_weight=0.40,
        verbose=True,
    )
    monitor.register(pitot)

    print(f"\n{'Tick':<6} {'Phase':<22} {'Pitot V':>8} {'RawDiv':>8} "
          f"{'LoRA':>5} {'W':>6} {'Action'}")
    print("-" * 80)

    for tick in range(1, 46):
        if tick <= 5:
            phase, radar_t, pitot_t = "Normal cruise", healthy_tel(), healthy_tel()

        elif tick <= 20:
            phase, radar_t, pitot_t = "Pitot freeze (1st)", healthy_tel(), frozen_tel()
            if tick == 6:
                z_r = radar.encode(radar_t); z_p = pitot.encode(pitot_t)
                div = router.divergence(z_r, z_p)
                dec = router.route(radar.get_confidence(z_r), pitot.get_confidence(z_p), div)
                # Pass enc_pitot so resolve() injects LoRA internally if W >= min_trust.
                # Manual load_lora() removed — trust gate now enforced inside resolve().
                arouter.resolve(z_r, z_p, dec, radar.get_confidence(z_r), pitot.get_confidence(z_p),
                                enc_pitot=pitot)
                monitor.notify_lora_loaded(pitot, adapter_type="pitot_freeze")
                print(f"\n  ⚡ [Tick {tick}] Pitot freeze. System 1 hit. LoRA loaded.\n")

        elif tick <= 25:
            phase, radar_t, pitot_t = "Ice cleared", healthy_tel(), healthy_tel()

        elif tick <= 35:
            phase, radar_t, pitot_t = "Pitot freeze (2nd)", healthy_tel(), frozen_tel()
            if tick == 26:
                print(f"\n  🧊 [Tick {tick}] Ice returns. Temporal decay re-arm check...\n")

        else:
            phase, radar_t, pitot_t = "Normal cruise again", healthy_tel(), healthy_tel()

        result = monitor.tick(encoder_a=radar, encoder_b=pitot,
                              telemetry_a=radar_t, telemetry_b=pitot_t)

        lora_str = "YES" if result["lora_loaded"] else "no"
        W_str    = f"{result['active_adapter_weight']:.3f}" if result["lora_loaded"] else "-"
        revival  = " 🔁REVIVAL" if result["revival_flagged"] else ""
        action   = result["action_taken"] if result["action_taken"] != "none" else ""

        print(f"{tick:<6} {phase:<22} {pitot_t['velocity']:>8.1f} "
              f"{result['raw_divergence']:>8.4f} {lora_str:>5} {W_str:>6}  "
              f"{action}{revival}")

    print("\n" + "=" * 72)
    print("FINAL STATUS")
    print("=" * 72)
    for name, s in monitor.status().items():
        print(f"  {name:<14} lora={s['lora_loaded']}  "
              f"adapter='{s['active_adapter_type']}'  "
              f"last_div={s['last_divergence']:.4f}")

    monitor.temporal_audit()
    print()


if __name__ == "__main__":
    main()
