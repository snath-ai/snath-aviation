# Snath Aviation

Sensor fault resolution for aviation safety systems.

---

Snath Aviation routes aircraft control decisions through sensor failures by measuring the geometric distance between two independent sensor streams — Radar and Pitot — rather than combining them. When the streams agree, the aircraft flies normally. When they confidently disagree, the system identifies the faulty sensor, reconstructs the safe trajectory from the healthy one, and resolves the same failure type from memory on subsequent encounters.

The routing core contains no trainable weights. It operates on the geometric distance between the two streams and cannot be retrained, fine-tuned, or caused to forget. The learning system operates entirely on the peripheral encoders, improving their geometry overnight without touching the routing core. The two systems — frozen router and adaptive encoders — are structurally decoupled.

Built on [Lár](https://github.com/snath-ai/Lar-JEPA).

*Dedicated to the 228 aboard Air France Flight 447, 1 June 2009.*

---

## How it works

### The routing core

The `AviationDivergenceRouter` is mathematically frozen. Its routing logic is 12 lines. It has no trainable weights and cannot be retrained. An aviation authority can read the routing invariants and formally verify them.

The divergence metric is total variation distance between the probability vectors of the two streams:

```
D = L1(softmax(z_radar) - softmax(z_pitot)) / sqrt(dim)
```

Normalising by probability vectors makes D magnitude-invariant — a frozen Pitot tube and a healthy Radar produce a large D regardless of velocity or altitude. A NaN guard forces `D = 2.0` if either encoder fails, routing unconditionally to STRUCTURAL_IMPASSE.

The router receives three scalars: radar confidence, pitot confidence, divergence. It never sees the raw sensor vectors. Four outcomes:

- D below threshold: fly normally
- Both confident, D above threshold: sensor disagreement detected — log, learn, resolve from memory
- Both confidence below floor: total signal loss — AI disconnects, revert to manual
- R4 firewall (TrajectoryCommitKernel): if the perturbed trajectory scores below 0.5, the AI disconnects regardless of upstream outcome. The learning system cannot command a maneuver that violates flight physics.

### The encoders

**RadarEncoder** maps velocity and altitude to a 3-dimensional latent vector. Confidence is an SNR proxy: `sigmoid((|z - 0.5|.mean() - 0.15) * 10)`. When the Radar is healthy, confidence is approximately 0.98.

**PitotEncoder** maps airspeed to the same 3D space. When the Pitot tube freezes, the latent collapses and confidence drops to approximately 0.05. This collapse is the geometric signal the router detects.

Both encoders support LoRA injection. A signed rank-1 matrix pair (A, B) can be loaded into either encoder: `adapted = base + (base @ lora_A @ lora_B)`. The `load_lora()` method verifies the HMAC signature and checks the `target_encoder` field before injecting — a GPS Spoof LoRA is silently rejected by the Pitot encoder.

### Default Mode Network

When the router returns `TRIGGER_REPLAN`, the event is HMAC-signed and appended to a local JSONL queue. During the consolidation cycle, `AviationDMN.consolidate()` clusters events by winner (radar wins, pitot wins) and trains two artifacts per cluster:

**System 1 — JSON centroid cache.** L1 distance from the incoming broken latent to the centroid. Distance below 0.2 is a match: instant `COMMIT_TRAJECTORY` override with no matrix computation. Sub-millisecond response.

**System 2 — PyTorch LoRA.** Signed rank-1 matrices trained to minimise L1 loss between the faulty stream and the winning stream. Injecting the adapter into the faulty encoder warps its latent geometry to match the healthy stream. The router measures a divergence near zero. The failure is resolved at the source.

**AviationHealthMonitor** manages the adapter lifecycle across continuous telemetry. It auto-detaches LoRA when raw divergence drops below the safe threshold for a sustained window (the sensor has physically recovered), re-classifies any new spike through System 1 before re-arming, and applies temporal decay — `W = exp(-lambda * delta_t)` — at re-arm time. Adapters trained on weather-induced failures (lambda = 0.50) decay faster than hardware defects (lambda = 0.02). If the adapter's trust weight falls below 0.40, the monitor refuses to re-arm and flags the event for a fresh training cycle.

**Empirical results.** Live telemetry from UAL2298 (OpenSky Network): simulated Pitot freeze, raw divergence 1.59. After DMN sleep cycle and LoRA injection: divergence 0.05. Large-scale synthetic validation (N = 2000 flights, N = 500 holdout): System 1 hit rate 100%, System 2 divergence reduction 93.3%.

---

## Pipeline

```
[Radar] [Pitot]     independent encoders, no shared state
    |       |
    v       v
AviationDivergenceRouter  (frozen, zero weights)
    |
    +-- COMMIT_TRAJECTORY  --> fly normally
    +-- TRIGGER_REPLAN     --> AviationAdapterRouter (System 1 cache lookup)
    |                              +-- cache hit  --> override, load LoRA (System 2)
    |                              +-- cache miss --> log to D_hard, flag for DMN
    +-- STRUCTURAL_IMPASSE --> autopilot disconnect
```

`aviation_full_stack.py` runs all ten ABCs on `lar.GraphExecutor` with HMAC-signed audit logging. `demo_real_world.py` runs the full closed loop on live OpenSky telemetry: failure detection, D_hard logging, DMN training, and next-flight resolution in a single execution.

---

## Getting started

```bash
python aviation_full_stack.py    # all ten ABCs, GraphExecutor, 8 audited steps
python demo_real_world.py        # live OpenSky telemetry + full DMN closed loop
python demo_large_scale.py       # N=2000 synthetic validation
python demo_health_monitor.py    # LoRA lifecycle: arm / detach / re-arm
python demo_full_33.py           # 33-invariant pipeline
python test_33_invariants.py     # formal invariant test suite
```

---

## Research

The routing invariants, the Safety-Learning Equivalence theorem, and the proof of domain isomorphism are in:

- Sajeev, A.V. (2026). *Divergence Is Not Noise: Multi-Stream Routing Without Modal Fusion and the Safety-Learning Equivalence.* [doi.org/10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781)
- Sajeev, A.V. (2026). *Universal Cognitive Routing: A Ten-Abstract-Base-Class Specification for Domain-Agnostic Agent Execution.* [doi.org/10.5281/zenodo.20278775](https://doi.org/10.5281/zenodo.20278775)
- Sajeev, A.V. (2026). *Architecture Is All You Need: Pre-Registration and Protocol for Empirical Validation of the Lár Training Loop.* [doi.org/10.5281/zenodo.20419182](https://doi.org/10.5281/zenodo.20419182)

Snath Aviation is one of three domain instantiations proving the architecture's universality. The other two are Snath Locus (CRISPR drug screening) and Snath Basis (quantitative finance). All three implement identical V1–V6 routing invariants without domain-specific modification to the routing core.

ADS-B telemetry in `demo_real_world.py` and `d_hard_live.jsonl` is sourced from the [OpenSky Network](https://opensky-network.org) — a public receiver network aggregating aircraft transponder broadcasts freely available for research use.
