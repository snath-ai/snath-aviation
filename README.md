# Snath Aviation

**Cognitive sensor-fault architecture for safety-critical flight systems.**

*Dedicated to the 228 aboard Air France Flight 447, 1 June 2009 — lost because the mathematics existed but had not been built.*

---

On 1 June 2009, all three Pitot tubes on AF447 froze simultaneously. They reported 0 m/s. The radar and GPS reported 274 m/s. The autopilot fused all sensor data into a single incoherent state, disconnected, and handed three woken pilots a dark dashboard in a storm over the Atlantic. The aircraft stalled. It fell 38,000 feet in 4 minutes and 24 seconds.

The autopilot was not defective. The pilots were not negligent. **The cognitive architecture was wrong.** It had no mathematical framework for determining which sensors to trust when they contradicted each other. It treated the contradiction as incoherence to be discarded rather than signal to be preserved.

Snath Aviation preserves the contradiction. It identifies which sensor failed geometrically in under one millisecond and resolves it from memory without sounding an alarm. The routing core contains zero trainable weights and can be formally verified by a regulator. All learning is confined to peripheral encoders, separated from the routing logic by 33 named invariants.

Built on [Lár-JEPA](https://github.com/snath-ai/Lar-JEPA) · Apache 2.0

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — PERCEPTION                                                   │
│  RadarEncoder ──────────────────────────────── PitotEncoder             │
│  (velocity, altitude → z_radar ∈ ℝ³)           (airspeed → z_pitot ∈ ℝ³) │
│  Invariants M1–M3: independent, no shared state                         │
└──────────────────────┬──────────────────────────┬───────────────────────┘
                       │                          │
                       ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — ROUTING (zero trainable weights — mathematically frozen)     │
│                                                                         │
│  D = L1(softmax(z_radar) − softmax(z_pitot)) / √dim                    │
│                                                                         │
│  both confident, D < 0.5  → COMMIT_TRAJECTORY  (fly normally)          │
│  both confident, D ≥ 0.5  → TRIGGER_REPLAN     (sensor conflict)       │
│  either confidence < 0.1  → STRUCTURAL_IMPASSE (total signal loss)      │
│  trajectory score < 0.5   → STRUCTURAL_IMPASSE (R4 physics firewall)   │
│                                                                         │
│  Invariants V1–V6, I1–I6, A1–A6, P1–P6, R1–R4                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ TRIGGER_REPLAN
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — LEARNING                                                     │
│                                                                         │
│  System 1 (< 1 ms)                  System 2 (async, overnight)        │
│  JSON centroid cache lookup         PyTorch LoRA injection              │
│  "have I seen this geometry?"       "structurally heal the encoder"     │
│  trust-invariant identification     perishable correction               │
│                                     W = exp(−λ · Δt), λ ∈ {0.50, 0.02} │
│                                                                         │
│  AviationDMN  ·  AviationAdapterRouter  ·  AviationHealthMonitor       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The routing core

`AviationDivergenceRouter` is frozen permanently. It has no parameters, does not update from experience, and cannot be retrained. Its entire logic is a probability-vector divergence and four routing rules. A regulator can read the invariants, formally verify them, and certify them — something no dense neural network with millions of learned weights has ever achieved in commercial aviation.

**The divergence metric:**

```python
p_radar = softmax(z_radar)
p_pitot = softmax(z_pitot)
D = float(np.sum(np.abs(p_radar - p_pitot)) / np.sqrt(dim))
```

Total variation distance normalised by `√dim`. Using probability vectors rather than raw activations makes the metric magnitude-invariant: a frozen Pitot at 0 m/s and a healthy Radar at 274 m/s produce the same large divergence regardless of velocity or altitude. A NaN guard forces `D = 2.0` if either encoder fails, routing unconditionally to `STRUCTURAL_IMPASSE`.

**The router receives three scalars — `c_radar`, `c_pitot`, `D` — and never sees the underlying vectors.** This is Invariant V4 (Content Blindness): the routing function cannot overfit to any sensor-specific content and works identically across pitot freezes, GPS spoofs, and failure modes not yet encountered.

**The 33 invariants:**

| Stage | Invariants | Role |
|---|---|---|
| Perception | M1–M3 | Independent stream encoding; no shared state |
| Routing | V1–V6 | Frozen geometric divergence; content blindness |
| Fault location | I1–I6 | Cross-attention fault targeting on the graph skeleton |
| Counterfactual | A1–A6, P1–P6 | Safe trajectory reconstruction from the trusted stream |
| Execution | R1–R4 | Physics-gated trajectory commit; R4 is the final firewall |

`test_33_invariants.py` verifies all 33 formally on every commit.

---

## The encoders

**`RadarEncoder`** maps velocity and altitude into a 3-dimensional latent probability vector over manoeuvre classes. Confidence is an SNR proxy: `sigmoid((|z − 0.5|.mean() − 0.15) × 10)`. A healthy Radar returns confidence ≈ 0.98.

**`PitotEncoder`** maps airspeed into the same 3D latent space. When the Pitot tube freezes, the latent collapses toward `[0.00, 0.05, 0.69]` and confidence drops to ≈ 0.05. This collapse is the geometric signal the router detects — not a threshold alarm, but a structural divergence between two independent views of the same physical aircraft.

Both encoders implement `load_lora(pt_path)`. A signed rank-1 matrix pair `(A, B)` can be injected: `adapted = base + (base @ lora_A @ lora_B)`. Before injecting, the adapter's HMAC signature and `target_encoder` field are verified — a GPS Spoof LoRA addressed to the Radar encoder is silently rejected by the Pitot encoder. This is LoRA Sovereignty: the mathematical identity of every adapter is cryptographically enforced.

---

## The D_hard curriculum

Every `TRIGGER_REPLAN` event is HMAC-signed and appended to `d_hard.jsonl` with its full provenance: raw latent vectors, confidence scalars, divergence scalar, and the eventual outcome (which stream was right). This queue is the aircraft's long-term episodic memory.

During the DMN consolidation cycle (`AviationDMN.consolidate()`), events are clustered by their geometric pattern of failure. For each cluster, two artefacts are trained:

**System 1 — JSON centroid cache.** The centroid of the broken stream's latent vectors during this failure type. At inference, `AviationAdapterRouter` computes the L1 distance from the incoming broken latent to all cached centroids. A match (distance < 0.2) overrides `TRIGGER_REPLAN` with `COMMIT_TRAJECTORY` in under one millisecond, with no matrix multiplication.

**System 2 — PyTorch LoRA.** Rank-1 matrices `(A, B)` trained by AdamW to minimise `||faulty_latent + (faulty_latent @ A @ B) − target_latent||₁`. Injecting the adapter into the faulty encoder warps its geometry to match the trusted stream. The router measures a divergence near zero on the next encounter — the failure resolves before the alarm fires.

This is Safety-Learning Equivalence (Invariant V6): the same event that constitutes a safety flag (`TRIGGER_REPLAN`) is the event that constitutes a training example for the adapter that prevents the same mistake. The safety invariants and the curriculum construction invariants are identical.

---

## System 1 + System 2

The architecture is explicitly modelled on Kahneman's dual-process theory. The naming is mathematically precise.

**System 1 (identification, trust-invariant):** The JSON centroid cache. It fires regardless of how old the paired LoRA adapter is. The geometric fingerprint of a Pitot freeze failure — the spatial pattern of `[0.00, 0.05, 0.69]` in latent space — does not expire. A centroid trained on a 2022 winter storm still correctly identifies a 2025 pitot freeze as the same failure class. Identification is durable.

**System 2 (correction, perishable):** The LoRA adapter. It encodes a correction derived from a specific aircraft generation, altitude envelope, and atmospheric condition. A delta trained on one sensor variant may be wrong in sign for a successor variant three years later. System 2 is therefore gated by the temporal trust score before injection.

**These two trust profiles are architecturally separated.** System 1 fires unconditionally on a centroid match. System 2 checks trust independently and falls back to System 1-only operation if the adapter is stale — the system still names the failure correctly and routes safely, it simply does not apply an untrusted correction.

---

## Temporal decay gate

Adapters accumulate as the fleet learns. Not all accumulated knowledge remains trustworthy.

```
W = exp(−λ · Δt)

where Δt = years since the adapter was trained
      λ  = failure-class decay constant
```

| Failure class | λ | Trust half-life |
|---|---|---|
| `weather_induced` (ice, turbulence) | 0.50 | 1.4 years |
| `hardware_struct` (manufacturing defect) | 0.02 | 34.7 years |

Adapters with `W < 0.40` are refused before injection. The temporal trust score and the refusal decision are recorded in the HMAC-signed audit trail on every inference call.

**Why the two classes decay at different rates:** A pitot freeze pattern is driven by atmospheric physics — icing conditions vary by season, route, and climate. A hardware structural defect (bent sensor bracket, faulty wiring) is driven by manufacturing — the physical failure mode does not change with the weather. The system knows the difference and trusts accordingly.

`AviationHealthMonitor` manages the full adapter lifecycle: auto-detaching LoRA when raw divergence drops below the safe threshold (sensor recovered), re-classifying any new spike through System 1 before re-arming, and applying the trust check at re-arm time. If the adapter trust falls below 0.40 at re-arm, the monitor refuses to load it and flags the event for a fresh DMN consolidation cycle.

---

## Human oversight

`LethalTrifectaGuard` is an unbypassable architectural block. The system cannot simultaneously hold (1) a divergent sensor reading, (2) a synthetic trajectory reconstruction, and (3) autonomous execution authority. If all three conditions are met simultaneously, the guard intercepts execution and throws `STRUCTURAL_IMPASSE` regardless of confidence scores.

`AviationJuryNode` implements EU AI Act Art. 14 meaningful human oversight. When the routing kernel proposes a counterfactual trajectory but confidence is borderline, the AI halts and presents the pilot with a structured approval prompt on the Multi-Function Display. The pilot's decision and rationale are cryptographically signed into the Flight Data Recorder's audit ledger.

`IncidentReporterNode` captures every `CRITICAL` or `HIGH` severity event — threshold violations, impasses, sensor losses — and logs them to an immutable JSONL audit file within the 24-hour mandatory reporting window required by EU AI Act Art. 72–74.

---

## Empirical results

**Single flight (ADS-B demo — UAL2298, OpenSky Network):**

| Stage | Divergence | Decision |
|---|---|---|
| Raw (Pitot frozen) | 1.59 | TRIGGER_REPLAN |
| After DMN cycle + LoRA | 0.05 | COMMIT_TRAJECTORY |

**Large-scale synthetic validation (N = 2,000 flights, N = 1,000 anomaly holdout):**

| Metric | Result |
|---|---|
| System 1 hit rate | 100% |
| Average raw divergence (alarm threshold > 0.5) | 0.93 |
| Average System 2 divergence (post-correction) | 0.22 |
| System 2 divergence reduction | 76.7% |

**LoRA lifecycle (45-tick continuous telemetry simulation):**

```
Ticks  1– 5:  Clean cruise.  D = 0.00.  No LoRA.
Tick   6:     Pitot freezes. System 1 hit in < 1ms. LoRA loaded.
Ticks  7–20:  Frozen.        D = 1.67.  LoRA active, correcting.
Tick  25:     🟢 LoRA detached automatically. Ice cleared.
Tick  26:     🧊 Ice returns. System 1 re-classifies → pitot_freeze.
              LoRA re-armed safely from typed cache.
Tick  40:     🟢 LoRA detached again. Clean flight.
```

---

## Getting started

```bash
# Full pipeline — all ten ABCs, GraphExecutor, HMAC audit trail
python aviation_full_stack.py

# Live ADS-B demo — OpenSky telemetry, full DMN closed loop
python demo_real_world.py

# Large-scale synthetic validation — N = 2,000 flights
python demo_large_scale.py

# LoRA lifecycle — arm / monitor / detach / re-arm
python demo_health_monitor.py

# Full 33-invariant pipeline
python demo_full_33.py

# Formal invariant test suite
python test_33_invariants.py

# Temporal decay regression tests (7 tests)
python test_temporal_decay.py
```

No dependencies beyond `torch`, `numpy`, and the Lár engine. `_lar.py` bootstraps the engine path automatically.

---

## Research

The routing invariants, the Safety-Learning Equivalence theorem, and the empirical proof of domain universality are formally established in:

- Sajeev, A.V. (2026). *Universal Cognitive Routing: A Ten-Abstract-Base-Class Specification for Domain-Agnostic Agent Execution.* [doi.org/10.5281/zenodo.20278775](https://doi.org/10.5281/zenodo.20278775)
- Sajeev, A.V. (2026). *Divergence Is Not Noise: Multi-Stream Routing Without Modal Fusion and the Safety-Learning Equivalence.* [doi.org/10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781)
- Sajeev, A.V. (2026). *Architecture Is All You Need: Pre-Registration and Protocol for Empirical Validation of the Lár Training Loop.* [doi.org/10.5281/zenodo.20419182](https://doi.org/10.5281/zenodo.20419182)
- Sajeev, A.V. (2026). *Snath Robotics: Multi-Stream Divergence Routing for Humanoid Robotics.* [doi.org/10.5281/zenodo.20517446](https://doi.org/10.5281/zenodo.20517446)

---

## Domain isomorphism

Snath Aviation is one of four production instantiations proving that the V1–V6 routing contract is domain-agnostic:

| Repo | Domain | Stream A | Stream B | Failure class |
|---|---|---|---|---|
| [Snath Basis](https://github.com/snath-ai/snath-basis) | Quantitative finance | Fundamental analysis | Market signals | `market_regime` / `structural` |
| **Snath Aviation** | Aviation sensor routing | Radar | Pitot tube | `weather_induced` / `hardware_struct` |
| [Snath Robotics](https://github.com/snath-ai/snath-robotics) | Humanoid sensor routing | Vision | Proprioception | `environmental_transient` / `hardware_structural` |
| [Snath Research](https://github.com/snath-ai/snath-research) | Scientific claim verification | Paper claims | Peer reviews | `scope_overclaim` / `methodology_gap` |

The temporal decay formula `W = exp(−λ · Δt)`, the identification/correction trust asymmetry, and the System 1/System 2 pipeline are **identical across all instantiations**. The λ constants and failure-class labels are the only domain-specific parameters. This is the empirical claim of universal cognitive routing: the same mathematical spine governs financial markets, aviation safety, humanoid robotics, and scientific publishing without modification.

---

*Apache 2.0 — Snath AI Open Source Research Initiative*
