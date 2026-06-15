# Snath Aviation: Architecture

This document is the technical reference for the Snath Aviation cognitive architecture. It describes every component, every data flow, every invariant, and every design decision — with precise references to the codebase.

---

## Overview: The Three-Layer Architecture

Snath Aviation is built in three completely decoupled layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LAYER 3: LEARNING                            │
│          AviationDMN · BasisDMN · AviationAdapterRouter             │
│          AviationHealthMonitor · Typed LoRA Cache                   │
│                                                                     │
│   Trains overnight. Heals geometry. Never touches Layer 2.          │
├─────────────────────────────────────────────────────────────────────┤
│                        LAYER 2: ROUTING                             │
│                   AviationDivergenceRouter                          │
│          (ZERO trainable parameters — mathematically frozen)        │
│                                                                     │
│   Pure L1 geometry. FAA-certifiable. Cannot be retrained.           │
├─────────────────────────────────────────────────────────────────────┤
│                        LAYER 1: PERCEPTION                          │
│            RadarEncoder · PitotEncoder (torch.nn.Module)            │
│                                                                     │
│   Independent neural encoders. Learned via PyTorch LoRA.            │
└─────────────────────────────────────────────────────────────────────┘
```

**The cardinal rule:** Learning flows upward from Layer 3 into Layer 1 only. Layer 2 is mathematically immutable. Nothing from the learning layer ever modifies the routing core.

---

## Layer 1: Perception — The Independent Encoders

### The Difference Between Architecture and Engine
Snath Aviation is structurally a **JEPA (Joint Embedding Predictive Architecture)** because it maps different modalities (air pressure and inertial movement) into a shared latent space and predicts counterfactual vectors. However, the *engine* executing this math is **PyTorch**. The encoders are `torch.nn.Module` subclasses using Linear projection layers, and the adapters use PyTorch `.pt` LoRA weights. There are no generative models (like LLMs or Diffusion) in the flight loop—only deterministic linear algebra.

### Design: Stream Independence (Invariant V1)

Both encoders are `torch.nn.Module` subclasses that also implement `AbstractModalEncoder` from the Lár-JEPA engine. They are completely independent — they do not share parameters, activations, or state.

```
Raw Telemetry
      │
      ├──────────────────────────┐
      │                          │
      ▼                          ▼
RadarEncoder                PitotEncoder
(torch.nn.Module)           (torch.nn.Module)
      │                          │
      │  Base proj: 3→3 identity │  Base proj: 3→3 identity
      │  LoRA: base + (x @ A @ B)│  LoRA: base + (x @ A @ B)
      │                          │
      ▼                          ▼
   z_radar ∈ ℝ³             z_pitot ∈ ℝ³
   c_radar ∈ [0,1]          c_pitot ∈ [0,1]
```

### Latent Space Geometry

Both encoders project telemetry into a 3-dimensional latent manifold normalised to `[0, 1]³`:

```python
# RadarEncoder
vel_norm  = min(velocity / 300.0, 1.0)   # normalised velocity
alt_norm  = min(altitude / 15000.0, 1.0) # normalised altitude
z_radar   = [vel_norm, 0.90, alt_norm]   # confidence axis is 0.90 for Radar

# PitotEncoder (healthy)
z_pitot   = [vel_norm, 0.90, alt_norm]   # matches Radar when both healthy

# PitotEncoder (frozen)
z_pitot   = [0.00, 0.05, alt_norm]       # velocity and confidence collapse
```

When both sensors are healthy, `z_radar ≈ z_pitot`. When a sensor fails, the L1 distance between them spikes.

### LoRA Injection: W' = W + AB

Each encoder has three additional attributes injected by the DMN sleep cycle:

```python
self.lora_A: torch.Tensor | None = None   # shape: (3, 1)
self.lora_B: torch.Tensor | None = None   # shape: (1, 3)

def encode(self, telemetry):
    base = self.proj(raw)
    if self.lora_A is not None:
        adapted = base + torch.matmul(torch.matmul(base, self.lora_A), self.lora_B)
        return adapted.numpy()
    return base.numpy()
```

The LoRA matrices are Rank-1: `A ∈ ℝ^(d×1)`, `B ∈ ℝ^(1×d)`. Their product `AB ∈ ℝ^(d×d)` is a rank-1 perturbation of the identity — the minimum-norm modification needed to close the geometric gap between the faulty and healthy streams.

### load_lora() — Surgical Loading

```python
def load_lora(self, pt_path: str) -> None:
    state = torch.load(pt_path, weights_only=True)
    
    # 1. HMAC Sovereignty Verification
    a_hash = hashlib.sha256(state["A"].numpy().tobytes()).hexdigest()[:16]
    b_hash = hashlib.sha256(state["B"].numpy().tobytes()).hexdigest()[:16]
    expected_sig = hmac.new(KEY, f"pitot|{a_hash}|{b_hash}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(state.get("hmac_hex", ""), expected_sig):
        raise ValueError("HMAC VERIFICATION FAILED")

    # 2. Type-Checking
    if state.get("target_encoder") == "pitot":
        self.lora_A = state["A"]
        self.lora_B = state["B"]
```

The `.pt` file contains a `target_encoder` field. The encoder only accepts weights explicitly intended for it — a GPS Spoof LoRA targeting the Radar encoder will be silently rejected by the Pitot encoder's `load_lora()`.

---

## Layer 2: Routing — The Frozen Mathematical Core

### Co-Embedding vs. Modal Fusion (Invariant V4)
A critical distinction in Snath Aviation is that the sensors are **Co-Embedded**, not fused. Modal fusion (like concatenating vectors) destroys the contradiction signal by averaging out the broken sensor with the healthy one. Instead, `Snath Aviation` maps both vectors to the same 3D coordinate system but never lets them touch. The router merely acts as a ruler, measuring the distance (Divergence) between them. This satisfies **Invariant V4: Content Blindness**.

### The AviationDivergenceRouter

```python
class AviationDivergenceRouter(AbstractDivergenceRouter):
    def divergence(self, z_a, z_b):
        # 1. NaN Guard (Fail-Safe)
        if np.isnan(z_a).any() or np.isnan(z_b).any(): return 2.0  # Forces IMPASSE
        
        # 2. Probability-Vector (Total Variation) Divergence
        p_a = np.exp(z_a) / np.sum(np.exp(z_a))
        p_b = np.exp(z_b) / np.sum(np.exp(z_b))
        dim = len(p_a)
        return float(np.sum(np.abs(p_a - p_b)) / np.sqrt(dim))

    def route(self, c_a, c_b, divergence):
        if divergence > 0.5 and c_a > 0.8 and c_b > 0.8:
            return RouteDecision.TRIGGER_REPLAN
        return RouteDecision.COMMIT_TRAJECTORY
```

This routing logic implements **Probability-Vector (Total Variation) Divergence**. As proven in the `Snath Locus` AIA theorem, routing on raw L1 distance is amplitude-sensitive and prone to false-positive `REPLAN` triggers. By applying a `softmax` to map latents to probability vectors before calculating divergence, we get a magnitude-invariant distance that mathematically halves the false-positive REPLAN rate. Additionally, the **NaN Guard** traps catastrophic encoder failure, preventing corrupt signals from evaluating to `False` on the threshold check and bypassing the router into a dangerous `COMMIT` state.

**It will never change.** There are no hyperparameters to tune, no weights to update, no hidden state to corrupt. The FAA can read these lines and formally verify them.

### The Routing Decision Matrix

| Divergence | Confidence A | Confidence B | Decision |
|---|---|---|---|
| D < 0.5 | any | any | `COMMIT_TRAJECTORY` — sensors agree |
| D ≥ 0.5 | c_a > 0.8 | c_b > 0.8 | `TRIGGER_REPLAN` — both confident, disagree |
| D ≥ 0.5 | c_a ≤ 0.8 | any | `COMMIT_TRAJECTORY` — low-confidence sensor deferred |
| — | c_a < 0.1 | c_b < 0.1 | `STRUCTURAL_IMPASSE` — no signal at all |

### The 33 Invariants: Full Pipeline

The router is one stage in a 33-invariant pipeline. Each stage has a strict mathematical contract:

```
STAGE 1: PERCEPTION (M1–M3)
  M1: encode(x).shape[-1] == output_dim         (output dimension contract)
  M2: modality is a stable, unique string        (stream identity contract)
  M3: encode() is deterministic given same input (reproducibility)

STAGE 2: ROUTING (V1–V6)
  V1: stream_a and stream_b share no parameters  (independence)
  V2: divergence(z_a, z_b) ≥ 0 always           (non-negativity)
  V3: divergence is not required to be symmetric  (asymmetry allowed)
  V4: route() sees only scalars (c_a, c_b, d)    (content-blindness)
  V5: every (c_a, c_b, d) maps to exactly one RouteDecision (completeness)
  V6: STRUCTURAL_IMPASSE == max-signal safety event (equivalence theorem)

STAGE 3: FAULT LOCATION (I1–I6, A1–A6)
  I1–I6: Environmental state encoding invariants
  A1–A6: Cross-attention fault targeting invariants

STAGE 4: COUNTERFACTUAL (P1–P6)
  P1–P6: Perturbation operator and wildtype reconstruction invariants

STAGE 5: EXECUTION (R1–R4)
  R1: trajectory score ∈ [0, 1]
  R2: score > 0.5 → COMMIT_TRAJECTORY
  R3: score ≤ 0.5 → STRUCTURAL_IMPASSE
  R4: trajectory must be physically reachable     (the un-bypassable firewall)
```

**R4 is the final safety invariant.** If any upstream component — including a malfunctioning LoRA adapter — generates a trajectory that violates physical reachability, R4 rejects it. The AI self-disconnects. It is physically impossible for the learning system to command a maneuver that kills the aircraft.

---

### Regulatory Compliance: EU AI Act & AEPD Guardrails

Snath Aviation natively integrates the `lar.compliance` package to enforce European Union regulatory law at runtime:

1. **Lethal Trifecta Guard (AEPD Rule of 2)**: Before the `AviationRoutingKernel` can emit a `COMMIT_TRAJECTORY`, it evaluates the `LethalTrifectaGuard`. If the system detects **(1) Untrusted Input** (sensor divergence `> 0.5`) + **(2) Sensitive Data** (flight paths) + **(3) Autonomous Action** (commit maneuver), it mathematically blocks execution unless there is a recorded `approve` from the `AviationJuryNode` (the Pilot-in-the-Loop). Attempting to bypass this throws an unrecoverable `LethalTrifectaError`.
2. **Post-Market Monitoring (Art. 72–74)**: If the system encounters a `STRUCTURAL_IMPASSE` (due to NaN failures, missing human approval, or critical trajectory thresholds), the `IncidentReporterNode` automatically intercepts the failure and writes a `CRITICAL` or `HIGH` severity incident report to an immutable JSON-L ledger. This perfectly satisfies the EU AI Act's 24-hour mandatory reporting window for high-risk AI failures.
3. **Deployer Transparency & Synthetic Content Marking (Art. 13 & Art. 50)**: To prevent catastrophic mode-confusion, the AI never secretly overwrites a physical sensor dial when a LoRA adapter is active. If the Pitot freezes and the Radar LoRA patches the vector, the system must explicitly flag the output as a synthetic counterfactual to the pilot (e.g., `PITOT FAIL. SYNTHETIC AIRSPEED ENGAGED (RADAR LORA)`). This ensures the human deployer is never misled into believing a broken physical sensor is functioning properly.

---

## Layer 3: Learning — The DMN Memory System

### The Teacher-Student JEPA Mapping
While the flight loop acts as a deterministic router, the `AviationDMN` sleep cycle is a mathematically perfect implementation of Yann LeCun's **Teacher-Student JEPA architecture**:
1. **Teacher Encoder (y-encoder)**: The healthy sensor (e.g., Radar) serves as the ground-truth target.
2. **Student Encoder (x-encoder)**: The faulty sensor (e.g., Pitot) provides the corrupted, masked context.
3. **Predictor**: The PyTorch LoRA Adapter.
During the DMN sleep cycle, the system minimizes the L1 loss between the Student + Predictor and the Teacher. At inference time (during flight), the system uses *only* the Student + Predictor to hallucinate the safe aerodynamic state, even if the Teacher is completely offline.

### Data Flow: From Failure to Adaptation

```
Sensor Failure
      │
      ▼
AviationDivergenceRouter → TRIGGER_REPLAN
      │
      ▼
DHardEvent (HMAC-signed)
      │
      └──► d_hard_live.jsonl (Tier 1 episodic queue, flat-file)
                │
           [Sleep Cycle]
                │
                ▼
        AviationDMN.consolidate()
                │
        ┌───────┴───────┐
        ▼               ▼
  adapter_X.json   adapter_X.pt
  (System 1)       (System 2)
  Fast Centroid    PyTorch LoRA
```

### The DHardEvent: Tamper-Evident Provenance

Every anomaly is logged as a `DHardEvent` with full provenance:

```python
@dataclass
class DHardEvent:
    asof:       str     # timestamp of the event
    scenario_id: str    # flight identifier
    decision:   str     # "TRIGGER_REPLAN"
    basis:      float   # the divergence value
    conf_a:     float   # radar confidence
    conf_b:     float   # pitot confidence
    v_a:        list    # radar latent vector
    v_b:        list    # pitot latent vector (the broken one)
    winner:     str     # "radar" | "pitot" (resolved ground truth)
    sig:        str     # HMAC-SHA256 signature
```

The HMAC signature covers all immutable observation fields. The ground-truth label (`winner`) is appended later without invalidating the signature — a cryptographic proof that the observation was not modified retroactively.

### AviationDMN.consolidate(): The Sleep Cycle

```
D_hard events (resolved)
        │
        ▼
Cluster by failure type:
  "radar wins" events  → Pitot Freeze cluster
  "pitot wins" events  → GPS Spoof cluster
        │
        ├──── SYSTEM 1 ────────────────────────────────────────────
        │     centroid = mean(v_a or v_b across cluster)
        │     save adapter_pitot_freeze.json
        │       { "type": "pitot_freeze",
        │         "centroid_v_a": [0.743, 0.900, 0.691],
        │         "trust": "radar" }
        │
        └──── SYSTEM 2 ────────────────────────────────────────────
              target_t = tensor of winner stream vectors
              faulty_t = tensor of loser stream vectors

              A = nn.Parameter(randn(3, 1) × 0.01)
              B = nn.Parameter(randn(1, 3) × 0.01)
              optimizer = AdamW([A, B], lr=0.1)

              for 100 epochs:
                  adapted = faulty_t + (faulty_t @ A @ B)
                  loss = L1(adapted, target_t)
                  loss.backward(); optimizer.step()

              save adapter_pitot_freeze.pt
                { "A": tensor, "B": tensor,
                  "target_encoder": "pitot",
                  "final_loss": 0.0179 }
```

The LoRA training objective is: find the minimum-norm rank-1 matrix `AB` such that applying it to the faulty stream's latent output makes it indistinguishable from the healthy stream's latent output under L1 distance.

### The Adapter File System

Every adapter pair lives as two files in `models/adapters/`:

```
models/adapters/
├── adapter_pitot_freeze.json   ← System 1 centroid (lightweight, fast)
├── adapter_pitot_freeze.pt     ← System 2 LoRA weights (deep, structural)
├── adapter_gps_spoof.json
└── adapter_gps_spoof.pt
```

As the fleet accumulates experience, this directory grows. Each new failure type adds exactly one `.json` + `.pt` pair. The System 1 spatial index scales at O(log N) — adding 10,000 adapters does not meaningfully slow inference.

---

## The Kahneman Hybrid Inference Pipeline

### Full Request Flow (Anomaly Present)

```
New Flight Telemetry
        │
        ▼
┌───────────────────┐      ┌───────────────────┐
│   RadarEncoder    │      │   PitotEncoder    │
│   z_r = [0.74,   │      │   z_p = [0.00,   │
│          0.90,   │      │          0.05,   │
│          0.69]   │      │          0.69]   │
└─────────┬─────────┘      └────────┬──────────┘
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
           AviationDivergenceRouter
           divergence = |z_r - z_p|₁ = 1.59
           route(0.98, 0.90, 1.59) → TRIGGER_REPLAN
                       │
                       ▼
           AviationAdapterRouter (System 1)
           cosine_sim(z_p, centroid_pitot_freeze) = 0.98
           → Cache Hit! Decision → COMMIT_TRAJECTORY
           → Instruction: load 'adapter_pitot_freeze.pt' into PitotEncoder
                       │
           ┌───────────┴───────────────┐
           │ System 1 Result           │ System 2 Action
           │ COMMIT_TRAJECTORY         │ pitot.load_lora(adapter.pt)
           │ (instant, <1ms)           │ z_p' = z_p + (z_p @ A @ B)
           │                           │ z_p' = [0.71, 0.88, 0.68]
           └───────────┬───────────────┘
                       │
                       ▼
           AviationDivergenceRouter (re-evaluated)
           divergence = |z_r - z_p'|₁ = 0.05
           route(0.98, 0.90, 0.05) → COMMIT_TRAJECTORY
                       │
                       ▼
           Aircraft continues safely. No alarm. No intervention.
```

### AviationAdapterRouter.resolve()

```python
def resolve(self, z_radar, z_pitot, base_decision, c_radar, c_pitot):
    if base_decision != RouteDecision.TRIGGER_REPLAN:
        return base_decision, "no divergence to resolve"

    # System 1: spatial L1 search over JSON centroids
    for adapter in self.adapters:
        if adapter["type"] == "pitot_freeze":
            dist = np.sum(np.abs(z_pitot - np.array(adapter["centroid_v_a"])))
            if dist < 0.2:
                return RouteDecision.COMMIT_TRAJECTORY, "Cache Hit (Pitot Freeze)"
        elif adapter["type"] == "gps_spoof":
            dist = np.sum(np.abs(z_radar - np.array(adapter["centroid_v_b"])))
            if dist < 0.2:
                return RouteDecision.COMMIT_TRAJECTORY, "Cache Hit (GPS Spoof)"

    return RouteDecision.TRIGGER_REPLAN, "Cache Miss. System 2 required."
```

---

## The LoRA Lifecycle: AviationHealthMonitor

### State Machine

```
                     ┌──────────────────────────┐
                     │     HEALTHY (no LoRA)     │
                     └──────────────────────────┘
                          │              ▲
        divergence ≥ 0.5  │              │  consecutive_safe_ticks
        System 1 hit       │              │  ≥ recovery_window
                          ▼              │
                     ┌──────────────────────────┐
                     │   ACTIVE (LoRA loaded)   │
                     │   adapter_type = "X"     │
                     └──────────────────────────┘
                          │              ▲
        raw_div < 0.5     │              │  raw_div spikes back ≥ 0.5
        (ticking up)       │              │  System 1 re-classifies → "X"
                          ▼              │
                     ┌──────────────────────────┐
                     │     RECOVERING           │
                     │ (LoRA still loaded but   │
                     │  counting safe ticks)    │
                     └──────────────────────────┘
```

### The Typed Cache: Cross-Contamination Prevention

```python
# WRONG (old design) — keyed only by encoder name:
self._lora_cache: dict[str, tuple] = {}
# "pitot" → (A_freeze, B_freeze)
# If a different anomaly occurs on the pitot, the wrong LoRA is re-armed!

# CORRECT (current design) — keyed by (encoder_name, adapter_type):
self._lora_cache: dict[tuple[str, str], tuple] = {}
# ("pitot", "pitot_freeze") → (A_freeze, B_freeze)
# ("pitot", "gps_spoof")    → (A_spoof,  B_spoof)
# A GPS Spoof LoRA can never be loaded onto a Pitot Freeze and vice versa.
```

### Re-arm Safety: System 1 Re-classification

When divergence returns after a LoRA detachment, the monitor **never blindly loads the previously cached adapter**. It routes the new anomaly through System 1 to re-classify it geometrically:

```python
# On re-arm attempt:
sys1_dec, sys1_note = self.adapter_router.resolve(z_a, z_b_raw, ...)

# Parse the failure type from System 1's note:
if "Pitot Freeze" in sys1_note:   adapter_type = "pitot_freeze"
elif "GPS Spoof" in sys1_note:    adapter_type = "gps_spoof"

# Load ONLY the typed cache entry that matches the re-classified failure:
cached = self._lora_cache.get((encoder_name, adapter_type))
```

If the new anomaly is a different type from the previous one, the monitor correctly identifies it as a new unknown failure and flags it for a new DMN consolidation cycle.

### Temporal Decay: Adapter Trust Degradation

A LoRA adapter trained on Pitot freeze events from three northern-winter seasons ago may not accurately represent today's atmospheric conditions on a different route. The `AviationHealthMonitor` implements a **Temporal Decay** model — borrowed from the TemporalNode in Snath Locus — that assigns each cached adapter a time-decaying trust weight evaluated at every re-arm attempt.

**Trust Weight Formula:**

```
W = exp(-λ · Δt)
```

- `Δt` = years elapsed since the adapter was written by `AviationDMN.consolidate()`
- `λ` = decay rate, set per failure class

**Decay Rate Table:**

| `failure_class` | λ | Rationale |
|---|---|---|
| `weather_induced` (ice, turbulence) | 0.50 | Seasonal atmospheric patterns change year-to-year — fast decay |
| `hardware_struct` (manufacturing defects, bent sensors) | 0.02 | Intrinsic to the aircraft's physical construction — near-permanent |
| *(default)* | 0.10 | General sensor drift and unexplained anomalies |

**Decision Logic at Re-arm:**

| Trust Weight | Decision |
|---|---|
| `W ≥ 0.40` | **RE-ARM** — adapter is within acceptable trust envelope |
| `W < 0.40` | **STALE** — refuse to load; flag event for fresh DMN consolidation |

When a stale refusal fires, the monitor logs a `STALE_ADAPTER` event and routes the anomaly as if no cached adapter existed — triggering a new DMN sleep cycle to train a fresh adapter on current data. This mirrors TemporalNode's behaviour: when evidence quality falls below the viability floor, the correct response is not to act on degraded evidence but to demand new evidence.

**Revival Signaling:** When a failure type that was previously detached re-appears after a gap (e.g., `pitot_freeze` returns after the aircraft had 30 clean flights), the monitor fires a `REVIVAL_SIGNAL` event to the maintenance system. A recurring failure pattern is qualitatively different from an isolated one — it warrants structured investigation, not just another re-arm.

**Adapter File Format (post-decay support):** The `.json` files written by `AviationDMN` now include two additional fields:

```json
{
  "type": "pitot_freeze",
  "centroid_v_a": [0.743, 0.900, 0.691],
  "trust": "radar",
  "created_at": "2026-01-15T03:22:11Z",
  "failure_class": "weather_induced"
}
```

**`temporal_audit()` Output Format:**

```
=== Temporal Adapter Audit ===
('pitot', 'pitot_freeze')  age=0.87y  λ=0.50  W=0.644  [██████░░░░] TRUSTED
('radar', 'gps_spoof')     age=2.10y  λ=0.10  W=0.810  [████████░░] TRUSTED
('pitot', 'ice_crystal')   age=4.50y  λ=0.50  W=0.105  [█░░░░░░░░░] ⚠ STALE
```

---



---

## Future Graph Architecture: Lár Engine Integration

While the current reference implementation runs procedurally (e.g., `demo_real_world.py`), the theoretical model maps perfectly to the core graph executor of the Lár engine (`lar/node.py`). Two specific Lár nodes provide massive architectural benefits for aviation:

### 1. BatchNode (Redundant Sensor Parallelism)
In a real A380 or Boeing 777, there are dozens of sensor streams (3x Pitot, 2x Radar, GPS, IMU, Angle of Attack). A `BatchNode` fans out execution to process all streams concurrently. 
**Crucial Safety Feature:** The Lár `BatchNode` implements a `branch_timeout`. If a Pitot tube physically shorts out and hangs the avionics bus, the `branch_timeout` drops the dead sensor and synthesizes the remaining streams. The main cognitive loop never freezes.

### 2. HumanJuryNode (Pilot-in-the-Loop Override)
The Lár `HumanJuryNode` satisfies EU AI Act Art 14 (Automation Boundary) by pausing execution for a human reviewer. 
**Aviation Mapping:** This is the Multi-Function Display (MFD) prompt for the Pilots. If the `AviationRoutingKernel` computes a borderline trajectory safety score (e.g., `0.55`), it triggers the `AviationJuryNode` (`HumanJuryNode`). The AI asks the human pilots to explicitly `COMMIT_TRAJECTORY` or `DISCONNECT_AUTOPILOT`. The decision is cryptographically recorded in the Flight Data Recorder (via an `authority_ledger`), mirroring Cockpit Voice Recorder (CVR) compliance.

---

## Component Reference

| File | Component | Role |
|---|---|---|
| `demo_real_world.py` | `RadarEncoder` | Stream A encoder (torch.nn.Module) |
| `demo_real_world.py` | `PitotEncoder` | Stream B encoder (torch.nn.Module + LoRA) |
| `demo_real_world.py` | `AviationDivergenceRouter` | Frozen routing core (V1–V6) |
| `demo_real_world.py` | `AviationFaultLocator` | Cross-attention fault targeting (I1–I6, A1–A6) |
| `demo_real_world.py` | `AviationPerturbationOperator` | Counterfactual trajectory (P1–P6) |
| `demo_real_world.py` | `AviationRoutingKernel` | Physics safety gate (R1–R4) |
| `dhard.py` | `DHardEvent` | HMAC-signed anomaly log entry |
| `dhard.py` | `DHardQueue` | Persistent JSONL event queue |
| `dmn/aviation_dmn.py` | `AviationDMN` | Sleep cycle: trains JSON + LoRA adapters |
| `dmn/adapter_router.py` | `AviationAdapterRouter` | System 1: O(log N) spatial cache |
| `dmn/health_monitor.py` | `AviationHealthMonitor` | LoRA lifecycle manager + typed re-arm cache |

---

## Invariant Violations and Their Consequences

| Violation | Consequence |
|---|---|
| LoRA outputs a latent vector far from the safe manifold | AviationDivergenceRouter sees high divergence → TRIGGER_REPLAN |
| TRIGGER_REPLAN propagates to Fault Locator | FaultLocator identifies and targets the anomalous encoder |
| Counterfactual trajectory reconstruction fails | AviationRoutingKernel scores < 0.5 |
| Score < 0.5 | Router throws STRUCTURAL_IMPASSE — AI self-disconnects |
| STRUCTURAL_IMPASSE | Aircraft reverts to hardware fallbacks. Pilots alerted. |

At no point in this chain does the learning system (LoRA) have the authority to bypass the physical safety invariants. The worst case outcome of a malfunctioning adapter is graceful AI disconnection — not a fatal maneuver.

---

## Empirical Verification Summary

| Test | Command | Result |
|---|---|---|
| 33-invariant pipeline | `python demo_full_33.py` | All 33 invariants pass |
| Live OpenSky telemetry | `python demo_real_world.py` | UAL2298: div 1.59 → 0.05 post-LoRA |
| Large scale (N=2000) | `python demo_large_scale.py` | System 1: 100% hit rate; System 2: 76.7% div reduction |
| LoRA lifecycle (45 ticks) | `python demo_health_monitor.py` | Detach at tick 25, re-arm at tick 26, detach at tick 40 |

---

*Built on the Lár-JEPA cognitive architecture (Apache 2.0).*
*Reference: Divergence Is Not Noise: Multi-Stream Routing Without Modal Fusion and the Safety-Learning Equivalence (Sajeev, 2026).*
