# Snath Aviation: Edge Cases and Re-Planning

While the architecture of Snath Aviation is mathematically robust, aviation is a domain of infinite edge cases. This document outlines how the Lár-JEPA architecture handles worst-case scenarios compared to standard autopilots, and provides a deep dive into the mechanics of the `TRIGGER_REPLAN` scenario and the `AviationPerturbationOperator`.

---

## 1. Edge Case Superiority

### Edge Case A: The "Dual Failure" (Volcanic Ash)
**The Scenario:** The plane flies through a cloud of volcanic ash. *Both* the Pitot tube and the Radar fail at the exact same time, both reporting `0 m/s`.
* **Standard Autopilot:** It reads `0 m/s` airspeed from the blended sensor feed, assumes the plane is stalling, violently pitches the nose down to gain speed, and flies the plane into the ocean. 
* **Snath Aviation:** Because both sensors failed to `0`, their *confidence scalars* (`c_a` and `c_b`) drop to near-zero (e.g., `0.05`). The Router hits the rule: `c_a < 0.1 and c_b < 0.1 -> STRUCTURAL_IMPASSE`. The AI instantly realizes it is completely blind, refuses to guess, and hands manual control to the pilot.

### Edge Case B: The "Silent Poison" (Sensor Drift)
**The Scenario:** A sensor doesn't snap off; it slowly drifts out of calibration, feeding slightly wrong data into the system over hours.
* **Standard Autopilot:** The drift is averaged into the fused sensor state. The autopilot slowly banks the plane off course without triggering hard-coded threshold alarms.
* **Snath Aviation:** Because Snath uses **Co-Embedding** (Invariant V4), the drifted sensor slowly pulls away from the healthy sensor in the 3D latent space. The Divergence scalar slowly climbs. The moment Divergence crosses `0.5`, the Router mathematically severs the drifting sensor and triggers a REPLAN. 

### Edge Case C: The "Hallucination"
**The Scenario:** The AI pulls a PyTorch LoRA to heal a broken sensor, but the Neural Network hallucinates a wildly incorrect flight path.
* **Standard Deep Neural Network (God Model):** It executes the hallucinated maneuver.
* **Snath Aviation:** The LoRA hallucinates the bad vector. But before the plane can move, the `AviationRoutingKernel` runs **Invariant R4 (Physical Reachability)**. It sees the vector violates the laws of physics. Furthermore, the **Lethal Trifecta Guard** physically blocks the maneuver because the pilot hasn't clicked `approve`. The AI throws an impasse and disconnects. The hallucination is trapped safely inside the software.

---

## 2. What a REPLAN Scenario Looks Like in Action

When the `AviationDivergenceRouter` detects a confident disagreement (e.g., Radar says 274 m/s, Pitot says 0 m/s), it issues a `TRIGGER_REPLAN`. Here is exactly what happens next in real-time (System 1):

1. **The Freeze:** The autopilot temporarily pauses the execution of the flight control maneuver. 
2. **The Reflex (System 1):** The `AviationAdapterRouter` takes the broken Pitot vector and searches the cached `.json` centroids in O(log N) time. It asks: *"Have I seen this geometry before?"*
3. **The Match:** It finds a match (`"type": "pitot_freeze"`).
4. **The Restructuring (System 2):** It loads the corresponding `.pt` PyTorch LoRA weights into the Pitot sensor's neural network layer.
5. **The Healing:** The LoRA mathematically warps the broken Pitot data to match the healthy Radar data. The Divergence drops from `0.8` back to `0.0`.
6. **The Human Check (Lethal Trifecta):** The system generates the new flight trajectory, flags it as `SYNTHETIC AIRSPEED ENGAGED (RADAR LORA)`, and asks the `AviationJuryNode` (the Pilot) to approve.
7. **The Commit:** The pilot approves, and the plane flies safely on the reconstructed vector.

If step 3 fails (no match found), the system instantly escalates to a `STRUCTURAL_IMPASSE` and disconnects.

---

## 3. How the Perturbation Operator Works

The magic of Step 5 ("The Healing") is driven by the `AviationPerturbationOperator`, which implements the predictive core of the JEPA architecture.

In standard physics, to see what happens when a sensor fails, you would have to physically break a sensor on a real plane. In a JEPA, we do it in **latent space** using Counterfactual Prediction:

```
z_predicted = z_broken + α · Δ
```

* `z_broken`: The latent vector currently coming from the frozen Pitot tube.
* `Δ (Delta)`: The perturbation vector. In `Snath Aviation`, the PyTorch LoRA matrices ($A$ and $B$) actually *represent* this perturbation vector mathematically! When we do `base + (base @ A @ B)`, we are applying the exact minimum-norm perturbation required to shift the broken geometry back to the healthy geometry.
* `α (Alpha)`: The magnitude (usually 1.0 for a full heal).

The Perturbation Operator takes the broken context (the Student), applies the learned LoRA weights (the Predictor), and generates the counterfactual healthy state (the Teacher's target) without needing to query the real world. It mathematically hallucinates the truth!
