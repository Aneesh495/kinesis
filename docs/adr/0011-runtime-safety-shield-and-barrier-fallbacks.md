# ADR-0011: Runtime Safety Shield and Barrier Function Fallbacks

## Status
Accepted

## Context
Latent trajectory planners (CEM, MPPI) and neural policies operate by optimizing predicted rewards. Under physical shifts, model inaccuracies, or out-of-distribution observations, the planner may propose actions that violate physical limits, trigger mechanical resonance, or lead to irreversible states. An active runtime supervisor is required to guarantee physical safety without relying entirely on the world model's optimism.

## Decision
We implement an online safety supervisor architecture in `kinesis.safety.shield`:
1. `Multi-Criterion Audit`: Every action proposal is audited against two independent safety gates:
   - Epistemic Uncertainty Gate: Rejects actions when world model ensemble disagreement $\sigma_{\text{epistemic}} > \sigma_{\max}$.
   - Barrier Margin Gate: Verifies discrete Control Barrier Function (CBF) invariance condition $h(x_{t+1}) - h(x_t) \ge -\gamma h(x_t)$.
2. `Safe Fallback Policy`: If either safety condition is violated, the shield overrides the proposed action with a verified conservative fallback policy (e.g. passive damping, zero torque, or Riccati LQR).
3. `Smooth Recovery`: When observations return to trusted safe sets, control authority is seamlessly restored to the primary latent planner.

## Consequences
### Positive
- Enforces strict forward invariance of defined safe sets.
- Prevents catastrophic damage during online exploratory planning.
- Decouples performance optimization from safety enforcement.

### Negative
- Conservative barrier limits may prematurely restrict aggressive task-achieving maneuvers.
- Requires defining domain-specific barrier certificate functions for each physical platform.
