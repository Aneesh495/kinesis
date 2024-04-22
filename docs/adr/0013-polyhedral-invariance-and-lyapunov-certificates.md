# ADR-0013: Polyhedral Invariance and Lyapunov Stability Certificates

## Status
Accepted

## Context
Data-driven policies and world models are susceptible to drift and unbounded trajectories when encountering unseen operational disturbances. Empirical testing over a handful of rollouts does not provide rigorous guarantees that a system will asymptotically converge or remain inside admissible state envelopes. Formal certificates of Lyapunov decay and invariant set containment are required.

## Decision
We implement analytical verification engines in `kinesis.analytic.lyapunov` and `kinesis.safety.polyhedral`:
1. `Discrete Lyapunov Equation Solver`: Solve the discrete Lyapunov equation $A^\top P A - P = -Q$ for positive definite $P \succ 0$.
2. `Quantitative Energy Certificates`: Compute quadratic Lyapunov energy $V(x) = x^\top P x$ along candidate or executed trajectories. Certify asymptotic stability when $V(x_{t+1}) < V(x_t)$ strictly decreases monotonically.
3. `Polyhedral Constraint Auditor`: Represent safe operation envelopes as convex polyhedra $\mathcal{S} = \{ x \in \mathbb{R}^n \mid H x \le h \}$. Compute normalized Euclidean distances to all constraint hyperplanes.
4. `Quadratic Programming Projection`: Project violating states back onto the safe set boundary via minimum Euclidean distance quadratic optimization:
   $$\min_y \frac{1}{2} \|y - x\|_2^2 \quad \text{subject to} \quad H y \le h$$

## Consequences
### Positive
- Provides mathematically rigorous certification of trajectory stability and decay rates.
- Generates Euclidean safety margins that can be monitored by runtime safety shields.
- Guarantees feasible safe set projections when disturbances push the system across boundaries.

### Negative
- Solving quadratic programs for set projection incurs slight numerical overhead during real-time loops.
