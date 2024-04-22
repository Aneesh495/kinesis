# ADR-0012: Counterfactual Twin-World Interventions and Causal Auditing

## Status
Accepted

## Context
Standard policy evaluation compares average returns across disparate randomized episodes, confounding physical shift effects with variations in initial conditions and environment noise. To isolate the true causal impact of an intervention (e.g. $\operatorname{do}(\text{mass} = 1.8 \times)$ or an action override $\operatorname{do}(u_t = u'_t)$), we must evaluate counterfactuals: what would have happened under the exact same noise realizations if the intervention had taken place?

## Decision
We implement a Twin-World simulation engine in `kinesis.evaluation.twin_world`:
1. `Aligned Twin Execution`: Concurrently initialize a factual environment $E_F$ and a counterfactual environment $E_{CF}$ sharing the exact same random seed and initial state.
2. `Targeted Pearl Interventions`: Apply do-calculus interventions to either:
   - Physical shift profiles: $\operatorname{do}(\text{profile} = P_{CF})$.
   - Action trajectory modifications: $\operatorname{do}(u_t = u'_t)$ for designated timesteps $t$.
3. `Causal Divergence Metrics`: Track per-step observation divergence $\|o_t^F - o_t^{CF}\|_2$, reward gap $r_t^F - r_t^{CF}$, and estimate finite-time Lyapunov divergence exponents:
   $$\lambda \approx \frac{1}{T} \ln \frac{\|d_T\|_2}{\|d_0\|_2}$$

## Consequences
### Positive
- Isolates individual treatment effects without variance from randomized initial states.
- Quantifies physical sensitivity and stability margins under precise physical parameter perturbations.
- Provides causal counterfactual benchmarks for evaluating adaptive control policies.

### Negative
- Doubles simulator execution time since two parallel physics instances are stepped synchronously.
