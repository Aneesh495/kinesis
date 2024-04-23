# ADR-0014: Pareto Frontier Optimization for Multi-Objective Risk Planning

## Status
Accepted

## Context
In continuous control under epistemic uncertainty, single-scalar reward objectives fail to represent the trade-off between maximizing task progress, minimizing epistemic model risk, and reducing actuator wear. Ad-hoc scalar weight tuning often results in either hyper-conservative policies that freeze or reckless policies that exploit model delusions. A structured multi-objective Pareto optimization framework is required.

## Decision
We implement multi-objective trajectory selection in `kinesis.planning.pareto`:
1. `Multi-Objective Evaluation`: Evaluate candidate action sequences across vector objectives $J(u) = [J_{\text{reward}}, -J_{\text{risk}}, -J_{\text{effort}}]^\top$.
2. `Non-Dominated Sorting`: Partition candidate trajectories into non-dominated Pareto fronts $\mathcal{F}_0, \mathcal{F}_1, \dots$ based on strict component-wise dominance.
3. `Crowding Distance Diversity`: Preserve solution diversity along the Pareto front using NSGA-II crowding distance metrics.
4. `Chebyshev Utopia Selection`: Select the operating compromise trajectory using weighted Chebyshev metric minimization to the Utopia point:
   $$\min_{c \in \mathcal{F}_0} \max_m w_m (u^*_m - J_m(c))$$

## Consequences
### Positive
- Prevents scalarization collapse where single objective weights ignore catastrophic risks.
- Maintains a diverse set of fallback plans along the trade-off boundary.
- Allows operators to adjust risk-aversion weights dynamically at runtime without re-training models.

### Negative
- Evaluating non-dominated sorting and crowding distances scales quadratically $\mathcal{O}(M N^2)$ with the candidate population size.
