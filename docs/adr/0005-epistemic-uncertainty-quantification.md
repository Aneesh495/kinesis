# ADR-0005: Epistemic Uncertainty Quantification via Ensemble Disagreement

## Status
Accepted

## Context
Standard neural world models often exhibit overconfident hallucinations in out-of-distribution regions of state-space. If a trajectory planner evaluates candidate actions using a single overconfident model, it can select catastrophic actions that exploit model error rather than genuine environment reward dynamics.

## Decision
We implement an ensemble committee architecture in `kinesis.uncertainty.ensemble`:
1. `Committee Size`: Minimum $M = 3$ world models trained with diverse initializations and independent data shuffling.
2. `Disagreement Metric`: Empirical variance across committee predictions in reward space and latent feature space:
   $$\sigma^2(s, a) = \frac{1}{M-1} \sum_{m=1}^M \left(f_m(s, a) - \bar{f}(s, a)\right)^2$$
3. `Uncertainty Penalty`: Objective functions penalize high disagreement:
   $$J(s, a) = \hat{r}_{mean}(s, a) - \lambda \cdot \sigma_{disagreement}(s, a)$$

Unit tests verify that out-of-distribution observation inputs produce higher disagreement scores than training data by a statistically significant margin.

## Consequences
### Positive
- Prevents planning algorithms from exploiting model hallucinations.
- Guides exploration toward under-explored regions during active collection.
- Supplies clear runtime telemetry when the physical system enters unmodeled regimes.

### Negative
- Multiplies inference and imagination compute costs linearly by ensemble size $M$.
