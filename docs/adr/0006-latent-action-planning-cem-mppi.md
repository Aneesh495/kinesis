# ADR-0006: Latent Action Planning via CEM and MPPI Path Integrals

## Status
Accepted

## Context
Model-predictive trajectory planning in learned latent spaces requires stochastic optimization algorithms capable of navigating non-convex reward surfaces without getting trapped in local extrema or suffering from divergence when gradients are noisy.

## Decision
We implement two complementary stochastic latent trajectory planners in `kinesis.planning`:
1. `CrossEntropyMethodPlanner (CEM)`:
   - Iteratively fits a time-varying diagonal Gaussian distribution over action sequences.
   - Evaluates $K$ candidate sequences in parallel via vectorized latent imagination.
   - Selects top $E$ elite sequences and refits mean and variance with momentum smoothing.
   - Warm-starts consecutive timesteps by shifting previous solution distributions forward.
2. `ModelPredictivePathIntegralPlanner (MPPI)`:
   - Perturbs a nominal control trajectory with Gaussian noise innovations.
   - Computes path cost expectations using importance sampling with exponential weights $w_k \propto \exp(-S_k / \lambda)$.
   - Updates nominal trajectory as a convex combination of candidate perturbations.

All imagination rollouts execute in PyTorch with `torch.no_grad()` to ensure zero memory leaks and deterministic computation.

## Consequences
### Positive
- Allows zero-shot policy synthesis directly from learned dynamics and reward predictions.
- Eliminates reliance on policy gradient estimation through long recurrence horizons.
- Naturally incorporates epistemic uncertainty penalties during rollout evaluation.

### Negative
- Increases per-step action latency compared to feedforward policy inference.
