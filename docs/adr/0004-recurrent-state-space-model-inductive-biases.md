# ADR-0004: Recurrent State-Space Model Inductive Biases

## Status
Accepted

## Context
Purely deterministic recurrent models (such as LSTMs or GRUs) struggle with multi-modal transitions, stochastic environments, and compounding prediction drift. Conversely, purely stochastic state models often lose multi-step temporal dependencies.

## Decision
We implement a Recurrent State-Space Model (RSSM) featuring split deterministic and stochastic representations:
1. `Deterministic Path`: Gated Recurrent Unit (GRU) producing memory representation $h_t = \text{GRU}(h_{t-1}, [z_{t-1}, a_{t-1}])$.
2. `Stochastic Path`: Diagonal Gaussian distribution $z_t \sim \mathcal{N}(\mu, \sigma^2)$ reparameterized for backpropagation.
3. `Balanced KL Divergence`: Gradient updates balanced between prior and posterior via $\alpha = 0.8$ weighting and a free-bits margin $\beta = 1.0$.
4. `Modular Prediction Heads`: Observation reconstruction head, reward prediction head, discount/continue prediction head, and optional constraint cost head.

We verify causal prefix invariance in unit tests: appending transitions to future timesteps cannot alter past hidden states or priors.

## Consequences
### Positive
- Robust multi-step rollouts without accumulating exponential compounding errors.
- Expressive representations capable of capturing non-Markovian partial observability.
- Natural separation between deterministic history integration and stochastic transition variability.

### Negative
- Requires careful KL divergence balancing and clipping to prevent posterior collapse or exploding latents.
