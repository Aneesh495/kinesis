# ADR-0010: Normalizing Flow Prior Densities for Multi-Modal Transitions

## Status
Accepted

## Context
Standard world model formulations (e.g. Dreamer) assume diagonal Gaussian distributions for latent transition dynamics: $p(s_t \mid s_{t-1}, a_{t-1}) = \mathcal{N}(\mu, \sigma^2 I)$. In continuous physical manipulation and locomotion tasks, contact mechanics and friction stick-slip transitions frequently induce multi-modal, highly non-Gaussian state distributions. Enforcing a unimodal Gaussian assumption forces the model to average over discrete modes, leading to blurred predictions and poor planning.

## Decision
We implement RealNVP Normalizing Flows in `kinesis.models.flow` to model expressive latent prior distributions:
1. `Affine Coupling Layers`: Partition the latent state $z = [z_a, z_b]$ and transform using conditioned scale $s$ and translation $t$ networks:
   $$y_a = z_a$$
   $$y_b = z_b \odot \exp(s(z_a)) + t(z_a)$$
2. `Exact Invertibility and Jacobian`: The Jacobian matrix is triangular, yielding exact log determinant in $\mathcal{O}(d)$ time:
   $$\log \left| \det \frac{\partial y}{\partial z} \right| = \sum_i s(z_a)_i$$
3. `Base Gaussian Sampling`: Generate complex prior samples by passing standard normal base vectors through a cascade of alternating coupling layers conditioned on deterministic recurrent state $h_t$.

## Consequences
### Positive
- Captures multi-modal post-contact dynamics without mode collapse.
- Provides exact likelihood evaluation for variational lower bounds.
- Allows rapid parallel sampling during Monte Carlo trajectory rollouts.

### Negative
- Increases forward and backward pass latency compared to diagonal Gaussian heads.
- Requires careful numerical clipping on exponentiated scale parameters to prevent gradient explosions.
