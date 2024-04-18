# ADR-0009: Symplectic Hamiltonian Neural Dynamics for Energy Preservation

## Status
Accepted

## Context
Standard feedforward or recurrent neural network dynamics models lack geometric inductive biases. When simulating continuous physical robotics systems over extended prediction horizons, unconstrained models often suffer from spurious energy gain or artificial numerical damping. This causes physical unfaithfulness and pathological plans in trajectory optimizers.

## Decision
We implement Symplectic Hamiltonian Neural Networks in `kinesis.models.symplectic`:
1. `Canonical Coordinates`: Partition latent state representation into generalized positions $q$ and conjugate momenta $p$.
2. `Learned Scalar Hamiltonian`: Parameterize total mechanical energy via neural potential and kinetic energy networks $H_\theta(q, p) = T_\theta(p) + V_\theta(q)$.
3. `Symplectic Leapfrog Integrator`: Advance state using exact volume-preserving symplectic leapfrog integration:
   $$p_{t + 1/2} = p_t - \frac{\Delta t}{2} \frac{\partial V}{\partial q}(q_t)$$
   $$q_{t + 1} = q_t + \Delta t \frac{\partial T}{\partial p}(p_{t + 1/2})$$
   $$p_{t + 1} = p_{t + 1/2} - \frac{\Delta t}{2} \frac{\partial V}{\partial q}(q_{t + 1})$$
4. `Control Coupling`: Couple external actuator control inputs through port-Hamiltonian input matrices $g(q) u$.

## Consequences
### Positive
- Enforces Liouville phase-space volume conservation by construction.
- Eliminates unphysical energy drift and divergence during multi-step latent trajectory rollouts.
- Provides physically grounded energy coordinates for safety and passivity verification.

### Negative
- Requires even latent dimensionality to accommodate paired canonical coordinate coordinates $(q, p)$.
- Increases computational graph depth due to second-order autodiff gradients during leapfrog updates.
