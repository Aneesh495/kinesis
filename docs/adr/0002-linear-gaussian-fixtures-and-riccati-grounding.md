# ADR-0002: Linear Gaussian Control Fixtures and Riccati Grounding

## Status
Accepted

## Context
Validating complex learned latent dynamics models directly on high-dimensional nonlinear contact tasks makes it difficult to isolate model learning failures from optimization bugs. In classical control theory, discrete-time Linear Quadratic Gaussian (LQG) control provides exact closed-form optimal solutions via discrete Algebraic Riccati Equations (DARE) and Kalman filters.

## Decision
We implement a fully analytical discrete-time linear Gaussian control fixture library in `kinesis.analytic`:
1. `LinearSystem`: Discrete-time state space model $x_{t+1} = A x_t + B u_t + w_t$ and $y_t = C x_t + v_t$.
2. `solve_dare_discrete`: Discrete Riccati equation solver using Schur vector decomposition and symplectic matrix pencil algorithms.
3. `SteadyStateKalmanFilter`: Optimal state estimator for process noise $W$ and measurement noise $V$.
4. `LQGController`: Optimal output feedback regulator combining Kalman state estimation and LQR gain feedback.

Unit tests verify that analytic Riccati solutions match within $10^{-6}$ Frobenius norm and that the certainty equivalence principle holds identically.

## Consequences
### Positive
- Provides a mathematically grounded baseline against which neural world models can be benchmarked on canonical linear dynamics.
- Eliminates guesswork during development of model-predictive planning algorithms.
- Serves as an unassailable positive ground-truth fixture for unit and property tests.

### Negative
- Requires maintaining analytic linear algebra fixtures alongside neural network code.
