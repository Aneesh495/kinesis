# ADR-0015: Warm-Start Memory Caching for Receding Horizon Optimization

## Status
Accepted

## Context
Model Predictive Control (MPC) and latent planners (CEM, MPPI, CMA-ES) must optimize multi-step action trajectories within tight real-time control deadlines. In classic implementations, optimization begins from white noise at every control step, discarding valuable optimization work completed at the previous step. This leads to chattering action sequences and slow convergence.

## Decision
We implement temporal shift and rollout memory caching in `kinesis.planning`:
1. `Temporal Receding Horizon Shift`: Store the optimal action sequence $u_{0:H-1}^*$ at step $t$. At step $t+1$, advance the sequence forward by 1 step: $[u_1^*, \dots, u_{H-1}^*, u_{\text{term}}]$, where $u_{\text{term}}$ is padded via repetition or nominal action.
2. `Consistency Verification`: Compute divergence $\|u_0^* - u_{\text{executed}}\|_2$. If external disturbance or safety overrides significantly alter executed actions beyond $\delta_{\max}$, immediately invalidate the cache.
3. `Latent Trajectory Rollout Cache`: Cache evaluated trajectory rollouts using quantized BLAKE2b hashes of initial states and action trajectories with LRU eviction to deduplicate identical forward simulations.

## Consequences
### Positive
- Accelerates latent planner convergence by 3x to 5x through warm-started distribution means.
- Smooths control inputs and reduces actuator chatter across consecutive receding horizons.
- Safely invalidates cached plans upon detecting unexpected disturbances.

### Negative
- Requires maintaining cache state across closed-loop control steps.
