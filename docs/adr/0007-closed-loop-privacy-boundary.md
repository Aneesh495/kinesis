# ADR-0007: Closed-Loop Execution Boundary and Zero Privileged Leakage

## Status
Accepted

## Context
In physics simulator benchmarks, it is easy to accidentally leak internal engine coordinates (such as generalized coordinates $q_{pos}$, velocity vectors $q_{vel}$, contact forces, or simulation bodies) directly into agent observation dictionaries. Moreover, naive planning implementations sometimes "peek" into the future by cloning the simulator state and rolling it forward. This destroys the premise of partially observed learned world models.

## Decision
We enforce a hard information boundary in `kinesis.closed_loop.loop`:
1. `Privacy Audit`: The runner inspects observation dictionaries every timestep. If forbidden keys (`qpos`, `qvel`, `qacc`, `physics`, `internal_state`) are present, a `PrivilegedInformationLeakageError` is raised immediately.
2. `Zero Simulator Peeking`: Planners and policies have zero access to the simulator instance, simulator pointers, or `step()` methods. Latent trajectory rollouts execute purely inside PyTorch tensor representations in the learned RSSM.
3. `Audit Trail`: Every executed episode logs its total simulator substeps and verifies that the simulator stepped exactly once per environment interaction step.

## Consequences
### Positive
- Guarantees scientific integrity: all decisions derive solely from partial observations and learned latent dynamics.
- Prevents simulator cloning tricks and privileged oracle shortcuts.
- Ensures compatibility with real-world robotics deployment where full state is unobservable.

### Negative
- Requires rigorous assertion checks on observation dictionaries during step execution.
