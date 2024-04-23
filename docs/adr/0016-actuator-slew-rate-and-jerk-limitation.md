# ADR-0016: Actuator Slew-Rate and Acceleration Jerk Limitation

## Status
Accepted

## Context
Aggressive model-based trajectory optimizers frequently generate discontinuous, high-frequency step commands between control cycles ($u_t \to u_{t+1}$). On physical continuous robots, instantaneous torque reversals cause gearbox backlash shocks, motor drive overcurrent trips, and unmodeled high-frequency structural resonance.

## Decision
We implement a discrete rate and jerk filter in `kinesis.safety.slew_rate`:
1. `Slew-Rate Velocity Limiting`: Restrict first-order rate of change:
   $$|\Delta u_t| = |u_t - u_{t-1}| \le \dot{u}_{\max} \Delta t$$
2. `Acceleration Jerk Limiting`: Restrict second-order derivative:
   $$|(\Delta u_t / \Delta t) - (\Delta u_{t-1} / \Delta t)| \le \ddot{u}_{\max} \Delta t$$
3. `Static Envelope Enclosure`: Ensure filtered commands remain strictly within declared `ActionSpec` physical bounds.
4. `Telemetry Accounting`: Track filter interventions and rate saturation events in runtime diagnostic logs.

## Consequences
### Positive
- Prevents actuator chatter and mechanical shock loads.
- Ensures physically realizable torque command trajectories.
- Improves world model predictive accuracy by keeping actions within smooth dynamical regimes.

### Negative
- Introduces small effective phase lag during emergency braking or rapid swingup phases.
