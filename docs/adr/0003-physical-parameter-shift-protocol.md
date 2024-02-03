# ADR-0003: Physical Parameter Shift Protocol and Probe Verification

## Status
Accepted

## Context
A major claim in world-model research is robustness under distribution shifts. However, shifts in the literature are often poorly specified, varied across evaluation runs, or applied post-hoc without verifying that actual physical dynamics diverged from the nominal system.

## Decision
We define 6 frozen physical shift profiles in `kinesis.shifts` in addition to the nominal baseline:
1. `actuator_attenuation`: 50% scale reduction in actuator gear transmission.
2. `actuator_delay`: Discrete 2-step FIFO delay queue injected into the actuator signal.
3. `friction_perturbation`: 60% reduction in surface contact friction coefficient.
4. `mass_perturbation`: 80% increase in distal body masses.
5. `sensor_degradation`: High Gaussian sensor noise ($\sigma = 0.15$) and 10% random sensor dropout.
6. `combined_stress`: Simultaneous application of attenuation, delay, friction, mass, and sensor perturbations.

Every shift profile is audited with a dynamics probe test (`probe_dynamics_divergence`) that steps identical action sequences through nominal and shifted environments and confirms non-zero state divergence.

## Consequences
### Positive
- Strict reproducibility across research benchmarks with frozen perturbation coefficients.
- Quantitative verification that physical shifts actually alter simulation trajectories.
- Systematic stress-testing of epistemic uncertainty estimators and adaptive controllers.

### Negative
- Evaluating across all 7 configurations increases benchmark compute requirements by $7\times$.
