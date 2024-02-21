# Kinesis Dataset Card: Continuous Control Demonstrations and Replay Data

## Dataset Overview
- **Dataset Name**: Kinesis Offline Replay and Exploration Traces
- **Domain**: Continuous partially observed robotic locomotion and manipulation benchmarks (`dm_control`).
- **Storage Format**: Individual JSON-serialized `Episode` files containing sequences of validated, immutable `Transition` records.
- **Integrity**: Every episode file is checksummed via SHA-256 and tracked in cryptographic manifests.

## Tasks and Schema
1. `cartpole_swingup`:
   - Observation fields: `position` ($\mathbb{R}^3$, cart position and pendulum angle cosine/sine), `velocity` ($\mathbb{R}^2$).
   - Action space: $\mathbb{R}^1$ bounded in $[-1, 1]$.
2. `walker_walk`:
   - Observation fields: `orientations` ($\mathbb{R}^{14}$), `height` ($\mathbb{R}^1$), `velocity` ($\mathbb{R}^9$).
   - Action space: $\mathbb{R}^6$ continuous joint torques bounded in $[-1, 1]$.
3. `finger_spin`:
   - Observation fields: `position` ($\mathbb{R}^4$), `velocity` ($\mathbb{R}^3$), `touch` ($\mathbb{R}^2$).
   - Action space: $\mathbb{R}^2$ continuous actuator torques bounded in $[-1, 1]$.

## Data Splits and Shift Variations
- **Nominal Split**: Standard dynamics with uncorrupted observations.
- **Shifted Splits**:
  - `actuator_attenuation`: $50\%$ actuator scale.
  - `actuator_delay`: 2-step FIFO action queue.
  - `friction_perturbation`: $60\%$ reduced friction.
  - `mass_perturbation`: $80\%$ increased link mass.
  - `sensor_degradation`: $\sigma = 0.15$ Gaussian noise and $10\%$ sensor dropout.
  - `combined_stress`: Simultaneous application of all 5 physical and sensory perturbations.

## Privacy and Anonymization
All recorded trajectories contain zero user data or privileged simulator pointers. All observations strictly conform to public partially observed sensor specifications.
