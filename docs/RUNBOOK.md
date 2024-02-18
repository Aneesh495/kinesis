# Kinesis Operational Runbook

This runbook guides researchers and operators through bootstrapping the environment, executing data collection, training world models, evaluating distribution shift robustness, running benchmarks, and generating acceptance evidence.

---

## 1. Environment Bootstrap and Diagnostics

Ensure Python 3.12 and `uv` package manager are available on your system.

```bash
# 1. Bootstrap virtual environment and dependencies
make bootstrap

# 2. Run system doctor to verify CPU/GPU/MPS, BLAS, and MuJoCo toolchain
make doctor
```

Expected Output:
- Platform OS, Architecture, and Python version printed.
- MuJoCo rendering and physics stepping verified.
- Torch hardware acceleration (CUDA or Apple Silicon MPS) detected.

---

## 2. Running Analytic Control Verification

Test the discrete-time linear Gaussian control fixtures, Riccati solver, and Kalman filter:

```bash
make test-control
```

Expected Output:
- 33 test items passing across controllability, observability, Riccati recursion, and closed-loop LQG regulation.

---

## 3. Data Collection and Replay Buffer Management

Collect demonstration and exploration trajectories into an immutable replay buffer:

```bash
# Collect 10 episodes on cartpole swingup
make collect TASK=cartpole_swingup SEED=42

# Verify replay buffer integrity and checkpoint recovery
make test-replay
```

---

## 4. World Model Training

Train the recurrent state-space model with balanced KL divergence:

```bash
make train TASK=cartpole_swingup SEED=42 RUN_ID=cartpole_rssm_v1
```

---

## 5. Evaluation and Distribution Shift Study

Evaluate trained policies under nominal dynamics and all 6 physical shift profiles:

```bash
# Run shift study across nominal, attenuation, delay, friction, mass, and sensor noise
make shift-study TASK=cartpole_swingup SEED=42
```

---

## 6. Throughput Benchmarks and Profiling

Run latency profiling across single-step transitions and multi-step rollouts:

```bash
make bench
```

---

## 7. Acceptance Evidence Generation and Independent Verification

Generate a complete cryptographic acceptance bundle and audit it independently:

```bash
# 1. Generate acceptance bundle with raw trajectories, checkpoints, and SHA-256 manifest
make acceptance RUN_ID=acceptance_run_v1

# 2. Independently verify bundle integrity, returns, and privacy boundaries
make verify EVIDENCE=evidence/acceptance_run_v1
```

Expected Output:
- `[VERIFIED] Evidence bundle evidence/acceptance_run_v1 passed independent audit successfully!`
- Zero privileged state leakage detected.
