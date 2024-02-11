# Kinesis Experiment Plan and Evaluation Protocol

This document specifies the scientific hypotheses, benchmark matrices, statistical evaluation criteria, and ablation studies designed for the Kinesis world-model and epistemic uncertainty-aware control laboratory.

## 1. Scientific Hypotheses

### Hypothesis H1: Epistemic Uncertainty Mitigates OOD Hallucination Exploitation
*Statement*: Latent trajectory planners (CEM, MPPI) optimizing objectives with ensemble disagreement penalties will exhibit significantly lower decision regret and higher policy returns under physical distribution shifts than unpenalized planners.
*Falsification Criterion*: If penalized planners achieve equal or lower mean returns across the 6 frozen shift profiles compared to unpenalized planners ($p > 0.05$ under a two-tailed Wilcoxon signed-rank test), H1 is rejected.

### Hypothesis H2: Balanced KL Prevents Posterior Collapse Without Information Loss
*Statement*: Weighting KL updates with $\alpha = 0.8$ on the prior and $0.2$ on the posterior, coupled with a free-bits threshold ($\beta = 1.0$), preserves multi-step latent predictive accuracy better than standard unweighted KL ($\alpha = 0.5$).
*Falsification Criterion*: If unweighted KL yields equal or lower multi-step observation reconstruction MSE over 15-step imagined horizons, H2 is rejected.

### Hypothesis H3: Analytic LQG Grounds Neural Latent World Models
*Statement*: On canonical linear Gaussian dynamics, a neural RSSM combined with CEM planning approaches the theoretical LQG return upper-bound within a bounded gap ($\le 10\%$).

## 2. Evaluation Matrix

The evaluation suite spans 3 continuous control tasks across 7 distribution conditions with 5 independent evaluation seeds each:

| Task | Action Dim | Observation Dim | Shift Conditions | Evaluation Seeds |
| :--- | :--- | :--- | :--- | :--- |
| `cartpole_swingup` | 1 | 5 | Nominal + 6 Shifts | 42, 43, 44, 45, 46 |
| `walker_walk` | 6 | 24 | Nominal + 6 Shifts | 42, 43, 44, 45, 46 |
| `finger_spin` | 2 | 9 | Nominal + 6 Shifts | 42, 43, 44, 45, 46 |

### Shift Conditions:
1. `nominal`: Baseline DM Control dynamics and unperturbed observation channel.
2. `actuator_attenuation`: $50\%$ reduction in actuator gear scale.
3. `actuator_delay`: 2-step FIFO action delay queue.
4. `friction_perturbation`: $60\%$ reduction in contact friction.
5. `mass_perturbation`: $80\%$ increase in distal link mass.
6. `sensor_degradation`: Gaussian observation noise ($\sigma = 0.15$) and $10\%$ random sensor dropout.
7. `combined_stress`: Simultaneous application of all 5 physical and sensory perturbations.

## 3. Evaluation Metrics

For each episode $e$ of length $T$:
- **Cumulative Episode Return**: $R = \sum_{t=0}^{T-1} r_t$
- **Total Constraint Cost**: $C = \sum_{t=0}^{T-1} c_t$
- **1-Step Prediction MSE**: $\text{MSE}_1 = \frac{1}{T} \sum_{t=1}^T \|\hat{x}_t - x_t\|_2^2$
- **15-Step Rollout Prediction MSE**: $\text{MSE}_{15} = \frac{1}{T-15} \sum_{t=1}^{T-15} \|\hat{x}_{t+15} - x_{t+15}\|_2^2$
- **Epistemic Disagreement**: Mean ensemble variance across imagined rollout steps.
- **Counterfactual Decision Regret**: $\Delta R = R_{nominal} - R_{shifted}$ under paired random seeds.

## 4. Ablation Study Protocol

The ablation runner evaluates 4 discrete ablations:
1. `full_model`: Standard RSSM with ensemble disagreement penalty and balanced KL.
2. `no_uncertainty_penalty`: Latent planner with $\lambda = 0.0$ (no epistemic penalty).
3. `unweighted_kl`: RSSM trained with standard symmetric KL ($\alpha = 0.5$).
4. `random_policy`: Untrained random uniform action selection baseline.
