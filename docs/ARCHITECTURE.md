# Kinesis Architecture Specification

This document details the architectural design, component interactions, mathematical contracts, and runtime boundaries of the Kinesis world-model and control laboratory.

## 1. High-Level System Architecture

The Kinesis laboratory is structured into modular layers spanning environment interaction, replay persistence, world-model learning, uncertainty quantification, latent trajectory planning, closed-loop execution, and independent verification.

```mermaid
flowchart TD
    Env["MuJoCo / DM Control Suite"] --> ObsWrapper["Partial Observation Wrapper"]
    ObsWrapper --> Replay["Bounded Replay Buffer"]
    Replay --> WMTrainer["World Model Trainer"]
    WMTrainer --> RSSM["Recurrent State Space Model (RSSM)"]
    RSSM --> Ensemble["Uncertainty Ensemble Committee"]
    Ensemble --> Planner["Latent Planner (CEM / MPPI)"]
    Planner --> Policy["Planner Policy"]
    Policy --> ClosedLoop["Closed Loop Runner"]
    ClosedLoop --> Env
    ClosedLoop --> Evidence["Acceptance Evidence Builder"]
    Evidence --> Verifier["Independent Cryptographic Verifier"]
```

## 2. Environment Adapter and Information Boundary

The environment layer enforces strict partial observation. Simulator state (including generalized coordinates $q_{pos}$, velocities $q_{vel}$, actuators, and internal contact forces) is strictly confined to environment step transitions. Policy decisions receive only the wrapped partial observation dictionary.

```mermaid
sequenceDiagram
    participant Policy as Policy / Planner
    participant Loop as Closed Loop Runner
    participant Wrap as Observation Wrapper
    participant Sim as DM Control Simulator

    Policy->>Loop: action proposal
    Loop->>Loop: verify privacy (no privileged state)
    Loop->>Sim: step(action)
    Sim-->>Wrap: raw sensor observation (no privileged state)
    Wrap->>Wrap: add delay, noise, and dropout
    Wrap-->>Loop: partial observation dict
    Loop-->>Policy: partial observation dict
```

## 3. Recurrent State-Space Model (RSSM) Architecture

The world model factorizes transition dynamics into a deterministic recurrent path ($h_t$) and a stochastic latent path ($z_t$).

$$h_t = \text{GRU}(h_{t-1}, [z_{t-1}, a_{t-1}])$$
$$q(z_t \mid h_t, x_t) = \mathcal{N}(\mu_{post}(h_t, x_t), \Sigma_{post}(h_t, x_t))$$
$$p(z_t \mid h_t) = \mathcal{N}(\mu_{prior}(h_t), \Sigma_{prior}(h_t))$$

```mermaid
flowchart LR
    Zprev["z_{t-1}"] --> GRU["Deterministic GRU Cell"]
    Aprev["a_{t-1}"] --> GRU
    Hprev["h_{t-1}"] --> GRU
    GRU --> Hcurr["h_t"]
    Hcurr --> Prior["Prior Head: p(z_t | h_t)"]
    Hcurr --> Post["Posterior Head: q(z_t | h_t, e_t)"]
    Obs["Observation x_t"] --> Enc["Encoder e_t"]
    Enc --> Post
    Post --> Zcurr["Stochastic Latent z_t"]
    Hcurr --> Feat["Feature [h_t, z_t]"]
    Zcurr --> Feat
    Feat --> Dec["Observation Decoder"]
    Feat --> Rew["Reward Head"]
    Feat --> Cont["Continue Head"]
    Feat --> Cost["Cost Head"]
```

## 4. Balanced KL Divergence Loss and Regularization

To prevent posterior collapse while regularizing the latent representations, training balances gradient updates between the prior and posterior distributions with a free-bits margin.

```mermaid
flowchart TD
    PostDist["Posterior q(z_t | h_t, x_t)"] --> KL["Kullback-Leibler Divergence"]
    PriorDist["Prior p(z_t | h_t)"] --> KL
    KL --> Clip["Max(KL, Free Bits Margin)"]
    Clip --> Split["Balanced Scaling"]
    Split --> Alpha["Alpha * KL(stop_grad(q), p)"]
    Split --> Beta["(1 - Alpha) * KL(q, stop_grad(p))"]
    Alpha --> TotalLoss["Total World Model Objective"]
    Beta --> TotalLoss
```

## 5. Epistemic Uncertainty Quantification via Ensemble Disagreement

Epistemic uncertainty is estimated using an ensemble committee of world models trained with distinct parameter initializations and data mini-batch schedules. Variance across predictions signals epistemic novelty.

```mermaid
flowchart TD
    LatentFeat["Latent State Feature [h_t, z_t]"] --> Model1["World Model 1"]
    LatentFeat --> Model2["World Model 2"]
    LatentFeat --> Model3["World Model 3"]
    Model1 --> Pred1["Reward / Dynamics Pred 1"]
    Model2 --> Pred2["Reward / Dynamics Pred 2"]
    Model3 --> Pred3["Reward / Dynamics Pred 3"]
    Pred1 --> VarianceCalc["Empirical Ensemble Variance"]
    Pred2 --> VarianceCalc
    Pred3 --> VarianceCalc
    VarianceCalc --> Disagreement["Epistemic Disagreement Penalty"]
```

## 6. Cross-Entropy Method (CEM) Latent Planning

CEM optimizes action sequences strictly in the learned latent space without interacting with the real simulator.

```mermaid
flowchart TD
    Init["Initialize Action Gaussian N(mu_0, sigma_0)"] --> Sample["Sample K Action Sequences"]
    Sample --> Rollout["Imagined Latent Rollouts in RSSM"]
    Rollout --> Score["Score: Predicted Return - Beta * Uncertainty"]
    Score --> Elite["Select Top-E Elite Candidates"]
    Elite --> Refit["Refit Action Gaussian Distribution"]
    Refit --> Check{"Max Iterations Reached?"}
    Check -- No --> Sample
    Check -- Yes --> Exec["Execute First Action in Environment"]
```

## 7. Model Predictive Path Integral (MPPI) Control

MPPI samples stochastic perturbations around a nominal control trajectory, computing importance weights via exponential trajectory returns.

```mermaid
flowchart TD
    Nominal["Nominal Action Sequence u_{0:H}"] --> Perturb["Generate K Perturbations: u + epsilon"]
    Perturb --> LatentSim["Rollout Imagined Latent States"]
    LatentSim --> TrajCost["Compute Trajectory Costs S(tau_k)"]
    TrajCost --> Softmax["Softmax Weights: w_k = exp(-S_k / lambda)"]
    Softmax --> Update["Update Nominal: u* = sum(w_k * (u + eps_k))"]
    Update --> Step["Shift Action Window and Execute First Action"]
```

## 8. Closed-Loop Execution and Online Adaptation

The closed-loop runner synchronizes policy interaction with environment stepping, logging immutable transition tuples while monitoring for distribution shifts.

```mermaid
sequenceDiagram
    participant Runner as Closed Loop Runner
    participant Policy as Latent Policy
    participant Adapt as Adaptation Manager
    participant Env as Environment

    Runner->>Env: reset()
    Env-->>Runner: initial observation
    loop Step Loop
        Runner->>Policy: act(observation)
        Policy-->>Runner: action
        Runner->>Env: step(action)
        Env-->>Runner: transition (reward, done, next_obs)
        Runner->>Adapt: record_prediction_error(error)
        alt High Error Detected
            Adapt->>Policy: trigger online parameter adaptation
        end
    end
```

## 9. Shift Robustness and Counterfactual Evaluation

Evaluation exercises trained agents against nominal conditions and 6 frozen physical shift profiles across identical paired random seeds.

```mermaid
flowchart TD
    EvalSeed["Evaluation Random Seed S"] --> NomEnv["Nominal Environment"]
    EvalSeed --> AttenEnv["Actuator Attenuation Shift"]
    EvalSeed --> DelayEnv["Actuator Delay Shift"]
    EvalSeed --> FricEnv["Friction Reduction Shift"]
    EvalSeed --> MassEnv["Mass Increase Shift"]
    EvalSeed --> SensorEnv["Sensor Degradation Shift"]
    EvalSeed --> StressEnv["Combined Stress Shift"]
    NomEnv --> Metrics["Counterfactual Gap and Regret Analysis"]
    AttenEnv --> Metrics
    DelayEnv --> Metrics
    FricEnv --> Metrics
    MassEnv --> Metrics
    SensorEnv --> Metrics
    StressEnv --> Metrics
```

## 10. Acceptance Bundle Building and Independent Cryptographic Verification

Acceptance bundles record raw trajectory traces, trained model checkpoints, environment manifests, and JSON summaries. Every file is registered with SHA-256 digests in MANIFEST.json.

```mermaid
flowchart TD
    Build["make acceptance"] --> RawTraces["Raw Trajectory JSON Files"]
    Build --> Checkpoints["Model Checkpoints .pt"]
    Build --> EnvSpec["ENVIRONMENT.json"]
    Build --> Summary["ACCEPTANCE.json"]
    RawTraces --> ManifestGen["Generate MANIFEST.json (SHA-256)"]
    Checkpoints --> ManifestGen
    EnvSpec --> ManifestGen
    Summary --> ManifestGen
    ManifestGen --> Bundle["Complete Evidence Bundle"]
    Bundle --> Verifier["make verify"]
    Verifier --> HashCheck["Verify SHA-256 Digest of Every File"]
    Verifier --> ReturnCheck["Recompute Cumulative Episode Returns from Step Rewards"]
    Verifier --> PrivacyAudit["Verify Zero Privileged State in Observation Keys"]
    Verifier --> PassFail["Pass / Reject Audit Verdict"]
```
