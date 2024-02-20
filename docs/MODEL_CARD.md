# Kinesis Model Card: Recurrent State-Space World Model (RSSM)

## Model Overview
- **Model Name**: Kinesis Stochastic Recurrent State-Space Model (RSSM)
- **Architecture Type**: Hybrid deterministic GRU recurrent dynamics with diagonal Gaussian stochastic latents.
- **Framework**: PyTorch 2.x
- **Intended Domain**: Continuous robotic locomotion and control under partial observability and physical parameter shifts.

## Model Architecture Details
- **Deterministic Recurrent Core**: Single-layer GRU cell ($h_t \in \mathbb{R}^{d_h}$, default $d_h = 256$ or $512$).
- **Stochastic Latent Path**: Diagonal multivariate Gaussian ($z_t \in \mathbb{R}^{d_z}$, default $d_z = 32$).
- **Observation Encoder**: Multi-Layer Perceptron (MLP) mapping partial observations $x_t \in \mathbb{R}^{d_x}$ to feature embeddings $e_t \in \mathbb{R}^{d_e}$.
- **Posterior Head**: MLP mapping concatenated $[h_t, e_t]$ to mean $\mu_q$ and standard deviation $\sigma_q$.
- **Prior Head**: MLP mapping $h_t$ to mean $\mu_p$ and standard deviation $\sigma_p$.
- **Decoders and Heads**:
  - Observation Reconstruction Head: Gaussian likelihood $\hat{x}_t \sim \mathcal{N}(\mu_{rec}, I)$.
  - Reward Prediction Head: Scalar MSE regression $\hat{r}_t$.
  - Continue Head: Binary Bernoulli classification $\hat{\gamma}_t$ indicating termination probability.
  - Constraint Cost Head: Scalar cost prediction $\hat{c}_t$.

## Training Procedure
- **Objective**: Multi-step Variational Evidence Lower Bound (ELBO) with balanced KL divergence ($\alpha = 0.8$) and free-bits threshold ($\beta = 1.0$).
- **Optimization**: AdamW optimizer, learning rate $3 \times 10^{-4}$, gradient norm clipping at $100.0$.
- **Sequence Length**: 50 transitions sampled across episode chunks.

## Evaluation and Validation
- **Causal Prefix Invariance**: Future sequence padding cannot alter past representations.
- **OOD Disagreement**: Ensemble committees exhibit monotonic increase in prediction variance under out-of-distribution physical shifts.
- **Zero Privileged State**: Strict assertion that no internal simulator generalized coordinates enter the model inputs.
