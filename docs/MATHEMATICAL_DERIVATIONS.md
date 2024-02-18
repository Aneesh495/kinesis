# Kinesis Mathematical Derivations

This document provides rigorous mathematical formulations and derivations for the analytical fixtures, learned world models, uncertainty estimators, and planning algorithms implemented across Kinesis.

---

## 1. Discrete Linear Quadratic Gaussian (LQG) Control

### 1.1 State Space Formulation
Consider a discrete-time linear time-invariant system with Gaussian perturbations:
$$x_{t+1} = A x_t + B u_t + w_t, \quad w_t \sim \mathcal{N}(0, W)$$
$$y_t = C x_t + v_t, \quad v_t \sim \mathcal{N}(0, V)$$
where $x_t \in \mathbb{R}^n$, $u_t \in \mathbb{R}^m$, and $y_t \in \mathbb{R}^p$.

### 1.2 Discrete Algebraic Riccati Equation (DARE)
For infinite-horizon regulation with state cost matrix $Q \succeq 0$ and control cost matrix $R \succ 0$:
$$J = \sum_{t=0}^{\infty} \left( x_t^\top Q x_t + u_t^\top R u_t \right)$$
The optimal state-feedback gain $K$ is given by:
$$K = (R + B^\top P B)^{-1} B^\top P A$$
where $P \succ 0$ is the unique stabilizing solution to the discrete algebraic Riccati equation:
$$P = A^\top P A - A^\top P B (R + B^\top P B)^{-1} B^\top P A + Q$$

### 1.3 Steady-State Kalman Filter
The optimal state estimator $\hat{x}_{t|t}$ satisfies:
$$\hat{x}_{t|t-1} = A \hat{x}_{t-1|t-1} + B u_{t-1}$$
$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + L (y_t - C \hat{x}_{t|t-1})$$
where the steady-state Kalman gain $L$ is:
$$L = \Sigma_\infty C^\top (C \Sigma_\infty C^\top + V)^{-1}$$
and the prior covariance $\Sigma_\infty$ solves the estimation Riccati equation:
$$\Sigma_\infty = A \left[ \Sigma_\infty - \Sigma_\infty C^\top (C \Sigma_\infty C^\top + V)^{-1} C \Sigma_\infty \right] A^\top + W$$

By the Separation Principle of stochastic control, the optimal control law is $u_t = -K \hat{x}_{t|t}$.

---

## 2. Recurrent State-Space World Model (RSSM)

### 2.1 Latent Transition Factorization
The world model factorizes the sequence likelihood into deterministic state $h_t$ and stochastic state $z_t$:
$$h_t = f_{det}(h_{t-1}, z_{t-1}, a_{t-1})$$
$$p(z_t \mid h_t) = \mathcal{N}(\mu_{prior}(h_t), \text{diag}(\sigma_{prior}^2(h_t)))$$
$$q(z_t \mid h_t, x_t) = \mathcal{N}(\mu_{post}(h_t, x_t), \text{diag}(\sigma_{post}^2(h_t, x_t)))$$

### 2.2 Variational Evidence Lower Bound (ELBO)
The model parameters $\theta$ (decoder, prior, dynamics) and variational parameters $\phi$ (encoder, posterior) maximize the multi-step ELBO:
$$\mathcal{L}(\theta, \phi) = \sum_{t=1}^T \mathbb{E}_{q(z_t \mid h_t, x_t)} \left[ \ln p_\theta(x_t \mid h_t, z_t) + \ln p_\theta(r_t \mid h_t, z_t) \right] - \sum_{t=1}^T D_{KL}\left( q_\phi(z_t \mid h_t, x_t) \parallel p_\theta(z_t \mid h_t) \right)$$

### 2.3 Balanced KL Divergence Loss
To train the prior toward the posterior without collapsing the posterior representation:
$$\mathcal{L}_{KL} = \alpha D_{KL}\left( \text{stop\_grad}(q_\phi) \parallel p_\theta \right) + (1 - \alpha) D_{KL}\left( q_\phi \parallel \text{stop\_grad}(p_\theta) \right)$$
with a free-bits floor $\beta$:
$$\mathcal{L}_{KL, balanced} = \max\left( \beta, \mathcal{L}_{KL} \right)$$
where $\alpha = 0.8$ prioritizes training the predictive prior dynamics.

---

## 3. Epistemic Disagreement Formulation

Given an ensemble of $M$ independently initialized world models $\{f^{(1)}, f^{(2)}, \dots, f^{(M)}\}$, the epistemic uncertainty for a latent state-action pair $(s, a)$ is defined as the empirical variance across committee outputs:
$$\mu(s, a) = \frac{1}{M} \sum_{m=1}^M f^{(m)}(s, a)$$
$$\sigma_{epistemic}^2(s, a) = \frac{1}{M - 1} \sum_{m=1}^M \left( f^{(m)}(s, a) - \mu(s, a) \right)^2$$
For multidimensional predictions, the total disagreement penalty is the trace of the empirical covariance matrix.

---

## 4. Latent Action Trajectory Optimization

### 4.1 Cross-Entropy Method (CEM)
CEM models candidate action trajectories $U = [a_0, a_1, \dots, a_{H-1}] \in \mathbb{R}^{H \times d_a}$ as a factorized Gaussian:
$$q(U) = \prod_{t=0}^{H-1} \mathcal{N}(\mu_t, \text{diag}(\sigma_t^2))$$
At iteration $i$:
1. Sample $K$ candidate action trajectories: $U^{(k)} \sim q^{(i)}(U)$ for $k \in \{1, \dots, K\}$.
2. Evaluate rollouts in learned RSSM latent space:
   $$J(U^{(k)}) = \sum_{t=0}^{H-1} \left( \hat{r}_t^{(k)} - \lambda \cdot \sigma_{epistemic, t}^{(k)} \right)$$
3. Sort and select the $E$ elite candidates with highest return $J(U^{(k)})$.
4. Update distribution parameters with smoothing factor $\alpha_{cem} \in [0, 1]$:
   $$\mu_t^{(i+1)} = \alpha_{cem} \mu_t^{(i)} + (1 - \alpha_{cem}) \left( \frac{1}{E} \sum_{e=1}^E U_{t, elite}^{(e)} \right)$$
   $$\sigma_t^{2, (i+1)} = \alpha_{cem} \sigma_t^{2, (i)} + (1 - \alpha_{cem}) \left( \frac{1}{E} \sum_{e=1}^E (U_{t, elite}^{(e)} - \mu_t^{(i+1)})^2 \right)$$

### 4.2 Model Predictive Path Integral (MPPI)
Given a nominal action sequence $u = [u_0, \dots, u_{H-1}]$, MPPI samples perturbations $\epsilon^{(k)} \sim \mathcal{N}(0, \Sigma)$. The cost-to-go of trajectory $k$ is $S(U^{(k)})$.
Candidate weights follow Boltzmann importance distributions:
$$w_k = \frac{\exp\left( - \frac{1}{\lambda} S(U^{(k)}) \right)}{\sum_{j=1}^K \exp\left( - \frac{1}{\lambda} S(U^{(j)}) \right)}$$
The updated nominal control sequence is:
$$u_t^* = u_t + \sum_{k=1}^K w_k \epsilon_{t}^{(k)}$$
The planner executes $a_0 = u_0^*$ and shifts the sequence forward for warm-starting the subsequent step.
