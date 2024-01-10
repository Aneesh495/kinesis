"""Discrete-time Kalman filter for optimal linear state estimation under partial observability."""

from dataclasses import dataclass
import numpy as np

from kinesis.analytic.system import LinearGaussianSystem


@dataclass
class KalmanState:
    """Belief state: Gaussian mean and covariance."""
    mean: np.ndarray
    covariance: np.ndarray

    def copy(self) -> "KalmanState":
        return KalmanState(mean=self.mean.copy(), covariance=self.covariance.copy())


class KalmanFilter:
    """Discrete-time Kalman filter:

    Time update (Predict):
      x_{t|t-1} = A x_{t-1|t-1} + B u_{t-1}
      Sigma_{t|t-1} = A Sigma_{t-1|t-1} A^T + W

    Measurement update (Correct):
      y_tilde = y_t - C x_{t|t-1}
      S_t = C Sigma_{t|t-1} C^T + V
      L_t = Sigma_{t|t-1} C^T S_t^{-1}
      x_{t|t} = x_{t|t-1} + L_t y_tilde
      Sigma_{t|t} = (I - L_t C) Sigma_{t|t-1}
    """

    def __init__(self, sys: LinearGaussianSystem, init_mean: np.ndarray, init_cov: np.ndarray) -> None:
        self.sys = sys.augment_for_delay()
        self.state_dim = self.sys.state_dim
        self.obs_dim = self.sys.obs_dim

        if init_mean.shape != (self.state_dim,):
            raise ValueError(f"init_mean shape {init_mean.shape} != ({self.state_dim},)")
        if init_cov.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"init_cov shape {init_cov.shape} != ({self.state_dim}, {self.state_dim})")

        self.state = KalmanState(
            mean=init_mean.astype(np.float64).copy(),
            covariance=0.5 * (init_cov + init_cov.T).astype(np.float64),
        )

        # Store intermediate prior state for diagnostic access
        self.prior_state = self.state.copy()
        self.last_innovation = np.zeros(self.obs_dim, dtype=np.float64)
        self.last_innovation_cov = np.eye(self.obs_dim, dtype=np.float64)

    def predict(self, u: np.ndarray) -> KalmanState:
        """Propagate state and covariance forward using dynamics and control input."""
        u_arr = np.asarray(u, dtype=np.float64)
        if u_arr.ndim == 1:
            u_col = u_arr.reshape(-1, 1)
        else:
            u_col = u_arr

        A, B, W = self.sys.A, self.sys.B, self.sys.W

        mean_pred = (A @ self.state.mean.reshape(-1, 1) + B @ u_col).ravel()
        cov_pred = A @ self.state.covariance @ A.T + W
        cov_pred = 0.5 * (cov_pred + cov_pred.T)

        self.prior_state = KalmanState(mean=mean_pred, covariance=cov_pred)
        self.state = self.prior_state.copy()
        return self.state

    def update(self, y: np.ndarray) -> KalmanState:
        """Incorporate new noisy observation y_t to update posterior belief."""
        y_arr = np.asarray(y, dtype=np.float64)
        C, V = self.sys.C, self.sys.V

        # Innovation: y - C x_prior
        y_pred = (C @ self.prior_state.mean.reshape(-1, 1)).ravel()
        innovation = y_arr - y_pred
        innovation_cov = C @ self.prior_state.covariance @ C.T + V
        innovation_cov = 0.5 * (innovation_cov + innovation_cov.T)

        # Kalman gain L_t = Sigma C^T S^{-1}
        gain = np.linalg.solve(innovation_cov, C @ self.prior_state.covariance).T

        # Posterior update
        mean_post = self.prior_state.mean + (gain @ innovation.reshape(-1, 1)).ravel()
        I = np.eye(self.state_dim, dtype=np.float64)
        cov_post = (I - gain @ C) @ self.prior_state.covariance
        cov_post = 0.5 * (cov_post + cov_post.T)

        self.last_innovation = innovation
        self.last_innovation_cov = innovation_cov
        self.state = KalmanState(mean=mean_post, covariance=cov_post)
        return self.state

    def log_likelihood(self, y: np.ndarray) -> float:
        """Compute Gaussian log-likelihood of observation y under current innovation distribution."""
        dim = self.obs_dim
        sign, logdet = np.linalg.slogdet(self.last_innovation_cov)
        if sign <= 0:
            return -np.inf
        inv_cov = np.linalg.inv(self.last_innovation_cov)
        quad = float(self.last_innovation.T @ inv_cov @ self.last_innovation)
        ll = -0.5 * (dim * np.log(2 * np.pi) + logdet + quad)
        return float(ll)
