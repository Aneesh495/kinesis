"""Discrete-time Unscented Kalman Filter (UKF) using scaled unscented transform."""

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class UKFState:
    """State estimate and covariance for Unscented Kalman Filter."""
    mean: np.ndarray
    covariance: np.ndarray
    step: int = 0


class UnscentedKalmanFilter:
    """Unscented Kalman Filter with scaled unscented transform."""

    def __init__(
        self,
        f: Callable[[np.ndarray, np.ndarray], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        process_noise_cov: np.ndarray,
        measurement_noise_cov: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.f = f
        self.h = h
        self.W = np.asarray(process_noise_cov, dtype=np.float64)
        self.V = np.asarray(measurement_noise_cov, dtype=np.float64)
        self.n = self.W.shape[0]
        self.m = self.V.shape[0]

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Scaling parameter lambda
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lam)

        # Compute weights
        num_sigmas = 2 * self.n + 1
        self.wm = np.zeros(num_sigmas, dtype=np.float64)
        self.wc = np.zeros(num_sigmas, dtype=np.float64)

        self.wm[0] = self.lam / (self.n + self.lam)
        self.wc[0] = self.wm[0] + (1.0 - self.alpha**2 + self.beta)

        for i in range(1, num_sigmas):
            self.wm[i] = 1.0 / (2.0 * (self.n + self.lam))
            self.wc[i] = self.wm[i]

    def _generate_sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Generate 2n + 1 symmetric sigma points from mean and covariance."""
        sigmas = np.zeros((2 * self.n + 1, self.n), dtype=np.float64)
        sigmas[0] = mean

        # Cholesky factor of scaled covariance
        try:
            L = np.linalg.cholesky((self.n + self.lam) * cov)
        except np.linalg.LinAlgError:
            # Fallback to SVD if covariance slightly lost positive-definiteness
            u, s, vt = np.linalg.svd(cov)
            L = u @ np.diag(np.sqrt(np.maximum(s, 1e-12) * (self.n + self.lam)))

        for i in range(self.n):
            sigmas[1 + i] = mean + L[:, i]
            sigmas[1 + self.n + i] = mean - L[:, i]

        return sigmas

    def predict(self, state: UKFState, u: np.ndarray) -> tuple[UKFState, np.ndarray]:
        """Propagate sigma points through nonlinear dynamics f and compute prior statistics."""
        sigmas = self._generate_sigma_points(state.mean, state.covariance)
        sigmas_f = np.zeros_like(sigmas)

        for i in range(len(sigmas)):
            sigmas_f[i] = self.f(sigmas[i], u)

        # Predicted mean
        x_pred = np.sum(self.wm[:, None] * sigmas_f, axis=0)

        # Predicted covariance
        diff = sigmas_f - x_pred[None, :]
        P_pred = diff.T @ (self.wc[:, None] * diff) + self.W
        P_pred = 0.5 * (P_pred + P_pred.T)

        return UKFState(mean=x_pred, covariance=P_pred, step=state.step), sigmas_f

    def update(self, state_pred: UKFState, sigmas_f: np.ndarray, y: np.ndarray) -> tuple[UKFState, np.ndarray]:
        """Transform propagated sigmas through observation function h and update state."""
        # Measurement sigma points
        sigmas_h = np.zeros((len(sigmas_f), self.m), dtype=np.float64)
        for i in range(len(sigmas_f)):
            sigmas_h[i] = self.h(sigmas_f[i])

        y_pred = np.sum(self.wm[:, None] * sigmas_h, axis=0)
        y_diff = sigmas_h - y_pred[None, :]
        x_diff = sigmas_f - state_pred.mean[None, :]

        # Innovation covariance
        S = y_diff.T @ (self.wc[:, None] * y_diff) + self.V
        S = 0.5 * (S + S.T)

        # Cross covariance
        P_xy = x_diff.T @ (self.wc[:, None] * y_diff)

        # Kalman gain
        K = P_xy @ np.linalg.inv(S)
        innovation = y - y_pred
        x_updated = state_pred.mean + K @ innovation
        P_updated = state_pred.covariance - K @ S @ K.T
        P_updated = 0.5 * (P_updated + P_updated.T)

        return UKFState(mean=x_updated, covariance=P_updated, step=state_pred.step + 1), innovation

    def step(self, state: UKFState, u: np.ndarray, y: np.ndarray) -> tuple[UKFState, np.ndarray]:
        """Execute complete UKF prediction and update cycle."""
        pred, sigmas_f = self.predict(state, u)
        return self.update(pred, sigmas_f, y)
