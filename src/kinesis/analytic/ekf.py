"""Discrete-time Extended Kalman Filter (EKF) with Joseph-form covariance updates."""

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class EKFState:
    """State estimate and covariance for Extended Kalman Filter."""
    mean: np.ndarray
    covariance: np.ndarray
    step: int = 0


class ExtendedKalmanFilter:
    """Discrete-time Extended Kalman Filter for nonlinear dynamical systems."""

    def __init__(
        self,
        f: Callable[[np.ndarray, np.ndarray], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        process_noise_cov: np.ndarray,
        measurement_noise_cov: np.ndarray,
        f_jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        h_jacobian: Callable[[np.ndarray], np.ndarray] | None = None,
        eps: float = 1e-6,
    ) -> None:
        self.f = f
        self.h = h
        self.W = np.asarray(process_noise_cov, dtype=np.float64)
        self.V = np.asarray(measurement_noise_cov, dtype=np.float64)
        self.f_jacobian = f_jacobian or self._approximate_f_jacobian
        self.h_jacobian = h_jacobian or self._approximate_h_jacobian
        self.eps = eps
        self.state_dim = self.W.shape[0]
        self.obs_dim = self.V.shape[0]

    def _approximate_f_jacobian(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute numerical Jacobian of state transition function via central differences."""
        n = len(x)
        jac = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            dx = np.zeros(n, dtype=np.float64)
            dx[i] = self.eps
            f_plus = self.f(x + dx, u)
            f_minus = self.f(x - dx, u)
            jac[:, i] = (f_plus - f_minus) / (2.0 * self.eps)
        return jac

    def _approximate_h_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Compute numerical Jacobian of observation function via central differences."""
        n = len(x)
        m = self.obs_dim
        jac = np.zeros((m, n), dtype=np.float64)
        for i in range(n):
            dx = np.zeros(n, dtype=np.float64)
            dx[i] = self.eps
            h_plus = self.h(x + dx)
            h_minus = self.h(x - dx)
            jac[:, i] = (h_plus - h_minus) / (2.0 * self.eps)
        return jac

    def predict(self, state: EKFState, u: np.ndarray) -> EKFState:
        """Execute state prediction step."""
        x_pred = self.f(state.mean, u)
        F = self.f_jacobian(state.mean, u)
        P_pred = F @ state.covariance @ F.T + self.W
        # Symmetrize covariance
        P_pred = 0.5 * (P_pred + P_pred.T)
        return EKFState(mean=x_pred, covariance=P_pred, step=state.step)

    def update(self, state_pred: EKFState, y: np.ndarray) -> tuple[EKFState, np.ndarray]:
        """Execute observation update step using Joseph-form covariance update."""
        y_pred = self.h(state_pred.mean)
        innovation = y - y_pred
        H = self.h_jacobian(state_pred.mean)
        S = H @ state_pred.covariance @ H.T + self.V
        S = 0.5 * (S + S.T)

        K = state_pred.covariance @ H.T @ np.linalg.inv(S)
        x_updated = state_pred.mean + K @ innovation

        # Joseph form: P = (I - KH) P (I - KH)^T + K V K^T guarantees positive semi-definiteness
        I = np.eye(self.state_dim, dtype=np.float64)
        IKH = I - K @ H
        P_updated = IKH @ state_pred.covariance @ IKH.T + K @ self.V @ K.T
        P_updated = 0.5 * (P_updated + P_updated.T)

        return EKFState(mean=x_updated, covariance=P_updated, step=state_pred.step + 1), innovation

    def step(self, state: EKFState, u: np.ndarray, y: np.ndarray) -> tuple[EKFState, np.ndarray]:
        """Execute full predict-and-update filter cycle."""
        pred = self.predict(state, u)
        return self.update(pred, y)
