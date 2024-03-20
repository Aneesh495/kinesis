"""Recursive Least Squares (RLS) online parameter estimator with exponential forgetting."""

from dataclasses import dataclass
import numpy as np


@dataclass
class RLSState:
    """State of Recursive Least Squares estimator."""
    theta: np.ndarray  # Estimated parameter vector [p]
    covariance: np.ndarray  # Parameter covariance P [p, p]
    step: int = 0


@dataclass
class RLSTelemetry:
    """Step telemetry from RLS estimator."""
    prior_error: float
    posterior_error: float
    gain_norm: float
    trace_covariance: float


class RecursiveLeastSquares:
    """Online parameter identifier with exponential forgetting factor and covariance reset."""

    def __init__(
        self,
        num_parameters: int,
        forgetting_factor: float = 0.98,
        initial_covariance_scale: float = 100.0,
        covariance_trace_max: float = 1e5,
        covariance_trace_min: float = 1e-4,
    ) -> None:
        self.p = num_parameters
        self.lam = float(forgetting_factor)
        assert 0.0 < self.lam <= 1.0, f"Forgetting factor must be in (0, 1], got {self.lam}"

        self.p_scale = initial_covariance_scale
        self.trace_max = covariance_trace_max
        self.trace_min = covariance_trace_min

    def initial_state(self, initial_theta: np.ndarray | None = None) -> RLSState:
        """Create initial state with isotropic covariance."""
        if initial_theta is not None:
            theta = np.asarray(initial_theta, dtype=np.float64).copy()
        else:
            theta = np.zeros(self.p, dtype=np.float64)

        cov = np.eye(self.p, dtype=np.float64) * self.p_scale
        return RLSState(theta=theta, covariance=cov, step=0)

    def update(self, state: RLSState, regressor: np.ndarray, measurement: float) -> tuple[RLSState, RLSTelemetry]:
        """Execute one RLS parameter estimation update step."""
        phi = np.asarray(regressor, dtype=np.float64)
        y = float(measurement)
        assert phi.shape == (self.p,), f"Regressor shape {phi.shape} must match param dim {self.p}"

        # 1. Prior error
        y_pred_prior = float(np.dot(phi, state.theta))
        prior_err = y - y_pred_prior

        # 2. Gain vector: K = P * phi / (lambda + phi^T * P * phi)
        P_phi = state.covariance @ phi
        denom = self.lam + float(phi.T @ P_phi)
        K = P_phi / denom

        # 3. Parameter update
        theta_next = state.theta + K * prior_err

        # 4. Covariance update: P_next = (1 / lambda) * (P - K * phi^T * P)
        P_next = (1.0 / self.lam) * (state.covariance - np.outer(K, phi.T @ state.covariance))
        P_next = 0.5 * (P_next + P_next.T)  # Ensure symmetry

        # Covariance trace bounding to prevent windup or collapse
        tr = float(np.trace(P_next))
        if tr > self.trace_max:
            P_next = P_next * (self.trace_max / tr)
        elif tr < self.trace_min:
            P_next = P_next + np.eye(self.p) * 1e-3

        # 5. Posterior error
        y_pred_post = float(np.dot(phi, theta_next))
        post_err = y - y_pred_post

        telemetry = RLSTelemetry(
            prior_error=prior_err,
            posterior_error=post_err,
            gain_norm=float(np.linalg.norm(K)),
            trace_covariance=float(np.trace(P_next)),
        )

        return RLSState(theta=theta_next, covariance=P_next, step=state.step + 1), telemetry
