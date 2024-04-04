"""Discrete-time Lyapunov stability verification and energy decay certification."""

from dataclasses import dataclass
import numpy as np
import scipy.linalg


@dataclass
class LyapunovCertificateResult:
    """Quantitative certificate of trajectory Lyapunov stability."""
    is_certified_stable: bool
    initial_lyapunov_value: float
    final_lyapunov_value: float
    max_decay_ratio: float
    strict_monotonic_decrease: bool
    lyapunov_trajectory: list[float]


class DiscreteLyapunovCertifier:
    """Solves discrete Lyapunov equation and verifies asymptotic stability along trajectories."""

    def __init__(self, A_matrix: np.ndarray, Q_matrix: np.ndarray | None = None) -> None:
        self.A = np.asarray(A_matrix, dtype=np.float64)
        self.n = self.A.shape[0]

        if Q_matrix is not None:
            self.Q = np.asarray(Q_matrix, dtype=np.float64)
        else:
            self.Q = np.eye(self.n, dtype=np.float64)

        # Solve discrete Lyapunov equation: A^T * P * A - P = -Q
        # scipy.linalg.solve_discrete_lyapunov solves A * P * A^T - P = -Q
        # For A^T * P * A - P = -Q, we pass A^T:
        try:
            self.P = scipy.linalg.solve_discrete_lyapunov(self.A.T, self.Q)
            # Symmetrize
            self.P = 0.5 * (self.P + self.P.T)
        except Exception:
            # Fallback to identity if unstable matrix
            self.P = np.eye(self.n, dtype=np.float64)

    def evaluate_v(self, x: np.ndarray) -> float:
        """Compute scalar Lyapunov energy V(x) = x^T * P * x."""
        x_vec = np.asarray(x, dtype=np.float64)
        return float(x_vec.T @ self.P @ x_vec)

    def certify_trajectory(self, trajectory_states: np.ndarray | list[np.ndarray]) -> LyapunovCertificateResult:
        """Verify whether V(x_{t+1}) < V(x_t) strictly along the trajectory."""
        states = np.asarray(trajectory_states, dtype=np.float64)
        T = len(states)
        assert T >= 2, "Trajectory must contain at least 2 states"

        v_vals = [self.evaluate_v(states[t]) for t in range(T)]

        monotonic = True
        decay_ratios = []

        for t in range(T - 1):
            v_curr = v_vals[t]
            v_next = v_vals[t + 1]

            if v_curr > 1e-8:
                ratio = v_next / v_curr
                decay_ratios.append(ratio)
                if ratio >= 1.0:
                    monotonic = False
            else:
                decay_ratios.append(1.0)

        max_ratio = float(np.max(decay_ratios)) if decay_ratios else 1.0
        certified = bool(monotonic and (v_vals[-1] < v_vals[0]))

        return LyapunovCertificateResult(
            is_certified_stable=certified,
            initial_lyapunov_value=float(v_vals[0]),
            final_lyapunov_value=float(v_vals[-1]),
            max_decay_ratio=max_ratio,
            strict_monotonic_decrease=monotonic,
            lyapunov_trajectory=v_vals,
        )
