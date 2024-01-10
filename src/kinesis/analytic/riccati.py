"""Discrete-time Riccati recursion and finite-horizon LQR optimal gain computation."""

from dataclasses import dataclass
import numpy as np

from kinesis.analytic.system import LinearGaussianSystem


@dataclass(frozen=True)
class LQRTrajectorySolution:
    """Exact finite-horizon LQR backward solution."""
    gains: tuple[np.ndarray, ...]  # K_0, K_1, ..., K_{N-1}
    cost_matrices: tuple[np.ndarray, ...]  # P_0, P_1, ..., P_N
    horizon: int

    def compute_cost_to_go(self, t: int, x: np.ndarray) -> float:
        """Evaluate quadratic value function V_t(x) = x^T P_t x."""
        if not (0 <= t <= self.horizon):
            raise IndexError(f"Time step {t} out of horizon bounds [0, {self.horizon}]")
        P_t = self.cost_matrices[t]
        val = x.T @ P_t @ x
        return float(np.squeeze(val))

    def get_action(self, t: int, x: np.ndarray) -> np.ndarray:
        """Compute optimal control action u_t^* = -K_t x_t."""
        if not (0 <= t < self.horizon):
            raise IndexError(f"Time step {t} out of control bounds [0, {self.horizon - 1}]")
        K_t = self.gains[t]
        return -K_t @ x


def solve_finite_horizon_lqr(sys: LinearGaussianSystem, horizon: int) -> LQRTrajectorySolution:
    """Solve the finite-horizon discrete-time Riccati recursion backwards from terminal time N."""
    if horizon < 1:
        raise ValueError(f"Horizon must be >= 1, got {horizon}")

    # If delay exists, augment system first
    augmented_sys = sys.augment_for_delay()
    A, B, Q, R, Q_f = augmented_sys.A, augmented_sys.B, augmented_sys.Q, augmented_sys.R, augmented_sys.Q_f

    # P_N = Q_f
    P_list = [Q_f.copy()]
    K_list = []

    # Backward dynamic programming
    for _ in range(horizon - 1, -1, -1):
        P_next = P_list[-1]

        # S_t = R + B^T P_{t+1} B
        S_t = R + B.T @ P_next @ B

        # K_t = S_t^{-1} B^T P_{t+1} A
        K_t = np.linalg.solve(S_t, B.T @ P_next @ A)

        # P_t = Q + A^T P_{t+1} A - A^T P_{t+1} B K_t
        P_t = Q + A.T @ P_next @ A - A.T @ P_next @ B @ K_t

        # Symmetrize for numerical stability
        P_t = 0.5 * (P_t + P_t.T)

        K_list.append(K_t)
        P_list.append(P_t)

    # Reverse to forward chronological order
    K_forward = tuple(reversed(K_list))
    P_forward = tuple(reversed(P_list))

    return LQRTrajectorySolution(
        gains=K_forward,
        cost_matrices=P_forward,
        horizon=horizon
    )


def solve_dare(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, max_iter: int = 1000, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Solve Discrete-time Algebraic Riccati Equation (DARE) using value iteration."""
    P = Q.copy()
    for _ in range(max_iter):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ K
        P_next = 0.5 * (P_next + P_next.T)
        if np.max(np.abs(P_next - P)) < tol:
            K_opt = np.linalg.solve(R + B.T @ P_next @ B, B.T @ P_next @ A)
            return P_next, K_opt
        P = P_next
    raise RuntimeError("DARE did not converge within maximum iterations")
