"""Discrete-time H-infinity robust state-feedback controller via game-theoretic Riccati recursion."""

from dataclasses import dataclass
import numpy as np


@dataclass
class HInfinityResult:
    """Result of discrete H-infinity Riccati recursion."""
    P: np.ndarray  # Riccati storage matrix (n, n)
    K_u: np.ndarray  # Robust control feedback gain (m, n)
    K_w: np.ndarray  # Worst-case disturbance gain (d, n)
    gamma: float
    converged: bool
    iterations: int


class DiscreteHInfinityController:
    """Discrete-time H-infinity robust minimax state-feedback controller."""

    def __init__(
        self,
        A: np.ndarray,
        B1: np.ndarray,  # Disturbance input matrix (n, d)
        B2: np.ndarray,  # Control input matrix (n, m)
        Q: np.ndarray,   # State penalty matrix (n, n)
        R: np.ndarray,   # Control penalty matrix (m, m)
        gamma: float = 5.0,  # H-infinity attenuation level
        max_iters: int = 200,
        tol: float = 1e-6,
    ) -> None:
        self.A = np.asarray(A, dtype=np.float64)
        self.B1 = np.asarray(B1, dtype=np.float64)
        self.B2 = np.asarray(B2, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.gamma = float(gamma)
        self.max_iters = max_iters
        self.tol = tol

        self.n = self.A.shape[0]
        self.d = self.B1.shape[1]
        self.m = self.B2.shape[1]

    def solve(self) -> HInfinityResult:
        """Solve game-theoretic discrete algebraic Riccati equation."""
        P = np.array(self.Q, dtype=np.float64)
        converged = False

        for it in range(self.max_iters):
            # Check adversary matrix condition: gamma^2 * I - B1^T * P * B1 > 0
            M_w = (self.gamma**2) * np.eye(self.d) - self.B1.T @ P @ self.B1
            eig_Mw = np.linalg.eigvalsh(M_w)
            if np.any(eig_Mw <= 0):
                # Gamma is too small for stability bound
                return HInfinityResult(
                    P=P,
                    K_u=np.zeros((self.m, self.n)),
                    K_w=np.zeros((self.d, self.n)),
                    gamma=self.gamma,
                    converged=False,
                    iterations=it + 1,
                )

            inv_Mw = np.linalg.inv(M_w)
            # Effective P including adversary worst-case response
            P_tilde = P + P @ self.B1 @ inv_Mw @ self.B1.T @ P

            # Controller gain: Ku = (R + B2^T * P_tilde * B2)^(-1) * B2^T * P_tilde * A
            M_u = self.R + self.B2.T @ P_tilde @ self.B2
            Ku = np.linalg.inv(M_u) @ (self.B2.T @ P_tilde @ self.A)

            # Closed loop transition under control
            A_cl = self.A - self.B2 @ Ku

            # Next Riccati update
            # P_next = A_cl^T * P_tilde * A_cl + Q + Ku^T * R * Ku
            P_next = self.A.T @ P_tilde @ self.A - (self.A.T @ P_tilde @ self.B2) @ np.linalg.inv(M_u) @ (self.B2.T @ P_tilde @ self.A) + self.Q
            P_next = 0.5 * (P_next + P_next.T)

            diff = np.max(np.abs(P_next - P))
            P = P_next
            if diff < self.tol:
                converged = True
                break

        # Compute worst-case disturbance gain: Kw = inv_Mw * B1^T * P * (A - B2 * Ku)
        inv_Mw = np.linalg.inv((self.gamma**2) * np.eye(self.d) - self.B1.T @ P @ self.B1)
        Kw = inv_Mw @ self.B1.T @ P @ (self.A - self.B2 @ Ku)

        return HInfinityResult(
            P=P,
            K_u=Ku,
            K_w=Kw,
            gamma=self.gamma,
            converged=converged,
            iterations=it + 1,
        )

    def compute_action(self, x: np.ndarray, result: HInfinityResult) -> np.ndarray:
        """Compute state feedback control action u = -Ku * x."""
        return -result.K_u @ x
