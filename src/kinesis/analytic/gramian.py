"""Discrete-time Controllability and Observability Gramian solvers and metrics."""

from dataclasses import dataclass
import numpy as np
import scipy.linalg


@dataclass
class GramianAnalysisResult:
    """Quantitative evaluation of controllability and observability."""
    is_controllable: bool
    is_observable: bool
    controllability_rank: int
    observability_rank: int
    system_dimension: int
    controllability_gramian: np.ndarray
    observability_gramian: np.ndarray
    min_controllability_singular_value: float
    min_observability_singular_value: float
    controllability_condition_number: float
    observability_condition_number: float


class DiscreteGramianAnalyzer:
    """Analyzes discrete linear systems via finite and infinite horizon Gramians."""

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.n = self.A.shape[0]

        if self.B.ndim == 1:
            self.B = self.B[:, None]
        if self.C.ndim == 1:
            self.C = self.C[None, :]

        self.m = self.B.shape[1]
        self.p = self.C.shape[0]

    def controllability_matrix(self) -> np.ndarray:
        """Construct the discrete controllability matrix [B, AB, ..., A^(n-1)B]."""
        blocks = [self.B]
        curr = self.B
        for _ in range(1, self.n):
            curr = self.A @ curr
            blocks.append(curr)
        return np.hstack(blocks)

    def observability_matrix(self) -> np.ndarray:
        """Construct the discrete observability matrix [C; CA; ...; CA^(n-1)]."""
        blocks = [self.C]
        curr = self.C
        for _ in range(1, self.n):
            curr = curr @ self.A
            blocks.append(curr)
        return np.vstack(blocks)

    def finite_controllability_gramian(self, horizon: int) -> np.ndarray:
        """Compute finite-horizon controllability Gramian sum_{k=0}^{N-1} A^k B B^T (A^T)^k."""
        W_c = np.zeros((self.n, self.n), dtype=np.float64)
        curr = self.B
        for _ in range(horizon):
            W_c += curr @ curr.T
            curr = self.A @ curr
        return 0.5 * (W_c + W_c.T)

    def finite_observability_gramian(self, horizon: int) -> np.ndarray:
        """Compute finite-horizon observability Gramian sum_{k=0}^{N-1} (A^T)^k C^T C A^k."""
        W_o = np.zeros((self.n, self.n), dtype=np.float64)
        curr = self.C
        for _ in range(horizon):
            W_o += curr.T @ curr
            curr = curr @ self.A
        return 0.5 * (W_o + W_o.T)

    def infinite_controllability_gramian(self) -> np.ndarray:
        """Solve A W_c A^T - W_c = -B B^T for stable systems."""
        try:
            W_c = scipy.linalg.solve_discrete_lyapunov(self.A, self.B @ self.B.T)
            return 0.5 * (W_c + W_c.T)
        except Exception:
            return self.finite_controllability_gramian(horizon=50)

    def infinite_observability_gramian(self) -> np.ndarray:
        """Solve A^T W_o A - W_o = -C^T C for stable systems."""
        try:
            W_o = scipy.linalg.solve_discrete_lyapunov(self.A.T, self.C.T @ self.C)
            return 0.5 * (W_o + W_o.T)
        except Exception:
            return self.finite_observability_gramian(horizon=50)

    def analyze(self, horizon: int | None = None) -> GramianAnalysisResult:
        """Run full structural analysis of controllability and observability."""
        c_mat = self.controllability_matrix()
        o_mat = self.observability_matrix()

        c_rank = int(np.linalg.matrix_rank(c_mat))
        o_rank = int(np.linalg.matrix_rank(o_mat))

        if horizon is not None:
            W_c = self.finite_controllability_gramian(horizon)
            W_o = self.finite_observability_gramian(horizon)
        else:
            W_c = self.infinite_controllability_gramian()
            W_o = self.infinite_observability_gramian()

        s_c = np.linalg.svd(W_c, compute_uv=False)
        s_o = np.linalg.svd(W_o, compute_uv=False)

        min_s_c = float(s_c[-1]) if len(s_c) > 0 else 0.0
        min_s_o = float(s_o[-1]) if len(s_o) > 0 else 0.0

        cond_c = float(s_c[0] / max(s_c[-1], 1e-12)) if len(s_c) > 0 else float("inf")
        cond_o = float(s_o[0] / max(s_o[-1], 1e-12)) if len(s_o) > 0 else float("inf")

        return GramianAnalysisResult(
            is_controllable=(c_rank == self.n),
            is_observable=(o_rank == self.n),
            controllability_rank=c_rank,
            observability_rank=o_rank,
            system_dimension=self.n,
            controllability_gramian=W_c,
            observability_gramian=W_o,
            min_controllability_singular_value=min_s_c,
            min_observability_singular_value=min_s_o,
            controllability_condition_number=cond_c,
            observability_condition_number=cond_o,
        )
