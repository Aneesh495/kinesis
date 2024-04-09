"""Convex polyhedral invariant set auditor and constraint projection filter."""

from dataclasses import dataclass
import numpy as np
import scipy.optimize


@dataclass
class PolyhedralAuditResult:
    """Detailed audit metrics of a state relative to a polyhedral constraint set."""
    is_contained: bool
    euclidean_safety_margin: float
    raw_slack: np.ndarray
    most_critical_constraint_idx: int
    projected_safe_state: np.ndarray


class PolyhedralConstraintSet:
    """Represents convex polyhedral set S = { x in R^n | H * x <= h }."""

    def __init__(self, H: np.ndarray, h: np.ndarray) -> None:
        self.H = np.asarray(H, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64).flatten()

        assert self.H.ndim == 2, "H must be a 2D matrix"
        assert self.H.shape[0] == len(self.h), "Row count of H must match dimension of h"

        self.num_constraints = self.H.shape[0]
        self.state_dim = self.H.shape[1]

        # Compute row-wise Euclidean norms of H for normalized distance
        self.row_norms = np.linalg.norm(self.H, axis=1)
        # Avoid division by zero for zero rows
        self.row_norms = np.maximum(self.row_norms, 1e-12)

    def contains(self, x: np.ndarray, tolerance: float = 1e-7) -> bool:
        """Check if H * x <= h + tol elementwise."""
        x_vec = np.asarray(x, dtype=np.float64).flatten()
        return bool(np.all(self.H @ x_vec <= self.h + tolerance))

    def compute_slack(self, x: np.ndarray) -> np.ndarray:
        """Compute slack vector s = h - H * x (positive implies interior)."""
        x_vec = np.asarray(x, dtype=np.float64).flatten()
        return self.h - (self.H @ x_vec)

    def compute_normalized_margin(self, x: np.ndarray) -> float:
        """Compute minimum normalized Euclidean distance to the boundary (positive inside)."""
        slack = self.compute_slack(x)
        normalized_slack = slack / self.row_norms
        return float(np.min(normalized_slack))

    def project_onto_set(self, x: np.ndarray) -> np.ndarray:
        """Project state x onto polyhedral set via quadratic programming min ||y - x||^2."""
        x_vec = np.asarray(x, dtype=np.float64).flatten()
        if self.contains(x_vec):
            return x_vec.copy()

        def objective(y: np.ndarray) -> float:
            diff = y - x_vec
            return float(0.5 * np.dot(diff, diff))

        def jacobian(y: np.ndarray) -> np.ndarray:
            return y - x_vec

        constraints = scipy.optimize.LinearConstraint(
            self.H,
            -np.inf,
            self.h,
            keep_feasible=False,
        )

        res = scipy.optimize.minimize(
            fun=objective,
            x0=x_vec,
            jac=jacobian,
            constraints=[constraints],
            method="SLSQP",
            options={"ftol": 1e-9, "maxiter": 200},
        )

        if res.success:
            return np.asarray(res.x, dtype=np.float64)
        return x_vec

    def audit(self, x: np.ndarray) -> PolyhedralAuditResult:
        """Run comprehensive audit on state x against all polyhedral hyperplanes."""
        x_vec = np.asarray(x, dtype=np.float64).flatten()
        slack = self.compute_slack(x_vec)
        normalized_slack = slack / self.row_norms

        min_idx = int(np.argmin(normalized_slack))
        min_margin = float(normalized_slack[min_idx])
        contained = bool(min_margin >= -1e-7)

        projected = self.project_onto_set(x_vec)

        return PolyhedralAuditResult(
            is_contained=contained,
            euclidean_safety_margin=min_margin,
            raw_slack=slack,
            most_critical_constraint_idx=min_idx,
            projected_safe_state=projected,
        )
