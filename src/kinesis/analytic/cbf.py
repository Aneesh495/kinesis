"""Discrete-time Control Barrier Function (CBF) safety filter with Quadratic Program projection."""

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class CBFFilterResult:
    """Telemetry from CBF safety filter."""
    filtered_action: np.ndarray
    nominal_action: np.ndarray
    barrier_value: float
    barrier_next_pred: float
    safety_violation_mitigated: bool
    qp_converged: bool


class DiscreteControlBarrierFilter:
    """Discrete-time Control Barrier Function safety filter projected via QP."""

    def __init__(
        self,
        f: Callable[[np.ndarray, np.ndarray], np.ndarray],
        barrier_func: Callable[[np.ndarray], float],
        barrier_decay_gamma: float = 0.2,  # gamma in (0, 1)
        action_bounds: tuple[float, float] = (-1.0, 1.0),
        eps: float = 1e-5,
    ) -> None:
        self.f = f
        self.h = barrier_func
        self.gamma = float(barrier_decay_gamma)
        self.u_min, self.u_max = action_bounds
        self.eps = eps

    def _gradient_h_next(self, x: np.ndarray, u_nom: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute gradient of next-step barrier function with respect to action u."""
        m = len(u_nom)
        grad = np.zeros(m, dtype=np.float64)
        x_next_nom = self.f(x, u_nom)
        h_next_nom = self.h(x_next_nom)

        for j in range(m):
            du = np.zeros(m, dtype=np.float64)
            du[j] = self.eps
            x_next_p = self.f(x, u_nom + du)
            x_next_m = self.f(x, u_nom - du)
            grad[j] = (self.h(x_next_p) - self.h(x_next_m)) / (2.0 * self.eps)

        return grad, h_next_nom

    def filter_action(self, x: np.ndarray, u_nom: np.ndarray) -> CBFFilterResult:
        """Filter nominal action to satisfy discrete CBF condition: h(x_{k+1}) - h(x_k) >= -gamma * h(x_k)."""
        u_nom = np.asarray(u_nom, dtype=np.float64)
        m = len(u_nom)
        h_curr = self.h(x)

        # Target minimum next barrier value: h(x_{next}) >= (1 - gamma) * h(x)
        h_threshold = (1.0 - self.gamma) * h_curr

        # Gradient of h(x_next) with respect to u at u_nom:
        grad_u, h_next_nom = self._gradient_h_next(x, u_nom)

        # If nominal action already satisfies barrier constraint:
        if h_next_nom >= h_threshold:
            u_clipped = np.clip(u_nom, self.u_min, self.u_max)
            return CBFFilterResult(
                filtered_action=u_clipped,
                nominal_action=u_nom,
                barrier_value=h_curr,
                barrier_next_pred=h_next_nom,
                safety_violation_mitigated=False,
                qp_converged=True,
            )

        # Constraint: grad_u^T (u - u_nom) + h_next_nom >= h_threshold
        # <=> grad_u^T delta_u >= h_threshold - h_next_nom
        # Let deficit = h_threshold - h_next_nom > 0
        # Minimize 0.5 * ||delta_u||_2^2  subject to  grad_u^T delta_u >= deficit
        deficit = h_threshold - h_next_nom
        grad_norm_sq = np.dot(grad_u, grad_u)

        if grad_norm_sq < 1e-8:
            # Control has no authority over barrier function; clip nominal
            u_clipped = np.clip(u_nom, self.u_min, self.u_max)
            return CBFFilterResult(
                filtered_action=u_clipped,
                nominal_action=u_nom,
                barrier_value=h_curr,
                barrier_next_pred=h_next_nom,
                safety_violation_mitigated=False,
                qp_converged=False,
            )

        # Closed-form minimum projection onto half-space:
        delta_u = (deficit / grad_norm_sq) * grad_u
        u_filtered = np.clip(u_nom + delta_u, self.u_min, self.u_max)

        # Check resulting next barrier
        x_next_filtered = self.f(x, u_filtered)
        h_next_filtered = self.h(x_next_filtered)

        return CBFFilterResult(
            filtered_action=u_filtered,
            nominal_action=u_nom,
            barrier_value=h_curr,
            barrier_next_pred=h_next_filtered,
            safety_violation_mitigated=True,
            qp_converged=True,
        )
