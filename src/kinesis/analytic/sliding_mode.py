"""Discrete-time Sliding Mode Controller (SMC) with boundary layer chatter alleviation."""

from dataclasses import dataclass
import numpy as np


@dataclass
class SMCResult:
    """Telemetry from sliding mode controller step."""
    action: np.ndarray
    sliding_surface_value: np.ndarray
    equivalent_action: np.ndarray
    switching_action: np.ndarray
    in_boundary_layer: bool


class DiscreteSlidingModeController:
    """Discrete-time sliding mode controller with saturation boundary layer."""

    def __init__(
        self,
        A_nominal: np.ndarray,
        B_nominal: np.ndarray,
        sliding_matrix_C: np.ndarray,
        switching_gain: float | np.ndarray = 0.5,
        boundary_layer_thickness: float = 0.05,
        action_bounds: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.A = np.asarray(A_nominal, dtype=np.float64)
        self.B = np.asarray(B_nominal, dtype=np.float64)
        self.C = np.asarray(sliding_matrix_C, dtype=np.float64)

        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.k = self.C.shape[0]
        assert self.k == self.m, f"Sliding matrix output dim {self.k} must match action dim {self.m}"

        self.CB = self.C @ self.B
        self.CB_inv = np.linalg.inv(self.CB)
        self.CA = self.C @ self.A

        if np.isscalar(switching_gain):
            self.K_sw = np.ones(self.m, dtype=np.float64) * switching_gain
        else:
            self.K_sw = np.asarray(switching_gain, dtype=np.float64)

        self.epsilon = boundary_layer_thickness
        self.action_min, self.action_max = action_bounds

    def compute_sliding_surface(self, x: np.ndarray) -> np.ndarray:
        """Compute sliding surface vector s = C * x."""
        return self.C @ x

    def compute_action(self, x: np.ndarray) -> SMCResult:
        """Compute robust sliding mode control action."""
        # 1. Sliding surface
        s = self.compute_sliding_surface(x)

        # 2. Equivalent control: u_eq = -(CB)^(-1) * CA * x
        u_eq = -self.CB_inv @ (self.CA @ x)

        # 3. Switching control with saturation boundary layer
        # sat(s / epsilon)
        s_norm = np.linalg.norm(s)
        in_layer = bool(s_norm <= self.epsilon)

        sat_s = np.clip(s / max(self.epsilon, 1e-6), -1.0, 1.0)
        u_sw = -self.K_sw * sat_s

        # 4. Total control with bounds clipping
        u_raw = u_eq + u_sw
        u_clipped = np.clip(u_raw, self.action_min, self.action_max)

        return SMCResult(
            action=u_clipped,
            sliding_surface_value=s,
            equivalent_action=u_eq,
            switching_action=u_sw,
            in_boundary_layer=in_layer,
        )
