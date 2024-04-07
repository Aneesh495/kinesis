"""Discrete-time frequency response, Bode analysis, and stability margin computation."""

from dataclasses import dataclass
import numpy as np


@dataclass
class BodeResponseResult:
    """Frequency response spectrum and stability margins."""
    frequencies: np.ndarray
    magnitude_db: np.ndarray
    phase_degrees: np.ndarray
    gain_margin_db: float | None
    phase_margin_degrees: float | None
    gain_crossover_freq: float | None
    phase_crossover_freq: float | None
    is_open_loop_stable: bool


class DiscreteBodeAnalyzer:
    """Computes discrete frequency response G(exp(j * w * dt)) and Bode margins."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray | None = None,
        dt: float = 0.05,
    ) -> None:
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.n = self.A.shape[0]
        self.dt = float(dt)

        if self.B.ndim == 1:
            self.B = self.B[:, None]
        if self.C.ndim == 1:
            self.C = self.C[None, :]

        if D is not None:
            self.D = np.asarray(D, dtype=np.float64)
            if self.D.ndim == 0:
                self.D = np.array([[self.D]])
        else:
            self.D = np.zeros((self.C.shape[0], self.B.shape[1]), dtype=np.float64)

    def evaluate_frequency_response(self, w: float) -> complex:
        """Evaluate scalar transfer function G(exp(j * w * dt))."""
        z = np.exp(1j * w * self.dt)
        # G(z) = C * (z * I - A)^(-1) * B + D
        zI_minus_A = z * np.eye(self.n) - self.A
        try:
            inv = np.linalg.inv(zI_minus_A)
            resp = self.C @ inv @ self.B + self.D
            return complex(resp[0, 0])
        except np.linalg.LinAlgError:
            return complex(np.nan, np.nan)

    def compute_bode(
        self,
        num_points: int = 200,
        w_min: float = 0.01,
        w_max: float | None = None,
    ) -> BodeResponseResult:
        """Compute Bode magnitude and phase across logarithmic frequency sweep."""
        nyquist = np.pi / self.dt
        if w_max is None or w_max >= nyquist:
            w_max = nyquist * 0.99

        w_grid = np.logspace(np.log10(w_min), np.log10(w_max), num_points)
        responses = [self.evaluate_frequency_response(w) for w in w_grid]

        mags = np.abs(responses)
        mag_db = 20.0 * np.log10(np.maximum(mags, 1e-12))
        phases_rad = np.unwrap(np.angle(responses))
        phases_deg = np.rad2deg(phases_rad)

        # Gain crossover: where mag_db crosses 0 dB
        gain_cross_freq = None
        phase_margin = None
        for i in range(len(mag_db) - 1):
            if (mag_db[i] >= 0.0 and mag_db[i + 1] < 0.0) or (mag_db[i] <= 0.0 and mag_db[i + 1] > 0.0):
                # Linear interpolation for frequency
                frac = (0.0 - mag_db[i]) / (mag_db[i + 1] - mag_db[i])
                gain_cross_freq = float(w_grid[i] + frac * (w_grid[i + 1] - w_grid[i]))
                interpolated_phase = phases_deg[i] + frac * (phases_deg[i + 1] - phases_deg[i])
                # Phase margin = 180 + phase
                # Normalize phase to [-180, 180] relative to -180
                phase_margin = float(180.0 + (interpolated_phase % 360.0 if interpolated_phase < 0 else interpolated_phase - 360.0))
                break

        # Phase crossover: where phase crosses -180 degrees
        phase_cross_freq = None
        gain_margin = None
        target_phase = -180.0
        for i in range(len(phases_deg) - 1):
            p1, p2 = phases_deg[i], phases_deg[i + 1]
            if (p1 >= target_phase and p2 < target_phase) or (p1 <= target_phase and p2 > target_phase):
                frac = (target_phase - p1) / (p2 - p1)
                phase_cross_freq = float(w_grid[i] + frac * (w_grid[i + 1] - w_grid[i]))
                interpolated_mag_db = mag_db[i] + frac * (mag_db[i + 1] - mag_db[i])
                gain_margin = float(-interpolated_mag_db)
                break

        # Stability check
        eigvals = np.linalg.eigvals(self.A)
        is_stable = bool(np.all(np.abs(eigvals) < 1.0))

        return BodeResponseResult(
            frequencies=w_grid,
            magnitude_db=mag_db,
            phase_degrees=phases_deg,
            gain_margin_db=gain_margin,
            phase_margin_degrees=phase_margin,
            gain_crossover_freq=gain_cross_freq,
            phase_crossover_freq=phase_cross_freq,
            is_open_loop_stable=is_stable,
        )
