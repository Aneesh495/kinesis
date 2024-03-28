"""Phase space trajectory portraits and vector field diagnostics."""

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class PhasePortraitMetrics:
    """Quantitative metrics extracted from phase portrait."""
    orbit_enclosed_area: float
    max_phase_velocity: float
    apparent_limit_cycle: bool
    quiver_grid_shape: tuple[int, int]


class PhasePortraitAnalyzer:
    """Analyzes 2D phase-space dynamics (q, q_dot) and renders ASCII vector field portraits."""

    def __init__(self, q_bounds: tuple[float, float] = (-np.pi, np.pi), v_bounds: tuple[float, float] = (-8.0, 8.0)) -> None:
        self.q_min, self.q_max = q_bounds
        self.v_min, self.v_max = v_bounds

    def compute_metrics(self, q_traj: np.ndarray, v_traj: np.ndarray) -> PhasePortraitMetrics:
        """Extract geometric phase metrics from discrete trajectory."""
        q = np.asarray(q_traj, dtype=np.float64)
        v = np.asarray(v_traj, dtype=np.float64)
        assert len(q) == len(v), "Length mismatch between q and v trajectories"

        # Approximate enclosed area via Green's theorem / shoelace formula
        # Area = 0.5 * |sum(q_i * v_{i+1} - q_{i+1} * v_i)|
        area = 0.5 * np.abs(np.sum(q[:-1] * v[1:] - q[1:] * v[:-1]))

        # Max velocity in phase space
        phase_vel = np.sqrt(np.diff(q)**2 + np.diff(v)**2)
        max_vel = float(np.max(phase_vel)) if len(phase_vel) > 0 else 0.0

        # Heuristic limit cycle detection: start and end of trajectory within small epsilon
        is_cycle = bool(len(q) > 20 and np.hypot(q[-1] - q[0], v[-1] - v[0]) < 0.2)

        return PhasePortraitMetrics(
            orbit_enclosed_area=float(area),
            max_phase_velocity=max_vel,
            apparent_limit_cycle=is_cycle,
            quiver_grid_shape=(15, 15),
        )

    def render_ascii_portrait(
        self,
        q_traj: np.ndarray,
        v_traj: np.ndarray,
        width: int = 40,
        height: int = 15,
    ) -> str:
        """Render discrete phase space trajectory as text grid."""
        grid = [["." for _ in range(width)] for _ in range(height)]

        # Map bounds to grid coordinates
        def to_grid(q_val, v_val):
            col = int((q_val - self.q_min) / max(self.q_max - self.q_min, 1e-6) * (width - 1))
            row = int((self.v_max - v_val) / max(self.v_max - self.v_min, 1e-6) * (height - 1))
            return max(0, min(height - 1, row)), max(0, min(width - 1, col))

        # Draw trajectory
        for i, (qv, vv) in enumerate(zip(q_traj, v_traj)):
            r, c = to_grid(qv, vv)
            char = "S" if i == 0 else ("E" if i == len(q_traj) - 1 else "#")
            grid[r][c] = char

        lines = ["+" + "-" * width + "+"]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * width + "+")
        lines.append(f"q: [{self.q_min:.1f}, {self.q_max:.1f}] | v: [{self.v_min:.1f}, {self.v_max:.1f}] (S:start, E:end)")

        return "\n".join(lines)
