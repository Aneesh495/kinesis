"""Discrete-time backwards reachability analysis and set invariance verification."""

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class ReachabilityResult:
    """Telemetry from backwards reachable set check."""
    is_safe: bool
    distance_to_boundary: float
    violating_horizon_step: int | None
    trajectory_states: list[np.ndarray]


class BackwardsReachabilityVerifier:
    """Verifies that future states along control rollouts stay within safe invariant bounds."""

    def __init__(
        self,
        safe_box_min: np.ndarray,
        safe_box_max: np.ndarray,
        horizon: int = 15,
    ) -> None:
        self.box_min = np.asarray(safe_box_min, dtype=np.float64)
        self.box_max = np.asarray(safe_box_max, dtype=np.float64)
        self.horizon = horizon
        self.n = len(self.box_min)
        assert len(self.box_max) == self.n, "Dimension mismatch between min and max bounds"
        assert np.all(self.box_min <= self.box_max), "Min bounds must be <= max bounds"

    def distance_to_boundary(self, x: np.ndarray) -> float:
        """Signed distance to boundary of safe box (positive inside, negative outside)."""
        dist_to_min = x - self.box_min
        dist_to_max = self.box_max - x
        all_margins = np.concatenate([dist_to_min, dist_to_max])
        return float(np.min(all_margins))

    def verify_trajectory(
        self,
        initial_state: np.ndarray,
        action_sequence: np.ndarray,
        transition_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> ReachabilityResult:
        """Simulate trajectory forward and verify invariance within safe set."""
        curr_x = np.asarray(initial_state, dtype=np.float64).copy()
        traj = [curr_x.copy()]
        min_distance = self.distance_to_boundary(curr_x)

        if min_distance < 0.0:
            return ReachabilityResult(
                is_safe=False,
                distance_to_boundary=min_distance,
                violating_horizon_step=0,
                trajectory_states=traj,
            )

        H = min(len(action_sequence), self.horizon)
        for t in range(H):
            u_t = action_sequence[t]
            curr_x = transition_fn(curr_x, u_t)
            traj.append(curr_x.copy())

            dist = self.distance_to_boundary(curr_x)
            min_distance = min(min_distance, dist)
            if dist < 0.0:
                return ReachabilityResult(
                    is_safe=False,
                    distance_to_boundary=dist,
                    violating_horizon_step=t + 1,
                    trajectory_states=traj,
                )

        return ReachabilityResult(
            is_safe=True,
            distance_to_boundary=min_distance,
            violating_horizon_step=None,
            trajectory_states=traj,
        )
