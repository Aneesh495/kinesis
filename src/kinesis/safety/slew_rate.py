"""Actuator slew-rate and acceleration jerk limiting filter."""

from dataclasses import dataclass
import numpy as np


@dataclass
class SlewRateResult:
    """Detailed output and telemetry from the slew rate filter."""
    filtered_action: np.ndarray
    raw_action: np.ndarray
    was_rate_limited: bool
    was_accel_limited: bool
    slew_rate: np.ndarray
    acceleration: np.ndarray | None


class ActuatorRateLimiter:
    """Constrains actuator rate of change and acceleration jerk to prevent mechanical damage."""

    def __init__(
        self,
        max_rate: float | np.ndarray,
        max_accel: float | np.ndarray | None = None,
        action_min: float | np.ndarray | None = None,
        action_max: float | np.ndarray | None = None,
        dt: float = 0.05,
    ) -> None:
        self.max_rate = np.asarray(max_rate, dtype=np.float64)
        self.max_accel = np.asarray(max_accel, dtype=np.float64) if max_accel is not None else None
        self.action_min = np.asarray(action_min, dtype=np.float64) if action_min is not None else None
        self.action_max = np.asarray(action_max, dtype=np.float64) if action_max is not None else None
        self.dt = float(dt)

        self.prev_action: np.ndarray | None = None
        self.prev_rate: np.ndarray | None = None
        self.intervention_count = 0
        self.total_steps = 0

    def reset(self, initial_action: np.ndarray | None = None) -> None:
        """Reset internal filter state."""
        self.prev_action = np.asarray(initial_action, dtype=np.float64).copy() if initial_action is not None else None
        self.prev_rate = np.zeros_like(self.prev_action) if self.prev_action is not None else None
        self.intervention_count = 0
        self.total_steps = 0

    def filter(self, action: np.ndarray) -> SlewRateResult:
        """Filter command action through rate and acceleration limits."""
        u_raw = np.asarray(action, dtype=np.float64)
        self.total_steps += 1

        if self.prev_action is None:
            # First action
            u_filtered = u_raw.copy()
            if self.action_min is not None and self.action_max is not None:
                u_filtered = np.clip(u_filtered, self.action_min, self.action_max)
            self.prev_action = u_filtered.copy()
            self.prev_rate = np.zeros_like(u_filtered)
            return SlewRateResult(
                filtered_action=u_filtered,
                raw_action=u_raw,
                was_rate_limited=False,
                was_accel_limited=False,
                slew_rate=np.zeros_like(u_filtered),
                acceleration=None,
            )

        # Desired rate of change
        delta_u = u_raw - self.prev_action
        rate = delta_u / self.dt
        rate_limited = False
        accel_limited = False
        accel = None

        # Check acceleration limit if configured and prev_rate exists
        if self.max_accel is not None and self.prev_rate is not None:
            delta_rate = rate - self.prev_rate
            accel = delta_rate / self.dt
            max_delta_rate = self.max_accel * self.dt

            clipped_delta_rate = np.clip(delta_rate, -max_delta_rate, max_delta_rate)
            if not np.allclose(clipped_delta_rate, delta_rate):
                accel_limited = True
                rate = self.prev_rate + clipped_delta_rate

        # Check rate limit
        max_delta_u = self.max_rate * self.dt
        effective_delta_u = rate * self.dt
        clipped_delta_u = np.clip(effective_delta_u, -max_delta_u, max_delta_u)
        if not np.allclose(clipped_delta_u, effective_delta_u):
            rate_limited = True

        u_filtered = self.prev_action + clipped_delta_u

        # Enforce static action bounds
        if self.action_min is not None and self.action_max is not None:
            u_filtered = np.clip(u_filtered, self.action_min, self.action_max)

        # Compute actual realized rate
        actual_rate = (u_filtered - self.prev_action) / self.dt
        if rate_limited or accel_limited:
            self.intervention_count += 1

        self.prev_action = u_filtered.copy()
        self.prev_rate = actual_rate.copy()

        return SlewRateResult(
            filtered_action=u_filtered,
            raw_action=u_raw,
            was_rate_limited=rate_limited,
            was_accel_limited=accel_limited,
            slew_rate=actual_rate,
            acceleration=accel,
        )
