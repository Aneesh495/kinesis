"""Temporal shift warm-start memory cache for latent trajectory optimization."""

from dataclasses import dataclass
import numpy as np


@dataclass
class WarmStartConfig:
    """Configuration for MPC warm-start trajectory shift and invalidation."""
    horizon: int
    action_dim: int
    shift_steps: int = 1
    terminal_mode: str = "repeat_last"  # "repeat_last", "zero", or "nominal"
    nominal_action: float = 0.0
    relaxation_factor: float = 1.0
    max_divergence_norm: float = 5.0


@dataclass
class WarmStartSnapshot:
    """Saved warm-start plan state."""
    action_sequence: np.ndarray
    step_index: int
    plan_cost: float | None


class WarmStartCache:
    """Maintains and shifts optimized action plans across successive MPC control steps."""

    def __init__(self, config: WarmStartConfig) -> None:
        self.config = config
        self.current_plan: np.ndarray | None = None
        self.current_cost: float | None = None
        self.step_counter: int = 0
        self.is_active: bool = False

    def reset(self) -> None:
        """Clear cache state for a new episode."""
        self.current_plan = None
        self.current_cost = None
        self.step_counter = 0
        self.is_active = False

    def invalidate(self) -> None:
        """Explicitly invalidate current warm-start plan due to disturbance or shift."""
        self.current_plan = None
        self.is_active = False

    def update(self, optimal_actions: np.ndarray, cost: float | None = None) -> None:
        """Store the newly optimized action sequence of shape (H, action_dim)."""
        actions = np.asarray(optimal_actions, dtype=np.float64)
        assert actions.ndim == 2, "Action sequence must have shape (H, action_dim)"
        assert actions.shape[1] == self.config.action_dim, "Action dim mismatch"

        self.current_plan = actions.copy()
        self.current_cost = cost
        self.step_counter += 1
        self.is_active = True

    def get_warm_start(self, target_horizon: int | None = None) -> np.ndarray | None:
        """Retrieve shifted trajectory for the next planning iteration."""
        if not self.is_active or self.current_plan is None:
            return None

        H = target_horizon if target_horizon is not None else self.config.horizon
        plan = self.current_plan
        curr_H = len(plan)

        shift = self.config.shift_steps
        if shift >= curr_H:
            self.invalidate()
            return None

        # Shift forward by removing the first `shift` steps
        shifted = plan[shift:].copy()

        # Generate terminal padding actions to match target horizon H
        needed_padding = H - len(shifted)
        if needed_padding > 0:
            if self.config.terminal_mode == "repeat_last":
                terminal_act = plan[-1]
                padding = np.tile(terminal_act, (needed_padding, 1))
            elif self.config.terminal_mode == "zero":
                padding = np.zeros((needed_padding, self.config.action_dim), dtype=np.float64)
            else:
                padding = np.full((needed_padding, self.config.action_dim), self.config.nominal_action, dtype=np.float64)
            result = np.vstack([shifted, padding])
        else:
            result = shifted[:H]

        return result

    def verify_consistency(self, executed_action: np.ndarray) -> bool:
        """Check if executed action diverged significantly from planned first action."""
        if self.current_plan is None:
            return True
        planned_first = self.current_plan[0]
        divergence = float(np.linalg.norm(np.asarray(executed_action) - planned_first))
        if divergence > self.config.max_divergence_norm:
            self.invalidate()
            return False
        return True
