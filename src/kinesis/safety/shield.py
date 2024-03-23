"""Runtime safety shield intercepting hazardous actions and switching to conservative fallbacks."""

from dataclasses import dataclass
from typing import Callable
import numpy as np

from kinesis.contracts.action import ActionSpec
from kinesis.policies.base import BasePolicy


@dataclass
class ShieldDecision:
    """Intervention decision from runtime safety shield."""
    executed_action: np.ndarray
    proposed_action: np.ndarray
    intervened: bool
    trigger_reason: str
    uncertainty_value: float
    safety_margin: float


class RuntimeSafetyShield:
    """Monitors epistemic risk and barrier safety margins, executing fallback when violated."""

    def __init__(
        self,
        fallback_policy: BasePolicy,
        action_spec: ActionSpec,
        max_allowed_uncertainty: float = 1.5,
        min_safety_barrier_margin: float = 0.05,
    ) -> None:
        self.fallback = fallback_policy
        self.action_spec = action_spec
        self.max_unc = float(max_allowed_uncertainty)
        self.min_margin = float(min_safety_barrier_margin)
        self.total_interventions = 0

    def evaluate_and_filter(
        self,
        observation: dict[str, np.ndarray],
        proposed_action: np.ndarray,
        estimated_uncertainty: float = 0.0,
        barrier_margin: float = 1.0,
    ) -> ShieldDecision:
        """Audit proposed action and execute fallback if risk or barrier violated."""
        prop_act = np.asarray(proposed_action, dtype=np.float32)
        intervened = False
        reason = "safe"

        # Check uncertainty violation
        if estimated_uncertainty > self.max_unc:
            intervened = True
            reason = f"uncertainty_exceeded_{estimated_uncertainty:.2f}_gt_{self.max_unc:.2f}"
        # Check barrier violation
        elif barrier_margin < self.min_margin:
            intervened = True
            reason = f"barrier_breached_{barrier_margin:.2f}_lt_{self.min_margin:.2f}"

        if intervened:
            self.total_interventions += 1
            exec_act = self.fallback.act(observation, deterministic=True)
        else:
            exec_act = self.action_spec.clip(prop_act)

        return ShieldDecision(
            executed_action=exec_act,
            proposed_action=prop_act,
            intervened=intervened,
            trigger_reason=reason,
            uncertainty_value=float(estimated_uncertainty),
            safety_margin=float(barrier_margin),
        )
