"""Adaptive planning horizon scheduler dynamically modulated by epistemic uncertainty."""

from dataclasses import dataclass
import numpy as np


@dataclass
class HorizonScheduleConfig:
    """Configuration for uncertainty-driven dynamic horizon scheduling."""
    min_horizon: int = 3
    max_horizon: int = 25
    initial_horizon: int = 5
    uncertainty_threshold: float = 0.5
    confidence_window: int = 5
    decay_ratio: float = 0.7
    growth_step: int = 1


@dataclass
class HorizonAdaptationEvent:
    """Record of a horizon adjustment event."""
    step: int
    previous_horizon: int
    new_horizon: int
    observed_uncertainty: float
    reason: str  # "uncertainty_spike", "confidence_expansion", "steady"


class AdaptiveHorizonScheduler:
    """Dynamically anneals or contracts MPC planning horizon according to model fidelity."""

    def __init__(self, config: HorizonScheduleConfig | None = None) -> None:
        self.config = config if config is not None else HorizonScheduleConfig()
        self.current_horizon = int(self.config.initial_horizon)
        self.step_count = 0
        self.low_uncertainty_run = 0
        self.history: list[HorizonAdaptationEvent] = []

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.current_horizon = int(self.config.initial_horizon)
        self.step_count = 0
        self.low_uncertainty_run = 0
        self.history.clear()

    def step(self, epistemic_uncertainty: float) -> int:
        """Update and return planning horizon based on current uncertainty reading."""
        u_val = float(epistemic_uncertainty)
        self.step_count += 1
        prev_h = self.current_horizon
        reason = "steady"

        if u_val > self.config.uncertainty_threshold:
            # Epistemic risk spike: contract horizon to avoid error compounding
            contracted = int(np.floor(self.current_horizon * self.config.decay_ratio))
            new_h = max(self.config.min_horizon, contracted)
            self.low_uncertainty_run = 0
            if new_h != prev_h:
                reason = "uncertainty_spike"
            self.current_horizon = new_h
        else:
            # Model is confident: accumulate consecutive low-uncertainty steps
            self.low_uncertainty_run += 1
            if self.low_uncertainty_run >= self.config.confidence_window:
                expanded = self.current_horizon + self.config.growth_step
                new_h = min(self.config.max_horizon, expanded)
                self.low_uncertainty_run = 0
                if new_h != prev_h:
                    reason = "confidence_expansion"
                self.current_horizon = new_h

        self.history.append(
            HorizonAdaptationEvent(
                step=self.step_count,
                previous_horizon=prev_h,
                new_horizon=self.current_horizon,
                observed_uncertainty=u_val,
                reason=reason,
            )
        )

        return self.current_horizon
