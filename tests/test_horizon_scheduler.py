"""Unit tests for adaptive planning horizon scheduler."""

import pytest

from kinesis.planning.horizon_scheduler import (
    AdaptiveHorizonScheduler,
    HorizonScheduleConfig,
    HorizonAdaptationEvent,
)


def test_horizon_scheduler_growth():
    """Verify horizon expands after sustained period of low uncertainty."""
    config = HorizonScheduleConfig(
        min_horizon=3,
        max_horizon=10,
        initial_horizon=4,
        uncertainty_threshold=0.5,
        confidence_window=3,
        growth_step=1,
    )
    scheduler = AdaptiveHorizonScheduler(config)
    assert scheduler.current_horizon == 4

    # 2 steps with low uncertainty: horizon remains 4
    assert scheduler.step(0.1) == 4
    assert scheduler.step(0.1) == 4

    # 3rd step triggers expansion to 5
    h = scheduler.step(0.1)
    assert h == 5
    assert scheduler.history[-1].reason == "confidence_expansion"


def test_horizon_scheduler_contraction_on_spike():
    """Verify horizon contracts immediately when uncertainty exceeds threshold."""
    config = HorizonScheduleConfig(
        min_horizon=3,
        max_horizon=20,
        initial_horizon=10,
        uncertainty_threshold=0.5,
        decay_ratio=0.5,
    )
    scheduler = AdaptiveHorizonScheduler(config)

    # Uncertainty spike to 0.8
    h = scheduler.step(0.8)
    assert h == 5  # floor(10 * 0.5)
    assert scheduler.history[-1].reason == "uncertainty_spike"

    # Second spike contracts to min_horizon = 3
    h2 = scheduler.step(0.9)
    assert h2 == 3
