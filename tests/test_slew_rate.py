"""Unit tests for actuator slew-rate and acceleration limiter."""

import numpy as np
import pytest

from kinesis.safety.slew_rate import ActuatorRateLimiter, SlewRateResult


def test_slew_rate_limiter_step_response():
    """Verify rate limiter clips large sudden jump in commanded action."""
    dt = 0.1
    # Max rate = 1.0 unit / sec -> max step delta = 0.1
    limiter = ActuatorRateLimiter(
        max_rate=1.0,
        action_min=-2.0,
        action_max=2.0,
        dt=dt,
    )
    limiter.reset(initial_action=np.array([0.0]))

    # Step jump from 0.0 to 1.0
    res1 = limiter.filter(np.array([1.0]))
    assert isinstance(res1, SlewRateResult)
    assert res1.was_rate_limited is True
    # Should only increase by 0.1
    np.testing.assert_allclose(res1.filtered_action, np.array([0.1]))

    # Second step command
    res2 = limiter.filter(np.array([1.0]))
    assert res2.was_rate_limited is True
    np.testing.assert_allclose(res2.filtered_action, np.array([0.2]))


def test_slew_rate_within_limits():
    """Verify signals within permissible rate limits pass through unchanged."""
    dt = 0.1
    limiter = ActuatorRateLimiter(
        max_rate=5.0,
        dt=dt,
    )
    limiter.reset(initial_action=np.array([0.0, 0.0]))

    # Command within limit: delta = 0.1 -> rate = 1.0 < 5.0
    res = limiter.filter(np.array([0.1, -0.1]))
    assert res.was_rate_limited is False
    np.testing.assert_allclose(res.filtered_action, np.array([0.1, -0.1]))


def test_slew_rate_acceleration_jerk_limit():
    """Verify acceleration jerk limiter smooths rate transitions."""
    dt = 0.1
    # Max rate = 10.0, max accel = 2.0 / sec^2 -> max delta rate = 0.2 / sec
    limiter = ActuatorRateLimiter(
        max_rate=10.0,
        max_accel=2.0,
        dt=dt,
    )
    limiter.reset(initial_action=np.array([0.0]))

    # Step command
    res1 = limiter.filter(np.array([5.0]))
    # First rate should be constrained by acceleration
    assert res1.was_accel_limited is True
    assert res1.slew_rate[0] <= 0.2 + 1e-6
