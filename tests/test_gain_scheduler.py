"""Unit tests for Robust Policy Gain Scheduler."""

import numpy as np
import pytest

from kinesis.adaptation.gain_scheduler import OperatingRegime, RobustGainScheduler


def test_gain_scheduler_exact_match():
    """Verify that operating point matching a regime recovers its gains."""
    r_nom = OperatingRegime(
        name="nominal",
        parameter_vector=np.array([1.0, 1.0]),
        feedback_gain=np.array([[2.0, 0.5]]),
        uncertainty_weight=1.0,
    )
    r_fric = OperatingRegime(
        name="low_friction",
        parameter_vector=np.array([0.4, 1.0]),
        feedback_gain=np.array([[1.0, 0.2]]),
        uncertainty_weight=2.5,
    )

    scheduler = RobustGainScheduler([r_nom, r_fric], kernel_bandwidth=0.1)

    # Query at exact low friction point
    res = scheduler.schedule(np.array([0.4, 1.0]))
    assert res.dominant_regime == "low_friction"
    assert res.regime_weights["low_friction"] > 0.95
    assert np.allclose(res.interpolated_gain, r_fric.feedback_gain, atol=0.05)


def test_gain_scheduler_smooth_interpolation():
    """Verify smooth convex interpolation between regimes."""
    r1 = OperatingRegime(name="regime1", parameter_vector=np.array([0.0]), feedback_gain=np.eye(2))
    r2 = OperatingRegime(name="regime2", parameter_vector=np.array([1.0]), feedback_gain=2.0 * np.eye(2))

    scheduler = RobustGainScheduler([r1, r2], kernel_bandwidth=0.5)

    # Midpoint query
    res = scheduler.schedule(np.array([0.5]))
    assert np.isclose(res.regime_weights["regime1"], 0.5, atol=1e-3)
    assert np.isclose(res.regime_weights["regime2"], 0.5, atol=1e-3)
    assert np.allclose(res.interpolated_gain, 1.5 * np.eye(2))
