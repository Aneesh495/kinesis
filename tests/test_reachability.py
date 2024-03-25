"""Unit tests for Backwards Reachability verifier."""

import numpy as np
import pytest

from kinesis.safety.reachability import BackwardsReachabilityVerifier


def test_reachability_safe_trajectory():
    """Verify safe trajectory verification within bounds."""
    verifier = BackwardsReachabilityVerifier(
        safe_box_min=np.array([-2.0, -2.0]),
        safe_box_max=np.array([2.0, 2.0]),
        horizon=10,
    )

    def f(x, u):
        return x * 0.9 + u * 0.1

    x0 = np.array([0.5, -0.5])
    actions = np.zeros((10, 2))

    res = verifier.verify_trajectory(x0, actions, f)
    assert res.is_safe is True
    assert res.distance_to_boundary > 0.0
    assert res.violating_horizon_step is None
    assert len(res.trajectory_states) == 11


def test_reachability_boundary_violation():
    """Verify detection of boundary breach along trajectory."""
    verifier = BackwardsReachabilityVerifier(
        safe_box_min=np.array([-1.0]),
        safe_box_max=np.array([1.0]),
        horizon=5,
    )

    def f(x, u):
        return x + u

    x0 = np.array([0.8])
    # Action pushes x over 1.0 at step 1
    actions = np.array([[0.5], [0.5]])

    res = verifier.verify_trajectory(x0, actions, f)
    assert res.is_safe is False
    assert res.violating_horizon_step == 1
    assert res.distance_to_boundary < 0.0
