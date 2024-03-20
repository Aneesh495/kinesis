"""Unit tests for Recursive Least Squares (RLS) parameter estimator."""

import numpy as np
import pytest

from kinesis.adaptation.rls import RecursiveLeastSquares


def test_rls_stationary_parameter_identification():
    """Verify RLS converges to ground truth parameter vector."""
    rng = np.random.default_rng(42)
    theta_true = np.array([1.5, -0.8, 2.2], dtype=np.float64)

    rls = RecursiveLeastSquares(num_parameters=3, forgetting_factor=0.99, initial_covariance_scale=50.0)
    state = rls.initial_state()

    for _ in range(80):
        phi = rng.normal(size=3)
        y = float(np.dot(phi, theta_true) + rng.normal(scale=0.01))
        state, telem = rls.update(state, phi, y)

    assert np.allclose(state.theta, theta_true, atol=0.05)
    assert telem.trace_covariance < 10.0


def test_rls_abrupt_shift_tracking():
    """Verify that exponential forgetting factor allows tracking sudden physical parameter shifts."""
    rng = np.random.default_rng(123)
    rls = RecursiveLeastSquares(num_parameters=2, forgetting_factor=0.92)
    state = rls.initial_state()

    theta_initial = np.array([1.0, 2.0])
    for _ in range(50):
        phi = rng.normal(size=2)
        y = float(np.dot(phi, theta_initial))
        state, _ = rls.update(state, phi, y)

    assert np.allclose(state.theta, theta_initial, atol=0.1)

    # Abrupt physical shift occurs: friction / mass drops by 50%
    theta_shifted = np.array([0.5, 1.0])
    for _ in range(60):
        phi = rng.normal(size=2)
        y = float(np.dot(phi, theta_shifted))
        state, _ = rls.update(state, phi, y)

    # RLS should adapt to the new shifted parameters
    assert np.allclose(state.theta, theta_shifted, atol=0.1)
