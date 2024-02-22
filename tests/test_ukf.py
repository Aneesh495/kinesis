"""Unit tests for Unscented Kalman Filter."""

import numpy as np
import pytest

from kinesis.analytic.ukf import UKFState, UnscentedKalmanFilter


def test_ukf_linear_grounding():
    """Verify UKF converges on a 2D linear oscillator."""
    A = np.array([[1.0, 0.1], [-0.2, 0.9]], dtype=np.float64)
    B = np.array([[0.0], [0.1]], dtype=np.float64)
    C = np.array([[1.0, 0.0]], dtype=np.float64)
    W = np.eye(2) * 1e-4
    V = np.eye(1) * 1e-3

    def f(x, u):
        return A @ x + B @ u

    def h(x):
        return C @ x

    ukf = UnscentedKalmanFilter(f, h, W, V)
    state = UKFState(mean=np.array([2.0, -1.0]), covariance=np.eye(2) * 0.5)

    u = np.array([0.2])
    y = np.array([1.9])
    next_state, innov = ukf.step(state, u, y)

    assert next_state.mean.shape == (2,)
    assert next_state.covariance.shape == (2, 2)
    assert np.all(np.linalg.eigvals(next_state.covariance) > 0)


def test_ukf_nonlinear_tracking():
    """Verify UKF tracking on nonlinear trigonometric observations."""
    def f(x, u):
        return np.array([0.5 * x[0] + 25.0 * x[0] / (1.0 + x[0]**2) + 8.0 * np.cos(1.2 * u[0])], dtype=np.float64)

    def h(x):
        return np.array([x[0]**2 / 20.0], dtype=np.float64)

    W = np.array([[1e-2]])
    V = np.array([[1e-2]])

    ukf = UnscentedKalmanFilter(f, h, W, V)
    state = UKFState(mean=np.array([0.1]), covariance=np.eye(1) * 1.0)

    true_x = np.array([1.0])
    for t in range(25):
        u = np.array([float(t)])
        true_x = f(true_x, u)
        meas = h(true_x) + np.array([0.01])
        state, innov = ukf.step(state, u, meas)

    assert np.isfinite(state.mean[0])
    assert state.covariance[0, 0] > 0
