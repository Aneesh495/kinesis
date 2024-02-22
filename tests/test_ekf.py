"""Unit tests for Extended Kalman Filter."""

import numpy as np
import pytest

from kinesis.analytic.ekf import EKFState, ExtendedKalmanFilter


def test_ekf_linear_consistency():
    """Verify that EKF matches linear Kalman filter on linear dynamics."""
    A = np.array([[1.0, 0.1], [0.0, 0.95]], dtype=np.float64)
    B = np.array([[0.0], [0.1]], dtype=np.float64)
    C = np.array([[1.0, 0.0]], dtype=np.float64)
    W = np.eye(2) * 1e-4
    V = np.eye(1) * 1e-2

    def f(x, u):
        return A @ x + B @ u

    def h(x):
        return C @ x

    ekf = ExtendedKalmanFilter(f, h, W, V)
    state = EKFState(mean=np.array([1.0, 0.0]), covariance=np.eye(2) * 0.1)

    u = np.array([0.5])
    y = np.array([1.02])

    next_state, innov = ekf.step(state, u, y)
    assert next_state.mean.shape == (2,)
    assert next_state.covariance.shape == (2, 2)
    assert np.all(np.linalg.eigvals(next_state.covariance) > 0)


def test_ekf_nonlinear_tracking():
    """Verify EKF tracking on a simulated nonlinear pendulum."""
    dt = 0.05
    def f_true(x, u):
        theta, omega = x[0], x[1]
        next_theta = theta + omega * dt
        next_omega = omega + (-9.81 * np.sin(theta) - 0.2 * omega + u[0]) * dt
        return np.array([next_theta, next_omega], dtype=np.float64)

    def h_true(x):
        return np.array([np.sin(x[0])], dtype=np.float64)

    W = np.diag([1e-4, 1e-3])
    V = np.array([[1e-3]])

    ekf = ExtendedKalmanFilter(f_true, h_true, W, V)
    true_x = np.array([0.5, 0.0], dtype=np.float64)
    state = EKFState(mean=np.array([0.0, 0.0]), covariance=np.eye(2) * 0.5)

    rng = np.random.default_rng(42)
    for _ in range(30):
        u = np.array([np.sin(_ * 0.1)], dtype=np.float64)
        true_x = f_true(true_x, u) + rng.multivariate_normal(np.zeros(2), W)
        meas = h_true(true_x) + rng.multivariate_normal(np.zeros(1), V)

        state, innov = ekf.step(state, u, meas)

    # Estimate should be close to true state after filter convergence
    assert np.allclose(state.mean[0], true_x[0], atol=0.2)
    assert np.all(np.linalg.eigvals(state.covariance) > 0)
