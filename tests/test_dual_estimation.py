"""Unit tests for Dual Unscented Kalman Filter concurrent estimation."""

import numpy as np
import pytest

from kinesis.adaptation.dual_estimation import DualUnscentedKalmanFilter


def test_dual_ukf_state_and_parameter_tracking():
    """Verify Dual UKF tracks state and identifies parameter."""
    dt = 0.05
    # True system: dx = (w * x + u) * dt
    # where w is unknown damping parameter
    w_true = -0.5

    def f_param(x, u, w):
        return x + (w[0] * x[0] + u[0]) * dt

    def h_param(x, w):
        return np.array([x[0]])

    dual = DualUnscentedKalmanFilter(
        f_param=f_param,
        h_param=h_param,
        state_dim=1,
        param_dim=1,
        obs_dim=1,
        state_process_cov=np.array([[1e-4]]),
        param_process_cov=np.array([[1e-4]]),
        state_meas_cov=np.array([[1e-3]]),
        param_meas_cov=np.array([[1e-3]]),
    )

    state = dual.initial_state(
        x0_mean=np.array([1.0]),
        x0_cov=np.array([[0.5]]),
        w0_mean=np.array([-0.1]),  # Initial guess
        w0_cov=np.array([[1.0]]),
    )

    true_x = np.array([1.0])
    for step in range(40):
        u = np.array([np.sin(step * 0.2)])
        true_x = f_param(true_x, u, np.array([w_true]))
        meas = true_x + np.random.normal(scale=0.01, size=1)

        state, telem = dual.step(state, u, meas)

    # State estimate should be close to true state
    assert np.isclose(state.state_est.mean[0], true_x[0], atol=0.2)
    # Parameter estimate should move toward -0.5
    assert state.param_est.mean[0] < -0.2
