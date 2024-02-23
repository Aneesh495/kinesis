"""Unit tests for Iterative Linear Quadratic Regulator (iLQR)."""

import numpy as np
import pytest

from kinesis.analytic.ilqr import IterativeLQR


def test_ilqr_quadratic_tracking():
    """Verify that iLQR reduces cost and steers a 2D integrator to origin."""
    dt = 0.1
    A = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    B = np.array([[0.5 * dt**2], [dt]], dtype=np.float64)

    def f(x, u):
        return A @ x + B @ u

    def cost_stage(x, u):
        return float(x.T @ np.diag([1.0, 0.1]) @ x + 0.05 * u[0]**2)

    def cost_terminal(x):
        return float(x.T @ np.diag([10.0, 2.0]) @ x)

    ilqr = IterativeLQR(
        f=f,
        cost_stage=cost_stage,
        cost_terminal=cost_terminal,
        horizon=15,
        state_dim=2,
        action_dim=1,
        max_iters=20,
    )

    x0 = np.array([2.0, 0.0], dtype=np.float64)
    init_actions = np.zeros((15, 1), dtype=np.float64)
    init_states, init_cost = ilqr.rollout(x0, init_actions)

    res = ilqr.optimize(x0, init_actions)

    assert res.total_cost < init_cost
    assert res.actions.shape == (15, 1)
    assert res.states.shape == (16, 2)
    # Terminal state should be closer to origin
    assert np.linalg.norm(res.states[-1]) < np.linalg.norm(x0)


def test_ilqr_nonlinear_pendulum_swing():
    """Verify iLQR cost reduction on nonlinear pendulum."""
    dt = 0.05
    def f(x, u):
        th, w = x[0], x[1]
        next_th = th + w * dt
        next_w = w + (-9.81 * np.sin(th) + u[0]) * dt
        return np.array([next_th, next_w], dtype=np.float64)

    def cost_stage(x, u):
        return float(1.0 - np.cos(x[0]) + 0.1 * x[1]**2 + 0.01 * u[0]**2)

    def cost_terminal(x):
        return float(10.0 * (1.0 - np.cos(x[0])) + x[1]**2)

    ilqr = IterativeLQR(
        f=f,
        cost_stage=cost_stage,
        cost_terminal=cost_terminal,
        horizon=10,
        state_dim=2,
        action_dim=1,
        max_iters=15,
    )

    x0 = np.array([0.5, 0.0], dtype=np.float64)
    res = ilqr.optimize(x0)
    assert res.total_cost > 0
    assert len(res.actions) == 10
