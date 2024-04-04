"""Unit tests for Differential Dynamic Programming (DDP)."""

import numpy as np
import pytest

from kinesis.analytic.ddp import DifferentialDynamicProgramming


def test_ddp_nonlinear_optimization():
    """Verify that full DDP with dynamics Hessians reduces trajectory cost."""
    dt = 0.1
    # Nonlinear dynamics with quadratic velocity drag
    def f(x, u):
        pos, vel = x[0], x[1]
        next_pos = pos + vel * dt
        next_vel = vel + (-0.1 * vel * abs(vel) + u[0]) * dt
        return np.array([next_pos, next_vel], dtype=np.float64)

    def cost_stage(x, u):
        return float(x[0]**2 + 0.1 * x[1]**2 + 0.05 * u[0]**2)

    def cost_terminal(x):
        return float(10.0 * x[0]**2 + x[1]**2)

    ddp = DifferentialDynamicProgramming(
        f=f,
        cost_stage=cost_stage,
        cost_terminal=cost_terminal,
        horizon=10,
        state_dim=2,
        action_dim=1,
        max_iters=15,
    )

    x0 = np.array([2.0, 0.0])
    init_actions = np.zeros((10, 1))
    _, init_cost = ddp.rollout(x0, init_actions)

    res = ddp.optimize(x0, init_actions)
    assert res.total_cost < init_cost
    assert res.actions.shape == (10, 1)
    assert res.states.shape == (11, 2)
    assert np.linalg.norm(res.states[-1]) < np.linalg.norm(x0)
