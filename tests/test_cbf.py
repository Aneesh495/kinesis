"""Unit tests for discrete-time Control Barrier Function (CBF) filter."""

import numpy as np
import pytest

from kinesis.analytic.cbf import DiscreteControlBarrierFilter


def test_cbf_preserves_safe_action():
    """Verify that safe actions pass through the filter unperturbed."""
    dt = 0.1
    def f(x, u):
        return x + np.array([u[0] * dt, 0.0])

    # Safe set: x[0] <= 2.0  => barrier h(x) = 2.0 - x[0] >= 0
    def h(x):
        return 2.0 - x[0]

    filter_cbf = DiscreteControlBarrierFilter(f, h, barrier_decay_gamma=0.2, action_bounds=(-1.0, 1.0))
    x = np.array([0.0, 0.0])
    u_safe = np.array([0.5])

    res = filter_cbf.filter_action(x, u_safe)
    assert res.safety_violation_mitigated is False
    assert np.allclose(res.filtered_action, u_safe)


def test_cbf_intervenes_on_unsafe_action():
    """Verify that dangerous action toward boundary is overridden by CBF."""
    dt = 0.1
    def f(x, u):
        return x + np.array([u[0] * dt, 0.0])

    # Safe set: x[0] <= 1.0 => h(x) = 1.0 - x[0]
    def h(x):
        return 1.0 - x[0]

    filter_cbf = DiscreteControlBarrierFilter(f, h, barrier_decay_gamma=0.1, action_bounds=(-2.0, 2.0))
    x = np.array([0.95, 0.0])
    # Dangerous action pushing forward past 1.0
    u_dangerous = np.array([1.5])

    res = filter_cbf.filter_action(x, u_dangerous)
    assert res.safety_violation_mitigated is True
    # Action should be reduced or made negative to prevent breaching barrier
    assert res.filtered_action[0] < u_dangerous[0]
    # Prediction of next state barrier should remain non-negative
    assert res.barrier_next_pred >= 0.0
