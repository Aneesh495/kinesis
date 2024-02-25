"""Unit tests for discrete-time H-infinity robust control synthesizer."""

import numpy as np
import pytest

from kinesis.analytic.h_infinity import DiscreteHInfinityController


def test_h_infinity_convergence_and_stability():
    """Verify H-infinity controller stabilizes discrete system and produces valid Riccati solution."""
    dt = 0.1
    A = np.array([[1.0, dt], [0.0, 0.95]], dtype=np.float64)
    B1 = np.array([[0.0], [0.1]], dtype=np.float64)  # Disturbance
    B2 = np.array([[0.0], [dt]], dtype=np.float64)   # Control
    Q = np.diag([1.0, 0.1])
    R = np.eye(1) * 0.1

    controller = DiscreteHInfinityController(A, B1, B2, Q, R, gamma=5.0)
    res = controller.solve()

    assert res.converged is True
    assert res.K_u.shape == (1, 2)
    assert res.K_w.shape == (1, 2)
    assert np.all(np.linalg.eigvals(res.P) > 0)

    # Closed-loop matrix A - B2 * Ku should have eigenvalues strictly inside the unit circle
    A_cl = A - B2 @ res.K_u
    eigs = np.abs(np.linalg.eigvals(A_cl))
    assert np.all(eigs < 1.0)


def test_h_infinity_infeasibility_under_small_gamma():
    """Verify that setting gamma too small reports non-convergence."""
    A = np.array([[1.5]], dtype=np.float64)
    B1 = np.array([[1.0]], dtype=np.float64)
    B2 = np.array([[1.0]], dtype=np.float64)
    Q = np.array([[1.0]], dtype=np.float64)
    R = np.array([[1.0]], dtype=np.float64)

    # With gamma = 0.1, robust L2 gain bound is impossible
    controller = DiscreteHInfinityController(A, B1, B2, Q, R, gamma=0.1)
    res = controller.solve()

    assert res.converged is False
