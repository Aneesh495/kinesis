"""Unit tests for discrete-time Lyapunov stability certification."""

import numpy as np
import pytest

from kinesis.analytic.lyapunov import DiscreteLyapunovCertifier, LyapunovCertificateResult


def test_lyapunov_stable_system():
    """Verify Lyapunov certification for a strictly asymptotically stable linear system."""
    # Stable matrix with eigenvalues well inside unit circle: 0.8, 0.5
    A = np.array([[0.8, 0.1], [0.0, 0.5]], dtype=np.float64)
    Q = np.eye(2, dtype=np.float64)

    certifier = DiscreteLyapunovCertifier(A, Q)

    # Check P is positive definite
    eigvals = np.linalg.eigvalsh(certifier.P)
    assert np.all(eigvals > 0.0)

    # Check Lyapunov equation: A.T @ P @ A - P = -Q
    diff = A.T @ certifier.P @ A - certifier.P + Q
    np.testing.assert_allclose(diff, np.zeros((2, 2)), atol=1e-10)

    # Simulate trajectory
    x0 = np.array([2.0, -1.5], dtype=np.float64)
    traj = [x0]
    curr = x0
    for _ in range(15):
        curr = A @ curr
        traj.append(curr)

    result = certifier.certify_trajectory(traj)
    assert isinstance(result, LyapunovCertificateResult)
    assert result.is_certified_stable is True
    assert result.strict_monotonic_decrease is True
    assert result.final_lyapunov_value < result.initial_lyapunov_value
    assert result.max_decay_ratio < 1.0


def test_lyapunov_unstable_system():
    """Verify Lyapunov certification rejects an unstable trajectory."""
    # Stable certifier setup
    A_stable = np.array([[0.8, 0.0], [0.0, 0.8]], dtype=np.float64)
    certifier = DiscreteLyapunovCertifier(A_stable)

    # Trajectory that grows exponentially
    x0 = np.array([1.0, 1.0], dtype=np.float64)
    traj = [x0 * (1.2 ** t) for t in range(5)]

    result = certifier.certify_trajectory(traj)
    assert result.is_certified_stable is False
    assert result.strict_monotonic_decrease is False
    assert result.final_lyapunov_value > result.initial_lyapunov_value
