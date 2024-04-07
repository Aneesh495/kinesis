"""Unit tests for discrete Bode frequency response and margin analyzer."""

import numpy as np
import pytest

from kinesis.analytic.bode import DiscreteBodeAnalyzer, BodeResponseResult


def test_bode_first_order_lag():
    """Verify frequency response of a stable first-order lowpass system."""
    # Discrete first order lag: x[k+1] = a * x[k] + (1 - a) * u[k]
    a = 0.8
    A = np.array([[a]])
    B = np.array([[1.0 - a]])
    C = np.array([[1.0]])
    dt = 0.05

    analyzer = DiscreteBodeAnalyzer(A, B, C, dt=dt)
    res = analyzer.compute_bode(num_points=100, w_min=0.1, w_max=20.0)

    assert isinstance(res, BodeResponseResult)
    assert res.is_open_loop_stable is True
    assert len(res.frequencies) == 100
    assert len(res.magnitude_db) == 100
    assert len(res.phase_degrees) == 100

    # Low frequency DC gain should be approximately 0 dB (gain = 1.0)
    assert abs(res.magnitude_db[0]) < 0.5
    # High frequency magnitude should roll off (negative dB)
    assert res.magnitude_db[-1] < res.magnitude_db[0]
    # Phase should lag (negative degrees)
    assert res.phase_degrees[-1] < 0.0


def test_bode_second_order_system():
    """Verify frequency response of a second order harmonic oscillator."""
    dt = 0.05
    # Discretized oscillator
    omega_n = 5.0
    zeta = 0.2
    A_cont = np.array([[0.0, 1.0], [-omega_n**2, -2.0 * zeta * omega_n]])
    # Forward Euler / matrix exp approximation
    A_disc = np.eye(2) + A_cont * dt
    B_disc = np.array([[0.0], [dt]])
    C_disc = np.array([[1.0, 0.0]])

    analyzer = DiscreteBodeAnalyzer(A_disc, B_disc, C_disc, dt=dt)
    res = analyzer.compute_bode(num_points=150, w_min=0.5, w_max=25.0)

    assert isinstance(res, BodeResponseResult)
    assert res.magnitude_db.shape == (150,)
