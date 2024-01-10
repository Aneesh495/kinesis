"""Analytic linear Gaussian control fixtures, Riccati solvers, and Kalman filters."""

from kinesis.analytic.system import LinearGaussianSystem, make_damped_harmonic_oscillator
from kinesis.analytic.riccati import LQRTrajectorySolution, solve_dare, solve_finite_horizon_lqr
from kinesis.analytic.kalman import KalmanFilter, KalmanState
from kinesis.analytic.lqg import (
    LQGSimulationResult,
    compute_analytic_expected_cost,
    simulate_lqg_closed_loop,
)

__all__ = [
    "LinearGaussianSystem",
    "make_damped_harmonic_oscillator",
    "LQRTrajectorySolution",
    "solve_dare",
    "solve_finite_horizon_lqr",
    "KalmanFilter",
    "KalmanState",
    "LQGSimulationResult",
    "compute_analytic_expected_cost",
    "simulate_lqg_closed_loop",
]
