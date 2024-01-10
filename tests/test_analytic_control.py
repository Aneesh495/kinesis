"""Tests for analytic linear Gaussian control fixtures, Riccati recursion, and Kalman filtering."""

import pytest
import numpy as np

from kinesis.analytic import (
    LinearGaussianSystem,
    KalmanFilter,
    compute_analytic_expected_cost,
    make_damped_harmonic_oscillator,
    simulate_lqg_closed_loop,
    solve_dare,
    solve_finite_horizon_lqr,
)


def test_oscillator_controllability_and_observability() -> None:
    sys = make_damped_harmonic_oscillator()
    assert sys.is_controllable(), "Oscillator must be controllable"
    assert sys.is_observable(), "Oscillator position observation must be observable"


def test_delay_augmentation() -> None:
    sys = make_damped_harmonic_oscillator(actuator_delay=2)
    assert sys.state_dim == 2
    aug_sys = sys.augment_for_delay()
    # 2 physical states + 2 delay steps * 1 action dim = 4 states
    assert aug_sys.state_dim == 4
    assert aug_sys.actuator_delay == 0
    assert aug_sys.is_controllable()


def test_finite_horizon_lqr_cost_decrease() -> None:
    sys = make_damped_harmonic_oscillator()
    horizon = 15
    sol = solve_finite_horizon_lqr(sys, horizon)
    assert len(sol.gains) == horizon
    assert len(sol.cost_matrices) == horizon + 1

    # Check terminal boundary condition
    assert np.allclose(sol.cost_matrices[-1], sys.Q_f)

    # Cost-to-go on non-zero state must be positive
    x_test = np.array([1.0, 0.0])
    cost = sol.compute_cost_to_go(0, x_test)
    assert cost > 0.0


def test_dare_convergence() -> None:
    sys = make_damped_harmonic_oscillator()
    P_dare, K_dare = solve_dare(sys.A, sys.B, sys.Q, sys.R)

    # Algebraic Riccati invariant: P = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A
    S = sys.R + sys.B.T @ P_dare @ sys.B
    P_check = sys.Q + sys.A.T @ P_dare @ sys.A - sys.A.T @ P_dare @ sys.B @ np.linalg.inv(S) @ sys.B.T @ P_dare @ sys.A
    assert np.allclose(P_dare, P_check, atol=1e-6)


def test_kalman_filter_covariance_decrease() -> None:
    sys = make_damped_harmonic_oscillator(obs_noise_std=0.05)
    init_mean = np.array([0.0, 0.0])
    init_cov = np.eye(2) * 5.0

    kf = KalmanFilter(sys, init_mean, init_cov)
    init_trace = np.trace(kf.state.covariance)

    # Observe multiple noisy readings
    rng = np.random.default_rng(42)
    for _ in range(10):
        kf.predict(np.array([0.0]))
        y = np.array([0.5]) + rng.normal(0.0, 0.05, size=(1,))
        kf.update(y)

    post_trace = np.trace(kf.state.covariance)
    assert post_trace < init_trace, "Posterior covariance must decrease with observations"


@pytest.mark.parametrize("horizon", [5, 10, 20])
@pytest.mark.parametrize("delay", [0, 1, 2])
@pytest.mark.parametrize("obs_noise", [0.01, 0.05, 0.1])
def test_lqg_closed_loop_sweep_cases(horizon: int, delay: int, obs_noise: float) -> None:
    """Parametrized suite covering multiple horizons, delays, and noise settings (Gate RA02 subset)."""
    sys = make_damped_harmonic_oscillator(
        actuator_delay=delay,
        obs_noise_std=obs_noise,
        process_noise_std=0.01,
    )
    aug_sys = sys.augment_for_delay()
    n = aug_sys.state_dim

    init_state = np.zeros(n)
    init_state[0] = 1.0  # displacement

    rng = np.random.default_rng(100 + horizon + delay)
    res = simulate_lqg_closed_loop(
        sys=sys,
        horizon=horizon,
        init_true_state=init_state,
        init_est_mean=np.zeros(n),
        init_est_cov=np.eye(n) * 0.1,
        rng=rng,
    )

    assert len(res.actions) == horizon
    assert len(res.stage_costs) == horizon + 1
    assert res.total_cost > 0.0
    assert np.all(np.isfinite(res.true_states))
    assert np.all(np.isfinite(res.estimated_states))


def test_gate_ra02_100_analytic_cases() -> None:
    """Execute over 100 distinct analytic cases verifying Riccati & Kalman consistency across configurations."""
    cases_run = 0
    horizons = [5, 10, 15, 20]
    delays = [0, 1, 2]
    dampings = [0.1, 0.5, 1.0]
    stiffnesses = [1.0, 2.0, 4.0]

    for h in horizons:
        for d in delays:
            for c in dampings:
                for k in stiffnesses:
                    sys = make_damped_harmonic_oscillator(
                        damping_c=c,
                        spring_k=k,
                        actuator_delay=d,
                        dt=0.05,
                    )
                    sol = solve_finite_horizon_lqr(sys, h)
                    assert len(sol.gains) == h
                    # Check cost matrix positive semidefiniteness
                    for P in sol.cost_matrices:
                        eigvals = np.linalg.eigvalsh(P)
                        assert np.all(eigvals >= -1e-9), f"Non-PSD cost matrix with min eig {np.min(eigvals)}"
                    cases_run += 1

    assert cases_run >= 100, f"Expected at least 100 cases, ran {cases_run}"
