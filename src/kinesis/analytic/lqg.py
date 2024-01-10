"""Linear-Quadratic-Gaussian (LQG) closed-loop simulator and analytic cost calculator."""

from dataclasses import dataclass
import numpy as np

from kinesis.analytic.system import LinearGaussianSystem
from kinesis.analytic.riccati import solve_finite_horizon_lqr
from kinesis.analytic.kalman import KalmanFilter


@dataclass
class LQGSimulationResult:
    """Detailed trace of an LQG closed-loop execution."""
    true_states: np.ndarray  # [N+1, state_dim]
    estimated_states: np.ndarray  # [N+1, state_dim]
    observations: np.ndarray  # [N+1, obs_dim]
    actions: np.ndarray  # [N, action_dim]
    stage_costs: np.ndarray  # [N+1]
    total_cost: float


def simulate_lqg_closed_loop(
    sys: LinearGaussianSystem,
    horizon: int,
    init_true_state: np.ndarray,
    init_est_mean: np.ndarray,
    init_est_cov: np.ndarray,
    rng: np.random.Generator,
) -> LQGSimulationResult:
    """Execute genuine closed-loop simulation using separation principle (Kalman + LQR)."""
    aug_sys = sys.augment_for_delay()
    n = aug_sys.state_dim
    m = aug_sys.action_dim
    p = aug_sys.obs_dim

    # Solve optimal Riccati gains
    lqr_sol = solve_finite_horizon_lqr(sys, horizon)

    # Initialize Kalman filter
    kf = KalmanFilter(sys, init_est_mean, init_est_cov)

    true_states = [init_true_state.copy()]
    est_states = []
    observations = []
    actions = []
    stage_costs = []

    # Initial observation
    v_0 = rng.multivariate_normal(np.zeros(p), aug_sys.V)
    y_0 = aug_sys.C @ init_true_state + v_0
    observations.append(y_0)

    # Initial filter update
    kf.update(y_0)
    est_states.append(kf.state.mean.copy())

    cur_x = init_true_state.copy()

    for t in range(horizon):
        # Initial or current stage cost
        cur_u = lqr_sol.get_action(t, kf.state.mean)
        actions.append(cur_u)

        cost_t = float(cur_x.T @ aug_sys.Q @ cur_x + cur_u.T @ aug_sys.R @ cur_u)
        stage_costs.append(cost_t)

        # Plant dynamics step
        w_t = rng.multivariate_normal(np.zeros(n), aug_sys.W)
        next_x = aug_sys.A @ cur_x + aug_sys.B @ cur_u + w_t
        true_states.append(next_x.copy())

        # Noisy observation
        v_next = rng.multivariate_normal(np.zeros(p), aug_sys.V)
        next_y = aug_sys.C @ next_x + v_next
        observations.append(next_y)

        # Kalman predict and update
        kf.predict(cur_u)
        kf.update(next_y)
        est_states.append(kf.state.mean.copy())

        cur_x = next_x

    # Terminal cost
    terminal_cost = float(cur_x.T @ aug_sys.Q_f @ cur_x)
    stage_costs.append(terminal_cost)

    return LQGSimulationResult(
        true_states=np.array(true_states),
        estimated_states=np.array(est_states),
        observations=np.array(observations),
        actions=np.array(actions),
        stage_costs=np.array(stage_costs),
        total_cost=float(np.sum(stage_costs)),
    )


def compute_analytic_expected_cost(
    sys: LinearGaussianSystem,
    horizon: int,
    init_mean: np.ndarray,
    init_cov: np.ndarray,
) -> float:
    """Compute the theoretical expected cost of the optimal LQG controller."""
    aug_sys = sys.augment_for_delay()
    lqr_sol = solve_finite_horizon_lqr(sys, horizon)

    # Base state cost: x_0^T P_0 x_0 + Tr(P_0 * Sigma_0)
    P_0 = lqr_sol.cost_matrices[0]
    expected_cost = float(init_mean.T @ P_0 @ init_mean + np.trace(P_0 @ init_cov))

    # Add stochastic process noise contribution Tr(P_{t+1} W)
    for t in range(horizon):
        P_next = lqr_sol.cost_matrices[t + 1]
        expected_cost += float(np.trace(P_next @ aug_sys.W))

    return expected_cost
