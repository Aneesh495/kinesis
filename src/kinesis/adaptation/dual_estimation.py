"""Dual Unscented Kalman Filter for concurrent latent state and physical parameter identification."""

from dataclasses import dataclass
from typing import Callable
import numpy as np

from kinesis.analytic.ukf import UKFState, UnscentedKalmanFilter


@dataclass
class DualUKFState:
    """Joint container for state filter and parameter filter estimates."""
    state_est: UKFState
    param_est: UKFState


@dataclass
class DualUKFTelemetry:
    """Telemetry from dual estimation step."""
    state_innovation: np.ndarray
    param_innovation: np.ndarray
    state_trace: float
    param_trace: float


class DualUnscentedKalmanFilter:
    """Interleaved dual UKF estimating hidden state x and physical parameter vector w concurrently."""

    def __init__(
        self,
        f_param: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],  # f(x, u, w)
        h_param: Callable[[np.ndarray, np.ndarray], np.ndarray],              # h(x, w)
        state_dim: int,
        param_dim: int,
        obs_dim: int,
        state_process_cov: np.ndarray,
        param_process_cov: np.ndarray,
        state_meas_cov: np.ndarray,
        param_meas_cov: np.ndarray,
    ) -> None:
        self.f_param = f_param
        self.h_param = h_param
        self.n = state_dim
        self.p = param_dim
        self.m = obs_dim

        self.W_x = np.asarray(state_process_cov, dtype=np.float64)
        self.W_w = np.asarray(param_process_cov, dtype=np.float64)
        self.V_x = np.asarray(state_meas_cov, dtype=np.float64)
        self.V_w = np.asarray(param_meas_cov, dtype=np.float64)

    def initial_state(
        self,
        x0_mean: np.ndarray,
        x0_cov: np.ndarray,
        w0_mean: np.ndarray,
        w0_cov: np.ndarray,
    ) -> DualUKFState:
        """Initialize dual filter states."""
        s_x = UKFState(mean=np.asarray(x0_mean, dtype=np.float64), covariance=np.asarray(x0_cov, dtype=np.float64))
        s_w = UKFState(mean=np.asarray(w0_mean, dtype=np.float64), covariance=np.asarray(w0_cov, dtype=np.float64))
        return DualUKFState(state_est=s_x, param_est=s_w)

    def step(
        self,
        dual_state: DualUKFState,
        u_prev: np.ndarray,
        y_curr: np.ndarray,
    ) -> tuple[DualUKFState, DualUKFTelemetry]:
        """Interleaved state and parameter estimation step."""
        curr_w = dual_state.param_est.mean
        curr_x = dual_state.state_est.mean

        # 1. State Filter Step using current parameter estimate w
        def f_state(x, u):
            return self.f_param(x, u, curr_w)

        def h_state(x):
            return self.h_param(x, curr_w)

        filter_x = UnscentedKalmanFilter(f_state, h_state, self.W_x, self.V_x)
        next_state_x, innov_x = filter_x.step(dual_state.state_est, u_prev, y_curr)

        # 2. Parameter Filter Step: random walk on parameters w_{k+1} = w_k + r_k
        # Observation is expected measurement d_k = h(f(x_{k-1}, u_{k-1}, w), w)
        def f_param_step(w, u):
            return w  # Stationary parameter model

        def h_param_step(w):
            x_pred = self.f_param(curr_x, u_prev, w)
            return self.h_param(x_pred, w)

        filter_w = UnscentedKalmanFilter(f_param_step, h_param_step, self.W_w, self.V_w)
        next_state_w, innov_w = filter_w.step(dual_state.param_est, np.zeros(1), y_curr)

        telemetry = DualUKFTelemetry(
            state_innovation=innov_x,
            param_innovation=innov_w,
            state_trace=float(np.trace(next_state_x.covariance)),
            param_trace=float(np.trace(next_state_w.covariance)),
        )

        return DualUKFState(state_est=next_state_x, param_est=next_state_w), telemetry
