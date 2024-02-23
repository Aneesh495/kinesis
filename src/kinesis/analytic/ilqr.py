"""Iterative Linear Quadratic Regulator (iLQR) for nonlinear trajectory optimization."""

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class iLQRResult:
    """Result of iLQR trajectory optimization."""
    states: np.ndarray  # (T + 1, n)
    actions: np.ndarray  # (T, m)
    feedback_gains: np.ndarray  # (T, m, n)
    feedforward_gains: np.ndarray  # (T, m)
    total_cost: float
    iterations: int
    converged: bool


class IterativeLQR:
    """Iterative Linear Quadratic Regulator (iLQR) algorithm."""

    def __init__(
        self,
        f: Callable[[np.ndarray, np.ndarray], np.ndarray],
        cost_stage: Callable[[np.ndarray, np.ndarray], float],
        cost_terminal: Callable[[np.ndarray], float],
        horizon: int,
        state_dim: int,
        action_dim: int,
        max_iters: int = 50,
        tol: float = 1e-4,
        eps: float = 1e-5,
    ) -> None:
        self.f = f
        self.cost_stage = cost_stage
        self.cost_terminal = cost_terminal
        self.horizon = horizon
        self.n = state_dim
        self.m = action_dim
        self.max_iters = max_iters
        self.tol = tol
        self.eps = eps

    def _jacobian_f(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute Jacobians f_x and f_u via central differences."""
        fx = np.zeros((self.n, self.n), dtype=np.float64)
        fu = np.zeros((self.n, self.m), dtype=np.float64)

        for i in range(self.n):
            dx = np.zeros(self.n, dtype=np.float64)
            dx[i] = self.eps
            fx[:, i] = (self.f(x + dx, u) - self.f(x - dx, u)) / (2.0 * self.eps)

        for j in range(self.m):
            du = np.zeros(self.m, dtype=np.float64)
            du[j] = self.eps
            fu[:, j] = (self.f(x, u + du) - self.f(x, u - du)) / (2.0 * self.eps)

        return fx, fu

    def _cost_derivatives_stage(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradients and Hessians of stage cost via finite differences."""
        lx = np.zeros(self.n, dtype=np.float64)
        lu = np.zeros(self.m, dtype=np.float64)
        lxx = np.zeros((self.n, self.n), dtype=np.float64)
        luu = np.zeros((self.m, self.m), dtype=np.float64)
        lux = np.zeros((self.m, self.n), dtype=np.float64)

        c0 = self.cost_stage(x, u)

        # Gradients
        for i in range(self.n):
            dx = np.zeros(self.n, dtype=np.float64)
            dx[i] = self.eps
            lx[i] = (self.cost_stage(x + dx, u) - self.cost_stage(x - dx, u)) / (2.0 * self.eps)

        for j in range(self.m):
            du = np.zeros(self.m, dtype=np.float64)
            du[j] = self.eps
            lu[j] = (self.cost_stage(x, u + du) - self.cost_stage(x, u - du)) / (2.0 * self.eps)

        # Hessians
        for i in range(self.n):
            for j in range(self.n):
                dxi = np.zeros(self.n, dtype=np.float64)
                dxj = np.zeros(self.n, dtype=np.float64)
                dxi[i] = self.eps
                dxj[j] = self.eps
                c_pp = self.cost_stage(x + dxi + dxj, u)
                c_pm = self.cost_stage(x + dxi - dxj, u)
                c_mp = self.cost_stage(x - dxi + dxj, u)
                c_mm = self.cost_stage(x - dxi - dxj, u)
                lxx[i, j] = (c_pp - c_pm - c_mp + c_mm) / (4.0 * self.eps**2)

        for i in range(self.m):
            for j in range(self.m):
                dui = np.zeros(self.m, dtype=np.float64)
                duj = np.zeros(self.m, dtype=np.float64)
                dui[i] = self.eps
                duj[j] = self.eps
                c_pp = self.cost_stage(x, u + dui + duj)
                c_pm = self.cost_stage(x, u + dui - duj)
                c_mp = self.cost_stage(x, u - dui + duj)
                c_mm = self.cost_stage(x, u - dui - duj)
                luu[i, j] = (c_pp - c_pm - c_mp + c_mm) / (4.0 * self.eps**2)

        # Regularize Hessians to ensure positive semi-definiteness
        lxx = 0.5 * (lxx + lxx.T)
        luu = 0.5 * (luu + luu.T) + np.eye(self.m) * 1e-4

        return lx, lu, lxx, luu, lux

    def _cost_derivatives_terminal(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute terminal cost gradient and Hessian."""
        vx = np.zeros(self.n, dtype=np.float64)
        vxx = np.zeros((self.n, self.n), dtype=np.float64)

        for i in range(self.n):
            dx = np.zeros(self.n, dtype=np.float64)
            dx[i] = self.eps
            vx[i] = (self.cost_terminal(x + dx) - self.cost_terminal(x - dx)) / (2.0 * self.eps)

        for i in range(self.n):
            for j in range(self.n):
                dxi = np.zeros(self.n, dtype=np.float64)
                dxj = np.zeros(self.n, dtype=np.float64)
                dxi[i] = self.eps
                dxj[j] = self.eps
                c_pp = self.cost_terminal(x + dxi + dxj)
                c_pm = self.cost_terminal(x + dxi - dxj)
                c_mp = self.cost_terminal(x - dxi + dxj)
                c_mm = self.cost_terminal(x - dxi - dxj)
                vxx[i, j] = (c_pp - c_pm - c_mp + c_mm) / (4.0 * self.eps**2)

        vxx = 0.5 * (vxx + vxx.T)
        return vx, vxx

    def rollout(self, x0: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, float]:
        """Rollout dynamics forward given initial state and action sequence."""
        states = np.zeros((self.horizon + 1, self.n), dtype=np.float64)
        states[0] = x0
        cost = 0.0

        for t in range(self.horizon):
            cost += self.cost_stage(states[t], actions[t])
            states[t + 1] = self.f(states[t], actions[t])

        cost += self.cost_terminal(states[-1])
        return states, cost

    def optimize(self, x0: np.ndarray, initial_actions: np.ndarray | None = None) -> iLQRResult:
        """Run iLQR backward-forward optimization iterations."""
        if initial_actions is None:
            actions = np.zeros((self.horizon, self.m), dtype=np.float64)
        else:
            actions = np.array(initial_actions, dtype=np.float64)

        states, total_cost = self.rollout(x0, actions)

        k_gains = np.zeros((self.horizon, self.m), dtype=np.float64)
        K_gains = np.zeros((self.horizon, self.m, self.n), dtype=np.float64)
        mu = 1e-4
        converged = False

        for it in range(self.max_iters):
            # Backward pass
            Vx, Vxx = self._cost_derivatives_terminal(states[-1])
            diverged = False

            for t in reversed(range(self.horizon)):
                fx, fu = self._jacobian_f(states[t], actions[t])
                lx, lu, lxx, luu, lux = self._cost_derivatives_stage(states[t], actions[t])

                Qx = lx + fx.T @ Vx
                Qu = lu + fu.T @ Vx
                Qxx = lxx + fx.T @ Vxx @ fx
                Quu = luu + fu.T @ Vxx @ fu + mu * np.eye(self.m)
                Qux = lux + fu.T @ Vxx @ fx

                # Invert Quu
                try:
                    Quu_inv = np.linalg.inv(Quu)
                except np.linalg.LinAlgError:
                    diverged = True
                    break

                kt = -Quu_inv @ Qu
                Kt = -Quu_inv @ Qux

                k_gains[t] = kt
                K_gains[t] = Kt

                # Update Value function
                Vx = Qx + Kt.T @ Quu @ kt + Kt.T @ Qu + Qux.T @ kt
                Vxx = Qxx + Kt.T @ Quu @ Kt + Kt.T @ Qux + Qux.T @ Kt
                Vxx = 0.5 * (Vxx + Vxx.T)

            if diverged:
                mu *= 5.0
                continue

            # Forward pass with backtracking line search
            alpha = 1.0
            accepted = False
            best_cost = total_cost
            best_states = states
            best_actions = actions

            for _ in range(6):
                new_states = np.zeros_like(states)
                new_actions = np.zeros_like(actions)
                new_states[0] = x0

                for t in range(self.horizon):
                    new_actions[t] = actions[t] + alpha * k_gains[t] + K_gains[t] @ (new_states[t] - states[t])
                    new_states[t + 1] = self.f(new_states[t], new_actions[t])

                new_cost = sum(self.cost_stage(new_states[t], new_actions[t]) for t in range(self.horizon))
                new_cost += self.cost_terminal(new_states[-1])

                if new_cost < total_cost:
                    accepted = True
                    best_cost = new_cost
                    best_states = new_states
                    best_actions = new_actions
                    break
                alpha *= 0.5

            if accepted:
                cost_diff = abs(total_cost - best_cost)
                total_cost = best_cost
                states = best_states
                actions = best_actions
                mu = max(1e-6, mu * 0.5)
                if cost_diff < self.tol:
                    converged = True
                    break
            else:
                mu *= 2.0

        return iLQRResult(
            states=states,
            actions=actions,
            feedback_gains=K_gains,
            feedforward_gains=k_gains,
            total_cost=total_cost,
            iterations=it + 1,
            converged=converged,
        )
