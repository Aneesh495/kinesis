"""Symplectic Hamiltonian Neural Network (HNN) latent dynamics conserving phase-space energy."""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class HamiltonianState:
    """Canonical phase-space coordinates."""
    q: torch.Tensor  # Generalized coordinates [B, d]
    p: torch.Tensor  # Generalized conjugate momenta [B, d]


class HamiltonianEnergyNet(nn.Module):
    """Scalar neural network parameterizing total energy H(q, p)."""

    def __init__(self, coordinate_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * coordinate_dim, hidden_dim),
            nn.Tanh(),  # Smooth C^2 activation required for Hamiltonian gradients
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute scalar Hamiltonian energy H(q, p) for each batch element."""
        qp = torch.cat([q, p], dim=-1)
        return self.net(qp).squeeze(-1)


class SymplecticHamiltonianDynamics(nn.Module):
    """Discrete-time symplectic integrator driven by Hamiltonian gradients and actuated torques."""

    def __init__(
        self,
        coordinate_dim: int,
        action_dim: int,
        dt: float = 0.05,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.coord_dim = coordinate_dim
        self.action_dim = action_dim
        self.dt = dt

        self.energy_net = HamiltonianEnergyNet(coordinate_dim, hidden_dim)

        # Control input actuation matrix g(q): maps action to generalized force
        self.actuation_net = nn.Sequential(
            nn.Linear(coordinate_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, coordinate_dim * action_dim),
        )

    def hamiltonian_gradients(self, q: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute dH/dq and dH/dp via torch.autograd."""
        q_grad = q.clone().detach().requires_grad_(True)
        p_grad = p.clone().detach().requires_grad_(True)

        H = self.energy_net(q_grad, p_grad).sum()
        dH_dq, dH_dp = torch.autograd.grad(H, [q_grad, p_grad], create_graph=True)
        return dH_dq, dH_dp

    def forward_leapfrog(
        self,
        state: HamiltonianState,
        action: torch.Tensor,
    ) -> HamiltonianState:
        """Execute single-step symplectic leapfrog / Störmer-Verlet integration."""
        q, p = state.q, state.p
        B = q.shape[0]

        # 1. Half-step coordinate update: q_{t+1/2} = q_t + 0.5 * dt * dH/dp(q_t, p_t)
        dH_dq0, dH_dp0 = self.hamiltonian_gradients(q, p)
        q_mid = q + 0.5 * self.dt * dH_dp0

        # 2. Generalized control force at q_{t+1/2}
        act_mat = self.actuation_net(q_mid).view(B, self.coord_dim, self.action_dim)
        control_force = torch.bmm(act_mat, action.unsqueeze(-1)).squeeze(-1)

        # 3. Full-step momentum update: p_{t+1} = p_t - dt * dH/dq(q_mid, p_t) + dt * control_force
        dH_dq_mid, _ = self.hamiltonian_gradients(q_mid, p)
        p_next = p - self.dt * dH_dq_mid + self.dt * control_force

        # 4. Final half-step coordinate update: q_{t+1} = q_{t+1/2} + 0.5 * dt * dH/dp(q_mid, p_next)
        _, dH_dp_next = self.hamiltonian_gradients(q_mid, p_next)
        q_next = q_mid + 0.5 * self.dt * dH_dp_next

        return HamiltonianState(q=q_next, p=p_next)
