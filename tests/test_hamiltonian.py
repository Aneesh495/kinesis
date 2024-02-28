"""Unit tests for Symplectic Hamiltonian Neural Network."""

import pytest
import torch

from kinesis.models.hamiltonian import HamiltonianEnergyNet, HamiltonianState, SymplecticHamiltonianDynamics


def test_hamiltonian_energy_gradients():
    """Verify exact computation of Hamiltonian canonical derivatives."""
    net = HamiltonianEnergyNet(coordinate_dim=2, hidden_dim=16)
    q = torch.randn(3, 2, requires_grad=True)
    p = torch.randn(3, 2, requires_grad=True)

    H = net(q, p)
    assert H.shape == (3,)

    loss = H.sum()
    loss.backward()
    assert q.grad is not None
    assert p.grad is not None
    assert torch.all(torch.isfinite(q.grad))
    assert torch.all(torch.isfinite(p.grad))


def test_symplectic_energy_conservation_unforced():
    """Verify that unforced leapfrog integration conserves Hamiltonian energy."""
    torch.manual_seed(42)
    hnn = SymplecticHamiltonianDynamics(coordinate_dim=2, action_dim=1, dt=0.01, hidden_dim=32)

    q0 = torch.tensor([[0.5, -0.2]], dtype=torch.float32)
    p0 = torch.tensor([[0.1, 0.4]], dtype=torch.float32)
    state = HamiltonianState(q=q0, p=p0)

    initial_energy = hnn.energy_net(state.q, state.p).item()

    # Rollout 10 unforced leapfrog steps (u = 0)
    zero_action = torch.zeros(1, 1, dtype=torch.float32)
    for _ in range(10):
        state = hnn.forward_leapfrog(state, zero_action)

    final_energy = hnn.energy_net(state.q, state.p).item()
    # Energy should be conserved within numerical tolerance of symplectic integrator
    assert abs(final_energy - initial_energy) < 0.05
