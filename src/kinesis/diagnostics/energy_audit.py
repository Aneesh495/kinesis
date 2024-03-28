"""Mechanical energy conservation and dissipative work auditor."""

from dataclasses import dataclass
import numpy as np


@dataclass
class EnergyAuditReport:
    """Quantitative summary of mechanical energy exchange along trajectory."""
    initial_energy: float
    final_energy: float
    peak_kinetic_energy: float
    total_control_work: float
    estimated_dissipation: float
    unphysical_energy_generated: bool
    energy_conservation_residual: float


class MechanicalEnergyAuditor:
    """Audits mechanical energy balance: E_final = E_initial + W_control - W_dissipated."""

    def __init__(self, mass: float = 1.0, gravity: float = 9.81) -> None:
        self.mass = float(mass)
        self.g = float(gravity)

    def audit_trajectory(
        self,
        positions: np.ndarray,      # [T, d]
        velocities: np.ndarray,     # [T, d]
        actions: np.ndarray,        # [T - 1, m]
        dt: float = 0.05,
    ) -> EnergyAuditReport:
        """Calculate kinetic, potential, and control work balances across trajectory."""
        pos = np.asarray(positions, dtype=np.float64)
        vel = np.asarray(velocities, dtype=np.float64)
        act = np.asarray(actions, dtype=np.float64)
        T = len(pos)
        assert len(vel) == T, "Positions and velocities length mismatch"

        # Kinetic energy: 0.5 * m * ||v||^2
        speeds_sq = np.sum(vel**2, axis=-1)
        kinetic = 0.5 * self.mass * speeds_sq

        # Potential energy: m * g * height (using first coordinate as height/displacement)
        height = pos[:, 0]
        potential = self.mass * self.g * height
        total_energy = kinetic + potential

        # Control work: sum_t u_t * v_t * dt
        control_work = 0.0
        for t in range(min(len(act), T - 1)):
            # Dot product of action and velocity
            u_dim = min(len(act[t]), vel.shape[-1])
            power = np.dot(act[t][:u_dim], vel[t][:u_dim])
            control_work += power * dt

        delta_e = total_energy[-1] - total_energy[0]
        # Energy balance: delta_e = W_control - W_dissipated => W_dissipated = W_control - delta_e
        dissipated = control_work - delta_e

        # If system gained more energy than control work supplied, unphysical energy was generated!
        unphysical = bool(dissipated < -0.1)
        residual = abs(delta_e - (control_work - max(0.0, dissipated)))

        return EnergyAuditReport(
            initial_energy=float(total_energy[0]),
            final_energy=float(total_energy[-1]),
            peak_kinetic_energy=float(np.max(kinetic)),
            total_control_work=float(control_work),
            estimated_dissipation=float(dissipated),
            unphysical_energy_generated=unphysical,
            energy_conservation_residual=float(residual),
        )
