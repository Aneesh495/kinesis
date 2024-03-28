"""Unit tests for Mechanical Energy auditor."""

import numpy as np
import pytest

from kinesis.diagnostics.energy_audit import MechanicalEnergyAuditor


def test_energy_audit_passive_dissipation():
    """Verify that damped passive trajectory correctly reports positive dissipation."""
    auditor = MechanicalEnergyAuditor(mass=1.0, gravity=9.81)

    # Ball dropped from height 10.0 to 5.0, slowing down due to air resistance
    T = 20
    pos = np.linspace(10.0, 5.0, T)[:, None]
    vel = np.linspace(-2.0, -0.5, T)[:, None]
    actions = np.zeros((T - 1, 1))

    report = auditor.audit_trajectory(pos, vel, actions)
    assert report.unphysical_energy_generated is False
    assert report.final_energy < report.initial_energy
    assert report.estimated_dissipation > 0.0


def test_energy_audit_unphysical_flag():
    """Verify that spontaneous energy increase without control work is flagged as unphysical."""
    auditor = MechanicalEnergyAuditor(mass=1.0, gravity=9.81)

    # Trajectory spontaneously gaining speed and height without control work
    T = 10
    pos = np.linspace(1.0, 10.0, T)[:, None]
    vel = np.linspace(0.0, 5.0, T)[:, None]
    actions = np.zeros((T - 1, 1))  # No work done

    report = auditor.audit_trajectory(pos, vel, actions)
    assert report.unphysical_energy_generated is True
    assert report.estimated_dissipation < 0.0
