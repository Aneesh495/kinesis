"""Unit tests for Runtime Safety Shield."""

import numpy as np
import pytest

from kinesis.contracts.action import ActionSpec
from kinesis.policies.fallback import SafeFallbackPolicy
from kinesis.safety.shield import RuntimeSafetyShield


def test_safety_shield_passthrough_when_safe():
    """Verify that safe proposed actions pass through unaltered."""
    action_spec = ActionSpec(shape=(1,), minimum=np.array([-1.0], dtype=np.float32), maximum=np.array([1.0], dtype=np.float32))
    fallback = SafeFallbackPolicy(action_spec)
    shield = RuntimeSafetyShield(fallback, action_spec, max_allowed_uncertainty=1.0, min_safety_barrier_margin=0.1)

    obs = {"position": np.array([0.5])}
    prop_act = np.array([0.4], dtype=np.float32)

    decision = shield.evaluate_and_filter(obs, prop_act, estimated_uncertainty=0.2, barrier_margin=0.8)
    assert decision.intervened is False
    assert np.allclose(decision.executed_action, prop_act)
    assert shield.total_interventions == 0


def test_safety_shield_intervenes_on_high_uncertainty():
    """Verify that excessive epistemic uncertainty triggers fallback."""
    action_spec = ActionSpec(shape=(1,), minimum=np.array([-1.0], dtype=np.float32), maximum=np.array([1.0], dtype=np.float32))
    fallback = SafeFallbackPolicy(action_spec)
    shield = RuntimeSafetyShield(fallback, action_spec, max_allowed_uncertainty=1.0)

    obs = {"position": np.array([0.5])}
    prop_act = np.array([0.9], dtype=np.float32)

    decision = shield.evaluate_and_filter(obs, prop_act, estimated_uncertainty=2.5, barrier_margin=0.8)
    assert decision.intervened is True
    assert "uncertainty_exceeded" in decision.trigger_reason
    assert shield.total_interventions == 1
    # Executed action should be fallback damping action
    assert np.allclose(decision.executed_action, np.zeros(1))
