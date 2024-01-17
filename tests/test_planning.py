"""Tests for uncertainty ensemble disagreement, CEM and MPPI latent planning, and policy execution."""

import pytest
import numpy as np
import torch

from kinesis.contracts import (
    ActionSpec,
    ObservationChannel,
    ObservationSpec,
)
from kinesis.models import WorldModel
from kinesis.planning import (
    CrossEntropyMethodPlanner,
    LatentPlannerPolicy,
    ModelPredictivePathIntegralPlanner,
    PlanningObjectiveWeights,
    TrajectoryObjective,
)
from kinesis.uncertainty import WorldModelEnsemble


@pytest.fixture
def dummy_setup() -> tuple[ObservationSpec, ActionSpec, WorldModelEnsemble]:
    ch = ObservationChannel(name="q", shape=(4,))
    obs_spec = ObservationSpec(channels=(ch,))
    act_spec = ActionSpec(
        shape=(2,),
        minimum=np.array([-1.0, -1.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0], dtype=np.float32),
    )

    # 3 independently initialized world models for committee
    models = [
        WorldModel(obs_dim=4, action_dim=2, embed_dim=16, deter_dim=32, stoch_dim=8, hidden_dim=32)
        for _ in range(3)
    ]
    ensemble = WorldModelEnsemble(models=models)
    return obs_spec, act_spec, ensemble


def test_ensemble_disagreement_calculation(dummy_setup: tuple[ObservationSpec, ActionSpec, WorldModelEnsemble]) -> None:
    _, act_spec, ensemble = dummy_setup
    B, H = 4, 5
    actions = torch.randn(B, H, 2)
    init_states = [m.rssm.initial_state(B, torch.device("cpu")) for m in ensemble.models]

    rews, discs, costs, disags, feats = ensemble.imagine_rollout_with_disagreement(init_states, actions)
    assert rews.shape == (B, H)
    assert disags.shape == (B, H)
    assert torch.all(disags >= 0.0), "Ensemble variance must be non-negative"


def test_trajectory_objective_penalization() -> None:
    obj = TrajectoryObjective(weights=PlanningObjectiveWeights(uncertainty_beta=2.0, constraint_weight=5.0))
    B, H = 2, 4
    rews = torch.ones(B, H) * 2.0
    discs = torch.ones(B, H)
    costs = torch.zeros(B, H)
    costs[1, :] = 1.0  # Candidate 1 has constraint violations
    disags = torch.zeros(B, H)
    disags[1, :] = 1.5  # Candidate 1 has high disagreement
    actions = torch.zeros(B, H, 1)

    scores = obj.score_trajectories(rews, discs, costs, disags, actions)
    assert scores[0] > scores[1], "Candidate with higher uncertainty and violations must have lower score"


def test_cem_planner_optimization(dummy_setup: tuple[ObservationSpec, ActionSpec, WorldModelEnsemble]) -> None:
    _, act_spec, ensemble = dummy_setup
    cem = CrossEntropyMethodPlanner(
        ensemble=ensemble,
        action_spec=act_spec,
        horizon=6,
        num_candidates=20,
        num_iterations=2,
    )
    init_states = [m.rssm.initial_state(1, torch.device("cpu")) for m in ensemble.models]
    res = cem.plan(init_states)

    assert act_spec.validate(res.selected_action)
    assert res.best_action_sequence.shape == (6, 2)
    assert np.all(res.best_action_sequence >= act_spec.minimum)
    assert np.all(res.best_action_sequence <= act_spec.maximum)
    assert res.iterations_completed == 2


def test_mppi_planner_optimization(dummy_setup: tuple[ObservationSpec, ActionSpec, WorldModelEnsemble]) -> None:
    _, act_spec, ensemble = dummy_setup
    mppi = ModelPredictivePathIntegralPlanner(
        ensemble=ensemble,
        action_spec=act_spec,
        horizon=6,
        num_candidates=20,
        temperature=0.5,
    )
    init_states = [m.rssm.initial_state(1, torch.device("cpu")) for m in ensemble.models]
    res = mppi.plan(init_states)

    assert act_spec.validate(res.selected_action)
    assert res.best_action_sequence.shape == (6, 2)


def test_latent_planner_policy_execution_and_fallback(
    dummy_setup: tuple[ObservationSpec, ActionSpec, WorldModelEnsemble]
) -> None:
    obs_spec, act_spec, ensemble = dummy_setup
    cem = CrossEntropyMethodPlanner(
        ensemble=ensemble,
        action_spec=act_spec,
        horizon=4,
        num_candidates=10,
        num_iterations=1,
    )
    policy = LatentPlannerPolicy(
        planner=cem,
        observation_spec=obs_spec,
        action_spec=act_spec,
        timeout_sec=5.0,
    )

    obs = {"q": np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)}
    action = policy.act(obs)
    assert act_spec.validate(action)
    assert policy.fallback_policy.metrics.total_invocations == 0

    # Test timeout trigger fallback with ultra-tight deadline
    tight_policy = LatentPlannerPolicy(
        planner=cem,
        observation_spec=obs_spec,
        action_spec=act_spec,
        timeout_sec=0.000001,  # Impractical deadline forcing fallback
    )
    action_fb = tight_policy.act(obs)
    assert act_spec.validate(action_fb)
    assert tight_policy.fallback_policy.metrics.timeout_triggers > 0
