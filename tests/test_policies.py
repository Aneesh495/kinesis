"""Tests for policy baselines, random controllers, heuristics, PPO, and imagined actor-critic."""

import pytest
import numpy as np
import torch

from kinesis.contracts import (
    ActionSpec,
    ObservationChannel,
    ObservationSpec,
)
from kinesis.models import WorldModel
from kinesis.policies import (
    CartpoleEnergyHeuristic,
    FingerTrackingHeuristic,
    ImaginedActorCritic,
    ModelFreeActorCritic,
    RandomPolicy,
    SafeFallbackPolicy,
    WalkerPeriodicHeuristic,
)


@pytest.fixture
def dummy_specs() -> tuple[ObservationSpec, ActionSpec]:
    ch = ObservationChannel(name="obs", shape=(4,))
    obs_spec = ObservationSpec(channels=(ch,))
    act_spec = ActionSpec(
        shape=(2,),
        minimum=np.array([-1.0, -1.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0], dtype=np.float32),
    )
    return obs_spec, act_spec


def test_random_and_fallback_policies(dummy_specs: tuple[ObservationSpec, ActionSpec]) -> None:
    obs_spec, act_spec = dummy_specs
    rand_policy = RandomPolicy(act_spec, seed=42)
    obs = {"obs": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)}

    act = rand_policy.act(obs)
    assert act_spec.validate(act)

    fallback = SafeFallbackPolicy(act_spec)
    fb_act = fallback.act(obs)
    assert act_spec.validate(fb_act)
    assert np.allclose(fb_act, 0.0)
    assert fallback.metrics.total_invocations == 1


def test_heuristic_controllers() -> None:
    act_spec_1 = ActionSpec(shape=(1,), minimum=np.array([-1.0]), maximum=np.array([1.0]))
    cartpole_h = CartpoleEnergyHeuristic(act_spec_1)
    act_cp = cartpole_h.act({"position": np.array([0.0, 0.5, 0.8]), "velocity": np.array([0.0, 0.2])})
    assert act_spec_1.validate(act_cp)

    act_spec_6 = ActionSpec(shape=(6,), minimum=-np.ones(6), maximum=np.ones(6))
    walker_h = WalkerPeriodicHeuristic(act_spec_6)
    act_walk = walker_h.act({})
    assert act_spec_6.validate(act_walk)

    act_spec_2 = ActionSpec(shape=(2,), minimum=-np.ones(2), maximum=np.ones(2))
    finger_h = FingerTrackingHeuristic(act_spec_2)
    act_fin = finger_h.act({})
    assert act_spec_2.validate(act_fin)


def test_model_free_actor_critic_and_gae(dummy_specs: tuple[ObservationSpec, ActionSpec]) -> None:
    obs_spec, act_spec = dummy_specs
    ppo = ModelFreeActorCritic(obs_spec, act_spec, hidden_dim=32)

    obs = {"obs": np.array([0.5, -0.2, 0.1, 0.8], dtype=np.float32)}
    act = ppo.act(obs)
    assert act_spec.validate(act)

    # Test GAE
    rewards = torch.tensor([1.0, 0.5, 0.0, 2.0])
    values = torch.tensor([1.2, 1.0, 0.8, 1.5])
    dones = torch.tensor([0.0, 0.0, 0.0, 1.0])
    next_val = torch.tensor(0.0)

    advs, rets = ppo.compute_gae(rewards, values, next_val, dones)
    assert len(advs) == 4
    assert len(rets) == 4
    assert torch.all(torch.isfinite(advs))

    # Update step
    b_obs = torch.randn(8, 4)
    b_acts = torch.randn(8, 2)
    b_logp = torch.zeros(8)
    b_rets = torch.randn(8)
    b_advs = torch.randn(8)

    res = ppo.update_step(b_obs, b_acts, b_logp, b_rets, b_advs)
    assert "policy_loss" in res
    assert "value_loss" in res


def test_imagined_actor_critic_training(dummy_specs: tuple[ObservationSpec, ActionSpec]) -> None:
    obs_spec, act_spec = dummy_specs
    wm = WorldModel(obs_dim=4, action_dim=2, embed_dim=32, deter_dim=64, stoch_dim=16, hidden_dim=64)
    ac = ImaginedActorCritic(
        world_model=wm,
        action_spec=act_spec,
        observation_spec=obs_spec,
        horizon=6,
    )

    obs = {"obs": np.array([0.2, 0.1, -0.3, 0.4], dtype=np.float32)}
    action = ac.act(obs)
    assert act_spec.validate(action)

    # Train on imagined trajectory from initial state
    B = 2
    init_s = wm.rssm.initial_state(B, device=torch.device("cpu"))
    train_res = ac.train_on_imagined_trajectories(init_s)
    assert "actor_loss" in train_res
    assert "critic_loss" in train_res
    assert np.isfinite(train_res["actor_loss"])
    assert np.isfinite(train_res["critic_loss"])
