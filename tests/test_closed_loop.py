"""Tests for closed-loop execution runner, online adaptation, and information boundaries."""

import pytest
import numpy as np
import torch

from kinesis.closed_loop import (
    ClosedLoopRunner,
    OnlineAdaptationManager,
)
from kinesis.envs import DMControlAdapter
from kinesis.models import WorldModel
from kinesis.policies import CartpoleEnergyHeuristic, RandomPolicy
from kinesis.replay import BoundedReplayBuffer
from kinesis.shifts import build_shifted_environment
from kinesis.training import WorldModelTrainer


def test_closed_loop_runner_execution_and_budget() -> None:
    env = DMControlAdapter("cartpole_swingup", action_repeat=2, seed=42)
    policy = RandomPolicy(action_spec=env.action_spec, seed=42)
    buf = BoundedReplayBuffer(capacity_transitions=500, chunk_size=20)

    runner = ClosedLoopRunner(replay_buffer=buf)
    ep, exec_res = runner.run_episode(env=env, policy=policy, seed=42, max_steps=25)

    assert ep.length == 26  # initial + 25 steps
    assert exec_res.steps == 25
    assert not exec_res.privileged_leakage_detected
    assert buf.total_transitions == 26
    env.close()


def test_world_model_training_step_from_replay() -> None:
    env = DMControlAdapter("cartpole_swingup", action_repeat=2, seed=42)
    policy = CartpoleEnergyHeuristic(action_spec=env.action_spec)
    buf = BoundedReplayBuffer(capacity_transitions=500, chunk_size=20)

    runner = ClosedLoopRunner(replay_buffer=buf)
    # Collect 2 short episodes to have enough transitions for sequence sampling
    runner.run_episode(env=env, policy=policy, seed=42, max_steps=20)
    runner.run_episode(env=env, policy=policy, seed=43, max_steps=20)

    obs_dim = env.observation_spec.total_flattened_dim
    act_dim = env.action_spec.dim
    wm = WorldModel(obs_dim=obs_dim, action_dim=act_dim, embed_dim=32, deter_dim=64, stoch_dim=16, hidden_dim=64)

    trainer = WorldModelTrainer(model=wm, lr=1e-3)
    metrics = trainer.train_batch(replay_buffer=buf, batch_size=2, sequence_length=8)

    assert metrics.step == 1
    assert metrics.total_loss > 0.0
    assert metrics.grad_norm > 0.0
    env.close()


def test_online_adaptation_on_shifted_environment() -> None:
    shifted_env = build_shifted_environment("cartpole_swingup", profile_name="actuator_attenuation", seed=42)
    policy = CartpoleEnergyHeuristic(action_spec=shifted_env.action_spec)

    obs_dim = shifted_env.observation_spec.total_flattened_dim
    act_dim = shifted_env.action_spec.dim
    wm = WorldModel(obs_dim=obs_dim, action_dim=act_dim, embed_dim=32, deter_dim=64, stoch_dim=16, hidden_dim=64)

    adapt_mgr = OnlineAdaptationManager(world_model=wm, adaptation_update_budget=5, batch_size=2, sequence_length=6)
    res = adapt_mgr.evaluate_and_adapt(
        shifted_env=shifted_env,
        policy=policy,
        shift_name="actuator_attenuation",
        seed=42,
        pre_shift_return=15.0,
        adaptation_episodes=2,
    )

    assert res.shift_name == "actuator_attenuation"
    assert res.adaptation_steps_budget == 5
    assert res.recovery_delay_steps > 0
    shifted_env.close()
