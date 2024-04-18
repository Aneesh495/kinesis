"""End-to-end integration pipeline: data collection, RSSM training, and latent MPC control."""

import numpy as np
import pytest
import torch

from kinesis.closed_loop import ClosedLoopRunner
from kinesis.envs.dm_control_adapter import DMControlAdapter
from kinesis.models.world_model import WorldModel
from kinesis.planning.cem import CrossEntropyMethodPlanner
from kinesis.planning.planner_policy import LatentPlannerPolicy
from kinesis.policies.random import RandomPolicy
from kinesis.replay.buffer import BoundedReplayBuffer
from kinesis.training.trainer import WorldModelTrainer


def test_end_to_end_collect_train_plan_pipeline():
    """Verify entire lifecycle from collection to training to closed-loop latent MPC execution."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Environment and initial exploration policy
    env = DMControlAdapter("cartpole_swingup", action_repeat=2, seed=seed)
    random_policy = RandomPolicy(action_spec=env.action_spec, seed=seed)
    replay_buf = BoundedReplayBuffer(capacity_transitions=500, chunk_size=20)

    runner = ClosedLoopRunner(replay_buffer=replay_buf)

    # 2. Collect initial bootstrap episodes
    runner.run_episode(env=env, policy=random_policy, seed=seed, max_steps=15)
    runner.run_episode(env=env, policy=random_policy, seed=seed + 1, max_steps=15)

    assert replay_buf.total_episodes >= 2
    assert replay_buf.total_transitions >= 30

    # 3. Instantiate compact World Model and train on replay buffer
    obs_dim = env.observation_spec.total_flattened_dim
    act_dim = env.action_spec.dim

    world_model = WorldModel(
        obs_dim=obs_dim,
        action_dim=act_dim,
        embed_dim=32,
        deter_dim=64,
        stoch_dim=16,
        hidden_dim=64,
    )

    trainer = WorldModelTrainer(model=world_model, lr=1e-3)
    metrics = trainer.train_batch(replay_buffer=replay_buf, batch_size=2, sequence_length=6)

    assert metrics.step == 1
    assert np.isfinite(metrics.total_loss)
    assert metrics.total_loss > 0.0

    # 4. Latent trajectory planner with trained world model
    planner = CrossEntropyMethodPlanner(
        ensemble=world_model,
        action_spec=env.action_spec,
        horizon=4,
        num_candidates=10,
        num_iterations=2,
    )

    planner_policy = LatentPlannerPolicy(
        planner=planner,
        observation_spec=env.observation_spec,
        action_spec=env.action_spec,
        timeout_sec=1.0,
    )

    # 5. Closed-loop execution with LatentPlannerPolicy
    test_ep, exec_res = runner.run_episode(
        env=env,
        policy=planner_policy,
        seed=seed + 10,
        max_steps=5,
    )

    # 6. Verify closed loop assertions and privacy boundaries
    assert test_ep.length == 6
    assert exec_res.steps == 5
    assert not exec_res.privileged_leakage_detected
    assert np.isfinite(exec_res.total_return)

    env.close()
