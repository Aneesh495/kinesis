"""Performance benchmarking, memory profiling, and latency characterization."""

from dataclasses import dataclass
import os
import resource
import time
from typing import Any
import numpy as np
import torch

from kinesis.contracts import ActionSpec, ObservationChannel, ObservationSpec
from kinesis.envs import DMControlAdapter
from kinesis.models import WorldModel
from kinesis.planning import CrossEntropyMethodPlanner
from kinesis.replay import BoundedReplayBuffer
from kinesis.uncertainty import WorldModelEnsemble


@dataclass(frozen=True)
class BenchmarkProfileResult:
    """Rigorous profile measurements with short and long repetitions."""
    encoder_latency_ms: float
    imagine_step_latency_ms: float
    cem_plan_latency_ms: float
    env_step_latency_ms: float
    replay_sample_throughput_seq_per_sec: float
    training_throughput_updates_per_sec: float
    peak_rss_mib: float
    short_reps: int = 30
    long_reps: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoder_latency_ms": self.encoder_latency_ms,
            "imagine_step_latency_ms": self.imagine_step_latency_ms,
            "cem_plan_latency_ms": self.cem_plan_latency_ms,
            "env_step_latency_ms": self.env_step_latency_ms,
            "replay_sample_throughput_seq_per_sec": self.replay_sample_throughput_seq_per_sec,
            "training_throughput_updates_per_sec": self.training_throughput_updates_per_sec,
            "peak_rss_mib": self.peak_rss_mib,
            "short_reps": self.short_reps,
            "long_reps": self.long_reps,
        }


def get_current_rss_mib() -> float:
    """Query current resident set size in MiB."""
    # On macOS ru_maxrss is in bytes, on Linux in kilobytes
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return float(usage / (1024 * 1024))
    return float(usage / 1024)


def run_comprehensive_benchmark(task_name: str = "cartpole_swingup") -> BenchmarkProfileResult:
    """Execute fixed 30 short-run and 10 longer workload repetitions measuring latency and throughput."""
    device = torch.device("cpu")
    env = DMControlAdapter(task_name, action_repeat=2, seed=42)
    obs_dim = env.observation_spec.total_flattened_dim
    act_dim = env.action_spec.dim

    wm = WorldModel(obs_dim=obs_dim, action_dim=act_dim, embed_dim=32, deter_dim=64, stoch_dim=16, hidden_dim=64)
    wm.to(device)

    # 1. Measure encoder latency (30 reps)
    sample_obs = torch.randn(1, obs_dim, device=device)
    enc_times = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = wm.encoder(sample_obs)
        enc_times.append(time.perf_counter() - t0)
    enc_latency_ms = float(np.mean(enc_times) * 1000.0)

    # 2. Measure imagine step latency (30 reps)
    init_s = wm.rssm.initial_state(1, device)
    sample_act = torch.zeros(1, act_dim, device=device)
    img_times = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = wm.rssm.imagine_step(init_s, sample_act)
        img_times.append(time.perf_counter() - t0)
    img_latency_ms = float(np.mean(img_times) * 1000.0)

    # 3. Measure CEM plan latency (10 reps)
    planner = CrossEntropyMethodPlanner(
        ensemble=wm,
        action_spec=env.action_spec,
        horizon=6,
        num_candidates=20,
        num_iterations=2,
    )
    plan_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = planner.plan(init_s)
        plan_times.append(time.perf_counter() - t0)
    plan_latency_ms = float(np.mean(plan_times) * 1000.0)

    # 4. Measure environment stepping latency (30 reps)
    t0 = env.reset(seed=42)
    env_times = []
    zero_act = env.action_spec.zero_action()
    for _ in range(30):
        t_start = time.perf_counter()
        _ = env.step(zero_act)
        env_times.append(time.perf_counter() - t_start)
    env_latency_ms = float(np.mean(env_times) * 1000.0)

    # 5. Measure replay sample throughput (10 reps)
    buf = BoundedReplayBuffer(capacity_transitions=2000, chunk_size=20)
    for ep in range(5):
        t = env.reset(seed=ep)
        buf.add_transition(t)
        for s in range(30):
            t = env.step(zero_act)
            buf.add_transition(t)

    sample_throughputs = []
    for _ in range(10):
        t_start = time.perf_counter()
        _ = buf.sample_sequences(batch_size=16, sequence_length=8)
        dur = time.perf_counter() - t_start
        sample_throughputs.append(16.0 / max(1e-6, dur))
    sample_tp = float(np.mean(sample_throughputs))

    # 6. Training throughput (10 reps)
    optimizer = torch.optim.Adam(wm.parameters(), lr=1e-3)
    train_times = []
    for _ in range(10):
        batch = buf.sample_sequences(batch_size=8, sequence_length=8)
        obs_t = torch.cat(list(batch["observations"].values()), dim=-1).to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)
        discounts = batch["discounts"].to(device)
        costs = batch["constraint_costs"].to(device)
        is_first = batch["is_first"].to(device)

        t_start = time.perf_counter()
        optimizer.zero_grad()
        loss_res = wm.compute_losses(obs_t, actions, rewards, discounts, costs, is_first)
        loss_res.total_loss.backward()
        optimizer.step()
        train_times.append(time.perf_counter() - t_start)
    train_tp = float(1.0 / np.mean(train_times))

    env.close()
    peak_rss = get_current_rss_mib()

    return BenchmarkProfileResult(
        encoder_latency_ms=enc_latency_ms,
        imagine_step_latency_ms=img_latency_ms,
        cem_plan_latency_ms=plan_latency_ms,
        env_step_latency_ms=env_latency_ms,
        replay_sample_throughput_seq_per_sec=sample_tp,
        training_throughput_updates_per_sec=train_tp,
        peak_rss_mib=peak_rss,
    )
