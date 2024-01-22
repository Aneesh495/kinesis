"""Tests for trajectory inspection viewer and performance benchmark profiler."""

from pathlib import Path
import pytest
import numpy as np

from kinesis.contracts import (
    Episode,
    EpisodeID,
    ShiftCategory,
    StepType,
    TerminationReason,
    Transition,
    TransitionID,
)
from kinesis.viewer import (
    TrajectoryViewer,
    get_current_rss_mib,
    run_comprehensive_benchmark,
)


def _make_dummy_episode() -> Episode:
    transitions = []
    for i in range(10):
        t = Transition(
            transition_id=TransitionID(f"t_{i}"),
            episode_id=EpisodeID("ep_viewer"),
            step_index=i,
            step_type=StepType.FIRST if i == 0 else (StepType.LAST if i == 9 else StepType.MID),
            observation={"q": np.array([float(i), 0.5], dtype=np.float32)},
            action=np.array([0.1, -0.2], dtype=np.float32),
            reward=float(i),
            constraint_cost=0.01,
            discount=0.0 if i == 9 else 1.0,
            time_delta=0.02,
        )
        transitions.append(t)

    return Episode(
        episode_id=EpisodeID("ep_viewer"),
        task_name="cartpole_swingup",
        seed=42,
        transitions=tuple(transitions),
        termination_reason=TerminationReason.TASK_SUCCESS,
        shift_category=ShiftCategory.NOMINAL,
    )


def test_trajectory_viewer_ascii_and_plot(tmp_path: Path) -> None:
    ep = _make_dummy_episode()

    # ASCII summary
    summary_str = TrajectoryViewer.render_ascii_summary(ep)
    assert "Episode Inspection Summary" in summary_str
    assert "Cumulative Return" in summary_str

    # Plot rendering
    out_plot = tmp_path / "trajectory_audit.png"
    TrajectoryViewer.render_diagnostic_plot(ep, out_plot, imagined_rewards=[0.5] * 10)
    assert out_plot.exists()
    assert out_plot.stat().st_size > 1000  # Non-trivial image file


def test_rss_memory_query() -> None:
    rss = get_current_rss_mib()
    assert rss > 0.0, "RSS memory must be positive"


def test_run_benchmark_profiler() -> None:
    res = run_comprehensive_benchmark(task_name="cartpole_swingup")
    assert res.encoder_latency_ms > 0.0
    assert res.imagine_step_latency_ms > 0.0
    assert res.cem_plan_latency_ms > 0.0
    assert res.peak_rss_mib > 0.0
