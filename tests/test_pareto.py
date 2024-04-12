"""Unit tests for multi-objective Pareto frontier planner."""

import numpy as np
import pytest

from kinesis.planning.pareto import (
    CandidateTrajectory,
    ParetoFrontierOptimizer,
    ParetoFrontierResult,
)


def test_pareto_domination_and_frontier():
    """Verify Pareto dominance filtering correctly isolates non-dominated points."""
    opt = ParetoFrontierOptimizer(num_objectives=2)

    # 4 candidates with objectives [reward, -risk]
    # c1: high reward (10), high risk (-5)
    # c2: medium reward (8), low risk (-2)
    # c3: dominated by c2 (reward 6, risk -3)
    # c4: dominated by c1 (reward 9, risk -6)
    dummy_act = np.zeros((5, 1))

    c1 = CandidateTrajectory(actions=dummy_act, objectives=np.array([10.0, -5.0]))
    c2 = CandidateTrajectory(actions=dummy_act, objectives=np.array([8.0, -2.0]))
    c3 = CandidateTrajectory(actions=dummy_act, objectives=np.array([6.0, -3.0]))
    c4 = CandidateTrajectory(actions=dummy_act, objectives=np.array([9.0, -6.0]))

    candidates = [c1, c2, c3, c4]
    front = opt.compute_non_dominated_front(candidates)

    # Only c1 and c2 should be on the Pareto front
    assert len(front) == 2
    assert c1 in front
    assert c2 in front
    assert c3 not in front
    assert c4 not in front


def test_pareto_crowding_and_compromise():
    """Verify crowding distance calculation and Chebyshev compromise selection."""
    opt = ParetoFrontierOptimizer(num_objectives=2)
    dummy_act = np.zeros((5, 1))

    c1 = CandidateTrajectory(actions=dummy_act, objectives=np.array([10.0, 0.0]))
    c2 = CandidateTrajectory(actions=dummy_act, objectives=np.array([5.0, 5.0]))
    c3 = CandidateTrajectory(actions=dummy_act, objectives=np.array([0.0, 10.0]))

    candidates = [c1, c2, c3]
    res = opt.optimize_frontier(candidates, weights=np.array([0.5, 0.5]))

    assert isinstance(res, ParetoFrontierResult)
    assert res.frontier_size == 3
    assert res.total_candidates == 3

    # Boundary points receive infinite crowding distance
    assert c1.crowding_distance == float("inf")
    assert c3.crowding_distance == float("inf")
    assert c2.crowding_distance > 0.0

    # Compromise solution with equal weights should pick balanced c2
    assert res.best_compromise == c2
