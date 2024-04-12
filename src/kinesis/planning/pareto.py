"""Multi-objective Pareto frontier optimizer balancing reward, risk, and control effort."""

from dataclasses import dataclass
import numpy as np


@dataclass(eq=False)
class CandidateTrajectory:
    """A candidate trajectory with multiple evaluated objective values."""
    actions: np.ndarray  # Shape (H, action_dim)
    objectives: np.ndarray  # Shape (num_objectives,), all normalized for maximization
    rank: int = 0
    crowding_distance: float = 0.0


@dataclass
class ParetoFrontierResult:
    """Analysis of the non-dominated Pareto frontier."""
    pareto_candidates: list[CandidateTrajectory]
    frontier_size: int
    total_candidates: int
    best_compromise: CandidateTrajectory
    objective_nadir: np.ndarray
    objective_utopia: np.ndarray


class ParetoFrontierOptimizer:
    """Computes non-dominated Pareto fronts and selects compromise solutions."""

    def __init__(self, num_objectives: int = 3) -> None:
        self.num_objectives = int(num_objectives)

    def dominates(self, obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
        """Return True if candidate A dominates candidate B (assuming maximization)."""
        all_ge = np.all(obj_a >= obj_b - 1e-9)
        strictly_gt = np.any(obj_a > obj_b + 1e-9)
        return bool(all_ge and strictly_gt)

    def compute_non_dominated_front(
        self,
        candidates: list[CandidateTrajectory],
    ) -> list[CandidateTrajectory]:
        """Filter candidates to strictly non-dominated set (Front 0)."""
        N = len(candidates)
        if N == 0:
            return []

        domination_counts = [0] * N
        dominated_by_me: list[list[int]] = [[] for _ in range(N)]

        for i in range(N):
            for j in range(i + 1, N):
                if self.dominates(candidates[i].objectives, candidates[j].objectives):
                    dominated_by_me[i].append(j)
                    domination_counts[j] += 1
                elif self.dominates(candidates[j].objectives, candidates[i].objectives):
                    dominated_by_me[j].append(i)
                    domination_counts[i] += 1

        front_indices = [i for i in range(N) if domination_counts[i] == 0]
        front = [candidates[i] for i in front_indices]
        for c in front:
            c.rank = 0

        return front

    def compute_crowding_distance(self, front: list[CandidateTrajectory]) -> None:
        """Assign NSGA-II crowding distances to preserve population diversity."""
        L = len(front)
        if L <= 2:
            for c in front:
                c.crowding_distance = float("inf")
            return

        for c in front:
            c.crowding_distance = 0.0

        for m in range(self.num_objectives):
            front.sort(key=lambda c: c.objectives[m])
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            obj_min = front[0].objectives[m]
            obj_max = front[-1].objectives[m]
            norm_range = obj_max - obj_min

            if norm_range > 1e-9:
                for i in range(1, L - 1):
                    dist = (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / norm_range
                    front[i].crowding_distance += float(dist)

    def select_best_compromise(
        self,
        front: list[CandidateTrajectory],
        weights: np.ndarray | None = None,
    ) -> CandidateTrajectory:
        """Select best compromise solution via weighted Chebyshev distance to Utopia."""
        assert len(front) > 0, "Front cannot be empty"
        if len(front) == 1:
            return front[0]

        all_objs = np.array([c.objectives for c in front])
        utopia = np.max(all_objs, axis=0)

        if weights is None:
            w = np.ones(self.num_objectives) / self.num_objectives
        else:
            w = np.asarray(weights, dtype=np.float64)
            w = w / np.sum(w)

        # Minimize max weighted regret: min_c max_m w_m * (utopia_m - obj_m)
        best_cand = front[0]
        best_regret = float("inf")

        for c in front:
            regrets = w * (utopia - c.objectives)
            max_regret = float(np.max(regrets))
            if max_regret < best_regret:
                best_regret = max_regret
                best_cand = c

        return best_cand

    def optimize_frontier(
        self,
        candidates: list[CandidateTrajectory],
        weights: np.ndarray | None = None,
    ) -> ParetoFrontierResult:
        """Run non-dominated sorting, compute diversity, and select compromise trajectory."""
        front = self.compute_non_dominated_front(candidates)
        self.compute_crowding_distance(front)

        all_objs = np.array([c.objectives for c in candidates])
        nadir = np.min(all_objs, axis=0) if len(candidates) > 0 else np.zeros(self.num_objectives)
        utopia = np.max(all_objs, axis=0) if len(candidates) > 0 else np.zeros(self.num_objectives)

        best = self.select_best_compromise(front, weights=weights)

        return ParetoFrontierResult(
            pareto_candidates=front,
            frontier_size=len(front),
            total_candidates=len(candidates),
            best_compromise=best,
            objective_nadir=nadir,
            objective_utopia=utopia,
        )
