"""Evaluation campaign, independent metrics, counterfactual comparisons, error gap analysis, and ablations."""

from kinesis.evaluation.metrics import (
    BenchmarkMetrics,
    aggregate_population_metrics,
    recompute_episode_metrics,
)
from kinesis.evaluation.counterfactual import (
    CounterfactualEvaluator,
    PairedEvaluationSummary,
)
from kinesis.evaluation.error_gap import (
    ErrorGapAnalysisResult,
    PredictionControlErrorGapAnalyzer,
)
from kinesis.evaluation.ablations import (
    AblationMatrixRunner,
    AblationResult,
)

__all__ = [
    "BenchmarkMetrics",
    "aggregate_population_metrics",
    "recompute_episode_metrics",
    "CounterfactualEvaluator",
    "PairedEvaluationSummary",
    "ErrorGapAnalysisResult",
    "PredictionControlErrorGapAnalyzer",
    "AblationMatrixRunner",
    "AblationResult",
]
