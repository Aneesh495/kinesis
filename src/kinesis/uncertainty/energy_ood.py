"""Energy-based out-of-distribution detector for identifying runtime distribution shift."""

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


@dataclass
class OODDetectionResult:
    """Output from energy-based OOD detector."""
    energy_score: float
    is_ood: bool
    anomaly_probability: float
    threshold: float


class EnergyOODDetector:
    """Detects out-of-distribution latent states and observations using free energy scores."""

    def __init__(self, temperature: float = 1.0, percentile_threshold: float = 95.0) -> None:
        self.temperature = float(temperature)
        self.percentile = float(percentile_threshold)
        self.fitted = False
        self.energy_threshold = 0.0
        self.min_energy = 0.0
        self.max_energy = 1.0

    def compute_energy(self, logits: torch.Tensor | np.ndarray) -> np.ndarray:
        """Compute free energy score E(x) = -T * log(sum(exp(logits / T)))."""
        if isinstance(logits, np.ndarray):
            logits_tensor = torch.from_numpy(logits)
        else:
            logits_tensor = logits

        # Free energy: negative log-sum-exp
        scaled = logits_tensor / self.temperature
        energy = -self.temperature * torch.logsumexp(scaled, dim=-1)
        return energy.detach().cpu().numpy()

    def fit(self, in_distribution_logits: torch.Tensor | np.ndarray) -> float:
        """Fit energy threshold on in-distribution calibration samples."""
        energies = self.compute_energy(in_distribution_logits)
        self.energy_threshold = float(np.percentile(energies, self.percentile))
        self.min_energy = float(np.min(energies))
        self.max_energy = float(np.max(energies))
        self.fitted = True
        return self.energy_threshold

    def detect(self, logits: torch.Tensor | np.ndarray) -> OODDetectionResult:
        """Evaluate if sample is out-of-distribution."""
        assert self.fitted, "EnergyOODDetector must be fitted on in-distribution data before detection"
        energies = self.compute_energy(logits)
        score = float(np.mean(energies))

        # Higher energy indicates out-of-distribution
        is_ood = bool(score > self.energy_threshold)

        # Normalized anomaly probability
        span = max(self.max_energy - self.min_energy, 1e-6)
        prob = float(np.clip((score - self.min_energy) / span, 0.0, 1.0))

        return OODDetectionResult(
            energy_score=score,
            is_ood=is_ood,
            anomaly_probability=prob,
            threshold=self.energy_threshold,
        )
