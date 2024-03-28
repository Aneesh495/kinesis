"""Actuator chatter and Fourier Power Spectral Density (PSD) frequency analyzer."""

from dataclasses import dataclass
import numpy as np


@dataclass
class SpectralAnalysisResult:
    """Quantitative frequency and chatter analysis of control signals."""
    dominant_frequency_hz: float
    total_power: float
    high_frequency_chatter_ratio: float
    is_excessive_chatter: bool
    frequencies: np.ndarray
    power_spectral_density: np.ndarray


class ActuatorSpectralAnalyzer:
    """Analyzes control signal frequency spectrum to detect actuator chatter and mechanical resonance."""

    def __init__(
        self,
        sampling_rate_hz: float = 20.0,
        high_freq_cutoff_hz: float = 5.0,
        chatter_ratio_threshold: float = 0.35,
    ) -> None:
        self.fs = float(sampling_rate_hz)
        self.cutoff = float(high_freq_cutoff_hz)
        self.threshold = float(chatter_ratio_threshold)

    def analyze(self, control_signal: np.ndarray) -> SpectralAnalysisResult:
        """Compute one-sided Power Spectral Density via Discrete Fourier Transform."""
        u = np.asarray(control_signal, dtype=np.float64).flatten()
        N = len(u)
        assert N >= 4, "Control signal must contain at least 4 samples for spectral analysis"

        # Remove DC component (mean)
        u_detrended = u - np.mean(u)

        # Compute FFT
        fft_vals = np.fft.rfft(u_detrended)
        freqs = np.fft.rfftfreq(N, d=1.0 / self.fs)

        # Power spectral density
        psd = (np.abs(fft_vals) ** 2) / N

        total_power = float(np.sum(psd))
        if total_power < 1e-12:
            return SpectralAnalysisResult(
                dominant_frequency_hz=0.0,
                total_power=0.0,
                high_frequency_chatter_ratio=0.0,
                is_excessive_chatter=False,
                frequencies=freqs,
                power_spectral_density=psd,
            )

        # Dominant frequency (excluding DC if freqs[0] == 0)
        dom_idx = int(np.argmax(psd[1:])) + 1 if len(psd) > 1 else 0
        dom_freq = float(freqs[dom_idx])

        # High frequency power above cutoff
        high_freq_mask = freqs >= self.cutoff
        high_freq_power = float(np.sum(psd[high_freq_mask]))
        chatter_ratio = high_freq_power / total_power

        excessive = bool(chatter_ratio > self.threshold)

        return SpectralAnalysisResult(
            dominant_frequency_hz=dom_freq,
            total_power=total_power,
            high_frequency_chatter_ratio=chatter_ratio,
            is_excessive_chatter=excessive,
            frequencies=freqs,
            power_spectral_density=psd,
        )
