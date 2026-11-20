"""
preprocessing.py
================
Module 2: fMRI Signal Preprocessing and Conditioning.
Applies zero-phase temporal bandpass filtering (0.01 - 0.10 Hz),
linear detrending, temporal z-score normalization, and motion scrubbing.
"""

import sys
import numpy as np
from pathlib import Path
from scipy import signal
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FMR_TR, LOW_FREQ, HIGH_FREQ


class FMRIPreprocessor:
    """Performs rigorous resting-state fMRI BOLD signal conditioning."""

    def __init__(
        self,
        tr: float = FMR_TR,
        low_freq: float = LOW_FREQ,
        high_freq: float = HIGH_FREQ
    ):
        self.tr = tr
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.nyquist = 0.5 * (1.0 / self.tr)

    def bandpass_filter(
        self,
        timeseries: np.ndarray,
        order: int = 3
    ) -> np.ndarray:
        """
        Applies a zero-phase forward-backward Butterworth bandpass filter (0.01–0.10 Hz).
        Eliminates respiratory/cardiac artifacts and slow drifts without introducing phase distortion.
        
        Parameters:
            timeseries: Array of shape (timepoints, n_regions)
            order: Filter order (default 3)
            
        Returns:
            Filtered timeseries array of same shape.
        """
        low = self.low_freq / self.nyquist
        high = self.high_freq / self.nyquist
        
        # Ensure cutoff frequencies stay within valid bounds (0 < Wn < 1)
        low = max(0.001, min(low, 0.98))
        high = max(low + 0.01, min(high, 0.99))

        b, a = signal.butter(order, [low, high], btype='bandpass')
        # Zero-phase forward-backward digital filter
        filtered = signal.filtfilt(b, a, timeseries, axis=0)
        return filtered

    def detrend(self, timeseries: np.ndarray) -> np.ndarray:
        """Removes linear scanner heating drift along the temporal axis."""
        return signal.detrend(timeseries, axis=0, type='linear')

    def standardize(self, timeseries: np.ndarray) -> np.ndarray:
        """
        Z-score normalizes each region's time series to zero mean and unit variance.
        Ensures correlation calculations are scale-invariant across participants.
        """
        means = np.mean(timeseries, axis=0, keepdims=True)
        stds = np.std(timeseries, axis=0, keepdims=True)
        stds[stds == 0] = 1.0  # Prevent division by zero
        return (timeseries - means) / stds

    def scrub_motion(
        self,
        timeseries: np.ndarray,
        framewise_displacement: Optional[np.ndarray] = None,
        fd_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects and flags high-motion volumes exceeding the Framewise Displacement threshold.
        """
        if framewise_displacement is None:
            # Estimate proxy FD from mean frame-to-frame BOLD signal changes
            diff = np.diff(timeseries, axis=0)
            fd_proxy = np.mean(np.abs(diff), axis=1)
            fd = np.insert(fd_proxy, 0, 0.0)
        else:
            fd = framewise_displacement

        scrub_mask = fd <= fd_threshold
        scrubbed_ts = timeseries[scrub_mask, :]
        return scrubbed_ts, scrub_mask

    def process(
        self,
        timeseries: np.ndarray,
        apply_filter: bool = True,
        apply_detrend: bool = True,
        apply_standardize: bool = True
    ) -> np.ndarray:
        """Executes full preprocessing pipeline on a single subject's BOLD time series."""
        out = timeseries.copy()
        if apply_detrend:
            out = self.detrend(out)
        if apply_filter:
            out = self.bandpass_filter(out)
        if apply_standardize:
            out = self.standardize(out)
        return out


if __name__ == "__main__":
    # Quick self-test
    np.random.seed(42)
    sample_raw = np.random.normal(0, 1, (180, 116)) + np.linspace(0, 3, 180)[:, None]
    preprocessor = FMRIPreprocessor()
    processed = preprocessor.process(sample_raw)
    print(f"[+] Raw shape: {sample_raw.shape}, Processed shape: {processed.shape}")
    print(f"[+] Processed mean: {np.mean(processed):.4f}, std: {np.std(processed):.4f}")
