"""
preprocessing.py
================
Biomedical ECG Preprocessing Pipeline.

Covers:
1. Loading raw ECG signals from PhysioNet MIT-BIH recordings or raw arrays.
2. Baseline wander removal (respiratory drift, < 0.5 Hz).
3. High-frequency noise suppression (EMG tremor, powerline interference, > 45 Hz).
4. Zero-phase bandpass filtering (Butterworth IIR with forward-backward filtfilt).
5. Notch filtering (50 Hz / 60 Hz AC interference).
6. Amplitude normalization (Z-score, MinMax, Robust IQR).
7. Comprehensive time-domain and frequency-domain (PSD) visual comparison.
"""

import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import wfdb
try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False


class ECGPreprocessor:
    """
    Standardized ECG Preprocessing pipeline for noise reduction
    and signal conditioning prior to deep learning segmentation.
    """
    
    def __init__(self, sampling_rate=360, lowcut=0.5, highcut=45.0, order=3, powerline=60.0):
        """
        Parameters:
        -----------
        sampling_rate : int
            Sampling frequency of the ECG signal in Hz (MIT-BIH default = 360 Hz).
        lowcut : float
            Lower cutoff frequency in Hz for baseline wander removal (default = 0.5 Hz).
        highcut : float
            Upper cutoff frequency in Hz for high-frequency noise removal (default = 45.0 Hz).
        order : int
            Filter order for the Butterworth filter (default = 3).
        powerline : float
            Powerline frequency to notch filter in Hz (default = 60.0 Hz for US/MIT-BIH).
        """
        self.fs = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.powerline = powerline
        
    def butter_bandpass(self, lowcut=None, highcut=None, order=None):
        """
        Designs Butterworth bandpass filter coefficients (b, a).
        Nyquist frequency = fs / 2.
        """
        low = lowcut if lowcut is not None else self.lowcut
        high = highcut if highcut is not None else self.highcut
        ord_val = order if order is not None else self.order
        
        nyq = 0.5 * self.fs
        low_norm = low / nyq
        high_norm = high / nyq
        
        # Ensure cutoff frequencies are within valid (0, 1) normalized range
        low_norm = max(0.001, min(low_norm, 0.99))
        high_norm = max(low_norm + 0.01, min(high_norm, 0.999))
        
        b, a = signal.butter(ord_val, [low_norm, high_norm], btype='bandpass')
        return b, a

    def apply_bandpass(self, ecg_signal):
        """
        Applies zero-phase forward-backward Butterworth bandpass filter.
        Uses scipy.signal.filtfilt to prevent phase distortion and preserve QRS morphology.
        """
        b, a = self.butter_bandpass()
        filtered = signal.filtfilt(b, a, ecg_signal)
        return filtered

    def apply_notch(self, ecg_signal, freq=None, quality_factor=30.0):
        """
        Applies an IIR notch filter to suppress powerline interference (50 Hz or 60 Hz).
        """
        f0 = freq if freq is not None else self.powerline
        nyq = 0.5 * self.fs
        if f0 >= nyq:
            return ecg_signal  # Notch frequency exceeds Nyquist limit, skip
            
        w0 = f0 / nyq
        b, a = signal.iirnotch(w0, quality_factor)
        filtered = signal.filtfilt(b, a, ecg_signal)
        return filtered

    def remove_baseline_wander(self, ecg_signal):
        """
        Removes low-frequency baseline drift using Butterworth high-pass filter (0.5 Hz).
        """
        nyq = 0.5 * self.fs
        norm_cutoff = self.lowcut / nyq
        b, a = signal.butter(self.order, norm_cutoff, btype='highpass')
        return signal.filtfilt(b, a, ecg_signal)

    def normalize(self, ecg_signal, method='zscore'):
        """
        Normalizes the ECG signal amplitude to stabilize neural network training.
        
        Methods:
        --------
        'zscore' : (x - mean) / std (Standardizes amplitude to mean=0, std=1)
        'minmax' : Scaled between 0 and 1
        'robust' : (x - median) / IQR (Resistant to extreme voltage outliers)
        """
        if method == 'zscore':
            std = np.std(ecg_signal)
            if std == 0:
                return ecg_signal
            return (ecg_signal - np.mean(ecg_signal)) / std
            
        elif method == 'minmax':
            s_min = np.min(ecg_signal)
            s_max = np.max(ecg_signal)
            if s_max - s_min == 0:
                return ecg_signal
            return (ecg_signal - s_min) / (s_max - s_min)
            
        elif method == 'robust':
            med = np.median(ecg_signal)
            q75, q25 = np.percentile(ecg_signal, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                return ecg_signal
            return (ecg_signal - med) / iqr
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def clean_neurokit(self, ecg_signal, method='neurokit'):
        """
        Cleans ECG signal using NeuroKit2 biomedical library routines.
        Methods include: 'neurokit', 'biosppy', 'pantompkins1985'.
        """
        if not HAS_NEUROKIT:
            return self.apply_bandpass(ecg_signal)
        return nk.ecg_clean(ecg_signal, sampling_rate=self.fs, method=method)

    def preprocess_pipeline(self, raw_signal, normalize_method='zscore', use_neurokit=False):
        """
        Complete end-to-end preprocessing pipeline for a continuous raw ECG lead:
        1. Input validation & NaN handling
        2. Zero-phase Bandpass Filtering (0.5 - 45 Hz)
        3. Notch Filtering (60 Hz powerline hum)
        4. Amplitude Normalization (Z-score / MinMax)
        
        Returns:
        --------
        cleaned_signal : np.ndarray
        """
        # 1. Validation
        signal_clean = np.array(raw_signal, dtype=np.float64)
        if np.isnan(signal_clean).any():
            # Interpolate NaNs if present
            nans = np.isnan(signal_clean)
            x_vals = lambda z: z.nonzero()[0]
            signal_clean[nans] = np.interp(x_vals(nans), x_vals(~nans), signal_clean[~nans])
            
        # 2. Bandpass filtering
        if use_neurokit and HAS_NEUROKIT:
            filtered = self.clean_neurokit(signal_clean)
        else:
            filtered = self.apply_bandpass(signal_clean)
            filtered = self.apply_notch(filtered)
            
        # 3. Normalization
        normalized = self.normalize(filtered, method=normalize_method)
        return normalized


def plot_preprocessing_comparison(raw_ecg, clean_ecg, fs=360, duration_sec=8, 
                                  record_id="100", lead_name="MLII", save_path="figures/ecg_preprocessing_comparison.png"):
    """
    Plots side-by-side time-domain and frequency-domain (PSD) comparisons
    illustrating the exact engineering effects of baseline removal and noise suppression.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    n_samples = int(duration_sec * fs)
    time_axis = np.arange(n_samples) / fs
    
    raw_seg = raw_ecg[:n_samples]
    clean_seg = clean_ecg[:n_samples]
    
    # Calculate Power Spectral Density (PSD) using Welch's Method
    freqs_raw, psd_raw = signal.welch(raw_ecg, fs=fs, nperseg=min(len(raw_ecg), 1024))
    freqs_clean, psd_clean = signal.welch(clean_ecg, fs=fs, nperseg=min(len(clean_ecg), 1024))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"ECG Preprocessing Pipeline Validation - Record {record_id} ({lead_name})", fontsize=14, fontweight='bold')
    
    # Plot 1: Raw vs Clean Time-Domain Waveform Overlay
    axes[0].plot(time_axis, raw_seg, color='#7f7f7f', alpha=0.7, label='Raw ECG (with Baseline Drift & Noise)', lw=1.2)
    axes[0].set_title(f"1. Raw Unfiltered ECG Signal ({duration_sec}s window)", fontsize=11, fontweight='semibold')
    axes[0].set_ylabel("Amplitude (mV)", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='upper right')
    
    # Plot 2: Cleaned and Normalized ECG Signal
    axes[1].plot(time_axis, clean_seg, color='#1f77b4', lw=1.3, label='Preprocessed & Normalized ECG (Bandpass 0.5-45 Hz + Z-Score)')
    axes[1].set_title(f"2. Conditioned ECG Signal (Isoelectric Baseline Stabilized)", fontsize=11, fontweight='semibold')
    axes[1].set_ylabel("Normalized Amplitude", fontsize=10)
    axes[1].set_xlabel("Time (seconds)", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='upper right')
    
    # Plot 3: Frequency Spectrum (Power Spectral Density - Welch Method)
    axes[2].semilogy(freqs_raw, psd_raw, color='#d62728', alpha=0.7, label='Raw Signal PSD (Showing <0.5 Hz Wander & High-Freq Noise)')
    axes[2].semilogy(freqs_clean, psd_clean, color='#2ca02c', lw=1.5, label='Filtered Signal PSD (Passband 0.5 - 45 Hz Preserved)')
    axes[2].axvspan(0, 0.5, color='gray', alpha=0.15, label='Filtered Baseline Wander (< 0.5 Hz)')
    axes[2].axvspan(45, fs/2, color='red', alpha=0.1, label='Suppressed High-Freq Noise (> 45 Hz)')
    axes[2].set_title("3. Frequency Domain Analysis: Power Spectral Density (PSD)", fontsize=11, fontweight='semibold')
    axes[2].set_xlabel("Frequency (Hz)", fontsize=10)
    axes[2].set_ylabel("Power Density (V²/Hz)", fontsize=10)
    axes[2].set_xlim(0, 100)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Preprocessing comparison chart saved to: {save_path}")


def run_demo():
    """
    Demonstrates the preprocessing pipeline on a downloaded record.
    """
    record_path = "data/raw/100"
    if not os.path.exists(f"{record_path}.dat"):
        print("[!] Record 100 not found locally. Running download first...")
        from download_data import download_mitbih
        download_mitbih(records=['100'])
        
    print(f"\n[*] Loading record 100 from '{record_path}'...")
    record = wfdb.rdrecord(record_path)
    fs = record.fs
    sig_names = record.sig_name
    
    # Select Lead MLII (Channel 0)
    lead_idx = 0
    if 'MLII' in sig_names:
        lead_idx = sig_names.index('MLII')
    raw_signal = record.p_signal[:, lead_idx]
    
    print(f"[*] Initial raw signal stats:")
    print(f"    • Lead: {sig_names[lead_idx]}")
    print(f"    • Total Samples: {len(raw_signal):,}")
    print(f"    • Mean: {np.mean(raw_signal):.4f} mV, Std: {np.std(raw_signal):.4f} mV")
    print(f"    • Min : {np.min(raw_signal):.4f} mV, Max: {np.max(raw_signal):.4f} mV")
    
    # Initialize preprocessor
    preprocessor = ECGPreprocessor(sampling_rate=fs, lowcut=0.5, highcut=45.0, order=3, powerline=60.0)
    
    print("\n[*] Applying Preprocessing Pipeline:")
    print("    1. Zero-phase Butterworth Bandpass (0.5 Hz - 45 Hz, Order 3)")
    print("    2. IIR Notch Filter (60 Hz powerline hum)")
    print("    3. Z-score Normalization ((x - mean) / std)")
    
    clean_signal = preprocessor.preprocess_pipeline(raw_signal, normalize_method='zscore')
    
    print(f"[*] Preprocessed signal stats:")
    print(f"    • Mean: {np.mean(clean_signal):.6f} (stabilized at ~0.0)")
    print(f"    • Std : {np.std(clean_signal):.6f} (scaled to 1.0)")
    print(f"    • Min : {np.min(clean_signal):.4f}, Max: {np.max(clean_signal):.4f}")
    
    # Generate visual comparison
    plot_preprocessing_comparison(
        raw_ecg=raw_signal,
        clean_ecg=clean_signal,
        fs=fs,
        duration_sec=8,
        record_id="100",
        lead_name=sig_names[lead_idx],
        save_path="figures/ecg_preprocessing_comparison.png"
    )
    
    # Test also on Record 119 if available
    rec119_path = "data/raw/119"
    if os.path.exists(f"{rec119_path}.dat"):
        print(f"\n[*] Also processing Record 119 (frequent PVC arrhythmias)...")
        rec119 = wfdb.rdrecord(rec119_path)
        raw119 = rec119.p_signal[:, 0]
        clean119 = preprocessor.preprocess_pipeline(raw119, normalize_method='zscore')
        plot_preprocessing_comparison(
            raw_ecg=raw119,
            clean_ecg=clean119,
            fs=rec119.fs,
            duration_sec=8,
            record_id="119",
            lead_name=rec119.sig_name[0],
            save_path="figures/ecg_preprocessing_record_119.png"
        )
        
    print("\n[+] Phase 2 Preprocessing execution completed successfully!")


if __name__ == '__main__':
    run_demo()
