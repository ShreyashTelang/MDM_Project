"""
segmentation.py
===============
ECG R-Peak Detection, Beat Segmentation, and Patient-Level Dataset Builder.

Covers:
1. Automated R-peak detection (Pan-Tompkins / NeuroKit2 / SciPy adaptive threshold).
2. Ground-truth cardiologist fiducial extraction from MIT-BIH annotations.
3. Fixed-window beat segmentation (360 samples = 1.0 second: 144 pre-R, 216 post-R).
4. Heart Rate (BPM) calculation from R-R intervals.
5. Patient-level dataset partitioning (DS1 Train/Val vs DS2 Unseen Test) preventing data leakage.
6. Exporting .npy tensors (X_train, y_train, X_val, y_val, X_test, y_test) and metadata.json.
7. Diagnostic visualization of segmented waveforms across all 4 arrhythmia classes.
"""

import os
import json
import glob
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import wfdb

from preprocessing import ECGPreprocessor

try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False

# Class definitions
CLASS_NAMES = ['Normal Beat', 'Supraventricular Ectopic (SVEB)', 'Premature Ventricular (PVC)', 'Other Arrhythmia']

# AAMI EC57 to 4-Class Mapping
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal & Bundle Branch
    'A': 1, 'a': 1, 'J': 1, 'S': 1,           # Supraventricular Ectopics (PAC / Atrial)
    'V': 2, 'E': 2,                           # Ventricular Ectopics (PVC)
    'F': 3, '/': 3, 'f': 3, 'Q': 3, '?': 3   # Fusion, Paced & Unclassified
}

# Standard Inter-Patient Partition (de Chazal et al., 2004)
DS1_TRAIN_VAL_RECORDS = [
    '101', '106', '108', '109', '112', '114', '115', '116',
    '118', '119', '122', '124', '201', '203', '205', '207',
    '208', '209', '215', '220', '223', '230'
]

DS2_TEST_RECORDS = [
    '100', '103', '105', '111', '113', '117', '121', '123',
    '200', '202', '210', '212', '213', '214', '219', '221',
    '222', '228', '231', '232', '233', '234'
]


def detect_r_peaks_pantompkins(ecg_signal, fs=360):
    """
    Standard Pan-Tompkins QRS Detection Algorithm implementation.
    Steps:
      1. Bandpass filter (5 - 15 Hz)
      2. Five-point derivative operator
      3. Amplitude squaring
      4. Moving window integration (MWI)
      5. Dual-threshold peak search
    """
    signal_arr = np.array(ecg_signal, dtype=np.float64)
    nyq = 0.5 * fs
    
    # 1. Bandpass 5 - 15 Hz
    b, a = signal.butter(2, [5.0 / nyq, 15.0 / nyq], btype='bandpass')
    filtered = signal.filtfilt(b, a, signal_arr)
    
    # 2. Derivative
    diff = np.diff(filtered)
    diff = np.pad(diff, (1, 0), mode='edge')
    
    # 3. Squaring
    squared = diff ** 2
    
    # 4. Moving Window Integration (approx 150 ms)
    window_len = int(0.150 * fs)
    mwi = np.convolve(squared, np.ones(window_len) / window_len, mode='same')
    
    # 5. Peak Finding on MWI
    min_dist = int(0.200 * fs)  # Refractory period: minimum 200 ms between heartbeats (max 300 bpm)
    threshold = np.mean(mwi) + 0.5 * np.std(mwi)
    peaks, _ = signal.find_peaks(mwi, height=threshold, distance=min_dist)
    
    # Refine peak to the local maximum in original filtered signal within +/- 50 ms window
    search_win = int(0.050 * fs)
    r_peaks = []
    for p in peaks:
        left = max(0, p - search_win)
        right = min(len(signal_arr), p + search_win)
        if right > left:
            local_max = left + np.argmax(np.abs(signal_arr[left:right]))
            r_peaks.append(local_max)
            
    r_peaks = np.unique(r_peaks)
    return r_peaks


def detect_r_peaks(ecg_signal, fs=360, method='auto'):
    """
    Detects R-peaks using NeuroKit2 (if installed) or native Pan-Tompkins.
    """
    if method in ['auto', 'neurokit'] and HAS_NEUROKIT:
        try:
            _, rpeaks_dict = nk.ecg_peaks(ecg_signal, sampling_rate=fs)
            return np.array(rpeaks_dict['ECG_R_Peaks'], dtype=int)
        except Exception:
            pass
    return detect_r_peaks_pantompkins(ecg_signal, fs=fs)


def calculate_heart_rate(r_peaks, fs=360):
    """
    Calculates estimated Heart Rate (BPM) from R-R peak sample intervals.
    Returns:
    --------
    bpm : float
        Mean heart rate in beats per minute.
    rr_intervals_ms : np.ndarray
        Array of R-R intervals in milliseconds.
    """
    if len(r_peaks) < 2:
        return 0.0, np.array([])
        
    diffs = np.diff(r_peaks)
    rr_ms = (diffs / fs) * 1000.0
    
    # Filter physiologically plausible R-R intervals (300 ms to 2000 ms, corresponding to 30 - 200 BPM)
    valid_rr = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
    if len(valid_rr) == 0:
        valid_rr = rr_ms
        
    mean_rr_sec = np.median(valid_rr) / 1000.0
    bpm = 60.0 / mean_rr_sec
    return float(np.round(bpm, 1)), valid_rr


def extract_beat_windows(ecg_signal, r_peaks, window_before=144, window_after=216):
    """
    Segments continuous ECG around R-peaks into fixed-size windows.
    Default: 144 pre-R + 216 post-R = 360 samples (exactly 1.0 second at 360 Hz).
    """
    total_samples = len(ecg_signal)
    total_window = window_before + window_after
    beats = []
    valid_peaks = []
    
    for r in r_peaks:
        start = r - window_before
        end = r + window_after
        if start >= 0 and end <= total_samples:
            beat = ecg_signal[start:end]
            beats.append(beat)
            valid_peaks.append(r)
            
    if len(beats) == 0:
        return np.empty((0, total_window)), np.array([], dtype=int)
        
    return np.array(beats, dtype=np.float32), np.array(valid_peaks, dtype=int)


def segment_record(rec_id, data_dir='data/raw', preprocessor=None, window_before=144, window_after=216):
    """
    Processes a single MIT-BIH record:
    1. Reads Lead MLII
    2. Conditions signal via preprocessor
    3. Matches cardiologist annotations to 4 classes
    4. Segments beats into window arrays
    """
    rec_path = os.path.join(data_dir, rec_id)
    record = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, 'atr')
    
    # Select Lead MLII
    sig_names = record.sig_name
    lead_idx = 0
    if 'MLII' in sig_names:
        lead_idx = sig_names.index('MLII')
    elif 'II' in sig_names:
        lead_idx = sig_names.index('II')
        
    raw_signal = record.p_signal[:, lead_idx]
    
    # Preprocess
    if preprocessor is None:
        preprocessor = ECGPreprocessor(sampling_rate=record.fs)
    clean_signal = preprocessor.preprocess_pipeline(raw_signal)
    
    # Filter annotations for valid AAMI classes
    r_peaks = []
    labels = []
    symbols = []
    
    for sample_idx, sym in zip(ann.sample, ann.symbol):
        if sym in AAMI_MAPPING:
            class_idx = AAMI_MAPPING[sym]
            r_peaks.append(sample_idx)
            labels.append(class_idx)
            symbols.append(sym)
            
    # Extract windows
    beats = []
    beat_labels = []
    beat_symbols = []
    total_len = len(clean_signal)
    
    for r, lbl, sym in zip(r_peaks, labels, symbols):
        start = r - window_before
        end = r + window_after
        if start >= 0 and end <= total_len:
            beat = clean_signal[start:end]
            beats.append(beat)
            beat_labels.append(lbl)
            beat_symbols.append(sym)
            
    beats = np.array(beats, dtype=np.float32)
    beat_labels = np.array(beat_labels, dtype=np.int32)
    return beats, beat_labels, beat_symbols, clean_signal, record.fs


def build_dataset(data_dir='data/raw', output_dir='data/processed',
                  window_before=144, window_after=216, val_ratio=0.15):
    """
    Partitions all available records into:
    - Train & Validation Sets (DS1 Patients)
    - Unseen Test Set (DS2 Patients)
    
    Prevents intra-patient data leakage by strictly keeping test patients out of training.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    hea_files = glob.glob(os.path.join(data_dir, "*.hea"))
    available_records = sorted([os.path.basename(f).replace('.hea', '') for f in hea_files])
    
    if len(available_records) == 0:
        raise FileNotFoundError(f"No records found in '{data_dir}'. Run download_data.py first.")
        
    print("=" * 65)
    print(f" BUILDING DATASET FROM {len(available_records)} PATIENT RECORDS ")
    print("=" * 65)
    
    preprocessor = ECGPreprocessor(sampling_rate=360)
    
    train_val_records = [r for r in available_records if r in DS1_TRAIN_VAL_RECORDS]
    test_records = [r for r in available_records if r in DS2_TEST_RECORDS]
    
    # If a record isn't in standard benchmark, place in train_val
    for r in available_records:
        if r not in train_val_records and r not in test_records:
            train_val_records.append(r)
            
    # If no records in test partition, split available records evenly
    if len(test_records) == 0:
        split_idx = int(len(available_records) * 0.7)
        train_val_records = available_records[:split_idx]
        test_records = available_records[split_idx:]
        
    print(f" • DS1 Training & Validation Records : {train_val_records}")
    print(f" • DS2 Completely Unseen Test Records: {test_records}")
    
    # 1. Process DS1 (Train/Val)
    ds1_beats, ds1_labels = [], []
    for r in train_val_records:
        print(f"    --> Segmenting DS1 Record {r}...", end=' ', flush=True)
        beats, labels, _, _, _ = segment_record(r, data_dir, preprocessor, window_before, window_after)
        ds1_beats.append(beats)
        ds1_labels.append(labels)
        print(f"[{len(beats):,} beats]")
        
    ds1_beats = np.concatenate(ds1_beats, axis=0)
    ds1_labels = np.concatenate(ds1_labels, axis=0)
    
    # 2. Process DS2 (Unseen Test)
    ds2_beats, ds2_labels = [], []
    for r in test_records:
        print(f"    --> Segmenting DS2 Record {r}...", end=' ', flush=True)
        beats, labels, _, _, _ = segment_record(r, data_dir, preprocessor, window_before, window_after)
        ds2_beats.append(beats)
        ds2_labels.append(labels)
        print(f"[{len(beats):,} beats]")
        
    ds2_beats = np.concatenate(ds2_beats, axis=0)
    ds2_labels = np.concatenate(ds2_labels, axis=0)
    
    # Split DS1 into Train and Validation sets (85% train, 15% validation)
    np.random.seed(42)
    indices = np.arange(len(ds1_beats))
    np.random.shuffle(indices)
    
    val_size = int(len(indices) * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_train = ds1_beats[train_idx]
    y_train = ds1_labels[train_idx]
    
    X_val = ds1_beats[val_idx]
    y_val = ds1_labels[val_idx]
    
    X_test = ds2_beats
    y_test = ds2_labels
    
    # Reshape for 1D CNN: (num_samples, window_size, 1_channel)
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # Compute balanced class weights for training: w_j = Total / (K * count_j)
    total_train = len(y_train)
    num_classes = len(CLASS_NAMES)
    class_weights = {}
    train_counts = {}
    for c in range(num_classes):
        cnt = int(np.sum(y_train == c))
        train_counts[CLASS_NAMES[c]] = cnt
        class_weights[c] = float(np.round(total_train / (num_classes * max(cnt, 1)), 4))
        
    print("\n" + "=" * 65)
    print(" DATASET SUMMARY ")
    print("=" * 65)
    print(f" • Training Set   (X_train): {X_train.shape} | Labels (y_train): {y_train.shape}")
    print(f" • Validation Set (X_val)  : {X_val.shape}   | Labels (y_val)  : {y_val.shape}")
    print(f" • Test Set       (X_test) : {X_test.shape}  | Labels (y_test) : {y_test.shape}")
    print(f"\nCalculated Training Class Weights (to handle class imbalance):")
    for c in range(num_classes):
        print(f"   Class {c} ({CLASS_NAMES[c]:<30}): {train_counts[CLASS_NAMES[c]]:5d} beats | Weight = {class_weights[c]:.4f}")
        
    # Save .npy arrays to disk
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    # Save metadata JSON
    metadata = {
        'sampling_rate': 360,
        'window_size': window_before + window_after,
        'window_before': window_before,
        'window_after': window_after,
        'classes': CLASS_NAMES,
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'test_samples': int(len(X_test)),
        'ds1_train_records': train_val_records,
        'ds2_test_records': test_records,
        'class_weights': class_weights
    }
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\n[+] Successfully saved preprocessed tensors and metadata to '{output_dir}/'")
    
    # Visualization 1: Representative Beat per Class
    plot_representative_beats(X_train, y_train, window_before, window_after, save_path="figures/segmented_beat_classes.png")
    
    # Visualization 2: Dataset Partition Distribution
    plot_split_distribution(y_train, y_val, y_test, save_path="figures/dataset_split_distribution.png")
    
    return metadata


def plot_representative_beats(X, y, window_before=144, window_after=216, save_path="figures/segmented_beat_classes.png"):
    """
    Plots an average template and sample heartbeat waveform for each of the 4 classes.
    """
    fs = 360
    total_len = window_before + window_after
    time_ms = (np.arange(total_len) - window_before) / fs * 1000.0  # Time in ms relative to R-peak (0 ms)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    colors = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    
    for c in range(4):
        c_idx = np.where(y == c)[0]
        ax = axes[c]
        if len(c_idx) > 0:
            # Plot individual sample beats (semi-transparent)
            sample_subset = c_idx[:min(10, len(c_idx))]
            for idx in sample_subset:
                ax.plot(time_ms, X[idx, :, 0], color=colors[c], alpha=0.25, lw=1.0)
                
            # Plot mean beat template
            mean_beat = np.mean(X[c_idx, :, 0], axis=0)
            ax.plot(time_ms, mean_beat, color=colors[c], lw=2.2, label=f"Mean Template (n={len(c_idx):,})")
            
        ax.axvline(0, color='gray', linestyle='--', alpha=0.7, label='R-Peak Center')
        ax.set_title(f"Class {c}: {CLASS_NAMES[c]}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time relative to R-Peak (ms)", fontsize=10)
        ax.set_ylabel("Normalized Amplitude", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize=9)
        
    plt.suptitle("Segmented ECG Heartbeat Waveforms (1.0 Second / 360 Samples Centered on R-Peak)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Representative beats chart saved to: {save_path}")


def plot_split_distribution(y_train, y_val, y_test, save_path="figures/dataset_split_distribution.png"):
    """
    Plots the count of beats per class across Train, Validation, and Test sets.
    """
    splits = ['Train (DS1)', 'Validation (DS1)', 'Unseen Test (DS2)']
    counts = np.zeros((4, 3))
    
    for c in range(4):
        counts[c, 0] = np.sum(y_train == c)
        counts[c, 1] = np.sum(y_val == c)
        counts[c, 2] = np.sum(y_test == c)
        
    x = np.arange(4)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(11, 6))
    rects1 = ax.bar(x - width, counts[:, 0], width, label=splits[0], color='#1f77b4')
    rects2 = ax.bar(x, counts[:, 1], width, label=splits[1], color='#2ca02c')
    rects3 = ax.bar(x + width, counts[:, 2], width, label=splits[2], color='#d62728')
    
    ax.set_title("Heartbeat Distribution Across Inter-Patient Partitions (DS1 vs DS2)", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel("Number of Heartbeats", fontsize=11)
    ax.set_yscale('log')  # Log scale to easily see minority classes
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Dataset partition chart saved to: {save_path}")


if __name__ == '__main__':
    build_dataset(data_dir='data/raw', output_dir='data/processed')
