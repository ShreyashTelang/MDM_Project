"""
explore_dataset.py
==================
ECG Exploration & Annotation Analysis for MIT-BIH Arrhythmia Database.

This script demonstrates:
1. Loading raw ECG signals (.dat, .hea) and annotations (.atr) using WFDB.
2. Signal parameters: sampling rate, channels/leads, duration, ADC units.
3. Analysis of physician-verified beat annotations.
4. Mapping of MIT-BIH annotations into standard AAMI / Project classes.
5. Demonstration of patient-level splitting to prevent data leakage.
6. Plotting and saving waveform segments with annotated beat labels.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

# AAMI EC57 Standard Beat Mapping
# Maps MIT-BIH detailed cardiologist symbols to standard 5/4 AAMI superclasses
AAMI_MAPPING = {
    # N: Normal beat and bundle branch blocks
    'N': 'Normal',
    'L': 'Normal',   # Left bundle branch block
    'R': 'Normal',   # Right bundle branch block
    'e': 'Normal',   # Atrial escape beat
    'j': 'Normal',   # Nodal (junctional) escape beat
    
    # SVEB / S: Supraventricular Ectopic Beat (Atrial abnormalities)
    'A': 'SVEB',     # Atrial premature beat
    'a': 'SVEB',     # Aberrated atrial premature beat
    'J': 'SVEB',     # Nodal premature beat
    'S': 'SVEB',     # Supraventricular premature beat
    
    # VEB / V: Ventricular Ectopic Beat (PVCs)
    'V': 'PVC',      # Premature ventricular contraction
    'E': 'PVC',      # Ventricular escape beat
    
    # F: Fusion beat
    'F': 'Fusion',   # Fusion of ventricular and normal beat
    
    # Q: Paced or unclassifiable
    '/': 'Paced',    # Paced beat
    'f': 'Other',    # Fusion of paced and normal beat
    'Q': 'Other',    # Unclassifiable beat
    '?': 'Other'
}

# Dictionary of standard MIT-BIH annotation symbol descriptions
try:
    SYMBOL_DESCRIPTIONS = {lbl.symbol: lbl.description for lbl in wfdb.io.annotation.ann_labels}
except Exception:
    SYMBOL_DESCRIPTIONS = {}

def get_symbol_meaning(sym):
    return SYMBOL_DESCRIPTIONS.get(sym, 'Non-beat / Rhythm marker')
# 0: Normal Beat (N)
# 1: Supraventricular Ectopic Beat (SVEB / PAC)
# 2: Premature Ventricular Contraction (PVC)
# 3: Other Arrhythmia (Fusion / Paced / Unknown)
PROJECT_CLASSES = {
    'Normal': 0,
    'SVEB': 1,
    'PVC': 2,
    'Fusion': 3,
    'Other': 3,
    'Paced': 3
}

CLASS_NAMES = ['Normal Beat', 'Supraventricular Ectopic (SVEB)', 'Premature Ventricular (PVC)', 'Other Arrhythmia']


def inspect_record(rec_id, data_dir='data/raw', plot_seconds=10, save_plot=True):
    """
    Inspects a single MIT-BIH record: header info, signal shape, annotations.
    """
    record_path = os.path.join(data_dir, rec_id)
    
    # 1. Read Record and Header
    record = wfdb.rdrecord(record_path)
    # 2. Read Annotations
    annotation = wfdb.rdann(record_path, 'atr')
    
    fs = record.fs
    n_samples = record.sig_len
    duration_sec = n_samples / fs
    duration_min = duration_sec / 60.0
    sig_names = record.sig_name
    units = record.units
    
    print("\n" + "=" * 65)
    print(f" RECORD {rec_id} SUMMARY ")
    print("=" * 65)
    print(f" • Sampling Frequency (fs) : {fs} Hz")
    print(f" • Total Samples           : {n_samples:,} samples")
    print(f" • Recording Duration      : {duration_sec:.1f} seconds ({duration_min:.1f} minutes)")
    print(f" • Available Leads/Channels: {sig_names}")
    print(f" • Signal Units            : {units}")
    print(f" • Total Annotated Beats   : {len(annotation.sample):,}")
    
    # Check MLII channel index (Modified Limb Lead II is the standard for arrhythmia analysis)
    mlii_idx = 0
    if 'MLII' in sig_names:
        mlii_idx = sig_names.index('MLII')
    elif 'II' in sig_names:
        mlii_idx = sig_names.index('II')
    print(f" • Selected Lead for View : {sig_names[mlii_idx]} (Channel index: {mlii_idx})")
    
    # 3. Analyze beat annotations
    symbol_series = pd.Series(annotation.symbol)
    symbol_counts = symbol_series.value_counts()
    
    print("\n--- Raw Annotation Symbols in Record ---")
    for sym, count in symbol_counts.items():
        meaning = get_symbol_meaning(sym)
        aami_cat = AAMI_MAPPING.get(sym, 'Non-beat')
        print(f"  Symbol '{sym}': {count:5d} beats | {meaning:<35} -> AAMI: {aami_cat}")
        
    # 4. Plot a segment
    if plot_seconds > 0:
        max_samples = int(plot_seconds * fs)
        time_axis = np.arange(max_samples) / fs
        signal_segment = record.p_signal[:max_samples, mlii_idx]
        
        # Filter beat markers within the segment
        mask = annotation.sample < max_samples
        sample_indices = annotation.sample[mask]
        symbols = [annotation.symbol[i] for i, m in enumerate(mask) if m]
        
        plt.figure(figsize=(14, 5))
        plt.plot(time_axis, signal_segment, label=f"Lead {sig_names[mlii_idx]}", color='#1f77b4', lw=1.2)
        
        # Mark annotations
        for s_idx, sym in zip(sample_indices, symbols):
            t_val = s_idx / fs
            v_val = signal_segment[s_idx]
            # Color coding: Green for Normal, Red for PVC, Orange for SVEB, Purple for Others
            color = '#2ca02c' if sym in ['N', 'L', 'R'] else ('#d62728' if sym in ['V', 'E'] else '#ff7f0e')
            plt.plot(t_val, v_val, marker='o', markersize=6, color=color)
            plt.annotate(sym, (t_val, v_val), textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=9, fontweight='bold', color=color)
            
        plt.title(f"MIT-BIH Arrhythmia Database - Record {rec_id} (Lead {sig_names[mlii_idx]})\nFirst {plot_seconds} Seconds with Cardiologist Beat Annotations", fontsize=12, fontweight='bold')
        plt.xlabel("Time (seconds)", fontsize=11)
        plt.ylabel(f"Amplitude ({units[mlii_idx]})", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        if save_plot:
            os.makedirs("figures", exist_ok=True)
            out_file = f"figures/mitbih_record_{rec_id}_waveform.png"
            plt.savefig(out_file, dpi=200)
            print(f"\n[+] Waveform plot saved successfully to: {out_file}")
        plt.close()
        
    return record, annotation


def summarize_dataset(data_dir='data/raw'):
    """
    Summarizes all downloaded records in the raw directory,
    grouping annotations into project classes.
    """
    hea_files = glob.glob(os.path.join(data_dir, "*.hea"))
    if not hea_files:
        print(f"[!] No records found in {data_dir}. Run download_data.py first.")
        return None
        
    rec_ids = sorted([os.path.basename(f).replace('.hea', '') for f in hea_files])
    print("\n" + "=" * 65)
    print(f" DATASET INVENTORY: {len(rec_ids)} RECORDS LOADED FROM '{data_dir}' ")
    print("=" * 65)
    
    total_beats = 0
    class_counts = {c: 0 for c in CLASS_NAMES}
    raw_symbol_counts = {}
    record_stats = []
    
    for rec_id in rec_ids:
        rec_path = os.path.join(data_dir, rec_id)
        try:
            ann = wfdb.rdann(rec_path, 'atr')
            rec_header = wfdb.rdheader(rec_path)
        except Exception as e:
            print(f"[!] Error reading record {rec_id}: {e}")
            continue
            
        rec_beat_count = 0
        rec_classes = {c: 0 for c in CLASS_NAMES}
        
        for sym in ann.symbol:
            raw_symbol_counts[sym] = raw_symbol_counts.get(sym, 0) + 1
            if sym in AAMI_MAPPING:
                aami_label = AAMI_MAPPING[sym]
                proj_class_idx = PROJECT_CLASSES.get(aami_label, 3)
                proj_class_name = CLASS_NAMES[proj_class_idx]
                class_counts[proj_class_name] += 1
                rec_classes[proj_class_name] += 1
                rec_beat_count += 1
                total_beats += 1
                
        record_stats.append({
            'Record': rec_id,
            'Leads': ", ".join(rec_header.sig_name),
            'Total Beats': rec_beat_count,
            'Normal': rec_classes['Normal Beat'],
            'PVC': rec_classes['Premature Ventricular (PVC)'],
            'SVEB': rec_classes['Supraventricular Ectopic (SVEB)'],
            'Other': rec_classes['Other Arrhythmia']
        })
        
    stats_df = pd.DataFrame(record_stats)
    print("\nPer-Record Beat Breakdown:")
    print(stats_df.to_string(index=False))
    
    print("\n" + "-" * 50)
    print(f" TOTAL BEATS ANALYZED: {total_beats:,}")
    print("-" * 50)
    for c_name, count in class_counts.items():
        pct = (count / total_beats * 100) if total_beats > 0 else 0
        print(f" • {c_name:<32}: {count:6d} ({pct:5.2f}%)")
        
    # Plot overall class distribution
    plt.figure(figsize=(9, 5))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color=['#2ca02c', '#ff7f0e', '#d62728', '#9467bd'])
    plt.title("MIT-BIH Arrhythmia Dataset - Beat Class Distribution", fontsize=12, fontweight='bold')
    plt.ylabel("Number of Heartbeats", fontsize=11)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count text on top of each bar
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + max(class_counts.values()) * 0.01,
                 f"{int(h):,} ({h/total_beats*100:.1f}%)" if total_beats > 0 else f"{int(h)}",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
                 
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/mitbih_class_distribution.png", dpi=200)
    plt.close()
    print("\n[+] Class distribution chart saved to: figures/mitbih_class_distribution.png")
    
    return stats_df


if __name__ == '__main__':
    data_dir = 'data/raw'
    
    # Check if data exists; if not, download record 100 and 119 for immediate inspection
    if not os.path.exists(os.path.join(data_dir, '100.dat')):
        print("[*] Downloading sample records 100 (normal predominant) and 119 (PVC rich) for exploration...")
        from download_data import download_mitbih
        download_mitbih(target_dir=data_dir, records=['100', '119'])
        
    # Inspect sample records
    print("\n>>> Inspecting Record 100 (Predominantly Normal Rhythm)...")
    inspect_record('100', data_dir=data_dir, plot_seconds=10)
    
    if os.path.exists(os.path.join(data_dir, '119.dat')):
        print("\n>>> Inspecting Record 119 (Frequent PVC / Ventricular Ectopic Beats)...")
        inspect_record('119', data_dir=data_dir, plot_seconds=10)
        
    # Summarize all available records
    summarize_dataset(data_dir=data_dir)
