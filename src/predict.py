"""
predict.py
==========
Phase 7: End-to-End ECG Prediction and Clinical Screening Pipeline.

Accepts continuous raw ECG signals (CSV, TXT, numpy, or WFDB records),
performs rigorous validation, applies conditioning filters, detects R-peaks,
calculates Heart Rate (BPM), runs batch inference with the 1D CNN,
and outputs a comprehensive diagnostic screening report.
"""

import os
import json
import numpy as np
import pandas as pd

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocessing import ECGPreprocessor
from segmentation import detect_r_peaks_pantompkins, calculate_heart_rate, extract_beat_windows

CLASS_NAMES = ['Normal Beat', 'Supraventricular Ectopic (SVEB)', 'Premature Ventricular (PVC)', 'Other Arrhythmia']
DEFAULT_FS = 360
MODEL_PATH = "models/ecg_cnn_model.keras"


class ECGInferenceEngine:
    """
    Production-ready inference engine for automated ECG waveform analysis.
    """
    
    def __init__(self, model_path=MODEL_PATH, config_path="models/inference_config.json"):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = {}
        self.preprocessor = None
        self._load_model_and_config()
        
    def _load_model_and_config(self):
        """Loads serialized 1D CNN model and inference specifications."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at '{self.model_path}'. Run src/train.py first.")
            
        print(f"[*] Loading 1D CNN model from '{self.model_path}'...", flush=True)
        self.model = keras.models.load_model(self.model_path)
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'sampling_rate': DEFAULT_FS,
                'window_before': 144,
                'window_after': 216,
                'classes': CLASS_NAMES
            }
            
        fs = self.config.get('sampling_rate', DEFAULT_FS)
        self.preprocessor = ECGPreprocessor(sampling_rate=fs, lowcut=0.5, highcut=45.0, order=3, powerline=60.0)
        print("[+] Inference engine initialized successfully.", flush=True)

    def validate_and_load_signal(self, input_source):
        """
        Validates and parses raw ECG input from file path, DataFrame, or numpy array.
        
        Handles:
        - Missing or corrupted file
        - Empty signal
        - Non-numeric / malformed rows
        - Flatline / zero variance signals (disconnected leads)
        - Insufficient signal duration (< 1.0 second)
        """
        if input_source is None:
            raise ValueError("Input ECG signal cannot be None.")
            
        signal_data = None
        
        # 1. Parse from file path
        if isinstance(input_source, str):
            if not os.path.exists(input_source):
                raise FileNotFoundError(f"ECG file does not exist: {input_source}")
                
            if os.path.getsize(input_source) == 0:
                raise ValueError("Uploaded ECG file is empty (0 bytes).")
                
            ext = os.path.splitext(input_source)[1].lower()
            
            try:
                if ext in ['.csv', '.txt']:
                    # Try reading with pandas, auto-detecting header and delimiter
                    df = pd.read_csv(input_source, sep=None, engine='python')
                    
                    # If multiple columns, look for ECG/voltage column or take first numerical column
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    if len(num_cols) == 0:
                        raise ValueError("No numeric voltage data found in CSV/TXT file.")
                        
                    # Prefer columns named 'ecg', 'mlii', 'lead', 'voltage', 'signal'
                    chosen_col = num_cols[0]
                    for col in num_cols:
                        if any(k in str(col).lower() for k in ['ecg', 'mlii', 'lead', 'volt', 'sig']):
                            chosen_col = col
                            break
                    signal_data = df[chosen_col].values.astype(np.float64)
                    
                elif ext == '.npy':
                    signal_data = np.load(input_source).flatten().astype(np.float64)
                    
                else:
                    raise ValueError(f"Unsupported file format '{ext}'. Please upload .csv, .txt, or .npy.")
            except Exception as e:
                raise ValueError(f"Failed to read ECG file: {str(e)}")
                
        # 2. Parse from numpy array or list
        elif isinstance(input_source, (np.ndarray, list)):
            signal_data = np.array(input_source, dtype=np.float64).flatten()
            
        else:
            raise ValueError("Invalid input format. Expected file path, numpy array, or list.")
            
        # 3. Rigorous Signal Integrity Checks
        if len(signal_data) == 0:
            raise ValueError("Signal is empty. No valid numeric samples found.")
            
        # Check for NaNs/Infs
        if np.isnan(signal_data).any() or np.isinf(signal_data).any():
            valid_mask = np.isfinite(signal_data)
            if np.sum(valid_mask) < 360:
                raise ValueError("Signal contains too many NaN/corrupted values.")
            x = np.arange(len(signal_data))
            signal_data = np.interp(x, x[valid_mask], signal_data[valid_mask])
            
        # Minimum sample requirement: at least 1.0 second (360 samples)
        fs = self.config.get('sampling_rate', DEFAULT_FS)
        if len(signal_data) < fs:
            raise ValueError(f"Signal is too short ({len(signal_data)} samples). Minimum required is {fs} samples (1.0 second).")
            
        # Check for disconnected electrode / flatline
        std_val = np.std(signal_data)
        if std_val < 1e-4:
            raise ValueError("Signal is flatline or has near-zero variance. Check electrode connection.")
            
        return signal_data

    def predict_ecg(self, input_source, max_waveform_points=2000):
        """
        Executes complete screening pipeline on raw ECG:
        1. Validation & Loading
        2. Bandpass (0.5 - 45 Hz) & Notch Filtering
        3. Automated R-peak detection & Heart Rate (BPM)
        4. Heartbeat Windowing (360 samples per beat)
        5. 1D CNN Batch Classification
        6. Clinical Aggregation & Report Generation
        """
        # 1. Validation
        raw_signal = self.validate_and_load_signal(input_source)
        fs = self.config.get('sampling_rate', DEFAULT_FS)
        duration_sec = len(raw_signal) / fs
        
        # 2. Preprocessing
        clean_signal = self.preprocessor.preprocess_pipeline(raw_signal, normalize_method='zscore')
        
        # 3. R-Peak Detection & Heart Rate
        r_peaks = detect_r_peaks_pantompkins(clean_signal, fs=fs)
        bpm, rr_intervals = calculate_heart_rate(r_peaks, fs=fs)
        
        # If no R-peaks detected
        if len(r_peaks) == 0:
            return {
                'status': 'Error: No valid QRS complexes detected',
                'prediction': 'Unclassifiable',
                'confidence': 0.0,
                'heart_rate': 0.0,
                'total_beats': 0,
                'disclaimer': 'Educational screening tool only. Not for medical diagnosis.'
            }
            
        # 4. Beat Segmentation
        win_before = self.config.get('window_before', 144)
        win_after = self.config.get('window_after', 216)
        beats, valid_peaks = extract_beat_windows(clean_signal, r_peaks, win_before, win_after)
        
        if len(beats) == 0:
            return {
                'status': 'Error: Heartbeats too close to signal edges for complete windowing',
                'prediction': 'Unclassifiable',
                'confidence': 0.0,
                'heart_rate': bpm,
                'total_beats': 0,
                'disclaimer': 'Educational screening tool only. Not for medical diagnosis.'
            }
            
        # 5. 1D CNN Batch Prediction
        beats_input = np.expand_dims(beats, axis=-1)  # (num_beats, 360, 1)
        probabilities = self.model.predict(beats_input, verbose=0)
        predicted_classes = np.argmax(probabilities, axis=-1)
        max_confidences = np.max(probabilities, axis=-1)
        
        # 6. Clinical Aggregation
        total_beats = len(predicted_classes)
        class_counts = {c: int(np.sum(predicted_classes == c)) for c in range(4)}
        
        normal_count = class_counts[0]
        sveb_count = class_counts[1]
        pvc_count = class_counts[2]
        other_count = class_counts[3]
        abnormal_count = sveb_count + pvc_count + other_count
        abnormal_pct = (abnormal_count / total_beats) * 100.0
        
        # Determine dominant clinical rhythm
        if pvc_count > 0 and (pvc_count / total_beats >= 0.05 or pvc_count >= 3):
            primary_class_idx = 2  # PVC
            pred_label = "Possible Arrhythmia (Premature Ventricular Contraction detected)"
            status_text = "Abnormal pattern detected — professional medical evaluation is recommended."
            is_abnormal = True
        elif sveb_count > 0 and (sveb_count / total_beats >= 0.05 or sveb_count >= 3):
            primary_class_idx = 1  # SVEB
            pred_label = "Possible Arrhythmia (Supraventricular Ectopic Beat detected)"
            status_text = "Abnormal pattern detected — professional medical evaluation is recommended."
            is_abnormal = True
        elif other_count > 0 and (other_count / total_beats >= 0.05 or other_count >= 3):
            primary_class_idx = 3  # Other
            pred_label = "Possible Arrhythmia (Unclassified Rhythm / Paced pattern detected)"
            status_text = "Abnormal pattern detected — professional medical evaluation is recommended."
            is_abnormal = True
        else:
            primary_class_idx = 0  # Normal
            pred_label = "Normal Sinus Rhythm"
            status_text = "No abnormal pattern detected by the model."
            is_abnormal = False
            
        overall_confidence = float(np.mean(max_confidences) * 100.0)
        
        # 7. Prepare Downsampled Waveform for Web Rendering (Chart.js / Plotly)
        # Limit to first N samples for fast browser charting
        plot_limit = min(len(clean_signal), max_waveform_points)
        time_axis = (np.arange(plot_limit) / fs).tolist()
        raw_sample = raw_signal[:plot_limit].tolist()
        clean_sample = clean_signal[:plot_limit].tolist()
        
        # Filter peaks that fall inside the plot limit
        peaks_in_view = [int(p) for p in valid_peaks if p < plot_limit]
        peak_times = [float(p / fs) for p in peaks_in_view]
        peak_voltages = [float(clean_signal[p]) for p in peaks_in_view]
        
        # Beat breakdown dictionary
        breakdown = {
            'Normal': {'count': normal_count, 'percentage': round(normal_count / total_beats * 100, 1)},
            'SVEB': {'count': sveb_count, 'percentage': round(sveb_count / total_beats * 100, 1)},
            'PVC': {'count': pvc_count, 'percentage': round(pvc_count / total_beats * 100, 1)},
            'Other': {'count': other_count, 'percentage': round(other_count / total_beats * 100, 1)}
        }
        
        # Build final response object
        response = {
            'success': True,
            'prediction': pred_label,
            'primary_class': CLASS_NAMES[primary_class_idx],
            'confidence': round(overall_confidence, 1),
            'status': status_text,
            'is_abnormal': is_abnormal,
            'heart_rate_bpm': bpm,
            'total_beats_analyzed': total_beats,
            'recording_duration_sec': round(duration_sec, 2),
            'sampling_rate_hz': fs,
            'breakdown': breakdown,
            'waveform': {
                'time': [round(t, 4) for t in time_axis],
                'raw': [round(v, 4) for v in raw_sample],
                'clean': [round(v, 4) for v in clean_sample],
                'peak_times': [round(t, 4) for t in peak_times],
                'peak_voltages': [round(v, 4) for v in peak_voltages]
            },
            'disclaimer': (
                "This system is intended for educational and research purposes only. "
                "It does not provide medical diagnosis or replace evaluation by a qualified healthcare professional."
            )
        }
        
        return response


# Singleton instance for server reuse
_engine_instance = None

def get_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ECGInferenceEngine()
    return _engine_instance


if __name__ == '__main__':
    # Quick self-test
    engine = get_engine()
    test_file = "data/raw/100"
    if os.path.exists(f"{test_file}.dat"):
        import wfdb
        rec = wfdb.rdrecord(test_file)
        res = engine.predict_ecg(rec.p_signal[:, 0])
        print("\n=== Test Inference on Record 100 ===")
        print(f"Prediction : {res['prediction']}")
        print(f"Confidence : {res['confidence']}%")
        print(f"Heart Rate : {res['heart_rate_bpm']} BPM")
        print(f"Status     : {res['status']}")
        print(f"Total Beats: {res['total_beats_analyzed']}")
        print(f"Breakdown  : {res['breakdown']}")
