"""
test_pipeline.py
================
Phase 10: Automated Unit & Integration Test Suite.

Verifies:
1. Signal preprocessing (Butterworth bandpass, notch, normalization)
2. QRS R-peak detection & Heart Rate (BPM) estimation
3. 1D CNN model loading & inference output tensor validity
4. Edge cases & error handling (empty signal, flatline, short duration)
5. Database CRUD operations (save, retrieve, delete)
6. Flask web endpoints & JSON REST APIs
"""

import os
import sys
import unittest
import numpy as np

# Ensure root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from src.preprocessing import ECGPreprocessor
from src.segmentation import detect_r_peaks_pantompkins, calculate_heart_rate, extract_beat_windows
from src.predict import get_engine
from src.database import db
from app import app


class TestECGPipeline(unittest.TestCase):
    """Automated test suite for the complete ECG screening system."""

    def setUp(self):
        self.fs = 360
        self.duration = 5.0
        n_samples = int(self.fs * self.duration)
        t = np.arange(n_samples) / self.fs

        # Synthetic clean ECG proxy: periodic pulses at 1.2 Hz (72 BPM) + 0.2 Hz baseline drift
        pulse = np.zeros(n_samples)
        for i in range(int(self.duration * 1.2)):
            idx = int((i + 0.5) * (self.fs / 1.2))
            if idx < n_samples:
                pulse[idx] = 1.5  # R-peak

        # Add baseline wander (0.2 Hz) and powerline hum (60 Hz)
        baseline = 0.5 * np.sin(2 * np.pi * 0.2 * t)
        noise = 0.05 * np.sin(2 * np.pi * 60 * t)
        self.synthetic_signal = pulse + baseline + noise

    def test_01_preprocessing(self):
        """Test Butterworth bandpass, notch, and Z-score normalization."""
        pre = ECGPreprocessor(sampling_rate=self.fs, lowcut=0.5, highcut=45.0, order=3, powerline=60.0)
        clean = pre.preprocess_pipeline(self.synthetic_signal, normalize_method='zscore')

        self.assertEqual(len(clean), len(self.synthetic_signal))
        self.assertAlmostEqual(np.mean(clean), 0.0, places=4, msg="Mean should be zero-centered.")
        self.assertAlmostEqual(np.std(clean), 1.0, places=4, msg="Std should be unit variance.")
        self.assertFalse(np.isnan(clean).any(), "Cleaned signal must not contain NaNs.")

    def test_02_r_peak_detection(self):
        """Test Pan-Tompkins R-peak detector and BPM computation."""
        r_peaks = detect_r_peaks_pantompkins(self.synthetic_signal, fs=self.fs)
        self.assertGreater(len(r_peaks), 2, "Should detect multiple R-peaks.")

        bpm, rr = calculate_heart_rate(r_peaks, fs=self.fs)
        self.assertGreater(bpm, 40.0, "Heart rate should be physiologically valid.")
        self.assertLess(bpm, 180.0, "Heart rate should be physiologically valid.")

    def test_03_beat_windowing(self):
        """Test slicing 360-sample cardiac cycles around R-peaks."""
        r_peaks = np.array([200, 560, 920, 1280])
        beats, valid_peaks = extract_beat_windows(self.synthetic_signal, r_peaks, window_before=144, window_after=216)

        self.assertGreater(len(beats), 0)
        self.assertEqual(beats.shape[1], 360, "Each beat window must be exactly 360 samples.")

    def test_04_cnn_inference_validity(self):
        """Test 1D CNN model inference and output probability distribution."""
        engine = get_engine()
        self.assertIsNotNone(engine.model, "Model must load successfully.")

        # Test batch input of shape (1, 360, 1)
        dummy_beat = np.zeros((1, 360, 1), dtype=np.float32)
        dummy_beat[0, 144, 0] = 3.0  # Synthetic R peak
        probs = engine.model.predict(dummy_beat, verbose=0)

        self.assertEqual(probs.shape, (1, 4), "Output probabilities must have shape (batch, 4).")
        self.assertAlmostEqual(np.sum(probs), 1.0, places=4, msg="Softmax probabilities must sum to 1.0.")

    def test_05_error_handling(self):
        """Test pipeline robustness against empty, flatline, or corrupted inputs."""
        engine = get_engine()

        # 1. Empty input
        with self.assertRaises(ValueError):
            engine.validate_and_load_signal([])

        # 2. Flatline / zero-variance
        flatline = np.zeros(720)
        with self.assertRaises(ValueError):
            engine.validate_and_load_signal(flatline)

        # 3. Insufficient duration (< 360 samples)
        short_sig = np.random.randn(100)
        with self.assertRaises(ValueError):
            engine.validate_and_load_signal(short_sig)

    def test_06_database_crud(self):
        """Test saving, retrieving, and deleting history records."""
        test_result = {
            'prediction': 'Unit Test Normal',
            'primary_class': 'Normal Beat',
            'confidence': 99.0,
            'heart_rate_bpm': 72.0,
            'status': 'Unit test pass',
            'is_abnormal': False,
            'total_beats_analyzed': 5,
            'breakdown': {'Normal': {'count': 5, 'percentage': 100.0}}
        }
        rec_id = db.save_analysis("test_signal.csv", test_result)
        self.assertIsNotNone(rec_id, "Record ID should be generated.")

        records = db.get_all_history(limit=10)
        self.assertGreater(len(records), 0)

        # Test delete
        deleted = db.delete_record(rec_id)
        self.assertTrue(deleted)

    def test_07_flask_endpoints(self):
        """Test Flask web routes and REST API."""
        client = app.test_client()

        # Pages
        for route in ['/', '/analysis', '/history', '/about']:
            res = client.get(route)
            self.assertEqual(res.status_code, 200, f"Route {route} failed.")

        # Health API
        h_res = client.get('/api/health')
        self.assertEqual(h_res.status_code, 200)
        self.assertEqual(h_res.get_json()['status'], 'healthy')

        # Sample API
        s_res = client.get('/api/sample/normal')
        self.assertEqual(s_res.status_code, 200)
        self.assertTrue(s_res.get_json()['success'])


if __name__ == '__main__':
    unittest.main()
