# AI-Based ECG Abnormality Detection and Classification Using 1D CNNs

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://tensorflow.org/)
[![Flask 3.x](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academic capstone project developed by a Computer Engineering student for automated cardiac arrhythmia screening from ambulatory electrocardiogram (ECG) signals using deep learning.

> **Healthcare Safety Requirement**: This system is designed as an **educational and research preliminary screening tool**, not as a medical diagnostic device. It does not provide definitive medical diagnosis or replace evaluation by a qualified healthcare professional.

---

## 📌 Project Architecture

```
MDM/
│
├── app.py                      # Flask web application & REST API
├── requirements.txt            # Python dependencies
├── README.md                   # Quickstart guide & project overview
├── DOCUMENTATION.md            # Complete 18-chapter academic thesis document
├── .env.example                # Environment variables template
│
├── models/
│   ├── ecg_cnn_model.keras     # Trained 1D CNN model (44k params, 603 KB)
│   ├── inference_config.json   # Model runtime parameters
│   ├── training_history.json   # Epoch loss/accuracy progression
│   ├── baseline_rf_model.joblib# Random forest benchmark model
│   └── test_evaluation_metrics.json
│
├── data/
│   ├── raw/                    # Downloaded PhysioNet MIT-BIH recordings (.dat, .hea, .atr)
│   ├── processed/              # Segmented beat tensors (X_train, y_train, X_test, etc.)
│   └── sample_uploads/         # Pre-loaded clinical sample CSVs for testing
│
├── src/
│   ├── download_data.py        # Automated PhysioNet dataset downloader
│   ├── explore_dataset.py      # Dataset inspection & annotation analysis
│   ├── preprocessing.py        # Zero-phase Butterworth & notch filters
│   ├── segmentation.py         # Pan-Tompkins R-peak detector & window slicing
│   ├── baseline_model.py       # Random forest baseline with 15 manual features
│   ├── train.py                # 1D CNN model architecture & training pipeline
│   ├── evaluate.py             # Rigorous unseen test patient evaluation
│   ├── predict.py              # Inference engine for file & array inputs
│   └── database.py             # Dual MySQL connector with SQLite fallback
│
├── templates/                  # Frontend HTML5 views
│   ├── index.html              # Home page
│   ├── analysis.html           # Interactive ECG upload & Chart.js waveform screening
│   ├── history.html            # Database history viewer
│   └── about.html              # Academic methodology & architecture
│
├── static/
│   ├── css/style.css           # Clinical responsive stylesheet
│   └── js/script.js            # Chart.js waveform rendering & AJAX engine
│
├── database/
│   ├── schema.sql              # MySQL DDL table schema
│   ├── setup_db.py             # Database setup, migration & seeding utility
│   └── ecg_history.db          # Active SQLite local fallback database
│
├── figures/                    # Saved diagnostic charts & confusion matrices
└── tests/
    └── test_pipeline.py        # Automated unit & integration test suite
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- Python 3.11 installed
- (Optional) MySQL Server (XAMPP / WAMP / MySQL 8.0). If MySQL is not running, the application automatically uses SQLite (`database/ecg_history.db`).

### 2. Installation
Clone or navigate to the workspace directory:
```bash
cd c:\Users\shrey\OneDrive\Documents\Desktop\MDM
py -3.11 -m pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
py -3.11 app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 🧪 Pipeline Stage Commands

To reproduce any phase of the research pipeline:

```bash
# Phase 1: Download & inspect MIT-BIH Arrhythmia dataset
py -3.11 src/download_data.py
py -3.11 src/explore_dataset.py

# Phase 2: Run ECG Preprocessing & generate filter validation plots
py -3.11 src/preprocessing.py

# Phase 3: Segment heartbeats & partition into inter-patient splits (DS1 vs DS2)
py -3.11 src/segmentation.py

# Phase 4: Train & evaluate Baseline Machine Learning model (Random Forest)
py -3.11 src/baseline_model.py

# Phase 5: Train 1D Convolutional Neural Network
py -3.11 src/train.py

# Phase 6: Evaluate on completely unseen test patients (DS2)
py -3.11 src/evaluate.py

# Phase 9: Initialize & seed database history
py -3.11 database/setup_db.py --seed

# Phase 10: Run automated unit & integration test suite
py -3.11 tests/test_pipeline.py
```

---

## 📊 Summary of Results

Evaluated on **13,910 heartbeats** from completely unseen test patients (De Chazal DS2 benchmark):

| Model | PVC Arrhythmia Sensitivity (Recall) | PVC ROC-AUC | Overall Accuracy | Model Size |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Baseline** | 84.18% | 0.881 | 89.99% | 7.02 MB |
| **Proposed 1D CNN** | **91.73%** | **0.970** | 87.46% | **603 KB** |

- **Sensitivity Gain**: $+7.55\%$ improvement in catching ventricular ectopic arrhythmias.
- **ROC-AUC**: Exceptional $0.970$ area under curve for Premature Ventricular Contractions.
- **Inter-Patient Partitioning**: Zero intra-patient data leakage.

For complete academic documentation, refer to **[`DOCUMENTATION.md`](file:///c:/Users/shrey/OneDrive/Documents/Desktop/MDM/DOCUMENTATION.md)**.
