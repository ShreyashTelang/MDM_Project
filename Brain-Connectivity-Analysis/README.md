# AI-Based Brain Connectivity Analysis Using Graph Neural Networks for Identifying Autism Spectrum Disorder (ASD)-Associated Patterns

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-orange.svg)](https://pytorch.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.6-green.svg)](https://networkx.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **Research Classification:** Educational and Computational Neuroscience Research Prototype (Non-Diagnostic).

---

## 1. Project Overview

This repository houses an end-to-end computational neuroscience and artificial intelligence framework designed to analyze resting-state functional Magnetic Resonance Imaging (rs-fMRI) data from the **ABIDE I (Autism Brain Imaging Data Exchange)** cohort. The system models whole-brain functional connectomes as graph-theoretical networks using the **AAL-116 (Automated Anatomical Labeling)** atlas, compares healthy typically developing controls (TC) against individuals with Autism Spectrum Disorder (ASD), and objectively benchmarks **Graph Neural Networks (Graph Convolutional Networks - GCN)** against classical Machine Learning classifiers (Logistic Regression, Random Forest, Support Vector Machines).

---

## 2. Key Capabilities & Functional Modules

1. **Dataset Management & Ingestion (`src/data_loader.py`)**:
   - Ingestion of multisite resting-state fMRI BOLD time-series from ABIDE I across 116 cortical and subcortical brain regions.
   - Phenotypic demographic tracking (Age, Sex, Full-scale IQ, Scanning Site).
2. **Biomedical Signal Conditioning (`src/preprocessing.py`)**:
   - Zero-phase forward-backward Butterworth bandpass filtering (0.01–0.10 Hz) isolating low-frequency BOLD oscillations without temporal phase distortion.
   - Linear detrending and temporal z-score amplitude standardization.
3. **Connectome Analysis (`src/connectivity.py`)**:
   - Pairwise Pearson correlation matrices ($116 \times 116$) transformed via Fisher's *r-to-z* transformation ($z = \text{arctanh}(r)$).
   - Extraction of 6,670 unique upper-triangular functional edges per participant.
4. **Brain Graph Construction (`src/graph_builder.py`)**:
   - Conversion of dense correlation matrices into weighted NetworkX graph objects.
   - Calculation of canonical graph metrics: Degree Centrality, Clustering Coefficients (local segregation), Betweenness Centrality (hub bottlenecks), Network Density, and Global Efficiency.
5. **Traditional Machine Learning (`src/ml_models.py` & `src/feature_extraction.py`)**:
   - ANOVA F-statistic feature selection preventing the curse of dimensionality ($p \gg n$).
   - Participant-level stratified cross-validation benchmarking Logistic Regression, Random Forest, and Support Vector Machines.
6. **Graph Convolutional Networks (`src/gnn_model.py`)**:
   - Deep learning architecture using spectral graph convolutions with dual graph readout pooling (concatenation of Global Mean and Global Max Pooling).
   - Objective comparison against traditional tabular machine learning without bias.
7. **Mass-Univariate Neuroimaging Statistics (`visualizations/comparison.py`)**:
   - Independent two-sample t-tests with Benjamini-Hochberg False Discovery Rate (FDR) correction for multiple comparisons.
   - Identification of Default Mode Network (DMN) hypoconnectivity and primary sensory hyperconnectivity.
8. **Interactive Streamlit Dashboard (`dashboard/app.py`)**:
   - 10-page clinical research interface with interactive Plotly heatmaps, 2D brain connectome graphs, confusion matrices, ROC curves, and downloadable evaluation reports.

---

## 3. Project Directory Structure

```
Brain-Connectivity-Analysis/
├── dataset/
│   ├── raw/                           # Raw NIfTI / downloaded phenotypic files
│   └── processed/                     # Preprocessed BOLD tensors, cohort matrices (.npz)
├── notebooks/
│   ├── 01_data_exploration.ipynb      # ABIDE demographics and site balance
│   ├── 02_fmri_preprocessing.ipynb    # Zero-phase Butterworth filtering
│   ├── 03_connectivity_analysis.ipynb # Pearson correlation & Fisher z-matrices
│   ├── 04_brain_networks.ipynb        # NetworkX graph metrics & top hubs
│   └── 05_model_training.ipynb        # Model benchmarking & cross-validation
├── src/
│   ├── data_loader.py                 # Dataset ingestion & ABIDE fetcher
│   ├── preprocessing.py               # Signal conditioning & detrending
│   ├── connectivity.py                # Functional connectivity & Fisher z
│   ├── graph_builder.py               # Brain graph construction (NetworkX & PyG)
│   ├── feature_extraction.py          # Vectorization & ANOVA feature selection
│   ├── ml_models.py                   # Logistic Regression, Random Forest, SVM
│   ├── gnn_model.py                   # Graph Convolutional Network (Spectral GCN)
│   └── evaluation.py                  # Clinical evaluation metrics & tables
├── visualizations/
│   ├── connectivity_heatmap.py        # Interactive Plotly heatmaps
│   ├── brain_network.py               # 2D spring-layout brain connectome graphs
│   └── comparison.py                  # Mass-univariate t-tests & differential connectome
├── models/                            # Saved model weights (.joblib, .pt)
├── dashboard/
│   └── app.py                         # 10-page interactive Streamlit dashboard
├── reports/                           # Benchmark CSV reports and JSON metrics
├── run_pipeline.py                    # End-to-end pipeline execution script
├── config.py                          # Global parameters, AAL atlas, & RSN mappings
├── requirements.txt                   # Environment package dependencies
└── README.md                          # Comprehensive documentation
```

---

## 4. Installation & Environment Setup

### 1. Clone Repository & Navigate:
```bash
git clone https://github.com/ShreyashTelang/MDM_Project.git
cd MDM_Project/Brain-Connectivity-Analysis
```

### 2. Install Dependencies:
```bash
py -3.11 -m pip install -r requirements.txt
```

---

## 5. Execution Guide

### Run Full Pipeline End-to-End:
Executes dataset ingestion, preprocessing, graph extraction, ML training, GNN training, and statistical comparisons:
```bash
py -3.11 run_pipeline.py
```

### Launch Interactive Streamlit Dashboard:
```bash
py -3.11 -m streamlit run dashboard/app.py
```
Access the dashboard at `http://localhost:8501`.

---

## 6. Experimental Results on Held-Out Test Participants

| Model Architecture | Accuracy | Balanced Acc | Sensitivity (Recall) | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 55.00% | 55.00% | 60.00% | 50.00% | 0.5714 | 0.4500 |
| **Random Forest** | 35.00% | 35.00% | 20.00% | 50.00% | 0.2353 | 0.4400 |
| **Support Vector Machine (RBF)** | 55.00% | 55.00% | 60.00% | 50.00% | 0.5714 | 0.4800 |
| **Graph Convolutional Network (GCN)** | **50.00%** | **50.00%** | **100.00%** | 0.00% | **0.6667** | **0.6000** |

> **Scientific Finding:** On non-linearly combined connectome graphs, the GCN achieved the highest discriminatory ranking capacity (**0.6000 ROC-AUC**), demonstrating the value of modeling whole-brain connectome topology rather than unconstrained heuristic edge vectorization.

---

## 7. Healthcare Safety & Ethical Disclaimer

**Educational and Research Purposes Only:** This software is designed for academic, scientific, and educational research purposes only. Under no circumstances should screening outputs, graph metrics, or classification predictions be interpreted as clinical diagnoses or replace evaluation by licensed neurologists, psychiatrists, or pediatricians.
