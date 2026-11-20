"""
run_pipeline.py
===============
End-to-End Orchestrator for Brain Connectivity Analysis & GNN Training.
Executes all research pipeline phases:
1. Dataset Acquisition (ABIDE I AAL-116)
2. Signal Conditioning & Preprocessing
3. Functional Connectivity Matrix Computation
4. Brain Graph Topological Metric Extraction
5. Traditional Machine Learning Benchmarking (LR, RF, SVM)
6. Graph Neural Network (GCN) Training & Evaluation
7. Healthy Control vs ASD Statistical Comparison
8. Persisting Artifacts for Dashboard & Research Reports.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "visualizations"))

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR,
    TEST_SIZE, RANDOM_STATE, AAL_REGIONS
)
from src.data_loader import ABIDELoader
from src.preprocessing import FMRIPreprocessor
from src.connectivity import FunctionalConnectivity
from src.graph_builder import BrainGraphBuilder
from src.feature_extraction import ConnectomeFeatureExtractor
from src.ml_models import BaselineMLModels
from src.gnn_model import GNNTrainer
from src.evaluation import create_model_comparison_df
from visualizations.comparison import compare_cohort_connectivity


def run_full_pipeline(n_subjects: int = 100):
    print("=" * 80)
    print("AI-BASED BRAIN CONNECTIVITY ANALYSIS USING GRAPH NEURAL NETWORKS")
    print("RESEARCH EXPERIMENTAL PIPELINE EXECUTION")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Phase 1: Dataset Acquisition
    # --------------------------------------------------------------------------
    print("\n[PHASE 1] Fetching and Structuring ABIDE I Dataset...")
    loader = ABIDELoader()
    pheno_df, timeseries_dict = loader.fetch_dataset(n_subjects=n_subjects)
    summary = loader.get_dataset_summary()
    print(f"  Cohort: {summary['total_subjects']} participants ({summary['asd_count']} ASD, {summary['tc_count']} TC)")
    print(f"  Sites: {list(summary['sites'].keys())}")

    # --------------------------------------------------------------------------
    # Phase 2: fMRI Signal Conditioning & Preprocessing
    # --------------------------------------------------------------------------
    print("\n[PHASE 2] Applying Zero-Phase Bandpass Filtering & Detrending...")
    preprocessor = FMRIPreprocessor()
    sub_ids = loader.subject_ids
    labels = np.array(loader.labels)

    preprocessed_ts = []
    for sid in sub_ids:
        raw_ts = timeseries_dict[sid]
        clean_ts = preprocessor.process(raw_ts)
        preprocessed_ts.append(clean_ts)

    preprocessed_stack = np.stack(preprocessed_ts, axis=0)
    print(f"  Conditioned BOLD tensor shape: {preprocessed_stack.shape} (Subjects, TRs, Regions)")

    # --------------------------------------------------------------------------
    # Phase 3: Functional Connectivity Matrix Computation
    # --------------------------------------------------------------------------
    print("\n[PHASE 3] Computing Pearson Correlation & Fisher r-to-z Matrices...")
    fc = FunctionalConnectivity(apply_fisher_z=True)
    fc_matrices = np.array([fc.compute_matrix(ts) for ts in preprocessed_stack])
    print(f"  Connectivity Matrices tensor: {fc_matrices.shape} (116 x 116 per participant)")

    # --------------------------------------------------------------------------
    # Phase 4: Graph Construction & Brain Network Analysis
    # --------------------------------------------------------------------------
    print("\n[PHASE 4] Building Brain Graphs & Extracting Topological Metrics...")
    graph_builder = BrainGraphBuilder()
    graph_metrics_list = []
    for i in range(len(sub_ids)):
        G = graph_builder.build_networkx_graph(fc_matrices[i])
        m = graph_builder.compute_graph_metrics(G)
        graph_metrics_list.append(m)

    mean_density = np.mean([m["network_density"] for m in graph_metrics_list])
    mean_clustering = np.mean([m["avg_clustering_coefficient"] for m in graph_metrics_list])
    print(f"  Average Network Density: {mean_density:.4f}")
    print(f"  Average Clustering Coefficient: {mean_clustering:.4f}")

    # --------------------------------------------------------------------------
    # Phase 5: Participant-Level Partitioning & Feature Extraction
    # --------------------------------------------------------------------------
    print("\n[PHASE 5] Participant-Level Partitioning & Classical ML Training...")
    # Stratified split to prevent patient overlap and data leakage
    idx_train, idx_test = train_test_split(
        np.arange(len(labels)),
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE
    )

    feature_extractor = ConnectomeFeatureExtractor(n_top_features=250)
    X_edges = feature_extractor.extract_edge_features(fc_matrices)
    X_train_raw, X_test_raw = X_edges[idx_train], X_edges[idx_test]
    y_train, y_test = labels[idx_train], labels[idx_test]

    # Select top features fitted strictly on training partition
    X_train_sel, selector = feature_extractor.fit_transform_features(X_train_raw, y_train)
    X_test_sel = feature_extractor.transform_features(X_test_raw)
    print(f"  Selected {X_train_sel.shape[1]} informative functional connectivity features.")

    # Train Baseline Models
    ml_runner = BaselineMLModels()
    ml_results = ml_runner.train_and_evaluate_all(X_train_sel, y_train, X_test_sel, y_test)

    # --------------------------------------------------------------------------
    # Phase 6: Graph Neural Network (Spectral GCN) Training
    # --------------------------------------------------------------------------
    print("\n[PHASE 6] Training Graph Convolutional Network (GCN)...")
    mats_train, mats_test = fc_matrices[idx_train], fc_matrices[idx_test]

    gnn_trainer = GNNTrainer()
    gnn_metrics = gnn_trainer.train_spectral_gcn(
        mats_train, y_train, mats_test, y_test,
        epochs=35, batch_size=16
    )
    all_results = dict(ml_results)
    all_results["Graph Convolutional Network (GCN)"] = gnn_metrics

    # --------------------------------------------------------------------------
    # Phase 7: Healthy vs ASD Statistical Comparison
    # --------------------------------------------------------------------------
    print("\n[PHASE 7] Conducting Cohort Mass-Univariate Statistical Tests...")
    comparison_stats = compare_cohort_connectivity(fc_matrices, labels, AAL_REGIONS)
    print(f"  Calculated t-tests across {len(comparison_stats['df_edges'])} unique connections.")
    print(f"  Top Alteration: {comparison_stats['top_hypoconnected'][0]['region_1']} <-> {comparison_stats['top_hypoconnected'][0]['region_2']} (Diff: {comparison_stats['top_hypoconnected'][0]['diff']:.3f})")

    # --------------------------------------------------------------------------
    # Phase 8: Persisting Precomputed Results & Benchmark Reports
    # --------------------------------------------------------------------------
    print("\n[PHASE 8] Persisting Models, Matrices, and Evaluation Reports...")
    comparison_df = create_model_comparison_df(all_results)
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON ON HELD-OUT PARTICIPANTS:")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    # Save summary table
    comparison_df.to_csv(REPORTS_DIR / "model_comparison_report.csv", index=False)
    with open(REPORTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save processed cohort data for interactive Streamlit dashboard
    np.savez_compressed(
        PROCESSED_DATA_DIR / "cohort_data.npz",
        fc_matrices=fc_matrices,
        labels=labels,
        sub_ids=np.array(sub_ids),
        idx_train=idx_train,
        idx_test=idx_test,
        mean_tc=comparison_stats["mean_tc"],
        mean_asd=comparison_stats["mean_asd"],
        diff_matrix=comparison_stats["diff_matrix"],
        t_stats=comparison_stats["t_stats"],
        p_values=comparison_stats["p_values"]
    )
    pheno_df.to_csv(PROCESSED_DATA_DIR / "phenotypic_metadata.csv", index=False)
    joblib.dump(selector, MODELS_DIR / "feature_selector.joblib")
    
    # Save training curves
    with open(REPORTS_DIR / "gnn_history.json", "w") as f:
        json.dump(gnn_trainer.history, f, indent=2)

    print(f"\n[+] Full pipeline completed successfully! Artifacts stored in '{PROCESSED_DATA_DIR}' and '{REPORTS_DIR}'")


if __name__ == "__main__":
    run_full_pipeline(n_subjects=100)
