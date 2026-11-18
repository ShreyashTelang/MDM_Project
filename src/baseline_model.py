"""
baseline_model.py
=================
Phase 4: Traditional Machine Learning Baseline Benchmark.

Extracts handcrafted statistical and morphological features from each segmented
ECG heartbeat and trains an ensemble Random Forest classifier.

This establishes an academic reference baseline to scientifically demonstrate:
1. The limitations of manual feature engineering on noisy physiological signals.
2. The empirical performance gains achieved by 1D Convolutional Neural Networks (Phase 5).
"""

import os
import json
import joblib
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

CLASS_NAMES = ['Normal Beat', 'Supraventricular Ectopic (SVEB)', 'Premature Ventricular (PVC)', 'Other Arrhythmia']
FEATURE_NAMES = [
    'mean', 'std', 'variance', 'skewness', 'kurtosis',
    'min_val', 'max_val', 'peak_to_peak', 'r_peak_amp',
    'pre_r_energy', 'post_r_energy', 'energy_ratio',
    'total_energy', 'rms', 'zero_crossings'
]


def extract_features_matrix(X, r_index=144):
    """
    Vectorized high-speed feature extraction for batch of beats.
    Input: (N, 360, 1) -> Output: (N, 15)
    """
    sig = X.squeeze(-1)  # (N, 360)
    N, L = sig.shape
    
    mean_val = np.mean(sig, axis=1)
    std_val = np.std(sig, axis=1)
    var_val = np.var(sig, axis=1)
    
    std_safe = np.where(std_val == 0, 1e-8, std_val)[:, np.newaxis]
    norm_sig = (sig - mean_val[:, np.newaxis]) / std_safe
    skew_val = np.mean(norm_sig ** 3, axis=1)
    kurt_val = np.mean(norm_sig ** 4, axis=1) - 3.0  # Fisher kurtosis
    
    min_val = np.min(sig, axis=1)
    max_val = np.max(sig, axis=1)
    ptp_val = max_val - min_val
    r_peak_amp = sig[:, min(r_index, L - 1)]
    
    pre_energy = np.sum(sig[:, :r_index] ** 2, axis=1)
    post_energy = np.sum(sig[:, r_index:] ** 2, axis=1)
    energy_ratio = post_energy / (pre_energy + 1e-8)
    total_energy = pre_energy + post_energy
    rms = np.sqrt(np.mean(sig ** 2, axis=1))
    
    zero_crossings = np.sum(np.diff(np.signbit(sig), axis=1) != 0, axis=1)
    
    features = np.column_stack([
        mean_val, std_val, var_val, skew_val, kurt_val,
        min_val, max_val, ptp_val, r_peak_amp,
        pre_energy, post_energy, energy_ratio,
        total_energy, rms, zero_crossings
    ]).astype(np.float32)
    
    return features


def train_and_evaluate_baseline(data_dir='data/processed', models_dir='models', figures_dir='figures'):
    """
    Runs feature extraction, trains Random Forest, evaluates on unseen test set,
    and exports benchmark reports and figures.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 65)
    print(" PHASE 4: BASELINE MACHINE LEARNING MODEL (RANDOM FOREST) ")
    print("=" * 65)
    
    # 1. Load preprocessed tensors
    print("[*] Loading segmented heartbeat tensors from 'data/processed/'...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    with open(os.path.join(data_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    r_idx = metadata.get('window_before', 144)
    
    # 2. Extract Features
    print(f"[*] Extracting {len(FEATURE_NAMES)} handcrafted features per heartbeat:")
    print("    --> Features: Statistical moments, amplitude peaks, temporal energies, RMS, zero-crossings...")
    
    print("    --> Extracting training features...", end=' ', flush=True)
    feat_train = extract_features_matrix(X_train, r_index=r_idx)
    print(f"[Done: {feat_train.shape}]")
    
    print("    --> Extracting validation features...", end=' ', flush=True)
    feat_val = extract_features_matrix(X_val, r_index=r_idx)
    print(f"[Done: {feat_val.shape}]")
    
    print("    --> Extracting test features...", end=' ', flush=True)
    feat_test = extract_features_matrix(X_test, r_index=r_idx)
    print(f"[Done: {feat_test.shape}]")
    
    # 3. Train Random Forest Classifier
    print("\n[*] Training Random Forest Classifier (100 estimators, balanced class weights)...")
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(feat_train, y_train)
    print("[+] Model training completed.")
    
    # Save trained baseline model
    model_save_path = os.path.join(models_dir, "baseline_rf_model.joblib")
    joblib.dump(rf_clf, model_save_path)
    print(f"[+] Saved baseline model to: {model_save_path}")
    
    # 4. Evaluate on Validation Set
    val_preds = rf_clf.predict(feat_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"\n[*] Validation Accuracy: {val_acc * 100:.2f}%")
    
    # 5. Evaluate on Unseen Test Set (DS2)
    print("\n" + "=" * 65)
    print(" BASELINE EVALUATION ON UNSEEN TEST PATIENTS (DS2) ")
    print("=" * 65)
    test_preds = rf_clf.predict(feat_test)
    test_probs = rf_clf.predict_proba(feat_test)
    
    test_acc = accuracy_score(y_test, test_preds)
    prec_macro = precision_score(y_test, test_preds, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, test_preds, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, test_preds, average='macro', zero_division=0)
    
    print(f" • Overall Test Accuracy : {test_acc * 100:.2f}%")
    print(f" • Macro Precision       : {prec_macro * 100:.2f}%")
    print(f" • Macro Recall          : {rec_macro * 100:.2f}%")
    print(f" • Macro F1-Score        : {f1_macro * 100:.2f}%")
    
    print("\n--- Detailed Classification Report ---")
    clf_report = classification_report(y_test, test_preds, target_names=CLASS_NAMES, digits=4)
    print(clf_report)
    
    # 6. Save Baseline Metrics for Viva / Final Comparison
    metrics_summary = {
        'model_name': 'Random Forest Classifier (Baseline Handcrafted Features)',
        'test_accuracy': float(np.round(test_acc, 4)),
        'macro_precision': float(np.round(prec_macro, 4)),
        'macro_recall': float(np.round(rec_macro, 4)),
        'macro_f1': float(np.round(f1_macro, 4)),
        'per_class': {}
    }
    for c_idx, c_name in enumerate(CLASS_NAMES):
        mask = (y_test == c_idx)
        c_acc = float(np.mean(test_preds[mask] == y_test[mask])) if np.sum(mask) > 0 else 0.0
        c_rec = float(recall_score(y_test == c_idx, test_preds == c_idx, zero_division=0))
        c_prec = float(precision_score(y_test == c_idx, test_preds == c_idx, zero_division=0))
        metrics_summary['per_class'][c_name] = {
            'samples': int(np.sum(mask)),
            'sensitivity_recall': round(c_rec, 4),
            'precision': round(c_prec, 4)
        }
        
    with open(os.path.join(models_dir, "baseline_metrics.json"), 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[+] Saved baseline metrics to: {os.path.join(models_dir, 'baseline_metrics.json')}")
    
    # 7. Plot Baseline Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=[c.split('(')[0].strip() for c in CLASS_NAMES],
                yticklabels=[c.split('(')[0].strip() for c in CLASS_NAMES])
    plt.title(f"Baseline Random Forest - Confusion Matrix (Unseen Test Patients)\nOverall Accuracy: {test_acc*100:.2f}% | Macro F1: {f1_macro*100:.2f}%", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class (Cardiologist Ground Truth)", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "baseline_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[+] Confusion matrix plot saved to: {cm_path}")
    
    # 8. Plot Feature Importance
    importances = rf_clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(FEATURE_NAMES)), importances[sorted_idx], color='#1f77b4', align='center')
    plt.yticks(range(len(FEATURE_NAMES)), [FEATURE_NAMES[i] for i in sorted_idx], fontsize=10)
    plt.xlabel("Gini Feature Importance", fontsize=11)
    plt.title("Random Forest - Handcrafted Feature Importance Ranking", fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    feat_imp_path = os.path.join(figures_dir, "baseline_feature_importance.png")
    plt.savefig(feat_imp_path, dpi=200)
    plt.close()
    print(f"[+] Feature importance plot saved to: {feat_imp_path}")
    
    return rf_clf, metrics_summary


if __name__ == '__main__':
    train_and_evaluate_baseline()
