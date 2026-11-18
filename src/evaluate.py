"""
evaluate.py
===========
Phase 6: Deep Learning Model Evaluation on Completely Unseen Test Patients (DS2).

Evaluates the trained 1D CNN model (models/ecg_cnn_model.keras) against
the 13,910 heartbeats from DS2 patients to compute:
1. Overall Accuracy
2. Sensitivity (Recall) per class
3. Specificity per class
4. Precision per class
5. F1-Score per class
6. Multi-class ROC-AUC analysis with One-vs-Rest curves
7. Confusion Matrix heatmap
8. Direct comparative benchmark between Baseline ML vs 1D CNN
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ['Normal Beat', 'Supraventricular Ectopic (SVEB)', 'Premature Ventricular (PVC)', 'Other Arrhythmia']
SHORT_NAMES = ['Normal', 'SVEB', 'PVC', 'Other']


def calculate_clinical_metrics(y_true, y_pred, y_probs, num_classes=4):
    """
    Computes rigorous clinical metrics including Sensitivity, Specificity,
    Precision, F1-Score, and ROC-AUC per class.
    """
    overall_acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Binarize labels for multi-class ROC-AUC
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    
    per_class_metrics = {}
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    for c in range(num_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        # ROC AUC for class c
        try:
            if y_true_bin.shape[1] == num_classes and np.sum(y_true_bin[:, c]) > 0:
                fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_probs[:, c])
                roc_auc = float(auc(fpr, tpr))
            else:
                roc_auc = 0.0
        except Exception:
            roc_auc = 0.0
            
        per_class_metrics[CLASS_NAMES[c]] = {
            'samples': int(np.sum(y_true == c)),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'sensitivity_recall': float(np.round(sensitivity, 4)),
            'specificity': float(np.round(specificity, 4)),
            'precision': float(np.round(precision, 4)),
            'f1_score': float(np.round(f1, 4)),
            'roc_auc': float(np.round(roc_auc, 4))
        }
        
    summary = {
        'overall_accuracy': float(np.round(overall_acc, 4)),
        'macro_precision': float(np.round(prec_macro, 4)),
        'macro_recall': float(np.round(rec_macro, 4)),
        'macro_f1': float(np.round(f1_macro, 4)),
        'per_class': per_class_metrics
    }
    return summary, cm, y_true_bin


def plot_confusion_matrix(cm, test_acc, macro_f1, save_path="figures/cnn_confusion_matrix.png"):
    """Plots counts and normalized percentage confusion matrix."""
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES)
    plt.title(f"1D CNN Confusion Matrix (Unseen Test Patients - DS2)\nAccuracy: {test_acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}%",
              fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Cardiac Class", fontsize=11)
    plt.ylabel("Cardiologist Reference Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] 1D CNN Confusion Matrix saved to: {save_path}")


def plot_roc_curves(y_true_bin, y_probs, save_path="figures/cnn_roc_curves.png"):
    """Plots One-vs-Rest ROC curves for each arrhythmia class."""
    plt.figure(figsize=(9, 6))
    colors = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    
    for c in range(len(CLASS_NAMES)):
        if np.sum(y_true_bin[:, c]) > 0:
            fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_probs[:, c])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[c], lw=2.0,
                     label=f"{SHORT_NAMES[c]} (AUC = {roc_auc:.3f})")
                     
    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("1D CNN Multi-Class ROC Curves (Unseen Test Patients - DS2)", fontsize=12, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] ROC Curves saved to: {save_path}")


def plot_benchmark_comparison(cnn_summary, baseline_path="models/baseline_metrics.json",
                              save_path="figures/model_comparison_benchmark.png"):
    """
    Plots a side-by-side grouped bar chart comparing the Phase 4 Baseline
    (Random Forest) vs. Phase 5 (1D CNN) to demonstrate the deep learning upgrade.
    """
    if not os.path.exists(baseline_path):
        return
        
    with open(baseline_path, 'r') as f:
        baseline_metrics = json.load(f)
        
    metrics = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score']
    baseline_vals = [
        baseline_metrics['test_accuracy'] * 100,
        baseline_metrics['macro_precision'] * 100,
        baseline_metrics['macro_recall'] * 100,
        baseline_metrics['macro_f1'] * 100
    ]
    cnn_vals = [
        cnn_summary['overall_accuracy'] * 100,
        cnn_summary['macro_precision'] * 100,
        cnn_summary['macro_recall'] * 100,
        cnn_summary['macro_f1'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline ML (Random Forest)', color='#7f7f7f')
    rects2 = ax.bar(x + width/2, cnn_vals, width, label='Proposed 1D CNN', color='#1f77b4')
    
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('Academic Benchmark: Baseline Handcrafted ML vs Proposed 1D CNN\n(Evaluated on Completely Unseen Test Patients - DS2)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(fontsize=11)
    
    # Add values on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1f77b4')
                    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Comparative benchmark chart saved to: {save_path}")


def run_evaluation(data_dir='data/processed', model_path='models/ecg_cnn_model.keras',
                   models_dir='models', figures_dir='figures'):
    """Main evaluation runner."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 65)
    print(" PHASE 6: 1D CNN MODEL EVALUATION ON UNSEEN TEST PATIENTS (DS2) ")
    print("=" * 65)
    
    # 1. Load Model & Test Data
    print(f"[*] Loading trained 1D CNN from '{model_path}'...")
    model = keras.models.load_model(model_path)
    
    print(f"[*] Loading unseen test tensors from '{data_dir}'...")
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    print(f"    • Test Beats Count : {len(X_test):,} beats from unseen patient records")
    print(f"    • Tensor Dimension : {X_test.shape}")
    
    # 2. Inference
    print("\n[*] Performing batch inference across 13,910 test beats...")
    y_probs = model.predict(X_test, batch_size=128, verbose=1)
    y_pred = np.argmax(y_probs, axis=-1)
    
    # 3. Calculate Clinical Metrics
    summary, cm, y_true_bin = calculate_clinical_metrics(y_test, y_pred, y_probs)
    
    print("\n" + "=" * 65)
    print(" UNSEEN TEST SET CLINICAL PERFORMANCE RESULTS ")
    print("=" * 65)
    print(f" • Overall Accuracy      : {summary['overall_accuracy'] * 100:.2f}%")
    print(f" • Macro Precision       : {summary['macro_precision'] * 100:.2f}%")
    print(f" • Macro Recall (Sens.)  : {summary['macro_recall'] * 100:.2f}%")
    print(f" • Macro F1-Score        : {summary['macro_f1'] * 100:.2f}%")
    
    print("\n--- Detailed Per-Class Clinical Breakdown ---")
    print(f"{'Class Name':<32} | {'Samples':<7} | {'Sens/Recall':<11} | {'Specificity':<11} | {'Precision':<10} | {'F1-Score':<8} | {'ROC-AUC':<7}")
    print("-" * 100)
    for c_name, m in summary['per_class'].items():
        print(f"{c_name:<32} | {m['samples']:<7d} | {m['sensitivity_recall']*100:9.2f}% | {m['specificity']*100:9.2f}% | {m['precision']*100:8.2f}% | {m['f1_score']*100:7.2f}% | {m['roc_auc']:7.3f}")
        
    print("\n--- Standard Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    
    # 4. Save Metrics
    metrics_path = os.path.join(models_dir, "test_evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"[+] Saved evaluation metrics to: {metrics_path}")
    
    # 5. Visualizations
    plot_confusion_matrix(cm, summary['overall_accuracy'], summary['macro_f1'],
                          save_path=os.path.join(figures_dir, "cnn_confusion_matrix.png"))
    plot_roc_curves(y_true_bin, y_probs,
                    save_path=os.path.join(figures_dir, "cnn_roc_curves.png"))
    plot_benchmark_comparison(summary, baseline_path=os.path.join(models_dir, "baseline_metrics.json"),
                              save_path=os.path.join(figures_dir, "model_comparison_benchmark.png"))
                              
    return summary


if __name__ == '__main__':
    run_evaluation()
