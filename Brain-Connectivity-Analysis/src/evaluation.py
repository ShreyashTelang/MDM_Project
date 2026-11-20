"""
evaluation.py
=============
Module 8: Model Evaluation and Benchmark Metrics.
Computes Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity,
F1-Score, ROC-AUC, and Confusion Matrices across participant-level test partitions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes rigorous clinical evaluation metrics for binary classification (ASD vs TC).
    
    Parameters:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Predicted class labels (0 or 1)
        y_prob: Predicted probabilities for the positive class (ASD)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        spec = 0.0

    # ROC-AUC calculation
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = 0.50
    else:
        auc = float(bal_acc)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def create_model_comparison_df(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Formats multiple model evaluation results into an ordered benchmark DataFrame."""
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            "Accuracy": f"{metrics.get('accuracy', 0.0) * 100:.2f}%",
            "Balanced Acc": f"{metrics.get('balanced_accuracy', 0.0) * 100:.2f}%",
            "Sensitivity (Recall)": f"{metrics.get('recall_sensitivity', 0.0) * 100:.2f}%",
            "Specificity": f"{metrics.get('specificity', 0.0) * 100:.2f}%",
            "F1-Score": f"{metrics.get('f1', 0.0):.4f}",
            "ROC-AUC": f"{metrics.get('roc_auc', 0.0):.4f}"
        })
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    y_t = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_p = np.array([0, 1, 0, 0, 1, 0, 1, 1])
    y_s = np.array([0.1, 0.9, 0.2, 0.4, 0.85, 0.15, 0.7, 0.6])
    res = evaluate_predictions(y_t, y_p, y_s)
    print("--- Test Evaluation Output ---")
    for k, v in res.items():
        print(f"  {k}: {v}")
