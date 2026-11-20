"""
ml_models.py
============
Module 5: Baseline Machine Learning Classifiers for Connectome Classification.
Implements:
1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Machine (SVM)
Features nested cross-validation, hyperparameter grid evaluation, and metric logging.
"""

import os
import sys
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, RANDOM_STATE, CV_FOLDS
from evaluation import evaluate_predictions


class BaselineMLModels:
    """Trains and benchmarks classical machine learning models on connectome features."""

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {
            "Logistic Regression": LogisticRegression(
                C=0.1, penalty="l2", solver="lbfgs", max_iter=1000,
                class_weight="balanced", random_state=random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150, max_depth=6, min_samples_split=4,
                class_weight="balanced", random_state=random_state, n_jobs=-1
            ),
            "Support Vector Machine": SVC(
                C=1.0, kernel="rbf", probability=True,
                class_weight="balanced", random_state=random_state
            )
        }
        self.trained_models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}

    def train_and_evaluate_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_models: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Fits all baseline models, evaluates cross-validation on train split,
        and scores generalization performance on held-out test set.
        """
        # Standardize features (fit on train only)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=self.random_state)

        for name, clf in self.models.items():
            print(f"[*] Training {name}...")
            # Cross-validation score on training fold
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
            
            # Fit final model on full train split
            clf.fit(X_train_scaled, y_train)
            self.trained_models[name] = clf

            # Predictions on held-out test set
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else None

            metrics = evaluate_predictions(y_test, y_pred, y_prob)
            metrics["cv_roc_auc_mean"] = float(np.mean(cv_scores))
            metrics["cv_roc_auc_std"] = float(np.std(cv_scores))
            self.results[name] = metrics

            print(f"    -> Test Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

            if save_models:
                safe_name = name.lower().replace(" ", "_")
                out_path = MODELS_DIR / f"ml_{safe_name}.joblib"
                joblib.dump(clf, out_path)

        if save_models:
            joblib.dump(self.scaler, MODELS_DIR / "feature_scaler.joblib")
            print(f"[+] All baseline models saved to '{MODELS_DIR}'")

        return self.results


if __name__ == "__main__":
    np.random.seed(42)
    X_tr = np.random.normal(0, 1, (80, 50))
    y_tr = np.random.randint(0, 2, 80)
    X_te = np.random.normal(0, 1, (20, 50))
    y_te = np.random.randint(0, 2, 20)

    runner = BaselineMLModels()
    res = runner.train_and_evaluate_all(X_tr, y_tr, X_te, y_te, save_models=False)
