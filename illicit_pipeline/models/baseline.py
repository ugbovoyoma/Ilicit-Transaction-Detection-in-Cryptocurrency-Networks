"""
Baseline ML models: Logistic Regression, Random Forest, XGBoost, and an ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    name: str
    model: object
    y_pred: np.ndarray
    y_proba: np.ndarray
    metrics: Dict[str, float]
    report: str
    confusion: np.ndarray


def _labels_to_binary(y: pd.Series | np.ndarray) -> np.ndarray:
    mapping = {"1": 1, "2": 0, 1: 1, 2: 0, "illicit": 1, "licit": 0}
    return np.array([mapping.get(val, val) for val in y]).astype(int)


def _evaluate(
    y_true: pd.Series, y_pred_labels: pd.Series | np.ndarray, y_proba: np.ndarray, positive_label: int = 1
) -> Tuple[Dict[str, float], str, np.ndarray]:
    y_true_binary = _labels_to_binary(y_true)
    y_pred_binary = _labels_to_binary(y_pred_labels)

    metrics = {
        "auc_roc": roc_auc_score(y_true_binary, y_proba),
        "auc_pr": average_precision_score(y_true_binary, y_proba, pos_label=positive_label),
        "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
        "f1": 0.0,
        "mcc": matthews_corrcoef(y_true_binary, y_pred_binary),
    }
    metrics["f1"] = (
        0
        if metrics["precision"] + metrics["recall"] == 0
        else 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
    )

    report = classification_report(
        y_true_binary,
        y_pred_binary,
        labels=[0, 1],
        target_names=["Licit", "Illicit"],
        zero_division=0,
    )
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    return metrics, report, cm


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> ModelResult:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, list(model.classes_).index("1")]

    metrics, report, cm = _evaluate(y_test, y_pred, y_proba)
    return ModelResult("Logistic Regression", model, y_pred, y_proba, metrics, report, cm)


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> ModelResult:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, list(model.classes_).index("1")]

    metrics, report, cm = _evaluate(y_test, y_pred, y_proba)
    return ModelResult("Random Forest", model, y_pred, y_proba, metrics, report, cm)


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> ModelResult:
    y_train_numeric = _labels_to_binary(y_train)
    y_test_numeric = _labels_to_binary(y_test)
    scale_pos_weight = (y_train_numeric == 0).sum() / max((y_train_numeric == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train_numeric)

    y_pred_numeric = model.predict(X_test)
    y_pred = np.where(y_pred_numeric == 1, "1", "2")
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics, report, cm = _evaluate(pd.Series(y_test_numeric), y_pred_numeric, y_proba)
    return ModelResult("XGBoost", model, y_pred, y_proba, metrics, report, cm)


def ensemble_by_auc(y_true: pd.Series, *results: ModelResult) -> ModelResult:
    """Weighted average ensemble using each model's AUC-ROC."""
    aucs = [res.metrics["auc_roc"] for res in results]
    total_auc = sum(aucs)
    if total_auc == 0:
        weights = [1 / len(aucs)] * len(aucs)
    else:
        weights = [auc / total_auc for auc in aucs]

    y_proba = sum(w * res.y_proba for w, res in zip(weights, results))
    y_pred = np.where(y_proba >= 0.5, "1", "2")

    metrics, report, cm = _evaluate(y_true, y_pred, y_proba)
    return ModelResult("Ensemble (ML)", None, y_pred, y_proba, metrics, report, cm)
