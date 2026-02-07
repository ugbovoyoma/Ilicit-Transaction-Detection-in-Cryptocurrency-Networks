"""
Preprocessing utilities: temporal split, pseudo-labeling, and SMOTE balancing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from .config import LABEL_MAP


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_full: pd.DataFrame
    test_full: pd.DataFrame
    baseline_features: List[str]


def temporal_split(data: pd.DataFrame, cutoff: int = 35) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into early (train) and late (test) periods."""
    train = data[data["time_step"] <= cutoff].copy()
    test = data[data["time_step"] > cutoff].copy()
    return train, test


def pseudo_label_and_balance(
    data: pd.DataFrame, cutoff: int = 35, confidence_threshold: float = 0.95
) -> PreparedData:
    """
    Apply pseudo-labeling on unlabeled train-period data and balance classes with SMOTE.
    Mirrors the hybrid approach from the original script.
    """
    # Base labeled/unlabeled split
    labeled_data = data[data["class_label"].isin(["1", "2"])]
    unlabeled_data = data[data["class_label"] == "unknown"]

    feature_cols = [col for col in labeled_data.columns if col.startswith("feature_")]

    early_periods = labeled_data[labeled_data["time_step"] <= cutoff]
    late_periods = labeled_data[labeled_data["time_step"] > cutoff]

    X_train_temp = early_periods[feature_cols]
    y_train_temp = early_periods["class_label"]
    X_test = late_periods[feature_cols]
    y_test = late_periods["class_label"]

    # Train initial model
    initial_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    initial_model.fit(X_train_temp, y_train_temp)

    # Pseudo-label high confidence unlabeled samples in train window
    unlabeled_train = unlabeled_data[unlabeled_data["time_step"] <= cutoff]
    if len(unlabeled_train) > 0:
        pseudo_probs = initial_model.predict_proba(unlabeled_train[feature_cols])
        pseudo_labels = initial_model.predict(unlabeled_train[feature_cols])
        max_probs = np.max(pseudo_probs, axis=1)
        high_conf_mask = max_probs > confidence_threshold
        selected_pseudo_features = unlabeled_train.loc[high_conf_mask, feature_cols]
        selected_pseudo_labels = pseudo_labels[high_conf_mask]
    else:
        selected_pseudo_features = pd.DataFrame(columns=feature_cols)
        selected_pseudo_labels = np.array([])

    # Combine real labels with pseudo-labels
    X_enhanced = pd.concat([X_train_temp, selected_pseudo_features], ignore_index=True)
    y_enhanced = pd.concat(
        [y_train_temp, pd.Series(selected_pseudo_labels, dtype=y_train_temp.dtype)],
        ignore_index=True,
    )

    # Handle NaNs
    if X_enhanced.isnull().sum().sum() > 0:
        X_enhanced = X_enhanced.fillna(X_enhanced.mean())

    smote = SMOTE(sampling_strategy=0.2, random_state=42)
    X_final, y_final = smote.fit_resample(X_enhanced, y_enhanced)

    # Return baseline features and full splits for downstream models
    return PreparedData(
        X_train=pd.DataFrame(X_final, columns=feature_cols),
        y_train=pd.Series(y_final),
        X_test=X_test,
        y_test=y_test,
        train_full=early_periods,
        test_full=late_periods,
        baseline_features=feature_cols,
    )

