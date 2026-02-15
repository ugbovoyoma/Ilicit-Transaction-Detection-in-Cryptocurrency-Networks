"""
Top-level orchestration for the modular pipeline.

"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from . import analysis
from .config import DATA_DIR, DLConfig
from .data_loading import load_raw_datasets, merge_datasets
from .features import TEMPORAL_FEATURES, GRAPH_FEATURES, add_graph_features, add_temporal_features
from .models import baseline as baseline_models
from .models.deep import (
    prepare_graph_data,
    prepare_lstm_datasets,
    train_graphsage,
    train_lstm,
)
from .preprocessing import PreparedData, pseudo_label_and_balance, temporal_split


def _unique(seq: List[str]) -> List[str]:
    return list(dict.fromkeys(seq))


def run_baselines(run_analysis: bool = True) -> Dict[str, baseline_models.ModelResult]:
    """
    Run data loading, feature engineering, pseudo-labeling/SMOTE, and baseline ML models.
    Returns a dictionary of ModelResult objects keyed by model name.
    """
    classes, edgelist, features = load_raw_datasets(DATA_DIR)
    bundle = merge_datasets(classes, edgelist, features)

    if run_analysis:
        analysis.print_dataset_overview(bundle)
        analysis.temporal_summary(bundle.merged)
        analysis.feature_snapshot(bundle.merged)

    # Feature engineering
    data_with_features = add_temporal_features(bundle.merged)
    data_with_features = add_graph_features(data_with_features, bundle.edgelist)

    # Preprocessing: pseudo-label + SMOTE + temporal split
    prepared = pseudo_label_and_balance(data_with_features)
    # Fill NaNs in test using train means (non-leaky)
    train_means = prepared.X_train.mean()
    X_test = prepared.X_test.fillna(train_means)

    lr_res = baseline_models.train_logistic_regression(prepared.X_train, prepared.y_train, X_test, prepared.y_test)
    rf_res = baseline_models.train_random_forest(prepared.X_train, prepared.y_train, X_test, prepared.y_test)
    xgb_res = baseline_models.train_xgboost(prepared.X_train, prepared.y_train, X_test, prepared.y_test)
    ens_res = baseline_models.ensemble_by_auc(prepared.y_test, lr_res, rf_res, xgb_res)

    results = {
        lr_res.name: lr_res,
        rf_res.name: rf_res,
        xgb_res.name: xgb_res,
        ens_res.name: ens_res,
    }
    return results


def run_deep_models(
    data_with_features: pd.DataFrame,
    edgelist: pd.DataFrame,
    prepared: PreparedData,
    config: DLConfig,
) -> Dict[str, object]:
    """
    Optionally train LSTM and GraphSAGE models. Returns dict of DeepResult.
    """
    outputs: Dict[str, object] = {}

    # LSTM
    lstm_features = _unique(prepared.baseline_features + TEMPORAL_FEATURES)
    train_loader, test_loader, _ = prepare_lstm_datasets(prepared.train_full, prepared.test_full, lstm_features, config)
    lstm_result = train_lstm(train_loader, test_loader, input_dim=len(lstm_features), config=config)
    outputs[lstm_result.name] = lstm_result

    # GraphSAGE
    gnn_features = _unique(prepared.baseline_features + GRAPH_FEATURES + ["time_sin", "time_cos"])
    train_ids = set(prepared.train_full["node_ID"])
    test_ids = set(prepared.test_full["node_ID"])
    graph_data = prepare_graph_data(
        data_with_features,
        edgelist,
        gnn_features,
        train_mask_ids=train_ids,
        test_mask_ids=test_ids,
    )
    gnn_result = train_graphsage(graph_data, config)
    outputs[gnn_result.name] = gnn_result

    return outputs


def run_full_pipeline(include_deep: bool = False, run_analysis: bool = True) -> Dict[str, object]:
    """
    Convenience wrapper to run the full pipeline. Deep models are disabled by
    default to keep runtime reasonable; set `include_deep=True` to train them.
    """
    classes, edgelist, features = load_raw_datasets(DATA_DIR)
    bundle = merge_datasets(classes, edgelist, features)

    if run_analysis:
        analysis.print_dataset_overview(bundle)
        analysis.temporal_summary(bundle.merged)
        analysis.feature_snapshot(bundle.merged)

    data_with_features = add_temporal_features(bundle.merged)
    data_with_features = add_graph_features(data_with_features, bundle.edgelist)

    prepared = pseudo_label_and_balance(data_with_features)
    train_means = prepared.X_train.mean()
    X_test = prepared.X_test.fillna(train_means)

    lr_res = baseline_models.train_logistic_regression(prepared.X_train, prepared.y_train, X_test, prepared.y_test)
    rf_res = baseline_models.train_random_forest(prepared.X_train, prepared.y_train, X_test, prepared.y_test)
    xgb_res = baseline_models.train_xgboost(prepared.X_train, prepared.y_train, X_test, prepared.y_test)
    ens_res = baseline_models.ensemble_by_auc(prepared.y_test, lr_res, rf_res, xgb_res)

    results: Dict[str, object] = {
        lr_res.name: lr_res,
        rf_res.name: rf_res,
        xgb_res.name: xgb_res,
        ens_res.name: ens_res,
    }

    if include_deep:
        deep_config = DLConfig()
        deep_results = run_deep_models(data_with_features, bundle.edgelist, prepared, deep_config)
        results.update(deep_results)

    return results
