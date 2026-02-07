"""
Lightweight exploratory analysis helpers.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loading import DatasetBundle


def print_dataset_overview(bundle: DatasetBundle) -> None:
    merged = bundle.merged
    edgelist = bundle.edgelist

    print(f"Shape: {merged.shape}")
    print(f"Columns: {merged.columns.tolist()}")
    print(f"Number of edges in the transaction network: {len(edgelist)}")

    class_counts = merged["class_label"].value_counts()
    class_percentages = merged["class_label"].value_counts(normalize=True) * 100
    print("\nClass Distribution:")
    for class_label, count in class_counts.items():
        percentage = class_percentages[class_label]
        print(f"{class_label}: {count} transactions ({percentage:.2f}%)")

    missing_values = merged.isnull().sum()
    print(f"\nTotal missing values: {missing_values.sum()}")
    if missing_values.sum() > 0:
        print("Columns with missing values:")
        print(missing_values[missing_values > 0])


def temporal_summary(merged: pd.DataFrame) -> None:
    print(f"Time steps range: {merged['time_step'].min()} to {merged['time_step'].max()}")
    print(f"Unique time steps: {merged['time_step'].nunique()}")
    time_dist = merged["time_step"].value_counts().sort_index()
    print("\nTransactions per time step (first 10):")
    print(time_dist.head(10))

    time_class_dist = merged.groupby(["time_step", "class_label"]).size().unstack(fill_value=0)
    print("\nClass distribution by time (first 10):")
    print(time_class_dist.head(10))

    # Illicit percentage by time
    if "1" in time_class_dist.columns:
        time_class_dist["total"] = time_class_dist.sum(axis=1)
        time_class_dist["illicit_pct"] = time_class_dist["1"] / time_class_dist["total"] * 100
        print("\nIllicit transaction percentage by time:")
        print(time_class_dist[["illicit_pct"]].round(2).head(10))


def feature_snapshot(merged: pd.DataFrame) -> None:
    preview_cols = [f"feature_{i}" for i in range(1, 6)]
    print("\nFeature stats (first five features):")
    print(merged[preview_cols].describe())

    feature_stats = merged[preview_cols].agg(["min", "max", "mean", "std"])
    print("\nQuick stats:")
    print(feature_stats)

