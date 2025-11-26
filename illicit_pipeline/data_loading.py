"""
Data loading utilities.

Responsible for reading the Elliptic Bitcoin CSVs and producing a merged
DataFrame that downstream steps can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from .config import DATA_DIR


@dataclass
class DatasetBundle:
    classes: pd.DataFrame
    edgelist: pd.DataFrame
    features: pd.DataFrame
    merged: pd.DataFrame


def load_raw_datasets(data_dir: str | Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three CSVs from disk."""
    data_dir = Path(data_dir)
    with tqdm(total=3, desc="Loading CSV files", unit="file") as pbar:
        classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
        pbar.update(1)
        edgelist = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")
        pbar.update(1)
        features = pd.read_csv(data_dir / "elliptic_txs_features.csv")
        pbar.update(1)
    return classes, edgelist, features


def merge_datasets(
    classes: pd.DataFrame, edgelist: pd.DataFrame, features: pd.DataFrame
) -> DatasetBundle:
    """Rename columns and merge features/classes into a single frame."""
    classes = classes.copy()
    edgelist = edgelist.copy()
    features = features.copy()

    classes.columns = ["node_ID", "class_label"]
    features.columns = ["node_ID", "time_step"] + [f"feature_{i}" for i in range(1, 166)]

    merged = features.merge(classes, on="node_ID", how="left")

    return DatasetBundle(
        classes=classes,
        edgelist=edgelist,
        features=features,
        merged=merged,
    )

