"""
Central configuration for the modular illicit transaction detection pipeline.

"""

from __future__ import annotations

import torch
from dataclasses import dataclass

# Paths and labels
DATA_DIR = "./dataset/rawData/elliptic_bitcoin_dataset"
LABEL_MAP = {"2": 0, "1": 1}  # 0 = licit, 1 = illicit


@dataclass
class DLConfig:
    # LSTM
    LSTM_HIDDEN_DIM: int = 64
    LSTM_NUM_LAYERS: int = 1
    LSTM_DROPOUT: float = 0.3
    LSTM_BIDIRECTIONAL: bool = False
    # GNN
    GNN_TYPE: str = "GraphSAGE"
    GNN_HIDDEN_DIM: int = 128
    GNN_NUM_LAYERS: int = 2
    GNN_DROPOUT: float = 0.3
    GNN_AGGR: str = "mean"
    # Hybrid
    HYBRID_FUSION_DIM: int = 256
    HYBRID_DROPOUT: float = 0.3
    HYBRID_CONCAT_METHOD: str = "concatenate"
    USE_ATTENTION_FUSION: bool = True
    XGB_EMBEDDING_DIM: int = 16
    # Training
    BATCH_SIZE: int = 256
    NUM_EPOCHS: int = 30
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-5
    PATIENCE: int = 5
    # Loss
    USE_CLASS_WEIGHTS: bool = False
    CLASS_WEIGHT_RATIO: float = 5.0
    FOCAL_LOSS: bool = False
    # Device
    FORCE_CPU: bool = False
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    # Reproducibility
    RANDOM_SEED: int = 42
    DETERMINISTIC: bool = True
    # Persistence
    SAVE_DIR: str = "models/"
    LSTM_MODEL_PATH: str = "best_lstm_model.pth"
    GNN_MODEL_PATH: str = "best_gnn_model.pth"
    HYBRID_MODEL_PATH: str = "best_hybrid_model.pth"
    # Evaluation
    METRICS: tuple[str, ...] = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "auc_pr",
        "mcc",
    )
    THRESHOLD_OPTIMIZATION: bool = True
    # Explainability
    USE_SHAP: bool = False
    SHAP_SAMPLES: int = 100


def select_device(force_cpu: bool = False) -> torch.device:
    """Pick an available device, respecting a forced CPU flag."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_num_workers(device: torch.device, fallback: int = 0) -> int:
    """Choose a sensible num_workers based on device type."""
    if device.type == "cuda":
        return 4
    if device.type == "mps":
        return 0
    return fallback

