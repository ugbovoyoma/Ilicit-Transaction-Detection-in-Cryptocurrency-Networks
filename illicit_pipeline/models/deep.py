"""
Deep learning models: LSTM (temporal) and GraphSAGE (graph-based).

These are simplified, modularized versions of the architectures in the
monolithic script. They are intended to be called from the pipeline orchestrator
and can be toggled on/off to control runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from ..config import DLConfig
from ..config import select_device, default_num_workers


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance - focuses on hard examples."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class SimpleLSTMFraudDetector(nn.Module):
    """Simplified LSTM for fraud detection."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        x = F.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        return self.fc2(x)


class GraphSAGEFraudDetector(nn.Module):
    """GraphSAGE model for fraud detection."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.fc = nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x)

    def get_embedding(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


@dataclass
class DeepResult:
    name: str
    y_pred: np.ndarray
    y_proba: np.ndarray
    metrics: Dict[str, float]


def prepare_lstm_datasets(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_names: list[str], config: DLConfig
) -> Tuple[DataLoader, DataLoader, torch.device]:
    device = select_device(config.FORCE_CPU)
    batch_size = config.BATCH_SIZE if device.type == "cuda" else min(config.BATCH_SIZE, 128)
    num_workers = default_num_workers(device, fallback=config.NUM_WORKERS if hasattr(config, "NUM_WORKERS") else 0)
    pin_memory = bool(getattr(config, "PIN_MEMORY", False))

    y_train = (train_df["class_label"] == "1").astype(int).values.astype(np.int64)
    y_test = (test_df["class_label"] == "1").astype(int).values.astype(np.int64)
    X_train = train_df[feature_names].values.astype(np.float32)
    X_test = test_df[feature_names].values.astype(np.float32)

    train_tensor = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_tensor = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(
        train_tensor, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_tensor, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, test_loader, device


def train_lstm(
    train_loader: DataLoader, test_loader: DataLoader, input_dim: int, config: DLConfig
) -> DeepResult:
    device = select_device(config.FORCE_CPU)
    model = SimpleLSTMFraudDetector(
        input_dim=input_dim,
        hidden_dim=config.LSTM_HIDDEN_DIM,
        num_layers=config.LSTM_NUM_LAYERS,
        dropout=config.LSTM_DROPOUT,
    ).to(device)

    # Class weights from training labels
    train_labels = torch.cat([y for _, y in train_loader]).cpu().numpy()
    class_weights_array = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights_array, dtype=torch.float32, device=device)

    criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(batch_y.numpy())

    y_prob = np.vstack(all_probs)[:, 1]
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    metrics = {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    return DeepResult("LSTM", y_pred, y_prob, metrics)


def prepare_graph_data(
    data: pd.DataFrame, edge_df: pd.DataFrame, feature_names: list[str], train_mask_ids: set, test_mask_ids: set
) -> Data:
    node_feature_df = data.set_index("node_ID")
    for col in feature_names:
        if col not in node_feature_df.columns:
            node_feature_df[col] = 0
    x = torch.tensor(node_feature_df[feature_names].fillna(0).values, dtype=torch.float)
    node_ids = node_feature_df.index.values
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    # Convert labels to numeric
    labels = node_feature_df["class_label"].map({"2": 0, "1": 1}).fillna(-1).values
    y = torch.tensor(labels, dtype=torch.long)

    edge_index_list = []
    for _, row in edge_df.iterrows():
        src_idx = node_to_idx.get(row["txId1"])
        dst_idx = node_to_idx.get(row["txId2"])
        if src_idx is not None and dst_idx is not None:
            edge_index_list.append([src_idx, dst_idx])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    train_mask = torch.tensor([node_ids[i] in train_mask_ids and labels[i] >= 0 for i in range(len(node_ids))])
    test_mask = torch.tensor([node_ids[i] in test_mask_ids and labels[i] >= 0 for i in range(len(node_ids))])

    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)


def train_graphsage(graph_data: Data, config: DLConfig) -> DeepResult:
    device = select_device(config.FORCE_CPU)
    model = GraphSAGEFraudDetector(
        in_channels=graph_data.num_node_features,
        hidden_channels=config.GNN_HIDDEN_DIM,
        num_layers=config.GNN_NUM_LAYERS,
        dropout=config.GNN_DROPOUT,
    ).to(device)
    graph_data = graph_data.to(device)

    train_labels_np = graph_data.y[graph_data.train_mask].cpu().numpy()
    class_weights = compute_class_weight("balanced", classes=np.unique(train_labels_np), y=train_labels_np)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = criterion(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        test_probs = F.softmax(out[graph_data.test_mask], dim=1).cpu().numpy()
        test_pred = out[graph_data.test_mask].argmax(dim=1).cpu().numpy()
        y_true = graph_data.y[graph_data.test_mask].cpu().numpy()

    y_prob = test_probs[:, 1]
    metrics = {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, test_pred),
    }
    return DeepResult("GraphSAGE", test_pred, y_prob, metrics)

