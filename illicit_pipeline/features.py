"""
Feature engineering routines.

Includes temporal feature creation and graph-based features derived from the
transaction network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm


TEMPORAL_FEATURES = [
    "time_step_normalized",
    "time_period_early",
    "time_period_mid",
    "time_period_late",
    "in_fraud_campaign",
    "rolling_tx_count_3",
    "time_since_last_tx",
    "tx_velocity",
    "in_burst_period",
    "is_rapid_tx",
    "tx_recurrence",
    "avg_time_appearance",
    "is_early_adopter",
    "is_late_entrant",
    "time_sin",
    "time_cos",
]

GRAPH_FEATURES = [
    "in_degree",
    "out_degree",
    "total_degree",
    "pagerank",
    "clustering_coef",
    "betweenness",
    "neighbor_avg_degree",
    "degree_ratio",
]


def add_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features in-place and return the updated DataFrame."""
    data = data.copy()
    data["time_step_normalized"] = (data["time_step"] - data["time_step"].min()) / (
        data["time_step"].max() - data["time_step"].min()
    )
    data["time_sin"] = np.sin(2 * np.pi * data["time_step"] / 49.0)
    data["time_cos"] = np.cos(2 * np.pi * data["time_step"] / 49.0)

    data["time_period_early"] = (data["time_step"] <= 16).astype(int)
    data["time_period_mid"] = ((data["time_step"] > 16) & (data["time_step"] <= 33)).astype(int)
    data["time_period_late"] = (data["time_step"] > 33).astype(int)
    fraud_campaign_periods = [9, 13, 15, 16, 20]
    data["in_fraud_campaign"] = data["time_step"].isin(fraud_campaign_periods).astype(int)

    # Rolling features per node
    rolling_features = []
    data_sorted = data.sort_values("time_step")
    for node_id, group in tqdm(
        data_sorted.groupby("node_ID"),
        desc="Temporal rolling features",
        leave=False,
    ):
        group = group.sort_values("time_step")
        rolling_count_3 = group["time_step"].rolling(window=3, min_periods=1).count()
        time_diff = group["time_step"].diff().fillna(0)
        time_span = group["time_step"].max() - group["time_step"].min()
        velocity = len(group) / (time_span + 1)
        temp_df = pd.DataFrame(
            {
                "node_ID": node_id,
                "time_step": group["time_step"],
                "rolling_tx_count_3": rolling_count_3.values,
                "time_since_last_tx": time_diff.values,
                "tx_velocity": velocity,
            },
            index=group.index,
        )
        rolling_features.append(temp_df)

    rolling_df = pd.concat(rolling_features, ignore_index=False)
    data = data.merge(
        rolling_df[["node_ID", "time_step", "rolling_tx_count_3", "time_since_last_tx", "tx_velocity"]],
        on=["node_ID", "time_step"],
        how="left",
    )
    data["rolling_tx_count_3"] = data["rolling_tx_count_3"].fillna(1)
    data["time_since_last_tx"] = data["time_since_last_tx"].fillna(0)
    data["tx_velocity"] = data["tx_velocity"].fillna(0)

    # Burst detection
    tx_freq_per_timestep = data.groupby("time_step").size()
    overall_mean_freq = tx_freq_per_timestep.mean()
    overall_std_freq = tx_freq_per_timestep.std()
    burst_threshold = overall_mean_freq + 2 * overall_std_freq
    burst_timesteps = tx_freq_per_timestep[tx_freq_per_timestep > burst_threshold].index.tolist()
    data["in_burst_period"] = data["time_step"].isin(burst_timesteps).astype(int)
    data["is_rapid_tx"] = ((data["time_since_last_tx"] > 0) & (data["time_since_last_tx"] <= 2)).astype(int)

    # Temporal aggregates
    early_threshold = data["time_step"].quantile(0.33)
    late_threshold = data["time_step"].quantile(0.67)
    data["is_early_adopter"] = (data["time_step"] <= early_threshold).astype(int)
    data["is_late_entrant"] = (data["time_step"] >= late_threshold).astype(int)
    tx_recurrence = data.groupby("node_ID").size().to_dict()
    data["tx_recurrence"] = data["node_ID"].map(tx_recurrence)
    avg_timestep = data.groupby("node_ID")["time_step"].mean().to_dict()
    data["avg_time_appearance"] = data["node_ID"].map(avg_timestep)

    return data


def add_graph_features(data: pd.DataFrame, edgelist: pd.DataFrame, train_cutoff: int = 35) -> pd.DataFrame:
    """
    Add graph-based features derived from the train-period transaction network.

    Uses only edges where both endpoints occur on or before `train_cutoff` to
    avoid temporal leakage.
    """
    data = data.copy()
    node_time_map = data.set_index("node_ID")["time_step"].to_dict()
    train_nodes_set = {n for n, t in node_time_map.items() if t <= train_cutoff}
    edge_mask = edgelist["txId1"].isin(train_nodes_set) & edgelist["txId2"].isin(train_nodes_set)
    filtered_edges = edgelist[edge_mask]

    G = nx.from_pandas_edgelist(filtered_edges, source="txId1", target="txId2", create_using=nx.DiGraph())

    # Degree features
    in_degree_dict = dict(G.in_degree())
    out_degree_dict = dict(G.out_degree())
    data["in_degree"] = data["node_ID"].map(in_degree_dict).fillna(0)
    data["out_degree"] = data["node_ID"].map(out_degree_dict).fillna(0)
    data["total_degree"] = data["in_degree"] + data["out_degree"]

    # PageRank
    pagerank_dict = nx.pagerank(G, max_iter=20, alpha=0.85)
    data["pagerank"] = data["node_ID"].map(pagerank_dict).fillna(0)

    # Clustering
    G_undirected = G.to_undirected()
    clustering_dict = nx.clustering(G_undirected)
    data["clustering_coef"] = data["node_ID"].map(clustering_dict).fillna(0)

    # Betweenness (approximate)
    k_sample = min(500, G.number_of_nodes())
    betweenness_dict = nx.betweenness_centrality(G, k=k_sample, normalized=True)
    data["betweenness"] = data["node_ID"].map(betweenness_dict).fillna(0)

    # Neighbor stats
    neighbor_avg_degree = {}
    for node in tqdm(G.nodes(), desc="Neighbor degrees", leave=False):
        neighbors = list(G.neighbors(node))
        neighbor_avg_degree[node] = np.mean([G.degree(n) for n in neighbors]) if neighbors else 0
    data["neighbor_avg_degree"] = data["node_ID"].map(neighbor_avg_degree).fillna(0)
    data["degree_ratio"] = np.where(
        data["out_degree"] > 0, data["in_degree"] / data["out_degree"], data["in_degree"]
    )

    return data

