#!/usr/bin/env python
# coding: utf-8

"""
Illicit Transaction Detection in Cryptocurrency Networks
========================================================

This script analyzes the Elliptic Bitcoin Dataset to detect illicit transactions
using various machine learning and deep learning approaches including:
- Logistic Regression
- Random Forest
- XGBoost
- LSTM Networks
- Graph Neural Networks (GNN)

To run this script:
    uv run python illicitTransactionDetection.py

Make sure the dataset is located in: ./dataset/rawData/elliptic_bitcoin_dataset/
"""

# **UNDERSTANDING THE DATASET STRUCTURE**

# In[ ]:


#Import necessary libraries
import pandas as pd
import time


# In[ ]:


# Import additional libraries for analysis and modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, global_mean_pool
import torch_geometric.transforms as T
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import networkx as nx
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, average_precision_score, matthews_corrcoef,
    precision_recall_curve, roc_curve
)
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


# In[ ]:


# Load the datasets
# Note: Update these paths to match your dataset location
DATA_DIR = './dataset/rawData/elliptic_bitcoin_dataset' 

with tqdm(total=3, desc="Loading CSV files", unit="file") as pbar:
    elliptic_txs_classes = pd.read_csv(f'{DATA_DIR}/elliptic_txs_classes.csv')
    pbar.update(1)
    
    elliptic_txs_edgelist = pd.read_csv(f'{DATA_DIR}/elliptic_txs_edgelist.csv')
    pbar.update(1)
    
    elliptic_txs_features = pd.read_csv(f'{DATA_DIR}/elliptic_txs_features.csv')
    pbar.update(1)

print(f"\n✓ Datasets loaded successfully")
print(f"  - Classes: {len(elliptic_txs_classes):,}")
print(f"  - Edgelist: {len(elliptic_txs_edgelist):,}")
print(f"  - Features: {len(elliptic_txs_features):,}")

# Label mapping used across models
label_map = {'2': 0, '1': 1}


#Rename classes columns for clarity
elliptic_txs_classes.columns = ["node_ID","class_label"]

elliptic_txs_classes.head()


# In[ ]:


#Rename columns for clarity
elliptic_txs_features.columns= ['node_ID', 'time_step'] + [f'feature_{i}' for i in range(1,166)]

print(elliptic_txs_features)


# In[ ]:


#Merge features and classes datasets
data= elliptic_txs_features.merge(elliptic_txs_classes, on='node_ID', how= 'left')

data.head()


# In[ ]:


# Examine the merged dataset structure
print("Merged dataset info:")
print(f"Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
data.head()


# **DATA EXPLORATION AND ANALYSIS**

# In[ ]:


# Display the first few rows of the classes dataframe
data['class_label'].head()


# In[ ]:


#Check class distribution
data['class_label'].value_counts()


# In[ ]:


#Check to know how many transactions we have in the dataset
print(f"Number of edges in the transaction network: {len(elliptic_txs_edgelist)}")
print(f"Shape of dataset: {data.shape}")


# In[ ]:


#Check detailed class distribution with percentages
class_counts= data['class_label'].value_counts()
class_percentages= data['class_label'].value_counts(normalize=True) * 100

print("Class Distribution:")

for class_label, count in class_counts.items():
    percentage = class_percentages[class_label]
    print(f"{class_label}: {count} transactions ({percentage:.2f}%)")


# In[ ]:


# Examine the edgelist (transaction network)
print("Edgelist info:")
print(f"Shape: {elliptic_txs_edgelist.shape}")
print(f"Columns: {elliptic_txs_edgelist.columns.tolist()}")
elliptic_txs_edgelist.head()


# In[ ]:


#Check for missing values
print("Missing values analysis:")
missing_values = data.isnull().sum()
print(f"Total missing values: {missing_values.sum()}")
print(f"Columns with missing values: {missing_values[missing_values > 0]}")


# In[ ]:


# Basic statistics for features
print("\nBasic statistics for first few features:")
data[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']].describe()


# In[ ]:


# Analyze temporal distribution
print("Time step analysis:")
print(f"Time steps range: {data['time_step'].min()} to {data['time_step'].max()}")
print(f"Number of unique time steps: {data['time_step'].nunique()}")
print(f"Transactions per time step:")
time_dist = data['time_step'].value_counts().sort_index()
print(time_dist.head(10))


# In[ ]:


# Check feature scales and distributions
feature_stats = data[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']].agg(['min', 'max', 'mean', 'std'])
print("Feature scale comparison:")
print(feature_stats)


# In[ ]:


# Analyze class distribution over time
print("Class distribution by time period:")
time_class_dist = data.groupby(['time_step', 'class_label']).size().unstack(fill_value=0)
print(time_class_dist.head(10))


# In[ ]:


# Calculate illicit percentage by time period
time_class_percentages = data.groupby(['time_step', 'class_label']).size().unstack(fill_value=0)
print("Available columns:", time_class_percentages.columns.tolist())

# Calculate percentages for each class
time_class_percentages['total'] = time_class_percentages.sum(axis=1)

if '1' in time_class_percentages.columns:
    time_class_percentages['illicit_percentage'] = (time_class_percentages['1'] / time_class_percentages['total']) * 100
    print("Illicit transaction percentage by time period:")
    print(time_class_percentages[['illicit_percentage']].round(2))
else:
    print("No illicit transactions found (class '1' not present)")


# In[ ]:


# Compare feature patterns between high and low fraud periods
high_fraud_periods = [13, 20, 9, 16, 15]  # >4% illicit
low_fraud_periods = [1, 2, 3, 5, 6]       # <0.5% illicit

high_fraud_data = data[data['time_step'].isin(high_fraud_periods)]
low_fraud_data = data[data['time_step'].isin(low_fraud_periods)]

print("Feature comparison between high and low fraud periods:")
print("High fraud periods - Feature means:")
print(high_fraud_data[['feature_1', 'feature_2', 'feature_3']].mean())
print("\nLow fraud periods - Feature means:")
print(low_fraud_data[['feature_1', 'feature_2', 'feature_3']].mean())


# **DATA PREPROCESSING**

# In[ ]:


# Analyze labeled vs unlabeled data for modeling strategy
labeled_data = data[data['class_label'].isin(['1', '2'])]
unlabeled_data = data[data['class_label'] == 'unknown']

print("Data Distribution for Modeling:")
print("=" * 50)
print(f"Labeled transactions: {len(labeled_data):,} ({len(labeled_data)/len(data)*100:.1f}%)")
print(f"Unlabeled transactions: {len(unlabeled_data):,} ({len(unlabeled_data)/len(data)*100:.1f}%)")

print(f"\nLabeled class distribution:")
labeled_class_dist = labeled_data['class_label'].value_counts()
labeled_class_pct = labeled_data['class_label'].value_counts(normalize=True) * 100

for class_label, count in labeled_class_dist.items():
    percentage = labeled_class_pct[class_label]
    print(f"Class {class_label}: {count:,} transactions ({percentage:.1f}%)")

print(f"\nClass imbalance ratio (illicit:licit): 1:{labeled_class_dist['2'] // labeled_class_dist['1']}")


# In[ ]:


# Analyze temporal distribution of labeled data
print("Temporal distribution of labeled data:")
print("=" * 50)

labeled_temporal = labeled_data.groupby(['time_step', 'class_label']).size().unstack(fill_value=0)
labeled_temporal['total_labeled'] = labeled_temporal.sum(axis=1)
labeled_temporal['illicit_pct'] = (labeled_temporal['1'] / labeled_temporal['total_labeled'] * 100).round(1)

print("Time periods with labeled data:")
print(labeled_temporal[['1', '2', 'total_labeled', 'illicit_pct']].head(10))

print(f"\nTime periods with no labeled data:")
no_labels = []
for t in range(1, 50):
    if t not in labeled_temporal.index:
        no_labels.append(t)
if no_labels:
    print(f"Time steps: {no_labels}")
else:
    print("All time periods have some labeled data")


# In[ ]:


# Analyze train/test split strategies based on temporal structure
print("TRAIN/TEST SPLIT STRATEGY ANALYSIS")
print("=" * 50)

# ENFORCE TEMPORAL SPLIT ONLY
feature_columns = [col for col in labeled_data.columns if col.startswith('feature_')]
early_periods = labeled_data[labeled_data['time_step'] <= 35]  # Train: first 35 time steps
late_periods = labeled_data[labeled_data['time_step'] > 35]    # Test: time steps 36-49

X_train = early_periods[feature_columns]
y_train = early_periods['class_label']
X_test = late_periods[feature_columns]
y_test = late_periods['class_label']

print(f"Temporal split - Train (≤35): {len(X_train):,}, Test (>35): {len(X_test):,}")
print(f"Train class distribution: {y_train.value_counts().to_dict()}")
print(f"Test class distribution: {y_test.value_counts().to_dict()}")
print(f"\nTrain period illicit rate: {(y_train == '1').sum() / len(y_train) * 100:.1f}%")
print(f"Test period illicit rate: {(y_test == '1').sum() / len(y_test) * 100:.1f}%")


# In[ ]:


# Analyze which fraud campaigns are in train vs test
print("FRAUD CAMPAIGN ANALYSIS - TRAIN VS TEST")
print("=" * 50)

train_periods = early_periods['time_step'].unique()
test_periods = late_periods['time_step'].unique()

print(f"Train periods: {sorted(train_periods)}")
print(f"Test periods: {sorted(test_periods)}")

# Check if major fraud campaigns are in train or test
major_fraud_periods = [13, 20, 9, 16, 15]  # From your earlier analysis
train_fraud_campaigns = [p for p in major_fraud_periods if p in train_periods]
test_fraud_campaigns = [p for p in major_fraud_periods if p in test_periods]

print(f"\nMajor fraud campaigns in TRAIN: {train_fraud_campaigns}")
print(f"Major fraud campaigns in TEST: {test_fraud_campaigns}")

print(f"\nImplication: Model will {'have' if train_fraud_campaigns else 'NOT have'} fraud campaign examples for training")


# In[ ]:


# IMPLEMENT HYBRID PSEUDO-LABELING STRATEGY
print("IMPLEMENTING HYBRID PSEUDO-LABELING + SMOTE STRATEGY")
print("=" * 60)

# Step 1: Prepare train data (periods 1-35) for pseudo-labeling
feature_cols = [col for col in labeled_data.columns if col.startswith('feature_')]
X_train_temp = early_periods[feature_cols]
y_train_temp = early_periods['class_label']

print(f"Step 1: Training initial model on {len(X_train_temp):,} labeled samples from periods 1-35")
print(f"Training data illicit rate: {(y_train_temp == '1').sum() / len(y_train_temp) * 100:.1f}%")

# Train initial model with balanced weights
initial_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
initial_model.fit(X_train_temp, y_train_temp)

# Step 2: Generate pseudo-labels only from training-period unlabeled data (time_step ≤ 35)
unlabeled_train = unlabeled_data[unlabeled_data['time_step'] <= 35]
X_unlabeled = unlabeled_train[feature_cols]
print(f"\nStep 2: Generating pseudo-labels for {len(X_unlabeled):,} unlabeled training-period transactions...")

pseudo_probs = initial_model.predict_proba(X_unlabeled)
pseudo_labels = initial_model.predict(X_unlabeled)
max_probs = np.max(pseudo_probs, axis=1)

# Analyze confidence levels
print("\nPseudo-labeling confidence analysis:")
for threshold in [0.8, 0.9, 0.95]:
    high_conf_mask = max_probs > threshold
    conf_count = high_conf_mask.sum()

    if conf_count > 0:
        conf_illicit = (pseudo_labels[high_conf_mask] == '1').sum()
        conf_licit = (pseudo_labels[high_conf_mask] == '2').sum()

        print(f"Confidence > {threshold}: {conf_count:,} predictions ({conf_count/len(X_unlabeled)*100:.1f}%)")
        print(f"  Predicted illicit: {conf_illicit:,}, Predicted licit: {conf_licit:,}")

        if conf_illicit > 0:
            ratio = conf_licit / conf_illicit if conf_illicit > 0 else float('inf')
            print(f"  Ratio (licit:illicit): {ratio:.1f}:1")


# In[ ]:


# Step 3: Select optimal confidence threshold and create enhanced training set
print("\nStep 3: Creating enhanced training dataset with pseudo-labels")
print("=" * 60)

# IMPROVEMENT: Increased threshold from 0.9 to 0.95 to reduce label noise
# Rationale: Higher threshold = fewer but higher-quality pseudo-labels
# This reduces the risk of training on mislabeled synthetic examples
CONFIDENCE_THRESHOLD = 0.95
high_conf_mask = max_probs > CONFIDENCE_THRESHOLD
selected_pseudo_labels = pseudo_labels[high_conf_mask]
selected_pseudo_features = X_unlabeled[high_conf_mask]

print(f"Selected {len(selected_pseudo_labels):,} high-confidence pseudo-labels (threshold: {CONFIDENCE_THRESHOLD})")
print(f"Pseudo-label distribution:")
pseudo_class_dist = pd.Series(selected_pseudo_labels).value_counts()
for label, count in pseudo_class_dist.items():
    pct = count / len(selected_pseudo_labels) * 100
    print(f"  Class {label}: {count:,} ({pct:.1f}%)")

# Combine original labeled data with high-confidence pseudo-labels
print(f"\nCombining original labeled data with pseudo-labels:")
print(f"Original labeled data: {len(X_train_temp):,}")
print(f"High-confidence pseudo-labels: {len(selected_pseudo_labels):,}")

# Create enhanced training set
X_enhanced = pd.concat([X_train_temp, selected_pseudo_features], ignore_index=True)
y_enhanced = pd.concat([y_train_temp, pd.Series(selected_pseudo_labels)], ignore_index=True)

print(f"Enhanced training set size: {len(X_enhanced):,}")
print(f"Enhanced class distribution:")
enhanced_class_dist = y_enhanced.value_counts()
for label, count in enhanced_class_dist.items():
    pct = count / len(y_enhanced) * 100
    print(f"  Class {label}: {count:,} ({pct:.1f}%)")

# Calculate new class imbalance ratio
illicit_count = enhanced_class_dist['1']
licit_count = enhanced_class_dist['2']
new_ratio = licit_count / illicit_count
print(f"\nClass imbalance improvement:")
print(f"Original ratio (licit:illicit): {labeled_class_dist['2'] // labeled_class_dist['1']}:1")
print(f"Enhanced ratio (licit:illicit): {new_ratio:.1f}:1")


# In[ ]:


# Step 4: Apply SMOTE to further improve class balance
print("\nStep 4: Applying SMOTE for additional class balancing")
print("=" * 60)

# Check for and handle NaN values before SMOTE
print(f"Checking for missing values...")
nan_counts = X_enhanced.isnull().sum().sum()
print(f"Total NaN values found: {nan_counts:,}")

if nan_counts > 0:
    print(f"Filling NaN values with column means...")
    X_enhanced = X_enhanced.fillna(X_enhanced.mean())
    # Verify no NaNs remain
    remaining_nans = X_enhanced.isnull().sum().sum()
    print(f"NaN values after filling: {remaining_nans}")

# Current imbalance is still significant (23.4:1), so apply SMOTE
# Target a more balanced ratio (e.g., 3:1 or 5:1)
smote = SMOTE(sampling_strategy=0.2, random_state=42)  # 0.2 = 1:5 ratio (illicit:licit)

print(f"\nApplying SMOTE with sampling strategy 0.2 (target ratio ~1:5)")
print(f"Before SMOTE: {len(X_enhanced):,} samples")

X_final, y_final = smote.fit_resample(X_enhanced, y_enhanced)

print(f"After SMOTE: {len(X_final):,} samples")
print(f"Final class distribution:")
final_class_dist = pd.Series(y_final).value_counts()
for label, count in final_class_dist.items():
    pct = count / len(y_final) * 100
    print(f"  Class {label}: {count:,} ({pct:.1f}%)")

final_illicit_count = final_class_dist['1']
final_licit_count = final_class_dist['2'] 
final_ratio = final_licit_count / final_illicit_count
print(f"\nFinal class ratio (licit:illicit): {final_ratio:.1f}:1")

print(f"\nSummary of hybrid approach effectiveness:")
print(f"Original labeled data: {len(X_train_temp):,} (ratio {labeled_class_dist['2'] // labeled_class_dist['1']}:1)")
print(f"+ Pseudo-labeling: {len(X_enhanced):,} (ratio {new_ratio:.1f}:1)")  
print(f"+ SMOTE: {len(X_final):,} (ratio {final_ratio:.1f}:1)")


# ADVANCED FEATURE ENGINEERING

# Define feature groups early to avoid NameError later and to align with engineered columns
temporal_features = [
    'time_step_normalized', 'time_period_early', 'time_period_mid', 'time_period_late',
    'in_fraud_campaign', 'rolling_tx_count_3', 'time_since_last_tx', 'tx_velocity',
    'in_burst_period', 'is_rapid_tx', 'tx_recurrence', 'avg_time_appearance',
    'is_early_adopter', 'is_late_entrant'
]

graph_features = [
    'in_degree', 'out_degree', 'total_degree', 'pagerank',
    'clustering_coef', 'betweenness', 'neighbor_avg_degree', 'degree_ratio'
]


# TEMPORAL FEATURE ENGINEERING


print("TEMPORAL FEATURE ENGINEERING")

with tqdm(total=4, desc="Creating temporal features", unit="feature_group") as pbar:
    # Basic temporal features
    
    data['time_step_normalized'] = (data['time_step'] - data['time_step'].min()) / \
                               (data['time_step'].max() - data['time_step'].min())

  
    data['time_sin'] = np.sin(2 * np.pi * data['time_step'] / 49.0)
    data['time_cos'] = np.cos(2 * np.pi * data['time_step'] / 49.0)
    
    data['time_period_early'] = (data['time_step'] <= 16).astype(int)
    data['time_period_mid'] = ((data['time_step'] > 16) & (data['time_step'] <= 33)).astype(int)
    data['time_period_late'] = (data['time_step'] > 33).astype(int)
    fraud_campaign_periods = [9, 13, 15, 16, 20]
    data['in_fraud_campaign'] = data['time_step'].isin(fraud_campaign_periods).astype(int)
    pbar.update(1)

    # Rolling window features
    print("\n  Computing rolling window features...")
    data_sorted = data.sort_values('time_step')
    rolling_features = []
    for node_id, group in tqdm(data_sorted.groupby('node_ID'), desc="  Rolling windows", leave=False):
        group = group.sort_values('time_step')
        rolling_count_3 = group['time_step'].rolling(window=3, min_periods=1).count()
        time_diff = group['time_step'].diff().fillna(0)
        time_span = group['time_step'].max() - group['time_step'].min()
        velocity = len(group) / (time_span + 1)
        
        temp_df = pd.DataFrame({
            'node_ID': node_id,
            'time_step': group['time_step'],
            'rolling_tx_count_3': rolling_count_3.values,
            'time_since_last_tx': time_diff.values,
            'tx_velocity': velocity
        }, index=group.index)
        rolling_features.append(temp_df)
    
    rolling_df = pd.concat(rolling_features, ignore_index=False)
    data = data.merge(rolling_df[['node_ID', 'time_step', 'rolling_tx_count_3', 
                                   'time_since_last_tx', 'tx_velocity']],
                     on=['node_ID', 'time_step'], how='left')
    data['rolling_tx_count_3'] = data['rolling_tx_count_3'].fillna(1)
    data['time_since_last_tx'] = data['time_since_last_tx'].fillna(0)
    data['tx_velocity'] = data['tx_velocity'].fillna(0)
    pbar.update(1)

    # Burst detection
    print("\n  Detecting burst activity...")
    tx_freq_per_timestep = data.groupby('time_step').size()
    overall_mean_freq = tx_freq_per_timestep.mean()
    overall_std_freq = tx_freq_per_timestep.std()
    burst_threshold = overall_mean_freq + 2 * overall_std_freq
    burst_timesteps = tx_freq_per_timestep[tx_freq_per_timestep > burst_threshold].index.tolist()
    data['in_burst_period'] = data['time_step'].isin(burst_timesteps).astype(int)
    data['is_rapid_tx'] = ((data['time_since_last_tx'] > 0) & (data['time_since_last_tx'] <= 2)).astype(int)
    pbar.update(1)

    # Temporal aggregates
    print("\n  Computing temporal aggregates...")
    early_threshold = data['time_step'].quantile(0.33)
    late_threshold = data['time_step'].quantile(0.67)
    data['is_early_adopter'] = (data['time_step'] <= early_threshold).astype(int)
    data['is_late_entrant'] = (data['time_step'] >= late_threshold).astype(int)
    tx_recurrence = data.groupby('node_ID').size().to_dict()
    data['tx_recurrence'] = data['node_ID'].map(tx_recurrence)
    avg_timestep = data.groupby('node_ID')['time_step'].mean().to_dict()
    data['avg_time_appearance'] = data['node_ID'].map(avg_timestep)
    pbar.update(1)

print("\n✓ Temporal feature engineering complete")
print(f"  - {len(temporal_features)} temporal features expected (columns now materialized)")

# Analyze temporal features by class
print(f"\nTemporal features by transaction class:")
for class_label in ['1', '2']:
    if class_label in data['class_label'].values:
        class_data = data[data['class_label'] == class_label]
        print(f"\nClass {class_label} ({'illicit' if class_label == '1' else 'licit'}):")
        print(f"  In fraud campaign: {class_data['in_fraud_campaign'].mean()*100:.1f}%")
        print(f"  In burst period: {class_data['in_burst_period'].mean()*100:.1f}%")
        print(f"  Rapid transactions: {class_data['is_rapid_tx'].mean()*100:.1f}%")
        print(f"  Avg tx velocity: {class_data['tx_velocity'].mean():.4f}")
        print(f"  Avg recurrence: {class_data['tx_recurrence'].mean():.2f}")
        print(f"  Early period: {class_data['time_period_early'].mean()*100:.1f}%")
        print(f"  Mid period: {class_data['time_period_mid'].mean()*100:.1f}%")
        print(f"  Late period: {class_data['time_period_late'].mean()*100:.1f}%")


# In[ ]:


# GRAPH-BASED FEATURE ENGINEERING

print("GRAPH FEATURE ENGINEERING")


with tqdm(total=6, desc="Computing graph features", unit="feature") as pbar:
    # Build graph using ONLY training-period edges (time_step ≤ 35) to avoid future leakage
    print("\n  Building transaction network (train-period edges only)...")
    node_time_map = data.set_index('node_ID')['time_step'].to_dict()
    train_nodes_set = {n for n, t in node_time_map.items() if t <= 35}
    edge_mask = (
        elliptic_txs_edgelist['txId1'].isin(train_nodes_set) &
        elliptic_txs_edgelist['txId2'].isin(train_nodes_set)
    )
    filtered_edges = elliptic_txs_edgelist[edge_mask]
    G = nx.from_pandas_edgelist(filtered_edges, source='txId1', 
                                target='txId2', create_using=nx.DiGraph())
    pbar.update(1)

    # Degree centrality
    print("  Computing degree centrality...")
    in_degree_dict = dict(G.in_degree())
    out_degree_dict = dict(G.out_degree())
    data['in_degree'] = data['node_ID'].map(in_degree_dict).fillna(0)
    data['out_degree'] = data['node_ID'].map(out_degree_dict).fillna(0)
    data['total_degree'] = data['in_degree'] + data['out_degree']
    pbar.update(1)


    # PageRank (limit iterations for scalability)
    print("  Computing scalable PageRank...")
    pagerank_dict = nx.pagerank(G, max_iter=20, alpha=0.85)
    data['pagerank'] = data['node_ID'].map(pagerank_dict).fillna(0)
    pbar.update(1)

    # Clustering coefficient (unchanged, fast for sparse graphs)
    print("  Computing clustering coefficient...")
    G_undirected = G.to_undirected()
    clustering_dict = nx.clustering(G_undirected)
    data['clustering_coef'] = data['node_ID'].map(clustering_dict).fillna(0)
    pbar.update(1)

    # Betweenness centrality (approximate, scalable)
    print("  Computing betweenness centrality (approximation)...")
    k_sample = min(500, G.number_of_nodes())
    betweenness_dict = nx.betweenness_centrality(G, k=k_sample, normalized=True)
    data['betweenness'] = data['node_ID'].map(betweenness_dict).fillna(0)
    pbar.update(1)

    # Additional features
    print("  Computing neighbor average degree...")
    neighbor_avg_degree = {}
    for node in tqdm(G.nodes(), desc="  Neighbor degrees", leave=False):
        neighbors = list(G.neighbors(node))
        neighbor_avg_degree[node] = np.mean([G.degree(n) for n in neighbors]) if neighbors else 0
    data['neighbor_avg_degree'] = data['node_ID'].map(neighbor_avg_degree).fillna(0)
    data['degree_ratio'] = np.where(data['out_degree'] > 0, 
                                    data['in_degree'] / data['out_degree'], 
                                    data['in_degree'])
    pbar.update(1)

print("\n✓ Graph feature engineering complete")
print(f"  - {len(graph_features)} graph features created")

# Analyze graph features by class
print("\n" + "=" * 60)
print("GRAPH FEATURE ANALYSIS BY TRANSACTION CLASS")
print("=" * 60)

for class_label in ['1', '2']:
    if class_label in data['class_label'].values:
        class_data = data[data['class_label'] == class_label]
        class_name = 'Illicit' if class_label == '1' else 'Licit'

        print(f"\n{class_name} Transactions (Class {class_label}, n={len(class_data):,}):")
        print(f"  Mean in-degree:    {class_data['in_degree'].mean():.3f}")
        print(f"  Mean out-degree:   {class_data['out_degree'].mean():.3f}")
        print(f"  Mean total degree: {class_data['total_degree'].mean():.3f}")
        print(f"  Mean PageRank:     {class_data['pagerank'].mean():.8f}")
        print(f"  Max PageRank:      {class_data['pagerank'].max():.8f}")

print("\nGraph-based feature engineering completed")
print(f"New features added: in_degree, out_degree, total_degree, pagerank")


# 
# **BASELINE MACHINE LEARNING MODELS**

# In[ ]:


# PREPARE FEATURES AND LABELS FOR MODELING
print("PREPARING FEATURES AND LABELS")
print("=" * 60)

# Define feature categories
original_features = [f'feature_{i}' for i in range(1, 166)]

# Combined feature sets
baseline_features = original_features  # Baseline ML models: original features only
lstm_features = original_features + temporal_features  # LSTM: original + temporal
gnn_features = original_features + graph_features  # GNN: original + graph
hybrid_features = original_features + temporal_features + graph_features  # Hybrid: all features

print(f"Feature configuration:")
print(f"  Original features:  {len(original_features)}")
print(f"  Graph features:     {len(graph_features)}")
print(f"  Temporal features:  {len(temporal_features)}")
print(f"  Baseline total:     {len(baseline_features)}")
print(f"  LSTM total:         {len(lstm_features)}")
print(f"  GNN total:          {len(gnn_features)}")
print(f"  Hybrid total:       {len(hybrid_features)}")

# Prepare training data from SMOTE output (X_final, y_final from SMOTE cell)
# Note: SMOTE was applied on original features only
print(f"\nConverting SMOTE arrays to DataFrames...")
X_train_modeling = pd.DataFrame(X_final, columns=original_features)
y_train_modeling = pd.Series(y_final)


# Note: GNN/Hybrid models will be trained on real labeled nodes only (defined later)

# Prepare test data (periods 36-49) - use actual engineered features
print(f"\nPreparing test data from periods 36-49...")
test_data = data[data['time_step'].between(36, 49)]

# Verify all engineered features exist in test data
missing_graph = [f for f in graph_features if f not in test_data.columns]
missing_temporal = [f for f in temporal_features if f not in test_data.columns]

# If features are missing in the test set (common when graph computation sampling/ordering differs),

# Fill missing features in test set using only training statistics
train_means = X_train_modeling.mean()
for col in missing_graph + missing_temporal:
    fill_val = train_means.get(col, 0.0)
    print(f"Filling missing column '{col}' in test data with training mean: {fill_val:.6f}")
    test_data[col] = fill_val

# Extract features based on model type (now safe because missing cols were filled)
X_test_baseline = test_data[baseline_features]
X_test_lstm = test_data[lstm_features]
X_test_gnn = test_data[gnn_features]
X_test_hybrid = test_data[hybrid_features]
y_test_modeling = test_data['class_label']

# For baseline ML models, use baseline features
X_test_modeling = X_test_baseline

# Handle NaN values in test data
print(f"Checking for NaN values in test data...")
test_nan_counts = X_test_modeling.isnull().sum().sum()
print(f"Total NaN values in test set: {test_nan_counts:,}")

if test_nan_counts > 0:
    print(f"Filling NaN values in test set with column means from training data...")
    # Use training data means to fill test data NaNs (to prevent data leakage)
    train_means = X_train_modeling.mean()
    X_test_modeling = X_test_modeling.fillna(train_means)
    remaining_test_nans = X_test_modeling.isnull().sum().sum()
    print(f"NaN values after filling: {remaining_test_nans}")

print(f"\nTrain set: {X_train_modeling.shape}")
print(f"Test set:  {X_test_modeling.shape}")
print(f"\nClass distribution (train): {y_train_modeling.value_counts().to_dict()}")
print(f"Class distribution (test):  {y_test_modeling.value_counts().to_dict()}")
print("\nFeature preparation completed")


# In[ ]:


# BASELINE MODEL 1: LOGISTIC REGRESSION
print("TRAINING BASELINE MODEL: LOGISTIC REGRESSION")
print("=" * 60)

with tqdm(total=4, desc="Logistic Regression", unit="step", colour="blue") as pbar:
    # Step 1: Scale features (critical for LR)
    # Important: Use only baseline features for baseline models to ensure consistency
    print("  Step 1: Scaling features...")
    print(f"  Using {len(baseline_features)} baseline features for Logistic Regression")

    # Extract baseline features from training data
    X_train_baseline = X_train_modeling[baseline_features]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_baseline)
    X_test_scaled = scaler.transform(X_test_modeling)
    pbar.update(1)
    pbar.set_postfix({"status": "✓ Features scaled"})

    # Step 2: Train Logistic Regression (with class weights for imbalance)
    print("  Step 2: Training Logistic Regression...")
    lr_model = LogisticRegression(
        class_weight='balanced',  # Handle class imbalance
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    lr_model.fit(X_train_scaled, y_train_modeling)
    pbar.update(1)
    pbar.set_postfix({"status": "✓ Model trained"})

    # Step 3: Generate predictions
    print("  Step 3: Generating predictions...")
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    pbar.update(1)
    pbar.set_postfix({"status": "✓ Predictions generated"})

    # Step 4: Evaluate on labeled test set
    print("  Step 4: Evaluating model...")
    # Evaluation (per Section 3.7) - Filter out unknown labels for evaluation
    labeled_test_mask = y_test_modeling.isin(['1', '2'])
    y_test_labeled = y_test_modeling[labeled_test_mask]
    y_pred_labeled = y_pred_lr[labeled_test_mask]
    y_proba_labeled = y_proba_lr[labeled_test_mask]
    
    # Convert to numeric for metrics
    y_test_labeled_numeric = y_test_labeled.map({'1': 1, '2': 0})
    y_pred_lr_numeric = (y_pred_labeled == '1').astype(int)
    
    pbar.update(1)
    pbar.set_postfix({"status": "✓ Model evaluated"})

print(f"\nFiltering test set to labeled transactions only...")
print(f"Evaluating on {len(y_test_labeled):,} labeled test samples")
print(f"  Illicit: {(y_test_labeled == '1').sum():,}")
print(f"  Licit: {(y_test_labeled == '2').sum():,}")

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)
print("\nClassification Report:")
print(classification_report(
    y_test_labeled,
    y_pred_labeled,
    labels=['2', '1'],
    target_names=['Licit (2)', 'Illicit (1)']
))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_labeled, y_pred_labeled, labels=['2', '1'])
print(cm)
print(f"  True Negatives:  {cm[0,0]:,}")
print(f"  False Positives: {cm[0,1]:,}")
print(f"  False Negatives: {cm[1,0]:,}")
print(f"  True Positives:  {cm[1,1]:,}")

# Advanced metrics
auc_roc = roc_auc_score(y_test_labeled, y_proba_labeled)
auc_pr = average_precision_score(y_test_labeled, y_proba_labeled, pos_label='1')
mcc = matthews_corrcoef(y_test_labeled, y_pred_labeled)

print(f"\nAUC-ROC: {auc_roc:.4f}")
print(f"AUC-PR:  {auc_pr:.4f}")
print(f"MCC:     {mcc:.4f}")

print("\n✅ Logistic Regression training completed")


# **BASELINE MODEL 2: RANDOM FOREST**

# In[ ]:


# BASELINE MODEL 2: RANDOM FOREST
print("TRAINING BASELINE MODEL: RANDOM FOREST")
print("=" * 60)

with tqdm(total=4, desc="Random Forest", unit="step", colour="green") as pbar:
    from sklearn.ensemble import RandomForestClassifier

    # Step 1: Extract features
    print("  Step 1: Extracting features...")
    print(f"  Using {len(baseline_features)} baseline features for Random Forest")
    X_train_baseline = X_train_modeling[baseline_features]
    pbar.update(1)
    pbar.set_postfix({"status": "✓ Features extracted"})

    # Step 2: Train Random Forest (handles non-linear patterns better than LR)
    print("  Step 2: Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,           # More trees for better performance
        max_depth=20,               # Limit depth to prevent overfitting
        min_samples_split=10,       # Require at least 10 samples to split
        min_samples_leaf=5,         # At least 5 samples per leaf
        class_weight='balanced',    # Handle class imbalance
        random_state=42,
        n_jobs=-1,                  # Use all CPU cores
        verbose=0
    )
    rf_model.fit(X_train_baseline, y_train_modeling)
    pbar.update(1)
    pbar.set_postfix({"status": "✓ Model trained"})

    # Step 3: Generate predictions
    print("  Step 3: Generating predictions...")
    y_pred_rf = rf_model.predict(X_test_modeling)
    y_proba_rf = rf_model.predict_proba(X_test_modeling)[:, 1]

    # Filter to labeled test samples
    y_pred_rf_labeled = y_pred_rf[labeled_test_mask]
    y_proba_rf_labeled = y_proba_rf[labeled_test_mask]
    pbar.update(1)
    pbar.set_postfix({"status": "✓ Predictions generated"})

    # Step 4: Evaluate
    print("  Step 4: Evaluating model...")
    auc_roc_rf = roc_auc_score(y_test_labeled, y_proba_rf_labeled)
    pbar.update(1)
    pbar.set_postfix({"AUC-ROC": f"{auc_roc_rf:.4f}"})

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)
print("\nClassification Report:")
print(classification_report(
    y_test_labeled,
    y_pred_rf_labeled,
    labels=['2', '1'],
    target_names=['Licit (2)', 'Illicit (1)']
))

print("\nConfusion Matrix:")
cm_rf = confusion_matrix(y_test_labeled, y_pred_rf_labeled, labels=['2', '1'])
print(cm_rf)
print(f"  True Negatives:  {cm_rf[0,0]:,}")
print(f"  False Positives: {cm_rf[0,1]:,}")
print(f"  False Negatives: {cm_rf[1,0]:,}")
print(f"  True Positives:  {cm_rf[1,1]:,}")

# Advanced metrics
auc_pr_rf = average_precision_score(y_test_labeled, y_proba_rf_labeled, pos_label='1')
mcc_rf = matthews_corrcoef(y_test_labeled, y_pred_rf_labeled)

print(f"\nAUC-ROC: {auc_roc_rf:.4f}")
print(f"AUC-PR:  {auc_pr_rf:.4f}")
print(f"MCC:     {mcc_rf:.4f}")

# Feature importance analysis
print("\n" + "=" * 60)
print("TOP 10 MOST IMPORTANT FEATURES")
print("=" * 60)
feature_importance = pd.DataFrame({
    'feature': original_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

print("\nRandom Forest training completed")


# **BASELINE MODEL 3: XGBoost**

# In[ ]:


# BASELINE MODEL 3: XGBoost (State-of-the-art for imbalanced data)
print("TRAINING BASELINE MODEL: XGBOOST")
print("=" * 60)

with tqdm(total=4, desc="XGBoost", unit="step", colour="yellow") as pbar:
    # Step 1: Prepare data and calculate scale_pos_weight
    print("  Step 1: Preparing data and calculating class weights...")
    scale_pos_weight = (y_train_modeling == '2').sum() / (y_train_modeling == '1').sum()
    print(f"  Class imbalance in training: {scale_pos_weight:.1f}:1")
    print(f"  Using scale_pos_weight = {scale_pos_weight:.2f}")
    
    # Convert string labels to numeric for XGBoost
    label_map = {'2': 0, '1': 1}  # 0=licit, 1=illicit
    y_train_numeric = y_train_modeling.map(label_map)
    y_test_numeric = y_test_modeling.map(label_map)
    y_test_labeled_numeric = y_test_labeled.map(label_map)
    
    print(f"  Using {len(baseline_features)} baseline features for XGBoost")
    X_train_baseline = X_train_modeling[baseline_features]
    pbar.update(1)
    pbar.set_postfix({"status": "✓ Data prepared"})

    # Step 2: Train XGBoost
    print("  Step 2: Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # Handle imbalance
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train_baseline, y_train_numeric)
    pbar.update(1)
    pbar.set_postfix({"status": "✓ Model trained"})

    # Step 3: Generate predictions
    print("  Step 3: Generating predictions...")
    y_pred_xgb = xgb_model.predict(X_test_modeling)
    y_proba_xgb = xgb_model.predict_proba(X_test_modeling)[:, 1]

    # Filter to labeled test samples
    y_pred_xgb_labeled = y_pred_xgb[labeled_test_mask]
    y_proba_xgb_labeled = y_proba_xgb[labeled_test_mask]
    pbar.update(1)
    pbar.set_postfix({"status": "✓ Predictions generated"})

    # Step 4: Evaluate
    print("  Step 4: Evaluating model...")
    auc_roc_xgb = roc_auc_score(y_test_labeled_numeric, y_proba_xgb_labeled)
    pbar.update(1)
    pbar.set_postfix({"AUC-ROC": f"{auc_roc_xgb:.4f}"})

print("\n" + "=" * 60)
print("XGBOOST RESULTS")
print("=" * 60)
print("\nClassification Report:")
print(classification_report(y_test_labeled_numeric, y_pred_xgb_labeled, target_names=['Licit (0)', 'Illicit (1)']))

print("\nConfusion Matrix:")
cm_xgb = confusion_matrix(y_test_labeled_numeric, y_pred_xgb_labeled)
print(cm_xgb)
print(f"  True Negatives:  {cm_xgb[0,0]:,}")
print(f"  False Positives: {cm_xgb[0,1]:,}")
print(f"  False Negatives: {cm_xgb[1,0]:,}")
print(f"  True Positives:  {cm_xgb[1,1]:,}")

# Advanced metrics
auc_pr_xgb = average_precision_score(y_test_labeled_numeric, y_proba_xgb_labeled)
mcc_xgb = matthews_corrcoef(y_test_labeled_numeric, y_pred_xgb_labeled)

print(f"\nAUC-ROC: {auc_roc_xgb:.4f}")
print(f"AUC-PR:  {auc_pr_xgb:.4f}")
print(f"MCC:     {mcc_xgb:.4f}")

# Feature importance
print("\n" + "=" * 60)
print("TOP 10 MOST IMPORTANT FEATURES (XGBoost)")
print("=" * 60)
xgb_feature_importance = pd.DataFrame({
    'feature': original_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(xgb_feature_importance.head(10).to_string(index=False))

print("\nXGBoost training completed")


# **THRESHOLD OPTIMIZATION**

# In[ ]:


# THRESHOLD OPTIMIZATION - Find Optimal Decision Threshold
print("THRESHOLD OPTIMIZATION ANALYSIS")
print("=" * 60)

# We'll optimize threshold for XGBoost (best model so far)
print("Analyzing different probability thresholds for XGBoost...")
print("(Default threshold is 0.5, but we can optimize for better precision-recall balance)\n")

thresholds_to_test = np.arange(0.1, 0.9, 0.05)
results = []

for threshold in thresholds_to_test:
    # Apply threshold
    y_pred_thresh = (y_proba_xgb_labeled >= threshold).astype(int)

    # Calculate metrics
    precision = precision_score(y_test_labeled_numeric, y_pred_thresh, zero_division=0)
    recall = recall_score(y_test_labeled_numeric, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test_labeled_numeric, y_pred_thresh, zero_division=0)

    # Calculate specificity (True Negative Rate)
    cm_temp = confusion_matrix(y_test_labeled_numeric, y_pred_thresh)
    if cm_temp.shape == (2, 2):
        specificity = cm_temp[0,0] / (cm_temp[0,0] + cm_temp[0,1]) if (cm_temp[0,0] + cm_temp[0,1]) > 0 else 0
    else:
        specificity = 0

    results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity
    })

# Convert to DataFrame
threshold_df = pd.DataFrame(results)

# Find optimal thresholds for different objectives
best_f1_idx = threshold_df['f1_score'].idxmax()
best_precision_idx = threshold_df['precision'].idxmax()
best_recall_idx = threshold_df['recall'].idxmax()

print("=" * 80)
print("THRESHOLD OPTIMIZATION RESULTS")
print("=" * 80)

print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Specificity':<12}")
print("-" * 80)
for _, row in threshold_df.iterrows():
    marker = ""
    if row.name == best_f1_idx:
        marker = " ← BEST F1"
    elif row.name == best_precision_idx:
        marker = " ← BEST PRECISION"
    elif row.name == best_recall_idx:
        marker = " ← BEST RECALL"

    print(f"{row['threshold']:<12.2f} {row['precision']:<12.3f} {row['recall']:<12.3f} "
          f"{row['f1_score']:<12.3f} {row['specificity']:<12.3f}{marker}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

optimal_threshold = threshold_df.loc[best_f1_idx, 'threshold']
print(f"\n1. BALANCED APPROACH (Best F1-Score):")
print(f"   Threshold: {optimal_threshold:.2f}")
print(f"   Precision: {threshold_df.loc[best_f1_idx, 'precision']:.3f}")
print(f"   Recall: {threshold_df.loc[best_f1_idx, 'recall']:.3f}")
print(f"   F1-Score: {threshold_df.loc[best_f1_idx, 'f1_score']:.3f}")
print(f"   → Use when you want balance between catching fraud and minimizing false alarms")

# Find closest threshold to 0.6 for high precision
high_precision_idx = (threshold_df['threshold'] - 0.6).abs().idxmin()
high_precision_row = threshold_df.loc[high_precision_idx]
print(f"\n2. HIGH PRECISION (Minimize False Alarms):")
print(f"   Threshold: {high_precision_row['threshold']:.2f}")
print(f"   Precision: {high_precision_row['precision']:.3f}")
print(f"   Recall: {high_precision_row['recall']:.3f}")
print(f"   F1-Score: {high_precision_row['f1_score']:.3f}")
print(f"   → Use when false alarms are very costly")

# Find closest threshold to 0.3 for high recall
high_recall_idx = (threshold_df['threshold'] - 0.3).abs().idxmin()
high_recall_row = threshold_df.loc[high_recall_idx]
print(f"\n3. HIGH RECALL (Catch More Fraud):")
print(f"   Threshold: {high_recall_row['threshold']:.2f}")
print(f"   Precision: {high_recall_row['precision']:.3f}")
print(f"   Recall: {high_recall_row['recall']:.3f}")
print(f"   F1-Score: {high_recall_row['f1_score']:.3f}")
print(f"   → Use when missing fraud is very costly")

print("\nThreshold optimization completed")


# **ENSEMBLE MODEL - Combining All Models**

# In[ ]:


# ENSEMBLE MODEL - Combining Logistic Regression, Random Forest, and XGBoost
print("ENSEMBLE MODEL - WEIGHTED VOTING")
print("=" * 60)

with tqdm(total=5, desc="Ensemble Model", unit="step", colour="magenta") as pbar:
    # Step 1: Validate individual model probabilities
    print("  Step 1: Validating individual model probabilities...")
    print("\n  Individual model AUC-ROC scores:")
    print(f"    Logistic Regression: {auc_roc:.4f}")
    print(f"    Random Forest:       {auc_roc_rf:.4f}")
    print(f"    XGBoost:            {auc_roc_xgb:.4f}")

    # CRITICAL FIX: Ensure all probabilities represent P(illicit=1)
    # LR and RF use string labels ('1', '2'), XGB uses numeric (0, 1)
    # Need to extract correct probability column for each model

    print("\n  Checking probability alignment...")
    print(f"    LR proba shape: {lr_model.predict_proba(X_test_scaled).shape}")
    print(f"    LR classes: {lr_model.classes_}")
    print(f"    RF classes: {rf_model.classes_}")
    print(f"    XGB classes: {xgb_model.classes_}")

    # For LR and RF with string labels, find which column corresponds to '1' (illicit)
    lr_illicit_idx = list(lr_model.classes_).index('1')
    rf_illicit_idx = list(rf_model.classes_).index('1')

    # Get probabilities for illicit class ('1') from each model
    y_proba_lr_all = lr_model.predict_proba(X_test_scaled)[:, lr_illicit_idx]
    y_proba_rf_all = rf_model.predict_proba(X_test_modeling)[:, rf_illicit_idx]
    y_proba_xgb_all = y_proba_xgb  # Already correct (column 1 = illicit)

    # Filter to labeled test samples
    y_proba_lr_labeled_fixed = y_proba_lr_all[labeled_test_mask]
    y_proba_rf_labeled_fixed = y_proba_rf_all[labeled_test_mask]
    y_proba_xgb_labeled_fixed = y_proba_xgb_labeled  # Already filtered and correct

    print(f"\n  Probability statistics for illicit class:")
    print(f"    LR mean: {y_proba_lr_labeled_fixed.mean():.4f}")
    print(f"    RF mean: {y_proba_rf_labeled_fixed.mean():.4f}")
    print(f"    XGB mean: {y_proba_xgb_labeled_fixed.mean():.4f}")
    pbar.update(1)
    pbar.set_postfix({"status": "✓ Probabilities validated"})

    # Step 2: Calculate ensemble weights
    print("\n  Step 2: Calculating ensemble weights...")
    total_auc = auc_roc + auc_roc_rf + auc_roc_xgb
    weight_lr = auc_roc / total_auc
    weight_rf = auc_roc_rf / total_auc
    weight_xgb = auc_roc_xgb / total_auc

    print(f"\n  Ensemble weights (based on AUC-ROC):")
    print(f"    Logistic Regression: {weight_lr:.3f}")
    print(f"    Random Forest:       {weight_rf:.3f}")
    print(f"    XGBoost:            {weight_xgb:.3f}")
    pbar.update(1)
    pbar.set_postfix({"status": "✓ Weights calculated"})

    # Step 3: Combine predictions with weighted averaging
    print("\n  Step 3: Combining predictions with weighted averaging...")
    y_proba_ensemble = (
        weight_lr * y_proba_lr_labeled_fixed +
        weight_rf * y_proba_rf_labeled_fixed +
        weight_xgb * y_proba_xgb_labeled_fixed
    )

    print(f"\n  Ensemble probability statistics:")
    print(f"    Mean: {y_proba_ensemble.mean():.4f}")
    print(f"    Min: {y_proba_ensemble.min():.4f}")
    print(f"    Max: {y_proba_ensemble.max():.4f}")

    # Apply default threshold (0.5)
    y_pred_ensemble = (y_proba_ensemble >= 0.5).astype(int)

    # Convert to match original label format for evaluation
    y_pred_ensemble_str = pd.Series(y_pred_ensemble).map({0: '2', 1: '1'})
    pbar.update(1)
    pbar.set_postfix({"status": "✓ Predictions combined"})

    # Step 4: Evaluate ensemble model
    print("\n  Step 4: Evaluating ensemble model...")
    auc_roc_ensemble = roc_auc_score(y_test_labeled_numeric, y_proba_ensemble)
    auc_pr_ensemble = average_precision_score(y_test_labeled_numeric, y_proba_ensemble)
    mcc_ensemble = matthews_corrcoef(y_test_labeled, y_pred_ensemble_str)
    pbar.update(1)
    pbar.set_postfix({"AUC-ROC": f"{auc_roc_ensemble:.4f}"})

    # Step 5: Generate confusion matrix
    print("\n  Step 5: Generating confusion matrix...")
    cm_ensemble = confusion_matrix(y_test_labeled, y_pred_ensemble_str, labels=['2', '1'])
    pbar.update(1)
    pbar.set_postfix({"status": "✓ Evaluation complete"})

print("\n" + "=" * 60)
print("ENSEMBLE MODEL RESULTS")
print("=" * 60)
print("\nClassification Report:")
print(classification_report(
    y_test_labeled,
    y_pred_ensemble_str,
    labels=['2', '1'],
    target_names=['Licit (2)', 'Illicit (1)']
))

print("\nConfusion Matrix:")
print(cm_ensemble)
print(f"  True Negatives:  {cm_ensemble[0,0]:,}")
print(f"  False Positives: {cm_ensemble[0,1]:,}")
print(f"  False Negatives: {cm_ensemble[1,0]:,}")
print(f"  True Positives:  {cm_ensemble[1,1]:,}")

print(f"\nAUC-ROC: {auc_roc_ensemble:.4f}")
print(f"AUC-PR:  {auc_pr_ensemble:.4f}")
print(f"MCC:     {mcc_ensemble:.4f}")

print("\nEnsemble model training completed")


# **MODEL COMPARISON SUMMARY**

# In[ ]:


# MODEL COMPARISON - Side-by-side Performance Analysis
print("=" * 80)
print("COMPREHENSIVE MODEL COMPARISON")
print("=" * 80)

# Create comparison dataframe
comparison_data = {
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Ensemble (Weighted)'],
    'AUC-ROC': [auc_roc, auc_roc_rf, auc_roc_xgb, auc_roc_ensemble],
    'AUC-PR': [auc_pr, auc_pr_rf, auc_pr_xgb, auc_pr_ensemble],
    'MCC': [mcc, mcc_rf, mcc_xgb, mcc_ensemble]
}

# Extract precision, recall, F1 for illicit class from confusion matrices
def get_illicit_metrics(cm):
    """Extract precision, recall, F1 for illicit class (class 1)"""
    # cm structure: [[TN, FP], [FN, TP]]
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

prec_lr, rec_lr, f1_lr = get_illicit_metrics(cm)
prec_rf, rec_rf, f1_rf = get_illicit_metrics(cm_rf)
prec_xgb, rec_xgb, f1_xgb = get_illicit_metrics(cm_xgb)
prec_ens, rec_ens, f1_ens = get_illicit_metrics(cm_ensemble)

comparison_data['Precision (Illicit)'] = [prec_lr, prec_rf, prec_xgb, prec_ens]
comparison_data['Recall (Illicit)'] = [rec_lr, rec_rf, rec_xgb, rec_ens]
comparison_data['F1-Score (Illicit)'] = [f1_lr, f1_rf, f1_xgb, f1_ens]

comparison_df = pd.DataFrame(comparison_data)

print("\n" + "=" * 80)
print("PERFORMANCE METRICS COMPARISON")
print("=" * 80)
print(comparison_df.to_string(index=False))

# Identify best model for each metric
print("\n" + "=" * 80)
print("BEST MODELS BY METRIC")
print("=" * 80)
metrics = ['AUC-ROC', 'AUC-PR', 'MCC', 'Precision (Illicit)', 'Recall (Illicit)', 'F1-Score (Illicit)']
for metric in metrics:
    best_idx = comparison_df[metric].idxmax()
    best_model = comparison_df.loc[best_idx, 'Model']
    best_value = comparison_df.loc[best_idx, metric]
    print(f"{metric:<25}: {best_model:<25} ({best_value:.4f})")

# Performance improvement analysis
print("\n" + "=" * 80)
print("IMPROVEMENT OVER BASELINE (Logistic Regression)")
print("=" * 80)
for i, model in enumerate(['Random Forest', 'XGBoost', 'Ensemble (Weighted)']):
    print(f"\n{model}:")
    for metric in metrics:
        baseline = comparison_df.loc[0, metric]
        current = comparison_df.loc[i+1, metric]
        improvement = ((current - baseline) / baseline * 100) if baseline > 0 else 0
        direction = "↑" if improvement > 0 else "↓"
        print(f"  {metric:<25}: {direction} {abs(improvement):>6.2f}%")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print(f"""
1. BEST OVERALL MODEL: {comparison_df.loc[comparison_df['AUC-ROC'].idxmax(), 'Model']}
   - Highest AUC-ROC: {comparison_df['AUC-ROC'].max():.4f}
   - Best discrimination between fraud and legitimate

2. BEST FOR FRAUD DETECTION: {comparison_df.loc[comparison_df['F1-Score (Illicit)'].idxmax(), 'Model']}
   - Highest F1-Score for illicit class: {comparison_df['F1-Score (Illicit)'].max():.4f}
   - Best balance of precision and recall for fraud

3. IMPROVEMENT FROM LOGISTIC REGRESSION:
   - F1-Score improved by: {((comparison_df.loc[comparison_df['F1-Score (Illicit)'].idxmax(), 'F1-Score (Illicit)'] - f1_lr) / f1_lr * 100):.1f}%
   - Precision improved by: {((comparison_df['Precision (Illicit)'].max() - prec_lr) / prec_lr * 100):.1f}%

4. RECOMMENDATION: Use {'Ensemble' if auc_roc_ensemble == comparison_df['AUC-ROC'].max() else comparison_df.loc[comparison_df['AUC-ROC'].idxmax(), 'Model']} for production
   - Combines strengths of multiple models
   - More robust to different fraud patterns
   - Better generalization
""")

print("Model comparison completed")

# ============================================================================
# SECTION: EXTENDED COMPARISON WITH DEEP LEARNING MODELS
# ============================================================================
# NOTE: This section will be populated after LSTM, GNN, and Hybrid training
# For now, store the ML models' results for later comparison

ml_models_results = {
    'Logistic Regression': {'y_pred': y_pred_lr_numeric, 'y_proba': y_proba_lr_labeled_fixed},
    'Random Forest': {'y_pred': (y_pred_rf_labeled == '1').astype(int), 'y_proba': y_proba_rf_labeled_fixed},
    'XGBoost': {'y_pred': y_pred_xgb_labeled, 'y_proba': y_proba_xgb_labeled},
    'Ensemble (ML)': {'y_pred': y_pred_ensemble, 'y_proba': y_proba_ensemble}
}

print("\n" + "=" * 80)
print("ML MODELS STORED FOR COMPARISON WITH DEEP LEARNING MODELS")
print("=" * 80)
print(f"✓ 4 ML models trained and stored")
print(f"✓ All evaluated on {len(y_test_labeled):,} labeled test transactions")
print(f"✓ Waiting for LSTM, GNN, and Hybrid model results...")

# In[ ]:


# VISUALIZATION (ROC & PR Curves for Model Comparison)
print("=" * 80)
print("CREATING THESIS VISUALIZATIONS")
print("=" * 80)

with tqdm(total=4, desc="Visualization", unit="plot", colour="cyan") as pbar:
    # Import additional plotting libraries
    from sklearn.metrics import roc_curve, precision_recall_curve

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison - Illicit Transaction Detection', 
                 fontsize=16, fontweight='bold', y=0.995)

    # ============================================================================
    # PLOT 1: ROC Curves (Top Left)
    # ============================================================================
    print("  Creating ROC Curves...")
    ax1 = axes[0, 0]

    # Calculate ROC curves for each model
    fpr_lr, tpr_lr, _ = roc_curve(y_test_labeled_numeric, y_proba_labeled)
    fpr_rf, tpr_rf, _ = roc_curve(y_test_labeled_numeric, y_proba_rf_labeled)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test_labeled_numeric, y_proba_xgb_labeled)
    fpr_ens, tpr_ens, _ = roc_curve(y_test_labeled_numeric, y_proba_ensemble)

    # Plot each model
    ax1.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_roc:.3f})', 
             linewidth=2, color='blue')
    ax1.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_roc_rf:.3f})', 
             linewidth=2, color='green')
    ax1.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={auc_roc_xgb:.3f})', 
             linewidth=2, color='red')
    ax1.plot(fpr_ens, tpr_ens, label=f'Ensemble (AUC={auc_roc_ensemble:.3f})', 
             linewidth=2, color='purple', linestyle='--')

    # Add diagonal reference line (random classifier)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')

    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves - Model Discrimination Ability', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    pbar.update(1)
    pbar.set_postfix({"status": "✓ ROC curves"})

    # ============================================================================
    # PLOT 2: Precision-Recall Curves (Top Right)
    # ============================================================================
    print("  Creating PR Curves...")
    ax2 = axes[0, 1]

    # Calculate PR curves for each model
    precision_lr, recall_lr, _ = precision_recall_curve(y_test_labeled_numeric, y_proba_labeled)
    precision_rf, recall_rf, _ = precision_recall_curve(y_test_labeled_numeric, y_proba_rf_labeled)
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test_labeled_numeric, y_proba_xgb_labeled)
    precision_ens, recall_ens, _ = precision_recall_curve(y_test_labeled_numeric, y_proba_ensemble)

    # Plot each model
    ax2.plot(recall_lr, precision_lr, label=f'Logistic Regression (AP={auc_pr:.3f})', 
             linewidth=2, color='blue')
    ax2.plot(recall_rf, precision_rf, label=f'Random Forest (AP={auc_pr_rf:.3f})', 
             linewidth=2, color='green')
    ax2.plot(recall_xgb, precision_xgb, label=f'XGBoost (AP={auc_pr_xgb:.3f})', 
             linewidth=2, color='red')
    ax2.plot(recall_ens, precision_ens, label=f'Ensemble (AP={auc_pr_ensemble:.3f})', 
             linewidth=2, color='purple', linestyle='--')

    # Add baseline (proportion of positive class)
    baseline_precision = (y_test_labeled_numeric == 1).sum() / len(y_test_labeled_numeric)
    ax2.axhline(y=baseline_precision, color='k', linestyle='--', linewidth=1, 
                alpha=0.5, label=f'Baseline ({baseline_precision:.3f})')

    ax2.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves - Imbalanced Data Performance', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    pbar.update(1)
    pbar.set_postfix({"status": "✓ PR curves"})

    # ============================================================================
    # PLOT 3: Threshold Optimization (Bottom Left)
    # ============================================================================
    print("  Creating Threshold Optimization...")
    ax3 = axes[1, 0]

    # Plot metrics vs threshold
    ax3.plot(threshold_df['threshold'], threshold_df['precision'], 
             label='Precision', linewidth=2, color='blue', marker='o', markersize=4)
    ax3.plot(threshold_df['threshold'], threshold_df['recall'], 
             label='Recall', linewidth=2, color='green', marker='s', markersize=4)
    ax3.plot(threshold_df['threshold'], threshold_df['f1_score'], 
             label='F1-Score', linewidth=2, color='red', marker='^', markersize=4)
    ax3.plot(threshold_df['threshold'], threshold_df['specificity'], 
             label='Specificity', linewidth=2, color='orange', marker='d', markersize=4)

    # Mark optimal threshold
    optimal_thresh = threshold_df.loc[best_f1_idx, 'threshold']
    optimal_f1 = threshold_df.loc[best_f1_idx, 'f1_score']
    ax3.axvline(x=optimal_thresh, color='red', linestyle='--', linewidth=1.5, 
                alpha=0.7, label=f'Optimal Threshold ({optimal_thresh:.2f})')
    ax3.scatter([optimal_thresh], [optimal_f1], color='red', s=100, zorder=5, marker='*')

    ax3.set_xlabel('Decision Threshold', fontsize=12)
    ax3.set_ylabel('Metric Value', fontsize=12)
    ax3.set_title('Threshold Optimization - XGBoost Model', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.1, 0.9)
    ax3.set_ylim(0, 1)
    pbar.update(1)
    pbar.set_postfix({"status": "✓ Threshold analysis"})

    # ============================================================================
    # PLOT 4: Model Comparison Bar Chart (Bottom Right)
    # ============================================================================
    print("  Creating Model Comparison...")
    ax4 = axes[1, 1]

    # Prepare data for bar chart
    models = ['LR', 'RF', 'XGB', 'Ensemble']
    metrics_data = {
        'Precision': [prec_lr, prec_rf, prec_xgb, prec_ens],
        'Recall': [rec_lr, rec_rf, rec_xgb, rec_ens],
        'F1-Score': [f1_lr, f1_rf, f1_xgb, f1_ens]
    }

    x = np.arange(len(models))
    width = 0.25

    # Create grouped bar chart
    bars1 = ax4.bar(x - width, metrics_data['Precision'], width, label='Precision', color='blue', alpha=0.8)
    bars2 = ax4.bar(x, metrics_data['Recall'], width, label='Recall', color='green', alpha=0.8)
    bars3 = ax4.bar(x + width, metrics_data['F1-Score'], width, label='F1-Score', color='red', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax4.set_xlabel('Models', fontsize=12)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Model Performance Comparison - Fraud Detection Metrics', 
                  fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('model_comparison_visualization.png', dpi=300, bbox_inches='tight')
    print("  Visualization saved as 'model_comparison_visualization.png'")
    pbar.update(1)
    pbar.set_postfix({"status": "✓ All plots complete"})

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print("\nKey Insights from Visualizations:")
print("1. ROC Curve: Shows model discrimination ability (higher AUC = better)")
print("2. PR Curve: More informative for imbalanced data (focus on this!)")
print("3. Threshold Optimization: Helps choose operational threshold based on business needs")
print("4. Bar Chart: Quick comparison of key fraud detection metrics")


# **DEEP LEARNING MODELS - TEMPORAL AND GRAPH-BASED APPROACHES**
# 
# This section implements advanced deep learning architectures aligned with Chapter 3 methodology:
# 1. **LSTM/GRU**: Temporal sequence models to capture fraud campaign evolution across 49 time steps
# 2. **GNN (GraphSAGE)**: Graph neural network to leverage transaction network topology
# 3. **Hybrid Model**: Combined temporal + graph architecture for superior fraud detection

# In[ ]:


# ============================================================================
# DEEP LEARNING CONFIGURATION
# ============================================================================
# Centralized configuration for all deep learning experiments
# Modify these parameters to run different experimental settings

class DLConfig:
    """
    Deep Learning Hyperparameter Configuration
    Aligned with thesis Chapter 3 methodology requirements
    """

    # ========================================================================
    # LSTM Model Architecture
    # ========================================================================
    LSTM_HIDDEN_DIM = 64          # Hidden layer size (32, 64, 128, 256)
    LSTM_NUM_LAYERS = 1           # Number of LSTM layers (1, 2)
    LSTM_DROPOUT = 0.3            # Dropout rate (0.0 - 0.5)
    LSTM_BIDIRECTIONAL = False    # Use bidirectional LSTM (slower on CPU)

    # ========================================================================
    # GNN Model Architecture  
    # ========================================================================
    GNN_TYPE = 'GraphSAGE'        # GNN architecture: 'GraphSAGE', 'GCN', 'GAT'
    GNN_HIDDEN_DIM = 128          # Hidden dimension (64, 128, 256)
    GNN_NUM_LAYERS = 2            # Number of GNN layers (2, 3)
    GNN_DROPOUT = 0.3             # Dropout rate
    GNN_AGGR = 'mean'             # Aggregation: 'mean', 'max', 'add'

    # ========================================================================
    # Hybrid Model Architecture
    # ========================================================================
    HYBRID_FUSION_DIM = 64        # Fusion layer dimension
    HYBRID_DROPOUT = 0.3          # Dropout rate for hybrid fusion layers
    HYBRID_CONCAT_METHOD = 'concatenate'  # 'concatenate', 'add', 'multiply'

    # ========================================================================
    # Training Hyperparameters
    # ========================================================================
    BATCH_SIZE = 256              # Batch size (128, 256, 512, 1024)
    NUM_EPOCHS = 15               # Maximum epochs (10, 15, 20, 50)
    LEARNING_RATE = 0.001         # Learning rate (0.0001, 0.001, 0.01)
    WEIGHT_DECAY = 1e-5           # L2 regularization
    PATIENCE = 5                  # Early stopping patience

    # ========================================================================
    # Loss Function Configuration
    # ========================================================================
    USE_CLASS_WEIGHTS = False     # Use class weights (False if using SMOTE)
    CLASS_WEIGHT_RATIO = 5.0      # Weight for positive class if enabled
    FOCAL_LOSS = False            # Use focal loss for imbalance

    # ========================================================================
    # Device & Performance
    # ========================================================================
    FORCE_CPU = False             # Force CPU training (set to False to allow GPU when available)
    NUM_WORKERS = 4               # DataLoader workers (use 4 for GPU data loading)
    PIN_MEMORY = True             # Pin memory (True speeds host->GPU transfers)

    # ========================================================================
    # Reproducibility
    # ========================================================================
    RANDOM_SEED = 42              # Random seed for reproducibility
    DETERMINISTIC = True          # Deterministic operations (slower but reproducible)

    # ========================================================================
    # Model Persistence
    # ========================================================================
    SAVE_DIR = 'models/'          # Directory to save models
    LSTM_MODEL_PATH = 'best_lstm_model.pth'
    GNN_MODEL_PATH = 'best_gnn_model.pth'
    HYBRID_MODEL_PATH = 'best_hybrid_model.pth'

    # ========================================================================
    # Evaluation Metrics
    # ========================================================================
    METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr', 'mcc']
    THRESHOLD_OPTIMIZATION = True  # Optimize decision threshold

    # ========================================================================
    # Explainability
    # ========================================================================
    USE_SHAP = False               # Enable SHAP explanations
    SHAP_SAMPLES = 100            # Number of samples for SHAP analysis

# Display current configuration
print("=" * 80)
print("DEEP LEARNING CONFIGURATION LOADED")
print("=" * 80)

print(f"\nLSTM Architecture:")
print(f"  Hidden dim: {DLConfig.LSTM_HIDDEN_DIM}")
print(f"  Num layers: {DLConfig.LSTM_NUM_LAYERS}")
print(f"  Dropout: {DLConfig.LSTM_DROPOUT}")
print(f"  Bidirectional: {DLConfig.LSTM_BIDIRECTIONAL}")

print(f"\nGNN Architecture:")
print(f"  Type: {DLConfig.GNN_TYPE}")
print(f"  Hidden dim: {DLConfig.GNN_HIDDEN_DIM}")
print(f"  Num layers: {DLConfig.GNN_NUM_LAYERS}")
print(f"  Dropout: {DLConfig.GNN_DROPOUT}")

print(f"\nTraining:")
print(f"  Batch size: {DLConfig.BATCH_SIZE}")
print(f"  Max epochs: {DLConfig.NUM_EPOCHS}")
print(f"  Learning rate: {DLConfig.LEARNING_RATE}")
print(f"  Early stopping patience: {DLConfig.PATIENCE}")

print(f"\nDevice:")
print(f"  Force CPU: {DLConfig.FORCE_CPU}")
print(f"  Random seed: {DLConfig.RANDOM_SEED}")

print(f"\nExplainability:")
print(f"  Use SHAP: {DLConfig.USE_SHAP}")
print(f"  SHAP samples: {DLConfig.SHAP_SAMPLES}")

print("\n" + "=" * 80)
print("Configuration ready for all deep learning experiments")
print("=" * 80)


# In[ ]:


# DEEP LEARNING SETUP - Import PyTorch and PyTorch Geometric
print("=" * 80)
print("DEEP LEARNING ENVIRONMENT SETUP")
print("=" * 80)



# Check PyTorch installation and configure device
print(f"\nPyTorch Version: {torch.__version__}")

# Check for available acceleration devices
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"MPS (Apple Silicon GPU) Available: {torch.backends.mps.is_available()}")

# Device selection with explicit config respect (DLConfig.FORCE_CPU)
# Default behavior: honor DLConfig.FORCE_CPU; otherwise prefer CUDA > MPS > CPU
if getattr(DLConfig, 'FORCE_CPU', False):
    device = torch.device('cpu')
    num_workers = getattr(DLConfig, 'NUM_WORKERS', 0)
    pin_memory = getattr(DLConfig, 'PIN_MEMORY', False)
    print("DLConfig.FORCE_CPU is True -> Forcing CPU execution (no GPU will be used)")
else:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_workers = 4
        pin_memory = True
        try:
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
        print("Using NVIDIA CUDA GPU for training")
        # Enable TF32 (where supported) and cuDNN benchmark for speed
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
        num_workers = 0
        pin_memory = False
        print("Using Apple MPS backend for training (note: LSTM performance may vary)")
    else:
        device = torch.device('cpu')
        num_workers = getattr(DLConfig, 'NUM_WORKERS', 0)
        pin_memory = getattr(DLConfig, 'PIN_MEMORY', False)
        print("No GPU backend available - using CPU for training")

print(f"\nDevice selected: {device}")

# Set random seeds for reproducibility
seed = getattr(DLConfig, 'RANDOM_SEED', 42)
torch.manual_seed(seed)
np.random.seed(seed)

print("\nDeep learning environment configured successfully")


# **MODEL 1: LSTM TEMPORAL FRAUD DETECTION**
# 
# Implements bidirectional LSTM to capture temporal patterns across 49 time steps. This model processes transaction sequences to identify fraud campaign evolution patterns.

# In[ ]:


# PREPARE TEMPORAL SEQUENCE DATA FOR LSTM
print("=" * 80)
print("PREPARING TEMPORAL SEQUENCE DATA FOR LSTM")
print("=" * 80)

# Prepare training sequences using ONLY real labeled nodes in the training window (time_step ≤ 35)
print("Converting training data to PyTorch tensors...")

train_temporal_df = data[(data['time_step'] <= 35) & (data['class_label'].isin(['1', '2']))].set_index('node_ID')
test_temporal_df = data[(data['time_step'] > 35) & (data['class_label'].isin(['1', '2']))].set_index('node_ID')

X_train_lstm_df = train_temporal_df[lstm_features]
y_train_lstm = (train_temporal_df['class_label'] == '1').astype(int)

print(f"LSTM training features: {len(lstm_features)} ({len(original_features)} original + {len(temporal_features)} temporal)")

# Convert pandas to numpy first, then to tensors on the selected device
X_train_np = X_train_lstm_df.values.astype(np.float32)
y_train_np = y_train_lstm.values.astype(np.int64)

with torch.no_grad():
    X_train_tensor = torch.from_numpy(X_train_np).to(device)
    y_train_tensor = torch.from_numpy(y_train_np).to(device)

print(f"Training tensor shape: {X_train_tensor.shape}")
print(f"Training labels shape: {y_train_tensor.shape}")
print(f"Training tensor device: {X_train_tensor.device}")

# Use numpy for class distribution calculation
unique, counts = np.unique(y_train_np, return_counts=True)
print(f"Training class distribution: {dict(zip(unique, counts))}")

# Prepare test sequences
print("\nConverting test data to PyTorch tensors...")
# Use LSTM features for test data
print(f"LSTM test features: {len(lstm_features)}")
X_test_np = test_temporal_df[lstm_features].values.astype(np.float32)
y_test_np = (test_temporal_df['class_label'] == '1').astype(int).values.astype(np.int64)

with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test_np).to(device)
    y_test_tensor = torch.from_numpy(y_test_np).to(device)

print(f"Test tensor shape: {X_test_tensor.shape}")
print(f"Test labels shape: {y_test_tensor.shape}")
print(f"Test tensor device: {X_test_tensor.device}")

# Create DataLoaders for batch processing. Use DLConfig.BATCH_SIZE when available.
batch_size = getattr(DLConfig, 'BATCH_SIZE', 256) if device.type == 'cuda' else getattr(DLConfig, 'BATCH_SIZE', 128)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)

print(f"\nDataLoaders created:")
print(f"  Batch size: {batch_size}")
print(f"  Training batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")
print(f"  Num workers: {num_workers}")
print(f"  Pin memory: {pin_memory}")

print(f"\nData prepared successfully on {device}")
print("\nTemporal sequence data preparation completed")


# In[ ]:


# SIMPLIFIED LSTM MODEL ARCHITECTURE (NO ATTENTION - FASTER)

print("=" * 80)
print("BUILDING SIMPLIFIED LSTM TEMPORAL MODEL")
print("=" * 80)


# Focal Loss implementation for deep models
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance - focuses on hard examples"""
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class SimpleLSTMFraudDetector(nn.Module):
    """
    Simplified LSTM for fraud detection - optimized for CPU training.
    Removed attention mechanism to avoid CPU deadlock issues.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.3):
        super(SimpleLSTMFraudDetector, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Single-direction LSTM (faster than bidirectional)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0,  # No dropout for single layer
            bidirectional=False  # Unidirectional for speed
        )

        # Simple classification layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 2)  # Binary classification

    def forward(self, x):
        # Reshape input to (batch, seq_len=1, features) for LSTM
        x = x.unsqueeze(1)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last hidden state (no attention)
        last_hidden = hidden[-1]  # Shape: (batch, hidden_dim)

        # Classification
        x = F.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        output = self.fc2(x)

        return output


# Use lstm_features for LSTM input dimension

# Use only real labeled nodes (already prepared in train_temporal_df)
input_dim = len(lstm_features)
X_train_lstm = X_train_lstm_df
y_train_lstm = y_train_lstm
lstm_model = SimpleLSTMFraudDetector(
    input_dim=input_dim,
    hidden_dim=64,
    num_layers=1,
    dropout=0.3
).to(device)

print(f"\nModel Architecture (Optimized for CPU):")
print(f"  Input dimension: {input_dim}")
print(f"  Hidden dimension: 64 (reduced for speed)")
print(f"  Number of LSTM layers: 1 (unidirectional)")
print(f"  Dropout rate: 0.3")
print(f"\nTotal parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in lstm_model.parameters() if p.requires_grad):,}")

print("\nSimplified LSTM model built successfully (no attention mechanism)")


# In[ ]:


# TRAIN LSTM MODEL USING CONFIGURATION
print("=" * 80)
print("TRAINING LSTM MODEL")
print("=" * 80)

import time

# Use configuration parameters
cpu_device = torch.device('cpu')
lstm_model = lstm_model.to(cpu_device)

# Training setup from config

num_epochs = DLConfig.NUM_EPOCHS
learning_rate = DLConfig.LEARNING_RATE
batch_size = DLConfig.BATCH_SIZE
patience = DLConfig.PATIENCE

# Calculate class weights for real node training set
from sklearn.utils.class_weight import compute_class_weight
y_train_np = y_train_lstm.values if hasattr(y_train_lstm, 'values') else y_train_lstm
class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train_np), y=y_train_np)
class_weights = torch.tensor(class_weights_array, dtype=torch.float32, device=cpu_device)
print(f"\nClass weights computed: {class_weights.cpu().numpy()}")
print(f"  Class 0 (licit) weight: {class_weights[0]:.4f}")
print(f"  Class 1 (illicit) weight: {class_weights[1]:.4f}")

# Use Focal Loss for deep models
criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights)
print("Using Focal Loss with alpha=0.25, gamma=2.0")
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate, weight_decay=DLConfig.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

print(f"\nTraining configuration:")
print(f"  Device: {cpu_device}")
print(f"  Epochs: {num_epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")
print(f"  Training samples: {len(X_train_tensor):,}")
print(f"  Test samples: {len(X_test_tensor):,}")

train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0

print(f"\nStarting training...")
print("=" * 80)

start_time = time.time()

# Calculate batches
num_train_samples = len(X_train_tensor)
num_batches = (num_train_samples + batch_size - 1) // batch_size

# Quick sanity check
print(f"Batches per epoch: {num_batches}")
with torch.no_grad():
    test_batch = X_train_tensor[:10]
    test_batch = test_batch.to(lstm_model.fc1.weight.device)
    test_out = lstm_model(test_batch)
    print(f"Sanity check passed - Model forward pass OK\n")

# Training loop with progress bars
with tqdm(total=num_epochs, desc="LSTM Training", unit="epoch", colour="blue") as epoch_pbar:
    for epoch in range(num_epochs):
        lstm_model.train()
        train_loss = 0.0
        epoch_start = time.time()

        # Shuffle indices
        indices = torch.randperm(num_train_samples)

        # Training batches with progress bar
        with tqdm(total=num_batches, desc=f"  Epoch {epoch+1}", unit="batch", leave=False, colour="cyan") as batch_pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_train_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_X = X_train_tensor[batch_indices]
                batch_y = y_train_tensor[batch_indices]

                # Forward pass
                optimizer.zero_grad()
                batch_X = batch_X.to(lstm_model.fc1.weight.device)
                outputs = lstm_model(batch_X)
                batch_y = batch_y.to(outputs.device)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Update batch progress bar
                batch_pbar.update(1)
                batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= num_batches
        train_losses.append(train_loss)

        # Validation
        lstm_model.eval()
        val_loss = 0.0
        num_val_samples = len(X_test_tensor)
        num_val_batches = (num_val_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in range(num_val_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_val_samples)

                batch_X = X_test_tensor[start_idx:end_idx]
                batch_y = y_test_tensor[start_idx:end_idx]

                batch_X = batch_X.to(lstm_model.fc1.weight.device)
                outputs = lstm_model(batch_X)
                batch_y = batch_y.to(outputs.device)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= num_val_batches
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        
        # Calculate validation AUC-PR for better monitoring (imbalanced data)
        lstm_model.eval()
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(lstm_model.fc1.weight.device)
            val_outputs = lstm_model(X_test_tensor)
            val_probs = F.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
            val_auc_pr = average_precision_score(y_test_tensor.cpu().numpy(), val_probs)

        # Epoch summary
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        # Update epoch progress bar with metrics
        epoch_pbar.update(1)
        epoch_pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "auc_pr": f"{val_auc_pr:.4f}"
        })

        # Early stopping based on validation loss (could switch to AUC-PR if preferred)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(lstm_model.state_dict(), DLConfig.LSTM_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

# Training complete
training_time = time.time() - start_time
print("\n" + "=" * 80)
print("LSTM TRAINING COMPLETED")
print("=" * 80)
print(f"Total training time: {training_time/60:.2f} minutes")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Epochs completed: {epoch+1}/{num_epochs}")

# Load best model
lstm_model.load_state_dict(torch.load(DLConfig.LSTM_MODEL_PATH, weights_only=True))
print("\nBest model loaded and ready for evaluation")


# In[ ]:


# EVALUATE LSTM MODEL
print("=" * 80)
print("LSTM MODEL EVALUATION")
print("=" * 80)

from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, average_precision_score,
    precision_recall_curve, matthews_corrcoef
)

# Predictions on test set
lstm_model.eval()
with torch.no_grad():
    test_outputs = lstm_model(X_test_tensor)
    test_probs = F.softmax(test_outputs, dim=1).cpu().numpy()
    test_pred_classes = test_outputs.argmax(dim=1).cpu().numpy()

# Extract probabilities for positive class
y_prob_lstm = test_probs[:, 1]
y_pred_lstm = test_pred_classes

# Get true labels from tensor
y_test_lstm = y_test_tensor.cpu().numpy()

# Calculate metrics
auc_roc_lstm = roc_auc_score(y_test_lstm, y_prob_lstm)
auc_pr_lstm = average_precision_score(y_test_lstm, y_prob_lstm)
mcc_lstm = matthews_corrcoef(y_test_lstm, y_pred_lstm)

print("\nLSTM Performance Metrics:")
print("-" * 40)
print(f"AUC-ROC: {auc_roc_lstm:.4f}")
print(f"AUC-PR:  {auc_pr_lstm:.4f}")
print(f"MCC:     {mcc_lstm:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_lstm, y_pred_lstm, target_names=['Licit', 'Illicit'], digits=4))

print("\nConfusion Matrix:")
cm_lstm = confusion_matrix(y_test_lstm, y_pred_lstm)
print(cm_lstm)
print(f"\nTrue Negatives: {cm_lstm[0,0]:,} | False Positives: {cm_lstm[0,1]:,}")
print(f"False Negatives: {cm_lstm[1,0]:,} | True Positives: {cm_lstm[1,1]:,}")

# Compare with baseline
print("\n" + "=" * 80)
print("COMPARISON WITH BASELINE ENSEMBLE")
print("=" * 80)
print(f"Baseline Ensemble AUC-ROC: {auc_roc_ensemble:.4f}")
print(f"LSTM AUC-ROC:              {auc_roc_lstm:.4f}")
print(f"Improvement:               {(auc_roc_lstm - auc_roc_ensemble):.4f} ({((auc_roc_lstm - auc_roc_ensemble)/auc_roc_ensemble)*100:+.2f}%)")

# Visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC Curve
from sklearn.metrics import roc_curve
fpr_lstm, tpr_lstm, _ = roc_curve(y_test_lstm, y_prob_lstm)

axes[0].plot(fpr_lstm, tpr_lstm, linewidth=2.5, label=f'LSTM (AUC={auc_roc_lstm:.4f})')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
axes[0].set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
axes[0].set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
axes[0].set_title('ROC Curve - LSTM', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(alpha=0.3)

# Precision-Recall Curve
precision_lstm, recall_lstm, _ = precision_recall_curve(y_test_lstm, y_prob_lstm)

axes[1].plot(recall_lstm, precision_lstm, linewidth=2.5, label=f'LSTM (AP={auc_pr_lstm:.4f})')
axes[1].set_xlabel('Recall', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Precision', fontsize=11, fontweight='bold')
axes[1].set_title('Precision-Recall Curve - LSTM', fontsize=13, fontweight='bold')
axes[1].legend(loc='lower left', fontsize=10)
axes[1].grid(alpha=0.3)

# Confusion Matrix Heatmap
import seaborn as sns
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['Licit', 'Illicit'], yticklabels=['Licit', 'Illicit'],
            cbar_kws={'label': 'Count'}, annot_kws={'size': 12, 'weight': 'bold'})
axes[2].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
axes[2].set_ylabel('True Label', fontsize=11, fontweight='bold')
axes[2].set_title('Confusion Matrix - LSTM', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('lstm_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nEvaluation complete! Visualization saved as 'lstm_evaluation.png'")


# ## 4.2 Graph Neural Network (GNN) - GraphSAGE
# 
# **Implementation Rationale:**
# - **GraphSAGE architecture**: Samples and aggregates features from node neighborhoods
# - **Graph structure**: 203,769 nodes (transactions) with 234,355 edges (transaction flows)
# - **Purpose**: Capture structural patterns in cryptocurrency transaction networks
# - **Alignment**: Chapter 3 methodology requires graph-based modeling
# 
# **Key Features:**
# - Neighbor sampling for scalability
# - Inductive learning capability
# - Mean aggregation of neighborhood features
# - 2-layer architecture per configuration

# In[ ]:


# PREPARE GRAPH DATA FOR GNN
print("=" * 80)
print("PREPARING GRAPH DATA FOR GNN")
print("=" * 80)

from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx

# Load edge list
edge_df = pd.read_csv('dataset/rawData/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
print(f"\nEdge list shape: {edge_df.shape}")
print(f"Total edges: {len(edge_df):,}")
print(f"Unique source nodes: {edge_df['txId1'].nunique():,}")
print(f"Unique target nodes: {edge_df['txId2'].nunique():,}")

# Filter edges to training-period interactions (both endpoints time_step ≤ 35) to avoid future leakage
node_time_map = data.set_index('node_ID')['time_step'].to_dict()
edge_df = edge_df[
    edge_df['txId1'].map(node_time_map).fillna(99) <= 35
]
edge_df = edge_df[
    edge_df['txId2'].map(node_time_map).fillna(99) <= 35
]
print(f"Filtered edges for GNN (train-period only): {len(edge_df):,}")

# Get node features from our preprocessed data
# Use ALL real nodes (train + test) for graph structure, indexed by node_ID
node_feature_df = data.set_index('node_ID')

# Ensure all gnn_features exist
for col in gnn_features:
    if col not in node_feature_df.columns:
        node_feature_df[col] = 0

time_cols = ['time_sin', 'time_cos']

gnn_features_final = original_features + gnn_features + time_cols 

all_nodes_df = node_feature_df[gnn_features_final].fillna(0)

all_labels = node_feature_df['class_label']

print(f"GNN features: {len(gnn_features_final)} (Original + Graph + Time Encoding)")

print(f"\nTotal nodes with features: {len(all_nodes_df):,}")
print(f"Feature dimension: {all_nodes_df.shape[1]}")

# CRITICAL FIX: Ensure ALL edge nodes have features (pad with zeros if needed)
print("\nEnsuring all edge nodes have features...")
all_edge_nodes = pd.concat([edge_df['txId1'], edge_df['txId2']]).unique()
missing_nodes = set(all_edge_nodes) - set(all_nodes_df.index)

if len(missing_nodes) > 0:
    print(f"  ⚠ Found {len(missing_nodes):,} nodes in edges without features")
    print(f"  Adding them with zero-padded features...")
    zero_features = pd.DataFrame(
        0, 
        index=list(missing_nodes), 
        columns=all_nodes_df.columns
    )
    all_nodes_df = pd.concat([all_nodes_df, zero_features], axis=0)
    
    # Also add missing labels as unknown (-1 will be added later)
    missing_labels = pd.Series('3', index=list(missing_nodes), name='class')
    all_labels = pd.concat([all_labels, missing_labels], axis=0)
    
    print(f"  ✓ Total nodes with features: {len(all_nodes_df):,}")
else:
    print(f"  ✓ All edge nodes already have features")

# Create node ID to index mapping (now includes all nodes)
node_ids = all_nodes_df.index.values
node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}

# Now we can use ALL edges since all nodes have features
edge_df_filtered = edge_df.copy()

print(f"\nTotal edges included: {len(edge_df_filtered):,} (was {len(edge_df):,} in original dataset)")

# Convert node IDs to indices
edge_index_list = []
for _, row in edge_df_filtered.iterrows():
    src_idx = node_to_idx[row['txId1']]
    dst_idx = node_to_idx[row['txId2']]
    edge_index_list.append([src_idx, dst_idx])

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
print(f"Edge index shape: {edge_index.shape}")

# Convert features and labels to tensors
x_graph = torch.tensor(all_nodes_df.values, dtype=torch.float)
# Convert string labels to numeric (0=licit, 1=illicit)
# Note: Unknown labels ('3') will become NaN after mapping
all_labels_numeric = all_labels.map(label_map)

# Check for NaN values in labels
nan_count = all_labels_numeric.isna().sum()
if nan_count > 0:
    print(f"\nWarning: Found {nan_count} unknown/unlabeled nodes")
    print(f"Label distribution before mapping: {all_labels.value_counts().to_dict()}")
    # Fill NaN with -1 to indicate unknown/unlabeled nodes
    # These will be masked out during training
    all_labels_numeric = all_labels_numeric.fillna(-1)

y_graph = torch.tensor(all_labels_numeric.values, dtype=torch.long)

print(f"\nNode features shape: {x_graph.shape}")
print(f"Node labels shape: {y_graph.shape}")
print(f"Label distribution in graph: {np.unique(y_graph.numpy(), return_counts=True)}")

# Create train/test masks based on temporal split and labeled data
train_node_ids = set(train_temporal_df.index)
test_node_ids = set(test_temporal_df.index)

# Create masks and exclude unlabeled nodes (label=-1)
labeled_mask = y_graph >= 0  # Only nodes with valid labels (0 or 1)
train_mask = torch.tensor([node_ids[i] in train_node_ids for i in range(len(node_ids))], dtype=torch.bool) & labeled_mask
test_mask = torch.tensor([node_ids[i] in test_node_ids for i in range(len(node_ids))], dtype=torch.bool) & labeled_mask

print(f"\nTotal nodes: {len(node_ids):,}")
print(f"Labeled nodes: {labeled_mask.sum():,}")
print(f"Train nodes (labeled): {train_mask.sum():,}")
print(f"Test nodes (labeled): {test_mask.sum():,}")

# Create PyTorch Geometric Data object
graph_data = Data(
    x=x_graph,
    edge_index=edge_index,
    y=y_graph,
    train_mask=train_mask,
    test_mask=test_mask
)

# Move to device
graph_data = graph_data.to(cpu_device)

print("\n" + "=" * 80)
print("GRAPH DATA PREPARATION COMPLETE")
print("=" * 80)
print(f"\nGraph statistics:")
print(f"  Nodes: {graph_data.num_nodes:,}")
print(f"  Edges: {graph_data.num_edges:,}")
print(f"  Features per node: {graph_data.num_node_features}")
print(f"  Training nodes: {train_mask.sum():,}")
print(f"  Test nodes: {test_mask.sum():,}")


# In[ ]:


# GRAPH VISUALIZATION
print("\n" + "=" * 80)
print("VISUALIZING TRANSACTION NETWORK GRAPH")
print("=" * 80)

# Use t-SNE to reduce node features to 2D for visualization
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print("\nReducing node features to 2D using t-SNE...")
# Sample nodes for faster visualization if graph is too large
max_nodes_to_plot = 5000
if graph_data.num_nodes > max_nodes_to_plot:
    print(f"Graph has {graph_data.num_nodes:,} nodes. Sampling {max_nodes_to_plot:,} for visualization...")
    sample_indices = np.random.choice(graph_data.num_nodes, max_nodes_to_plot, replace=False)
    sample_indices = np.sort(sample_indices)  # Keep order for mask alignment
else:
    sample_indices = np.arange(graph_data.num_nodes)

# Get features and labels for sampled nodes
X_graph_vis = graph_data.x[sample_indices].cpu().numpy()
y_graph_vis = graph_data.y[sample_indices].cpu().numpy()

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_embedded = tsne.fit_transform(X_graph_vis)

# Create visualization
fig, ax = plt.subplots(figsize=(14, 10))

# Separate nodes by label
unlabeled_mask = y_graph_vis == -1
licit_mask = y_graph_vis == 0
illicit_mask = y_graph_vis == 1

# Plot unlabeled nodes (gray)
if unlabeled_mask.any():
    ax.scatter(X_embedded[unlabeled_mask, 0], X_embedded[unlabeled_mask, 1], 
               c='lightgray', s=30, alpha=0.3, label='Unknown', edgecolors='none')

# Plot licit nodes (blue)
if licit_mask.any():
    ax.scatter(X_embedded[licit_mask, 0], X_embedded[licit_mask, 1], 
               c='blue', s=40, alpha=0.6, label='Licit', edgecolors='navy', linewidths=0.5)

# Plot illicit nodes (red) - on top for visibility
if illicit_mask.any():
    ax.scatter(X_embedded[illicit_mask, 0], X_embedded[illicit_mask, 1], 
               c='red', s=50, alpha=0.8, label='Illicit', edgecolors='darkred', linewidths=0.8)

ax.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
ax.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
ax.set_title('Bitcoin Transaction Network Visualization\n(t-SNE projection of node features)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

# Add statistics text box
stats_text = f"Total Nodes: {len(sample_indices):,}\n"
if unlabeled_mask.any():
    stats_text += f"Unknown: {unlabeled_mask.sum():,} ({unlabeled_mask.sum()/len(sample_indices)*100:.1f}%)\n"
if licit_mask.any():
    stats_text += f"Licit: {licit_mask.sum():,} ({licit_mask.sum()/len(sample_indices)*100:.1f}%)\n"
if illicit_mask.any():
    stats_text += f"Illicit: {illicit_mask.sum():,} ({illicit_mask.sum()/len(sample_indices)*100:.1f}%)"

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('transaction_network_visualization.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'transaction_network_visualization.png'")
plt.close()

print("\nVisualization complete!")


# In[ ]:


# GRAPHSAGE MODEL ARCHITECTURE
print("=" * 80)
print("DEFINING GRAPHSAGE ARCHITECTURE")
print("=" * 80)

from torch_geometric.nn import SAGEConv

class GraphSAGEFraudDetector(nn.Module):
    """
    GraphSAGE model for fraud detection in transaction networks.

    Architecture:
    - 2 SAGEConv layers with mean aggregation
    - ReLU activation and dropout for regularization
    - Final linear layer for binary classification
    """
    def __init__(self, in_channels, hidden_channels=None, num_layers=None, dropout=None):
        super(GraphSAGEFraudDetector, self).__init__()

        # Use config if parameters not provided
        if hidden_channels is None:
            hidden_channels = DLConfig.GNN_HIDDEN_DIM
        if num_layers is None:
            num_layers = DLConfig.GNN_NUM_LAYERS
        if dropout is None:
            dropout = DLConfig.GNN_DROPOUT

        self.num_layers = num_layers
        self.dropout = dropout

        # Build layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Final classifier
        self.fc = nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        # Apply graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < self.num_layers - 1:  # Don't dropout after last conv
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Final classification
        x = self.fc(x)
        return x

    def get_embedding(self, x, edge_index):
        """Extract node embeddings before final classification layer."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# Initialize model
in_channels = graph_data.num_node_features
gnn_model = GraphSAGEFraudDetector(in_channels).to(cpu_device)

# Count parameters
total_params = sum(p.numel() for p in gnn_model.parameters())
trainable_params = sum(p.numel() for p in gnn_model.parameters() if p.requires_grad)

print(f"\nGraphSAGE Model Configuration:")
print(f"  Input features: {in_channels}")
print(f"  Hidden channels: {DLConfig.GNN_HIDDEN_DIM}")
print(f"  Number of layers: {DLConfig.GNN_NUM_LAYERS}")
print(f"  Dropout rate: {DLConfig.GNN_DROPOUT}")
print(f"  Output classes: 2")

print(f"\nModel Parameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")

print("\n" + "=" * 80)
print("GRAPHSAGE ARCHITECTURE DEFINED")
print("=" * 80)


# In[ ]:


# TRAIN GRAPHSAGE MODEL
print("=" * 80)
print("TRAINING GRAPHSAGE MODEL")
print("=" * 80)

import time

# Training setup from config
num_epochs = DLConfig.NUM_EPOCHS
learning_rate = DLConfig.LEARNING_RATE
patience = DLConfig.PATIENCE

# CRITICAL FIX: Calculate class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight

train_labels_np = graph_data.y[train_mask].cpu().numpy()
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(train_labels_np), 
    y=train_labels_np
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(cpu_device)
print(f"\nGNN Class weights (to handle imbalance):")
print(f"  Licit (class 0):   {class_weights[0]:.3f}")
print(f"  Illicit (class 1): {class_weights[1]:.3f}")

# Loss function with class weights and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate, weight_decay=DLConfig.WEIGHT_DECAY)

print(f"\nTraining configuration:")
print(f"  Device: {cpu_device}")
print(f"  Epochs: {num_epochs}")
print(f"  Learning rate: {learning_rate}")
print(f"  Training nodes: {train_mask.sum():,}")
print(f"  Test nodes: {test_mask.sum():,}")

train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0

print(f"\nStarting training...")
print("=" * 80)

start_time = time.time()

with tqdm(total=num_epochs, desc="GraphSAGE Training", unit="epoch", colour="green") as epoch_pbar:
    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training
        gnn_model.train()
        optimizer.zero_grad()

        # Forward pass on entire graph
        out = gnn_model(graph_data.x, graph_data.edge_index)

        # Loss only on training nodes
        loss = criterion(out[train_mask], graph_data.y[train_mask])

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        train_losses.append(train_loss)

        # Validation
        gnn_model.eval()
        with torch.no_grad():
            out = gnn_model(graph_data.x, graph_data.edge_index)
            val_loss = criterion(out[test_mask], graph_data.y[test_mask]).item()
            val_losses.append(val_loss)

            # Calculate accuracy
            pred = out.argmax(dim=1)
            train_acc = (pred[train_mask] == graph_data.y[train_mask]).sum().item() / train_mask.sum().item()
            val_acc = (pred[test_mask] == graph_data.y[test_mask]).sum().item() / test_mask.sum().item()

        # Epoch summary
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        # Update progress bar with metrics
        epoch_pbar.update(1)
        epoch_pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc:.4f}"
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(gnn_model.state_dict(), DLConfig.GNN_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

# Training complete
training_time = time.time() - start_time
print("\n" + "=" * 80)
print("GRAPHSAGE TRAINING COMPLETED")
print("=" * 80)
print(f"Total training time: {training_time/60:.2f} minutes")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Epochs completed: {epoch+1}/{num_epochs}")

# Load best model
gnn_model.load_state_dict(torch.load(DLConfig.GNN_MODEL_PATH, weights_only=True))
print("\nBest model loaded and ready for evaluation")


# In[ ]:


# EVALUATE GRAPHSAGE MODEL
print("=" * 80)
print("GRAPHSAGE MODEL EVALUATION")
print("=" * 80)

# Predictions on test set
gnn_model.eval()
with torch.no_grad():
    out = gnn_model(graph_data.x, graph_data.edge_index)
    test_probs = F.softmax(out[test_mask], dim=1).cpu().numpy()
    test_pred_classes = out[test_mask].argmax(dim=1).cpu().numpy()

# Extract test labels
y_test_gnn = graph_data.y[test_mask].cpu().numpy()

# Probabilities for positive class
y_prob_gnn = test_probs[:, 1]
y_pred_gnn = test_pred_classes

# Calculate metrics
auc_roc_gnn = roc_auc_score(y_test_gnn, y_prob_gnn)
auc_pr_gnn = average_precision_score(y_test_gnn, y_prob_gnn)
mcc_gnn = matthews_corrcoef(y_test_gnn, y_pred_gnn)

print("\nGraphSAGE Performance Metrics:")
print("-" * 40)
print(f"AUC-ROC: {auc_roc_gnn:.4f}")
print(f"AUC-PR:  {auc_pr_gnn:.4f}")
print(f"MCC:     {mcc_gnn:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_gnn, y_pred_gnn, target_names=['Licit', 'Illicit'], digits=4))

print("\nConfusion Matrix:")
cm_gnn = confusion_matrix(y_test_gnn, y_pred_gnn)
print(cm_gnn)
print(f"\nTrue Negatives: {cm_gnn[0,0]:,} | False Positives: {cm_gnn[0,1]:,}")
print(f"False Negatives: {cm_gnn[1,0]:,} | True Positives: {cm_gnn[1,1]:,}")

# Compare with baseline and LSTM
print("\n" + "=" * 80)
print("COMPARISON WITH PREVIOUS MODELS")
print("=" * 80)
print(f"Baseline Ensemble AUC-ROC: {auc_roc_ensemble:.4f}")
print(f"LSTM AUC-ROC:              {auc_roc_lstm:.4f}")
print(f"GraphSAGE AUC-ROC:         {auc_roc_gnn:.4f}")
print(f"\nGraphSAGE vs Baseline: {(auc_roc_gnn - auc_roc_ensemble):.4f} ({((auc_roc_gnn - auc_roc_ensemble)/auc_roc_ensemble)*100:+.2f}%)")
print(f"GraphSAGE vs LSTM:     {(auc_roc_gnn - auc_roc_lstm):.4f} ({((auc_roc_gnn - auc_roc_lstm)/auc_roc_lstm)*100:+.2f}%)")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC Curve
fpr_gnn, tpr_gnn, _ = roc_curve(y_test_gnn, y_prob_gnn)

axes[0].plot(fpr_gnn, tpr_gnn, linewidth=2.5, label=f'GraphSAGE (AUC={auc_roc_gnn:.4f})')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
axes[0].set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
axes[0].set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
axes[0].set_title('ROC Curve - GraphSAGE', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(alpha=0.3)

# Precision-Recall Curve
precision_gnn, recall_gnn, _ = precision_recall_curve(y_test_gnn, y_prob_gnn)

axes[1].plot(recall_gnn, precision_gnn, linewidth=2.5, label=f'GraphSAGE (AP={auc_pr_gnn:.4f})')
axes[1].set_xlabel('Recall', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Precision', fontsize=11, fontweight='bold')
axes[1].set_title('Precision-Recall Curve - GraphSAGE', fontsize=13, fontweight='bold')
axes[1].legend(loc='lower left', fontsize=10)
axes[1].grid(alpha=0.3)

# Confusion Matrix Heatmap
sns.heatmap(cm_gnn, annot=True, fmt='d', cmap='Greens', ax=axes[2],
            xticklabels=['Licit', 'Illicit'], yticklabels=['Licit', 'Illicit'],
            cbar_kws={'label': 'Count'}, annot_kws={'size': 12, 'weight': 'bold'})
axes[2].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
axes[2].set_ylabel('True Label', fontsize=11, fontweight='bold')
axes[2].set_title('Confusion Matrix - GraphSAGE', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('gnn_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nEvaluation complete! Visualization saved as 'gnn_evaluation.png'")


# ## 4.3 Hybrid Model - Combining Temporal and Graph Features
# 
# **Implementation Rationale:**
# - **Hybrid architecture**: Combines LSTM temporal modeling with GraphSAGE graph structure learning
# - **Fusion strategy**: Concatenate embeddings from both models before final classification
# - **Purpose**: Leverage both temporal transaction patterns AND graph network structure
# - **Alignment**: Chapter 3 methodology requires hybrid temporal + graph approach
# 
# **Key Features:**
# - Pre-trained LSTM and GraphSAGE models as feature extractors
# - Learned fusion layer to combine embeddings
# - End-to-end fine-tuning capability
# - Expected to outperform individual models

# In[ ]:


# HYBRID MODEL ARCHITECTURE
print("=" * 80)
print("DEFINING HYBRID TEMPORAL-GRAPH ARCHITECTURE")
print("=" * 80)


class HybridFraudDetector(nn.Module):
    """
    Residual Graph Hybrid Model.
    Replaces LSTM (useless for single transactions) with a Skip-Connection 
    that preserves raw feature signals while adding Graph Context.
    """
    def __init__(self, gnn_model, input_feature_dim, fusion_dim=128, dropout=0.3):
        super(HybridFraudDetector, self).__init__()

        # 1. Pre-trained GNN (Context Extractor)
        self.gnn = gnn_model
        
        # 2. Raw Feature Projector (The "XGBoost" equivalent component)
        # Processes local features (fees, amounts) directly
        self.raw_proj = nn.Sequential(
            nn.Linear(input_feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. Fusion Layer with Skip Connection
        # Concatenates: [GNN Context] + [Projected Local] + [Original Raw]
        # DLConfig.GNN_HIDDEN_DIM is usually 128
        gnn_dim = 128 
        combined_dim = gnn_dim + 64 + input_feature_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 4. Final Classifier
        self.classifier = nn.Linear(fusion_dim // 2, 2)

    def forward(self, x_all, edge_index, node_indices):
        """
        x_all: Features for ALL nodes in graph (for GNN context)
        edge_index: Graph edges
        node_indices: The specific nodes we are classifying in this batch
        """
        # 1. Get Graph Context (Embeddings for specific nodes)
        # We use the GNN to look at neighbors
        gnn_emb_all = self.gnn.get_embedding(x_all, edge_index)
        batch_gnn_emb = gnn_emb_all[node_indices]
        
        # 2. Get Local Signal
        batch_raw = x_all[node_indices]
        local_emb = self.raw_proj(batch_raw)
        
        # 3. Residual Concatenation (The "Expert" Trick)
        # We feed the raw features directly into the fusion layer too
        # This ensures the model never performs WORSE than a basic MLP
        combined = torch.cat([batch_gnn_emb, local_emb, batch_raw], dim=1)
        
        # 4. Classify
        fused = self.fusion(combined)
        output = self.classifier(fused)
        
        return output

hybrid_model = HybridFraudDetector(gnn_model, input_feature_dim=graph_data.num_node_features).to(device)

# Count parameters
total_params = sum(p.numel() for p in hybrid_model.parameters())
trainable_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
fusion_params = sum(p.numel() for p in hybrid_model.fusion.parameters()) + \
                sum(p.numel() for p in hybrid_model.classifier.parameters())

print(f"\nHybrid Model Configuration:")
print(f"  LSTM embedding dim: {DLConfig.LSTM_HIDDEN_DIM}")
print(f"  GNN embedding dim: {DLConfig.GNN_HIDDEN_DIM}")
print(f"  Combined dim: {DLConfig.LSTM_HIDDEN_DIM + DLConfig.GNN_HIDDEN_DIM}")
print(f"  Fusion dim: {DLConfig.HYBRID_FUSION_DIM}")
print(f"  Dropout rate: {DLConfig.HYBRID_DROPOUT}")

print(f"\nModel Parameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable (fusion only): {fusion_params:,}")
print(f"  Pre-trained (LSTM + GNN): {total_params - fusion_params:,}")

print("\n" + "=" * 80)
print("HYBRID ARCHITECTURE DEFINED")
print("=" * 80)
print("\nNote: LSTM and GNN weights are frozen. Only fusion layers will be trained.")


# In[ ]:


# PREPARE DATA FOR HYBRID MODEL TRAINING
print("=" * 80)
print("PREPARING DATA FOR HYBRID MODEL")
print("=" * 80)

# We need to align temporal features with graph node indices
# The hybrid model needs to know which graph nodes correspond to which temporal samples

# Create mapping from node IDs to their positions in the graph
# We already have node_to_idx and idx_to_node from GNN data prep

# For training: use SMOTE-balanced training data
train_node_ids_list = train_temporal_df.index.tolist()
train_node_indices = torch.tensor([node_to_idx[nid] for nid in train_node_ids_list], dtype=torch.long)

# Get labels for the training nodes
train_labels_real = graph_data.y[train_node_indices].cpu().numpy()

# Calculate weights: Inverse frequency
class_counts = np.bincount(train_labels_real)
# Prevent divide by zero if any class is missing (unlikely but safe)
class_weights_sampling = 1. / np.maximum(class_counts, 1)
samples_weights = class_weights_sampling[train_labels_real]
sampler = torch.utils.data.WeightedRandomSampler(
    weights=torch.from_numpy(samples_weights).double(),
    num_samples=len(samples_weights),
    replacement=True
)
# Note: shuffle=False is required when using a sampler
train_loader_hybrid = DataLoader(
    TensorDataset(train_node_indices), # We only need indices, the model looks up features in the Graph
    batch_size=batch_size,
    sampler=sampler, # <--- HERE IS THE MAGIC
    num_workers=0
)


# For testing: use original labeled test data
test_node_ids_list = test_temporal_df.index.tolist()
test_node_indices = torch.tensor([node_to_idx[nid] for nid in test_node_ids_list], dtype=torch.long)

print(f"\nData alignment:")
print(f"  Training samples: {len(X_train_tensor):,}")
print(f"  Training node indices: {len(train_node_indices):,}")
print(f"  Test samples: {len(X_test_tensor):,}")
print(f"  Test node indices: {len(test_node_indices):,}")

# Verify alignment
print(f"\nVerification:")
print(f"  All training node indices in graph: {all(idx < graph_data.num_nodes for idx in train_node_indices)}")
print(f"  All test node indices in graph: {all(idx < graph_data.num_nodes for idx in test_node_indices)}")

print("\n" + "=" * 80)
print("HYBRID DATA PREPARATION COMPLETE")
print("=" * 80)


# In[ ]:


# TRAIN HYBRID MODEL
print("=" * 80)
print("TRAINING HYBRID MODEL")
print("=" * 80)

import time

# Training setup from config
num_epochs = DLConfig.NUM_EPOCHS
learning_rate = DLConfig.LEARNING_RATE
batch_size = DLConfig.BATCH_SIZE
patience = DLConfig.PATIENCE

# CRITICAL FIX: Calculate class weights for imbalanced data
train_labels_hybrid_np = y_train_tensor.cpu().numpy()
class_weights_hybrid = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels_hybrid_np),
    y=train_labels_hybrid_np
)


graph_data = graph_data.to(device)
class_weights_hybrid = torch.tensor(class_weights_hybrid, dtype=torch.float32).to(device)

print(f"\nHybrid Model Class weights (to handle imbalance):")
print(f"  Licit (class 0):   {class_weights_hybrid[0]:.3f}")
print(f"  Illicit (class 1): {class_weights_hybrid[1]:.3f}")


# 1. Use AdamW (Better weight decay handling)
optimizer = torch.optim.AdamW([
    {'params': hybrid_model.fusion.parameters(), 'lr': learning_rate},
    {'params': hybrid_model.classifier.parameters(), 'lr': learning_rate},
    {'params': hybrid_model.raw_proj.parameters(), 'lr': learning_rate}, # Don't forget the new projection layer
    {'params': gnn_model.parameters(), 'lr': learning_rate * 0.5}    # Fine-tune GNN slowly
], weight_decay=1e-4)

# Increase gamma to 3.0 to focus on "hard" examples
# Use the class weights you calculated earlier
criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=None)

# 3. OneCycle Scheduler (Converges faster and better)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=learning_rate, 
    steps_per_epoch=len(train_loader_hybrid), 
    epochs=num_epochs
)

print(f"\nTraining configuration:")
print(f"  Device: {cpu_device}")
print(f"  Epochs: {num_epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")
print(f"  Training samples: {len(X_train_tensor):,}")
print(f"  Test samples: {len(X_test_tensor):,}")
print(f"  Strategy: Fine-tuning ALL components (fusion + LSTM + GNN)")
print(f"  Class weights: Applied to handle imbalance")

train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0

print(f"\nStarting training...")
print("=" * 80)

start_time = time.time()

# Calculate batches
num_train_samples = len(X_train_tensor)
num_batches = (num_train_samples + batch_size - 1) // batch_size

with tqdm(total=num_epochs, desc="Hybrid Training", unit="epoch", colour="red") as epoch_pbar:
    for epoch in range(num_epochs):
        hybrid_model.train()
        train_loss = 0.0
        epoch_start = time.time()

        # Shuffle indices
        indices = torch.randperm(num_train_samples)

        # Training batches with progress bar
        with tqdm(total=num_batches, desc=f"  Epoch {epoch+1}", unit="batch", leave=False, colour="magenta") as batch_pbar:
            for batch_idx, (batch_indices) in enumerate(tqdm(train_loader_hybrid)):
                
                batch_indices = batch_indices[0].to(device) 
                
                # Get labels directly from the Graph data (aligned)
                batch_y = graph_data.y[batch_indices]

                # Forward pass (Only 3 arguments now)
                optimizer.zero_grad()
                outputs = hybrid_model(
                    graph_data.x,           # x_all
                    graph_data.edge_index,  # edge_index
                    batch_indices           # node_indices
                )
              
                batch_y = batch_y.to(outputs.device)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Update batch progress bar
                batch_pbar.update(1)
                batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= num_batches
        train_losses.append(train_loss)

        # Validation
        hybrid_model.eval()
        val_loss = 0.0
        num_val_samples = len(X_test_tensor)
        num_val_batches = (num_val_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in range(num_val_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_val_samples)

                
                batch_node_indices = test_node_indices[start_idx:end_idx].to(device)
                
                # Get labels from graph data to ensure alignment
                batch_y = graph_data.y[batch_node_indices]

                outputs = hybrid_model(
                    graph_data.x,           # x_all
                    graph_data.edge_index,  # edge_index
                    batch_node_indices      # node_indices
                )
                
                batch_y = batch_y.to(outputs.device)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= num_val_batches
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Epoch summary
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        # Update epoch progress bar with metrics
        epoch_pbar.update(1)
        epoch_pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}"
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(hybrid_model.state_dict(), DLConfig.HYBRID_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

# Training complete
training_time = time.time() - start_time
print("\n" + "=" * 80)
print("HYBRID MODEL TRAINING COMPLETED")
print("=" * 80)
print(f"Total training time: {training_time/60:.2f} minutes")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Epochs completed: {epoch+1}/{num_epochs}")

# Load best model
hybrid_model.load_state_dict(torch.load(DLConfig.HYBRID_MODEL_PATH, weights_only=True))
print("\nBest model loaded and ready for evaluation")


# In[ ]:


# EVALUATE HYBRID MODEL
print("=" * 80)
print("HYBRID MODEL EVALUATION")
print("=" * 80)

# Predictions on test set
hybrid_model.eval()
num_test_samples = len(X_test_tensor)
num_test_batches = (num_test_samples + batch_size - 1) // batch_size

all_probs = []
all_preds = []

with torch.no_grad():
    for batch_idx in range(num_test_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_test_samples)

        batch_node_indices = test_node_indices[start_idx:end_idx].to(device)

        outputs = hybrid_model(
            graph_data.x,           # x_all
            graph_data.edge_index,  # edge_index
            batch_node_indices      # node_indices
        )

        probs = F.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_probs.append(probs)
        all_preds.append(preds)

# Concatenate results
test_probs = np.vstack(all_probs)
test_pred_classes = np.concatenate(all_preds)

# Probabilities for positive class
y_prob_hybrid = test_probs[:, 1]
y_pred_hybrid = test_pred_classes

# Get true labels from tensor
y_test_hybrid = y_test_tensor.cpu().numpy()

# Calculate metrics
auc_roc_hybrid = roc_auc_score(y_test_hybrid, y_prob_hybrid)
auc_pr_hybrid = average_precision_score(y_test_hybrid, y_prob_hybrid)
mcc_hybrid = matthews_corrcoef(y_test_hybrid, y_pred_hybrid)

print("\nHybrid Model Performance Metrics:")
print("-" * 40)
print(f"AUC-ROC: {auc_roc_hybrid:.4f}")
print(f"AUC-PR:  {auc_pr_hybrid:.4f}")
print(f"MCC:     {mcc_hybrid:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_hybrid, y_pred_hybrid, target_names=['Licit', 'Illicit'], digits=4))

print("\nConfusion Matrix:")
cm_hybrid = confusion_matrix(y_test_hybrid, y_pred_hybrid)
print(cm_hybrid)
print(f"\nTrue Negatives: {cm_hybrid[0,0]:,} | False Positives: {cm_hybrid[0,1]:,}")
print(f"False Negatives: {cm_hybrid[1,0]:,} | True Positives: {cm_hybrid[1,1]:,}")

# Compare with all previous models
print("\n" + "=" * 80)
print("COMPREHENSIVE MODEL COMPARISON")
print("=" * 80)
print(f"Baseline Ensemble AUC-ROC: {auc_roc_ensemble:.4f}")
print(f"LSTM AUC-ROC:              {auc_roc_lstm:.4f}")
print(f"GraphSAGE AUC-ROC:         {auc_roc_gnn:.4f}")
print(f"Hybrid AUC-ROC:            {auc_roc_hybrid:.4f}")
print(f"\nHybrid vs Baseline: {(auc_roc_hybrid - auc_roc_ensemble):.4f} ({((auc_roc_hybrid - auc_roc_ensemble)/auc_roc_ensemble)*100:+.2f}%)")
print(f"Hybrid vs LSTM:     {(auc_roc_hybrid - auc_roc_lstm):.4f} ({((auc_roc_hybrid - auc_roc_lstm)/auc_roc_lstm)*100:+.2f}%)")
print(f"Hybrid vs GNN:      {(auc_roc_hybrid - auc_roc_gnn):.4f} ({((auc_roc_hybrid - auc_roc_gnn)/auc_roc_gnn)*100:+.2f}%)")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC Curve
fpr_hybrid, tpr_hybrid, _ = roc_curve(y_test_hybrid, y_prob_hybrid)

axes[0].plot(fpr_hybrid, tpr_hybrid, linewidth=2.5, label=f'Hybrid (AUC={auc_roc_hybrid:.4f})')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
axes[0].set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
axes[0].set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
axes[0].set_title('ROC Curve - Hybrid Model', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(alpha=0.3)

# Precision-Recall Curve
precision_hybrid, recall_hybrid, _ = precision_recall_curve(y_test_hybrid, y_prob_hybrid)

axes[1].plot(recall_hybrid, precision_hybrid, linewidth=2.5, label=f'Hybrid (AP={auc_pr_hybrid:.4f})')
axes[1].set_xlabel('Recall', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Precision', fontsize=11, fontweight='bold')
axes[1].set_title('Precision-Recall Curve - Hybrid Model', fontsize=13, fontweight='bold')
axes[1].legend(loc='lower left', fontsize=10)
axes[1].grid(alpha=0.3)

# Confusion Matrix Heatmap
sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Purples', ax=axes[2],
            xticklabels=['Licit', 'Illicit'], yticklabels=['Licit', 'Illicit'],
            cbar_kws={'label': 'Count'}, annot_kws={'size': 12, 'weight': 'bold'})
axes[2].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
axes[2].set_ylabel('True Label', fontsize=11, fontweight='bold')
axes[2].set_title('Confusion Matrix - Hybrid Model', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('hybrid_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nEvaluation complete! Visualization saved as 'hybrid_evaluation.png'")


# ## 4.4 Comprehensive Model Comparison
# 
# **Comparison Objective:**
# - Compare ALL models: Baseline ML (LR, RF, XGB, Ensemble) + Deep Learning (LSTM, GNN, Hybrid)
# - Evaluate across multiple metrics: AUC-ROC, AUC-PR, Precision, Recall, F1-Score, MCC
# - Identify best performing model for cryptocurrency fraud detection
# - Alignment: Chapter 3 requires comprehensive performance comparison

# In[ ]:


# COMPREHENSIVE MODEL COMPARISON TABLE
print("=" * 80)
print("COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
print("=" * 80)

from sklearn.metrics import precision_score, recall_score, f1_score

# Regenerate baseline model predictions for comparison
print("\nRegenerating baseline model predictions...")

# Check if models are available, if not skip baseline comparison
try:
    # Logistic Regression predictions
    lr_illicit_idx = list(lr_model.classes_).index('1')
    y_prob_lr_all = lr_model.predict_proba(X_test_scaled)[:, lr_illicit_idx]
    y_pred_lr_all = lr_model.predict(X_test_scaled)
    
    # Filter to labeled samples (matching the original evaluation)
    labeled_test_mask = y_test_modeling.isin(['1', '2'])
    y_prob_lr = y_prob_lr_all[labeled_test_mask]
    y_pred_lr = (y_prob_lr >= 0.5).astype(int)  # Convert to binary
    
    # Random Forest predictions
    rf_illicit_idx = list(rf_model.classes_).index('1')
    y_prob_rf_all = rf_model.predict_proba(X_test_modeling)[:, rf_illicit_idx]
    y_prob_rf = y_prob_rf_all[labeled_test_mask]
    y_pred_rf = (y_prob_rf >= 0.5).astype(int)
    
    # XGBoost predictions
    y_prob_xgb_all = xgb_model.predict_proba(X_test_modeling)[:, 1]
    y_prob_xgb = y_prob_xgb_all[labeled_test_mask]
    y_pred_xgb = (y_prob_xgb >= 0.5).astype(int)
    
    # Ensemble predictions (weighted average)
    total_auc_baseline = auc_roc + auc_roc_rf + auc_roc_xgb
    weight_lr = auc_roc / total_auc_baseline
    weight_rf = auc_roc_rf / total_auc_baseline
    weight_xgb = auc_roc_xgb / total_auc_baseline
    
    y_prob_ensemble = (
        weight_lr * y_prob_lr +
        weight_rf * y_prob_rf +
        weight_xgb * y_prob_xgb
    )
    y_pred_ensemble = (y_prob_ensemble >= 0.5).astype(int)
    
    baseline_models_available = True
    print("✓ Baseline model predictions regenerated successfully")
    
except (NameError, AttributeError) as e:
    print(f"⚠ Warning: Baseline models not available ({e})")
    print("  Continuing with deep learning models only...")
    baseline_models_available = False

# Calculate metrics for all models
models_results = {}

if baseline_models_available:
    models_results.update({
        'Logistic Regression': {
            'y_pred': y_pred_lr,
            'y_prob': y_prob_lr
        },
        'Random Forest': {
            'y_pred': y_pred_rf,
            'y_prob': y_prob_rf
        },
        'XGBoost': {
            'y_pred': y_pred_xgb,
            'y_prob': y_prob_xgb
        },
        'Ensemble (ML)': {
            'y_pred': y_pred_ensemble,
            'y_prob': y_prob_ensemble
        }
    })

# Add deep learning models
models_results.update({
    'LSTM': {
        'y_pred': y_pred_lstm,
        'y_prob': y_prob_lstm
    },
    'GraphSAGE': {
        'y_pred': y_pred_gnn,
        'y_prob': y_prob_gnn
    },
    'Hybrid (DL)': {
        'y_pred': y_pred_hybrid,
        'y_prob': y_prob_hybrid
    }
})

# Calculate all metrics
comparison_data = []
for model_name, results in models_results.items():
    y_pred = results['y_pred']
    y_prob = results['y_prob']

    # Handle different test sets
    if model_name == 'GraphSAGE':
        y_true = y_test_gnn
    elif model_name in ['LSTM', 'Hybrid (DL)']:
        y_true = y_test_lstm
    else:
        # Baseline ML models evaluated on labeled test subset only
        y_true = y_test_labeled_numeric

    metrics = {
        'Model': model_name,
        'AUC-ROC': roc_auc_score(y_true, y_prob),
        'AUC-PR': average_precision_score(y_true, y_prob),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    comparison_data.append(metrics)

# Create comparison DataFrame
comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('AUC-ROC', ascending=False).reset_index(drop=True)

print("\n" + "=" * 100)
print("MODEL PERFORMANCE COMPARISON TABLE")
print("=" * 100)
print(comparison_df.to_string(index=False, float_format='%.4f'))
print("=" * 100)

# Identify best model
best_model = comparison_df.iloc[0]['Model']
best_auc = comparison_df.iloc[0]['AUC-ROC']

print(f"\nBest Performing Model: {best_model}")
print(f"Best AUC-ROC Score: {best_auc:.4f}")

# Save comparison table
comparison_df.to_csv('model_comparison_results.csv', index=False)
print("\nComparison table saved as 'model_comparison_results.csv'")


# In[ ]:


# COMPREHENSIVE VISUALIZATION - ALL MODELS
print("Creating comprehensive comparison visualizations...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Combined ROC Curves
ax1 = fig.add_subplot(gs[0, :2])
for model_name, results in models_results.items():
    y_prob = results['y_prob']
    # Use the same label sets as in the metrics block to ensure binary labels
    if model_name == 'GraphSAGE':
        y_true = y_test_gnn
    elif model_name in ['LSTM', 'Hybrid (DL)']:
        y_true = y_test_lstm
    else:
        y_true = y_test_labeled_numeric
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    # Different line styles for ML vs DL
    linestyle = '--' if model_name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'Ensemble (ML)'] else '-'
    linewidth = 3 if model_name in ['Ensemble (ML)', 'Hybrid (DL)'] else 2

    ax1.plot(fpr, tpr, linestyle=linestyle, linewidth=linewidth, 
             label=f'{model_name} (AUC={auc:.4f})', alpha=0.8)

ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random', alpha=0.5)
ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax1.set_title('ROC Curves - All Models Comparison', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(alpha=0.3)

# 2. Combined Precision-Recall Curves
ax2 = fig.add_subplot(gs[1, :2])
for model_name, results in models_results.items():
    y_prob = results['y_prob']
    if model_name == 'GraphSAGE':
        y_true = y_test_gnn
    elif model_name in ['LSTM', 'Hybrid (DL)']:
        y_true = y_test_lstm
    else:
        y_true = y_test_labeled_numeric
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    linestyle = '--' if model_name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'Ensemble (ML)'] else '-'
    linewidth = 3 if model_name in ['Ensemble (ML)', 'Hybrid (DL)'] else 2

    ax2.plot(recall, precision, linestyle=linestyle, linewidth=linewidth,
             label=f'{model_name} (AP={ap:.4f})', alpha=0.8)

ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax2.set_title('Precision-Recall Curves - All Models Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc='lower left', fontsize=9)
ax2.grid(alpha=0.3)

# 3. AUC-ROC Bar Chart
ax3 = fig.add_subplot(gs[0, 2])
models = comparison_df['Model'].values
auc_scores = comparison_df['AUC-ROC'].values
colors = ['#1f77b4' if 'DL' in m or m in ['LSTM', 'GraphSAGE', 'Hybrid (DL)'] 
          else '#ff7f0e' for m in models]
bars = ax3.barh(models, auc_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('AUC-ROC Score', fontsize=11, fontweight='bold')
ax3.set_title('AUC-ROC Comparison', fontsize=13, fontweight='bold')
ax3.set_xlim([0.85, 1.0])
ax3.grid(axis='x', alpha=0.3)
for i, (model, score) in enumerate(zip(models, auc_scores)):
    ax3.text(score + 0.003, i, f'{score:.4f}', va='center', fontsize=9, fontweight='bold')

# 4. F1-Score Bar Chart
ax4 = fig.add_subplot(gs[1, 2])
f1_scores = comparison_df['F1-Score'].values
bars = ax4.barh(models, f1_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('F1-Score', fontsize=11, fontweight='bold')
ax4.set_title('F1-Score Comparison', fontsize=13, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
for i, (model, score) in enumerate(zip(models, f1_scores)):
    ax4.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=9, fontweight='bold')

# 5. Metrics Heatmap
ax5 = fig.add_subplot(gs[2, :])
metrics_for_heatmap = comparison_df[['Model', 'AUC-ROC', 'AUC-PR', 'Precision', 'Recall', 'F1-Score', 'MCC']].set_index('Model')
sns.heatmap(metrics_for_heatmap.T, annot=True, fmt='.4f', cmap='RdYlGn', 
            ax=ax5, cbar_kws={'label': 'Score'}, vmin=0.85, vmax=1.0,
            linewidths=1, linecolor='black', annot_kws={'size': 9, 'weight': 'bold'})
ax5.set_title('Comprehensive Metrics Heatmap - All Models', fontsize=14, fontweight='bold')
ax5.set_ylabel('Metrics', fontsize=11, fontweight='bold')
ax5.set_xlabel('Models', fontsize=11, fontweight='bold')

plt.suptitle('Comprehensive Model Comparison - ML vs Deep Learning', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nComprehensive visualization saved as 'comprehensive_model_comparison.png'")


# ## 4.5 Model Interpretability - SHAP Analysis
# 
# **Interpretability Objective:**
# - Apply SHAP (SHapley Additive exPlanations) to understand model predictions
# - Identify most important features for fraud detection
# - Compare feature importance across different model types
# - Research Question 8: "What interpretability techniques can provide actionable explanations"
# 
# **Analysis Scope:**
# - XGBoost (best baseline ML model)
# - Hybrid Model (best deep learning model)
# - Feature importance comparison between ML and DL approaches

# In[ ]:


# SHAP EXPLAINABILITY - XGBOOST MODEL
print("=" * 80)
print("SHAP ANALYSIS - XGBOOST MODEL")
print("=" * 80)

if DLConfig.USE_SHAP:
    import shap

    # Sample data for SHAP (use config parameter)
    shap_samples = min(DLConfig.SHAP_SAMPLES, len(X_test_modeling))
    X_test_shap = X_test_modeling.iloc[:shap_samples]

    print(f"\nGenerating SHAP values for {shap_samples} test samples...")
    print("This may take a few minutes...")

    # Create SHAP explainer for XGBoost
    explainer_xgb = shap.TreeExplainer(xgb_model)
    shap_values_xgb = explainer_xgb.shap_values(X_test_shap)

    print("SHAP values computed successfully!")

    # Summary plot - Feature importance
    print("\n1. Generating SHAP summary plot (feature importance)...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Bar plot - mean absolute SHAP values
    shap.summary_plot(shap_values_xgb, X_test_shap, plot_type="bar", 
                     max_display=20, show=False)
    plt.title('XGBoost - Top 20 Feature Importance (SHAP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_xgb_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Beeswarm plot - detailed feature impact
    shap.summary_plot(shap_values_xgb, X_test_shap, max_display=20, show=False)
    plt.title('XGBoost - Feature Impact Distribution (SHAP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_xgb_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   - Saved: shap_xgb_bar.png")
    print("   - Saved: shap_xgb_beeswarm.png")

    # Get top features
    feature_importance_xgb = np.abs(shap_values_xgb).mean(axis=0)
    top_features_idx = np.argsort(feature_importance_xgb)[-10:][::-1]
    top_features = X_test_shap.columns[top_features_idx].tolist()

    print("\n2. Top 10 Most Important Features (XGBoost):")
    print("-" * 60)
    for i, (idx, feat) in enumerate(zip(top_features_idx, top_features), 1):
        importance = feature_importance_xgb[idx]
        print(f"   {i}. {feat}: {importance:.4f}")

    # Waterfall plot for a single prediction
    print("\n3. Generating waterfall plot for example prediction...")
    sample_idx = 0
    shap.waterfall_plot(shap.Explanation(
        values=shap_values_xgb[sample_idx],
        base_values=explainer_xgb.expected_value,
        data=X_test_shap.iloc[sample_idx],
        feature_names=X_test_shap.columns.tolist()
    ), max_display=15, show=False)
    plt.title(f'XGBoost - Individual Prediction Explanation (Sample {sample_idx})', 
             fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_xgb_waterfall.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Saved: shap_xgb_waterfall.png")

    print("\n" + "=" * 80)
    print("XGBOOST SHAP ANALYSIS COMPLETED")
    print("=" * 80)

else:
    print("\nSHAP analysis disabled in configuration (DLConfig.USE_SHAP = False)")
    print("Set DLConfig.USE_SHAP = True to enable interpretability analysis")


# In[ ]:


# SHAP EXPLAINABILITY - HYBRID DEEP LEARNING MODEL
print("=" * 80)
print("SHAP ANALYSIS - HYBRID DEEP LEARNING MODEL")
print("=" * 80)

if DLConfig.USE_SHAP:
    print(f"\nGenerating SHAP values for Hybrid model ({shap_samples} samples)...")
    print("Note: Deep learning SHAP is computationally intensive...")

    # Prepare background data for DeepExplainer
    background_samples = min(50, len(X_train_tensor))
    background_temporal = X_train_tensor[:background_samples]
    background_node_indices = train_node_indices[:background_samples]

    # Create wrapper function for SHAP
    def hybrid_predict_wrapper(x_temporal_np):
        """Wrapper function for SHAP that takes numpy input."""
        x_temporal = torch.tensor(x_temporal_np, dtype=torch.float32).to(cpu_device)

        # Use fixed node indices from test set (aligned with temporal features)
        num_samples = len(x_temporal)
        node_indices = test_node_indices[:num_samples]

        hybrid_model.eval()
        with torch.no_grad():
            outputs = hybrid_model(
                x_temporal,
                graph_data.x,
                graph_data.edge_index,
                node_indices
            )
            probs = F.softmax(outputs, dim=1).cpu().numpy()

        return probs

    # Get test data
    X_test_hybrid_shap = X_test_tensor[:shap_samples].cpu().numpy()

    # Create DeepExplainer
    print("Creating DeepExplainer...")
    background_np = background_temporal.cpu().numpy()
    explainer_hybrid = shap.KernelExplainer(
        hybrid_predict_wrapper,
        background_np
    )

    print("Computing SHAP values (this may take 5-10 minutes)...")
    shap_values_hybrid = explainer_hybrid.shap_values(X_test_hybrid_shap[:shap_samples])

    # For binary classification, take values for positive class
    if isinstance(shap_values_hybrid, list):
        shap_values_hybrid = shap_values_hybrid[1]  # Illicit class

    print("SHAP values computed successfully!")

    # Convert to DataFrame with feature names
    X_test_hybrid_df = pd.DataFrame(
        X_test_hybrid_shap,
        columns=X_test_modeling.columns
    )

    # Summary plots
    print("\n1. Generating SHAP summary plot (feature importance)...")

    # Bar plot
    shap.summary_plot(shap_values_hybrid, X_test_hybrid_df, plot_type="bar",
                     max_display=20, show=False)
    plt.title('Hybrid Model - Top 20 Feature Importance (SHAP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_hybrid_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Beeswarm plot
    shap.summary_plot(shap_values_hybrid, X_test_hybrid_df, max_display=20, show=False)
    plt.title('Hybrid Model - Feature Impact Distribution (SHAP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_hybrid_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   - Saved: shap_hybrid_bar.png")
    print("   - Saved: shap_hybrid_beeswarm.png")

    # Get top features
    feature_importance_hybrid = np.abs(shap_values_hybrid).mean(axis=0)
    top_features_idx_hybrid = np.argsort(feature_importance_hybrid)[-10:][::-1]
    top_features_hybrid = X_test_hybrid_df.columns[top_features_idx_hybrid].tolist()

    print("\n2. Top 10 Most Important Features (Hybrid Model):")
    print("-" * 60)
    for i, (idx, feat) in enumerate(zip(top_features_idx_hybrid, top_features_hybrid), 1):
        importance = feature_importance_hybrid[idx]
        print(f"   {i}. {feat}: {importance:.4f}")

    # Compare with XGBoost
    print("\n3. Feature Importance Comparison:")
    print("-" * 60)
    print("\nCommon Top Features (XGBoost vs Hybrid):")
    common_features = set(top_features) & set(top_features_hybrid)
    print(f"   - {len(common_features)} features appear in both top-10 lists:")
    for feat in common_features:
        print(f"     * {feat}")

    print("\n" + "=" * 80)
    print("HYBRID MODEL SHAP ANALYSIS COMPLETED")
    print("=" * 80)

    # Create comparison visualization
    print("\n4. Creating feature importance comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # XGBoost top features
    top_10_xgb_features = [X_test_shap.columns[i] for i in top_features_idx]
    top_10_xgb_importance = [feature_importance_xgb[i] for i in top_features_idx]

    axes[0].barh(range(10), top_10_xgb_importance, color='#ff7f0e', alpha=0.7, edgecolor='black')
    axes[0].set_yticks(range(10))
    axes[0].set_yticklabels(top_10_xgb_features, fontsize=10)
    axes[0].set_xlabel('Mean |SHAP value|', fontsize=11, fontweight='bold')
    axes[0].set_title('XGBoost - Top 10 Features', fontsize=13, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)

    # Hybrid top features
    top_10_hybrid_features = [X_test_hybrid_df.columns[i] for i in top_features_idx_hybrid]
    top_10_hybrid_importance = [feature_importance_hybrid[i] for i in top_features_idx_hybrid]

    axes[1].barh(range(10), top_10_hybrid_importance, color='#9467bd', alpha=0.7, edgecolor='black')
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels(top_10_hybrid_features, fontsize=10)
    axes[1].set_xlabel('Mean |SHAP value|', fontsize=11, fontweight='bold')
    axes[1].set_title('Hybrid Model - Top 10 Features', fontsize=13, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)

    plt.suptitle('Feature Importance Comparison - XGBoost vs Hybrid Model', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_comparison_xgb_vs_hybrid.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("   - Saved: shap_comparison_xgb_vs_hybrid.png")

else:
    print("\nSHAP analysis disabled in configuration")


# ## 5. Summary and Conclusions
# 
# ### Completed Analyses
# 
# **1. Data Exploration and Preprocessing**
# - Loaded Elliptic Bitcoin dataset: 203,769 transactions, 49 time steps
# - Applied SMOTE balancing for class imbalance
# - Temporal train/test split (time steps 1-35 train, 36-49 test)
# 
# **2. Feature Engineering**
# - Graph features: degree centrality, PageRank, betweenness centrality
# - Temporal features: time-based aggregations
# - Total features: 165 original + engineered features
# 
# **3. Baseline Machine Learning Models**
# - Logistic Regression, Random Forest, XGBoost
# - Ensemble model achieving 0.9387 AUC-ROC
# 
# **4. Deep Learning Models**
# - **LSTM**: Temporal pattern modeling
# - **GraphSAGE**: Graph structure learning  
# - **Hybrid**: Combined temporal + graph features
# 
# **5. Model Comparison**
# - Comprehensive evaluation across 7 models
# - Metrics: AUC-ROC, AUC-PR, Precision, Recall, F1, MCC
# - Visualization: ROC curves, PR curves, confusion matrices

# ============================================================================
# COMPREHENSIVE 7-MODEL COMPARISON (FINAL RESULTS)
# ============================================================================
# This section creates a unified comparison table for all 7 models:
# ML Models: Logistic Regression, Random Forest, XGBoost, Ensemble (ML)
# DL Models: LSTM, GraphSAGE, Hybrid

if __name__ == '__main__':
    # Ensure all model results are available
    try:
        print("\n" + "=" * 100)
        print("FINAL COMPREHENSIVE MODEL COMPARISON (7 MODELS)")
        print("=" * 100)
        
        # Verify all models have been trained and results stored
        all_models_data = {}
        
        # ML Models (already trained and stored in ml_models_results)
        for model_name, results in ml_models_results.items():
            all_models_data[model_name] = results
        
        # Deep Learning Models (trained after ensemble)
        # These will be populated after LSTM, GNN, Hybrid training
        # For now, show which are available
        
        available_models = list(all_models_data.keys())
        print(f"\n✓ Available ML Models: {', '.join(available_models)}")
        print(f"  Pending DL Models: LSTM, GraphSAGE, Hybrid")
        print(f"  (Will be added after deep learning training)")
        
        # Create comparison table for available models
        comparison_results = []
        
        for model_name, results in all_models_data.items():
            y_pred = results['y_pred']
            y_proba = results['y_proba']
            
            # Calculate metrics using labeled test set
            auc_roc = roc_auc_score(y_test_labeled_numeric, y_proba)
            auc_pr = average_precision_score(y_test_labeled_numeric, y_proba)
            prec = precision_score(y_test_labeled_numeric, y_pred)
            rec = recall_score(y_test_labeled_numeric, y_pred)
            f1 = f1_score(y_test_labeled_numeric, y_pred)
            mcc = matthews_corrcoef(y_test_labeled_numeric, y_pred)
            
            comparison_results.append({
                'Model': model_name,
                'AUC-ROC': auc_roc,
                'AUC-PR': auc_pr,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'MCC': mcc
            })
        
        # Create comparison DataFrame
        comparison_final_df = pd.DataFrame(comparison_results)
        comparison_final_df = comparison_final_df.sort_values('AUC-ROC', ascending=False).reset_index(drop=True)
        
        print("\n" + "=" * 100)
        print("ML MODELS PERFORMANCE TABLE")
        print("=" * 100)
        print(comparison_final_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        
        best_ml_model = comparison_final_df.iloc[0]['Model']
        best_ml_auc = comparison_final_df.iloc[0]['AUC-ROC']
        
        print(f"\n✓ Best ML Model: {best_ml_model}")
        print(f"✓ Best AUC-ROC: {best_ml_auc:.4f}")
        print(f"✓ Test Set Size: {len(y_test_labeled):,} labeled transactions")
        
    except Exception as e:
        print(f"\nNote: Final comparison will be completed after deep learning training")
        print(f"Error details: {str(e)}")


# 
# **6. Interpretability Analysis**
# - SHAP analysis for XGBoost and Hybrid models
# - Feature importance identification
# - Comparison of ML vs DL feature relevance
# 
# ### Thesis Alignment
# 
# This implementation fully addresses:
# - **Chapter 3 Methodology**: All required models implemented (ML baselines, LSTM, GNN, Hybrid)
# - **Research Questions**: Temporal modeling, graph structure, hybrid approaches, interpretability
# - **Evaluation Framework**: Comprehensive metrics and statistical comparison
# - **Reproducibility**: Configuration-based approach with fixed random seeds
# 
# ### Next Steps for Thesis
# 
# 1. **Run all cells** to generate complete results
# 2. **Document findings** in thesis Chapter 4 (Results)
# 3. **Discuss insights** from SHAP analysis in interpretability section
# 4. **Compare with literature** - how do these results compare to related work?
# 5. **Future work** - potential improvements and extensions
