# Illicit Transaction Detection in Cryptocurrency Networks

This repository contains the implementation code for an MSc dissertation submitted to the University of Ibadan, February 2026.

---

## Thesis Information

**Title:** Machine, Deep and Ensemble Learning For Illicit Transaction Detection In Cryptocurrency Networks: A Comparative Analysis Of Temporal, Graph, and Tabular Learning Architectures

**Author:** Ugbovo Ogheneyoma Precious  
**Matric No.:** 246087  
**Degree:** Master of Data and Information Science  
**Department:** Data and Information Science  
**Faculty:** Multidisciplinary Studies  
**Institution:** University of Ibadan  
**Supervisor:** Dr. Janet O. Adekannbi (B.Sc., M.Inf.Sc., PhD. Ibadan)  
**Date:** February 2026

---

## Abstract

Cryptocurrency networks have become an indispensable pillar of the digital economy, where decentralisation, transparency and effectiveness of transactions become a reality. However, the pseudo-anonymity of the networks has also attracted many illicit practices, such as moneylaundering and terror funding. Conventional rule-based systems, which are traditionally used in centralised banking, are not effective in evaluating the ever-increasing scale, temporal complexity, and graph-like nature of the transactions in blockchain. The main aim of this study is to perform a comparative analysis of temporal, graph, and tabular learning architectures to determine the most effective strategy for identifying illicit cryptocurrency transfers. The research adopted a quantitative study design that utilised the Elliptic Bitcoin transaction dataset. To deal with the intrinsic problem of imbalance classes in financial fraud data, the machine learning baselines used the Synthetic Minority Oversampling Technique, and the deep learning architectures used Focal Loss. Three different learning paradigms were systematically assessed in the research: tabular learning with the help of logistic regression, random forest, and gradient boosting; graph learning with GraphSAGE helping to capture the topological structure; and temporal learning with bi-directional long short-term memory networks helping to model transaction sequences. These models were tested individually and were then combined to create a Grand Ensemble strategy. The comparison showed that tabular learning structures are still highly effective when designed with aggregated neighbourhood features. In particular, the XGBoost model showed high results on an individual level, with a precision of 94.69% and an F1-score of 78.65%. Conversely, although the hybrid graph-temporal network was able to effectively capture latent structural dependencies, it had a slightly lower F1 score of 76.04%. The research established that the best solution was a combination of these strategies. The constructed Grand Ensemble that overlaid the feature sharpness of XGBoost with the structural embeddings of the hybrid network gained a higher precision of 94.70%, a recall of 67.48%, an F1-score of 78.81%, and a Matthews Correlation Coefficient of 0.7896. In addition, the model proved to have good discrimination with an area under the ROC curve of 0.9196 and an area under the precision-recall curve of 0.7591. This study concludes that while advanced tabular models are competitive, hybrid ensemble architectures provide the strongest defence against financial crime. It is suggested that financial intelligence departments should implement ensemble-based monitoring tools to reduce false positives while still achieving high detection rates. Also, it is suggested to integrate explainable AI tools, such as SHAP, to have the transparency required in regulatory compliance.

Keywords: Cryptocurrency, Anti-Money Laundering, Graph Neural Networks, Deep Learning, Ensemble Learning
Word Count: 409


---

## Research Objectives

The primary objective of this study is to evaluate the performance of temporal, graph-based, and tabular architectures, as well as identify the best ensemble strategy for identifying illicit cryptocurrency transactions using classification metrics.

The specific objectives are:

a) To determine the statistical distributions, topological forms and temporal patterns ofillicit and licit transaction processes by conducting an exploratory data analysis on the Elliptic dataset.

b) To derive high-resolution and informative temporal and topological features that could effectively capture subtle patterns in the cryptocurrency transaction graph.

c) To evaluate and compare the baseline performance of different traditional machine learning algorithms using standard metrics to measure the predictive strength of explicit statistical information.

d) To determine the detection efficacy of deep learning Bi-Directional LSTMs and GraphSAGE by benchmarking their performance metrics against the traditional
tabular models.

e) To derive a robust hybrid ensemble architecture that integrates tabular, temporal and graph-based learners, and to mathematically measure its impact on detection stability and the reduction of false positives.

f) To systematically compare the isolated architectures against the Grand Ensemble strategy to determine the optimal operational threshold that effectively balances detection recall with operational precision.
---

## Project Structure

```
illicit-transaction-detection/
├── illicit_pipeline/
│   ├── config.py                  # Central configuration and hyperparameters
│   ├── data_loading.py            # Dataset loading utilities
│   ├── preprocessing.py           # Data preprocessing and balancing
│   ├── features.py                # Temporal and graph feature engineering
│   ├── analysis.py                # Exploratory data analysis
│   ├── pipeline.py                # Pipeline orchestration
│   └── models/
│       ├── baseline.py            # ML models (LR, RF, XGBoost, Ensemble)
│       └── deep.py                # DL models (LSTM, GraphSAGE, Hybrid)
├── dataset/
│   └── rawData/
│       └── elliptic_bitcoin_dataset/
│           ├── elliptic_txs_classes.csv
│           ├── elliptic_txs_features.csv
│           └── elliptic_txs_edgelist.csv
├── run_pipeline.py                # Command-line entry point
├── illicitTransactionDetection.ipynb
├── model_comparison_results.csv
└── pyproject.toml
```

---

## Dataset

This research utilizes the Elliptic Bitcoin Dataset, a large-scale labeled transaction graph dataset.

| Statistic | Value |
|-----------|-------|
| Total Transactions | 203,769 |
| Labeled Transactions | 46,564 |
| Illicit Transactions | 4,545 (9.8%) |
| Licit Transactions | 42,019 (90.2%) |
| Unlabeled Transactions | 157,205 |
| Transaction Edges | 234,355 |
| Time Steps | 49 |
| Features per Transaction | 166 |

The dataset exhibits severe class imbalance (approximately 10:1 ratio), making it a challenging benchmark for fraud detection algorithms.

---

## Methodology

### Feature Engineering

**Temporal Features (16 features)**
- Normalized time steps and cyclical encodings (sine/cosine transformations)
- Time period indicators (early, mid, late)
- Fraud campaign detection flags
- Rolling transaction counts and velocity metrics
- Burst period detection and rapid transaction indicators

**Graph Features (8 features)**
- Degree centrality (in-degree, out-degree, total degree)
- PageRank scores
- Clustering coefficients
- Betweenness centrality
- Neighbor average degree and degree ratios

### Models Evaluated

| Category | Model | Description |
|----------|-------|-------------|
| Tabular | Logistic Regression | Linear baseline with L2 regularization |
| Tabular | Random Forest | Ensemble of 200 decision trees |
| Tabular | XGBoost | Gradient boosted trees with class weighting |
| Temporal | LSTM | Long Short-Term Memory network |
| Graph | GraphSAGE | Inductive graph neural network |
| Hybrid | Ensemble | Weighted average by AUC-ROC scores |
| Hybrid | DL Hybrid | LSTM and GraphSAGE fusion with attention |

### Class Imbalance Strategies

- SMOTE (Synthetic Minority Over-sampling Technique)
- Class weighting in loss functions
- Focal Loss for deep learning models
- Threshold optimization via precision-recall curves

---

## Results

### Model Performance Comparison

| Model | AUC-ROC | AUC-PR | Precision | Recall | F1-Score | MCC |
|-------|---------|--------|-----------|--------|----------|-----|
| XGBoost | 0.940 | 0.778 | 0.946 | 0.675 | 0.788 | 0.789 |
| Ensemble (ML) | 0.932 | 0.758 | 0.842 | 0.683 | 0.754 | 0.745 |
| Random Forest | 0.922 | 0.752 | 0.943 | 0.676 | 0.787 | 0.788 |
| Hybrid (DL) | 0.913 | 0.758 | 0.900 | 0.683 | 0.777 | 0.773 |
| LSTM | 0.904 | 0.646 | 0.212 | 0.804 | 0.336 | 0.349 |
| Logistic Regression | 0.878 | 0.315 | 0.177 | 0.795 | 0.290 | 0.302 |
| GraphSAGE | 0.876 | 0.565 | 0.228 | 0.707 | 0.344 | 0.339 |

### Key Findings

1. XGBoost achieves the best overall performance with an AUC-ROC of 0.94 and the highest precision (94.6%)
2. LSTM and GraphSAGE achieve high recall but suffer from low precision, producing many false positives
3. Temporal and graph features significantly improve detection accuracy over baseline feature sets
4. Combining machine learning models through ensemble methods yields consistent improvements over individual models

---

## Installation

### Prerequisites

- Python 3.13 or higher
- CUDA-compatible GPU (optional, for faster deep learning training)

### Setup

Clone the repository:

```bash
git clone https://github.com/yourusername/Illicit-Transaction-Detection-in-Cryptocurrency-Networks.git
cd Illicit-Transaction-Detection-in-Cryptocurrency-Networks
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS
```

Install dependencies:

```bash
pip install -e .
```

Download the Elliptic dataset from [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place the files in `dataset/rawData/elliptic_bitcoin_dataset/`.

---

## Usage

Run baseline models only:

```bash
python run_pipeline.py
```

Include deep learning models (LSTM and GraphSAGE):

```bash
python run_pipeline.py --deep
```

Skip exploratory analysis output:

```bash
python run_pipeline.py --no-analysis
```

Run the interactive Jupyter notebook:

```bash
jupyter lab illicitTransactionDetection.ipynb
```

---

## Configuration

Key hyperparameters can be modified in `illicit_pipeline/config.py`:

```python
@dataclass
class DLConfig:
    # LSTM
    LSTM_HIDDEN_DIM: int = 64
    LSTM_NUM_LAYERS: int = 1
    LSTM_DROPOUT: float = 0.3
    
    # GraphSAGE
    GNN_HIDDEN_DIM: int = 128
    GNN_NUM_LAYERS: int = 2
    GNN_DROPOUT: float = 0.3
    
    # Training
    BATCH_SIZE: int = 256
    NUM_EPOCHS: int = 30
    LEARNING_RATE: float = 0.001
    PATIENCE: int = 5
```

---

## Dependencies

- torch (>=2.5.0)
- torch-geometric (>=2.6.0)
- xgboost (>=3.1.1)
- scikit-learn (>=1.7.2)
- pandas (>=2.3.3)
- networkx (>=3.5)
- imbalanced-learn (>=0.14.0)
- shap (>=0.45.0)
- matplotlib (>=3.10.7)
- seaborn (>=0.13.2)

---

## References

1. Weber, M., et al. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. KDD Workshop on Anomaly Detection in Finance.

2. Elliptic. (2019). The Elliptic Data Set: Opening Up Machine Learning on the Blockchain. https://www.elliptic.co/

3. Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.

4. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.

5. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.

---

## Acknowledgments

I would like to thank Dr. Janet O. Adekannbi for her guidance and mentorship throughout this research. I also acknowledge Elliptic for providing the labeled Bitcoin transaction dataset, and the Faculty of Multidisciplinary Studies at the University of Ibadan for their support.

---

## License

This project is licensed under the MIT License.

---

## Contact

Ugbovo Ogheneyoma Precious  
ugbovoyoma@gmail.com
