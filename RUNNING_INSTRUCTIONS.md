# Running Instructions

## Python Script Version

The notebook has been converted to a Python script (`illicitTransactionDetection.py`) that you can run directly from the terminal.

### Prerequisites

1. **Dataset**: Make sure your dataset is located in the correct directory:
   ```
   ./dataset/rawData/elliptic_bitcoin_dataset/
   ├── elliptic_txs_classes.csv
   ├── elliptic_txs_edgelist.csv
   └── elliptic_txs_features.csv
   ```

2. **Dependencies**: All dependencies are managed by `uv` and defined in `pyproject.toml`.

### Running the Script

To run the entire analysis pipeline:

```bash
uv run python illicitTransactionDetection.py
```

This will execute the entire analysis including:
- Data loading and exploration
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training (Logistic Regression, Random Forest, XGBoost)
- Deep Learning models (LSTM, GNN)
- Model evaluation and comparison

### Running with UV

The `uv` tool automatically manages your virtual environment and dependencies:

```bash
# Install/sync dependencies
uv sync

# Run the script
uv run python illicitTransactionDetection.py
```

### Troubleshooting

**If you get "File not found" errors for the dataset:**
- Update the `DATA_DIR` variable in the script (line ~40) to point to your dataset location

**If you get import errors:**
- Run `uv sync` to install all dependencies
- Check that all packages in `pyproject.toml` are installed

**Memory issues:**
- The script processes a large dataset and trains multiple models
- Consider running on a machine with at least 16GB RAM
- For GPU acceleration, ensure CUDA is properly installed

### Output

The script will generate:
- Various plots and visualizations (saved as PNG files)
- Model performance metrics printed to console
- Saved model files (e.g., `best_lstm_model.pth`)
- Comparison visualizations

### Notes

- The original Jupyter notebook is still available: `illicitTransactionDetection.ipynb`
- This Python script is automatically generated from the notebook
- To regenerate: `jupyter nbconvert --to python illicitTransactionDetection.ipynb`
