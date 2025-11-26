# Ilicit-Transaction-Detection-in-Cryptocurrency-Networks

## Modular pipeline
The original `illicitTransactionDetection.py` remains untouched. A modular pipeline now lives alongside it:

- Core modules: `illicit_pipeline/config.py`, `illicit_pipeline/data_loading.py`, `illicit_pipeline/analysis.py`, `illicit_pipeline/features.py`, `illicit_pipeline/preprocessing.py`, `illicit_pipeline/models/`.
- Entrypoint: `run_pipeline.py`

Run baselines:
```
python run_pipeline.py
```

Include deep models (LSTM + GraphSAGE):
```
python run_pipeline.py --deep
```

Skip the exploratory analysis prints:
```
python run_pipeline.py --no-analysis
```
