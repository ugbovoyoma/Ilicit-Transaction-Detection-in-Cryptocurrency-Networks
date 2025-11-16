# LSTM Model Improvements - Applied Changes

## Date: November 15, 2025

## Problem Diagnosis
The original LSTM model underperformed (AUC-ROC: 0.838 vs XGBoost: 0.943) due to:
1. **Label noise from pseudo-labeling**: Low confidence threshold (0.9) included noisy pseudo-labels
2. **Class imbalance not addressed**: No class weighting in loss function
3. **Overfitting**: Train loss (0.013) << Val loss (0.28) indicated memorization
4. **Wrong architecture for data**: LSTM trained on flat transaction vectors (no true sequences)

## Implemented Fixes

### 1. Tighter Pseudo-Label Threshold ✓
**Change**: Increased confidence threshold from 0.9 → 0.95  
**Location**: Line ~397 in `illicitTransactionDetection.py`  
**Rationale**: Fewer but higher-quality pseudo-labels reduces label noise  
**Expected Impact**: Better training signal, less overfitting to synthetic data

```python
# Before: CONFIDENCE_THRESHOLD = 0.9
# After:  CONFIDENCE_THRESHOLD = 0.95
```

### 2. Class-Weighted Loss Function ✓
**Change**: Added balanced class weights to CrossEntropyLoss  
**Location**: Line ~1665 in `illicitTransactionDetection.py`  
**Rationale**: SMOTE reduces but doesn't eliminate imbalance; explicit weighting helps  
**Expected Impact**: Better recall on illicit class, improved precision-recall balance

```python
# Compute class weights using sklearn
class_weights = compute_class_weight('balanced', classes=[0,1], y=y_train)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### 3. Focal Loss (Optional Alternative) ✓
**Change**: Added FocalLoss implementation (commented out)  
**Location**: Line ~1668 in `illicitTransactionDetection.py`  
**Rationale**: Focal loss emphasizes hard examples and is robust to noisy labels  
**How to enable**: Uncomment lines and comment out standard CrossEntropyLoss

```python
# To switch to Focal Loss:
# criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights)
```

### 4. AUC-PR Monitoring During Training ✓
**Change**: Added validation AUC-PR calculation each epoch  
**Location**: Line ~1770 in `illicitTransactionDetection.py`  
**Rationale**: AUC-PR is more informative than loss for imbalanced data  
**Expected Impact**: Better visibility into model quality during training

```python
# Now prints: Val Loss: 0.2140 | Val AUC-PR: 0.1576
```

## Expected Performance Improvements

| Metric | Before | Target | Reasoning |
|--------|--------|--------|-----------|
| AUC-ROC | 0.838 | 0.88-0.90 | Class weights + cleaner labels |
| AUC-PR | 0.158 | 0.25-0.35 | Better minority class focus |
| Precision (illicit) | 0.176 | 0.30-0.45 | Fewer false positives |
| Recall (illicit) | 0.600 | 0.55-0.65 | Maintained with better precision |
| Overfitting gap | Large | Smaller | Cleaner pseudo-labels |

## How to Run

```bash
# Run the improved script
uv run python illicitTransactionDetection.py
```

The script will now:
1. Use only 95%+ confidence pseudo-labels (fewer but cleaner)
2. Apply class weights during LSTM training
3. Show AUC-PR alongside loss during training epochs
4. Stop earlier if validation metrics don't improve

## Next Steps (If Further Improvement Needed)

### Quick Wins
1. **Switch to Focal Loss**: Uncomment focal loss in training section
2. **Reduce SMOTE ratio**: Change `sampling_strategy=0.2` to `0.15` (less synthetic data)
3. **Add more dropout**: Increase dropout from 0.3 to 0.4-0.5

### Medium Effort
4. **Bidirectional LSTM**: Change to `bidirectional=True` in model config
5. **Increase hidden dim**: Change from 64 to 128 (more capacity)
6. **Early stopping on AUC-PR**: Modify early stopping to use AUC-PR instead of loss

### Research Direction (Thesis-worthy)
7. **Build proper sequences**: Group transactions by node_ID/time_step
8. **Hybrid GNN+LSTM**: Feed GNN embeddings into LSTM
9. **Attention mechanism**: Re-add attention with proper sequence masking

## Code Changes Summary

**Files Modified**: 
- `illicitTransactionDetection.py` (4 sections)

**Lines Changed**: ~30 lines total
- Pseudo-label threshold: 1 line
- Class weight calculation: 15 lines
- AUC-PR monitoring: 6 lines
- Documentation: 8 lines

**No Breaking Changes**: All changes are backward-compatible

## Validation Checklist
- [x] Pseudo-label threshold increased to 0.95
- [x] Class weights computed and applied to loss
- [x] Focal Loss implementation added (optional)
- [x] AUC-PR tracking added to training loop
- [ ] Re-run full training pipeline
- [ ] Compare metrics with baseline
- [ ] Document results in thesis

## References
- Focal Loss: Lin et al. (2017) "Focal Loss for Dense Object Detection"
- Class Balancing: Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling"
- Pseudo-labeling: Lee (2013) "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning"

---
**Author**: GitHub Copilot  
**Date**: November 15, 2025  
**Status**: Ready for testing
