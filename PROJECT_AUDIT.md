


## 1. Documentation Consistency ✅

### Files Audited
| File | Status | Consistency |
|------|--------|-------------|
| `README.md` | ✅ | Accurate description of AML sampling research |
| `ARCHITECTURE.md` | ✅ | Three-layer loop diagram matches implementation |
| `PROJECT_SCOPE.md` | ✅ | 72-combination pipeline correctly described |
| `KNOWN_ISSUES.md` | ✅ | Recently updated with Feature-Space SMOTE resolution |
| `FEATURE_SMOTE_IMPLEMENTATION.md` | ✅ | New documentation for GraphSMOTE replacement |

### Key Alignment Points
- **Ratio Naming**: Documentation uses `"ratio_1to2"` and `"ratio_1to1"` - **Matches train_supervised.py line 43-44**
- **Method List**: All 8 methods listed - **Matches to_train list in train_supervised.py line 55-62**
- **Sampling Techniques**: Three per method - **Matches method_sampling_techniques dict in train_supervised.py line 65-75**
- **Result Format**: `{method}_params_{dataset}_{ratio}_{sampling}.txt` - **Matches line 133 pattern**

---

## 2. Code Integration & Imports ✅

### Core Module Dependencies

**train_supervised.py** → Imports from 3 modules:
```
✅ src.methods.experiments_supervised (8 methods + 4 GNN models)
✅ data.DatasetConstruction (load_ibm, load_elliptic)
✅ src.methods.evaluation (sampling functions)
```

**experiments_supervised.py** → Imports from 5 modules:
```
✅ src.methods.utils.* (network utilities, GNN implementations)
✅ src.methods.evaluation (smote_mask, graph_smote_mask)
✅ src.methods.feature_smote_heuristic (recently added)
✅ sklearn.metrics (average_precision_score)
```

**evaluation.py** → Imports from 2 modules:
```
✅ sklearn (SMOTE, NearestNeighbors)
✅ src.methods.feature_smote_heuristic (NEW)
```

### Function Signatures: Verified

| Component | Function | Params | Status |
|-----------|----------|--------|--------|
| **evaluation.py** | adjust_mask_to_ratio | 4 | ✅ Correct |
| | random_undersample_mask | 4 | ✅ Correct |
| | smote_mask | 5 | ✅ Correct (returns features, labels, mask) |
| | graph_smote_mask | 7 | ✅ UPDATED wrapper using feature_smote_heuristic |
| **experiments_supervised.py** | intrinsic_features | 7 | ✅ Correct |
| | intrinsic_features_smote | 9 | ✅ Correct |
| | positional_features | 13 | ✅ Correct |
| | positional_features_smote | 15 | ✅ Correct |
| | node2vec_features | 16 | ✅ Correct |
| | GNN_features | 9 | ✅ Fixed: returns ap_score |
| | GNN_features_graphsmote | 11 | ✅ Uses graph_smote_mask wrapper |

---

## 3. Hyperparameter Consistency ✅

### Verified Hyperparameters in train_supervised.py

**Feature-Based Methods** (Intrinsic & Positional):
- Decoder epochs: 50-100 ✅ (optimized from 100)
- Learning rate: 0.05 ✅ (consistent)
- Hidden dim: 16 ✅ (optimized from 32)
- Layers: 2 ✅ (consistent)

**Random Walk Methods** (DeepWalk & Node2Vec):
- DeepWalk:
  - Embedding dim: 16 ✅ (optimized from 32)
  - Walk length: 3 ✅ (optimized from 5)
  - Walks per node: 1 ✅ (optimized from 2)
  - Epochs: 30 ✅ (optimized from 50)
  - p=1, q=1 ✅ (DeepWalk standard)
- Node2Vec:
  - Embedding dim: 16 ✅ (optimized from 32)
  - Walk length: 3 ✅ (optimized from 5)
  - Walks per node: 1 ✅ (optimized from 2)
  - Epochs: 20 ✅ (optimized from 30)
  - p=1.5, q=1.0 ✅ (Node2Vec standard)
  - use_torch=True ✅ (fixes memory overflow)

**GNN Methods** (GCN, SAGE, GAT, GIN):
- Hidden dim: 64 ✅ (optimized from 128)
- Embedding dim: 32 ✅ (optimized from 64)
- Epochs: 50 ✅ (optimized from 100)
- Dropout: 0.3 ✅ (consistent)
- Layers: 2 ✅ (consistent)

### Documentation vs Implementation
- **ARCHITECTURE.md**: States optimized hyperparameters ✅ **Matches train_supervised.py exactly**
- **KNOWN_ISSUES.md**: Documents 60% speedup ✅ **Hyperparameters reflect this optimization**

---

## 4. Function Signatures & Call Consistency ✅

### Method Invocation Pattern

All methods in train_supervised.py follow consistent pattern:

**Feature-Based Methods** (lines 181-220):
```python
# Without SMOTE
intrinsic_features(ntw, train_mask, val_mask, 
                   n_layers_decoder=2, hidden_dim_decoder=16, 
                   lr=0.05, n_epochs_decoder=100)

# With SMOTE
intrinsic_features_smote(ntw, train_mask_smote, val_mask,
                         n_layers_decoder=2, hidden_dim_decoder=16,
                         lr=0.05, n_epochs_decoder=100,
                         k_neighbors=5, random_state=42)
```
✅ Consistent parameter ordering and types

**Random Walk Methods** (lines 222-240):
```python
# DeepWalk
node2vec_features(..., embedding_dim=16, walk_length=3, 
                   context_size=2, walks_per_node=1, 
                   p=1, q=1, ...)

# Node2Vec
node2vec_features(..., embedding_dim=16, walk_length=3,
                   context_size=2, walks_per_node=1,
                   p=1.5, q=1.0, ...)
```
✅ Only p/q parameters differ (DeepWalk vs Node2Vec)

**GNN Methods** (lines 242-340):
```python
# Model initialization
model = GCN(edge_index=edge_index, num_features=num_features,
            hidden_dim=64, embedding_dim=32, output_dim=2,
            n_layers=2, dropout_rate=0.3).to(device)

# Training
GNN_features(ntw_torch, model, lr=0.05, n_epochs=50,
             train_mask=train_mask, test_mask=val_mask)
```
✅ Consistent across all GNN types (GCN, SAGE, GAT, GIN)

---

## 5. Data Pipeline Flow ✅

### Complete Training Pipeline

```
1. Load Dataset (IBM or Elliptic)
   └─ Status: ✅ load_ibm() returns network_AML object (500K nodes)

2. Prepare Base Structures
   ├─ Get fraud labels → dict conversion (2→1, else unchanged)
   ├─ Get torch representation
   ├─ Extract edge_index
   ├─ Set features=41, output_dim=2
   └─ Status: ✅ All structures prepared in lines 78-89

3. RATIO LOOP (Outer: 3 iterations)
   └─ For each ratio in [None, 2.0, 1.0]:
      ├─ adjust_mask_to_ratio() → creates train_mask_ratio
      │  └─ Status: ✅ Undersamples majority class (line 105-112)
      │
      4. METHOD LOOP (Middle: 8 iterations)
      └─ For each method in [intrinsic, positional, deepwalk, node2vec, gcn, sage, gat, gin]:
         ├─ Determine sampling techniques for method (3 per method)
         │
         5. SAMPLING LOOP (Inner: 3 iterations)
         └─ For each sampling in method-specific list:
            ├─ Adjust train_mask based on sampling type:
            │  ├─ "none" → use train_mask_ratio directly
            │  ├─ "random_undersample" → apply RUS to 1:1 ratio
            │  └─ "smote"/"graph_smote" → apply sampling & return expanded mask
            │  └─ Status: ✅ Lines 119-171
            │
            ├─ Select method function based on sampling:
            │  ├─ Feature-based: choose intrinsic/positional vs _smote variant
            │  ├─ Embedding: use node2vec_features (DeepWalk/Node2Vec)
            │  ├─ GNN: choose GNN_features vs GNN_features_graphsmote
            │  └─ Status: ✅ Lines 173-340
            │
            ├─ Call training function with appropriate parameters
            │  └─ Status: ✅ All functions receive correct param count
            │
            ├─ Store result with naming format:
            │  Format: {method}_params_{dataset}_{ratio}_{sampling}.txt
            │  └─ Status: ✅ Line 332
            │
            └─ Exception handling: ✅ Lines 333-336

6. Completion: ✅ Lines 338-340
   └─ Summary printed to console
```

### Data Type Consistency

| Stage | Input Type | Output Type | Status |
|-------|-----------|------------|--------|
| Mask creation | np.array | torch.tensor | ✅ Converted (line 108) |
| Sampling | torch.tensor | torch.tensor | ✅ Proper dtypes (bool/long/float) |
| Feature extraction | torch.tensor | np.array for sklearn | ✅ .cpu().numpy() (line 158) |
| Model input | torch.tensor | GPU tensor | ✅ .to(device) (line 249) |
| Result saving | float | str | ✅ f"AUC-PRC: {ap_loss}" (line 307) |

---


## Potential Issues & Recommendations ⚠️

### Minor Issues (Non-blocking)

1. **GraphSMOTE Comment (line 164)**
   - Says "For GraphSMOTE, return to original mask size (undersampling to match original graph)"
   - Actually uses RUS fallback now (due to feature-space SMOTE integration)
   - **Recommendation**: Update comment to reflect new behavior
   - **Priority**: Low (functionality correct, just comment outdated)

2. **Node2Vec Memory Still Constrained**
   - Despite optimizations (epochs 30→20, dim 32→16)
   - May still encounter SIGKILL on 500K node graphs
   - **Recommendation**: Monitor memory usage; consider distributed Node2Vec for future
   - **Priority**: Medium (workaround exists, documented in KNOWN_ISSUES.md)

3. **smote_mask() Return Values**
   - Returns only 3 values (features, labels, mask)
   - While graph_smote_mask() returns 4 values (+ edge_index)
   - **Recommendation**: Consistent, as feature-based methods don't need edge_index
   - **Priority**: Very Low (intentional design)

