# Pipeline Architecture Full Analysis

## Scope and revision note

This document describes the current supervised pipeline after the mask-isolation refactor. The analysis is based on the code in [scripts/train_supervised.py](../scripts/train_supervised.py), [src/methods/experiments_supervised.py](../src/methods/experiments_supervised.py), [src/utils/Network.py](../src/utils/Network.py), [data/DatasetConstruction.py](../data/DatasetConstruction.py), [src/methods/evaluation.py](../src/methods/evaluation.py), and [src/methods/utils/GNN.py](../src/methods/utils/GNN.py).

The central design principle in the current version is:

- **train, validation, and test are structurally separated**
- **sampling is restricted to the training portion only**
- **final reported metrics are computed on the held-out test split**

---

## 1. Dataset loading and adjustment logic

### 1.1 Entry points

The dataset loading entry points are:

- [data/DatasetConstruction.py](../data/DatasetConstruction.py): `load_ibm()`
- [data/DatasetConstruction.py](../data/DatasetConstruction.py): `load_elliptic()`
- [scripts/train_supervised.py](../scripts/train_supervised.py): the main driver calls `load_ibm()` or `load_elliptic()` based on the `--network` argument

### 1.2 What is loaded and what is adjusted?

#### IBM

The IBM loader in [data/DatasetConstruction.py](../data/DatasetConstruction.py) does the following:

1. Reads the raw transaction CSV from the IBM data directory.
2. Converts the timestamp column into a proper datetime object.
3. Removes self-transactions by filtering out rows where `Account == Account.1`.
4. Keeps the last 500,000 transactions for the experiment window.
5. Builds a feature table containing amount, currency, payment format, and derived time features (`Day`, `Hour`, `Minute`).
6. One-hot-encodes categorical columns.
7. Constructs a time-based split using the ordered transaction chronology.

Important point: **there is no front-end class-ratio adjustment before the split**. The loader does not perform random undersampling, SMOTE, or any global rebalancing before creating the masks.

#### Elliptic

The Elliptic loader in [data/DatasetConstruction.py](../data/DatasetConstruction.py) does the following:

1. Reads three files: feature matrix, edge list, and class labels.
2. Renames columns and maps the raw class labels to a compact form:
   - `unknown -> 2`
   - `1 -> 1`
   - `2 -> 0`
3. Uses the `time_step` column to build a temporal split.
4. Excludes unlabeled rows by applying a mask that requires `class != 2`.

Again, **no global sampling or ratio balancing is applied before the split**.

### 1.3 Pure-state confirmation

The pipeline is deliberately kept clean at the data-loading stage:

- No global RUS is applied before the split.
- No SMOTE is applied before the split.
- No graph-level resampling is applied before the split.
- The only preprocessing is feature engineering and categorical encoding.

Text-based flowchart:

```text
Raw CSV / raw feature files
    -> read into pandas DataFrame
    -> remove invalid/self edges
    -> engineer features / one-hot encode / derive time features
    -> build masks from chronology or time_step
    -> wrap into network_AML object
    -> pass to training/evaluation functions
```

### 1.4 The resulting object type

The loaded dataset is wrapped into a `network_AML` object in [src/utils/Network.py](../src/utils/Network.py). That object stores:

- the feature table
- the edge list
- the original masks
- the fraud-label dictionary
- a PyTorch Geometric `Data` object produced by `get_network_torch()`

---

## 2. Dataset split and mask flow

### 2.1 Where masks are created

The masks are created in [data/DatasetConstruction.py](../data/DatasetConstruction.py):

- IBM: `train_mask`, `val_mask`, `test_mask` are constructed by slicing the ordered transaction table into 60%, 20%, 20% portions.
- Elliptic: `train_mask`, `val_mask`, `test_mask` are constructed from `time_step` thresholds.

The masks are then passed into the `network_AML` constructor in [src/utils/Network.py](../src/utils/Network.py), where they are attached to the object and later exposed by `get_masks()`.

### 2.2 Ratio details

| Dataset | Split logic | Train | Validation | Test |
|---|---|---:|---:|---:|
| IBM | chronological 60/20/20 split | 60% | 20% | 20% |
| Elliptic | time-step based split | `time_step < 30` and labeled | `30 <= time_step < 40` and labeled | `time_step >= 40` and labeled |

### 2.3 Mask flow across files

The flow is:

```text
load_ibm() / load_elliptic()
    -> create train_mask / val_mask / test_mask
    -> network_AML(..., train_mask=..., val_mask=..., test_mask=...)
    -> ntw.get_masks()
    -> scripts/train_supervised.py receives train_mask, val_mask, test_mask
    -> pass masks into training/evaluation functions
```

### 2.4 Function-stage mapping

| Phase | File path | Core function | What it does |
|---|---|---|---|
| Train | [scripts/train_supervised.py](../scripts/train_supervised.py) | main loop in `__main__` | Loads the dataset, clones the original training mask, and dispatches to the method-specific training functions. |
| Train | [src/methods/experiments_supervised.py](../src/methods/experiments_supervised.py) | `intrinsic_features()` | Uses `train_mask` to fit a decoder on intrinsic features and evaluates on the test split. |
| Train | [src/methods/experiments_supervised.py](../src/methods/experiments_supervised.py) | `positional_features()` | Uses `train_mask` to build positional features and fit a decoder, then evaluates on the test split. |
| Train / Validation / Final Test | [src/methods/experiments_supervised.py](../src/methods/experiments_supervised.py) | `GNN_features()` | Trains on `train_mask`, computes validation loss/AP/F1 on `val_mask`, and computes final AP/F1 on `test_mask`. |
| Train / Validation / Final Test | [src/methods/experiments_supervised.py](../src/methods/experiments_supervised.py) | `GNN_features_graphsmote()` | Applies GraphSMOTE on the training portion, trains on the synthetic training subset, monitors `val_mask`, and reports final metrics on `test_mask`. |
| Mask packaging | [src/utils/Network.py](../src/utils/Network.py) | `get_network_torch()` | Packages the graph into a PyTorch Geometric `Data` object and attaches `train_mask`, `val_mask`, and `test_mask` to it. |

### 2.5 How masks are used in practice

The training driver in [scripts/train_supervised.py](../scripts/train_supervised.py) explicitly does this:

```python
train_mask, val_mask, test_mask = ntw.get_masks()
train_mask_ratio = train_mask.clone()
```

That means the original split is preserved as the starting point. Any sampling is later applied only to the training portion, while validation and test remain untouched.

---

## 3. Sampling and numerical computation

### 3.1 Sampling entry points

#### Feature-space methods

The feature-based methods are:

- `intrinsic`
- `positional`
- `deepwalk`
- `node2vec`

The training driver in [scripts/train_supervised.py](../scripts/train_supervised.py) dispatches them based on the selected sampling method (`none`, `random_undersample`, `smote`, `graph_smote`, `graph_ensemble_smote`).

The official SMOTE implementation is invoked in [src/methods/evaluation.py](../src/methods/evaluation.py) via `smote_mask()`.

#### GNN methods

The graph methods are:

- `gcn`
- `sage`
- `gat`
- `gin`

They use the graph-based helpers in [src/methods/evaluation.py](../src/methods/evaluation.py):

- `graph_smote_mask()`
- `graph_ensemble_smote_mask()`

### 3.2 Where SMOTE is applied

For intrinsic and positional methods, the workflow is:

```text
scripts/train_supervised.py
    -> select sampling='smote'
    -> call intrinsic_features_smote() / positional_features_smote()
    -> inside those functions, call smote_mask(train_subset_only)
    -> fit decoder on oversampled training data
    -> evaluate on untouched test split
```

For GNN methods, the workflow is:

```text
scripts/train_supervised.py
    -> select graph_smote or graph_ensemble_smote
    -> call GNN_features_graphsmote()
    -> inside that function, call graph_smote_mask() / graph_ensemble_smote_mask()
    -> create synthetic training graph data
    -> train the GNN on the synthetic training portion
    -> evaluate on val_mask and then test_mask
```

### 3.3 Strict isolation proof

The code enforces isolation in three concrete ways:

1. **The original split stays intact**
   - [scripts/train_supervised.py](../scripts/train_supervised.py) creates `train_mask_ratio = train_mask.clone()` and never overwrites `val_mask` or `test_mask`.

2. **Sampling operates on a training-only view**
   - The resampling helpers in [src/methods/evaluation.py](../src/methods/evaluation.py) are fed a masked training subset only.
   - The returned augmented data contains a new mask for synthetic samples, but the original validation and test masks are not modified.

3. **Evaluation uses the separate masks explicitly**
   - [src/methods/experiments_supervised.py](../src/methods/experiments_supervised.py) calls `evaluate_split(val_mask)` and `evaluate_split(test_mask)` in the GNN path.
   - Feature-based methods use the test split directly for final metric calculation.

Therefore, the sampling strategies are limited to the training portion:

- `none`
- `random_undersample`
- `smote`
- `graph_smote`
- `graph_ensemble_smote`

None of them are allowed to replace `val_mask` or `test_mask`.

### 3.4 Classifier architecture

#### Feature-based classifier

The feature-based decoder is a small MLP defined in [src/methods/utils/decoder.py](../src/methods/utils/decoder.py) and called via `Decoder_deep_norm`.

The current training script uses:

- `n_layers_decoder=2`
- `hidden_dim_decoder=16`
- `lr=0.05`
- `n_epochs_decoder=100` for intrinsic methods
- `n_epochs_decoder=50` for positional methods

The loss is `nn.CrossEntropyLoss()`.

#### GNN classifier

The graph models come from [src/methods/utils/GNN.py](../src/methods/utils/GNN.py):

- `GCN`
- `GraphSAGE`
- `GAT`
- `GIN`

The current driver uses a 2-layer GNN with hidden dimension 64 and embedding dimension 32, with dropout 0.3.

The output layer is a linear classifier that produces a 2-class prediction.

---

## 4. Metrics, downstream usage, and persistence

### 4.1 Validation metric usage

In the GNN path, the validation metrics are computed inside [src/methods/experiments_supervised.py](../src/methods/experiments_supervised.py) by `GNN_features()` and `GNN_features_graphsmote()`.

The validation loop does the following:

1. Runs one training epoch on `train_mask`
2. Calls `evaluate_split(val_mask)`
3. Prints training loss, validation loss, and validation AP

The current code does not implement explicit early stopping or checkpointing. The validation numbers are therefore used for **monitoring during training**, not for persistent model selection or checkpoint storage.

### 4.2 Test metric usage

At the end of training, the code explicitly switches to the held-out test split:

- `evaluate_split(test_mask)` inside `GNN_features()`
- `evaluate_split(test_mask)` inside `GNN_features_graphsmote()`

These functions compute:

- AUC-PRC via `average_precision_score()`
- F1 via a percentile-based threshold

For the feature-based methods, the final metric values are computed directly in functions such as:

- `intrinsic_features()`
- `positional_features()`

and returned as `(ap_score, f1)`.

### 4.3 Persistence and overwrite protection

The overwrite protection logic is in [scripts/train_supervised.py](../scripts/train_supervised.py). The main loop:

1. Constructs a result file path under `res/`
2. Checks whether the file already exists
3. If the file exists and is non-empty and contains the expected metric token, it skips computation
4. Otherwise it writes a new content string into the result file

The core logic is:

```python
for rf, res in zip(result_files_list, results):
    rf_path = Path(rf)
    if rf_path.exists():
        print(f"Skip existing result file: {rf_path}")
        continue
    with open(rf_path, "w") as f:
        f.write(res)
```

This is the current anti-overwrite mechanism.

Text-based flowchart:

```text
training run
    -> prepare result filename under res/
    -> check existence + metric token
    -> if valid existing file: skip
    -> else: compute metric and write result file
```

---

## Feedback assessment: what was improved and what remains important

The current code addresses the main concerns raised in the feedback in a materially stronger way:

### Resolved or strongly improved

- **Validation/test leakage is no longer implemented as `val_mask = test_mask`**
  - Verified by a repository search: no such assignment remains.
- **The train/validation/test split is preserved structurally**
  - The dataset loaders create distinct masks, and the training driver keeps them separate.
- **Sampling is constrained to the training portion**
  - The resampling helpers operate on the training subset and do not overwrite the validation/test masks.
- **SMOTE is now implemented through the official imbalanced-learn interface**
  - [src/methods/evaluation.py](../src/methods/evaluation.py) uses `imblearn.over_sampling.SMOTE` rather than the older hand-rolled directionality version.

### Remaining caveats to keep in mind

- **The current feature-based pipeline does not yet implement a separate validation loop**
  - The driver still evaluates the intrinsic/positional functions on the test split rather than a dedicated validation loop.
- **There is no explicit early stopping or best-model checkpointing**
  - Validation metrics are printed, but they are not used to save the best model.
- **The GNN GraphSMOTE/GraphENS path uses a training-only synthetic graph augmentation workflow**
  - The implementation is conceptually consistent, but it is still a training-time augmentation step rather than a full graph-level re-implementation of a separate benchmark pipeline.

Overall, the most important feedback issue—**the leakage between validation and test**—has been corrected in the current codebase.
