# Benchmarking Sampling Techniques for Imbalance Graph Learning in Anti-Money Laundering

This repository provides an end-to-end, technically rigorous benchmarking pipeline designed to evaluate various sampling and graph augmentation techniques for class-imbalanced node classification in Anti-Money Laundering (AML). 

By implementing strict **temporal splits** (chronological isolation) and a validation-guided **Early Stopping** mechanism, this pipeline prevents data leakage and overfitting, ensuring that reported model performance reflects true generalization capabilities.

---

## 📂 1. Pipeline Architecture & Data Flow

To ensure realistic AML modeling, our pipeline strictly enforces time-series isolation:
1. **Raw State Ingestion**: Features are loaded and engineered with zero front-end resampling or label proportion adjustments.
2. **Temporal Splitting**: Train, validation, and test datasets are partitioned sequentially based on timestamp or time step.
3. **OOS Sampling Isolation**: Sampling algorithms (e.g., RUS, SMOTE, GraphSMOTE, GraphENS) operate exclusively on the training subset (`train_mask`). Validation and test subsets remain completely untouched to serve as pure evaluation benchmarks.

```
                  Raw Transaction / Graph Data Ingestion
                                     |
                                     v
                       [ Temporal Split Partitioning ]
                      /              |              \
                     /               |               \
         Train Subset (60%)   Val Subset (20%)   Test Subset (20%)
                 |                   |                   |
                 v                   |                   |
    [ Train-Only Resampling ]        |                   |
    - RUS, SMOTE, GraphSMOTE         |                   |
    - GraphENS, ReweightedGS         |                   |
                 |                   |                   |
                 v                   v                   v
            Train Model -----> Evaluate Val AP -----> Final Evaluation
         (Forward Pass)     (Monitor Overfitting)    (Percentile Cutoff)
                                     |                   |
                                     +---> Early Stop?   v
                                     |    (Patience=10) [Metrics Output]
                                     v                  - AUC-PRC
                              Save best_model.pt        - F1_99
```

---

## 📊 2. Dataset Classification & Partitioning

The pipeline natively supports **five major benchmark datasets**:

| Dataset Name | Source / Type | Feature State & Engineering | Split Mechanism |
| :--- | :--- | :--- | :--- |
| **IBM HI-SMALL** | IBM / Synthetic | Features grouped by sliding window (delta=4h) to form transactional edges; categorical currencies and formats are one-hot encoded; `Day`, `Hour`, `Minute` are extracted. | **Chronological Slicing**: <br>60% Train / 20% Val / 20% Test |
| **IBM HI-MEDIUM** | IBM / Synthetic | Similar to Small; massive scale with higher density. Categorical attributes are one-hot encoded. | **Chronological Slicing**: <br>60% Train / 20% Val / 20% Test |
| **IBM LI-SMALL** | IBM / Synthetic | Lower fraud density synthetic transactions with structured currency columns one-hot encoded. | **Chronological Slicing**: <br>60% Train / 20% Val / 20% Test |
| **IBM LI-MEDIUM** | IBM / Synthetic | Lower density, medium-scale transaction graphs with currency-format dummy variables. | **Chronological Slicing**: <br>60% Train / 20% Val / 20% Test |
| **ELLIPTIC** | Bitcoin / Real-world | Anonymized Bitcoin transactions. Class 2 (unlabeled) is filtered out. Feature matrix is pre-processed using `nan_to_num`. | **Time-Step Based Split**: <br>Train: `time_step < 30`<br>Val: `30 <= time_step < 40`<br>Test: `time_step >= 40` |

---

## ⚙️ 3. Supported Methods & Resampling Techniques

### 🧠 Graph Representational & Feature-based Models (Baselines)
- **Intrinsic Features**: 2-layer MLP decoder running on native transactional features.
- **Positional Features**: Topology-based features (e.g., PageRank, Personalized PageRank) combined with an MLP decoder.
- **DeepWalk & Node2Vec**: Random walk-based graph embeddings mapped to downstream classifiers.
- **Graph Neural Networks (GNNs)**: 2-layer GNN architectures (`hidden_dim=64`, `embedding_dim=32`, `dropout=0.3`) including **GCN**, **GraphSAGE**, **GAT**, and **GIN**.

### 🔄 Resampling and Structural Augmentation Techniques
- **NONE**: Training on the original imbalanced split (baseline).
- **RUS**: Random Undersampling of the majority class inside the training mask.
- **SMOTE**: Feature-space synthetic minority over-sampling applied to intrinsic/positional features using `imblearn`.
- **GraphSMOTE**: Feature-space interpolation combined with heuristic edge reconstruction.
- **Reweighted GraphSMOTE**: GraphSMOTE augmented with continuous cosine-similarity edge weights.
- **Unweighted GraphSMOTE**: A GraphSMOTE variant mechanically identical to GraphSMOTE (SMOTE feature interpolation + unconditional k-NN edge attachment) apart from clip-bound/k-neighbor-clamp details. Formerly mislabeled `graph_ensemble_smote`/GraphENS in this repo; renamed once the real GraphENS was implemented (see below). Not part of the default sweep, but reachable via the `unweighted_graph_smote` sampler name.
- **GraphENS**: Full implementation of Park, Song & Yang (ICLR 2022), *"GraphENS: Neighbor-Aware Ego Network Synthesis for Class-Imbalanced Node Classification"* — see the dedicated section below.

---

## 📈 4. Training, Validation & Downstream Evaluation

### ⏹️ Early Stopping (Val AP Guided)
To resolve overfitting caused by severe class imbalance, the deep learning pipeline utilizes a dedicated **Early Stopping** class:
* **Metric Monitored**: `val_ap` (Average Precision / AUC-PRC on the validation split). *Note: AP is utilized rather than Cross-Entropy Loss, as Loss can easily be minimized by classifying all nodes as the majority class, whereas AP specifically focuses on minority class precision-recall trade-offs.*
* **Patience**: Defaulted to `10` epochs.
* **Checkpointing**: Saves the best-performing model parameter weights to `res/checkpoints/best_model_{method}_{result_tag}.pt`. Upon training completion or early termination, these optimal weights are automatically re-loaded prior to evaluating on the test dataset.

### 🎯 Test Evaluation & Metrics
The final reported figures are evaluated on the untouched test subset using:
- **AUC-PRC** (`average_precision_score`) via scikit-learn.
- **Percentile-Based F1-Score**: Converts predicted probabilities into hard predictions by flagging the top $N\%$ most suspicious transactions (default `percentile_q=99` for top 1% anomaly threshold).

---

## 🚀 5. How to Run

### Install Dependencies
Ensure you have the required GNN and resampling libraries installed:
```bash
pip install torch torch-geometric pandas numpy scikit-learn imbalanced-learn networkx tqdm
```

### Execution Commands

1. **Run a Single Supervised Experiment**
   Execute supervised GNN or tabular classifiers on a chosen dataset:
   ```bash
   python scripts/train_supervised.py --mode auc --network hi_small --seed 42
   ```

2. **Run Repeated Experiments with Multiple Seeds**
   To execute the baseline and sampling pipelines across multiple random seeds (e.g., `42`, `123`, `999`), utilize the bash script:
   ```bash
   export NETWORK_NAME="hi_small"  # Set to hi_small, hi_medium, li_small, li_medium, or elliptic
   bash scripts/run_supervised_repeated.sh auc
   ```

3. **Generate Summary Comparison Markdown Tables**
   Once your experiments under `res/` are completed, compile individual result files into a consolidated markdown matrix for each of the 5 datasets:
   ```bash
   python scripts/generate_all_tables.py
   ```
   This compiles all seeds and average configurations into:
   * `res/ratio_comparison_tables_auc_prc.md`
   * `res/ratio_comparison_tables_f1_99.md`

---

*This benchmark suite guarantees rigorous, leak-free AML modeling designed to evaluate how structural and feature-space resampling impacts model generalizability across diverse scale conditions.*

## 🆕 Additions: Focal Loss, GATSMOTE, and TNU

This update adds support for:

- **Focal Loss** (`focal`) — a drop-in loss module to focus training on hard minority examples. CLI flag: `--loss focal`, with hyperparameters `--focal-gamma` and `--focal-alpha` (supports `balanced` or numeric).
- **GATSMOTE** (`gatsmote`) — graph-aware oversampling that uses neighborhood attention to place edges for synthetic minority nodes. Key CLI params: `--gatsmote-k-neighbors`, `--gatsmote-attention-heads`, `--gatsmote-edge-threshold`, `--gatsmote-homophily-weight`, `--gatsmote-use-predicted-labels`.
- **Targeted Neighbourhood Undersampling (TNU)** (`tnu` / `targeted_neighbourhood_undersampling`) — graph-aware undersampling that removes noisy/dissimilar majority neighbors around minority nodes. CLI params: `--tnu-k-neighbors`, `--tnu-distance-metric`, `--tnu-remove-ratio`, `--tnu-noise-threshold`.

These methods are implemented with strict train-mask-only operations to avoid label leakage. The sampler canonical names are:

```
none, rus, smote, graph_smote, gatsmote, targeted_neighbourhood_undersampling, graphens, unweighted_graph_smote, reweighted_graph_smote
```

`graph_ensemble_smote` remains accepted as an alias for `graphens` (this is the name already used by the default GCN/SAGE/GAT/GIN sweep config).

The tuning script `scripts/train_supervised_tuned.py` includes an expanded imbalance ratio grid and accepts `--loss` and sampler-specific hyperparameters. Example commands are described below.

### Quick example: Focal + GATSMOTE on IBM (extreme imbalance)

```bash
python scripts/train_supervised_tuned.py --network hi_small --loss focal --focal-gamma 2.0 --focal-alpha balanced --gatsmote-k-neighbors 5 --gatsmote-attention-heads 1 --gatsmote-edge-threshold 0.5 --gatsmote-homophily-weight 1.0
```

### Note on imbalance ratios
The tuning grid now supports: `1:1, 1:2, 1:5, 1:10, 1:20, 1:50, 1:100, 1:200, 1:500, 1:1000, 1:2000`. When a ratio is infeasible for a dataset/split, the script will log a warning and skip that configuration.

## 🆕 GraphENS (full Algorithm 1 implementation)

`graphens` (alias: `graph_ensemble_smote`, for GCN/SAGE/GAT/GIN only) implements Park, Song & Yang, *"GraphENS: Neighbor-Aware Ego Network Synthesis for Class-Imbalanced Node Classification"* (ICLR 2022), Algorithm 1, in full — not a feature-space SMOTE approximation. Pure algorithm primitives live in `src/methods/graphens.py`; the per-epoch training loop lives in `GNN_features_graphens_with_predictions` (`src/methods/experiments_supervised.py`).

Unlike GraphSMOTE/GATSMOTE/TNU/Reweighted-GraphSMOTE, which sample a static augmented graph once before training starts, GraphENS **resynthesizes its minority ego networks every epoch**:
- Each epoch, minority (`v_minor`) and majority (`v_target`) training nodes are paired, mixed via a saliency-masked feature blend (`λ ~ Beta(2,2)`, masking driven by the target node's own gradient-based saliency from the previous epoch), and wired to a blended, degree-matched sample of both nodes' real neighbors — added as directed, **incoming-only** edges onto the synthetic node (never symmetrized), per the paper's message-passing footnote.
- The blend ratio `φ̂ = sigmoid(KL(ô_minor ‖ ô_target))` is provably ≥ 0.5 (KL divergence is non-negative), so the synthetic node's neighbor/feature distribution never leans more on the target node than on the minority source.
- Saliency and confidence (`ô`) are both recomputed at the end of every epoch and reused as next epoch's inputs; saliency reuses that epoch's own backward pass (no extra backward call).
- For the first `--graphens-warmup` epochs (and always for the very first epoch, since there is no prior-epoch state yet), a simpler path runs instead: plain mixup with no saliency masking, and synthetic nodes simply inherit `v_minor`'s real neighbors.

**Confirmed implementation choice on a paper/code discrepancy**: confidence aggregation `ô` mean-pools each node's raw logits across its ego network (itself + real neighbors) *then* applies softmax once — matching the official reference implementation (github.com/JoonHyung-Park/GraphENS) rather than Algorithm 1's literal text, which describes softmaxing each neighbor individually before averaging. The two are not equivalent; this repo deliberately follows the authors' validated code over their paper's prose. See the code comment on `graphens.aggregate_confidence` for detail.

CLI params (`scripts/train_supervised_tuned.py`):
- `--graphens-warmup` (default `5`) — epochs of the simple warmup path before switching to the full KL/saliency-blended path (paper tunes among `{1, 5}`).
- `--graphens-mask-k` (default `5.0`) — the `k` in `K = k·φ̂`, a small integer multiplier giving the **count** of features masked per synthetic node, not a 0–1 fraction (unlike the reference repo's `--keep_prob`; paper tunes among `{1, 5, 10}`).
- `--graphens-pred-temp` (default `1.0`) — temperature applied before the confidence-aggregation softmax (paper tunes among `{1, 2}`).

Evaluation always runs on the plain, unaugmented graph (confirmed against the reference implementation's own `test()`/validation-loss code, which never forwards on its per-epoch synthesized graph either) — only the training graph is resynthesized each epoch.
