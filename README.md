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
- **GraphENS (Graph Ensemble SMOTE)**: Structural and feature-space ensemble over-sampling tailored for GNN class imbalance.

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
