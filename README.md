# Benchmarking Sampling Techniques for Imbalance Graph Learning in Anti-Money Laundering

This repository provides an end-to-end, technically rigorous benchmarking pipeline designed to evaluate various sampling and graph augmentation techniques for class-imbalanced node classification in Anti-Money Laundering (AML).

By implementing strict **temporal splits** (chronological isolation) and a validation-guided **Early Stopping** mechanism, this pipeline prevents data leakage and overfitting, ensuring that reported model performance reflects true generalization capabilities.

---

## 1. Pipeline Architecture & Data Flow

To ensure realistic AML modeling, our pipeline strictly enforces time-series isolation:
1. **Raw State Ingestion**: Features are loaded and engineered with zero front-end resampling or label proportion adjustments.
2. **Temporal Splitting**: Train, validation, and test datasets are partitioned sequentially based on timestamp or time step.
3. **OOS Sampling Isolation**: Sampling algorithms (RUS, SMOTE, GraphSMOTE, GATSMOTE, TNU, GraphENS) operate exclusively on the training subset (`train_mask`). Nearest-neighbor searches and edge attachment for the graph-structural samplers are restricted to train-mask nodes only, so synthetic nodes/edges can never attach to validation or test nodes. Validation and test subsets remain completely untouched to serve as pure evaluation benchmarks.

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
    - GATSMOTE, TNU, GraphENS        |                   |
                 |                   |                   |
                 v                   v                   v
            Train Model -----> Evaluate Val AP -----> Final Evaluation
         (Forward Pass)     (Monitor Overfitting)    (Percentile Cutoff)
                                     |                   |
                                     +---> Early Stop?   v
                                     |    (Patience=10) [Metrics Output]
                                     v                  - AUC-PRC
                              Save best_model.pt        - F1_99 / F1_90
```

---

## 2. Dataset Classification & Partitioning

The pipeline natively supports **five major benchmark datasets**:

| Dataset Name | Source / Type | Feature State & Engineering | Split Mechanism |
| :--- | :--- | :--- | :--- |
| **IBM HI-SMALL** | IBM / Synthetic | Features grouped by sliding window (delta=4h) to form transactional edges; categorical currencies and formats are one-hot encoded; `Day`, `Hour`, `Minute` are extracted. | **Chronological Slicing**: <br>60% Train / 20% Val / 20% Test |
| **IBM HI-MEDIUM** | IBM / Synthetic | Similar to Small; massive scale with higher density. Categorical attributes are one-hot encoded. | **Chronological Slicing**: <br>60% Train / 20% Val / 20% Test |
| **IBM LI-SMALL** | IBM / Synthetic | Lower fraud density synthetic transactions with structured currency columns one-hot encoded. | **Chronological Slicing**: <br>60% Train / 20% Val / 20% Test |
| **IBM LI-MEDIUM** | IBM / Synthetic | Lower density, medium-scale transaction graphs with currency-format dummy variables. | **Chronological Slicing**: <br>60% Train / 20% Val / 20% Test |
| **ELLIPTIC** | Bitcoin / Real-world | Anonymized Bitcoin transactions. Class 2 (unlabeled) is filtered out. Feature matrix is pre-processed using `nan_to_num`. | **Time-Step Based Split**: <br>Train: `time_step < 30`<br>Val: `30 <= time_step < 40`<br>Test: `time_step >= 40` |

Note: for the IBM datasets, `Amount Received` and `Amount Paid` are used as raw, unscaled numeric features (no `StandardScaler`/normalization is applied anywhere in the loading pipeline). Values can range up to roughly `2.8e11`. See "Known Limitations" below for why this matters when tuning loss/regularization hyperparameters.

---

## 3. Supported Methods & Resampling Techniques

### Graph Representational & Feature-based Models (Baselines)
- **Intrinsic Features**: 2-layer MLP decoder running on native transactional features.
- **Positional Features**: Topology-based features (e.g., PageRank, Personalized PageRank) combined with an MLP decoder. Centrality/PageRank features for train/val are computed on a train+val-only induced subgraph (test nodes and their edges are excluded from that computation entirely) to avoid leaking test topology into the features used for training.
- **DeepWalk & Node2Vec**: Random walk-based graph embeddings mapped to a downstream classifier. The downstream classifier uses validation-guided Early Stopping, same as every other method below.
- **Graph Neural Networks (GNNs)**: 2-layer GNN architectures (`hidden_dim=64`, `embedding_dim=32`, `dropout=0.3`) including **GCN**, **GraphSAGE**, **GAT**, and **GIN**.
  - **GCN** and **GAT** are the two architectures that actually consume edge weights during message passing (GCN via `edge_weight`, GAT via `GATv2Conv`'s `edge_dim=1` projection) — this matters for GATSMOTE, whose learned edge probabilities are only ever able to influence the forward pass through these two architectures. GraphSAGE and GIN accept an `edge_attr` argument for a uniform call signature but do not use it, matching their original formulations (Hamilton et al. 2017; Xu et al. 2019).
  - **GraphSAGE** trains with real neighbor-sampling mini-batches (`torch_geometric.loader.NeighborLoader`) rather than a full-graph forward pass, matching its original sample-and-aggregate design. This requires `pyg-lib` or `torch-sparse` to be installed with a working compiled backend; if neither is available, training falls back automatically to full-graph forward passes (with a printed warning) instead of failing the run. Evaluation always runs on the full graph regardless.

### Resampling and Structural Augmentation Techniques
- **NONE**: Training on the original imbalanced split (baseline).
- **RUS**: Random Undersampling of the majority class inside the training mask.
- **SMOTE**: Feature-space synthetic minority over-sampling applied to intrinsic/positional/embedding features using `imblearn`.
- **GraphSMOTE**: Feature-space interpolation combined with k-NN edge attachment restricted to train-mask candidates. This is a non-parametric heuristic inspired by GraphSMOTE (Zhao, Zhang & Wang, WSDM 2021), not the paper's trainable edge-reconstruction decoder.
- **GATSMOTE**: A genuinely trainable `torch.nn.Module` (`GATEdgeGenerator`) implementing Liu et al., *"GATSMOTE"* (Mathematics 2022, DOI 10.3390/math10111799) — see the dedicated section below for the full mechanism and known caveats.
- **Targeted Neighbourhood Undersampling (TNU)**: Graph-aware undersampling that removes noisy/dissimilar majority-class neighbors around minority nodes.
- **GraphENS**: Full implementation of Park, Song & Yang (ICLR 2022), *"GraphENS: Neighbor-Aware Ego Network Synthesis for Class-Imbalanced Node Classification"* — see the dedicated section below.

---

## 4. Training, Validation & Downstream Evaluation

### Early Stopping (Val AP Guided)
To resolve overfitting caused by severe class imbalance, the deep learning pipeline utilizes a dedicated **Early Stopping** class:
* **Metric Monitored**: `val_ap` (Average Precision / AUC-PRC on the validation split). AP is used rather than raw loss, since loss can be trivially minimized by classifying every node as the majority class, whereas AP specifically tracks minority-class precision-recall trade-offs.
* **Patience**: Defaulted to `10` epochs.
* **Checkpointing**: Saves the best-performing model parameter weights to `res/checkpoints/best_model_{method}_{result_tag}.pt`. For GATSMOTE, the classifier and the `GATEdgeGenerator` are checkpointed and restored together (via a single `nn.ModuleList`), so the two never desynchronize at test time. Upon training completion or early termination, the optimal weights are automatically re-loaded prior to evaluating on the test dataset.

### Test Evaluation & Metrics
The final reported figures are evaluated on the untouched test subset using:
- **AUC-PRC** (`average_precision_score`) via scikit-learn.
- **Percentile-Based F1-Score**: Converts predicted probabilities into hard predictions by flagging the top N% most suspicious transactions. Both `F1_99` (top 1%) and `F1_90` (top 10%) are reported from the same prediction pass.

---

## 5. How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Execution Commands

1. **Run a single tuned sweep**
   `scripts/train_supervised_tuned.py` is the entry point; it sweeps every method (`intrinsic`, `positional`, `deepwalk`, `node2vec`, `gcn`, `sage`, `gat`, `gin`) across the full imbalance-ratio grid and every sampling technique applicable to that method, for one seed:
   ```bash
   python scripts/train_supervised_tuned.py --mode auc --network hi_small --seed 42
   ```
   Results are written as individual `.txt` files under `res/tuned/` (one per method / ratio / sampling combination), each containing `AUC-PRC`, `F1_99`, and `F1_90`.

2. **Run the same sweep across multiple seeds**
   To repeat the full sweep across seeds `42`, `123`, and `999` for a given dataset:
   ```bash
   export NETWORK_NAME="hi_small"  # hi_small, hi_medium, li_small, li_medium, or elliptic
   bash scripts/run_supervised_repeated.sh auc
   ```
   This simply invokes `train_supervised_tuned.py` once per seed; it does not need to pre-aggregate anything itself, since the table generator below does that directly from the per-seed result files.

3. **Generate comparison tables (mean +/- std across seeds)**
   Once your experiments under `res/` are complete, compile every result file into consolidated markdown tables:
   ```bash
   python scripts/generate_all_tables.py
   ```
   Result files are grouped by (method, dataset, sampling technique, ratio) and, wherever more than one `--seed` run exists for a given combination, reported as mean +/- sample standard deviation; a combination backed by only one run shows a bare value. This produces:
   * `tables/tuned_only_ratio_comparison_auc_prc.md`
   * `tables/tuned_only_ratio_comparison_f1_99.md`
   * `tables/tuned_only_ratio_comparison_f1_90.md`

---

*This benchmark suite aims for rigorous, leak-free AML modeling designed to evaluate how structural and feature-space resampling impacts model generalizability across diverse scale conditions.*

## Focal Loss, GATSMOTE, and TNU

- **Focal Loss** (`focal`) — a drop-in loss module to focus training on hard minority examples. CLI flags: `--loss focal`, `--focal-gamma` (default `2.0`), `--focal-alpha` (`balanced`, a numeric value, or unset).
- **GATSMOTE** (`gatsmote`) — trainable multi-head attention edge generator; see the dedicated section below for the mechanism. CLI flags: `--gatsmote-k-neighbors` (default `5`), `--gatsmote-heads` (default `4`), `--gatsmote-edge-threshold` (default `0.5`), `--gatsmote-lambda1` (locality-loss weight, default `0.2`), `--gatsmote-lambda2` (shortest-path-loss weight, default `0.05`), `--gatsmote-use-predicted-labels`.
- **Targeted Neighbourhood Undersampling (TNU)** (`tnu` / `targeted_neighbourhood_undersampling`) — graph-aware undersampling that removes noisy/dissimilar majority neighbors around minority nodes. CLI flags: `--tnu-k-neighbors` (default `10`), `--tnu-distance-metric` (`cosine` or `euclidean`, default `cosine`), `--tnu-remove-ratio` (defaults to following the outer imbalance-ratio sweep unless set explicitly), `--tnu-noise-threshold` (default `0.5`), `--tnu-min-majority-keep` (default `1`), `--tnu-preserve-minority-neighbors`.

These methods are implemented with strict train-mask-only operations to avoid label leakage. The sampler canonical names are:

```
none, rus, smote, graph_smote, gatsmote, targeted_neighbourhood_undersampling, graphens
```

`graph_ensemble_smote` remains accepted as an alias for `graphens` (this is the name already used by the default GCN/SAGE/GAT/GIN sweep config).

The tuning script `scripts/train_supervised_tuned.py` includes an expanded imbalance ratio grid and accepts `--loss` and sampler-specific hyperparameters. Example commands are described below.

### Quick example: Focal + GATSMOTE on IBM (extreme imbalance)

```bash
python scripts/train_supervised_tuned.py --network hi_small --loss focal --focal-gamma 2.0 --focal-alpha balanced --gatsmote-k-neighbors 5 --gatsmote-heads 4 --gatsmote-edge-threshold 0.5 --gatsmote-lambda1 0.2 --gatsmote-lambda2 0.05
```

### Note on imbalance ratios
The tuning grid supports: `1:1, 1:2, 1:5, 1:10, 1:20, 1:50, 1:100, 1:200, 1:500, 1:1000, 1:2000`, plus an "Original" (untouched) row. When a ratio is infeasible for a dataset/split, the script will log a warning and skip that configuration.

## GATSMOTE (trainable multi-head attention edge generator)

`gatsmote` implements Liu et al., *"GATSMOTE: Improving Imbalanced Node Classification on Graphs via Attention and Homophily"* (Mathematics 2022, DOI 10.3390/math10111799), as a genuinely trainable `torch.nn.Module` (`GATEdgeGenerator` in `src/methods/samplers.py`) — not a static distance/label heuristic. Synthetic minority nodes are produced once via SMOTE interpolation over the train mask, together with a fixed, train-mask-only k-NN candidate topology per synthetic node; from that point on, the module learns:

- **Multi-head attention** (`W`, `a` per head, `LeakyReLU(0.2)`) over each synthetic node's candidates, exactly matching GAT's scoring function.
- **Head fusion** through a learnable `nn.Linear(heads, 1)`, applied to the raw pre-softmax attention scores (matching the paper's Eq. 8/Algorithm 1 fusion order) rather than to post-softmax weights, then passed through a sigmoid to produce the edge connectivity probability `E^t in [0,1]`.
- **A locality auxiliary loss** (paper Eq. 10 / Hypothesis 1): a bilinear "push to the extremes" loss that pushes `E^t` toward 1 for highly feature-similar pairs and toward 0 for dissimilar pairs, rather than regressing `E^t` toward the graded similarity value itself.
- **A shortest-path auxiliary loss** (paper Eq. 11 / Hypothesis 2): rewards same-label pairs proportionally to the (capped) BFS hop-distance between the candidate node and the synthetic node's SMOTE parent — pairs message-passing can't already reach get pushed toward `E^t = 1`. A mismatched-label penalty is layered on top as a homophily regularizer (not itself part of Eq. 11, but compatible with it since the two apply to disjoint pair subsets).

Both auxiliary losses are added to the node-classification loss every training step (`loss_total = loss_node + lambda1 * loss_locality + lambda2 * loss_shortest`) and optimized jointly with the classifier via a single optimizer over both parameter sets. The classifier and the edge generator are checkpointed together for the same reason described in Section 4 above.

**GraphSAGE-embedding ablation, not the full pipeline**: this implementation operates on raw input node features rather than embeddings from a pretrained GraphSAGE extractor (the paper's primary Algorithm 1 pipeline, lines 2-16). This corresponds to the paper's secondary "raw representations" ablation configuration (Section 4.2.4), and keeps GATSMOTE's inputs consistent with every other sampling technique in this benchmark, which all operate on raw features.

## Known Limitations

- **GATSMOTE's default `--gatsmote-lambda1`/`--gatsmote-lambda2` were tuned on a small, well-scaled synthetic graph**, not on the real IBM/Elliptic data at production scale. A validation attempt against real `hi_small` data produced an unusable result (`loss_node` exploding into the millions) — that explosion is an artifact of the unscaled `Amount Received`/`Amount Paid` features noted in Section 2 (values up to `~2.8e11`), not a property of the loss terms themselves. Revisit these defaults once IBM feature normalization is addressed.
- **The shortest-path auxiliary loss has limited resolution on real, sparse AML graphs.** On the real HI-Small transaction graph (500k nodes, 631k edges), only a small fraction of candidate/synthetic-parent pairs resolve to a genuine hop count within the default `max_hops=4` cap; the graph is highly fragmented (hundreds of thousands of connected components, a large share of them singleton nodes), so most same-label pairs are in a different, unreachable component from their parent rather than merely far away. The mechanism still grades the resolvable minority correctly and remains directionally correct for the rest, but has low resolution among "far" pairs at this scale purely because of how fragmented the underlying transaction graph is, not because the hop cap is mistuned.
- **`GraphSAGE`'s neighbor-sampling mini-batch training requires `pyg-lib` or `torch-sparse`** with a working compiled backend for the current platform. Both are listed in `requirements.txt`, but on platforms without prebuilt wheels for them, training silently falls back to full-graph forward passes for GraphSAGE (a warning is printed when this happens).

## GraphENS (full Algorithm 1 implementation)

`graphens` (alias: `graph_ensemble_smote`, for GCN/SAGE/GAT/GIN only) implements Park, Song & Yang, *"GraphENS: Neighbor-Aware Ego Network Synthesis for Class-Imbalanced Node Classification"* (ICLR 2022), Algorithm 1, in full — not a feature-space SMOTE approximation. Pure algorithm primitives live in `src/methods/graphens.py`; the per-epoch training loop lives in `GNN_features_graphens_with_predictions` (`src/methods/experiments_supervised.py`).

Unlike GraphSMOTE/GATSMOTE/TNU, which sample a static augmented graph once before training starts, GraphENS **resynthesizes its minority ego networks every epoch**:
- Each epoch, minority (`v_minor`) and majority (`v_target`) training nodes are paired, mixed via a saliency-masked feature blend (`lambda ~ Beta(2,2)`, masking driven by the target node's own gradient-based saliency from the previous epoch), and wired to a blended, degree-matched sample of both nodes' real neighbors — added as directed, **incoming-only** edges onto the synthetic node (never symmetrized), per the paper's message-passing footnote.
- The blend ratio `phi_hat = sigmoid(KL(o_minor || o_target))` is provably >= 0.5 (KL divergence is non-negative), so the synthetic node's neighbor/feature distribution never leans more on the target node than on the minority source.
- Saliency and confidence (`o_hat`) are both recomputed at the end of every epoch and reused as next epoch's inputs; saliency reuses that epoch's own backward pass (no extra backward call).
- For the first `--graphens-warmup` epochs (and always for the very first epoch, since there is no prior-epoch state yet), a simpler path runs instead: plain mixup with no saliency masking, and synthetic nodes simply inherit `v_minor`'s real neighbors.

**Confirmed implementation choice on a paper/code discrepancy**: confidence aggregation `o_hat` mean-pools each node's raw logits across its ego network (itself + real neighbors) *then* applies softmax once — matching the official reference implementation (github.com/JoonHyung-Park/GraphENS) rather than Algorithm 1's literal text, which describes softmaxing each neighbor individually before averaging. The two are not equivalent; this repo deliberately follows the authors' validated code over their paper's prose. See the code comment on `graphens.aggregate_confidence` for detail.

CLI params (`scripts/train_supervised_tuned.py`):
- `--graphens-warmup` (default `5`) — epochs of the simple warmup path before switching to the full KL/saliency-blended path (paper tunes among `{1, 5}`).
- `--graphens-mask-k` (default `5.0`) — the `k` in `K = k * phi_hat`, a small integer multiplier giving the **count** of features masked per synthetic node, not a 0-1 fraction (unlike the reference repo's `--keep_prob`; paper tunes among `{1, 5, 10}`).
- `--graphens-pred-temp` (default `1.0`) — temperature applied before the confidence-aggregation softmax (paper tunes among `{1, 2}`).

Evaluation always runs on the plain, unaugmented graph (confirmed against the reference implementation's own `test()`/validation-loss code, which never forwards on its per-epoch synthesized graph either) — only the training graph is resynthesized each epoch.
