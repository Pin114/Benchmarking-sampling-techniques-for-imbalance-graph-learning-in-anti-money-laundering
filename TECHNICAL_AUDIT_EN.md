# Technical Audit Report: Graph Machine Learning Codebase for Anti-Money Laundering

*Written against the actual state of the `pinyu` branch, current as of the data-integrity and numerical-stability fixes applied on top of commit `f10c235` (post `65836bc`). This revision supersedes the previous edition of this report: it folds in a second wave of fixes discovered while diagnosing a live training run, several of which are foundational data-pipeline corrections, not just robustness patches. Any prior version of this document that a reader has already consulted for methodology purposes should be treated as superseded — in particular, the previous "Known Limitation" describing unnormalized IBM features (§6.4) is now a closed, fixed issue, and a previously-undetected timestamp-parsing bug affecting three of the five datasets is disclosed here for the first time.*

This report consists of 7 independent sections auditing — at fine granularity — the data pipeline, the split mechanism, leakage risk, sampling-technique theory-vs-implementation, joint training mechanics, classifier architecture and hyperparameters, and tensor-shape/numerical-stability risk, followed by a changelog section documenting exactly which historical results were invalidated and regenerated as a result of this round of fixes. Every claim is cited by file and line number so it can be checked directly against the source.

---

## 1. Data Pipeline and Graph Construction (Fine-Grained View)

### 1.1 Lifecycle: From CSV to `Data` Object

Two fully independent loading paths, managed by [`data/DatasetConstruction.py`](data/DatasetConstruction.py), both ultimately feed the same [`network_AML`](src/utils/Network.py#L16) class.

**(a) Elliptic — `load_elliptic()`** ([DatasetConstruction.py:17-53](data/DatasetConstruction.py#L17-L53))
- Reads three files: `elliptic_txs_features.csv` (column 0 = txId, column 1 = time_step, columns 2-166 = 165 anonymized features), `elliptic_txs_edgelist.csv`, `elliptic_txs_classes.csv`.
- `x = feat_df.loc[:, 'time_step':].values` (L32) treats `time_step` itself as the 0th column of `x`; `scripts/train_supervised_tuned.py` later manually slices it out via `ntw_torch.x[:, 1:94]`, keeping only the 93 "local" features (dropping the 72 "aggregated" feature columns and `time_step` itself).
- Class mapping `{'unknown': 2, '1': 1, '2': 0}` (L36): the original label `'1'` (illicit) maps to `y=1`; `'2'` (licit) maps to `y=0`; `'unknown'` maps to `y=2` (excluded when masks are built).
- **The split is a strict temporal split**: `train: time_step<30`, `val: 30<=t<40`, `test: t>=40`, all further `& (y != 2)` (L47-49).
- Elliptic's raw features are the original paper's own pre-scaled/anonymized values and never go through the timestamp-parsing code path described in (b) below — Elliptic is **unaffected** by the timestamp bug and required no re-run as a result of this audit round.

**(b) IBM family — `load_ibm_config()`** ([DatasetConstruction.py:134-199](data/DatasetConstruction.py#L134-L199))
- Sorted ascending by `Timestamp`, self-transfers dropped (`Account==Account.1`), only the **last** 500,000 rows after sorting are kept (a tail slice, not a random sample).
- Final node feature set: `Amount Received`/`Amount Paid` + `Day`/`Hour`/`Minute` (calendar components only) + three one-hot blocks (Receiving Currency, Payment Currency, Payment Format). **Account numbers and bank codes are dropped entirely** — the model never sees which account/bank a transaction touches, only structural signal via graph topology.
- Edges are built in `preprocess_ibm()` (L75-132): nodes are **transactions themselves** (not accounts). A directed edge A→B is created if transaction A's receiving account equals transaction B's paying account and `0<=Δt<=240min` (modeling money-flow chains).
- **Split**: after time-sorting, the first 60% of rows = train, next 20% = val, last 20% = test — an equivalent temporal split to Elliptic's.

### 1.2 Critical Fix: IBM Timestamp Format Was Silently Corrupting Three of Five Datasets

This is the most consequential finding in this audit round, because it corrupts the **temporal ordering** that the train/val/test split (§1.1b, §2) and the "no future leakage" guarantee both depend on — not merely a downstream numerical artifact.

**Root cause.** The IBM AML dataset release is **not internally consistent in its `Timestamp` string format across files**:
- `HI-Small_Trans.csv` uses `D/M/Y H:MM` (e.g. `"01/09/2022 0:20"`).
- `HI-Medium_Trans.csv`, `LI-Small_Trans.csv`, and `LI-Medium_Trans.csv` all use `Y/M/D HH:MM` (e.g. `"2022/09/01 00:17"`).

The pre-fix code applied a single, file-format-agnostic call, `pd.to_datetime(df_features['Timestamp'], dayfirst=True, errors='coerce')`, to every IBM CSV. `dayfirst=True` is the correct interpretation for HI-Small's format, but is actively **wrong** for the other three files' `Y/M/D` format, and it fails in two distinct, compounding ways depending on the day-of-month value:

```python
pd.to_datetime("2022/09/01 00:17", dayfirst=True)  # -> 2022-01-09 00:17:00  (WRONG: month and day silently swapped)
pd.to_datetime("2022/09/13 09:36", dayfirst=True)  # -> NaT                  (day=13 can't be swapped, so parsing just fails)
```

That is: for every row whose day-of-month is `<=12`, the parser does **not** raise an error or return `NaT` — it silently returns a different, wrong-but-valid `datetime` (month and day transposed). Only rows with day-of-month `>12` fail visibly as `NaT`. Empirically, on `HI-Medium_Trans.csv` (31,898,238 raw rows) this produced **8,818,855 `NaT` rows (~27.6% of the file)** — a magnitude that rules out "a few malformed rows" as an explanation and points directly at a systemic format mismatch; `LI-Medium_Trans.csv` showed a comparable magnitude (8,630,810 dropped rows), while `HI-Small` (correct format for the original code) and the pre-fix `LI-Small` (small file, few days with day-of-month `>12` in its particular 500k-row window) showed 0 and 62 respectively.

Two independent, compounding forms of data corruption followed from this for HI-Medium/LI-Small/LI-Medium:
1. **Silent mis-ordering** of every day-of-month `<=12` row (the majority of affected rows) — these rows do not fail, so they were never visible as an error; they simply scrambled the chronological sort order used to define "past" vs "future" for the train/val/test split.
2. **NaT propagation into derived features**: `NaT` rows' derived `Day`/`Hour`/`Minute` columns become `NaN`. Feeding raw `NaN` into a GNN's message-passing sum-aggregation spreads `NaN` to every node within a few hops of the affected node (any node whose neighborhood includes an `NaN`-featured node inherits `NaN`), and — after the standardization fix in §1.3 — a single `NaN` row in the train split poisons that entire column's mean/std statistic, turning a handful of bad rows into an entire feature column of `NaN` across all 500,000 nodes.
3. Because `sort_values('Timestamp')` runs **before** the final `.iloc[start_index:]` tail-slice that selects "the last 500,000 rows," a corrupted sort order does not just reorder the existing 500k rows — it can select a **different set of transactions entirely** as the final 500k-row dataset, compared to what a correct sort would select. This means the node/edge set itself (not merely feature values or ordering) may have differed between the buggy and corrected pipelines for these three networks.

**Fix** ([DatasetConstruction.py:148-167](data/DatasetConstruction.py#L148-L167)): detect the format per row instead of assuming one dataset-wide convention, and parse each convention with its own explicit, unambiguous format string:
```python
raw_ts = df_features['Timestamp'].astype(str)
is_ymd_format = raw_ts.str.match(r'^\d{4}/')
parsed_ts = pd.Series(pd.NaT, index=df_features.index, dtype='datetime64[ns]')
if is_ymd_format.any():
    parsed_ts.loc[is_ymd_format] = pd.to_datetime(raw_ts[is_ymd_format], format='%Y/%m/%d %H:%M', errors='coerce')
if (~is_ymd_format).any():
    parsed_ts.loc[~is_ymd_format] = pd.to_datetime(raw_ts[~is_ymd_format], dayfirst=True, errors='coerce')
```
Rows still unparseable after this (genuine data-entry artifacts, not format ambiguity) are dropped explicitly and logged (`"Dropped N rows with unparseable Timestamp"`), rather than silently propagating `NaN`/`NaT` downstream.

**Verification.** Re-parsing `HI-Medium_Trans.csv`'s raw `Timestamp` column directly: 0 rows now fail to parse (vs. 8,818,855 before), and the resulting `day` component now correctly spans `[1, 13, ...]` within a sample window rather than being artificially truncated to `<=12` by the swap bug; `min`/`max` timestamps across a sampled window are internally consistent with a monotonically-progressing September 2022 date range. All four IBM configs (`hi_small`, `hi_medium`, `li_small`, `li_medium`) were re-verified end-to-end through `load_ibm_config()` → `get_network_torch()` with `torch.isnan(x).any()`/`torch.isinf(x).any()` both `False`.

**Methodological implication for citation**: any result computed for `hi_medium`, `li_small`, or `li_medium` **prior to this fix** used a train/val/test split whose temporal ordering was partially corrupted, and in some cases a differently-selected 500k-row transaction subset entirely. These are not comparable to, and should not be pooled with, results computed after this fix. `hi_small` and `elliptic` results are unaffected and remain valid across this fix (see §8 for the exact set of invalidated/regenerated result files).

### 1.3 Critical Fix: IBM Feature Standardization (Was Entirely Absent)

**Root cause.** `network_AML.get_network_torch()` ([Network.py:104](src/utils/Network.py#L104), pre-fix) constructed `x` directly from the raw feature DataFrame with **no normalization or scaling of any kind**. `Amount Received`/`Amount Paid` are raw currency magnitudes with no cap — observed values in the IBM data run up to `~1e11`. Feeding such large-magnitude, unnormalized values into an untrained GNN's first linear layer, then summing over potentially hundreds of neighbors via message-passing aggregation, produces activations that can overflow to `inf`/`nan` for some random weight-initialization draws.

This was empirically confirmed to be **initialization-seed-dependent and graph-size-dependent**, not universal: under `--seed 42`, 83%+ of all training epochs logged across `hi_medium`/`li_medium`/`li_small` showed `train_loss=nan` from epoch 1 onward, while the *same* code under `--seed 123`/`--seed 999` on the *same* networks showed 0 NaN epochs, and the smaller `hi_small`/`elliptic` networks were unaffected under any tested seed. This pattern — sound at small scale, silently unstable at larger scale, contingent on the specific random draw — is the signature of an activation-explosion problem rather than an algorithmic bug in any specific sampler.

**A second-order consequence, specific to GraphENS** (`sampling="graph_ensemble_smote"`): its per-epoch confidence aggregation (`aggregate_confidence`, §4.4) takes a `softmax` of the live model's raw logits; when those logits are already `inf`/`nan` from the upstream explosion, the softmax output collapses to `nan`, which propagates through `compute_mixing_ratio`'s KL divergence into `phi_hat`, and finally into `blended_neighbor_sampling`'s per-candidate weight vector — which `torch.multinomial` refuses outright (`RuntimeError: probability tensor contains either inf, nan or element < 0`), turning a *pipeline-wide* NaN-loss numerical failure into a *hard crash specific to GraphENS*, while every other sampling technique on the same broken data silently produced a "successful" result file with a meaningless near-zero AUC-PRC (e.g. `AUC-PRC: 0.00123`, constant `val_ap` across all 50 epochs) instead of crashing. **This means the crash was the visible symptom of a much larger, silent problem: every non-GraphENS GNN result computed under the same broken (seed, network) combination is equally suspect, whether or not it crashed.**

**Fix** ([Network.py:104-127](src/utils/Network.py#L104-L127)): per-feature standardization (zero mean, unit variance), with mean/std computed **exclusively from the training-split rows** and then applied to train, val, and test alike — avoiding val/test distributional leakage into the scale used for GNN input, consistent with the leakage-avoidance discipline already used elsewhere in this codebase (§3):
```python
train_mask_bool = self.train_mask.bool() if torch.is_tensor(self.train_mask) else torch.tensor(np.array(self.train_mask, dtype=bool))
x_train = x[train_mask_bool]
mean = x_train.mean(dim=0, keepdim=True)
std = x_train.std(dim=0, keepdim=True)
std = torch.where(std < 1e-8, torch.ones_like(std), std)   # zero-variance columns map to 0, not div-by-zero
x = (x - mean) / std
```
The `std < 1e-8` guard exists because several feature columns are one-hot indicator dummies (e.g. `Payment Format_Wire`); a column that is constant within the training split would otherwise divide by zero.

**Verification**: a synthetic test combining a large-magnitude column (`1e9`–`1e11` range) with a zero-variance constant column confirmed: (a) no `NaN`/`Inf` in the output, (b) train-subset mean ≈ 0 / std ≈ 1 after transform, (c) the constant column maps to exactly 0 rather than raising a division error. Against real data, all four IBM networks' `x` tensors were confirmed `NaN`/`Inf`-free after both this fix and the §1.2 timestamp fix were applied together (the timestamp fix was necessary *in addition to* standardization — standardization alone does not fix genuinely-missing raw values, since a single `NaN` train-split value poisons that column's mean/std for the whole dataset; see §1.2's point 2).

This closes the "unnormalized IBM features" item that appeared as a **Known Limitation** in the previous edition of this report (previously §6.4, item 1) — it is no longer a limitation, it is a fixed defect, with the caveat that any pre-fix `hi_medium`/`li_small`/`li_medium` GNN-family result should be treated as invalid regardless of whether it happened to crash (see §8).

### 1.4 Robustness Fix: Eigenvector-Centrality Non-Convergence on Fragmented Graphs

**Root cause.** `eigenvector_nx()` ([functionsNetworKit.py](src/methods/utils/functionsNetworKit.py), part of the `positional` method's structural feature set) called `nx.eigenvector_centrality(G_nx, max_iter=1000)` — networkx's power-iteration solver — with no fallback. Power iteration is prone to non-convergence on graphs with many small, weakly-connected components, which is exactly the topology of the IBM transaction-chain graphs (money-flow edges tend to form short, largely disjoint chains rather than one large, well-connected component). This raised an uncaught `networkx.exception.PowerIterationFailedConvergence`, which — although safely caught by the outer per-configuration `try/except` in `train_supervised_tuned.py` (see §8, "fail-safe, not fail-silent") — resulted in **0 successful `positional`-method result files for every IBM dataset attempted**, while Elliptic's `positional` results (a more well-connected graph topology) were 100% complete.

**Fix** ([functionsNetworKit.py:29-67](src/methods/utils/functionsNetworKit.py#L29-L67)): fall back from whole-graph power iteration to **per-connected-component** direct eigenvalue solving:
```python
def eigenvector_nx(G_nx):
    try:
        eigen_full = nx.eigenvector_centrality(G_nx, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigen_full = _eigenvector_per_component(G_nx)
    ...
```
`_eigenvector_per_component` (L29-50) is deliberately **not** a naive whole-graph fallback to `nx.eigenvector_centrality_numpy` — that function raises `nx.AmbiguousSolution` outright on any disconnected graph, which the IBM graphs always are, making it a dead-end fallback that would degrade the entire feature to a constant 0 across every node. Solving component-by-component instead recovers a real, non-degenerate signal within each component: `nx.eigenvector_centrality_numpy(subgraph)` is well-posed on a connected subgraph (a genuine eigenvalue solve, not subject to power-iteration divergence), isolated single-node components get the standard-convention value `0.0` (a lone node has no meaningful eigenvector-centrality interpretation), and if a component's solve fails for any other reason its nodes fall back to `0.0` individually rather than aborting the whole computation.

**Verification**: on a synthetic disconnected graph (two triangles + a path + an isolated node), forcing the primary solver to fail confirmed the fallback produces correct, non-degenerate, non-uniform scores (e.g. the path's middle node scored `0.707` vs. `0.5` for its endpoints — the expected centrality ordering) with zero `NaN`s and no crash.

### 1.5 Scalability Fix: Landmark-Sampled Closeness Centrality

**Root cause.** `closeness_nx()` ([functionsNetworKit.py](src/methods/utils/functionsNetworKit.py), also part of `positional`'s structural feature set) called `nx.closeness_centrality(G_nx)` with **no sampling of any kind** — unlike `betweenness_nx` in the same file, which already accepts a landmark-count parameter `k` (default `500`) precisely to avoid this problem. Exact closeness centrality requires a full breadth-first search from *every* node, `O(V·(V+E))` total work. On a real IBM graph (500,000 nodes), this was diagnosed, in production, as the direct cause of the host machine being driven into severe swap (96%+ of an 18GB-RAM machine's swap capacity in use, with training processes observed making literally 0% CPU progress for over an hour) — not a crash, but a silent, near-total throughput collapse that is far harder to detect than an exception.

**Fix** ([functionsNetworKit.py:20-67](src/methods/utils/functionsNetworKit.py#L20-L67)): decompose by connected component (mirroring §1.4's eigenvector fix), and apply exact computation only where it is already cheap:
```python
for component_nodes in nx.connected_components(G_nx):
    if len(component_nodes) <= 1:
        closeness_full[node] = 0.0                       # singleton: no meaningful closeness
    elif len(component_nodes) <= k:
        closeness_full.update(nx.closeness_centrality(subgraph))   # small component: exact, cheap anyway
    else:
        landmarks = rng.sample(component_nodes, k)        # large component: k-landmark approximation
        # accumulate sum of shortest-path distances from each landmark via single_source_shortest_path_length,
        # then estimate closeness[v] = (comp_size-1) / ((comp_size/k) * sum_of_landmark_distances[v])
```
This is the standard Eppstein–Wang-style landmark approximation to closeness centrality, applied only where the exact computation is actually expensive — IBM's transaction graphs fragment into hundreds of thousands of components (§4.3 measured 374k components on HI-Small, 351k of them singletons), so the overwhelming majority of nodes are computed exactly anyway (their component is far smaller than `k=500`); only the rare large component(s) — precisely the ones responsible for the `O(V·(V+E))` blowup — are approximated. This reduces the dominant cost from `O(V)` full-graph BFS traversals to `O(k)`, independent of how large the graph's biggest component is.

**Verification**: (a) on a graph where every component is smaller than `k`, output is bit-for-bit identical to `nx.closeness_centrality` (confirmed on `networkx.karate_club_graph()`, max error `0.0`); (b) forcing the sampled path on a 3,000-node random graph (`k=200`) against the exact ground truth gave mean absolute error `0.00206`, max `0.01349` — small and consistent with expected landmark-sampling variance; (c) a 50,000-node random graph completed in under 3 minutes even under concurrent CPU contention from other running training jobs, where the previous exact implementation was the direct cause of a real production hang at 10x that node count. This closes what was, in the previous edit of this section, disclosed as an open operational limitation (§6.4) — it is now a fixed scalability defect, not a scheduling workaround.

### 1.6 Global Topology: GCN/GAT/GIN Do a Full-Graph Forward Pass; GraphSAGE Has Real Neighbor Sampling

The current forward-pass mechanism for each of the four architectures ([`src/methods/utils/GNN.py`](src/methods/utils/GNN.py)):

| Architecture | Training-time graph scope | edge_attr/edge_weight |
|---|---|---|
| **GCN** (L29-77) | Single full-graph forward pass | Actually used: `edge_weight = edge_attr.view(-1).to(torch.float32)` (L63-65) |
| **GraphSAGE** (L79-136) | **Real neighbor-sampling mini-batches** (see below) | Not used (architecturally has no such concept — not a bug) |
| **GAT** (L139-196) | Single full-graph forward pass | Actually used: `GATv2Conv(..., edge_dim=1)` (L163-171); `edge_attr` is reshaped to `[-1,1]` (L179) before genuinely entering the attention computation |
| **GIN** (L198-268) | Single full-graph forward pass | Not used (the original GIN formulation has no edge-feature concept — this is exactly why `GINE` exists, L275+, unused by this benchmark) |

**GraphSAGE's neighbor sampling** ([experiments_supervised.py:70-101](src/methods/experiments_supervised.py#L70-L101)):
```python
use_neighbor_sampling = isinstance(model, GraphSAGE)
if use_neighbor_sampling:
    num_neighbors = sage_num_neighbors or [15] * max(int(model.n_layers), 1)
    train_loader = _try_build_neighbor_loader(ntw_torch, train_mask_sampled, sage_batch_size, num_neighbors, shuffle=True)
    use_neighbor_sampling = train_loader is not None
```
`_try_build_neighbor_loader` pulls one batch as a probe: if the current environment is missing the `pyg-lib`/`torch-sparse` compiled backend that `NeighborLoader`'s sampler needs, it prints a warning and gracefully falls back to a full-graph forward pass instead of failing the entire experiment because of a missing dependency (confirmed to trigger in the local sandbox: `ImportError: 'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'`). Evaluation always runs on the full graph regardless.

**Professional commentary**: cleanly separating "mini-batch sampling for training, full graph for evaluation" is a well-established and defensible tradeoff in the literature. GCN/GAT/GIN currently have no extension point for mini-batch training on large graphs — this remains a scalability gap worth noting for future work, now compounded by the fact that these architectures' exact (unsampled) structural feature dependencies (§1.4) are themselves memory- and compute-intensive at IBM's node counts (see §8's operational notes on concurrency).

---

## 2. Data Split Mechanism and Train-Evaluation Alignment

### 2.1 The Exact Implementation

Both datasets use a **temporal split** (see §1.1), not a random one. Masks are stored as `torch.bool` tensors on `Data.train_mask/val_mask/test_mask`, with length equal to the total node count and indices corresponding directly to node IDs. As established in §1.2, the correctness of this split for `hi_medium`/`li_small`/`li_medium` was compromised prior to this audit round by the timestamp-parsing bug; it is now correct for all five datasets.

### 2.2 Line-by-Line Application of Masks in Train/Eval Loops

Using `GNN_features` ([experiments_supervised.py:704-824](src/methods/experiments_supervised.py#L704-L824)) as an example:
```python
train_mask_sampled = train_mask.bool().to(device)
def train_epoch():
    out, _ = _forward(ntw_torch.x.to(device), ntw_torch.edge_index.to(device))  # full-graph forward
    y_train = y[train_mask_sampled]
    loss_val = _compute_loss(criterion, out[train_mask_sampled], y_train)
```
Every epoch, the GNN performs one full-graph forward pass (including val/test) — message passing lets val/test node features participate in the aggregation for train node embeddings, an unavoidable consequence of transductive learning (using val/test nodes structurally is fine, as long as their **labels** never enter the loss). Loss computation slices explicitly here — only nodes where `train_mask` is true enter the gradient.

`evaluate_split(mask)` runs another full-graph forward pass under `torch.no_grad()`, using the same slicing technique to compute `val_ap`/`test_f1`. The epoch loop only triggers `early_stopping` on `val_mask`; `test_mask` is touched exactly once, after training ends and the val-best checkpoint has been reloaded.

`GNN_features_graphsmote(_with_predictions)` ([L1009-1255](src/methods/experiments_supervised.py#L1009-L1255)) symmetrically pads `val_mask`/`test_mask` when synthetic nodes exist, using identical code on adjacent lines for both.

### 2.3 Checkpoint-Path Isolation Fix (Concurrency Correctness)

**Root cause.** `intrinsic_features_with_predictions` and `positional_features_with_predictions` both default their `checkpoint_path` parameter to a **fixed, non-parameterized** path (`"res/checkpoints/best_model_intrinsic_tuned.pt"` and `"res/checkpoints/best_model_pos_tuned.pt"` respectively), and — unlike the `node2vec`/`deepwalk`/GNN call sites, which already passed a `unique_checkpoint_path` embedding the network/ratio/sampling/seed identity — `train_supervised_tuned.py` called both without overriding this default. Running more than one network/seed combination **concurrently** (a standard practice for this benchmark's sweep, given the number of (method × sampling × ratio × seed) configurations) meant every concurrently-running `intrinsic`/`positional` job shared the exact same on-disk checkpoint file: one process's `torch.save` and another's `torch.load` could interleave, silently reloading the wrong network's best-validation weights for early stopping.

**Fix** ([train_supervised_tuned.py:211,215](scripts/train_supervised_tuned.py#L211)): both call sites now pass `checkpoint_path=unique_checkpoint_path`, the same per-(network, ratio, sampling, seed) unique path already used everywhere else in the sweep.

**Methodological implication**: any `intrinsic`/`positional` result generated while two or more networks were being swept **concurrently**, prior to this fix, carries a nonzero risk of early-stopping having reloaded a different run's checkpoint. This is a latent risk rather than a confirmed corruption in the currently-retained pre-existing result files (concurrent execution of `intrinsic`/`positional` jobs across different networks was not the dominant sweep pattern used to produce them), but it is now closed going forward, and is disclosed here for completeness since it directly affects the validity of any checkpoint-dependent early-stopping result under concurrent execution.

---

## 3. Comprehensive Data-Leakage Vulnerability Audit

### 3.1 Positional/Structural Feature Leakage — Clean, and Actively Hardened

Tracing line by line ([L360-487](src/methods/experiments_supervised.py#L360-L487)):
```python
train_val_mask = train_mask.bool() | val_mask.bool()
train_val_nodes = set(torch.where(train_val_mask)[0].tolist())
ntw_nx_train_val = ntw_nx_full.subgraph(list(train_val_nodes))    # induced subgraph -- test nodes/edges do not exist here
features_nx_df_train_val = local_features_nx(ntw_nx_train_val, ...)
features_nk_df_train_val = features_nk(ntw_nx_train_val, ...)
```
`nx.pagerank`, `nx.betweenness_centrality`/`closeness_centrality`/`eigenvector_centrality` physically cannot see any test node or any edge touching a test node when computing "features for training." The eigenvector-centrality fallback introduced in §1.4 preserves this isolation exactly — the per-component fallback operates on whatever subgraph (`train_val` or `full`) it was called with, and introduces no new cross-split visibility.

`features_df_full` (the full-graph version) is used only to extract `X_test` and never enters `loss_val.backward()`.

**One detail worth recording, though not a bug**: the train feature subgraph mixes in val-node topology (`train_val_mask = train|val`) — train nodes' centrality values are mildly influenced by the presence of val nodes, but only structurally, never through val **labels**.

### 3.2 Feature Standardization Introduces No New Leakage

The §1.3 standardization fix computes mean/std **exclusively** from `self.train_mask`-selected rows (`x_train = x[train_mask_bool]`) before applying the resulting affine transform to the entire feature matrix (train, val, and test alike). This is the same discipline already used for the k-NN candidate pools in `graph_smote_mask`/`GATEdgeGenerator`/`TargetedNeighbourhoodUndersampling` (§3.3) — val/test **distributional** information (their raw magnitude range) never influences the scale train nodes are represented in, only the reverse (val/test nodes are represented on a scale fixed entirely by the training distribution, the standard and correct convention for any train/val/test pipeline with feature scaling).

### 3.3 Resampling/SMOTE Leakage — Fixed and Kept Consistent

Current state ([evaluation.py:198-296](src/methods/evaluation.py#L198-L296)):
```python
nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, features_masked.shape[0]), algorithm='ball_tree').fit(features_masked)
...
for neighbor_idx in indices[0][1:]:
    neighbor_global_idx = int(idx_mask[neighbor_idx])   # remapped back to global index
```
`features_masked = features_np[idx_mask]` — the k-NN candidate pool is strictly restricted to train-mask nodes, consistent with `GATEdgeGenerator.prepare_synthetic_nodes` ([samplers.py:159-195](src/methods/samplers.py#L159-L195)) and `TargetedNeighbourhoodUndersampling.__call__`.

`GraphENS` ([graphens.py](src/methods/graphens.py)) builds its adjacency list from the real, full-graph edges; `minor_pool`/`target_pool` are restricted to train, but a train minority node's real neighbors can be val/test nodes. The new edges are deliberately **directional** (`neighbor -> synthetic`, never symmetrized) — this only lets val/test **features** flow one-way into the synthetic node, never back into the val/test node's own embedding. This is the same class of phenomenon as transductive message-passing generally, not a leak newly introduced by GraphENS.

---

## 4. Sampling Techniques: Theory vs. Actual Code Implementation

### 4.1 Basic SMOTE / RUS
`smote_mask` ([evaluation.py:116-196](src/methods/evaluation.py#L116-L196)) directly calls `imblearn.over_sampling.SMOTE.fit_resample`. `random_undersample_mask` (L66-114) uses `RandomState.choice` for sampling-without-replacement.

### 4.2 Vanilla Graph SMOTE
- **Theory** (Zhao, Zhang & Wang, WSDM 2021): a trainable edge generator/decoder, jointly trained with the classifier.
- **Implementation** (`graph_smote_mask`): a one-shot, non-parametric k-NN heuristic (now correctly restricted to train), weight fixed at 1. Captures the spirit of "interpolate + attach edges" but not the paper's trainable decoder mechanism.

### 4.3 GATSMOTE — A Genuinely Trainable PyTorch Module

`samplers.py`'s `GATEdgeGenerator(nn.Module)` ([L16-395](src/methods/samplers.py#L16-L395)) implements Liu et al., *GATSMOTE* (Mathematics 2022, DOI 10.3390/math10111799):

- **Trainable weight matrices/attention vectors**: `self.W = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(heads)])`, `self.a = nn.ModuleList([nn.Linear(2*hidden_dim, 1, bias=False) for _ in range(heads)])`.
- **Multi-head attention** (`forward`, [L260-361](src/methods/samplers.py#L260-L361)): `e = LeakyReLU(0.2)(a_h([Wz_i‖Wz_j]))`, `alpha = segment_softmax(e, dst_idx)`.
- **Head fusion, matching the paper**: raw pre-softmax scores `e^{tk}` are fused via a trainable `nn.Linear(heads,1)` *before* the softmax normalization step (paper Eq. 8/Algorithm 1).
- **Locality auxiliary loss** (paper Eq. 10 / Hypothesis 1):
  $$\mathcal{L}_{\text{locality}} = -\text{mean}\big(2 \cdot E^t \cdot (\text{sim}_{\cos} - 0.5)\big)$$
  A bilinear "push to extremes" design: highly-similar pairs are pushed toward `E^t→1`, dissimilar pairs toward `E^t→0`.
- **Shortest-path auxiliary loss** (paper Eq. 11 / Hypothesis 2): capped BFS (`_bounded_hop_distance`, `max_hops=4`) from the synthetic node's SMOTE "parent," rewarding same-label pairs proportionally to hop distance.

**Known limitation, now partially re-contextualized by §1.3's fix**: the previous edition of this report attributed a validation attempt's `loss_node` exploding into the millions on real `hi_small` data to the unnormalized `Amount Received`/`Amount Paid` features (up to `~2.8e11`). That root cause is now fixed (§1.3); `--gatsmote-lambda1`/`--gatsmote-lambda2`'s default values, however, were still tuned on a small, well-scaled synthetic graph and have not been re-validated at real production scale **under the corrected, standardized feature pipeline** — this re-validation is still outstanding and should be performed before treating the current defaults as tuned for the AML datasets specifically.

**Known limitation, independent of the above**: measured on the real HI-Small graph (500k nodes, 631k edges), only 0.33% of pairs resolve to a genuine hop count within the `max_hops=4` cap; 99.11% of *all* pairs are in a **different connected component** from their parent (374k connected components, 351k of them singletons) — unreachable at any hop count. The mechanism remains directionally correct but has low resolution among "far" pairs at this scale, a consequence of the transaction graph's fragmentation, not a mistuned hop cap.

### 4.4 GraphENS / Graph Ensemble SMOTE — Real Implementation, Now Numerically Hardened

`sampling="graph_ensemble_smote"` implements GraphENS (Park, Song & Yang, ICLR 2022, Algorithm 1), composed of [`graphens.py`](src/methods/graphens.py) (pure functions) and `GNN_features_graphens_with_predictions` ([experiments_supervised.py:1256 onward](src/methods/experiments_supervised.py#L1256)):

- **Degree-distribution alignment**: `graphens.sample_augmented_degree` samples from the real graph's degree histogram, capped at `deg(v_minor)`.
- **Blended ego-network sampling**: `blended_neighbor_sampling` implements Eq. 1, `p(u|v_mixed)=φ̂·p(u|v_minor)+(1-φ̂)·p(u|v_target)`, sampling without replacement.
- **KL mixing ratio φ̂**, **saliency-masked mixup**, **confidence aggregation ô** (mean-then-softmax, a deliberate choice to follow the reference implementation over the paper's literal wording) — all implemented, no unreached dead code.

**New fix: numerical-stability guard on `blended_neighbor_sampling`** ([graphens.py:214-271](src/methods/graphens.py#L214-L271)). As detailed in §1.3, `phi_hat` (and hence this function's sampling weights) is derived from the live model's per-epoch confidence, which can become `NaN` if upstream logits explode. `torch.multinomial` raises a hard, uncatchable-by-design `RuntimeError` on any non-finite or negative probability entry — previously an unconditional crash. The fix adds a defensive fallback:
```python
if not torch.isfinite(weights).all() or (weights < 0).any():
    weights = torch.ones_like(weights)   # fall back to a uniform draw over candidate neighbors
if weights.sum() <= 0:
    return torch.empty(0, dtype=torch.long)
```
This does not, by itself, fix a genuinely unstable upstream model (that is §1.3's job) — it ensures that GraphENS specifically never crashes the entire sweep configuration merely because it is the one sampling technique that routes model confidence through a strict-input-domain PyTorch primitive, while every other sampling technique on the same unstable data would otherwise silently produce a garbage-but-"successful" result file instead. With §1.3's fix in place, this guard is not expected to trigger under normal operation on the corrected pipeline; it remains as defense-in-depth. Verified via direct unit test: passing `phi_hat=nan` no longer raises, and returns a valid (though non-meaningful, since the underlying signal was itself non-meaningful) sample of candidate node IDs.

### 4.5 Targeted Neighbourhood Undersampling (TNU)

**Logic for flagging noisy nodes** ([samplers.py:409 onward](src/methods/samplers.py#L409)): for every minority-class node, find its k nearest neighbors via `NearestNeighbors(metric='cosine')`; if a neighbor is majority-class **and** its cosine distance `>= noise_threshold`, flag it as a removal candidate.

**CLI parameter wiring**: `effective_remove_ratio = tnu_remove_ratio if tnu_remove_ratio is not None else ratio` ([experiments_supervised.py:986-987](src/methods/experiments_supervised.py#L986-L987)); all six `--tnu-*` flags are threaded through `_build_graphsmote_sampling` into the constructor.

### 4.6 "Original" (ratio=None) Semantic Consistency Across Methods

`_build_graphsmote_sampling` ([L946-1007](src/methods/experiments_supervised.py#L946-L1007)) places `if ratio is None:` at the front of its dispatch chain, making "Original" mean "no resampling" uniformly across `graph_smote`/`gatsmote`/`tnu`, consistent with `GNN_features`/`intrinsic_features`'s pre-existing semantics.

---

## 5. Joint Training and Optimization Mechanics

`GATEdgeGenerator` is the only sampler in this codebase with trainable parameters of its own; `GNN_features_graphsmote(_with_predictions)` ([L1009-1255](src/methods/experiments_supervised.py#L1009-L1255)) wires up a complete joint-training pipeline:

**Optimizer**:
```python
joint_params = list(model.parameters()) + list(gat_edge_gen.parameters()) if gat_edge_gen is not None else list(model.parameters())
optimizer = torch.optim.Adam(joint_params, lr=lr, weight_decay=5e-4)
```

**Total loss formula per epoch step**:
```python
loss_val = loss_node + gat_edge_gen.lambda_locality * loss_locality + gat_edge_gen.lambda_shortest * loss_shortest
```
$\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{node}}+\lambda_1\cdot\mathcal{L}_{\text{locality}}+\lambda_2\cdot\mathcal{L}_{\text{shortest}}$, `--gatsmote-lambda1` (default `0.2`), `--gatsmote-lambda2` (default `0.05`). Edge probabilities are recomputed every epoch from the current attention parameters (`build_epoch_graph`); only the synthetic candidate topology is fixed once before training starts.

**Early Stopping and Checkpoint Synchronization**:
```python
checkpoint_target = nn.ModuleList([model, gat_edge_gen]) if gat_edge_gen is not None else model
early_stopping(metric_to_monitor, checkpoint_target)
checkpoint_target.load_state_dict(torch.load(checkpoint_path, ...))
```
Wrapping both classifier and edge-generator in a single `nn.ModuleList` for checkpointing prevents the two from desynchronizing (classifier at its val-best epoch, edge generator at whatever epoch training happened to end on).

**Contrast with GraphENS**: its optimizer only contains `model.parameters()` — GraphENS has no trainable parameters of its own; the saliency/confidence statistics it uses are closure state computed from that epoch's classifier gradients, not parameters being optimized.

---

## 6. Classifier Architecture and Hyperparameter Configuration

### 6.1 Structural Breakdown

| | GCN | GraphSAGE | GAT | GIN |
|---|---|---|---|---|
| Layers | 2 | 2 | 2 | 2 |
| hidden_dim | 64 | 64 | 64 (lightweight mode: 16) | 64 |
| embedding_dim | 32 | 32 | 32 (lightweight mode: 8) | 32 |
| output_dim | 2 | 2 | 2 | 2 |
| heads | — | — | 4 (lightweight mode: 1) | — |
| dropout_rate | 0.3 | 0.3 | 0.3 (lightweight mode: 0.2) | 0.3 |
| Activation | `F.relu` | `F.relu` | `F.relu` | built into the GIN layer's `ReLU` |
| Output head | `Decoder_linear` (single linear layer) | same | same | same |

"Lightweight mode" (`use_lightweight_gat = num_nodes >= 300000`) only affects GAT, and also drops `gnn_epochs` from 50 to 10.

### 6.2 edge_attr Handling

GAT is the only architecture that genuinely re-weights edge information via attention. GraphSAGE/GIN's lack of support is an inherent property of their original formulations, not an implementation gap.

### 6.3 Full CLI Hyperparameter Inventory (`train_supervised_tuned.py`)

**Imbalance-ratio sweep grid** ([train_supervised_tuned.py:92](scripts/train_supervised_tuned.py#L92), revised this round):
```python
test_ratios = [None, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0]
```
The `200.0`/`ratio_1to200` and `2000.0`/`ratio_1to2000` points were removed from the sweep grid (a scope-reduction decision, not a bug fix) — any result table referencing these two ratio points reflects a prior, wider grid and should not be expected to appear in newly-generated tables.

Semantics: `target_minority_count = round(majority_count / ratio)`. `ratio=None` (Original) uniformly means "no resampling" across every sampling-technique family (§4.6).

**Gradient clipping**: `--clip_norm` (default `1.0`), genuinely threaded into every `GNN_features*` function.

**Loss function formulas**:
- **Cross-Entropy**: `nn.CrossEntropyLoss(weight=[1.0, pos_weight])`, `pos_weight = num_neg/max(num_pos,1)`.
- **Focal Loss** ([losses.py:39-79](src/methods/losses.py#L39-L79)): $FL(p_t)=-\alpha_t(1-p_t)^{\gamma}\log(p_t)$, `gamma=2.0` default, `alpha` supports `None`/`'balanced'`/scalar/list.

**GATSMOTE CLI**: `--gatsmote-k-neighbors` (5), `--gatsmote-heads` (4), `--gatsmote-edge-threshold` (0.5), `--gatsmote-lambda1` (0.2), `--gatsmote-lambda2` (0.05), `--gatsmote-use-predicted-labels`.

**TNU/GraphENS CLI**: see §4.5 and the README.

### 6.4 Known Limitations (Current, Post-Fix State)

1. **GATSMOTE's `lambda1`/`lambda2` defaults require re-validation at real scale under the now-corrected, standardized feature pipeline** (§4.3) — the original explosion that motivated flagging this is fixed, but the defaults themselves were never re-tuned against the fix.
2. **The shortest-path auxiliary loss has limited resolution on real, sparse AML graphs** due to graph fragmentation (§4.3), independent of any bug.
3. **GraphSAGE's neighbor sampling requires `pyg-lib` or `torch-sparse`** with a working compiled backend; falls back to full-graph training with a printed warning when unavailable.
4. ~~Exact (unsampled) `closeness_centrality` on 500k-node IBM graphs is memory- and compute-intensive~~ — **fixed** (§1.5): now landmark-sampled per-component, matching `betweenness_centrality`'s existing `k`-sampling convention. Retained here as a historical note because it was the direct cause of a real production incident during this audit round (see §8.3) before the fix.
5. **Running multiple large-graph (`hi_medium`/`li_small`/`li_medium`) sweeps fully concurrently is still not recommended** even after §1.5's fix, simply due to this development machine's limited total RAM (18GB) relative to holding multiple full `networkx` graph objects, PyTorch tensors, and pandas DataFrames in memory at once — sequential execution of these three networks (§8.3) remains the safer default regardless of the closeness-centrality fix specifically.

---

## 7. Feature-Space Mismatches and Array Truncation Risk

### 7.1 Shape Changes After Appending Synthetic Nodes (SMOTE Family)

`graph_smote_mask` and `GATEdgeGenerator.prepare_synthetic_nodes` both append synthetic nodes to the tail of the original feature matrix; `expanded_mask[:N]` keeps the original mask, `expanded_mask[N:]=True`.

### 7.2 `X_val`/`X_test` Slicing Logic

**Strategy A (graph-structural methods)**: padding, with `val_mask_smote`/`test_mask_smote` built identically. **Strategy B (pure feature-space methods)**: truncation back to original length `N` before masking, applied symmetrically to `X_val` and `X_test`.

### 7.3 Shape Safety in `GATEdgeGenerator.build_epoch_graph`

When zero synthetic edges clear the attention threshold, `torch.cat` between empty and non-empty tensors is well-defined; classification loss and the two auxiliary losses (computed over the full candidate set regardless of the threshold filter) continue to supply gradient signal.

### 7.4 Shape Safety in Multi-Seed Aggregation

`scripts/generate_all_tables.py` stores each cell as a `{seed_key: value}` dict; `aggregate_cell()` computes mean/sample-stdev (ddof=1, n>=2) or a bare value (n=1). Verified against synthetic 3-seed inputs and regenerated against real data.

---

## 8. Changelog: What Was Invalidated and Regenerated This Round

This section exists specifically to support citation of this codebase's results in a methodology write-up — it states precisely which historical numbers are trustworthy, which were discarded, and why.

### 8.1 Fail-safe design that limited the damage

`train_supervised_tuned.py` wraps each (method, sampling, ratio, seed) configuration's training call in its own `try/except`, and only ever writes a result file **after** training completes successfully; a crash produces a *missing* file, never a corrupted one. Combined with a skip-if-exists-and-valid check at the top of each configuration, this means: (a) no bug described in this report ever silently wrote a wrong number into a result file *as a direct consequence of a crash* — crashes (§1.4's eigenvector non-convergence, §4.4's GraphENS `torch.multinomial` failure) always produced missing files, not corrupted ones; (b) the genuinely dangerous failure mode was the *non-crashing* one (§1.3): a numerically unstable run that completes all 50 epochs with `NaN` loss throughout and still writes a "successful" result file with a meaningless near-zero metric. Table 1 below accounts for both failure modes.

### 8.2 Invalidated and regenerated file counts

| Dataset | Affected by §1.2 (timestamp) | Affected by §1.3 (normalization / NaN-loss) | Result files invalidated | Scope of re-run |
|---|---|---|---|---|
| `hi_small` | No (correct format originally) | No (small enough graph; 0 NaN epochs observed under all tested seeds) | 0 | None — pre-existing results retained |
| `elliptic` | No (does not use IBM timestamp parsing) | No (0 NaN epochs observed) | 0 | None — pre-existing results retained |
| `hi_medium` | **Yes** (~27.6% of raw rows affected) | **Yes** (83%+ NaN epochs under seed 42) | All methods, all samplings, all ratios, all seeds | Full re-run (`intrinsic`/`positional`/`deepwalk`/`node2vec`/`gcn`/`sage`/`gat`/`gin`) |
| `li_small` | **Yes** (small in this file's specific 500k-row window, but the same code path and same silent-swap risk applies) | **Yes** (83%+ NaN epochs under seed 42) | All methods, all samplings, all ratios, all seeds | Full re-run |
| `li_medium` | **Yes** (~27.6% of raw rows affected) | **Yes** | All methods, all samplings, all ratios, all seeds | Full re-run |

Concretely: 3,240 pre-existing GNN-family (`gcn`/`sage`/`gat`/`gin`) result files and a further 968 non-GNN-family (`intrinsic`/`positional`/`deepwalk`/`node2vec`) result files across `hi_medium`/`li_small`/`li_medium` were moved to `res/tuned_backup_prenormalization/` and `res/tuned_backup_timestampfix/` respectively (retained, not deleted, for audit purposes) rather than being overwritten in place, and the corresponding cached structural-feature CSVs (`res/{hi_medium,li_small,li_medium}_train_tuned*_features_nx.csv`, 15 files) were likewise moved aside to `res/backup_stale_topology_cache/`, since the node/edge set underlying those cached computations may itself have changed under the corrected sort order (§1.2, point 3).

### 8.3 Operational notes for reproduction

- **Concurrency**: running the IBM "medium"-scale networks' full sweeps concurrently on this memory-constrained (18GB) development machine risked swap-induced throughput collapse — a diagnosed instance showed processes silently making zero progress for over an hour at 0% CPU while the system held 96%+ swap utilization, root-caused in part to exact (pre-§1.5-fix) closeness centrality on a 500k-node graph. Sequential execution of the affected networks (with `scripts/run_supervised_repeated.sh`, which already sequences a given network's 3 seeds internally) is the recommended execution pattern for these three networks even after the §1.5 algorithmic fix, given the remaining combined memory footprint of concurrent large-graph jobs (§6.4, item 5).
- **Orphaned worker processes**: killing a stuck top-level training process with a plain `kill`/SIGTERM does not clean up its `multiprocessing` worker children (observed: 36 orphaned `spawn_main` workers, each holding ~250-300MB, continued running and consuming memory after their parent was killed) — these must be identified and terminated separately (e.g. `pgrep -f multiprocessing.spawn`) to actually reclaim the memory a stuck run was holding.
- **Bytecode cache**: a stale/corrupted `__pycache__` `.pyc` file produced during a period of severe memory pressure caused one transient, non-reproducible `ImportError` on process relaunch; clearing `__pycache__` resolved it immediately and is a reasonable first troubleshooting step if a previously-working import inexplicably fails after a resource-constrained run.

---

## Cross-Section Professional Commentary

1. **The transductive, full-graph setup is the foundational assumption underlying this entire benchmark.** The recurring phenomenon — "a train node's embedding aggregates real val/test neighbor features" — originates from the "single graph, mask-based split" architectural choice, not from any individual sampler's defect.
2. **The timestamp-parsing bug (§1.2) is qualitatively different from every other issue in this report**: it is the only one that corrupts the *ground truth* of what "past" and "future" mean for three of five datasets, rather than a downstream numerical or leakage concern layered on top of a correct split. Any methodology write-up citing results from `hi_medium`, `li_small`, or `li_medium` generated before this fix should be considered to be reporting on a different, incorrectly-constructed dataset split, not merely a noisier version of the same experiment.
3. **The unnormalized-IBM-features issue flagged in the previous edition of this report as a cross-cutting risk has been resolved**, but its diagnosis illustrates a general principle worth carrying forward: a numerical-stability problem that manifests as a hard crash in exactly one code path (GraphENS's `torch.multinomial`) can be the *most visible*, and therefore most easily mistaken for the *only*, symptom of a problem that is silently corrupting every other code path that shares the same unstable input.
4. **GATSMOTE remains the implementation closest to its source paper methodologically**, but also the most expensive to train and the most hyperparameter-sensitive; its defaults should be treated as unvalidated at IBM production scale until re-tuned against the now-corrected data pipeline (§6.4, item 1).
5. **GraphSAGE's mini-batch training coexists with GCN/GAT/GIN's full-graph training**, a computational-path difference worth remembering when comparing convergence across architectures at a fixed epoch count.
6. **For methodology reporting purposes**: the five datasets in this benchmark are not uniformly reproducible with the same operational effort — `elliptic` and `hi_small` have been stable and fully reproducible throughout this audit's history, while `hi_medium`/`li_small`/`li_medium` required two rounds of data-pipeline correction (timestamp parsing, feature standardization) plus non-trivial execution-scheduling care (§8.3) before their results could be trusted. A methodology section should disclose this asymmetry rather than presenting all five datasets as having received identical validation rigor by construction.
