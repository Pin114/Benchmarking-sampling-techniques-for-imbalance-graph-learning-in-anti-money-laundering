# Technical Audit Report: Graph Machine Learning Codebase for Anti-Money Laundering

*Written against the actual state of the `pinyu` branch (merged from `main`, after commit `f10c235`).*

This report consists of 7 independent sections, each auditing — at fine granularity — the data pipeline, the split mechanism, leakage risk, sampling-technique theory-vs-implementation, joint training mechanics, classifier architecture and hyperparameters, and tensor-shape risk. Each section closes with professional commentary. Every claim is cited by file and line number so it can be checked directly against the source.

---

## 1. Data Pipeline and Graph Construction (Fine-Grained View)

### 1.1 Lifecycle: From CSV to `Data` Object

Two fully independent loading paths, managed by [`data/DatasetConstruction.py`](data/DatasetConstruction.py), both ultimately feed the same [`network_AML`](src/utils/Network.py#L16) class.

**(a) Elliptic — `load_elliptic()`** ([DatasetConstruction.py:17-53](data/DatasetConstruction.py#L17-L53))
- Reads three files: `elliptic_txs_features.csv` (column 0 = txId, column 1 = time_step, columns 2-166 = 165 anonymized features), `elliptic_txs_edgelist.csv`, `elliptic_txs_classes.csv`.
- `x = feat_df.loc[:, 'time_step':].values` (L32) treats `time_step` itself as the 0th column of `x`; `scripts/train_supervised_tuned.py` later manually slices it out via `ntw_torch.x[:, 1:94]`, keeping only the 93 "local" features (dropping the 72 "aggregated" feature columns and `time_step` itself).
- Class mapping `{'unknown': 2, '1': 1, '2': 0}` (L36): the original label `'1'` (illicit) maps to `y=1`; `'2'` (licit) maps to `y=0`; `'unknown'` maps to `y=2` (excluded when masks are built).
- **The split is a strict temporal split**: `train: time_step<30`, `val: 30<=t<40`, `test: t>=40`, all further `& (y != 2)` (L47-49).

**(b) IBM family — `load_ibm_config()`** ([DatasetConstruction.py:134-199](data/DatasetConstruction.py#L134-L199))
- Sorted ascending by `Timestamp`, self-transfers dropped (`Account==Account.1`), only the **last** 500,000 rows after sorting are kept (L154-156 — a tail slice, not a random sample).
- Final node feature set: `Amount Received`/`Amount Paid` (**no normalization/scaling applied anywhere** — see the Known Limitations note in §6.4) + `Day`/`Hour`/`Minute` (calendar components only) + three one-hot blocks (Receiving Currency, Payment Currency, Payment Format). **Account numbers and bank codes are dropped entirely** — the model never sees which account/bank a transaction touches, only structural signal via graph topology.
- Edges are built in `preprocess_ibm()` (L75-132): nodes are **transactions themselves** (not accounts). A directed edge A→B is created if transaction A's receiving account equals transaction B's paying account and `0<=Δt<=240min` (modeling money-flow chains).
- **Split** (L186-195): after time-sorting, the first 60% of rows = train, next 20% = val, last 20% = test — an equivalent temporal split.

### 1.2 `network_AML` and Precise Graph Semantics

[`src/utils/Network.py`](src/utils/Network.py):

- **Edges are unconditionally symmetrized** (`_set_up_network_info`, L51-57): `self.directed` is never set to `True` by any call site; every directed edge is duplicated as its reverse and merged in — **directional semantics are erased before the graph ever reaches a GNN** (this applies to both IBM's money-flow direction and Elliptic's original payment direction).
- **`get_network_torch()`** (L104-134): builds a single `torch_geometric.data.Data(x, y, edge_index)` from **all** (train+val+test) nodes and edges at once; the three masks are just boolean tensors hanging off the same object. **There is only ever one graph** — this is a pure transductive, full-batch setup, and train/val/test are never split into separate subgraphs.
- **`get_features(full=False)`**: an Elliptic-specific branch — `full=False` → columns 2-94 (93 columns, local), `full=True` → columns 2-166 (165 columns, local+aggregated). `intrinsic_features*` uses the default `full=False` (93 columns), but `positional_features*` calls `ntw.get_features(full=True)` (165 columns) — **the two baseline pipelines' "intrinsic feature space" definitions differ**, which matters when comparing scores between them; they are not the same feature set.

### 1.3 Global Topology: GCN/GAT/GIN Do a Full-Graph Forward Pass; GraphSAGE Now Has Real Neighbor Sampling

Previously, every `GNN_features*` function signature carried `train_loader`/`val_loader`/`test_loader: DataLoader = None` parameters that were never referenced anywhere in the function bodies — **these dead parameters, and the `from torch.utils.data import DataLoader` import, have now been removed** (all 5 `GNN_features*` functions in [experiments_supervised.py](src/methods/experiments_supervised.py) currently have empty signatures for these three params).

The current forward-pass mechanism for each of the four architectures ([`src/methods/utils/GNN.py`](src/methods/utils/GNN.py)):

| Architecture | Training-time graph scope | edge_attr/edge_weight |
|---|---|---|
| **GCN** (L29-77) | Single full-graph forward pass | Actually used: `edge_weight = edge_attr.view(-1).to(torch.float32)` (L63-65) |
| **GraphSAGE** (L79-136) | **Real neighbor-sampling mini-batches** (see below) | Not used (architecturally has no such concept — not a bug, see comment at L120-124) |
| **GAT** (L139-196) | Single full-graph forward pass | Actually used: `GATv2Conv(..., edge_dim=1)` (L163-171); `edge_attr` is reshaped to `[-1,1]` (L179) before genuinely entering the attention computation |
| **GIN** (L198-268) | Single full-graph forward pass | Not used (the original GIN formulation has no edge-feature concept — this is exactly why `GINE` exists, L275+, unused by this benchmark; see comment at L207-210) |

**GraphSAGE's neighbor sampling** ([experiments_supervised.py:70-101](src/methods/experiments_supervised.py#L70-L101), wired into [`GNN_features`](src/methods/experiments_supervised.py#L704)/[`GNN_features_with_predictions`](src/methods/experiments_supervised.py#L826)):
```python
use_neighbor_sampling = isinstance(model, GraphSAGE)
if use_neighbor_sampling:
    num_neighbors = sage_num_neighbors or [15] * max(int(model.n_layers), 1)
    train_loader = _try_build_neighbor_loader(ntw_torch, train_mask_sampled, sage_batch_size, num_neighbors, shuffle=True)
    use_neighbor_sampling = train_loader is not None
```
`_build_neighbor_loader` constructs a `torch_geometric.loader.NeighborLoader` on a CPU copy of the graph; each mini-batch places its seed nodes (`batch.batch_size` of them) first, followed by the sampled multi-hop neighbors. Loss is computed only over the seed-node slice (`out[:seed_n]`). `_try_build_neighbor_loader` (L85-101) actually pulls one batch as a probe: if the current environment is missing the `pyg-lib`/`torch-sparse` compiled backend that `NeighborLoader`'s sampler needs, it prints a warning and **gracefully falls back to a full-graph forward pass** instead of failing the entire experiment because of a missing dependency. Evaluation (`evaluate_split`) **always** runs on the full graph regardless.

**Professional commentary**: cleanly separating "mini-batch sampling for training, full graph for evaluation" is a well-established and defensible tradeoff in the literature (model selection stays fair, since every architecture's val/test evaluation runs on the same unsampled graph). However, GCN/GAT/GIN currently have no extension point at all for mini-batch training on large graphs — extending memory efficiency to these three architectures on million-node-scale IBM datasets in the future would require a non-trivial architectural refactor, not just flipping a switch analogous to SAGE's.

---

## 2. Data Split Mechanism and Train-Evaluation Alignment

### 2.1 The Exact Implementation

Both datasets use a **temporal split** (see §1.1), not a random one. Masks are stored as `torch.bool` tensors on `Data.train_mask/val_mask/test_mask` ([Network.py:127-132](src/utils/Network.py#L127-L132)), with length equal to the total node count and indices corresponding directly to node IDs.

### 2.2 Line-by-Line Application of Masks in Train/Eval Loops

Using `GNN_features` ([experiments_supervised.py:704-824](src/methods/experiments_supervised.py#L704-L824)) as an example:
```python
train_mask_sampled = train_mask.bool().to(device)                 # sampling=="none" branch
def train_epoch():
    out, _ = _forward(ntw_torch.x.to(device), ntw_torch.edge_index.to(device))  # full-graph forward
    y_train = y[train_mask_sampled]                                  # labels of train nodes only
    loss_val = _compute_loss(criterion, out[train_mask_sampled], y_train)  # logits of train nodes only
```
Every epoch, the GNN performs one full-graph forward pass (including val/test) — message passing lets val/test node features participate in the aggregation for train node embeddings, which is an unavoidable consequence of transductive learning (using val/test nodes structurally is fine, as long as their **labels** never enter the loss). Loss computation slices explicitly here — only nodes where `train_mask` is true enter the gradient — this step is clean. **Note: `_compute_loss` calls no longer redundantly pass a `mask=` argument** (a bug that previously existed here — see §6.3).

`evaluate_split(mask)` runs another full-graph forward pass under `torch.no_grad()`, using the same slicing technique to compute `val_ap`/`test_f1`. The epoch loop only triggers `early_stopping` on `val_mask`; `test_mask` is touched exactly once, after training ends and the val-best checkpoint has been reloaded — the correct isolation order.

`GNN_features_graphsmote(_with_predictions)` ([L1009-1255](src/methods/experiments_supervised.py#L1009-L1255)) **symmetrically pads** `val_mask`/`test_mask` when synthetic nodes exist (synthetic nodes always get `False` appended to val/test masks), using identical code on adjacent lines for both — there is no "test is protected, val isn't" asymmetry.

### 2.3 Audit of Validation-Set Usage Rigor — All Methods Now Aligned

`node2vec_features`/`node2vec_features_with_predictions` ([L488-595](src/methods/experiments_supervised.py#L488-L595), [L596-703](src/methods/experiments_supervised.py#L596-L703)) now run their decoder training loop as:
```python
x_val = x[:val_mask.shape[0]][val_mask.bool()].to(device_decoder).squeeze()
early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path, monitor='val_ap')
for epoch in range(n_epochs_decoder):
    ...
    val_ap = average_precision_score(y_val.cpu().numpy(), val_output_softmax.cpu().numpy()[:, 1])
    early_stopping(val_ap, decoder)
    if early_stopping.early_stop: break
if os.path.exists(checkpoint_path):
    decoder.load_state_dict(torch.load(checkpoint_path, map_location=device_decoder))
```
This is exactly the same mechanism used by `intrinsic_features_with_predictions`/`positional_features_with_predictions`. `train_supervised_tuned.py` has also been updated to pass `patience=10, checkpoint_path=unique_checkpoint_path` at the call site.

**Deliberately left untouched**: Node2Vec's own random-walk pretraining (`node2vec_representation_torch`, [functionsTorch.py:42-172](src/methods/utils/functionsTorch.py#L42-L172)) still has no validation signal and runs a fixed `n_epochs` with no early stopping — this is a reasonable design choice for an unsupervised skip-gram objective (there is no natural supervised validation signal; using downstream AP for early stopping would require running the decoder every single embedding epoch, conflating the responsibilities of two separate training stages), not an oversight.

**Current state**: `intrinsic`, `positional`, `node2vec/deepwalk` (decoder stage), and every `GNN_features*` variant now **all** use `val_mask` for `EarlyStopping(monitor='val_ap')` plus best-weight reloading — validation mechanics are now aligned across every pipeline.

**Professional commentary**: when `use_torch=True`, Node2Vec restricts its graph to a `train∪test` node subgraph (near [experiments_supervised.py:466](src/methods/experiments_supervised.py): `active_nodes = train_mask.bool() | test_mask.bool()`, deliberately excluding val) — meaning the embedding-training stage is **structurally** exposed to test nodes and their edges (though never their labels). This is the same phenomenon, in essence, as a GNN's full-graph message passing every epoch; it's a common "structural contact" pattern in transductive representation learning, worth recording but not a bug.

---

## 3. Comprehensive Data-Leakage Vulnerability Audit

### 3.1 Positional/Structural Feature Leakage — `positional_features_with_predictions`: Clean, and Actively Hardened

Tracing line by line ([L360-487](src/methods/experiments_supervised.py#L360-L487)):
```python
train_val_mask = train_mask.bool() | val_mask.bool()
train_val_nodes = set(torch.where(train_val_mask)[0].tolist())
ntw_nx_train_val = ntw_nx_full.subgraph(list(train_val_nodes))    # induced subgraph -- test nodes/edges do not exist here
features_nx_df_train_val = local_features_nx(ntw_nx_train_val, ...)   # PageRank/density/RNC computed on this subgraph
features_nk_df_train_val = features_nk(ntw_nx_train_val, ...)         # Betweenness/Closeness/Eigenvector too
```
The code carries an explicit comment: `# Subgraph isolation to completely block test set structure leakage`. `nx.pagerank`, `nx.betweenness_centrality`/`closeness_centrality`/`eigenvector_centrality` ([functionsNetworKit.py:7-36](src/methods/utils/functionsNetworKit.py#L7-L36)) physically cannot see any test node or any edge touching a test node when computing "features for training".

`features_df_full` (the full-graph version) is used only to extract `X_test` and never enters `loss_val.backward()` — test features can see full-graph topology (a reasonable inference-time assumption) but this does not constitute a training-gradient leak.

**One detail worth recording, though not a bug**: the train feature subgraph mixes in val-node topology (`train_val_mask = train|val`) — train nodes' centrality values are mildly influenced by the presence of val nodes, but only structurally, never through val **labels** (`fraud_dict_train` has already been filtered down to pure-train via `fraud_dict_known` in `train_supervised_tuned.py`, guarded by an explicit `assert`).

### 3.2 Resampling/SMOTE Leakage — `graph_smote_mask`/`GATEdgeGenerator` Fixed and Kept Consistent

Current state ([evaluation.py:198-296](src/methods/evaluation.py#L198-L296)):
```python
nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, features_masked.shape[0]), algorithm='ball_tree').fit(features_masked)
...
for neighbor_idx in indices[0][1:]:
    neighbor_global_idx = int(idx_mask[neighbor_idx])   # remapped back to global index
```
`features_masked = features_np[idx_mask]` — the k-NN candidate pool is strictly restricted to train-mask nodes, consistent with `GATEdgeGenerator.prepare_synthetic_nodes` ([samplers.py:159-195](src/methods/samplers.py#L159-L195): `nbrs = NearestNeighbors(...).fit(features_masked)`) and `TargetedNeighbourhoodUndersampling.__call__` (which was already correct here). Synthetic minority nodes can only ever connect to real nodes inside the training set.

`GraphENS` ([graphens.py](src/methods/graphens.py)) builds its adjacency list from the real, full-graph edges; `minor_pool`/`target_pool` are restricted to train, but a train minority node's real neighbors can be val/test nodes. The new edges are deliberately **directional** (`neighbor -> synthetic`, per `blended_neighbor_sampling`'s docstring, "never symmetrized") — this only lets val/test **features** flow one-way into the synthetic node, and never lets the synthetic node's information flow back into the val/test node's own embedding. This is the same class of phenomenon as the "train nodes aggregate real neighbors that may be val/test" behavior that every transductive GNN already has by design — not a leak newly introduced by GraphENS.

**Current state**: the k-NN/homophily searches in `graph_smote_mask`, `GATEdgeGenerator`, and `TargetedNeighbourhoodUndersampling` are **all** now correctly restricted to train-mask nodes. `reweighted_graph_smote_mask`/`unweighted_graph_smote_mask` (which previously both had the same leak, and the former's weight formula additionally degenerated into the mathematical constant `exp(-1)≈0.368`) have been deleted entirely from `evaluation.py` and are no longer part of the attack surface.

---

## 4. Sampling Techniques: Theory vs. Actual Code Implementation

### 4.1 Basic SMOTE / RUS
`smote_mask` ([evaluation.py:116-196](src/methods/evaluation.py#L116-L196)) directly calls `imblearn.over_sampling.SMOTE.fit_resample` — a textbook implementation. `random_undersample_mask` (L66-114) uses `RandomState.choice` for sampling-without-replacement — fully consistent with theory.

### 4.2 Vanilla Graph SMOTE
- **Theory** (Zhao, Zhang & Wang, WSDM 2021): after SMOTE interpolation generates nodes, an **edge generator/decoder** is trained (a reconstruction loss against the observed adjacency matrix), producing connection probabilities, and jointly trained with the classifier.
- **Implementation** (`graph_smote_mask`): no trainable edge generator; edges are attached via k-NN (now correctly restricted to train), with weight fixed at 1 (no `edge_attr`). This is a **one-shot, non-parametric heuristic** that captures the spirit of "interpolate + attach edges" but has nothing to do with the paper's trainable decoder mechanism.

### 4.3 GATSMOTE — A Genuinely Trainable PyTorch Module, Now Refined to Match the Paper

`samplers.py`'s `GATEdgeGenerator(nn.Module)` ([L16-395](src/methods/samplers.py#L16-L395)) implements Liu et al., *GATSMOTE* (Mathematics 2022, DOI 10.3390/math10111799). Checking each claim:

- **Trainable weight matrices/attention vectors** (near [L46-47](src/methods/samplers.py#L46-L47)): `self.W = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(heads)])`, `self.a = nn.ModuleList([nn.Linear(2*hidden_dim, 1, bias=False) for _ in range(heads)])` — genuine `nn.Parameter`s, initialized with `nn.init.xavier_uniform_`.
- **Multi-head attention** (`forward`, [L260-361](src/methods/samplers.py#L260-L361)): `e = LeakyReLU(0.2)(a_h([Wz_i‖Wz_j]))`, `alpha = segment_softmax(e, dst_idx)` — the standard GAT formula, normalized per destination (synthetic) node.
- **Head fusion, refined to match the paper**: `logit_stack = torch.stack(head_logits, dim=-1)` (**the raw, pre-softmax scores `e^{tk}`**) → `edge_logits = self.fusion(logit_stack)` (a trainable `nn.Linear(heads,1)`) → `edge_probs = sigmoid(edge_logits)`. This is a deliberate design decision — the code comment explicitly records: "the paper fuses the raw pre-normalization score e^{tk} (Eq. 8/Algorithm 1) before the separate softmax normalization step" — fusion happens *before* softmax, not on already-normalized attention weights (this is the point at which a teammate corrected the implementation against the paper, after merging the earlier version I had built).
- **Locality auxiliary loss** (paper Eq. 10 / Hypothesis 1):
  $$\mathcal{L}_{\text{locality}} = -\text{mean}\big(2 \cdot E^t \cdot (\text{sim}_{\cos} - 0.5)\big)$$
  A bilinear "push to extremes" design: highly-similar pairs are pushed toward `E^t→1`, dissimilar pairs toward `E^t→0`, rather than regressing toward a graded similarity value (the earlier MSE-based formulation has been replaced by this bilinear form).
- **Shortest-path auxiliary loss** (paper Eq. 11 / Hypothesis 2): a capped BFS (`_bounded_hop_distance`, [L107-129](src/methods/samplers.py#L107-L129), `max_hops=4`) computes hop distance from the synthetic node's "SMOTE parent" (its nearest real minority-class neighbor in feature space, `indices[0][0]`), then rewards same-label pairs proportionally to that distance:
  ```python
  distance_weight = clamp(pair_hop_distance / max_hops, 0, 1)
  per_pair_coeff = mismatch - match * distance_weight   # same-label & structurally far -> push toward E^t=1; mismatched label -> push toward E^t=0
  loss_shortest = (per_pair_coeff * edge_probs).mean()
  ```
  This is much closer to the paper's core idea of "shortening paths between same-label nodes" than a binary label-mismatch penalty — bounded-BFS hop distance approximates the paper's exact path-count/matrix-power computation, keeping this tractable on large, sparse AML graphs.

**Known limitation (explicitly documented in code)**: measured on the real HI-Small graph (500k nodes, 631k edges), only 0.33% of pairs resolve to a genuine hop count within the `max_hops=4` cap; of the rest, 99.11% (of *all* pairs) are simply in a **different connected component** from their parent (this graph has 374k connected components, 351k of them singletons) — i.e. unreachable at any hop count, regardless of the cap. The mechanism still correctly grades the resolvable minority and remains directionally correct for the rest (pushing toward connections message-passing can't already reach), but has low resolution among "far" pairs at this scale — this is a consequence of how fragmented the real transaction graph is, not a mistuned hop cap.

**Joint training**: see Section 5. **Conclusion**: this is a genuinely trainable module — parameter updates, multi-head attention (including the paper's exact fusion order), and dual auxiliary-loss backpropagation are all present, and the implementation has already been corrected against the paper's formulas.

### 4.4 GraphENS / Graph Ensemble SMOTE — A Real Implementation, No Dead Code

`sampling="graph_ensemble_smote"` now corresponds to a **genuine implementation of GraphENS** (Park, Song & Yang, ICLR 2022, Algorithm 1), composed of [`graphens.py`](src/methods/graphens.py) (pure functions) and `GNN_features_graphens_with_predictions` ([experiments_supervised.py:1256 onward](src/methods/experiments_supervised.py#L1256)):

- **Degree-distribution alignment**: `graphens.sample_augmented_degree` samples from the real graph's degree histogram, capped at `deg(v_minor)` — genuinely implemented.
- **Blended ego-network sampling**: `blended_neighbor_sampling` implements Eq. 1, `p(u|v_mixed)=φ̂·p(u|v_minor)+(1-φ̂)·p(u|v_target)`, sampling without replacement — genuinely implemented.
- **KL mixing ratio φ̂**, **saliency-masked mixup**, **confidence aggregation ô** (mean-then-softmax; the docstring explicitly records this as a deliberate choice to follow the reference implementation's formula over the paper's literal wording) — all implemented in `graphens.py` and invoked by `train_epoch()`; **no unreached dead code was found**.

### 4.5 Targeted Neighbourhood Undersampling (TNU)

**Logic for flagging noisy nodes** ([samplers.py:409 onward](src/methods/samplers.py#L409)): for every minority-class node, find its k nearest neighbors via `NearestNeighbors(metric='cosine')`; if a neighbor is majority-class **and** its cosine distance `>= noise_threshold`, flag it as a removal candidate.

**CLI parameter wiring — now fixed, with clearly separated responsibilities**: the outer sweep's `ratio` drives the default via `effective_remove_ratio = tnu_remove_ratio if tnu_remove_ratio is not None else ratio` ([experiments_supervised.py:986-987](src/methods/experiments_supervised.py#L986-L987)); all six of `--tnu-k-neighbors`, `--tnu-distance-metric`, `--tnu-remove-ratio`, `--tnu-noise-threshold`, `--tnu-min-majority-keep`, and `--tnu-preserve-minority-neighbors` are now passed through `_build_graphsmote_sampling` into the `TargetedNeighbourhoodUndersampling(...)` constructor — no longer silently ignored.

### 4.6 "Original" (ratio=None) Semantic Consistency Across Methods — Fixed

`_build_graphsmote_sampling` ([L946-1007](src/methods/experiments_supervised.py#L946-L1007)) now places `if ratio is None:` at the very front of its if/elif chain:
```python
if ratio is None:
    x_smote, y_smote, train_mask_smote, edge_index_smote = ntw_torch.x, ntw_torch.y, train_mask.bool(), ntw_torch.edge_index
elif sampling_name == "gatsmote":
    ...
```
`ratio=None` (the "Original" row) now uniformly means "no resampling" for `graph_smote`/`gatsmote`/`tnu`, consistent with the pre-existing semantics used by other families such as `GNN_features`/`intrinsic_features` (`elif sampling_name == "none" or ratio is None:`). Before this fix, `graph_smote_mask(..., ratio=None, ...)` would internally fall back to `target_minority_count=majority_count` (full 1:1 rebalancing), and TNU's `remove_ratio=None` would remove **every** flagged noisy candidate — both contradicted the expectation that "Original" represents the native class distribution. This has been verified experimentally: after the fix, `y_true`'s length exactly matches the unaugmented test set.

---

## 5. Joint Training and Optimization Mechanics

`GATEdgeGenerator` is the only sampler in this codebase with trainable parameters of its own; `GNN_features_graphsmote(_with_predictions)` ([L1009-1255](src/methods/experiments_supervised.py#L1009-L1255)) wires up a complete joint-training pipeline for it:

**Optimizer**:
```python
joint_params = list(model.parameters()) + list(gat_edge_gen.parameters()) if gat_edge_gen is not None else list(model.parameters())
optimizer = torch.optim.Adam(joint_params, lr=lr, weight_decay=5e-4)
```

**Total loss formula per epoch step** (`train_epoch`):
```python
edge_index_epoch, edge_attr_epoch, loss_locality, loss_shortest = gat_edge_gen.build_epoch_graph(...)  # recomputed from the current W/a/fusion
loss_node = _compute_loss(criterion, out[train_mask], y[train_mask])
loss_val = loss_node + gat_edge_gen.lambda_locality * loss_locality + gat_edge_gen.lambda_shortest * loss_shortest
loss_val.backward()
```
That is, $\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{node}}+\lambda_1\cdot\mathcal{L}_{\text{locality}}+\lambda_2\cdot\mathcal{L}_{\text{shortest}}$, corresponding to the CLI flags `--gatsmote-lambda1` (default `0.2`) and `--gatsmote-lambda2` (default `0.05`). **Edge probabilities are not computed once and cached — they are recomputed every epoch using the current attention parameters** (`build_epoch_graph`, [samplers.py:363-395](src/methods/samplers.py#L363-L395)); only the synthetic *topology* (which candidate pairs exist) is fixed once before training starts (because re-drawing the candidate pool every epoch would make mask sizes unstable). Fixed candidate set + per-epoch recomputed attention weights is the only engineering-sound compromise for "joint training" here.

**Early Stopping and Checkpoint Synchronization**:
```python
checkpoint_target = nn.ModuleList([model, gat_edge_gen]) if gat_edge_gen is not None else model
early_stopping(metric_to_monitor, checkpoint_target)         # saves the state_dict of the whole module group
checkpoint_target.load_state_dict(torch.load(checkpoint_path, ...))   # reloaded together
```
**This is the fix for a real bug that previously existed**: if only `early_stopping(metric_to_monitor, model)` were called (saving just the classifier), then the `gat_edge_gen` weights at "the moment val performance was best" would never be saved — at final reload, the classifier would be the val-best version, but `gat_edge_gen` would be whatever state training happened to end on. The two would be out of sync, meaning the edge probabilities used at test time would be computed from an attention-parameter set the classifier was never actually trained alongside. Wrapping both in `nn.ModuleList([model, gat_edge_gen])` and saving/loading them as a unit eliminates this risk entirely.

**Contrast with GraphENS**: `GNN_features_graphens_with_predictions`'s optimizer only contains `model.parameters()`, because GraphENS has no trainable parameters of its own — the `saliency`/`confidence` statistics it uses each epoch are computed from that same step's classifier gradients and outputs (closure state carried across epochs, not parameters being optimized). It needs no second parameter set to keep in sync. This is a clean dividing line in this codebase between "genuine joint training" and "dynamic but not jointly trained".

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

"Lightweight mode" (`use_lightweight_gat = num_nodes >= 300000`, `train_supervised_tuned.py`) only affects GAT, and also drops `gnn_epochs` from 50 to 10 — GCN/SAGE/GIN are entirely unaffected on large graphs.

### 6.2 edge_attr Handling

See the table in §1.3. GAT is now the **only** architecture that genuinely re-weights edge information via attention (GCN's edge-weighted linear averaging is semantically weaker). GraphSAGE/GIN's lack of support is an inherent property of their original formulations having no edge-feature concept, not an implementation gap — `GNN.py` carries explicit comments on both explaining why.

### 6.3 Full CLI Hyperparameter Inventory (`train_supervised_tuned.py`)

**Imbalance-ratio sweep grid**:
```python
test_ratios = [None, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
```
Semantics: `target_minority_count = round(majority_count / ratio)`, where `ratio` is the target majority:minority ratio. `ratio=None` (Original) now consistently means "no resampling" across **every** sampling-technique family (see §4.6).

**Gradient clipping**: `--clip_norm` (default `1.0`) is now genuinely threaded into every `GNN_features*` function's `clip_norm` parameter, replacing the previously hardcoded `max_norm=1.0`.

**Loss function formulas**:
- **Cross-Entropy**: `nn.CrossEntropyLoss(weight=[1.0, pos_weight])`, `pos_weight = num_neg/max(num_pos,1)`, computed dynamically from whatever subset is current.
- **Focal Loss** ([losses.py:39-79](src/methods/losses.py#L39-L79)):
  $$FL(p_t)=-\alpha_t(1-p_t)^{\gamma}\log(p_t)$$
  `gamma` defaults to `2.0`; `alpha` supports `None`/`'balanced'`/a scalar/a list. **A previously fixed crash bug**: every GNN-related function called `_compute_loss` with `out`/`y` already filtered by `train_mask`, then redundantly passed that same, unfiltered-length mask into `FocalLoss.forward(..., mask=...)`, tripping its internal shape check and guaranteeing a `ValueError`. The fix was to stop re-passing `mask` on tensors that were already filtered — `--loss focal` now trains correctly with any GNN method (verified experimentally: it crashed before the fix and produces a normal loss curve after).

**GATSMOTE CLI** (`train_supervised_tuned.py`): `--gatsmote-k-neighbors` (5), `--gatsmote-heads` (4), `--gatsmote-edge-threshold` (0.5), `--gatsmote-lambda1` (0.2), `--gatsmote-lambda2` (0.05), `--gatsmote-use-predicted-labels` — all correctly wired through (the old `--gatsmote-attention-heads`/`--gatsmote-homophily-weight` flag names have been renamed to the above, matching the genuinely trainable module's actual parameters).

**TNU/GraphENS CLI**: see §4.5 and the "GraphENS" section of the README — all correctly wired through.

### 6.4 Known Limitations (Explicitly Documented in Code, Not Speculation From This Audit)

1. **`--gatsmote-lambda1`/`--gatsmote-lambda2`'s default values were tuned on a small, well-scaled synthetic graph**, not validated at real IBM/Elliptic production scale. A validation attempt against real `hi_small` data produced an unusable result (`loss_node` exploding into the millions) — this explosion is an artifact of the unscaled `Amount Received`/`Amount Paid` features noted in §1.1 (up to `~2.8e11`), not a property of the loss terms themselves. These defaults will need re-tuning once IBM feature normalization is addressed.
2. **The shortest-path auxiliary loss has limited resolution on real, sparse AML graphs** (see the Known Limitation in §4.3).
3. **GraphSAGE's neighbor sampling requires `pyg-lib` or `torch-sparse`** with a working compiled backend for the current platform. Both are listed in `requirements.txt`, but on platforms without prebuilt wheels for them, training silently falls back to full-graph forward passes (a warning is printed when this happens).

---

## 7. Feature-Space Mismatches and Array Truncation Risk

### 7.1 Shape Changes After Appending Synthetic Nodes (SMOTE Family)

`graph_smote_mask` and `GATEdgeGenerator.prepare_synthetic_nodes` both follow the same pattern: synthetic nodes are always appended to the tail of the original feature matrix (`vstack/cat([original N rows, synthetic M rows])`); `expanded_mask[:N]` keeps the original mask, `expanded_mask[N:]=True` (synthetic rows always count as train).

### 7.2 `X_val`/`X_test` Slicing Logic — Two Strategies, Both Verified Symmetric

**Strategy A (graph-structural methods, `GNN_features_graphsmote*`)**: padding. `n_synthetic = x_smote.shape[0] - ntw_torch.x.shape[0]`; `val_mask_smote`/`test_mask_smote` use identical code, the same `n_synthetic`, on adjacent lines, appending `torch.zeros(n_synthetic, dtype=torch.bool)` — there is no "test is protected, val isn't" asymmetry.

**Strategy B (pure feature-space methods, `intrinsic/positional/node2vec_features*`)**: truncation. `features_tensor[:val_mask.shape[0]]`/`[:test_mask.shape[0]]` first slice the potentially-lengthened tensor back to its original length N, then apply the original-length mask — `X_val` and `X_test` now **both** use this pattern (after the node2vec fix, the two are written symmetrically).

### 7.3 Shape Safety in `GATEdgeGenerator.build_epoch_graph`

```python
keep = edge_probs.detach() >= self.edge_threshold
kept_pairs = synthetic_pairs[:, keep]; kept_weights = edge_probs[keep]
dynamic_edge_index = torch.cat([kept_pairs, kept_pairs.flip(0)], dim=1)
full_edge_index = torch.cat([base_edge_index, dynamic_edge_index], dim=1)
full_edge_attr = torch.cat([base_weight, dynamic_edge_weight], dim=0)
```
When `keep` is all-`False` (possible early in training, before attention has learned any meaningful signal), `dynamic_edge_index`/`dynamic_edge_weight` become empty tensors; `torch.cat` between an empty and non-empty tensor is well-defined — the graph temporarily degenerates to "no synthetic edges", classification loss still computes normally, and `loss_locality`/`loss_shortest` (computed over the full `synthetic_pairs` set regardless of the `keep` filter) continue to supply a gradient signal pushing the attention parameters toward more meaningful `E^t` values. This design is naturally robust to the edge case of "no edges clear the threshold yet".

### 7.4 Shape Safety in `generate_all_tables.py`'s Multi-Seed Aggregation

`scripts/generate_all_tables.py` now stores each (metric, dataset, sampling, method, ratio) cell as a `{seed_key: value}` dictionary rather than a single scalar; `aggregate_cell()` computes `statistics.mean`/`statistics.stdev` (sample standard deviation, ddof=1, only when n>=2; a bare value is shown when n=1) over `values_by_seed.values()`. Verified against synthetic 3-seed files: `[0.50, 0.52, 0.48]` correctly resolves to `0.5000 ± 0.0200`, and the real tables under `tables/tuned_only_ratio_comparison_*.md` (including the newly added `F1_90` table) have been regenerated against real data.

**Current state**: the previously flagged "node2vec is missing X_val protection" issue has been resolved as part of the §2 fix (X_val now exists, and its truncation logic is symmetric with X_test); the graph-structural methods' padding logic was already symmetric and introduces no new risk; the new multi-seed aggregation logic has also been verified shape-safe.

---

## Cross-Section Professional Commentary

1. **The transductive, full-graph setup is the foundational assumption underlying this entire benchmark.** The recurring phenomenon in §1.3/§2.2/§3.2 — "a train node's embedding aggregates real val/test neighbor features" — originates from the "single graph, mask-based split" architectural choice, not from any individual sampler's defect.
2. **GATSMOTE is now the implementation closest to its source paper methodologically**, but it is also the most expensive to train and has the most hyperparameters — hyperparameter-search experiment design should probably allocate it more random seeds to reach stable conclusions.
3. **The IBM datasets' missing feature normalization is a root-cause issue that spans multiple modules.** It doesn't just blow up GATSMOTE's loss (§6.4, item 1) — any component whose behavior depends on stable gradient scale (Focal Loss's `(1-p_t)^γ` term, GATEdgeGenerator's dual auxiliary-loss weights) may carry similarly unvalidated risk on the IBM datasets. This deserves priority in the next round of fixes as a standalone issue, rather than being patched piecemeal wherever it happens to surface (e.g. only in GATSMOTE).
4. **GraphSAGE's mini-batch training now coexists with GCN/GAT/GIN's full-graph training**, which is a computational-path difference worth remembering when comparing across architectures: SAGE's definition of "one epoch" (multiple mini-batch gradient updates) differs from the other three's (a single full-graph gradient update) — comparisons like "which architecture converges faster at the same epoch count" need to account for this.
