import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from torch_geometric.utils import softmax as segment_softmax


def _coerce_torch_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.array(value)


class GATEdgeGenerator(nn.Module):
    """Trainable edge generator for GATSMOTE (Liu et al., Mathematics 2022, 10.3390/math10111799).

    Synthetic minority-class node features are produced once via SMOTE interpolation over the
    train mask, together with a fixed candidate-neighbor topology per synthetic node (k-NN in
    raw feature space, restricted to train-mask candidates -- see prepare_synthetic_nodes). Given
    that fixed candidate pool, this module learns per-head GAT-style attention (linear projection
    `W` + attention vector `a`, LeakyReLU + softmax normalization) over each synthetic node's
    candidates, fuses the heads through a learnable linear layer into an edge connectivity
    probability E^t in [0,1], and exposes two differentiable auxiliary losses (locality/cosine-
    similarity and homophily/label-mismatch) so the whole thing is optimized jointly with the
    downstream node classifier every training step.
    """

    def __init__(
        self,
        in_dim,
        k_neighbors=5,
        attention_heads=4,
        hidden_dim=None,
        edge_threshold=0.5,
        lambda_locality=1.0,
        lambda_shortest=1.0,
        ratio=None,
        use_predicted_labels_for_homophily=False,
        random_state=None,
    ):
        super().__init__()
        self.k_neighbors = int(k_neighbors)
        self.attention_heads = max(1, int(attention_heads))
        self.hidden_dim = int(hidden_dim) if hidden_dim else max(8, int(in_dim) // 2)
        self.edge_threshold = float(edge_threshold)
        self.lambda_locality = float(lambda_locality)
        self.lambda_shortest = float(lambda_shortest)
        self.ratio = ratio
        self.use_predicted_labels_for_homophily = use_predicted_labels_for_homophily
        self.random_state = random_state

        self.W = nn.ModuleList([nn.Linear(int(in_dim), self.hidden_dim, bias=False) for _ in range(self.attention_heads)])
        self.a = nn.ModuleList([nn.Linear(2 * self.hidden_dim, 1, bias=False) for _ in range(self.attention_heads)])
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fusion = nn.Linear(self.attention_heads, 1)  # gamma (weight) + beta (bias) head fusion

        for head_W in self.W:
            nn.init.xavier_uniform_(head_W.weight)
        for head_a in self.a:
            nn.init.xavier_uniform_(head_a.weight)
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def _empty_prepared(self, features, labels, mask, edge_index, device):
        return dict(
            features=features,
            labels=labels,
            mask=mask,
            edge_index=edge_index,
            synthetic_pairs=torch.empty((2, 0), dtype=torch.long, device=device),
            pair_label_match=torch.empty((0,), dtype=torch.bool, device=device),
            n_synthetic=0,
        )

    def prepare_synthetic_nodes(self, mask, features, labels, edge_index, predicted_labels=None):
        """One-shot, non-trainable step: SMOTE feature interpolation for the minority class plus
        a fixed candidate-neighbor topology per synthetic node, restricted to train-mask
        candidates only (mirrors the leak fix applied to graph_smote_mask in evaluation.py --
        fitting the k-NN search on unmasked features would let synthetic train nodes attach
        directly to val/test nodes based on their feature values). Call once per training run
        (not per epoch) -- the synthetic population and mask sizes must stay fixed across epochs
        so that `forward` can be called on the same graph every step with only the attention
        parameters changing.
        """
        is_torch_feat = isinstance(features, torch.Tensor)
        is_torch_labels = isinstance(labels, torch.Tensor)
        is_torch_mask = isinstance(mask, torch.Tensor)
        is_torch_edge = isinstance(edge_index, torch.Tensor)

        device = features.device if is_torch_feat else torch.device('cpu')
        dtype = features.dtype if is_torch_feat else torch.float32

        features_np = features.detach().cpu().numpy() if is_torch_feat else np.array(features)
        labels_np = labels.detach().cpu().numpy() if is_torch_labels else np.array(labels)
        mask_np = mask.detach().cpu().numpy().astype(bool) if is_torch_mask else np.array(mask).astype(bool)
        edge_index_np = edge_index.detach().cpu().numpy() if is_torch_edge else np.array(edge_index)

        mask_np = np.atleast_1d(mask_np).astype(bool)
        idx_mask = np.where(mask_np)[0]
        if idx_mask.size == 0:
            return self._empty_prepared(features, labels, mask, edge_index, device)

        features_masked = np.nan_to_num(features_np[idx_mask], nan=0.0, posinf=0.0, neginf=0.0)
        labels_masked = labels_np[idx_mask]
        unique_classes, class_counts = np.unique(labels_masked, return_counts=True)
        if unique_classes.size < 2:
            return self._empty_prepared(features, labels, mask, edge_index, device)

        minority_class = unique_classes[np.argmin(class_counts)]
        minority_count = int(np.min(class_counts))
        majority_count = int(np.max(class_counts))
        target_minority_count = int(np.round(majority_count / self.ratio)) if self.ratio is not None else majority_count
        if target_minority_count <= minority_count:
            return self._empty_prepared(features, labels, mask, edge_index, device)

        sampling_strategy = {minority_class: target_minority_count}
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=max(1, min(self.k_neighbors, minority_count - 1)),
            random_state=self.random_state,
        )
        X_smote, y_smote = smote.fit_resample(features_masked, labels_masked)

        n_original = features_masked.shape[0]
        n_synthetic = X_smote.shape[0] - n_original
        if n_synthetic <= 0:
            return self._empty_prepared(features, labels, mask, edge_index, device)

        clean_features = np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0)
        expanded_features_np = np.vstack([clean_features, X_smote[n_original:]])
        expanded_labels_np = np.concatenate([labels_np, y_smote[n_original:]])
        expanded_mask_np = np.zeros(len(expanded_labels_np), dtype=bool)
        expanded_mask_np[:len(mask_np)] = mask_np
        expanded_mask_np[len(labels_np):] = True

        # Candidate pool restricted to train-mask nodes only (features_masked / idx_mask),
        # matching the graph_smote_mask leak fix -- see evaluation.py.
        n_neighbors_query = min(self.k_neighbors + 1, features_masked.shape[0])
        nbrs = NearestNeighbors(n_neighbors=n_neighbors_query, algorithm='ball_tree').fit(features_masked)

        n_original_nodes = features_np.shape[0]
        synthetic_pairs = []
        pair_label_match = []
        for synthetic_idx in range(n_synthetic):
            synthetic_feature = X_smote[n_original + synthetic_idx].reshape(1, -1)
            _, indices = nbrs.kneighbors(synthetic_feature)
            candidate_sources = [int(idx_mask[int(rel_idx)]) for rel_idx in indices[0][1:]]
            if not candidate_sources:
                continue
            synthetic_global_idx = n_original_nodes + synthetic_idx
            synthetic_label = y_smote[n_original + synthetic_idx]
            if self.use_predicted_labels_for_homophily and predicted_labels is not None:
                pred_np = _coerce_torch_numpy(predicted_labels)
                neighbor_labels = pred_np[candidate_sources]
            else:
                neighbor_labels = labels_np[candidate_sources]
            for src, neighbor_label in zip(candidate_sources, neighbor_labels):
                synthetic_pairs.append([src, synthetic_global_idx])
                pair_label_match.append(bool(neighbor_label == synthetic_label))

        if not synthetic_pairs:
            return self._empty_prepared(features, labels, mask, edge_index, device)

        synthetic_pairs_t = torch.tensor(synthetic_pairs, dtype=torch.long, device=device).t().contiguous()
        pair_label_match_t = torch.tensor(pair_label_match, dtype=torch.bool, device=device)

        expanded_features = torch.from_numpy(expanded_features_np).to(dtype=dtype, device=device)
        expanded_labels_device = labels.device if is_torch_labels else device
        expanded_mask_device = mask.device if is_torch_mask else device
        expanded_labels = torch.from_numpy(expanded_labels_np).to(device=expanded_labels_device)
        expanded_mask = torch.from_numpy(expanded_mask_np).to(device=expanded_mask_device)
        expanded_edge_index = edge_index.to(device) if is_torch_edge else torch.from_numpy(edge_index_np).to(device)

        return dict(
            features=expanded_features,
            labels=expanded_labels,
            mask=expanded_mask,
            edge_index=expanded_edge_index,
            synthetic_pairs=synthetic_pairs_t,
            pair_label_match=pair_label_match_t,
            n_synthetic=n_synthetic,
        )

    def forward(self, features, synthetic_pairs, pair_label_match=None):
        """Trainable multi-head attention pass over `synthetic_pairs` ([2, P]: row0 = existing
        candidate node index, row1 = synthetic node index). Recomputed every training step since
        W/a/fusion change with each gradient update -- this is the joint end-to-end path, not a
        one-shot heuristic.

        Returns (edge_probs, loss_locality, loss_shortest):
          - edge_probs: E^t in [0,1] per pair, from multi-head GAT attention fused through a
            learnable nn.Linear(heads, 1).
          - loss_locality: pulls E^t toward the pair's cosine similarity in learned embedding
            space (maximize connectivity between feature-similar nodes).
          - loss_shortest: penalizes E^t on pairs whose labels mismatch (homophily regularizer).
        """
        if synthetic_pairs.numel() == 0:
            zero = features.sum() * 0.0
            return torch.empty((0,), dtype=features.dtype, device=features.device), zero, zero

        src_idx, dst_idx = synthetic_pairs[0], synthetic_pairs[1]
        head_alphas = []
        head_embeds_src = []
        head_embeds_dst = []
        for W_h, a_h in zip(self.W, self.a):
            z = W_h(features)
            z_src = z[src_idx]
            z_dst = z[dst_idx]
            e = self.leaky_relu(a_h(torch.cat([z_src, z_dst], dim=-1))).squeeze(-1)
            # Neighbor-wise softmax normalization per synthetic (destination) node, matching GAT.
            alpha = segment_softmax(e, dst_idx)
            head_alphas.append(alpha)
            head_embeds_src.append(z_src)
            head_embeds_dst.append(z_dst)

        alpha_stack = torch.stack(head_alphas, dim=-1)  # [P, heads]
        edge_logits = self.fusion(alpha_stack).squeeze(-1)  # [P]
        edge_probs = torch.sigmoid(edge_logits)

        mean_z_src = torch.stack(head_embeds_src, dim=0).mean(dim=0)
        mean_z_dst = torch.stack(head_embeds_dst, dim=0).mean(dim=0)
        cos_sim = F.cosine_similarity(mean_z_src, mean_z_dst, dim=-1)
        target_similarity = (cos_sim + 1.0) / 2.0
        loss_locality = F.mse_loss(edge_probs, target_similarity.detach())

        if pair_label_match is not None and pair_label_match.numel() > 0:
            mismatch = (~pair_label_match).to(edge_probs.dtype)
            loss_shortest = (edge_probs * mismatch).mean()
        else:
            loss_shortest = edge_probs.sum() * 0.0

        return edge_probs, loss_locality, loss_shortest

    def build_epoch_graph(self, base_edge_index, features, synthetic_pairs, pair_label_match=None):
        """Combines the fixed base edges with the current epoch's trainable synthetic edges.

        Synthetic pairs whose current E^t clears `edge_threshold` become (bidirectional) graph
        edges weighted by E^t; everything below is dropped from the topology used for message
        passing, though `edge_probs` (and hence loss_locality/loss_shortest) stays fully
        differentiable regardless of that hard cut.
        """
        edge_probs, loss_locality, loss_shortest = self.forward(features, synthetic_pairs, pair_label_match)
        device = features.device
        dtype = features.dtype

        keep = edge_probs.detach() >= self.edge_threshold
        if keep.any():
            kept_pairs = synthetic_pairs[:, keep]
            kept_weights = edge_probs[keep]
            dynamic_edge_index = torch.cat([kept_pairs, kept_pairs.flip(0)], dim=1)
            dynamic_edge_weight = torch.cat([kept_weights, kept_weights], dim=0)
        else:
            dynamic_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            dynamic_edge_weight = torch.empty((0,), dtype=dtype, device=device)

        base_weight = torch.ones(base_edge_index.shape[1], dtype=dtype, device=device)
        full_edge_index = torch.cat([base_edge_index, dynamic_edge_index], dim=1)
        full_edge_attr = torch.cat([base_weight, dynamic_edge_weight], dim=0)
        return full_edge_index, full_edge_attr, loss_locality, loss_shortest


class TargetedNeighbourhoodUndersampling:
    def __init__(self, k_neighbors=10, distance_metric='cosine', remove_ratio=None, target_classes=None, preserve_minority_neighbors=True, noise_threshold=0.5, min_majority_keep=1, random_state=None):
        self.k_neighbors = int(k_neighbors)
        self.distance_metric = distance_metric
        self.remove_ratio = remove_ratio
        self.target_classes = target_classes
        self.preserve_minority_neighbors = preserve_minority_neighbors
        self.noise_threshold = float(noise_threshold)
        self.min_majority_keep = max(1, int(min_majority_keep))
        self.random_state = random_state

    def __call__(self, mask, features, labels):
        is_torch_feat = isinstance(features, torch.Tensor)
        is_torch_labels = isinstance(labels, torch.Tensor)
        is_torch_mask = isinstance(mask, torch.Tensor)

        if is_torch_feat:
            features_np = features.cpu().numpy()
        else:
            features_np = np.array(features)
        if is_torch_labels:
            labels_np = labels.cpu().numpy()
        else:
            labels_np = np.array(labels)
        if is_torch_mask:
            mask_np = mask.cpu().numpy().astype(bool)
        else:
            mask_np = np.array(mask).astype(bool)

        mask_np = np.atleast_1d(mask_np).astype(bool)
        idx_mask = np.where(mask_np)[0]
        if idx_mask.size == 0:
            return mask

        train_labels = labels_np[idx_mask]
        unique_classes, counts = np.unique(train_labels, return_counts=True)
        if unique_classes.size < 2:
            return mask

        minority_class = unique_classes[np.argmin(counts)]
        majority_classes = [c for c in unique_classes if c != minority_class]
        if self.target_classes is not None:
            majority_classes = [c for c in majority_classes if c in self.target_classes]
        if not majority_classes:
            return mask

        majority_indices = np.where(np.isin(train_labels, majority_classes))[0]
        minority_indices = np.where(train_labels == minority_class)[0]
        if majority_indices.size == 0 or minority_indices.size == 0:
            return mask

        features_masked = np.nan_to_num(features_np[idx_mask], nan=0.0, posinf=0.0, neginf=0.0)
        # Nearest-neighbor search via sklearn.neighbors.NearestNeighbors (same API
        # GATEdgeGenerator already uses in this file) instead of materializing a dense
        # len(idx_mask) x len(idx_mask) pairwise matrix, which is infeasible at
        # real train_mask sizes (e.g. ~360TB at 300k nodes). metric='cosine'
        # matches the previous 1 - cosine_similarity distance scale exactly, so
        # noise_threshold's meaning is unchanged; sklearn runs this in chunked
        # brute-force under the hood since ball_tree/kd_tree don't support cosine.
        metric = 'cosine' if self.distance_metric == 'cosine' else 'euclidean'
        n_neighbors_query = min(self.k_neighbors + 1, features_masked.shape[0])
        nbrs = NearestNeighbors(n_neighbors=n_neighbors_query, metric=metric).fit(features_masked)
        query_distances, query_indices = nbrs.kneighbors(features_masked[minority_indices])

        remove_candidates = []
        for row in range(minority_indices.shape[0]):
            neighbor_ids = query_indices[row][1:]  # skip self (nearest, distance ~0)
            neighbor_scores = query_distances[row][1:]
            for neighbor_id, score in zip(neighbor_ids, neighbor_scores):
                if train_labels[neighbor_id] in majority_classes:
                    if float(score) >= self.noise_threshold:
                        remove_candidates.append(int(idx_mask[int(neighbor_id)]))

        if not remove_candidates:
            return mask

        # remove_ratio is a target majority:minority count ratio, matching the
        # semantics of `ratio` in random_undersample_mask (evaluation.py), not a
        # raw fraction of remove_candidates. Convert it into how many majority
        # nodes need to go to reach that target, then clamp to what the noisy-
        # neighbor pass actually flagged as removable.
        majority_count = int(majority_indices.size)
        minority_count = int(minority_indices.size)
        if self.remove_ratio is not None:
            desired_majority = int(np.round(minority_count * self.remove_ratio))
            target_remove = max(0, majority_count - desired_majority)
        else:
            target_remove = len(remove_candidates)
        target_remove = min(target_remove, max(0, len(remove_candidates) - self.min_majority_keep))
        if target_remove <= 0:
            return mask

        rng = np.random.RandomState(self.random_state)
        remove_indices = rng.choice(remove_candidates, size=target_remove, replace=False)
        new_mask = np.array(mask_np, copy=True)
        new_mask[remove_indices] = False

        if is_torch_mask:
            return torch.from_numpy(new_mask).to(mask.device)
        return new_mask
