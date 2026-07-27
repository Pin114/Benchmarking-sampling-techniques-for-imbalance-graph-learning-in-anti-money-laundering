import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE


def _coerce_torch_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    return np.array(value)


class GATSMOTE:
    def __init__(self, k_neighbors=5, attention_heads=1, edge_threshold=0.5, homophily_weight=1.0, ratio=None, use_predicted_labels_for_homophily=False, random_state=None):
        self.k_neighbors = int(k_neighbors)
        self.attention_heads = max(1, int(attention_heads))
        self.edge_threshold = float(edge_threshold)
        self.homophily_weight = float(homophily_weight)
        self.ratio = ratio
        self.use_predicted_labels_for_homophily = use_predicted_labels_for_homophily
        self.random_state = random_state

    def __call__(self, mask, features, labels, edge_index, predicted_labels=None):
        is_torch_feat = isinstance(features, torch.Tensor)
        is_torch_labels = isinstance(labels, torch.Tensor)
        is_torch_mask = isinstance(mask, torch.Tensor)
        is_torch_edge = isinstance(edge_index, torch.Tensor)

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
        if is_torch_edge:
            edge_index_np = edge_index.cpu().numpy()
        else:
            edge_index_np = np.array(edge_index)

        mask_np = np.atleast_1d(mask_np).astype(bool)
        idx_mask = np.where(mask_np)[0]
        if idx_mask.size == 0:
            return features, labels, mask, edge_index

        features_masked = np.nan_to_num(features_np[idx_mask], nan=0.0, posinf=0.0, neginf=0.0)
        labels_masked = labels_np[idx_mask]
        unique_classes, class_counts = np.unique(labels_masked, return_counts=True)
        if unique_classes.size < 2:
            return features, labels, mask, edge_index

        minority_class = unique_classes[np.argmin(class_counts)]
        minority_count = int(np.min(class_counts))
        majority_count = int(np.max(class_counts))
        target_minority_count = int(np.round(majority_count / self.ratio)) if self.ratio is not None else majority_count
        if target_minority_count <= minority_count:
            return features, labels, mask, edge_index

        sampling_strategy = {minority_class: target_minority_count}
        smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=max(1, min(int(self.k_neighbors), minority_count - 1)), random_state=self.random_state)
        X_smote, y_smote = smote.fit_resample(features_masked, labels_masked)

        n_original = features_masked.shape[0]
        n_synthetic = X_smote.shape[0] - n_original
        if n_synthetic <= 0:
            return features, labels, mask, edge_index

        expanded_features = np.vstack([np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0), X_smote[n_original:]])
        expanded_labels = np.concatenate([labels_np, y_smote[n_original:]])
        expanded_mask = np.zeros(len(expanded_labels), dtype=bool)
        expanded_mask[:len(mask_np)] = mask_np
        expanded_mask[len(labels_np):] = True

        clean_features = np.nan_to_num(features_np[idx_mask], nan=0.0, posinf=0.0, neginf=0.0)
        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, clean_features.shape[0]), algorithm='ball_tree').fit(clean_features)
        new_edges = []
        for synthetic_idx in range(n_synthetic):
            synthetic_feature = X_smote[n_original + synthetic_idx].reshape(1, -1)
            distances, indices = nbrs.kneighbors(synthetic_feature)
            candidate_sources = []
            for rel_idx in indices[0][1:]:
                candidate_sources.append(int(idx_mask[int(rel_idx)]))
            if not candidate_sources:
                continue
            if self.use_predicted_labels_for_homophily and predicted_labels is not None:
                pred_labels = np.array(predicted_labels)[candidate_sources]
                same_label = pred_labels == y_smote[n_original + synthetic_idx]
            else:
                same_label = np.array([labels_np[idx] == y_smote[n_original + synthetic_idx] for idx in candidate_sources])
            scores = np.array([1.0 if s else 0.0 for s in same_label], dtype=float)
            scores = scores * self.homophily_weight
            scores = np.maximum(scores, 0.0)
            attention_scores = np.exp(-np.clip(distances[0][1:], 1e-8, None)) + scores
            top_k = np.argsort(attention_scores)[::-1][:max(1, min(self.k_neighbors, len(candidate_sources)))]
            selected = [candidate_sources[i] for i in top_k if attention_scores[i] >= self.edge_threshold]
            if not selected:
                selected = [candidate_sources[i] for i in top_k]
            for neighbor_idx in selected:
                new_edges.append([int(neighbor_idx), int(features_np.shape[0] + synthetic_idx)])
                new_edges.append([int(features_np.shape[0] + synthetic_idx), int(neighbor_idx)])
        if new_edges:
            new_edges_array = np.array(new_edges).T
            expanded_edge_index = np.hstack([edge_index_np, new_edges_array]) if edge_index_np.size else new_edges_array
        else:
            expanded_edge_index = edge_index_np

        if is_torch_feat:
            expanded_features = torch.from_numpy(expanded_features).to(features.dtype).to(features.device)
        if is_torch_labels:
            expanded_labels = torch.from_numpy(expanded_labels).to(labels.device)
        if is_torch_mask:
            expanded_mask = torch.from_numpy(expanded_mask).to(mask.device)
        if is_torch_edge:
            expanded_edge_index = torch.from_numpy(expanded_edge_index).to(edge_index.device)
        return expanded_features, expanded_labels, expanded_mask, expanded_edge_index


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
        # GATSMOTE already uses in this file) instead of materializing a dense
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
