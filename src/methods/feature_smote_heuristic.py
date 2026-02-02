"""
Feature-Space SMOTE + Heuristic Edge Connection for GNN
This is a practical alternative to GraphSMOTE, avoiding complex edge generation modules
"""
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE


def feature_smote_with_heuristic_edges(
    mask,
    features,
    labels,
    edge_index,
    k_neighbors=5,
    heuristic='knn',
    random_state=None
):
    """
    Apply SMOTE in feature space, then use heuristic edge connection.
    
    Parameters:
    - mask: boolean mask indicating training samples
    - features: feature matrix (n_samples, n_features)
    - labels: label vector (n_samples,)
    - edge_index: original edge list (2, n_edges)
    - k_neighbors: k for k-NN graph in heuristic edge generation
    - heuristic: 'knn' (connect to k most similar nodes) or 'parent' (connect to parent node)
    - random_state: seed for reproducibility
    
    Returns:
    - expanded_features: augmented feature matrix
    - expanded_labels: augmented label vector
    - expanded_mask: mask for augmented training set
    - expanded_edge_index: augmented edge list with heuristic connections
    """
    
    # Convert to numpy
    is_torch = isinstance(features, torch.Tensor)
    if is_torch:
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
        edge_index_np = edge_index.cpu().numpy() if isinstance(edge_index, torch.Tensor) else np.array(edge_index)
    else:
        features_np = np.array(features)
        labels_np = np.array(labels)
        mask_np = np.array(mask)
        edge_index_np = np.array(edge_index)
    
    mask_np = np.atleast_1d(mask_np.astype(bool))
    
    # Step 1: Extract training features and labels
    idx_train = np.where(mask_np)[0]
    X_train = features_np[idx_train]
    y_train = labels_np[idx_train]
    
    # Ensure labels are binary (0 and 1 only) - filter out any other values like 2 (unknown)
    valid_idx = np.isin(y_train, [0, 1])
    if not np.all(valid_idx):
        print(f"Warning: Found {(~valid_idx).sum()} training samples with invalid labels. Filtering them out.")
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]
        idx_train = idx_train[valid_idx]
    
    # Step 2: Apply SMOTE
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    
    # Identify newly generated synthetic nodes
    n_original = X_train.shape[0]
    n_synthetic = X_smote.shape[0] - n_original
    
    # Step 3: Build expanded feature matrix and labels
    # All features = original features + synthetic features
    expanded_features = np.vstack([features_np, X_smote[n_original:]])
    expanded_labels = np.concatenate([labels_np, y_smote[n_original:]])
    
    # Step 4: Build expanded mask (mark only training set and synthetic nodes)
    expanded_mask = np.zeros(expanded_features.shape[0], dtype=bool)
    expanded_mask[idx_train] = True  # Original training set
    expanded_mask[-n_synthetic:] = True  # Synthetic nodes
    
    # Step 5: Heuristic edge connection
    if heuristic == 'knn':
        # Use k-NN graph to connect synthetic nodes to k most similar features
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree').fit(features_np)
        
        new_edges = []
        for i in range(n_synthetic):
            synthetic_idx = features_np.shape[0] + i  # Index of synthetic node in expanded matrix
            synthetic_feature = X_smote[n_original + i].reshape(1, -1)
            
            # Find k most similar original nodes
            distances, indices = nbrs.kneighbors(synthetic_feature)
            
            for neighbor_idx in indices[0][1:]:  # Skip self
                # Bidirectional edges
                new_edges.append([neighbor_idx, synthetic_idx])
                new_edges.append([synthetic_idx, neighbor_idx])
        
        # Merge original edges and new edges
        if new_edges:
            new_edges_array = np.array(new_edges).T
            expanded_edge_index = np.hstack([edge_index_np, new_edges_array])
        else:
            expanded_edge_index = edge_index_np
    
    elif heuristic == 'parent':
        # Connect synthetic nodes to their "parent nodes" (via SMOTE minority index)
        # Note: SMOTE synthetic_inds attribute records parent node for each synthetic node
        # But this requires internal SMOTE access, falling back to knn as alternative
        print("Warning: 'parent' heuristic requires SMOTE internal access. Falling back to 'knn'")
        return feature_smote_with_heuristic_edges(
            mask, features, labels, edge_index,
            k_neighbors=k_neighbors, heuristic='knn', random_state=random_state
        )
    
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")
    
    # Convert back to torch if needed
    if is_torch:
        expanded_features = torch.from_numpy(expanded_features).float()
        expanded_labels = torch.from_numpy(expanded_labels).long()
        expanded_mask = torch.from_numpy(expanded_mask).bool()
        expanded_edge_index = torch.from_numpy(expanded_edge_index).long()
    
    return expanded_features, expanded_labels, expanded_mask, expanded_edge_index
