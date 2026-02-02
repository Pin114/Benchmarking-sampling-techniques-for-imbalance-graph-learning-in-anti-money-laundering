"""
特徵空間 SMOTE + 啟發式連邊 for GNN
這是 GraphSMOTE 的實用替代方案，避免複雜的邊生成模組
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
    在特徵空間進行 SMOTE，然後用啟發式方法連邊。
    
    Parameters:
    - mask: boolean mask indicating training samples
    - features: feature matrix (n_samples, n_features)
    - labels: label vector (n_samples,)
    - edge_index: original edge list (2, n_edges)
    - k_neighbors: k for k-NN graph in heuristic edge generation
    - heuristic: 'knn' (連接到 k 個最相似的節點) or 'parent' (連接到父節點)
    - random_state: seed for reproducibility
    
    Returns:
    - expanded_features: augmented feature matrix
    - expanded_labels: augmented label vector
    - expanded_mask: mask for augmented training set
    - expanded_edge_index: augmented edge list with heuristic connections
    """
    
    # 轉換為 numpy
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
    
    # Step 1: 提取訓練集的特徵和標籤
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
    
    # Step 2: 應用 SMOTE
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    
    # 找出新生成的虛擬節點
    n_original = X_train.shape[0]
    n_synthetic = X_smote.shape[0] - n_original
    
    # Step 3: 建立擴展的特徵矩陣和標籤
    # 全部特徵 = 原始特徵 + 虛擬特徵
    expanded_features = np.vstack([features_np, X_smote[n_original:]])
    expanded_labels = np.concatenate([labels_np, y_smote[n_original:]])
    
    # Step 4: 建立擴展的 mask（只標記訓練集和虛擬節點）
    expanded_mask = np.zeros(expanded_features.shape[0], dtype=bool)
    expanded_mask[idx_train] = True  # 原始訓練集
    expanded_mask[-n_synthetic:] = True  # 虛擬節點
    
    # Step 5: 啟發式連邊
    if heuristic == 'knn':
        # 使用 k-NN 圖連接虛擬節點到特徵最相似的 k 個節點
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree').fit(features_np)
        
        new_edges = []
        for i in range(n_synthetic):
            synthetic_idx = features_np.shape[0] + i  # 虛擬節點在擴展矩陣中的索引
            synthetic_feature = X_smote[n_original + i].reshape(1, -1)
            
            # 找 k 個最相似的原始節點
            distances, indices = nbrs.kneighbors(synthetic_feature)
            
            for neighbor_idx in indices[0][1:]:  # 跳過自己
                # 雙向邊
                new_edges.append([neighbor_idx, synthetic_idx])
                new_edges.append([synthetic_idx, neighbor_idx])
        
        # 合併原始邊和新邊
        if new_edges:
            new_edges_array = np.array(new_edges).T
            expanded_edge_index = np.hstack([edge_index_np, new_edges_array])
        else:
            expanded_edge_index = edge_index_np
    
    elif heuristic == 'parent':
        # 連接虛擬節點到它的「父節點」（通過 SMOTE 的 minority index）
        # 注：SMOTE 中的 synthetic_inds 屬性記錄了每個虛擬節點的父節點
        # 但這需要訪問 SMOTE 內部，暫用 knn 作為備選
        print("Warning: 'parent' heuristic requires SMOTE internal access. Falling back to 'knn'")
        return feature_smote_with_heuristic_edges(
            mask, features, labels, edge_index,
            k_neighbors=k_neighbors, heuristic='knn', random_state=random_state
        )
    
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")
    
    # 轉換回 torch 如果需要
    if is_torch:
        expanded_features = torch.from_numpy(expanded_features).float()
        expanded_labels = torch.from_numpy(expanded_labels).long()
        expanded_mask = torch.from_numpy(expanded_mask).bool()
        expanded_edge_index = torch.from_numpy(expanded_edge_index).long()
    
    return expanded_features, expanded_labels, expanded_mask, expanded_edge_index
