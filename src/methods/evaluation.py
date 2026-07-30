import os
import sys
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE

class EarlyStopping:
    """
    Early Stopping controller that stops training when the validation metric (AP or Loss) has not improved.
    """
    def __init__(self, patience=10, verbose=True, delta=0.0, checkpoint_path='checkpoint.pt', monitor='val_ap'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_ap_max = -np.inf

    def __call__(self, val_metric, model):
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        if self.monitor == 'val_loss':
            score = -val_metric
        elif self.monitor == 'val_ap':
            score = val_metric
        else:
            raise ValueError(f"Unsupported monitor metric: {self.monitor}")
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] Counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        if self.verbose:
            if self.monitor == 'val_loss':
                print(f"[EarlyStopping] Val Loss decreased ({self.val_loss_min:.6f} --> {val_metric:.6f}). Saving model...")
                self.val_loss_min = val_metric
            elif self.monitor == 'val_ap':
                print(f"[EarlyStopping] Val AP increased ({self.val_ap_max:.6f} --> {val_metric:.6f}). Saving model...")
                self.val_ap_max = val_metric
        torch.save(model.state_dict(), self.checkpoint_path)

def random_undersample_mask(mask, labels, ratio=None, random_state=None):
    """
    Randomly undersamples the majority class inside the mask down to target ratio.
    If ratio is None or target ratio results in more majority than original, no undersampling is done.
    """
    is_torch = isinstance(mask, torch.Tensor)
    if is_torch:
        mask_np = mask.cpu().numpy().astype(bool)
        labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    else:
        mask_np = np.array(mask).astype(bool)
        labels_np = np.array(labels)
        
    mask_np = np.atleast_1d(mask_np)
    idx_mask = np.where(mask_np)[0]
    if idx_mask.size == 0:
        return mask
        
    labels_in_mask = labels_np[idx_mask]
    unique_classes, counts = np.unique(labels_in_mask, return_counts=True)
    if unique_classes.size <= 1:
        return mask
        
    minority_class = unique_classes[np.argmin(counts)]
    majority_class = unique_classes[np.argmax(counts)]
    minority_count = np.min(counts)
    majority_count = np.max(counts)
    
    if ratio is not None:
        desired_majority = int(np.round(minority_count * ratio))
    else:
        desired_majority = minority_count # default to 1:1
        
    if desired_majority >= majority_count:
        return mask
        
    rnd = np.random.RandomState(random_state)
    idx_minority = idx_mask[np.where(labels_in_mask == minority_class)[0]]
    idx_majority = idx_mask[np.where(labels_in_mask == majority_class)[0]]
    
    selected_majority = rnd.choice(idx_majority, size=desired_majority, replace=False)
    selected_all = np.concatenate([idx_minority, selected_majority])
    
    new_mask = np.zeros_like(mask_np, dtype=bool)
    new_mask[selected_all] = True
    
    if is_torch:
        return torch.from_numpy(new_mask).to(mask.device)
    return new_mask

def smote_mask(mask, features, labels, k_neighbors=5, ratio=None, random_state=None):
    """
    SMOTE applied within a mask with target ratio.
    """
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
        return features, labels, mask
        
    features_masked = features_np[idx_mask]
    features_masked = np.nan_to_num(features_masked, nan=0.0, posinf=0.0, neginf=0.0)
    labels_masked = labels_np[idx_mask]
    
    unique_classes, class_counts = np.unique(labels_masked, return_counts=True)
    if unique_classes.size < 2:
        return features, labels, mask
        
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]
    minority_count = np.min(class_counts)
    majority_count = np.max(class_counts)
    
    if ratio is not None:
        target_minority_count = int(np.round(majority_count / ratio))
    else:
        target_minority_count = majority_count
        
    if target_minority_count <= minority_count:
        return features, labels, mask
        
    sampling_strategy = {minority_class: target_minority_count}
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=max(1, min(int(k_neighbors), minority_count - 1)),
        random_state=random_state
    )
    X_smote, y_smote = smote.fit_resample(features_masked, labels_masked)
    
    n_original = features_masked.shape[0]
    n_synthetic = X_smote.shape[0] - n_original
    
    if n_synthetic <= 0:
        expanded_features = features_np
        expanded_labels = labels_np
    else:
        expanded_features = np.vstack([np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0), X_smote[n_original:]])
        expanded_labels = np.concatenate([labels_np, y_smote[n_original:]])
        
    expanded_mask = np.zeros(len(expanded_labels), dtype=bool)
    expanded_mask[:len(mask_np)] = mask_np
    expanded_mask[len(labels_np):] = True
    
    if is_torch_feat:
        expanded_features = torch.from_numpy(expanded_features).to(features.dtype).to(features.device)
    if is_torch_labels:
        expanded_labels = torch.from_numpy(expanded_labels).to(labels.device)
    if is_torch_mask:
        expanded_mask = torch.from_numpy(expanded_mask).to(mask.device)
        
    return expanded_features, expanded_labels, expanded_mask

def graph_smote_mask(mask, features, labels, edge_index, k_neighbors=5, ratio=None, similarity_metric='cosine', random_state=None):
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
        
    features_masked = features_np[idx_mask]
    features_masked = np.nan_to_num(features_masked, nan=0.0, posinf=0.0, neginf=0.0)
    labels_masked = labels_np[idx_mask]
    
    unique_classes, class_counts = np.unique(labels_masked, return_counts=True)
    if unique_classes.size < 2:
        return features, labels, mask, edge_index
        
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]
    minority_count = np.min(class_counts)
    majority_count = np.max(class_counts)
    
    if ratio is not None:
        target_minority_count = int(np.round(majority_count / ratio))
    else:
        target_minority_count = majority_count
        
    if target_minority_count <= minority_count:
        return features, labels, mask, edge_index
        
    sampling_strategy = {minority_class: target_minority_count}
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=max(1, min(int(k_neighbors), minority_count - 1)),
        random_state=random_state
    )
    X_smote, y_smote = smote.fit_resample(features_masked, labels_masked)
    
    n_original = features_masked.shape[0]
    n_synthetic = X_smote.shape[0] - n_original
    
    if n_synthetic <= 0:
        expanded_features = features_np
        expanded_labels = labels_np
        expanded_mask = mask_np
        expanded_edge_index = edge_index_np
    else:
        expanded_features = np.vstack([np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0), X_smote[n_original:]])
        expanded_labels = np.concatenate([labels_np, y_smote[n_original:]])
        expanded_mask = np.zeros(len(expanded_labels), dtype=bool)
        expanded_mask[:len(mask_np)] = mask_np
        expanded_mask[len(labels_np):] = True
        
        # Candidate pool for the synthetic node's graph neighbor is restricted to
        # train-mask nodes only (features_masked / idx_mask), not the full,
        # unmasked node set -- fitting on features_np here would let synthetic
        # train nodes attach directly to val/test nodes based on val/test's own
        # feature values, leaking test-set structure into the training graph.
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, features_masked.shape[0]), algorithm='ball_tree').fit(features_masked)
        new_edges = []
        for synthetic_idx in range(n_synthetic):
            synthetic_feature = X_smote[n_original + synthetic_idx].reshape(1, -1)
            _, indices = nbrs.kneighbors(synthetic_feature)
            for neighbor_idx in indices[0][1:]:
                neighbor_global_idx = int(idx_mask[neighbor_idx])
                new_edges.append([neighbor_global_idx, int(features_np.shape[0] + synthetic_idx)])
                new_edges.append([int(features_np.shape[0] + synthetic_idx), neighbor_global_idx])
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

