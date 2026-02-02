from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors
import random

from sklearn.ensemble import IsolationForest

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Ensure repository root is on sys.path so `import src...` works when running this file directly.
# File is at src/methods/evaluation.py -> repo root is two levels up.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC_PATH = os.path.join(ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.methods.utils.functionsNetworkX import *
from src.methods.utils.functionsNetworKit import *
from src.methods.utils.functionsTorch import *
from src.methods.utils.GNN import *
from utils.Network import *
from data.DatasetConstruction import *

from src.methods.utils.decoder import *
from src.methods.feature_smote_heuristic import feature_smote_with_heuristic_edges

from tqdm import tqdm

def positional_features_calc(
        ntw,
        alpha_pr: float,
        alpha_ppr: float=None,
        fraud_dict_train: dict=None,
        fraud_dict_test: dict=None,
        ntw_name: str=None, 
        use_intrinsic: bool = True
):
    print("networkx: ")
    ntw_nx = ntw.get_network_nx()
    features_nx_df = local_features_nx(ntw_nx, alpha_pr, alpha_ppr, fraud_dict_train=fraud_dict_train, ntw_name=ntw_name)

    ## Train NetworkKit
    print("networkit: ")
    try:
        ntw_nk = ntw.get_network_nk()
        features_nk_df = features_nk(ntw_nk, ntw_name=ntw_name)
    except NameError as e:
        # features_nk not defined in the codebase — fallback to NetworkX features
        print(f"[Warning] features_nk not found ({e}). Falling back to NetworkX features.")
        ntw_nx = ntw.get_network_nx()
        features_nk_df = features_nx(ntw_nx, ntw_name=ntw_name)
    except Exception as e:
        # Any other runtime error (including networkit failure) -> fallback to NetworkX
        print(f"[Warning] features_nk failed ({e}). Falling back to NetworkX features.")
        ntw_nx = ntw.get_network_nx()
        features_nk_df = features_nx(ntw_nx, ntw_name=ntw_name)

    ## Concatenate features
    if use_intrinsic:
        print("intrinsic and summary: ")
        X = ntw.get_features(full=True)
        features_df = pd.concat([X, features_nx_df, features_nk_df], axis=1)
    else:
        features_df = pd.concat([features_nx_df, features_nk_df], axis=1)
    features_df["fraud"] = [fraud_dict_test[x] for x in features_df.index]
    return features_df

def train_model_shallow(
        x_train, 
        y_train, 
        n_epochs_decoder: int, 
        lr: float,
        n_layers_decoder: int=2, 
        hidden_dim_decoder: int=10, 
        device_decoder: str="cpu"):
    decoder = Decoder_deep_norm(x_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    return(decoder)

def train_model_deep(data, model, train_mask, n_epochs, lr, batch_size, loader = None, use_intrinsic=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.
    criterion = nn.CrossEntropyLoss()  # Define loss function.

    if use_intrinsic:
        features = data.x
    else:
        features = torch.ones((data.x.shape[0], 1),dtype=torch.float32).to(device)

    def train_GNN():
        model.train()
        optimizer.zero_grad()
        y_hat, h = model(features, data.edge_index.to(device))
        y = data.y
        loss = criterion(y_hat[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        return(loss)

    for _ in range(n_epochs):
        loss_train = train_GNN()
        print('Epoch: {:03d}, Loss: {:.4f}'.format(_, loss_train))
    
def stratified_sampling(x_test, y_test):
    n_samples = x_test.shape[0]
    x_new, y_new = resample(x_test, y_test, n_samples=n_samples, stratify=y_test)
    return(x_new, y_new)


def random_undersample_majority(x, y, return_index=False, random_state=None):
    """
    Randomly undersample the majority class so that all classes have the same number
    of samples as the minority class.

    Works with numpy arrays or torch tensors. If `return_index` is True, also returns
    the indices (relative to the input arrays) that were kept.

    Parameters:
    - x: feature array (n_samples, ...)
    - y: label array (n_samples,) with integer class labels (e.g., 0/1)
    - return_index: whether to return selected indices
    - random_state: int or None

    Returns:
    - x_res, y_res  (and indices if return_index=True)
    """
    # Accept torch tensors or numpy arrays
    import numpy as _np
    try:
        import torch as _torch
    except Exception:
        _torch = None

    is_torch = _torch is not None and isinstance(y, _torch.Tensor)

    if is_torch:
        y_np = y.cpu().numpy()
    else:
        y_np = _np.array(y)

    classes, counts = _np.unique(y_np, return_counts=True)
    min_count = int(_np.min(counts))

    # for reproducibility
    rnd = _np.random.RandomState(random_state)

    selected_idx = []
    for cls in classes:
        idx_cls = _np.where(y_np == cls)[0]
        if idx_cls.shape[0] <= min_count:
            selected = idx_cls
        else:
            selected = rnd.choice(idx_cls, size=min_count, replace=False)
        selected_idx.append(selected)

    selected_idx = _np.concatenate(selected_idx)
    rnd.shuffle(selected_idx)

    # Build outputs
    if is_torch:
        x_np = x.cpu().numpy() if isinstance(x, _torch.Tensor) else _np.array(x)
    else:
        x_np = _np.array(x)

    x_res = x_np[selected_idx]
    y_res = y_np[selected_idx]

    if is_torch and _torch is not None:
        x_res = _torch.from_numpy(x_res).to(x.device) if isinstance(x, _torch.Tensor) else x_res
        y_res = _torch.from_numpy(y_res).to(y.device) if isinstance(y, _torch.Tensor) else y_res

    if return_index:
        return (x_res, y_res, selected_idx)
    return (x_res, y_res)


def random_undersample_mask(mask, labels, target_ratio=1.0, random_state=None):
    """
    Given a boolean mask (1D tensor or array) and a label vector for all samples,
    randomly undersample the majority class inside `mask` so that the majority count
    becomes `target_ratio * minority_count`.

    Returns a new boolean mask of the same shape.
    """
    import numpy as _np
    try:
        import torch as _torch
    except Exception:
        _torch = None

    # make sure mask_np is 1D and type-correct
    is_torch = _torch is not None and isinstance(mask, _torch.Tensor)

    if is_torch:
        # 轉換為 NumPy 陣列時，強制轉換為 bool 以避免 object_ 類型
        mask_np = mask.cpu().numpy().astype(_np.bool_)
        labels_np = labels.cpu().numpy() if isinstance(labels, _torch.Tensor) else _np.array(labels)
    else:
        # force to bool to avoid object_ dtype
        mask_np = _np.array(mask).astype(_np.bool_)
        labels_np = _np.array(labels)

    # make sure mask_np is 1D
    mask_np = _np.atleast_1d(mask_np) 

    # 獲取原始訓練遮罩 (mask) 中所有為 True 的樣本索引
    idx = _np.where(mask_np)[0]
    if idx.size == 0:
        # nothing to do
        return mask.clone() if is_torch and _torch is not None else mask_np

    lbls_in_mask = labels_np[idx]
    classes, counts = _np.unique(lbls_in_mask, return_counts=True) #計算在 mask 內部，各類別的樣本數量
    if classes.size <= 1:
        # single class inside mask
        return mask.clone() if is_torch and _torch is not None else mask_np

    minority_count = int(_np.min(counts)) #找出在 mask 內，少數類的數量
    desired_majority = int(_np.floor(minority_count * target_ratio))

    rnd = _np.random.RandomState(random_state)

    """遍歷 mask 內的類別：
    1. 如果是多數類（cnt > minority_count），則使用 rnd.choice 隨機抽樣出 desired_majority 個索引。
    2. 如果是少數類，則保留所有索引。"""

    # build new selected indices
    selected = []
    for cls, cnt in zip(classes, counts):
        idx_cls = idx[_np.where(lbls_in_mask == cls)[0]]
        if cnt <= minority_count:
            selected.append(idx_cls)
        else:
            take = min(desired_majority, idx_cls.shape[0])
            selected.append(rnd.choice(idx_cls, size=take, replace=False))

    selected = _np.concatenate(selected)
    
    # 獲取原始 mask 的尺寸，以確保返回的 mask 尺寸一致
    original_shape = mask.shape if is_torch else _np.array(mask).shape
    new_mask = _np.zeros(original_shape, dtype=_np.bool_) #創建一個與原始數據集大小相同的全新遮罩，所有值初始化為 False
         
    new_mask[selected] = True #在 new_mask 中，只將通過採樣被選中的索引位置設為 True

    if is_torch and _torch is not None:
        return _torch.from_numpy(new_mask)
    return new_mask

def evaluate_model_shallow_AUC(model, x_test, y_test, device = "cpu"):
    model.eval()

    x_test = x_test.to(device)
    y_test = y_test.to(device)
    y_pred = model(x_test)
    y_pred = y_pred.softmax(dim=1)

    AUC = roc_auc_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    AP = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])

    return(AUC, AP)

def evaluate_model_shallow_PRF(model, x_test, y_test, percentile_q = 99, device = "cpu"):
    model.eval()

    x_test = x_test.to(device)
    y_test = y_test.to(device)
    y_pred = model(x_test)
    y_pred = y_pred.softmax(dim=1)

    cutoff = np.percentile(y_pred.cpu().detach().numpy()[:,1], percentile_q)
    y_pred_hard = (y_pred[:,1] >= cutoff)*1
    precision = precision_score(y_test.cpu().detach().numpy(), y_pred_hard.cpu().detach().numpy())
    recall = recall_score(y_test.cpu().detach().numpy(), y_pred_hard.cpu().detach().numpy())
    F1 = f1_score(y_test.cpu().detach().numpy(), y_pred_hard.cpu().detach().numpy())

    return(precision, recall, F1)

def resample_mask(test_mask, p=0.5):
    sample_size = int(np.floor(test_mask.sum()*p))
    # Get indices where value is True
    true_indices = [i for i, val in enumerate(test_mask) if val]

    # Randomly select a subset of these indices
    sampled_indices = random.sample(true_indices, min(sample_size, len(true_indices)))

    # Create new tensor with False at all indices except the sampled ones
    output_tensor = torch.zeros_like(test_mask, dtype=torch.bool)
    output_tensor[sampled_indices] = True

    return output_tensor

import torch

def subsample_true_values_tensor(test_mask, p=0.5):
    # Get indices where value is True
    sample_size = int(np.floor(test_mask.sum()*p))
    true_indices = torch.where(test_mask)[0]

    # Randomly select a subset of these indices
    sampled_indices = true_indices[torch.randperm(true_indices.size(0))[:min(sample_size, true_indices.size(0))]]

    # Create new tensor with False at all indices except the sampled ones
    output_tensor = torch.zeros_like(test_mask, dtype=torch.bool)
    output_tensor[sampled_indices] = True

    return output_tensor

def evaluate_model_deep(data, model, test_mask, percentile_q_list = [99], n_samples=100, device = "cpu", loader = None, use_intrinsic=True):
    AUC_list = []
    AP_list = []

    precision_dict = dict()
    recall_dict = dict()
    F1_dict = dict()
    for percentile_q in percentile_q_list:
        precision_dict[percentile_q] = []
        recall_dict[percentile_q] = []
        F1_dict[percentile_q] = []

    model.eval()

    for _ in tqdm(range(n_samples)):
        test_mask_new = resample_mask(test_mask)
        if loader is None:
            model.eval()
            if use_intrinsic:
                features = data.x
            else:
                features = torch.ones((data.x.shape[0], 1), dtype=torch.float32).to(device)
            out, h = model(features, data.edge_index.to(device))
            y_hat = out[test_mask_new].to(device) # Prediction
            y = data.y[test_mask_new].to(device) # True value
            
        else:
            batch = next(iter(loader))
            batch = batch.to(device, 'edge_index')
            if use_intrinsic:
                features = batch.x
            else:
                features = torch.ones((batch.x.shape[0], 1), dtype=torch.float32).to(device)
            out, h = model(features, batch.edge_index)
            y_hat = out[:batch.batch_size] # Prediction
            y = batch.y[:batch.batch_size] # True value
        y_hat  = y_hat.softmax(dim=1) # Get probability of fraud
        
        AUC = roc_auc_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])
        AP = average_precision_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])

        AUC_list.append(AUC)
        AP_list.append(AP)

        for percentile_q in percentile_q_list:
            cutoff = np.percentile(y_hat.cpu().detach().numpy()[:,1], percentile_q)
            y_hat_hard = (y_hat[:,1] >= cutoff)*1
            precision = precision_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())
            recall = recall_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())
            F1 = f1_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())

            precision_dict[percentile_q].append(precision)
            recall_dict[percentile_q].append(recall)
            F1_dict[percentile_q].append(F1)

    return(AUC_list, AP_list, precision_dict, recall_dict, F1_dict)

def evaluate_model_deep_AUC(data, model, test_mask, device="cpu", loader=None, use_intrinsic=True):
    model.eval()
    test_mask_new = resample_mask(test_mask)
    if loader is None:
        model.eval()
        if use_intrinsic:
            features = data.x
        else:
            features = torch.ones((data.x.shape[0], 1), dtype=torch.float32).to(device)
        out, h = model(features, data.edge_index.to(device))
        y_hat = out[test_mask_new].to(device) # Prediction
        y = data.y[test_mask_new].to(device) # True value
        
    else:
        batch = next(iter(loader))
        batch = batch.to(device, 'edge_index')
        if use_intrinsic:
            features = batch.x
        else:
            features = torch.ones((batch.x.shape[0], 1), dtype=torch.float32).to(device)
        out, h = model(features, batch.edge_index)
        y_hat = out[:batch.batch_size] # Prediction
        y = batch.y[:batch.batch_size] # True value
    y_hat  = y_hat.softmax(dim=1) # Get probability of fraud
    
    AUC = roc_auc_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])
    AP = average_precision_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])

    return(AUC, AP)
    
def evaluate_model_deep_PRF(data, model, test_mask, percentile_q=90, device="cpu", loader=None, use_intrinsic=True):
    model.eval()
    test_mask_new = resample_mask(test_mask)
    if loader is None:
        model.eval()
        if use_intrinsic:
            features = data.x
        else:
            features = torch.ones((data.x.shape[0], 1), dtype=torch.float32).to(device)
        out, h = model(features, data.edge_index.to(device))
        y_hat = out[test_mask_new].to(device) # Prediction
        y = data.y[test_mask_new].to(device) # True value
        
    else:
        batch = next(iter(loader))
        batch = batch.to(device, 'edge_index')
        if use_intrinsic:
            features = batch.x
        else:
            features = torch.ones((batch.x.shape[0], 1), dtype=torch.float32).to(device)
        out, h = model(features, batch.edge_index)
        y_hat = out[:batch.batch_size] # Prediction
        y = batch.y[:batch.batch_size] # True value
    y_hat  = y_hat.softmax(dim=1) # Get probability of fraud
    
    cutoff = np.percentile(y_hat.cpu().detach().numpy()[:,1], percentile_q)
    y_hat_hard = (y_hat[:,1] >= cutoff)*1
    precision = precision_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())
    recall = recall_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())
    F1 = f1_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())

    return(precision, recall, F1)

def evaluate_if(model, x_test, y_test, percentile_q_list = [99], n_samples=100):
    AUC_list = []
    AP_list = []

    precision_dict = dict()
    recall_dict = dict()
    F1_dict = dict()

    x_test = x_test.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()

    for percentile_q in percentile_q_list:
        precision_dict[percentile_q] = []
        recall_dict[percentile_q] = []
        F1_dict[percentile_q] = []
    
    for _ in tqdm(range(n_samples)):
        x_new, y_new = stratified_sampling(x_test, y_test)

        model.fit(x_new)
        y_pred = model.score_samples(x_new)
        y_pred = -y_pred

        # inspect unique classes in y_new
        unique_classes = np.unique(y_new)
        
        if len(unique_classes) < 2:
            # if only one class present, assign neutral scores
            AUC = 0.5 
            AP = 0.0
        else:
            # 🎯 最終修正：確保 roc_auc_score 知道這是二元問題，並使用分數
            # 這裡我們只傳遞二元分數，並確保標籤是二元的。
            try:
                AUC = roc_auc_score(y_new, y_pred)
                AP = average_precision_score(y_new, y_pred)
            except ValueError:
                # catch any ValueError that might arise and assign neutral scores
                AUC = 0.5
                AP = 0.0

        AUC_list.append(AUC)
        AP_list.append(AP)

        for percentile_q in percentile_q_list:
            cutoff = np.percentile(y_pred, percentile_q)
            y_pred_hard = (y_pred >= cutoff)*1
            precision = precision_score(y_new, y_pred_hard)
            recall = recall_score(y_new, y_pred_hard)
            F1 = f1_score(y_new, y_pred_hard)

            precision_dict[percentile_q].append(precision)
            recall_dict[percentile_q].append(recall)
            F1_dict[percentile_q].append(F1)
        
    return(AUC_list, AP_list, precision_dict, recall_dict, F1_dict)

def save_results_TI(AUC_list, AP_list, model_name):
    res_dict = {'AUC': AUC_list, 'AP': AP_list}
    df = pd.DataFrame(res_dict)
    df.to_csv('res/'+model_name+'_TI.csv')

def save_results_TD(precision_dict, recall_dict, F1_dict, model_name):
    res_dict = dict()
    for key in precision_dict.keys():
        res_dict['precision_'+str(key)] = precision_dict[key]
        res_dict['recall_'+str(key)] = recall_dict[key]
        res_dict['F1_'+str(key)] = F1_dict[key]
    
    df = pd.DataFrame(res_dict)
    df.to_csv('res/'+model_name+'_TD.csv')


def smote_mask(mask, features, labels, k_neighbors=5, random_state=None):
    """
    SMOTE: Synthetic Minority Over-sampling Technique applied within a mask.
    
    Generates synthetic samples for the minority class using k-NN in feature space.
    Only operates on samples within the mask.

    Parameters:
    - mask: boolean mask (1D tensor or array) - which samples to resample
    - features: feature matrix (n_samples, n_features), numpy array or torch tensor
    - labels: label vector (n_samples,)
    - k_neighbors: number of nearest neighbors for synthetic sample generation
    - random_state: seed for reproducibility

    Returns:
    - expanded_features: feature matrix with synthetic samples (numpy array)
    - expanded_labels: label vector with synthetic labels (numpy array)
    - expanded_mask: boolean mask indicating which samples are in the training set
    """
    # Convert to numpy if needed
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

    # Get samples in mask
    idx_mask = np.where(mask_np)[0]
    features_masked = features_np[idx_mask]
    labels_masked = labels_np[idx_mask]

    # Identify minority and majority classes
    unique_classes, class_counts = np.unique(labels_masked, return_counts=True)
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]
    
    minority_count = np.min(class_counts)
    majority_count = np.max(class_counts)

    # Get indices of minority and majority samples
    idx_minority = np.where(labels_masked == minority_class)[0]
    idx_majority = np.where(labels_masked == majority_class)[0]

    # Fit kNN on majority class to find neighbors
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(idx_majority))).fit(features_masked[idx_majority])
    distances, indices = nbrs.kneighbors(features_masked[idx_minority])

    # Generate synthetic samples
    rnd = np.random.RandomState(random_state)
    n_synthetic = majority_count - minority_count  # Generate this many synthetic samples
    
    synthetic_features = []
    synthetic_labels = []

    # Only generate synthetic samples if needed (n_synthetic > 0)
    if n_synthetic > 0:
        for i in range(n_synthetic):
            # Randomly pick a minority sample
            minority_idx = rnd.choice(len(idx_minority))
            # Randomly pick one of its k nearest neighbors (from majority class)
            neighbor_idx = rnd.choice(min(k_neighbors, len(idx_majority)))
            
            x_minority = features_masked[idx_minority[minority_idx]]
            # Map back to original indices
            neighbor_original_idx = idx_majority[indices[minority_idx, neighbor_idx]]
            x_neighbor = features_masked[neighbor_original_idx]
            
            # Generate synthetic sample
            alpha = rnd.uniform(0, 1)
            synthetic_sample = x_minority + alpha * (x_neighbor - x_minority)
            synthetic_features.append(synthetic_sample)
            synthetic_labels.append(minority_class)

        # Combine original and synthetic samples
        synthetic_features = np.array(synthetic_features)
        synthetic_labels = np.array(synthetic_labels)
        
        expanded_features = np.vstack([features_np, synthetic_features])
        expanded_labels = np.concatenate([labels_np, synthetic_labels])
    else:
        # If already balanced (n_synthetic == 0), just return original
        expanded_features = features_np
        expanded_labels = labels_np

    # Create expanded mask
    expanded_mask = np.zeros(len(expanded_labels), dtype=bool)
    expanded_mask[:len(mask_np)] = mask_np
    # Mark synthetic samples as part of training set
    expanded_mask[len(labels_np):] = True

    # Convert back to torch if input was torch
    if is_torch_feat:
        expanded_features = torch.from_numpy(expanded_features).to(features.dtype).to(features.device)
    if is_torch_labels:
        expanded_labels = torch.from_numpy(expanded_labels).to(labels.device)
    if is_torch_mask:
        expanded_mask = torch.from_numpy(expanded_mask).to(mask.device)

    return expanded_features, expanded_labels, expanded_mask


def graph_smote_mask(mask, features, labels, edge_index, k_neighbors=5, 
                     similarity_metric='cosine', random_state=None):
    """
    GraphSMOTE: Feature-space SMOTE + Heuristic Edge Connection
    
    This is a practical replacement for complex graph-aware SMOTE methods.
    Strategy:
    1. Apply standard SMOTE to feature matrix (ignores graph structure)
    2. Connect synthetic nodes via k-NN heuristic in feature space
    3. Return expanded features + new adjacency matrix for GNN input
    
    Parameters:
    - mask: boolean mask (1D tensor or array)
    - features: feature matrix (n_samples, n_features)
    - labels: label vector (n_samples,)
    - edge_index: edge indices (2, n_edges) as torch tensor or numpy array
    - k_neighbors: number of nearest neighbors for heuristic edge connection
    - similarity_metric: 'cosine' or 'euclidean' (used for edge connection)
    - random_state: seed for reproducibility

    Returns:
    - expanded_features: feature matrix with synthetic samples
    - expanded_labels: label vector with synthetic labels
    - expanded_mask: boolean mask for training set
    - expanded_edge_index: edge indices (original edges + heuristic synthetic connections)
    """
    return feature_smote_with_heuristic_edges(
        mask=mask,
        features=features,
        labels=labels,
        edge_index=edge_index,
        k_neighbors=k_neighbors,
        heuristic='knn',
        random_state=random_state
    )


def adjust_mask_to_ratio(mask, labels, target_ratio, random_state=None):
    """
    調整mask中的class比例到指定的target_ratio。
    
    Parameters:
    - mask: boolean mask (1D tensor or array)
    - labels: label vector (n_samples,)
    - target_ratio: 目標比例 (majority_count / minority_count)
                   e.g., 1.0 for 1:1, 2.0 for 2:1 (majority:minority)
    - random_state: seed for reproducibility
    
    Returns:
    - new_mask: adjusted boolean mask
    - original_majority_count: 原始多數類數量
    - original_minority_count: 原始少數類數量
    - new_majority_count: 調整後多數類數量
    """
    is_torch = isinstance(mask, torch.Tensor)
    
    if is_torch:
        mask_np = mask.cpu().numpy().astype(bool)
        labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    else:
        mask_np = np.array(mask).astype(bool)
        labels_np = np.array(labels)
    
    mask_np = np.atleast_1d(mask_np)
    
    # Get samples in mask
    idx_mask = np.where(mask_np)[0]
    if idx_mask.size == 0:
        return (mask.clone() if is_torch else mask_np), 0, 0, 0
    
    labels_in_mask = labels_np[idx_mask]
    
    # Identify classes
    unique_classes, counts = np.unique(labels_in_mask, return_counts=True)
    if unique_classes.size <= 1:
        return (mask.clone() if is_torch else mask_np), 0, 0, 0
    
    minority_class = unique_classes[np.argmin(counts)]
    majority_class = unique_classes[np.argmax(counts)]
    
    minority_count = np.min(counts)
    majority_count = np.max(counts)
    
    # Calculate new majority count based on target ratio
    # target_ratio = new_majority_count / minority_count
    new_majority_count = int(np.round(target_ratio * minority_count))
    new_majority_count = min(new_majority_count, majority_count)  # Don't oversample
    
    # Get indices of each class
    idx_minority = idx_mask[np.where(labels_in_mask == minority_class)[0]]
    idx_majority = idx_mask[np.where(labels_in_mask == majority_class)[0]]
    
    # Random sampling for majority class
    rnd = np.random.RandomState(random_state)
    selected_majority = rnd.choice(idx_majority, size=new_majority_count, replace=False)
    
    # Combine selected indices
    selected_all = np.concatenate([idx_minority, selected_majority])
    
    # Create new mask
    new_mask = np.zeros_like(mask_np, dtype=bool)
    new_mask[selected_all] = True
    
    if is_torch:
        return torch.from_numpy(new_mask).to(mask.device), majority_count, minority_count, new_majority_count
    
    return new_mask, majority_count, minority_count, new_majority_count