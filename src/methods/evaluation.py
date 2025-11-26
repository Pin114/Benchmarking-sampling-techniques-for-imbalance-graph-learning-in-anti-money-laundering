from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import random

from sklearn.ensemble import IsolationForest

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

    # 確保 mask 是 1D 且類型正確，這能同時解決 ValueError 和 TypeError
    is_torch = _torch is not None and isinstance(mask, _torch.Tensor)

    if is_torch:
        # 轉換為 NumPy 陣列時，強制轉換為 bool 以避免 object_ 類型
        mask_np = mask.cpu().numpy().astype(_np.bool_)
        labels_np = labels.cpu().numpy() if isinstance(labels, _torch.Tensor) else _np.array(labels)
    else:
        # 對 NumPy 輸入，也強制轉換為 bool
        mask_np = _np.array(mask).astype(_np.bool_)
        labels_np = _np.array(labels)

    # 🎯 修正點 1: 確保 mask_np 至少是 1D，同時解決 ValueError 和 TypeError
    mask_np = _np.atleast_1d(mask_np) 

    idx = _np.where(mask_np)[0]
    if idx.size == 0:
        # nothing to do
        return mask.clone() if is_torch and _torch is not None else mask_np

    lbls_in_mask = labels_np[idx]
    classes, counts = _np.unique(lbls_in_mask, return_counts=True)
    if classes.size <= 1:
        # single class inside mask
        return mask.clone() if is_torch and _torch is not None else mask_np

    minority_count = int(_np.min(counts))
    desired_majority = int(_np.floor(minority_count * target_ratio))

    rnd = _np.random.RandomState(random_state)

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
    
    # 🎯 修正點 2: 統一使用原始輸入的 shape 來創建新 mask，並確保 dtype 為 bool
    # 由於在函式開始時，我們已確保 mask_np 是正確的 bool 類型且至少 1D，這裡可以安全使用原始 shape
    
    # 獲取原始 mask 的尺寸，以確保返回的 mask 尺寸一致
    original_shape = mask.shape if is_torch else _np.array(mask).shape
    new_mask = _np.zeros(original_shape, dtype=_np.bool_)
         
    new_mask[selected] = True

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

        AUC = roc_auc_score(y_new, y_pred)
        AP = average_precision_score(y_new, y_pred)

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