import sys
import os
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
from tqdm import tqdm

# Ensure repository root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC_PATH = os.path.join(ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.methods.utils.functionsNetworkX import *
try:
    from src.methods.utils.functionsNetworKit import *
except ImportError:
    pass
from src.methods.utils.functionsTorch import *
from src.methods.utils.GNN import *
from src.utils.Network import *
from src.methods.utils.decoder import *

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0.0, checkpoint_path='checkpoint.pt', monitor='val_ap'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_ap_max = -np.Inf

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

def positional_features_calc(ntw, alpha_pr: float, alpha_ppr: float=None, fraud_dict_train: dict=None, fraud_dict_test: dict=None, ntw_name: str=None, use_intrinsic: bool = True):
    print("networkx: ")
    ntw_nx = ntw.get_network_nx()
    features_nx_df = local_features_nx(ntw_nx, alpha_pr, alpha_ppr, fraud_dict_train=fraud_dict_train, ntw_name=ntw_name)
    print("networkit: ")
    try:
        ntw_nk = ntw.get_network_nk()
        features_nk_df = features_nk(ntw_nk, ntw_name=ntw_name)
    except NameError as e:
        print(f"[Warning] features_nk not found ({e}). Falling back to NetworkX features.")
        ntw_nx = ntw.get_network_nx()
        features_nk_df = features_nx(ntw_nx, ntw_name=ntw_name)
    except Exception as e:
        print(f"[Warning] features_nk failed ({e}). Falling back to NetworkX features.")
        ntw_nx = ntw.get_network_nx()
        features_nk_df = features_nx(ntw_nx, ntw_name=ntw_name)
    if use_intrinsic:
        print("intrinsic and summary: ")
        X = ntw.get_features(full=True)
        features_df = pd.concat([X, features_nx_df, features_nk_df], axis=1)
    else:
        features_df = pd.concat([features_nx_df, features_nk_df], axis=1)
    features_df["fraud"] = [fraud_dict_test[x] for x in features_df.index]
    return features_df

def train_model_shallow(x_train, y_train, n_epochs_decoder: int, lr: float, n_layers_decoder: int=2, hidden_dim_decoder: int=10, device_decoder: str="cpu"):
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
    return decoder

def train_model_deep(data, model, train_mask, n_epochs, lr, batch_size, loader=None, use_intrinsic=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    if use_intrinsic:
        features = data.x
    else:
        features = torch.ones((data.x.shape[0], 1), dtype=torch.float32).to(device)
    def train_GNN():
        model.train()
        optimizer.zero_grad()
        y_hat, h = model(features, data.edge_index.to(device))
        y = data.y
        loss = criterion(y_hat[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        return loss
    for _ in range(n_epochs):
        loss_train = train_GNN()
        print('Epoch: {:03d}, Loss: {:.4f}'.format(_, loss_train))

def stratified_sampling(x_test, y_test):
    n_samples = x_test.shape[0]
    x_new, y_new = resample(x_test, y_test, n_samples=n_samples, stratify=y_test)
    return x_new, y_new

def random_undersample_majority(x, y, return_index=False, random_state=None):
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
        return x_res, y_res, selected_idx
    return x_res, y_res

def random_undersample_mask(mask, labels, target_ratio=1.0, random_state=None):
    import numpy as _np
    try:
        import torch as _torch
    except Exception:
        _torch = None
    is_torch = _torch is not None and isinstance(mask, _torch.Tensor)
    if is_torch:
        mask_np = mask.cpu().numpy().astype(_np.bool_)
        labels_np = labels.cpu().numpy() if isinstance(labels, _torch.Tensor) else _np.array(labels)
    else:
        mask_np = _np.array(mask).astype(_np.bool_)
        labels_np = _np.array(labels)
    mask_np = _np.atleast_1d(mask_np)
    idx = _np.where(mask_np)[0]
    if idx.size == 0:
        return mask.clone() if is_torch and _torch is not None else mask_np
    lbls_in_mask = labels_np[idx]
    classes, counts = _np.unique(lbls_in_mask, return_counts=True)
    if classes.size <= 1:
        return mask.clone() if is_torch and _torch is not None else mask_np
    
    minority_idx_lbl = _np.argmin(counts)
    minority_count = int(counts[minority_idx_lbl])
    desired_majority = int(_np.floor(minority_count * target_ratio))
    
    rnd = _np.random.RandomState(random_state)
    selected = []
    for cls, cnt in zip(classes, counts):
        idx_cls = idx[_np.where(lbls_in_mask == cls)[0]]
        if cnt <= minority_count:
            selected.append(idx_cls)
        else:
            take = min(desired_majority, idx_cls.shape[0])
            selected.append(rnd.choice(idx_cls, size=take, replace=False))
    selected = _np.concatenate(selected)
    original_shape = mask.shape if is_torch else _np.array(mask).shape
    new_mask = _np.zeros(original_shape, dtype=_np.bool_)
    new_mask[selected] = True
    if is_torch and _torch is not None:
        return _torch.from_numpy(new_mask).to(mask.device)
    return new_mask

def evaluate_model_shallow_AUC(model, x_test, y_test, device="cpu", percentile_q=99):
    model.eval()
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    y_pred = model(x_test)
    y_pred = y_pred.softmax(dim=1)
    AUC = roc_auc_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    AP = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    cutoff = np.percentile(y_pred.cpu().detach().numpy()[:,1], percentile_q)
    y_pred_hard = (y_pred[:,1] >= cutoff)*1
    F1 = f1_score(y_test.cpu().detach().numpy(), y_pred_hard.cpu().detach().numpy())
    return AUC, AP, F1

def evaluate_model_shallow_PRF(model, x_test, y_test, percentile_q=99, device="cpu"):
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
    return precision, recall, F1

def resample_mask(test_mask, p=0.5):
    sample_size = int(np.floor(test_mask.sum()*p))
    true_indices = [i for i, val in enumerate(test_mask) if val]
    sampled_indices = random.sample(true_indices, min(sample_size, len(true_indices)))
    output_tensor = torch.zeros_like(test_mask, dtype=torch.bool)
    output_tensor[sampled_indices] = True
    return output_tensor

def subsample_true_values_tensor(test_mask, p=0.5):
    sample_size = int(np.floor(test_mask.sum()*p))
    true_indices = torch.where(test_mask)[0]
    sampled_indices = true_indices[torch.randperm(true_indices.size(0))[:min(sample_size, true_indices.size(0))]]
    output_tensor = torch.zeros_like(test_mask, dtype=torch.bool)
    output_tensor[sampled_indices] = True
    return output_tensor

def evaluate_model_deep(data, model, test_mask, percentile_q_list=[99], n_samples=100, device="cpu", loader=None, use_intrinsic=True):
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
            y_hat = out[test_mask_new].to(device)
            y = data.y[test_mask_new].to(device)
        else:
            batch = next(iter(loader))
            batch = batch.to(device, 'edge_index')
            if use_intrinsic:
                features = batch.x
            else:
                features = torch.ones((batch.x.shape[0], 1), dtype=torch.float32).to(device)
            out, h = model(features, batch.edge_index)
            y_hat = out[:batch.batch_size]
            y = batch.y[:batch.batch_size]
        y_hat = y_hat.softmax(dim=1)
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
    return AUC_list, AP_list, precision_dict, recall_dict, F1_dict

def evaluate_model_deep_AUC(data, model, test_mask, device="cpu", loader=None, use_intrinsic=True, percentile_q=90):
    model.eval()
    test_mask_new = resample_mask(test_mask)
    if loader is None:
        model.eval()
        if use_intrinsic:
            features = data.x
        else:
            features = torch.ones((data.x.shape[0], 1), dtype=torch.float32).to(device)
        out, h = model(features, data.edge_index.to(device))
        y_hat = out[test_mask_new].to(device)
        y = data.y[test_mask_new].to(device)
    else:
        batch = next(iter(loader))
        batch = batch.to(device, 'edge_index')
        if use_intrinsic:
            features = batch.x
        else:
            features = torch.ones((batch.x.shape[0], 1), dtype=torch.float32).to(device)
        out, h = model(features, batch.edge_index)
        y_hat = out[:batch.batch_size]
        y = batch.y[:batch.batch_size]
    y_hat = y_hat.softmax(dim=1)
    AUC = roc_auc_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])
    AP = average_precision_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])
    cutoff = np.percentile(y_hat.cpu().detach().numpy()[:,1], percentile_q)
    y_hat_hard = (y_hat[:,1] >= cutoff)*1
    F1 = f1_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())
    return AUC, AP, F1

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
        y_hat = out[test_mask_new].to(device)
        y = data.y[test_mask_new].to(device)
    else:
        batch = next(iter(loader))
        batch = batch.to(device, 'edge_index')
        if use_intrinsic:
            features = batch.x
        else:
            features = torch.ones((batch.x.shape[0], 1), dtype=torch.float32).to(device)
        out, h = model(features, batch.edge_index)
        y_hat = out[:batch.batch_size]
        y = batch.y[:batch.batch_size]
    y_hat = y_hat.softmax(dim=1)
    cutoff = np.percentile(y_hat.cpu().detach().numpy()[:,1], percentile_q)
    y_hat_hard = (y_hat[:,1] >= cutoff)*1
    precision = precision_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())
    recall = recall_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())
    F1 = f1_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())
    return precision, recall, F1

def evaluate_if(model, x_test, y_test, percentile_q_list=[99], n_samples=100):
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
        unique_classes = np.unique(y_new)
        if len(unique_classes) < 2:
            AUC = 0.5
            AP = 0.0
        else:
            try:
                AUC = roc_auc_score(y_new, y_pred)
                AP = average_precision_score(y_new, y_pred)
            except ValueError:
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
    return AUC_list, AP_list, precision_dict, recall_dict, F1_dict

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

def smote_mask(mask, features, labels, ratio=None, k_neighbors=5, random_state=None):
    """
    SMOTE applied within a mask, generating synthetic minority samples up to the target ratio.
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
        expanded_features = features_np
        expanded_labels = labels_np
        expanded_mask = np.zeros(len(labels_np), dtype=bool)
        expanded_mask[:len(mask_np)] = mask_np
        return expanded_features, expanded_labels, expanded_mask
        
    features_masked = features_np[idx_mask]
    features_masked = np.nan_to_num(features_masked, nan=0.0, posinf=0.0, neginf=0.0)
    labels_masked = labels_np[idx_mask]
    
    unique_classes, class_counts = np.unique(labels_masked, return_counts=True)
    if unique_classes.size < 2:
        expanded_features = features_np
        expanded_labels = labels_np
        expanded_mask = np.zeros(len(labels_np), dtype=bool)
        expanded_mask[:len(mask_np)] = mask_np
        return expanded_features, expanded_labels, expanded_mask
        
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]
    minority_count = np.min(class_counts)
    majority_count = np.max(class_counts)
    
    if ratio is None:
        target_minority_count = majority_count
    else:
        target_minority_count = int(np.round(majority_count / ratio))
        
    if target_minority_count <= minority_count:
        expanded_features = features_np
        expanded_labels = labels_np
        expanded_mask = np.zeros(len(labels_np), dtype=bool)
        expanded_mask[:len(mask_np)] = mask_np
        if is_torch_feat:
            expanded_features = torch.from_numpy(expanded_features).to(features.dtype).to(features.device)
        if is_torch_labels:
            expanded_labels = torch.from_numpy(expanded_labels).to(labels.device)
        if is_torch_mask:
            expanded_mask = torch.from_numpy(expanded_mask).to(mask.device)
        return expanded_features, expanded_labels, expanded_mask
        
    smote = SMOTE(
        sampling_strategy={minority_class: target_minority_count},
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

def reweighted_graph_smote_mask(mask, features, labels, edge_index, ratio=None, k_neighbors=5, similarity_metric='cosine', random_state=None):
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
        expanded_features = features_np
        expanded_labels = labels_np
        expanded_mask = mask_np
        expanded_edge_index = edge_index_np
        expanded_edge_weights = np.ones(max(edge_index_np.shape[1], 0), dtype=float) if edge_index_np.size else np.array([], dtype=float)
        if is_torch_feat:
            expanded_features = torch.from_numpy(expanded_features).to(features.dtype).to(features.device)
        if is_torch_labels:
            expanded_labels = torch.from_numpy(expanded_labels).to(labels.device)
        if is_torch_mask:
            expanded_mask = torch.from_numpy(expanded_mask).to(mask.device)
        if is_torch_edge:
            expanded_edge_index = torch.from_numpy(expanded_edge_index).to(edge_index.device)
            expanded_edge_weights = torch.from_numpy(expanded_edge_weights).to(edge_index.device)
        return expanded_features, expanded_labels, expanded_mask, expanded_edge_index, expanded_edge_weights

    features_masked = features_np[idx_mask]
    features_masked = np.nan_to_num(features_masked, nan=0.0, posinf=0.0, neginf=0.0)
    labels_masked = labels_np[idx_mask]
    
    unique_classes, class_counts = np.unique(labels_masked, return_counts=True)
    if unique_classes.size < 2:
        expanded_features = features_np
        expanded_labels = labels_np
        expanded_mask = mask_np
        expanded_edge_index = edge_index_np
        expanded_edge_weights = np.ones(max(edge_index_np.shape[1], 0), dtype=float) if edge_index_np.size else np.array([], dtype=float)
        if is_torch_feat:
            expanded_features = torch.from_numpy(expanded_features).to(features.dtype).to(features.device)
        if is_torch_labels:
            expanded_labels = torch.from_numpy(expanded_labels).to(labels.device)
        if is_torch_mask:
            expanded_mask = torch.from_numpy(expanded_mask).to(mask.device)
        if is_torch_edge:
            expanded_edge_index = torch.from_numpy(expanded_edge_index).to(edge_index.device)
            expanded_edge_weights = torch.from_numpy(expanded_edge_weights).to(edge_index.device)
        return expanded_features, expanded_labels, expanded_mask, expanded_edge_index, expanded_edge_weights

    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]
    minority_count = np.min(class_counts)
    majority_count = np.max(class_counts)

    if ratio is None:
        target_minority_count = majority_count
    else:
        target_minority_count = int(np.round(majority_count / ratio))

    if target_minority_count <= minority_count:
        expanded_features = features_np
        expanded_labels = labels_np
        expanded_mask = mask_np
        expanded_edge_index = edge_index_np
        expanded_edge_weights = np.ones(max(edge_index_np.shape[1], 0), dtype=float) if edge_index_np.size else np.array([], dtype=float)
        if is_torch_feat:
            expanded_features = torch.from_numpy(expanded_features).to(features.dtype).to(features.device)
        if is_torch_labels:
            expanded_labels = torch.from_numpy(expanded_labels).to(labels.device)
        if is_torch_mask:
            expanded_mask = torch.from_numpy(expanded_mask).to(mask.device)
        if is_torch_edge:
            expanded_edge_index = torch.from_numpy(expanded_edge_index).to(edge_index.device)
            expanded_edge_weights = torch.from_numpy(expanded_edge_weights).to(edge_index.device)
        return expanded_features, expanded_labels, expanded_mask, expanded_edge_index, expanded_edge_weights

    smote = SMOTE(
        sampling_strategy={minority_class: target_minority_count},
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
        expanded_edge_weights = np.ones(max(edge_index_np.shape[1], 0), dtype=float) if edge_index_np.size else np.array([], dtype=float)
    else:
        clean_features_np = np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0)
        expanded_features = np.vstack([clean_features_np, X_smote[n_original:]])
        expanded_labels = np.concatenate([labels_np, y_smote[n_original:]])
        expanded_mask = np.zeros(len(expanded_labels), dtype=bool)
        expanded_mask[:len(mask_np)] = mask_np
        expanded_mask[len(labels_np):] = True
        
        if edge_index_np.size > 0:
            base_edges = []
            base_weights = []
            if edge_index_np.shape[1] > 0:
                for edge_pair in edge_index_np.T:
                    base_edges.append([int(edge_pair[0]), int(edge_pair[1])])
                    base_weights.append(1.0)
            if base_edges:
                base_edges_array = np.array(base_edges).T
                expanded_edge_index = np.hstack([edge_index_np, base_edges_array]) if edge_index_np.size else base_edges_array
            else:
                expanded_edge_index = edge_index_np
            expanded_edge_weights = np.array(base_weights, dtype=float)
        else:
            expanded_edge_index = edge_index_np
            expanded_edge_weights = np.array([], dtype=float)
            
        if n_synthetic > 0:
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, features_masked.shape[0]), algorithm='ball_tree').fit(features_masked)
            new_edges = []
            new_weights = []
            for synthetic_idx in range(n_synthetic):
                synthetic_feature = X_smote[n_original + synthetic_idx].reshape(1, -1)
                distances, indices = nbrs.kneighbors(synthetic_feature)
                for neighbor_idx in indices[0][1:]:
                    neighbor_global_idx = int(idx_mask[neighbor_idx])
                    synthetic_global_idx = int(features_np.shape[0] + synthetic_idx)
                    new_edges.append([neighbor_global_idx, synthetic_global_idx])
                    new_edges.append([synthetic_global_idx, neighbor_global_idx])
                    if np.isfinite(distances[0][1:]).all() and distances[0][1:] is not None:
                        weight = float(np.exp(-np.mean(distances[0][1:]) / max(float(np.mean(distances[0][1:])), 1e-8)))
                    else:
                        weight = 1.0
                    new_weights.extend([min(1.0, max(0.0, weight)), min(1.0, max(0.0, weight))])
            if new_edges:
                new_edges_array = np.array(new_edges).T
                expanded_edge_index = np.hstack([edge_index_np, new_edges_array]) if edge_index_np.size else new_edges_array
                expanded_edge_weights = np.concatenate([expanded_edge_weights, np.array(new_weights, dtype=float)])
            else:
                expanded_edge_weights = np.ones(max(expanded_edge_index.shape[1], 0), dtype=float) if expanded_edge_index.size else np.array([], dtype=float)
        else:
            expanded_edge_weights = np.ones(max(expanded_edge_index.shape[1], 0), dtype=float) if expanded_edge_index.size else np.array([], dtype=float)
            
    if is_torch_feat:
        expanded_features = torch.from_numpy(expanded_features).to(features.dtype).to(features.device)
    if is_torch_labels:
        expanded_labels = torch.from_numpy(expanded_labels).to(labels.device)
    if is_torch_mask:
        expanded_mask = torch.from_numpy(expanded_mask).to(mask.device)
    if is_torch_edge:
        expanded_edge_index = torch.from_numpy(expanded_edge_index).to(edge_index.device)
        expanded_edge_weights = torch.from_numpy(expanded_edge_weights).to(edge_index.device)
    return expanded_features, expanded_labels, expanded_mask, expanded_edge_index, expanded_edge_weights

def graph_smote_mask(mask, features, labels, edge_index, ratio=None, k_neighbors=5, similarity_metric='cosine', random_state=None):
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
        expanded_features = features_np
        expanded_labels = labels_np
        expanded_mask = mask_np
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

    features_masked = features_np[idx_mask]
    features_masked = np.nan_to_num(features_masked, nan=0.0, posinf=0.0, neginf=0.0)
    labels_masked = labels_np[idx_mask]
    
    unique_classes, class_counts = np.unique(labels_masked, return_counts=True)
    if unique_classes.size < 2:
        expanded_features = features_np
        expanded_labels = labels_np
        expanded_mask = mask_np
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

    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]
    minority_count = np.min(class_counts)
    majority_count = np.max(class_counts)

    if ratio is None:
        target_minority_count = majority_count
    else:
        target_minority_count = int(np.round(majority_count / ratio))

    if target_minority_count <= minority_count:
        expanded_features = features_np
        expanded_labels = labels_np
        expanded_mask = mask_np
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

    smote = SMOTE(
        sampling_strategy={minority_class: target_minority_count},
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
        
        if n_synthetic > 0:
            clean_features = np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0)
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, clean_features.shape[0]), algorithm='ball_tree').fit(clean_features)
            new_edges = []
            for synthetic_idx in range(n_synthetic):
                synthetic_feature = X_smote[n_original + synthetic_idx].reshape(1, -1)
                _, indices = nbrs.kneighbors(synthetic_feature)
                for neighbor_idx in indices[0][1:]:
                    new_edges.append([int(neighbor_idx), int(features_np.shape[0] + synthetic_idx)])
                    new_edges.append([int(features_np.shape[0] + synthetic_idx), int(neighbor_idx)])
            if new_edges:
                new_edges_array = np.array(new_edges).T
                expanded_edge_index = np.hstack([edge_index_np, new_edges_array]) if edge_index_np.size else new_edges_array
            else:
                expanded_edge_index = edge_index_np
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

def graph_ensemble_smote_mask(mask, features, labels, edge_index, ratio=None, k_neighbors=5, random_state=None):
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
        expanded_features = features_np
        expanded_labels = labels_np
        expanded_mask = mask_np
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

    features_masked = features_np[idx_mask]
    features_masked = np.nan_to_num(features_masked, nan=0.0, posinf=1e5, neginf=-1e5)
    labels_masked = labels_np[idx_mask]
    
    unique_classes, class_counts = np.unique(labels_masked, return_counts=True)
    if unique_classes.size < 2:
        expanded_features = features_np
        expanded_labels = labels_np
        expanded_mask = mask_np
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

    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]
    minority_count = np.min(class_counts)
    majority_count = np.max(class_counts)

    if ratio is None:
        target_minority_count = majority_count
    else:
        target_minority_count = int(np.round(majority_count / ratio))

    if target_minority_count <= minority_count:
        expanded_features = features_np
        expanded_labels = labels_np
        expanded_mask = mask_np
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

    smote = SMOTE(
        sampling_strategy={minority_class: target_minority_count},
        k_neighbors=max(1, min(int(k_neighbors), minority_count - 1)),
        random_state=random_state
    )
    X_smote, y_smote = smote.fit_resample(features_masked, labels_masked)
    X_smote = np.nan_to_num(X_smote, nan=0.0, posinf=1e5, neginf=-1e5)
    n_original = features_masked.shape[0]
    n_synthetic = X_smote.shape[0] - n_original
    
    if n_synthetic <= 0:
        expanded_features = features_np
        expanded_labels = labels_np
        expanded_mask = mask_np
        expanded_edge_index = edge_index_np
    else:
        clean_features_np = np.nan_to_num(features_np, nan=0.0, posinf=1e5, neginf=-1e5)
        expanded_features = np.vstack([clean_features_np, X_smote[n_original:]])
        expanded_labels = np.concatenate([labels_np, y_smote[n_original:]])
        expanded_mask = np.zeros(len(expanded_labels), dtype=bool)
        expanded_mask[:len(mask_np)] = mask_np
        expanded_mask[len(labels_np):] = True
        
        if n_synthetic > 0 and edge_index_np.size > 0:
            clean_features_np = np.nan_to_num(features_np, nan=0.0, posinf=1e5, neginf=-1e5)
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, clean_features_np.shape[0]), algorithm='ball_tree').fit(clean_features_np)
            new_edges = []
            for synthetic_idx in range(n_synthetic):
                synthetic_feature = np.nan_to_num(X_smote[n_original + synthetic_idx].reshape(1, -1), nan=0.0, posinf=1e5, neginf=-1e5)
                _, indices = nbrs.kneighbors(synthetic_feature)
                for neighbor_idx in indices[0][1:]:
                    new_edges.append([neighbor_idx, int(clean_features_np.shape[0] + synthetic_idx)])
                    new_edges.append([int(clean_features_np.shape[0] + synthetic_idx), neighbor_idx])
            if new_edges:
                new_edges_array = np.array(new_edges).T
                expanded_edge_index = np.hstack([edge_index_np, new_edges_array]) if edge_index_np.size else new_edges_array
            else:
                expanded_edge_index = edge_index_np
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
