# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, f1_score

# 繼承原始 experiments_supervised 中的所有導入和輔助類別
from src.methods.experiments_supervised import *

def intrinsic_features(
    ntw, train_mask, test_mask, n_layers_decoder, hidden_dim_decoder, lr, n_epochs_decoder, percentile_q=99, clip_norm=1.0,
    sampling="none", target_ratio=None, k_neighbors=5, random_state=None, assert_tag=""
):
    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    X_train, y_train, X_test, y_test = ntw.get_train_test_split_intrinsic(train_mask, test_mask, device=device_decoder)
    if sampling == "smote" and target_ratio is not None:
        train_mask_np = train_mask.cpu().numpy().astype(bool) if isinstance(train_mask, torch.Tensor) else np.array(train_mask).astype(bool)
        X_full = ntw.get_features().values
        y_full = ntw.df_features['class'].values
        expanded_features, expanded_labels, expanded_mask = smote_mask(
            train_mask_np, X_full, y_full, k_neighbors=k_neighbors, random_state=random_state, target_ratio=target_ratio
        )
        assert_ratio_achieved(expanded_labels, expanded_mask, target_ratio, tag=assert_tag)
        X_train = torch.tensor(expanded_features[expanded_mask], dtype=torch.float32).to(device_decoder)
        y_train = torch.tensor(expanded_labels[expanded_mask], dtype=torch.long).to(device_decoder)
    decoder = Decoder_deep_norm(X_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    
    # 類別重加權防退化
    num_pos = int((y_train == 1).sum().item())
    num_neg = int((y_train == 0).sum().item())
    pos_weight = float(num_neg) / max(num_pos, 1)
    weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device_decoder)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=clip_norm)
        optimizer.step()
        
    decoder.eval()
    with torch.no_grad():
        y_pred = decoder(X_test)
        y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().numpy(), y_pred.cpu().numpy()[:,1])
    cutoff = np.percentile(y_pred.cpu().numpy()[:,1], percentile_q)
    y_pred_hard = (y_pred.cpu().numpy()[:,1] >= cutoff).astype(int)
    f1 = f1_score(y_test.cpu().numpy(), y_pred_hard)
    return (ap_score, f1)

def intrinsic_features_with_predictions(
    ntw, train_mask, test_mask, n_layers_decoder, hidden_dim_decoder, lr, n_epochs_decoder, clip_norm=1.0,
    sampling="none", target_ratio=None, k_neighbors=5, random_state=None, assert_tag=""
):
    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    X_train, y_train, X_test, y_test = ntw.get_train_test_split_intrinsic(train_mask, test_mask, device=device_decoder)
    if sampling == "smote" and target_ratio is not None:
        train_mask_np = train_mask.cpu().numpy().astype(bool) if isinstance(train_mask, torch.Tensor) else np.array(train_mask).astype(bool)
        X_full = ntw.get_features().values
        y_full = ntw.df_features['class'].values
        expanded_features, expanded_labels, expanded_mask = smote_mask(
            train_mask_np, X_full, y_full, k_neighbors=k_neighbors, random_state=random_state, target_ratio=target_ratio
        )
        assert_ratio_achieved(expanded_labels, expanded_mask, target_ratio, tag=assert_tag)
        X_train = torch.tensor(expanded_features[expanded_mask], dtype=torch.float32).to(device_decoder)
        y_train = torch.tensor(expanded_labels[expanded_mask], dtype=torch.long).to(device_decoder)
    decoder = Decoder_deep_norm(X_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    
    num_pos = int((y_train == 1).sum().item())
    num_neg = int((y_train == 0).sum().item())
    pos_weight = float(num_neg) / max(num_pos, 1)
    weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device_decoder)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=clip_norm)
        optimizer.step()
        
    decoder.eval()
    with torch.no_grad():
        y_pred = decoder(X_test)
        y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().numpy(), y_pred.cpu().numpy()[:,1])
    return ap_score, y_pred.cpu().numpy()[:,1], y_test.cpu().numpy()

def positional_features(
    ntw, train_mask, test_mask, alpha_pr: float, alpha_ppr: float, n_epochs_decoder: int, lr: float,
    fraud_dict_train: dict = None, fraud_dict_test: dict = None, n_layers_decoder: int = 2,
    hidden_dim_decoder: int = 5, ntw_name: str = None, use_intrinsic: bool = False, percentile_q: int = 99, clip_norm=1.0,
    sampling: str = "none", target_ratio: float = None, k_neighbors: int = 5, random_state: int = None, assert_tag: str = ""
):
    print("intrinsic and summary: ")
    X = ntw.get_features(full=True)
    print("networkx: ")
    ntw_nx = ntw.get_network_nx()
    features_nx_df = local_features_nx(ntw_nx, alpha_pr, alpha_ppr, fraud_dict_train=fraud_dict_train, ntw_name=ntw_name)
    print("networkit: ")
    ntw_nk = ntw.get_network_nk()
    features_nk_df = features_nk(ntw_nk, ntw_name=ntw_name)
    if use_intrinsic:
        features_df = pd.concat([X, features_nx_df, features_nk_df], axis=1)
    else:
        features_df = pd.concat([features_nx_df, features_nk_df], axis=1)
    features_df["fraud"] = [fraud_dict_test[x] for x in features_df.index]
    
    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if sampling == "smote" and target_ratio is not None:
        train_mask_np = train_mask.numpy().astype(bool) if isinstance(train_mask, torch.Tensor) else np.array(train_mask).astype(bool)
        x_all_features = features_df.drop(["PSP", "fraud"], axis=1, errors='ignore').values
        y_all = features_df["fraud"].values
        expanded_features, expanded_labels, expanded_mask = smote_mask(
            train_mask_np, x_all_features, y_all, k_neighbors=k_neighbors, random_state=random_state, target_ratio=target_ratio
        )
        assert_ratio_achieved(expanded_labels, expanded_mask, target_ratio, tag=assert_tag)
        x_train = torch.tensor(expanded_features[expanded_mask], dtype=torch.float32).to(device_decoder)
        y_train = torch.tensor(expanded_labels[expanded_mask], dtype=torch.long).to(device_decoder)
    else:
        features_df_train = features_df[train_mask.numpy()]
        x_train_features = features_df_train.drop(["PSP", "fraud"], axis=1, errors='ignore').values
        x_train = torch.tensor(x_train_features, dtype=torch.float32).to(device_decoder)
        y_train = torch.tensor(features_df_train["fraud"].values, dtype=torch.long).to(device_decoder)

    features_df_test = features_df[test_mask.numpy()]
    x_test_features = features_df_test.drop(["PSP", "fraud"], axis=1, errors='ignore').values
    x_test = torch.tensor(x_test_features, dtype=torch.float32).to(device_decoder)
    y_test = torch.tensor(features_df_test["fraud"].values, dtype=torch.long).to(device_decoder)

    decoder = Decoder_deep_norm(x_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    num_pos = int((y_train == 1).sum().item())
    num_neg = int((y_train == 0).sum().item())
    pos_weight = float(num_neg) / max(num_pos, 1)
    weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device_decoder)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=clip_norm)
        optimizer.step()

    decoder.eval()
    with torch.no_grad():
        y_pred = decoder(x_test)
        y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().numpy(), y_pred.cpu().numpy()[:,1])
    cutoff = np.percentile(y_pred.cpu().numpy()[:,1], percentile_q)
    y_pred_hard = (y_pred.cpu().numpy()[:,1] >= cutoff).astype(int)
    f1 = f1_score(y_test.cpu().numpy(), y_pred_hard)
    return (ap_score, f1)

def positional_features_with_predictions(
    ntw, train_mask, test_mask, alpha_pr: float, alpha_ppr: float, n_epochs_decoder: int, lr: float,
    fraud_dict_train: dict = None, fraud_dict_test: dict = None, n_layers_decoder: int = 2,
    hidden_dim_decoder: int = 5, ntw_name: str = None, use_intrinsic: bool = False, clip_norm=1.0,
    sampling: str = "none", target_ratio: float = None, k_neighbors: int = 5, random_state: int = None, assert_tag: str = ""
):
    print("intrinsic and summary: ")
    X = ntw.get_features(full=True)
    print("networkx: ")
    ntw_nx = ntw.get_network_nx()
    features_nx_df = local_features_nx(ntw_nx, alpha_pr, alpha_ppr, fraud_dict_train=fraud_dict_train, ntw_name=ntw_name)
    print("networkit: ")
    ntw_nk = ntw.get_network_nk()
    features_nk_df = features_nk(ntw_nk, ntw_name=ntw_name)
    if use_intrinsic:
        features_df = pd.concat([X, features_nx_df, features_nk_df], axis=1)
    else:
        features_df = pd.concat([features_nx_df, features_nk_df], axis=1)
    features_df["fraud"] = [fraud_dict_test[x] for x in features_df.index]
    
    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if sampling == "smote" and target_ratio is not None:
        train_mask_np = train_mask.numpy().astype(bool) if isinstance(train_mask, torch.Tensor) else np.array(train_mask).astype(bool)
        x_all_features = features_df.drop(["PSP", "fraud"], axis=1, errors='ignore').values
        y_all = features_df["fraud"].values
        expanded_features, expanded_labels, expanded_mask = smote_mask(
            train_mask_np, x_all_features, y_all, k_neighbors=k_neighbors, random_state=random_state, target_ratio=target_ratio
        )
        assert_ratio_achieved(expanded_labels, expanded_mask, target_ratio, tag=assert_tag)
        x_train = torch.tensor(expanded_features[expanded_mask], dtype=torch.float32).to(device_decoder)
        y_train = torch.tensor(expanded_labels[expanded_mask], dtype=torch.long).to(device_decoder)
    else:
        features_df_train = features_df[train_mask.numpy()]
        x_train_features = features_df_train.drop(["PSP", "fraud"], axis=1, errors='ignore').values
        x_train = torch.tensor(x_train_features, dtype=torch.float32).to(device_decoder)
        y_train = torch.tensor(features_df_train["fraud"].values, dtype=torch.long).to(device_decoder)

    features_df_test = features_df[test_mask.numpy()]
    x_test_features = features_df_test.drop(["PSP", "fraud"], axis=1, errors='ignore').values
    x_test = torch.tensor(x_test_features, dtype=torch.float32).to(device_decoder)
    y_test = torch.tensor(features_df_test["fraud"].values, dtype=torch.long).to(device_decoder)

    decoder = Decoder_deep_norm(x_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    num_pos = int((y_train == 1).sum().item())
    num_neg = int((y_train == 0).sum().item())
    pos_weight = float(num_neg) / max(num_pos, 1)
    weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device_decoder)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=clip_norm)
        optimizer.step()

    decoder.eval()
    with torch.no_grad():
        y_pred = decoder(x_test)
        y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().numpy(), y_pred.cpu().numpy()[:,1])
    return ap_score, y_pred.cpu().numpy()[:,1], y_test.cpu().numpy()

def node2vec_features(
    ntw_torch, train_mask, test_mask, embedding_dim, walk_length, context_size, walks_per_node, num_negative_samples, 
    p, q, lr=0.01, n_epochs=1, n_epochs_decoder=1, ntw_nx = None, use_torch = False, use_intrinsic=True, percentile_q=99, clip_norm=1.0
):
    if use_torch:
        active_nodes = (train_mask.bool() | test_mask.bool())
        active_idx = None
        if active_nodes.any():
            active_idx = torch.where(active_nodes)[0]
            node_map = {int(old_idx): new_idx for new_idx, old_idx in enumerate(active_idx.tolist())}
            edge_pairs = []
            for src, dst in ntw_torch.edge_index.t().tolist():
                if src in node_map and dst in node_map:
                    edge_pairs.append((node_map[src], node_map[dst]))
            if edge_pairs:
                filtered_edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            else:
                filtered_edge_index = torch.empty((2, 0), dtype=torch.long)
            filtered_graph = ntw_torch.clone()
            filtered_graph.edge_index = filtered_edge_index
            filtered_graph.x = ntw_torch.x[active_idx]
            filtered_graph.num_nodes = int(active_idx.shape[0])
            graph_for_n2v = filtered_graph
        else:
            graph_for_n2v = ntw_torch
        model_n2v = node2vec_representation_torch(
            graph_for_n2v, train_mask = train_mask, test_mask = test_mask, embedding_dim=embedding_dim, 
            walk_length=walk_length, context_size=context_size, walks_per_node=walks_per_node, 
            num_negative_samples=num_negative_samples, p=p, q=q, lr=lr, n_epochs=n_epochs
        )
        model_n2v.eval()
        x = model_n2v()
        x = x.detach().to('cpu')
        x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
        if active_nodes.any() and active_idx is not None and x.shape[0] != ntw_torch.num_nodes:
            x_full = torch.zeros((ntw_torch.num_nodes, x.shape[1]), dtype=x.dtype)
            x_full[active_idx] = x
            x = x_full
    else:
        model_n2v = node2vec_representation_nx(
            ntw_nx, embedding_dim=embedding_dim, walk_length=walk_length, context_size=context_size, 
            walks_per_node=walks_per_node, num_negative_samples=num_negative_samples, p=p, q=q
        )
        print("Saving node2vec embedding...")
        x_df = pd.DataFrame([model_n2v(n) for n in ntw_nx.nodes()], index=ntw_nx.nodes())
        x = torch.tensor(x_df.values).to('cpu')
        x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
        
    x_intrinsic = ntw_torch.x.detach().to('cpu')
    x_intrinsic = torch.nan_to_num(x_intrinsic, nan=0.0, posinf=1e5, neginf=-1e5)
    if use_intrinsic:
        x = torch.cat((x, x_intrinsic), 1)
        
    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    x_train = x[train_mask].to(device_decoder).squeeze()
    x_test = x[test_mask].to(device_decoder).squeeze()
    y_train = ntw_torch.y[train_mask].to(device_decoder).squeeze()
    y_test = ntw_torch.y[test_mask].to(device_decoder).squeeze()
    
    decoder = Decoder_deep_norm(x_train.shape[1], 2, 10).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    
    num_pos = int((y_train == 1).sum().item())
    num_neg = int((y_train == 0).sum().item())
    pos_weight = float(num_neg) / max(num_pos, 1)
    weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device_decoder)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=clip_norm)
        optimizer.step()
        
    decoder.eval()
    with torch.no_grad():
        y_pred = decoder(x_test)
        y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().numpy(), y_pred.cpu().numpy()[:,1])
    cutoff = np.percentile(y_pred.cpu().numpy()[:,1], percentile_q)
    y_pred_hard = (y_pred.cpu().numpy()[:,1] >= cutoff).astype(int)
    f1 = f1_score(y_test.cpu().numpy(), y_pred_hard)
    return (ap_score, f1)

def node2vec_features_with_predictions(
    ntw_torch, train_mask, test_mask, embedding_dim, walk_length, context_size, walks_per_node, num_negative_samples, 
    p, q, lr=0.01, n_epochs=1, n_epochs_decoder=1, ntw_nx = None, use_torch = False, use_intrinsic=True, clip_norm=1.0
):
    if use_torch:
        active_nodes = (train_mask.bool() | test_mask.bool())
        active_idx = None
        if active_nodes.any():
            active_idx = torch.where(active_nodes)[0]
            node_map = {int(old_idx): new_idx for new_idx, old_idx in enumerate(active_idx.tolist())}
            edge_pairs = []
            for src, dst in ntw_torch.edge_index.t().tolist():
                if src in node_map and dst in node_map:
                    edge_pairs.append((node_map[src], node_map[dst]))
            if edge_pairs:
                filtered_edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            else:
                filtered_edge_index = torch.empty((2, 0), dtype=torch.long)
            filtered_graph = ntw_torch.clone()
            filtered_graph.edge_index = filtered_edge_index
            filtered_graph.x = ntw_torch.x[active_idx]
            filtered_graph.num_nodes = int(active_idx.shape[0])
            graph_for_n2v = filtered_graph
        else:
            graph_for_n2v = ntw_torch
        model_n2v = node2vec_representation_torch(
            graph_for_n2v, train_mask = train_mask, test_mask = test_mask, embedding_dim=embedding_dim, 
            walk_length=walk_length, context_size=context_size, walks_per_node=walks_per_node, 
            num_negative_samples=num_negative_samples, p=p, q=q, lr=lr, n_epochs=n_epochs
        )
        model_n2v.eval()
        x = model_n2v()
        x = x.detach().to('cpu')
        x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
        if active_nodes.any() and active_idx is not None and x.shape[0] != ntw_torch.num_nodes:
            x_full = torch.zeros((ntw_torch.num_nodes, x.shape[1]), dtype=x.dtype)
            x_full[active_idx] = x
            x = x_full
    else:
        model_n2v = node2vec_representation_nx(
            ntw_nx, embedding_dim=embedding_dim, walk_length=walk_length, context_size=context_size, 
            walks_per_node=walks_per_node, num_negative_samples=num_negative_samples, p=p, q=q
        )
        print("Saving node2vec embedding...")
        x_df = pd.DataFrame([model_n2v(n) for n in ntw_nx.nodes()], index=ntw_nx.nodes())
        x = torch.tensor(x_df.values).to('cpu')
        x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
        
    x_intrinsic = ntw_torch.x.detach().to('cpu')
    x_intrinsic = torch.nan_to_num(x_intrinsic, nan=0.0, posinf=1e5, neginf=-1e5)
    if use_intrinsic:
        x = torch.cat((x, x_intrinsic), 1)
        
    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    x_train = x[train_mask].to(device_decoder).squeeze()
    x_test = x[test_mask].to(device_decoder).squeeze()
    y_train = ntw_torch.y[train_mask].to(device_decoder).squeeze()
    y_test = ntw_torch.y[test_mask].to(device_decoder).squeeze()
    
    decoder = Decoder_deep_norm(x_train.shape[1], 2, 10).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    
    num_pos = int((y_train == 1).sum().item())
    num_neg = int((y_train == 0).sum().item())
    pos_weight = float(num_neg) / max(num_pos, 1)
    weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device_decoder)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=clip_norm)
        optimizer.step()
        
    decoder.eval()
    with torch.no_grad():
        y_pred = decoder(x_test)
        y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().numpy(), y_pred.cpu().numpy()[:,1])
    return ap_score, y_pred.cpu().numpy()[:,1], y_test.cpu().numpy()

def GNN_features(
    ntw_torch: Data, model: nn.Module, lr: float, n_epochs: int,
    train_loader: DataLoader = None, val_loader: DataLoader = None, test_loader: DataLoader = None,
    train_mask: torch.Tensor = None, val_mask: torch.Tensor = None, test_mask: torch.Tensor = None,
    use_intrinsic: bool = True, percentile_q: int = 99, patience: int = 10,
    checkpoint_path: str = "res/checkpoints/best_model.pt", monitor: str = 'val_ap',
    clip_norm: float = 1.0
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor=monitor
    )
    def _mask_to_device(mask):
        if mask is None: return None
        return mask.bool().to(device)
    def _forward(x, edge_index):
        if use_intrinsic: return model(x, edge_index)
        ones = torch.ones((x.shape[0], 1), dtype=torch.float32, device=device)
        return model(ones, edge_index)
    def _build_weighted_criterion(y_subset):
        num_pos = int((y_subset == 1).sum().item())
        num_neg = int((y_subset == 0).sum().item())
        pos_weight = float(num_neg) / max(num_pos, 1)
        weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weight_tensor)
        
    def train_epoch():
        model.train()
        optimizer.zero_grad()
        out, _ = _forward(ntw_torch.x.to(device), ntw_torch.edge_index.to(device))
        y = ntw_torch.y.long().to(device)
        if train_mask is not None:
            train_dev = _mask_to_device(train_mask)
            y_train = y[train_dev]
            criterion = _build_weighted_criterion(y_train)
            loss = criterion(out[train_dev], y_train)
        else:
            criterion = _build_weighted_criterion(y)
            loss = criterion(out, y)
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        return loss.item()
        
    def evaluate_split(mask):
        model.eval()
        with torch.no_grad():
            if use_intrinsic:
                out, _ = model(ntw_torch.x.to(device), ntw_torch.edge_index.to(device))
            else:
                ones = torch.ones((ntw_torch.x.shape[0], 1), dtype=torch.float32, device=device)
                out, _ = model(ones, ntw_torch.edge_index.to(device))
            y = ntw_torch.y.long().to(device)
            if mask is not None:
                mask_dev = _mask_to_device(mask)
                out = out[mask_dev]
                y = y[mask_dev]
            else:
                return None
            if out.shape[0] == 0: return None
            criterion = _build_weighted_criterion(y)
            loss = criterion(out, y).item()
            y_hat = out.softmax(dim=1)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
            try:
                ap_score = average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy()[:, 1])
                cutoff = np.percentile(y_hat.cpu().numpy()[:, 1], percentile_q)
                y_pred_hard = (y_hat.cpu().numpy()[:, 1] >= cutoff).astype(int)
                f1 = f1_score(y.cpu().numpy(), y_pred_hard)
            except Exception:
                pred = out.argmax(dim=1)
                correct = pred == y
                ap_score = int(correct.sum()) / max(int(correct.shape[0]), 1)
                f1 = ap_score
            return {'loss': loss, 'ap': ap_score, 'f1': f1}
            
    for epoch in range(n_epochs):
        train_loss = train_epoch()
        if val_mask is not None:
            val_result = evaluate_split(val_mask)
            if val_result is not None:
                print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | val_loss={val_result['loss']:.6f} | val_ap={val_result['ap']:.6f}")
                metric_to_monitor = val_result['ap'] if monitor == 'val_ap' else val_result['loss']
                early_stopping(metric_to_monitor, model)
                if early_stopping.early_stop:
                    print(f"[GNN_features] 連續 {patience} 個 Epoch 未改善，終止訓練！")
                    break
            else:
                print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | val_result=None")
        else:
            print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f}")
            
    if val_mask is not None and os.path.exists(checkpoint_path):
        print(f"[GNN_features] 載入驗證集最優表現模型權重: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_result = evaluate_split(test_mask)
    if test_result is None:
        raise RuntimeError('Test evaluation returned no data.')
    return test_result['ap'], test_result['f1']

def GNN_features_with_predictions(
    ntw_torch: Data, model: nn.Module, lr: float, n_epochs: int,
    train_loader: DataLoader = None, val_loader: DataLoader = None, test_loader: DataLoader = None,
    train_mask: torch.Tensor = None, val_mask: torch.Tensor = None, test_mask: torch.Tensor = None,
    use_intrinsic: bool = True, patience: int = 10,
    checkpoint_path: str = "res/checkpoints/best_model.pt", monitor: str = 'val_ap',
    clip_norm: float = 1.0
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor=monitor
    )
    def _mask_to_device(mask):
        if mask is None: return None
        return mask.bool().to(device)
    def _forward(x, edge_index):
        if use_intrinsic: return model(x, edge_index)
        ones = torch.ones((x.shape[0], 1), dtype=torch.float32, device=device)
        return model(ones, edge_index)
    def _build_weighted_criterion(y_subset):
        num_pos = int((y_subset == 1).sum().item())
        num_neg = int((y_subset == 0).sum().item())
        pos_weight = float(num_neg) / max(num_pos, 1)
        weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weight_tensor)
        
    def train_epoch():
        model.train()
        optimizer.zero_grad()
        out, _ = _forward(ntw_torch.x.to(device), ntw_torch.edge_index.to(device))
        y = ntw_torch.y.long().to(device)
        if train_mask is not None:
            train_dev = _mask_to_device(train_mask)
            y_train = y[train_dev]
            criterion = _build_weighted_criterion(y_train)
            loss = criterion(out[train_dev], y_train)
        else:
            criterion = _build_weighted_criterion(y)
            loss = criterion(out, y)
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        return loss.item()
        
    def evaluate_split(mask):
        model.eval()
        with torch.no_grad():
            if use_intrinsic:
                out, _ = model(ntw_torch.x.to(device), ntw_torch.edge_index.to(device))
            else:
                ones = torch.ones((ntw_torch.x.shape[0], 1), dtype=torch.float32, device=device)
                out, _ = model(ones, ntw_torch.edge_index.to(device))
            y = ntw_torch.y.long().to(device)
            if mask is not None:
                mask_dev = _mask_to_device(mask)
                out = out[mask_dev]
                y = y[mask_dev]
            else:
                return None
            if out.shape[0] == 0: return None
            criterion = _build_weighted_criterion(y)
            loss = criterion(out, y).item()
            y_hat = out.softmax(dim=1)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
            try:
                ap_score = average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy()[:, 1])
            except Exception:
                pred = out.argmax(dim=1)
                correct = pred == y
                ap_score = int(correct.sum()) / max(int(correct.shape[0]), 1)
            return {'loss': loss, 'ap': ap_score, 'output': y_hat, 'y': y}
            
    for epoch in range(n_epochs):
        train_loss = train_epoch()
        if val_mask is not None:
            val_result = evaluate_split(val_mask)
            if val_result is not None:
                print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | val_loss={val_result['loss']:.6f} | val_ap={val_result['ap']:.6f}")
                metric_to_monitor = val_result['ap'] if monitor == 'val_ap' else val_result['loss']
                early_stopping(metric_to_monitor, model)
                if early_stopping.early_stop:
                    print(f"[GNN_features] 連續 {patience} 個 Epoch 未改善，終止訓練！")
                    break
            else:
                print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | val_result=None")
        else:
            print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f}")
            
    if val_mask is not None and os.path.exists(checkpoint_path):
        print(f"[GNN_features] 載入驗證集最優表現模型權重: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_result = evaluate_split(test_mask)
    if test_result is None:
        raise RuntimeError('Test evaluation returned no data.')
    return test_result['ap'], test_result['output'].cpu().numpy()[:, 1], test_result['y'].cpu().numpy()

def GNN_features_graphsmote(
    ntw_torch, model: nn.Module, lr: float, n_epochs: int,
    train_loader: DataLoader = None, val_loader: DataLoader = None, test_loader: DataLoader = None,
    train_mask: torch.Tensor = None, val_mask: torch.Tensor = None, test_mask: torch.Tensor = None,
    use_intrinsic: bool = True, k_neighbors: int = 5, random_state: int = None, percentile_q: int = 99,
    sampling: str = "graph_smote", patience: int = 10,
    checkpoint_path: str = "res/checkpoints/best_model_graphsmote.pt", monitor: str = 'val_ap',
    clip_norm: float = 1.0, target_ratio: float = None, assert_tag: str = ""
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor=monitor
    )
    def _build_weighted_criterion(y_subset):
        num_pos = int((y_subset == 1).sum().item())
        num_neg = int((y_subset == 0).sum().item())
        pos_weight = float(num_neg) / max(num_pos, 1)
        weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weight_tensor)
    def _mask_to_device(mask):
        if mask is None: return None
        return mask.bool().to(device)
    def _forward(x, edge_index, edge_attr=None):
        if use_intrinsic: return model(x, edge_index, edge_attr=edge_attr)
        ones = torch.ones((x.shape[0], 1), dtype=torch.float32, device=device)
        return model(ones, edge_index, edge_attr=edge_attr)
        
    if sampling == "reweighted_graph_smote":
        x_smote, y_smote, train_mask_smote, edge_index_smote, edge_attr_smote = reweighted_graph_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, random_state=random_state,
            target_ratio=target_ratio
        )
    elif sampling == "graph_ensemble_smote":
        x_smote, y_smote, train_mask_smote, edge_index_smote = graph_ensemble_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, random_state=random_state,
            target_ratio=target_ratio
        )
        edge_attr_smote = None
    else:
        x_smote, y_smote, train_mask_smote, edge_index_smote = graph_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, random_state=random_state,
            target_ratio=target_ratio
        )
        edge_attr_smote = None

    if target_ratio is not None:
        assert_ratio_achieved(y_smote, train_mask_smote, target_ratio, tag=assert_tag)

    if isinstance(y_smote, torch.Tensor):
        valid_labels = torch.isin(y_smote, torch.tensor([0, 1], device=y_smote.device))
        valid_mask = train_mask_smote & valid_labels
    else:
        valid_labels = np.isin(y_smote, [0, 1])
        valid_mask = torch.from_numpy(valid_labels).bool() & train_mask_smote

    if not valid_mask.any():
        print("Warning: post-SMOTE filter left 0 samples. Fallback to normal GNN_features.")
        return GNN_features(
            ntw_torch, model, lr, n_epochs, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
            use_intrinsic=use_intrinsic, percentile_q=percentile_q, patience=patience, checkpoint_path=checkpoint_path,
            monitor=monitor, clip_norm=clip_norm
        )
        
    ntw_torch_smote = ntw_torch.clone()
    ntw_torch_smote.x = x_smote.to(device)
    ntw_torch_smote.y = y_smote.long().to(device)
    ntw_torch_smote.edge_index = edge_index_smote.long().to(device)
    if edge_attr_smote is not None:
        ntw_torch_smote.edge_attr = edge_attr_smote.to(device=device, dtype=torch.float32)
    train_mask_smote = valid_mask.bool().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    def train_epoch():
        model.train()
        optimizer.zero_grad()
        edge_attr_for_epoch = ntw_torch_smote.edge_attr if hasattr(ntw_torch_smote, 'edge_attr') and ntw_torch_smote.edge_attr is not None else None
        out, h = _forward(ntw_torch_smote.x, ntw_torch_smote.edge_index, edge_attr=edge_attr_for_epoch)
        y = ntw_torch_smote.y
        valid_mask_batch = torch.isin(y, torch.tensor([0, 1], device=y.device))
        active_mask = train_mask_smote & valid_mask_batch
        if active_mask.any():
            y_hat_filtered = out[active_mask]
            y_filtered = y[active_mask]
            criterion = _build_weighted_criterion(y_filtered)
            loss = criterion(y_hat_filtered, y_filtered)
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            return loss.item()
        return 0.0
        
    def evaluate_split(mask):
        model.eval()
        with torch.no_grad():
            if use_intrinsic:
                out, h = model(ntw_torch.x.to(device), ntw_torch.edge_index.to(device))
            else:
                ones = torch.ones((ntw_torch.x.shape[0], 1), dtype=torch.float32, device=device)
                out, h = model(ones, ntw_torch.edge_index.to(device))
            y = ntw_torch.y.long().to(device)
            if mask is not None:
                mask_dev = _mask_to_device(mask)
                out = out[mask_dev]
                y = y[mask_dev]
            if out.shape[0] == 0: return None
            criterion = _build_weighted_criterion(y)
            loss = criterion(out, y).item()
            y_hat = out.softmax(dim=1)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
            try:
                ap_score = average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy()[:, 1])
                cutoff = np.percentile(y_hat.cpu().numpy()[:, 1], percentile_q)
                y_pred_hard = (y_hat.cpu().numpy()[:, 1] >= cutoff).astype(int)
                f1 = f1_score(y.cpu().numpy(), y_pred_hard)
            except Exception:
                pred = out.argmax(dim=1)
                correct = pred == y
                ap_score = int(correct.sum()) / max(int(correct.shape[0]), 1)
                f1 = ap_score
            return {'loss': loss, 'ap': ap_score, 'f1': f1}
            
    for epoch in range(n_epochs):
        train_loss = train_epoch()
        if val_mask is not None:
            val_result = evaluate_split(val_mask)
            if val_result is not None:
                print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | val_loss={val_result['loss']:.6f} | val_ap={val_result['ap']:.6f}")
                metric_to_monitor = val_result['ap'] if monitor == 'val_ap' else val_result['loss']
                early_stopping(metric_to_monitor, model)
                if early_stopping.early_stop:
                    print(f"[{sampling}] 連續 {patience} 個 Epoch 未改善，終止訓練！")
                    break
            else:
                print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | val_result=None")
        else:
            print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f}")
            
    if val_mask is not None and os.path.exists(checkpoint_path):
        print(f"[{sampling}] 載入驗證集最優表現模型權重: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_result = evaluate_split(test_mask)
    if test_result is None:
        raise RuntimeError('Test evaluation returned no data.')
    return test_result['ap'], test_result['f1']

def GNN_features_graphsmote_with_predictions(
    ntw_torch, model: nn.Module, lr: float, n_epochs: int,
    train_loader: DataLoader = None, val_loader: DataLoader = None, test_loader: DataLoader = None,
    train_mask: torch.Tensor = None, val_mask: torch.Tensor = None, test_mask: torch.Tensor = None,
    use_intrinsic: bool = True, k_neighbors: int = 5, random_state: int = None,
    sampling: str = "graph_smote", patience: int = 10,
    checkpoint_path: str = "res/checkpoints/best_model_graphsmote.pt", monitor: str = 'val_ap',
    clip_norm: float = 1.0, target_ratio: float = None, assert_tag: str = ""
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor=monitor
    )
    def _build_weighted_criterion(y_subset):
        num_pos = int((y_subset == 1).sum().item())
        num_neg = int((y_subset == 0).sum().item())
        pos_weight = float(num_neg) / max(num_pos, 1)
        weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weight_tensor)
    def _mask_to_device(mask):
        if mask is None: return None
        return mask.bool().to(device)
    def _forward(x, edge_index, edge_attr=None):
        if use_intrinsic: return model(x, edge_index, edge_attr=edge_attr)
        ones = torch.ones((x.shape[0], 1), dtype=torch.float32, device=device)
        return model(ones, edge_index, edge_attr=edge_attr)
        
    if sampling == "reweighted_graph_smote":
        x_smote, y_smote, train_mask_smote, edge_index_smote, edge_attr_smote = reweighted_graph_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, random_state=random_state,
            target_ratio=target_ratio
        )
    elif sampling == "graph_ensemble_smote":
        x_smote, y_smote, train_mask_smote, edge_index_smote = graph_ensemble_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, random_state=random_state,
            target_ratio=target_ratio
        )
        edge_attr_smote = None
    else:
        x_smote, y_smote, train_mask_smote, edge_index_smote = graph_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, random_state=random_state,
            target_ratio=target_ratio
        )
        edge_attr_smote = None

    if target_ratio is not None:
        assert_ratio_achieved(y_smote, train_mask_smote, target_ratio, tag=assert_tag)

    if isinstance(y_smote, torch.Tensor):
        valid_labels = torch.isin(y_smote, torch.tensor([0, 1], device=y_smote.device))
        valid_mask = train_mask_smote & valid_labels
    else:
        valid_labels = np.isin(y_smote, [0, 1])
        valid_mask = torch.from_numpy(valid_labels).bool() & train_mask_smote
        
    if not valid_mask.any():
        print("Warning: post-SMOTE filter left 0 samples. Fallback to normal GNN_with_predictions.")
        return GNN_features_with_predictions(
            ntw_torch, model, lr, n_epochs, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
            use_intrinsic=use_intrinsic, patience=patience, checkpoint_path=checkpoint_path, monitor=monitor, clip_norm=clip_norm
        )
        
    ntw_torch_smote = ntw_torch.clone()
    ntw_torch_smote.x = x_smote.to(device)
    ntw_torch_smote.y = y_smote.long().to(device)
    ntw_torch_smote.edge_index = edge_index_smote.long().to(device)
    if edge_attr_smote is not None:
        ntw_torch_smote.edge_attr = edge_attr_smote.to(device=device, dtype=torch.float32)
    train_mask_smote = valid_mask.bool().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    def train_epoch():
        model.train()
        optimizer.zero_grad()
        edge_attr_for_epoch = ntw_torch_smote.edge_attr if hasattr(ntw_torch_smote, 'edge_attr') and ntw_torch_smote.edge_attr is not None else None
        out, h = _forward(ntw_torch_smote.x, ntw_torch_smote.edge_index, edge_attr=edge_attr_for_epoch)
        y = ntw_torch_smote.y
        valid_mask_batch = torch.isin(y, torch.tensor([0, 1], device=y.device))
        active_mask = train_mask_smote & valid_mask_batch
        if active_mask.any():
            y_hat_filtered = out[active_mask]
            y_filtered = y[active_mask]
            criterion = _build_weighted_criterion(y_filtered)
            loss = criterion(y_hat_filtered, y_filtered)
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            return loss.item()
        return 0.0
        
    def evaluate_split(mask):
        model.eval()
        with torch.no_grad():
            if use_intrinsic:
                out, h = model(ntw_torch.x.to(device), ntw_torch.edge_index.to(device))
            else:
                ones = torch.ones((ntw_torch.x.shape[0], 1), dtype=torch.float32, device=device)
                out, h = model(ones, ntw_torch.edge_index.to(device))
            y = ntw_torch.y.long().to(device)
            if mask is not None:
                mask_dev = _mask_to_device(mask)
                out = out[mask_dev]
                y = y[mask_dev]
            if out.shape[0] == 0: return None
            criterion = _build_weighted_criterion(y)
            loss = criterion(out, y).item()
            y_hat = out.softmax(dim=1)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
            try:
                ap_score = average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy()[:, 1])
            except Exception:
                pred = out.argmax(dim=1)
                correct = pred == y
                ap_score = int(correct.sum()) / max(int(correct.shape[0]), 1)
            return {'loss': loss, 'ap': ap_score, 'output': y_hat, 'y': y}
            
    for epoch in range(n_epochs):
        train_loss = train_epoch()
        if val_mask is not None:
            val_result = evaluate_split(val_mask)
            if val_result is not None:
                print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | val_loss={val_result['loss']:.6f} | val_ap={val_result['ap']:.6f}")
                metric_to_monitor = val_result['ap'] if monitor == 'val_ap' else val_result['loss']
                early_stopping(metric_to_monitor, model)
                if early_stopping.early_stop:
                    print(f"[{sampling}] 連續 {patience} 個 Epoch 未改善，終止訓練！")
                    break
            else:
                print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | val_result=None")
        else:
            print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f}")
            
    if val_mask is not None and os.path.exists(checkpoint_path):
        print(f"[{sampling}] 載入驗證集最優表現模型權重: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_result = evaluate_split(test_mask)
    if test_result is None:
        raise RuntimeError('Test evaluation returned no data.')
    return test_result['ap'], test_result['output'].cpu().numpy()[:, 1], test_result['y'].cpu().numpy()
