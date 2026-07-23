# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.methods.utils.functionsNetworkX import *
from src.methods.utils.functionsNetworKit import *
from src.methods.utils.functionsTorch import *
from src.methods.utils.GNN import *
from src.utils.Network import *
from src.methods.utils.decoder import *
from src.methods.evaluation import (
    smote_mask,
    graph_smote_mask,
    reweighted_graph_smote_mask,
    graph_ensemble_smote_mask,
    EarlyStopping,
    random_undersample_mask
)
from sklearn.metrics import average_precision_score, f1_score
import os
import numpy as np
import pandas as pd

# =====================================================================
# 1. Intrinsic Features (With and Without Predictions)
# =====================================================================

def intrinsic_features(
    ntw, train_mask, val_mask, test_mask, n_layers_decoder, hidden_dim_decoder, lr, n_epochs_decoder, ratio=None, sampling="none", percentile_q=99, patience=10, checkpoint_path="res/checkpoints/best_model_intrinsic_eval.pt"
):
    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    y_tensor = torch.tensor(ntw.df_features['class'].values, dtype=torch.long)
    features_tensor = ntw.get_features_torch()
    
    if sampling == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool().to(device_decoder), y_tensor.to(device_decoder), ratio=ratio)
    elif sampling == "smote" and ratio is not None:
        features_tensor, y_tensor, train_mask_sampled = smote_mask(train_mask.bool().to(device_decoder), features_tensor.to(device_decoder), y_tensor.to(device_decoder), ratio=ratio)
    else:
        train_mask_sampled = train_mask.bool().to(device_decoder)
        
    X_train = features_tensor[train_mask_sampled.cpu()].to(device_decoder)
    y_train = y_tensor[train_mask_sampled.cpu()].to(device_decoder)
    
    X_val = features_tensor[val_mask.bool().cpu()].to(device_decoder)
    y_val = y_tensor[val_mask.bool().cpu()].to(device_decoder)
    
    X_test = features_tensor[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)
    y_test = y_tensor[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)
    
    decoder = Decoder_deep_norm(X_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    
    num_pos = int((y_train == 1).sum().item())
    num_neg = int((y_train == 0).sum().item())
    pos_weight = float(num_neg) / max(num_pos, 1)
    weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device_decoder)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor='val_ap'
    )
    
    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        decoder.eval()
        with torch.no_grad():
            val_output = decoder(X_val)
            val_output_softmax = val_output.softmax(dim=1)
            val_ap = average_precision_score(y_val.cpu().numpy(), val_output_softmax.cpu().numpy()[:,1])
            
        early_stopping(val_ap, decoder)
        if early_stopping.early_stop:
            break
            
    if os.path.exists(checkpoint_path):
        decoder.load_state_dict(torch.load(checkpoint_path, map_location=device_decoder))
        
    decoder.eval()
    y_pred = decoder(X_test)
    y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    cutoff = np.percentile(y_pred.cpu().detach().numpy()[:,1], percentile_q)
    y_pred_hard = (y_pred.cpu().detach().numpy()[:,1] >= cutoff).astype(int)
    f1 = f1_score(y_test.cpu().detach().numpy(), y_pred_hard)
    return (ap_score, f1)


def intrinsic_features_with_predictions(
    ntw, train_mask, val_mask, test_mask, n_layers_decoder, hidden_dim_decoder, lr, n_epochs_decoder, ratio=None, sampling="none", patience=10, checkpoint_path="res/checkpoints/best_model_intrinsic_tuned.pt"
):
    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    y_tensor = torch.tensor(ntw.df_features['class'].values, dtype=torch.long)
    features_tensor = ntw.get_features_torch()
    
    if sampling == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool().to(device_decoder), y_tensor.to(device_decoder), ratio=ratio)
    elif sampling == "smote" and ratio is not None:
        features_tensor, y_tensor, train_mask_sampled = smote_mask(train_mask.bool().to(device_decoder), features_tensor.to(device_decoder), y_tensor.to(device_decoder), ratio=ratio)
    else:
        train_mask_sampled = train_mask.bool().to(device_decoder)
        
    X_train = features_tensor[train_mask_sampled.cpu()].to(device_decoder)
    y_train = y_tensor[train_mask_sampled.cpu()].to(device_decoder)
    
    X_val = features_tensor[val_mask.bool().cpu()].to(device_decoder)
    y_val = y_tensor[val_mask.bool().cpu()].to(device_decoder)
    
    X_test = features_tensor[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)
    y_test = y_tensor[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)
    
    decoder = Decoder_deep_norm(X_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    
    num_pos = int((y_train == 1).sum().item())
    num_neg = int((y_train == 0).sum().item())
    pos_weight = float(num_neg) / max(num_pos, 1)
    weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device_decoder)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor='val_ap'
    )
    
    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        decoder.eval()
        with torch.no_grad():
            val_output = decoder(X_val)
            val_output_softmax = val_output.softmax(dim=1)
            val_ap = average_precision_score(y_val.cpu().numpy(), val_output_softmax.cpu().numpy()[:,1])
            
        early_stopping(val_ap, decoder)
        if early_stopping.early_stop:
            break
            
    if os.path.exists(checkpoint_path):
        decoder.load_state_dict(torch.load(checkpoint_path, map_location=device_decoder))
        
    decoder.eval()
    y_pred = decoder(X_test)
    y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    return ap_score, y_pred.cpu().detach().numpy()[:,1], y_test.cpu().detach().numpy()


# =====================================================================
# 2. Positional Features (With and Without Predictions)
# =====================================================================

def positional_features(
    ntw, train_mask, val_mask, test_mask, alpha_pr: float, alpha_ppr: float, n_epochs_decoder: int, lr: float, fraud_dict_train: dict = None, fraud_dict_test: dict = None, n_layers_decoder: int = 2, hidden_dim_decoder: int = 5, ntw_name: str = None, use_intrinsic: bool = False, percentile_q: int = 99, ratio=None, sampling="none", patience=10, checkpoint_path="res/checkpoints/best_model_pos_eval.pt"
):
    print("intrinsic and summary: ")
    X_full_df = ntw.get_features(full=True)
    
    print("networkx (Full & Subgraph): ")
    ntw_nx_full = ntw.get_network_nx()
    
    # [Point 1: Subgraph isolation to completely block test set structure leakage]
    train_val_mask = train_mask.bool() | val_mask.bool()
    train_val_nodes = set(torch.where(train_val_mask)[0].tolist())
    ntw_nx_train_val = ntw_nx_full.subgraph(list(train_val_nodes))
    
    # Calculate features on train-val subgraph
    features_nx_df_train_val = local_features_nx(
        ntw_nx_train_val, alpha_pr, alpha_ppr, 
        fraud_dict_train=fraud_dict_train, 
        ntw_name=ntw_name + "_train_val"
    )
    features_nk_df_train_val = features_nk(
        ntw_nx_train_val, 
        ntw_name=ntw_name + "_train_val_nk"
    )
    
    if use_intrinsic:
        X_train_val_df = X_full_df.loc[X_full_df.index.isin(features_nx_df_train_val.index)]
        features_df_train_val = pd.concat([X_train_val_df, features_nx_df_train_val, features_nk_df_train_val], axis=1)
    else:
        features_df_train_val = pd.concat([features_nx_df_train_val, features_nk_df_train_val], axis=1)
        
    features_df_train_val["fraud"] = [fraud_dict_test[x] for x in features_df_train_val.index]
    
    # Map back to full-sized zeros to allow downstream masking code to work unmodified
    N_nodes = int(ntw_nx_full.number_of_nodes())
    x_features_train_val = features_df_train_val.drop(["PSP", "fraud"], axis=1, errors='ignore').values
    D_features = x_features_train_val.shape[1]
    
    full_size_x_train_val = np.zeros((N_nodes, D_features), dtype=np.float32)
    for idx, row in features_df_train_val.iterrows():
        full_size_x_train_val[int(idx)] = row.drop(["PSP", "fraud"], errors='ignore').values
        
    features_tensor_train_val = torch.tensor(full_size_x_train_val, dtype=torch.float32)
    y_tensor_train_val = torch.tensor([fraud_dict_test.get(i, 0) for i in range(N_nodes)], dtype=torch.long)
    
    # Calculate features on full graph for validation/test evaluation
    features_nx_df_full = local_features_nx(
        ntw_nx_full, alpha_pr, alpha_ppr, 
        fraud_dict_train=fraud_dict_train, 
        ntw_name=ntw_name + "_full"
    )
    ntw_nk_full = ntw.get_network_nk()
    features_nk_df_full = features_nk(
        ntw_nk_full, 
        ntw_name=ntw_name + "_full_nk"
    )
    
    if use_intrinsic:
        features_df_full = pd.concat([X_full_df, features_nx_df_full, features_nk_df_full], axis=1)
    else:
        features_df_full = pd.concat([features_nx_df_full, features_nk_df_full], axis=1)
        
    features_df_full["fraud"] = [fraud_dict_test[x] for x in features_df_full.index]
    
    x_features_full = features_df_full.drop(["PSP", "fraud"], axis=1, errors='ignore').values
    features_tensor_full = torch.tensor(x_features_full, dtype=torch.float32)
    y_tensor_full = torch.tensor(features_df_full["fraud"].values, dtype=torch.long)
    
    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    
    if sampling == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool(), y_tensor_train_val, ratio=ratio)
    elif sampling == "smote" and ratio is not None:
        features_tensor_train_val, y_tensor_train_val, train_mask_sampled = smote_mask(train_mask.bool(), features_tensor_train_val, y_tensor_train_val, ratio=ratio)
    else:
        train_mask_sampled = train_mask.bool()
        
    # Extracted splits
    X_train = features_tensor_train_val[train_mask_sampled.cpu()].to(device_decoder)
    y_train = y_tensor_train_val[train_mask_sampled.cpu()].to(device_decoder)
    
    # [Point 2: Add validation set to non-GNN methods]
    X_val = features_tensor_train_val[val_mask.bool().cpu()].to(device_decoder)
    y_val = y_tensor_train_val[val_mask.bool().cpu()].to(device_decoder)
    
    X_test = features_tensor_full[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)
    y_test = y_tensor_full[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)
    
    decoder = Decoder_deep_norm(X_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    
    num_pos = int((y_train == 1).sum().item())
    num_neg = int((y_train == 0).sum().item())
    pos_weight = float(num_neg) / max(num_pos, 1)
    weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device_decoder)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor='val_ap'
    )
    
    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation evaluation
        decoder.eval()
        with torch.no_grad():
            val_out = decoder(X_val)
            val_out_softmax = val_out.softmax(dim=1)
            val_ap = average_precision_score(y_val.cpu().numpy(), val_out_softmax.cpu().numpy()[:,1])
            
        early_stopping(val_ap, decoder)
        if early_stopping.early_stop:
            break
            
    if os.path.exists(checkpoint_path):
        decoder.load_state_dict(torch.load(checkpoint_path, map_location=device_decoder))
        
    decoder.eval()
    y_pred = decoder(X_test)
    y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    cutoff = np.percentile(y_pred.cpu().detach().numpy()[:,1], percentile_q)
    y_pred_hard = (y_pred.cpu().detach().numpy()[:,1] >= cutoff).astype(int)
    f1 = f1_score(y_test.cpu().detach().numpy(), y_pred_hard)
    return (ap_score, f1)


def positional_features_with_predictions(
    ntw, train_mask, val_mask, test_mask, alpha_pr: float, alpha_ppr: float, n_epochs_decoder: int, lr: float, fraud_dict_train: dict = None, fraud_dict_test: dict = None, n_layers_decoder: int = 2, hidden_dim_decoder: int = 5, ntw_name: str = None, use_intrinsic: bool = False, ratio=None, sampling="none", patience=10, checkpoint_path="res/checkpoints/best_model_pos_tuned.pt"
):
    print("intrinsic and summary: ")
    X_full_df = ntw.get_features(full=True)
    
    print("networkx (Full & Subgraph): ")
    ntw_nx_full = ntw.get_network_nx()
    
    # [Point 1: Subgraph isolation to completely block test set structure leakage]
    train_val_mask = train_mask.bool() | val_mask.bool()
    train_val_nodes = set(torch.where(train_val_mask)[0].tolist())
    ntw_nx_train_val = ntw_nx_full.subgraph(list(train_val_nodes))
    
    # Calculate features on train-val subgraph
    features_nx_df_train_val = local_features_nx(
        ntw_nx_train_val, alpha_pr, alpha_ppr, 
        fraud_dict_train=fraud_dict_train, 
        ntw_name=ntw_name + "_train_val"
    )
    features_nk_df_train_val = features_nk(
        ntw_nx_train_val, 
        ntw_name=ntw_name + "_train_val_nk"
    )
    
    if use_intrinsic:
        X_train_val_df = X_full_df.loc[X_full_df.index.isin(features_nx_df_train_val.index)]
        features_df_train_val = pd.concat([X_train_val_df, features_nx_df_train_val, features_nk_df_train_val], axis=1)
    else:
        features_df_train_val = pd.concat([features_nx_df_train_val, features_nk_df_train_val], axis=1)
        
    features_df_train_val["fraud"] = [fraud_dict_test[x] for x in features_df_train_val.index]
    
    # Map back to full-sized zeros to allow downstream masking code to work unmodified
    N_nodes = int(ntw_nx_full.number_of_nodes())
    x_features_train_val = features_df_train_val.drop(["PSP", "fraud"], axis=1, errors='ignore').values
    D_features = x_features_train_val.shape[1]
    
    full_size_x_train_val = np.zeros((N_nodes, D_features), dtype=np.float32)
    for idx, row in features_df_train_val.iterrows():
        full_size_x_train_val[int(idx)] = row.drop(["PSP", "fraud"], errors='ignore').values
        
    features_tensor_train_val = torch.tensor(full_size_x_train_val, dtype=torch.float32)
    y_tensor_train_val = torch.tensor([fraud_dict_test.get(i, 0) for i in range(N_nodes)], dtype=torch.long)
    
    # Calculate features on full graph for validation/test evaluation
    features_nx_df_full = local_features_nx(
        ntw_nx_full, alpha_pr, alpha_ppr, 
        fraud_dict_train=fraud_dict_train, 
        ntw_name=ntw_name + "_full"
    )
    ntw_nk_full = ntw.get_network_nk()
    features_nk_df_full = features_nk(
        ntw_nk_full, 
        ntw_name=ntw_name + "_full_nk"
    )
    
    if use_intrinsic:
        features_df_full = pd.concat([X_full_df, features_nx_df_full, features_nk_df_full], axis=1)
    else:
        features_df_full = pd.concat([features_nx_df_full, features_nk_df_full], axis=1)
        
    features_df_full["fraud"] = [fraud_dict_test[x] for x in features_df_full.index]
    
    x_features_full = features_df_full.drop(["PSP", "fraud"], axis=1, errors='ignore').values
    features_tensor_full = torch.tensor(x_features_full, dtype=torch.float32)
    y_tensor_full = torch.tensor(features_df_full["fraud"].values, dtype=torch.long)
    
    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    
    if sampling == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool(), y_tensor_train_val, ratio=ratio)
    elif sampling == "smote" and ratio is not None:
        features_tensor_train_val, y_tensor_train_val, train_mask_sampled = smote_mask(train_mask.bool(), features_tensor_train_val, y_tensor_train_val, ratio=ratio)
    else:
        train_mask_sampled = train_mask.bool()
        
    # Extracted splits
    X_train = features_tensor_train_val[train_mask_sampled.cpu()].to(device_decoder)
    y_train = y_tensor_train_val[train_mask_sampled.cpu()].to(device_decoder)
    
    # [Point 2: Add validation set to non-GNN methods]
    X_val = features_tensor_train_val[val_mask.bool().cpu()].to(device_decoder)
    y_val = y_tensor_train_val[val_mask.bool().cpu()].to(device_decoder)
    
    X_test = features_tensor_full[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)
    y_test = y_tensor_full[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)
    
    decoder = Decoder_deep_norm(X_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    
    num_pos = int((y_train == 1).sum().item())
    num_neg = int((y_train == 0).sum().item())
    pos_weight = float(num_neg) / max(num_pos, 1)
    weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device_decoder)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor='val_ap'
    )
    
    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation evaluation
        decoder.eval()
        with torch.no_grad():
            val_out = decoder(X_val)
            val_out_softmax = val_out.softmax(dim=1)
            val_ap = average_precision_score(y_val.cpu().numpy(), val_out_softmax.cpu().numpy()[:,1])
            
        early_stopping(val_ap, decoder)
        if early_stopping.early_stop:
            break
            
    if os.path.exists(checkpoint_path):
        decoder.load_state_dict(torch.load(checkpoint_path, map_location=device_decoder))
        
    decoder.eval()
    y_pred = decoder(X_test)
    y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    return ap_score, y_pred.cpu().detach().numpy()[:,1], y_test.cpu().detach().numpy()


# =====================================================================
# 3. Node2Vec (No Graph Modifications Needed - Symmetrically Padded and Corrected)
# =====================================================================

def node2vec_features(
    ntw_torch, train_mask, val_mask, test_mask, embedding_dim, walk_length, context_size, walks_per_node, num_negative_samples, p, q, lr=0.01, n_epochs=1, n_epochs_decoder=1, ntw_nx=None, use_torch=False, use_intrinsic=True, percentile_q=99, ratio=None, sampling="none"
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
    else:
        graph_for_n2v = ntw_torch
        
    model_n2v = node2vec_representation_torch(
        graph_for_n2v, train_mask=train_mask, test_mask=test_mask, embedding_dim=embedding_dim, walk_length=walk_length, context_size=context_size, walks_per_node=walks_per_node, num_negative_samples=num_negative_samples, p=p, q=q, lr=lr, n_epochs=n_epochs
    )
    model_n2v.eval()
    x = model_n2v()
    x = x.detach().to('cpu')
    x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
    if active_nodes.any() and active_idx is not None and x.shape[0] != ntw_torch.num_nodes:
        x_full = torch.zeros((ntw_torch.num_nodes, x.shape[1]), dtype=x.dtype)
        x_full[active_idx] = x
        x = x_full
    x_intrinsic = ntw_torch.x.detach().to('cpu')
    x_intrinsic = torch.nan_to_num(x_intrinsic, nan=0.0, posinf=1e5, neginf=-1e5)
    if use_intrinsic:
        x = torch.cat((x, x_intrinsic), 1)
    y_tensor = ntw_torch.y.cpu()
    if sampling == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool(), y_tensor, ratio=ratio)
    elif sampling == "smote" and ratio is not None:
        x, y_tensor, train_mask_sampled = smote_mask(train_mask.bool(), x, y_tensor, ratio=ratio)
    elif sampling in ["graph_smote", "graph_ensemble_smote", "reweighted_graph_smote"] and ratio is not None:
        x, y_tensor, train_mask_sampled = smote_mask(train_mask.bool(), x, y_tensor, ratio=ratio)
    else:
        train_mask_sampled = train_mask.bool()
        
    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    x_train = x[train_mask_sampled.cpu()].to(device_decoder).squeeze()
    x_test = x[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder).squeeze()
    y_train = y_tensor[train_mask_sampled.cpu()].to(device_decoder).squeeze()
    y_test = ntw_torch.y[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder).squeeze()
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
        optimizer.step()
    decoder.eval()
    y_pred = decoder(x_test)
    y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    cutoff = np.percentile(y_pred.cpu().detach().numpy()[:,1], percentile_q)
    y_pred_hard = (y_pred.cpu().detach().numpy()[:,1] >= cutoff).astype(int)
    f1 = f1_score(y_test.cpu().detach().numpy(), y_pred_hard)
    return (ap_score, f1)


def node2vec_features_with_predictions(
    ntw_torch, train_mask, val_mask, test_mask, embedding_dim, walk_length, context_size, walks_per_node, num_negative_samples, p, q, lr=0.01, n_epochs=1, n_epochs_decoder=1, ntw_nx=None, use_torch=False, use_intrinsic=True, ratio=None, sampling="none"
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
    else:
        graph_for_n2v = ntw_torch
        
    model_n2v = node2vec_representation_torch(
        graph_for_n2v, train_mask=train_mask, test_mask=test_mask, embedding_dim=embedding_dim, walk_length=walk_length, context_size=context_size, walks_per_node=walks_per_node, num_negative_samples=num_negative_samples, p=p, q=q, lr=lr, n_epochs=n_epochs
    )
    model_n2v.eval()
    x = model_n2v()
    x = x.detach().to('cpu')
    x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
    if active_nodes.any() and active_idx is not None and x.shape[0] != ntw_torch.num_nodes:
        x_full = torch.zeros((ntw_torch.num_nodes, x.shape[1]), dtype=x.dtype)
        x_full[active_idx] = x
        x = x_full
    x_intrinsic = ntw_torch.x.detach().to('cpu')
    x_intrinsic = torch.nan_to_num(x_intrinsic, nan=0.0, posinf=1e5, neginf=-1e5)
    if use_intrinsic:
        x = torch.cat((x, x_intrinsic), 1)
    y_tensor = ntw_torch.y.cpu()
    if sampling == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool(), y_tensor, ratio=ratio)
    elif sampling == "smote" and ratio is not None:
        x, y_tensor, train_mask_sampled = smote_mask(train_mask.bool(), x, y_tensor, ratio=ratio)
    elif sampling in ["graph_smote", "graph_ensemble_smote", "reweighted_graph_smote"] and ratio is not None:
        x, y_tensor, train_mask_sampled = smote_mask(train_mask.bool(), x, y_tensor, ratio=ratio)
    else:
        train_mask_sampled = train_mask.bool()
        
    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    x_train = x[train_mask_sampled.cpu()].to(device_decoder).squeeze()
    x_test = x[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder).squeeze()
    y_train = y_tensor[train_mask_sampled.cpu()].to(device_decoder).squeeze()
    y_test = ntw_torch.y[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder).squeeze()
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
        optimizer.step()
    decoder.eval()
    y_pred = decoder(x_test)
    y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    return ap_score, y_pred.cpu().detach().numpy()[:,1], y_test.cpu().detach().numpy()


# =====================================================================
# 4. Standard GNN Methods (Validation, Slicing Protected)
# =====================================================================

def GNN_features(
    ntw_torch, model: nn.Module, lr: float, n_epochs: int, train_loader: DataLoader = None, val_loader: DataLoader = None, test_loader: DataLoader = None, train_mask: torch.Tensor = None, val_mask: torch.Tensor = None, test_mask: torch.Tensor = None, use_intrinsic: bool = True, percentile_q: int = 99, patience: int = 10, checkpoint_path: str = "res/checkpoints/best_model.pt", monitor: str = 'val_ap', ratio=None, sampling="none"
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor=monitor
    )
    y_tensor = ntw_torch.y.cpu()
    if sampling == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool(), y_tensor, ratio=ratio).to(device)
    elif sampling == "smote" and ratio is not None:
        x_np, y_np, train_mask_np = smote_mask(train_mask.bool(), ntw_torch.x, ntw_torch.y, ratio=ratio)
        ntw_torch = ntw_torch.clone()
        ntw_torch.x = x_np.to(device)
        ntw_torch.y = y_np.to(device)
        train_mask_sampled = train_mask_np.to(device)
    else:
        train_mask_sampled = train_mask.bool().to(device)
        
    def _mask_to_device(mask):
        if mask is None:
            return None
        return mask.bool().to(device)
        
    def _forward(x, edge_index):
        if use_intrinsic:
            return model(x, edge_index)
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
        train_dev = train_mask_sampled
        y_train = y[train_dev]
        criterion = _build_weighted_criterion(y_train)
        loss = criterion(out[train_dev], y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()
        
    def evaluate_split(mask):
        model.eval()
        with torch.no_grad():
            out, _ = _forward(ntw_torch.x.to(device), ntw_torch.edge_index.to(device))
            y = ntw_torch.y.long().to(device)
            mask_dev = _mask_to_device(mask)
            out_filtered = out[:mask_dev.shape[0]][mask_dev]
            y_filtered = y[:mask_dev.shape[0]][mask_dev]
            if out_filtered.shape[0] == 0:
                return None
            criterion = _build_weighted_criterion(y_filtered)
            loss = criterion(out_filtered, y_filtered).item()
            y_hat = out_filtered.softmax(dim=1)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
            ap_score = average_precision_score(y_filtered.cpu().numpy(), y_hat.cpu().numpy()[:, 1])
            cutoff = np.percentile(y_hat.cpu().numpy()[:, 1], percentile_q)
            y_pred_hard = (y_hat.cpu().numpy()[:, 1] >= cutoff).astype(int)
            f1 = f1_score(y_filtered.cpu().numpy(), y_pred_hard)
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
                    print(f"[GNN_features] Early Stop triggered!")
                    break
                    
    if val_mask is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_result = evaluate_split(test_mask)
    return test_result['ap'], test_result['f1']


def GNN_features_with_predictions(
    ntw_torch, model: nn.Module, lr: float, n_epochs: int, train_loader: DataLoader = None, val_loader: DataLoader = None, test_loader: DataLoader = None, train_mask: torch.Tensor = None, val_mask: torch.Tensor = None, test_mask: torch.Tensor = None, use_intrinsic: bool = True, patience: int = 10, checkpoint_path: str = "res/checkpoints/best_model.pt", monitor: str = 'val_ap', ratio=None, sampling="none"
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor=monitor
    )
    y_tensor = ntw_torch.y.cpu()
    if sampling == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool(), y_tensor, ratio=ratio).to(device)
    elif sampling == "smote" and ratio is not None:
        x_np, y_np, train_mask_np = smote_mask(train_mask.bool(), ntw_torch.x, ntw_torch.y, ratio=ratio)
        ntw_torch = ntw_torch.clone()
        ntw_torch.x = x_np.to(device)
        ntw_torch.y = y_np.to(device)
        train_mask_sampled = train_mask_np.to(device)
    else:
        train_mask_sampled = train_mask.bool().to(device)
        
    def _mask_to_device(mask):
        if mask is None:
            return None
        return mask.bool().to(device)
        
    def _forward(x, edge_index):
        if use_intrinsic:
            return model(x, edge_index)
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
        train_dev = train_mask_sampled
        y_train = y[train_dev]
        criterion = _build_weighted_criterion(y_train)
        loss = criterion(out[train_dev], y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()
        
    def evaluate_split(mask):
        model.eval()
        with torch.no_grad():
            out, _ = _forward(ntw_torch.x.to(device), ntw_torch.edge_index.to(device))
            y = ntw_torch.y.long().to(device)
            mask_dev = _mask_to_device(mask)
            out_filtered = out[:mask_dev.shape[0]][mask_dev]
            y_filtered = y[:mask_dev.shape[0]][mask_dev]
            if out_filtered.shape[0] == 0:
                return None
            criterion = _build_weighted_criterion(y_filtered)
            loss = criterion(out_filtered, y_filtered).item()
            y_hat = out_filtered.softmax(dim=1)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
            ap_score = average_precision_score(y_filtered.cpu().numpy(), y_hat.cpu().numpy()[:, 1])
            return {'loss': loss, 'ap': ap_score, 'output': y_hat, 'y': y_filtered}
            
    for epoch in range(n_epochs):
        train_loss = train_epoch()
        if val_mask is not None:
            val_result = evaluate_split(val_mask)
            if val_result is not None:
                print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | val_loss={val_result['loss']:.6f} | val_ap={val_result['ap']:.6f}")
                metric_to_monitor = val_result['ap'] if monitor == 'val_ap' else val_result['loss']
                early_stopping(metric_to_monitor, model)
                if early_stopping.early_stop:
                    print(f"[GNN_features] Early Stop triggered!")
                    break
                    
    if val_mask is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_result = evaluate_split(test_mask)
    return test_result['ap'], test_result['output'].cpu().numpy()[:, 1], test_result['y'].cpu().numpy()


# =====================================================================
# 5. GNN GraphSMOTE (Point 3 - Train-Evaluation Graph Match)
# =====================================================================

def GNN_features_graphsmote(
    ntw_torch, model: nn.Module, lr: float, n_epochs: int, train_loader: DataLoader = None, val_loader: DataLoader = None, test_loader: DataLoader = None, train_mask: torch.Tensor = None, val_mask: torch.Tensor = None, test_mask: torch.Tensor = None, use_intrinsic: bool = True, k_neighbors: int = 5, random_state: int = None, percentile_q: int = 99, sampling: str = "graph_smote", patience: int = 10, checkpoint_path: str = "res/checkpoints/best_model_graphsmote.pt", monitor: str = 'val_ap', ratio=None
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
        if mask is None:
            return None
        return mask.bool().to(device)
        
    def _forward(x, edge_index, edge_attr=None):
        if use_intrinsic:
            return model(x, edge_index, edge_attr=edge_attr)
        ones = torch.ones((x.shape[0], 1), dtype=torch.float32, device=device)
        return model(ones, edge_index, edge_attr=edge_attr)
        
    if sampling == "reweighted_graph_smote":
        x_smote, y_smote, train_mask_smote, edge_index_smote, edge_attr_smote = reweighted_graph_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, ratio=ratio, random_state=random_state
        )
    elif sampling == "graph_ensemble_smote":
        x_smote, y_smote, train_mask_smote, edge_index_smote = graph_ensemble_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, ratio=ratio, random_state=random_state
        )
        edge_attr_smote = None
    else: # "graph_smote"
        x_smote, y_smote, train_mask_smote, edge_index_smote = graph_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, ratio=ratio, random_state=random_state
        )
        edge_attr_smote = None
        
    ntw_torch_smote = ntw_torch.clone()
    ntw_torch_smote.x = x_smote.to(device)
    ntw_torch_smote.y = y_smote.long().to(device)
    ntw_torch_smote.edge_index = edge_index_smote.long().to(device)
    if edge_attr_smote is not None:
        ntw_torch_smote.edge_attr = edge_attr_smote.to(device=device, dtype=torch.float32)
        
    train_mask_smote = train_mask_smote.bool().to(device)
    
    # [Point 3: Create padded masks for expanded graph size validation/test symmetry]
    n_synthetic = int(x_smote.shape[0] - ntw_torch.x.shape[0])
    if val_mask is not None:
        val_mask_smote = torch.cat([val_mask.bool().cpu(), torch.zeros(n_synthetic, dtype=torch.bool)]).to(device)
    else:
        val_mask_smote = None
    test_mask_smote = torch.cat([test_mask.bool().cpu(), torch.zeros(n_synthetic, dtype=torch.bool)]).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    def train_epoch():
        model.train()
        optimizer.zero_grad()
        edge_attr_for_epoch = ntw_torch_smote.edge_attr if hasattr(ntw_torch_smote, 'edge_attr') and ntw_torch_smote.edge_attr is not None else None
        out, _ = _forward(ntw_torch_smote.x, ntw_torch_smote.edge_index, edge_attr=edge_attr_for_epoch)
        y = ntw_torch_smote.y
        active_mask = train_mask_smote
        if active_mask.any():
            y_hat_filtered = out[active_mask]
            y_filtered = y[active_mask]
            criterion = _build_weighted_criterion(y_filtered)
            loss = criterion(y_hat_filtered, y_filtered)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            return loss.item()
        return 0.0
        
    def evaluate_split(mask_smote):
        model.eval()
        with torch.no_grad():
            edge_attr_for_eval = ntw_torch_smote.edge_attr if hasattr(ntw_torch_smote, 'edge_attr') and ntw_torch_smote.edge_attr is not None else None
            out, _ = _forward(ntw_torch_smote.x, ntw_torch_smote.edge_index, edge_attr=edge_attr_for_eval)
            y = ntw_torch_smote.y
            mask_dev = mask_smote.bool().to(device)
            out_filtered = out[mask_dev]
            y_filtered = y[mask_dev]
            if out_filtered.shape[0] == 0:
                return None
            criterion = _build_weighted_criterion(y_filtered)
            loss = criterion(out_filtered, y_filtered).item()
            y_hat = out_filtered.softmax(dim=1)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
            ap_score = average_precision_score(y_filtered.cpu().numpy(), y_hat.cpu().numpy()[:, 1])
            cutoff = np.percentile(y_hat.cpu().numpy()[:, 1], percentile_q)
            y_pred_hard = (y_hat.cpu().numpy()[:, 1] >= cutoff).astype(int)
            f1 = f1_score(y_filtered.cpu().numpy(), y_pred_hard)
            return {'loss': loss, 'ap': ap_score, 'f1': f1}
            
    for epoch in range(n_epochs):
        train_loss = train_epoch()
        if val_mask_smote is not None:
            val_result = evaluate_split(val_mask_smote)
            if val_result is not None:
                print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | val_loss={val_result['loss']:.6f} | val_ap={val_result['ap']:.6f}")
                metric_to_monitor = val_result['ap'] if monitor == 'val_ap' else val_result['loss']
                early_stopping(metric_to_monitor, model)
                if early_stopping.early_stop:
                    print(f"[{sampling}] Early Stop triggered!")
                    break
                    
    if val_mask_smote is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_result = evaluate_split(test_mask_smote)
    return test_result['ap'], test_result['f1']


def GNN_features_graphsmote_with_predictions(
    ntw_torch, model: nn.Module, lr: float, n_epochs: int, train_loader: DataLoader = None, val_loader: DataLoader = None, test_loader: DataLoader = None, train_mask: torch.Tensor = None, val_mask: torch.Tensor = None, test_mask: torch.Tensor = None, use_intrinsic: bool = True, k_neighbors: int = 5, random_state: int = None, sampling: str = "graph_smote", patience: int = 10, checkpoint_path: str = "res/checkpoints/best_model_graphsmote.pt", monitor: str = 'val_ap', ratio=None
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
        if mask is None:
            return None
        return mask.bool().to(device)
        
    def _forward(x, edge_index, edge_attr=None):
        if use_intrinsic:
            return model(x, edge_index, edge_attr=edge_attr)
        ones = torch.ones((x.shape[0], 1), dtype=torch.float32, device=device)
        return model(ones, edge_index, edge_attr=edge_attr)
        
    if sampling == "reweighted_graph_smote":
        x_smote, y_smote, train_mask_smote, edge_index_smote, edge_attr_smote = reweighted_graph_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, ratio=ratio, random_state=random_state
        )
    elif sampling == "graph_ensemble_smote":
        x_smote, y_smote, train_mask_smote, edge_index_smote = graph_ensemble_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, ratio=ratio, random_state=random_state
        )
        edge_attr_smote = None
    else: # "graph_smote"
        x_smote, y_smote, train_mask_smote, edge_index_smote = graph_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, ratio=ratio, random_state=random_state
        )
        edge_attr_smote = None
        
    ntw_torch_smote = ntw_torch.clone()
    ntw_torch_smote.x = x_smote.to(device)
    ntw_torch_smote.y = y_smote.long().to(device)
    ntw_torch_smote.edge_index = edge_index_smote.long().to(device)
    if edge_attr_smote is not None:
        ntw_torch_smote.edge_attr = edge_attr_smote.to(device=device, dtype=torch.float32)
        
    train_mask_smote = train_mask_smote.bool().to(device)
    
    # [Point 3: Create padded masks for expanded graph size validation/test symmetry]
    n_synthetic = int(x_smote.shape[0] - ntw_torch.x.shape[0])
    if val_mask is not None:
        val_mask_smote = torch.cat([val_mask.bool().cpu(), torch.zeros(n_synthetic, dtype=torch.bool)]).to(device)
    else:
        val_mask_smote = None
    test_mask_smote = torch.cat([test_mask.bool().cpu(), torch.zeros(n_synthetic, dtype=torch.bool)]).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    def train_epoch():
        model.train()
        optimizer.zero_grad()
        edge_attr_for_epoch = ntw_torch_smote.edge_attr if hasattr(ntw_torch_smote, 'edge_attr') and ntw_torch_smote.edge_attr is not None else None
        out, _ = _forward(ntw_torch_smote.x, ntw_torch_smote.edge_index, edge_attr=edge_attr_for_epoch)
        y = ntw_torch_smote.y
        active_mask = train_mask_smote
        if active_mask.any():
            y_hat_filtered = out[active_mask]
            y_filtered = y[active_mask]
            criterion = _build_weighted_criterion(y_filtered)
            loss = criterion(y_hat_filtered, y_filtered)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            return loss.item()
        return 0.0
        
    def evaluate_split(mask_smote):
        model.eval()
        with torch.no_grad():
            edge_attr_for_eval = ntw_torch_smote.edge_attr if hasattr(ntw_torch_smote, 'edge_attr') and ntw_torch_smote.edge_attr is not None else None
            out, _ = _forward(ntw_torch_smote.x, ntw_torch_smote.edge_index, edge_attr=edge_attr_for_eval)
            y = ntw_torch_smote.y
            mask_dev = mask_smote.bool().to(device)
            out_filtered = out[mask_dev]
            y_filtered = y[mask_dev]
            if out_filtered.shape[0] == 0:
                return None
            criterion = _build_weighted_criterion(y_filtered)
            loss = criterion(out_filtered, y_filtered).item()
            y_hat = out_filtered.softmax(dim=1)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
            ap_score = average_precision_score(y_filtered.cpu().numpy(), y_hat.cpu().numpy()[:, 1])
            return {'loss': loss, 'ap': ap_score, 'output': y_hat, 'y': y_filtered}
            
    for epoch in range(n_epochs):
        train_loss = train_epoch()
        if val_mask_smote is not None:
            val_result = evaluate_split(val_mask_smote)
            if val_result is not None:
                print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | val_loss={val_result['loss']:.6f} | val_ap={val_result['ap']:.6f}")
                metric_to_monitor = val_result['ap'] if monitor == 'val_ap' else val_result['loss']
                early_stopping(metric_to_monitor, model)
                if early_stopping.early_stop:
                    print(f"[{sampling}] Early Stop triggered!")
                    break
                    
    if val_mask_smote is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_result = evaluate_split(test_mask_smote)
    return test_result['ap'], test_result['output'].cpu().numpy()[:, 1], test_result['y'].cpu().numpy()
