# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from src.methods.utils.functionsNetworkX import *
from src.methods.utils.functionsNetworKit import *
from src.methods.utils.functionsTorch import *
from src.methods.utils.GNN import *
from src.utils.Network import *
from src.methods.utils.decoder import *
from src.methods.evaluation import (
    smote_mask,
    graph_smote_mask,
    EarlyStopping,
    random_undersample_mask
)
from src.methods.losses import FocalLoss
from src.methods.samplers import GATEdgeGenerator, TargetedNeighbourhoodUndersampling
from src.methods import graphens
from sklearn.metrics import average_precision_score, f1_score
import os
import numpy as np
import pandas as pd

def _normalize_sampling_name(sampling):
    mapping = {
        "none": "none",
        "rus": "random_undersample",
        "random_undersample": "random_undersample",
        "smote": "smote",
        "graphsmote": "graph_smote",
        "graph_smote": "graph_smote",
        "gatsmote": "gatsmote",
        "tnu": "targeted_neighbourhood_undersampling",
        "targeted_neighbourhood_undersampling": "targeted_neighbourhood_undersampling",
        "graphens": "graphens",
        "graph_ensemble_smote": "graphens",
    }
    return mapping.get(sampling, sampling)

def _build_loss_criterion(y_subset, loss_name="ce", loss_kwargs=None, device=None):
    if loss_kwargs is None:
        loss_kwargs = {}
    if loss_name in {None, "ce", "cross_entropy"}:
        num_pos = int((y_subset == 1).sum().item())
        num_neg = int((y_subset == 0).sum().item())
        pos_weight = float(num_neg) / max(num_pos, 1)
        weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weight_tensor)
    if loss_name in {"weighted_ce"}:
        num_pos = int((y_subset == 1).sum().item())
        num_neg = int((y_subset == 0).sum().item())
        pos_weight = float(num_neg) / max(num_pos, 1)
        weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weight_tensor)
    if loss_name == "focal":
        alpha = loss_kwargs.get("alpha", None)
        gamma = loss_kwargs.get("gamma", 2.0)
        reduction = loss_kwargs.get("reduction", "mean")
        criterion = FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
        if device is not None:
            criterion = criterion.to(device=device)
        return criterion
    raise ValueError(f"Unsupported loss: {loss_name}")

def _compute_loss(criterion, out, y, mask=None):
    if isinstance(criterion, FocalLoss):
        return criterion(out, y, mask=mask)
    return criterion(out, y)

def _build_neighbor_loader(ntw_torch, input_mask, batch_size, num_neighbors, shuffle):
    """GraphSAGE's mean/pool aggregator is defined for a sample-and-aggregate neighborhood
    (Hamilton et al. 2017), but every GNN_features* pass here otherwise forwards the whole
    graph in one shot -- SAGE was getting no actual neighbor sampling. NeighborLoader is
    built on a CPU copy (its C++ sampler expects CPU tensors) with `input_mask` as the seed
    nodes; each yielded mini-batch places the seed nodes first (`batch.batch_size` of them),
    followed by their sampled multi-hop neighbors used only for message passing.
    """
    from torch_geometric.loader import NeighborLoader
    graph_cpu = ntw_torch.clone().cpu()
    return NeighborLoader(
        graph_cpu, num_neighbors=num_neighbors, batch_size=batch_size,
        input_nodes=input_mask.cpu(), shuffle=shuffle,
    )

def _try_build_neighbor_loader(ntw_torch, input_mask, batch_size, num_neighbors, shuffle):
    """NeighborLoader's sampler needs pyg-lib or torch-sparse compiled extensions; on an
    environment without either (GNN.py's own _ensure_torch_geometric_layers already warns
    about this at import time) it raises ImportError only once the loader is actually
    iterated, not at construction. Probe with one batch here so callers can fall back to
    full-graph training instead of crashing every GraphSAGE run in such environments.
    """
    try:
        loader = _build_neighbor_loader(ntw_torch, input_mask, batch_size, num_neighbors, shuffle)
        next(iter(loader))  # actually triggers the C++ sampler; construction alone does not.
        # Re-iterating the same DataLoader object below (a fresh `for batch in loader`) starts
        # a new iterator from scratch -- this probe batch isn't lost, just not reused.
        return loader
    except ImportError as e:
        print(f"[GNN_features] WARNING: neighbor-sampling unavailable ({e}); "
              f"falling back to full-graph training for GraphSAGE.")
        return None

# =====================================================================
# 1. Intrinsic Features (With and Without Predictions)
# =====================================================================
def intrinsic_features(
    ntw, train_mask, val_mask, test_mask, n_layers_decoder, hidden_dim_decoder, lr, n_epochs_decoder, ratio=None, sampling="none", percentile_q=99, patience=10, checkpoint_path="res/checkpoints/best_model_intrinsic_eval.pt", loss="ce", loss_kwargs=None, seed=None
):
    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    y_tensor = torch.tensor(ntw.df_features['class'].values, dtype=torch.long)
    features_tensor = ntw.get_features_torch()
    sampling_name = _normalize_sampling_name(sampling)

    if sampling_name == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool().to(device_decoder), y_tensor.to(device_decoder), ratio=ratio, random_state=seed)
    elif sampling_name == "smote" and ratio is not None:
        features_tensor, y_tensor, train_mask_sampled = smote_mask(train_mask.bool().to(device_decoder), features_tensor.to(device_decoder), y_tensor.to(device_decoder), ratio=ratio, random_state=seed)
    elif sampling_name == "targeted_neighbourhood_undersampling" and ratio is not None:
        sampler = TargetedNeighbourhoodUndersampling(remove_ratio=ratio, random_state=seed)
        train_mask_sampled = sampler(train_mask.bool().to(device_decoder), features_tensor.to(device_decoder), y_tensor.to(device_decoder))
    elif sampling_name == "none" or ratio is None:
        train_mask_sampled = train_mask.bool().to(device_decoder)
    else:
        raise ValueError(f"Unrecognized sampling technique: {sampling_name!r}")

    X_train = features_tensor[train_mask_sampled.cpu()].to(device_decoder)
    y_train = y_tensor[train_mask_sampled.cpu()].to(device_decoder)
    X_val = features_tensor[:val_mask.shape[0]][val_mask.bool().cpu()].to(device_decoder)
    y_val = y_tensor[:val_mask.shape[0]][val_mask.bool().cpu()].to(device_decoder)
    X_test = features_tensor[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)
    y_test = y_tensor[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)

    decoder = Decoder_deep_norm(X_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    criterion = _build_loss_criterion(y_train, loss_name=loss, loss_kwargs=loss_kwargs or {}, device=device_decoder)
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor='val_ap'
    )

    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(X_train)
        loss_val = _compute_loss(criterion, output, y_train)
        loss_val.backward()
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
    ntw, train_mask, val_mask, test_mask, n_layers_decoder, hidden_dim_decoder, lr, n_epochs_decoder, ratio=None, sampling="none", patience=10, checkpoint_path="res/checkpoints/best_model_intrinsic_tuned.pt", loss="ce", loss_kwargs=None, seed=None
):
    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    y_tensor = torch.tensor(ntw.df_features['class'].values, dtype=torch.long)
    features_tensor = ntw.get_features_torch()
    sampling_name = _normalize_sampling_name(sampling)

    if sampling_name == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool().to(device_decoder), y_tensor.to(device_decoder), ratio=ratio, random_state=seed)
    elif sampling_name == "smote" and ratio is not None:
        features_tensor, y_tensor, train_mask_sampled = smote_mask(train_mask.bool().to(device_decoder), features_tensor.to(device_decoder), y_tensor.to(device_decoder), ratio=ratio, random_state=seed)
    elif sampling_name == "targeted_neighbourhood_undersampling" and ratio is not None:
        sampler = TargetedNeighbourhoodUndersampling(remove_ratio=ratio, random_state=seed)
        train_mask_sampled = sampler(train_mask.bool().to(device_decoder), features_tensor.to(device_decoder), y_tensor.to(device_decoder))
    elif sampling_name == "none" or ratio is None:
        train_mask_sampled = train_mask.bool().to(device_decoder)
    else:
        raise ValueError(f"Unrecognized sampling technique: {sampling_name!r}")

    X_train = features_tensor[train_mask_sampled.cpu()].to(device_decoder)
    y_train = y_tensor[train_mask_sampled.cpu()].to(device_decoder)
    X_val = features_tensor[:val_mask.shape[0]][val_mask.bool().cpu()].to(device_decoder)
    y_val = y_tensor[:val_mask.shape[0]][val_mask.bool().cpu()].to(device_decoder)
    X_test = features_tensor[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)
    y_test = y_tensor[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)

    decoder = Decoder_deep_norm(X_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = _build_loss_criterion(y_train, loss_name=loss, loss_kwargs=loss_kwargs or {}, device=device_decoder)
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor='val_ap'
    )

    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(X_train)
        loss_val = _compute_loss(criterion, output, y_train)
        loss_val.backward()
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
    ntw, train_mask, val_mask, test_mask, alpha_pr: float, alpha_ppr: float, n_epochs_decoder: int, lr: float, fraud_dict_train: dict = None, fraud_dict_test: dict = None, n_layers_decoder: int = 2, hidden_dim_decoder: int = 5, ntw_name: str = None, use_intrinsic: bool = False, percentile_q: int = 99, ratio=None, sampling="none", patience=10, checkpoint_path="res/checkpoints/best_model_pos_eval.pt", loss="ce", loss_kwargs=None, seed=None
):
    print("intrinsic and summary: ")
    X_full_df = ntw.get_features(full=True)
    print("networkx (Full & Subgraph): ")
    ntw_nx_full = ntw.get_network_nx()

    # Subgraph isolation to completely block test set structure leakage
    train_val_mask = train_mask.bool() | val_mask.bool()
    train_val_nodes = set(torch.where(train_val_mask)[0].tolist())
    ntw_nx_train_val = ntw_nx_full.subgraph(list(train_val_nodes))

    # Calculate features on train-val subgraph with propagated seed
    features_nx_df_train_val = local_features_nx(
        ntw_nx_train_val, alpha_pr, alpha_ppr, fraud_dict_train=fraud_dict_train, ntw_name=ntw_name + "_train_val"
    )
    features_nk_df_train_val = features_nk(
        ntw_nx_train_val, ntw_name=ntw_name + "_train_val_nk", seed=seed
    )

    if use_intrinsic:
        X_train_val_df = X_full_df.loc[X_full_df.index.isin(features_nx_df_train_val.index)]
        features_df_train_val = pd.concat([X_train_val_df, features_nx_df_train_val, features_nk_df_train_val], axis=1)
    else:
        features_df_train_val = pd.concat([features_nx_df_train_val, features_nk_df_train_val], axis=1)

    features_df_train_val["fraud"] = [fraud_dict_test.get(x, 0) for x in features_df_train_val.index]

    # Map back to full-sized zeros to allow downstream masking code to work unmodified
    N_nodes = int(ntw_nx_full.number_of_nodes())
    x_features_train_val = features_df_train_val.drop(["PSP", "fraud"], axis=1, errors='ignore').values
    D_features = x_features_train_val.shape[1]
    full_size_x_train_val = np.zeros((N_nodes, D_features), dtype=np.float32)
    for idx, row in features_df_train_val.iterrows():
        full_size_x_train_val[int(idx)] = row.drop(["PSP", "fraud"], errors='ignore').values

    features_tensor_train_val = torch.tensor(full_size_x_train_val, dtype=torch.float32)
    y_tensor_train_val = torch.tensor([fraud_dict_test.get(i, 0) for i in range(N_nodes)], dtype=torch.long)

    # Calculate features on full graph for validation/test evaluation with propagated seed
    features_nx_df_full = local_features_nx(
        ntw_nx_full, alpha_pr, alpha_ppr, fraud_dict_train=fraud_dict_train, ntw_name=ntw_name + "_full"
    )
    ntw_nk_full = ntw.get_network_nk()
    features_nk_df_full = features_nk(
        ntw_nk_full, ntw_name=ntw_name + "_full_nk", seed=seed
    )

    if use_intrinsic:
        features_df_full = pd.concat([X_full_df, features_nx_df_full, features_nk_df_full], axis=1)
    else:
        features_df_full = pd.concat([features_nx_df_full, features_nk_df_full], axis=1)

    features_df_full["fraud"] = [fraud_dict_test.get(x, 0) for x in features_df_full.index]
    x_features_full = features_df_full.drop(["PSP", "fraud"], axis=1, errors='ignore').values
    features_tensor_full = torch.tensor(x_features_full, dtype=torch.float32)
    y_tensor_full = torch.tensor(features_df_full["fraud"].values, dtype=torch.long)

    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    sampling_name = _normalize_sampling_name(sampling)
    if sampling_name == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool(), y_tensor_train_val, ratio=ratio, random_state=seed)
    elif sampling_name == "smote" and ratio is not None:
        features_tensor_train_val, y_tensor_train_val, train_mask_sampled = smote_mask(train_mask.bool(), features_tensor_train_val, y_tensor_train_val, ratio=ratio, random_state=seed)
    elif sampling_name == "targeted_neighbourhood_undersampling" and ratio is not None:
        sampler = TargetedNeighbourhoodUndersampling(remove_ratio=ratio, random_state=seed)
        train_mask_sampled = sampler(train_mask.bool(), features_tensor_train_val, y_tensor_train_val)
    elif sampling_name == "none" or ratio is None:
        train_mask_sampled = train_mask.bool()
    else:
        raise ValueError(f"Unrecognized sampling technique: {sampling_name!r}")

    # Extracted splits
    X_train = features_tensor_train_val[train_mask_sampled.cpu()].to(device_decoder)
    y_train = y_tensor_train_val[train_mask_sampled.cpu()].to(device_decoder)
    X_val = features_tensor_train_val[:val_mask.shape[0]][val_mask.bool().cpu()].to(device_decoder)
    y_val = y_tensor_train_val[:val_mask.shape[0]][val_mask.bool().cpu()].to(device_decoder)
    X_test = features_tensor_full[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)
    y_test = y_tensor_full[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)

    decoder = Decoder_deep_norm(X_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = _build_loss_criterion(y_train, loss_name=loss, loss_kwargs=loss_kwargs or {}, device=device_decoder)
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor='val_ap'
    )

    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(X_train)
        loss_val = _compute_loss(criterion, output, y_train)
        loss_val.backward()
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
    ntw, train_mask, val_mask, test_mask, alpha_pr: float, alpha_ppr: float, n_epochs_decoder: int, lr: float, fraud_dict_train: dict = None, fraud_dict_test: dict = None, n_layers_decoder: int = 2, hidden_dim_decoder: int = 5, ntw_name: str = None, use_intrinsic: bool = False, ratio=None, sampling="none", patience=10, checkpoint_path="res/checkpoints/best_model_pos_tuned.pt", loss="ce", loss_kwargs=None, seed=None
):
    print("intrinsic and summary: ")
    X_full_df = ntw.get_features(full=True)
    print("networkx (Full & Subgraph): ")
    ntw_nx_full = ntw.get_network_nx()

    # Subgraph isolation to completely block test set structure leakage
    train_val_mask = train_mask.bool() | val_mask.bool()
    train_val_nodes = set(torch.where(train_val_mask)[0].tolist())
    ntw_nx_train_val = ntw_nx_full.subgraph(list(train_val_nodes))

    # Calculate features on train-val subgraph with propagated seed
    features_nx_df_train_val = local_features_nx(
        ntw_nx_train_val, alpha_pr, alpha_ppr, fraud_dict_train=fraud_dict_train, ntw_name=ntw_name + "_train_val"
    )
    features_nk_df_train_val = features_nk(
        ntw_nx_train_val, ntw_name=ntw_name + "_train_val_nk", seed=seed
    )

    if use_intrinsic:
        X_train_val_df = X_full_df.loc[X_full_df.index.isin(features_nx_df_train_val.index)]
        features_df_train_val = pd.concat([X_train_val_df, features_nx_df_train_val, features_nk_df_train_val], axis=1)
    else:
        features_df_train_val = pd.concat([features_nx_df_train_val, features_nk_df_train_val], axis=1)

    features_df_train_val["fraud"] = [fraud_dict_test.get(x, 0) for x in features_df_train_val.index]

    # Map back to full-sized zeros to allow downstream masking code to work unmodified
    N_nodes = int(ntw_nx_full.number_of_nodes())
    x_features_train_val = features_df_train_val.drop(["PSP", "fraud"], axis=1, errors='ignore').values
    D_features = x_features_train_val.shape[1]
    full_size_x_train_val = np.zeros((N_nodes, D_features), dtype=np.float32)
    for idx, row in features_df_train_val.iterrows():
        full_size_x_train_val[int(idx)] = row.drop(["PSP", "fraud"], errors='ignore').values

    features_tensor_train_val = torch.tensor(full_size_x_train_val, dtype=torch.float32)
    y_tensor_train_val = torch.tensor([fraud_dict_test.get(i, 0) for i in range(N_nodes)], dtype=torch.long)

    # Calculate features on full graph for validation/test evaluation with propagated seed
    features_nx_df_full = local_features_nx(
        ntw_nx_full, alpha_pr, alpha_ppr, fraud_dict_train=fraud_dict_train, ntw_name=ntw_name + "_full"
    )
    ntw_nk_full = ntw.get_network_nk()
    features_nk_df_full = features_nk(
        ntw_nk_full, ntw_name=ntw_name + "_full_nk", seed=seed
    )

    if use_intrinsic:
        features_df_full = pd.concat([X_full_df, features_nx_df_full, features_nk_df_full], axis=1)
    else:
        features_df_full = pd.concat([features_nx_df_full, features_nk_df_full], axis=1)

    features_df_full["fraud"] = [fraud_dict_test.get(x, 0) for x in features_df_full.index]
    x_features_full = features_df_full.drop(["PSP", "fraud"], axis=1, errors='ignore').values
    nan_count_full = int(np.isnan(x_features_full).sum())
    if nan_count_full > 0:
        print(f"[positional_features_with_predictions] WARNING: {nan_count_full} NaN values found in "
              f"features_tensor_full (test-time feature table) and replaced with 0.0. This should not "
              f"happen if the graph-conversion pipeline covers every node; investigate if this fires.")
    features_tensor_full = torch.nan_to_num(
        torch.tensor(x_features_full, dtype=torch.float32), nan=0.0, posinf=1e5, neginf=-1e5
    )
    y_tensor_full = torch.tensor(features_df_full["fraud"].values, dtype=torch.long)

    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    sampling_name = _normalize_sampling_name(sampling)
    if sampling_name == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool(), y_tensor_train_val, ratio=ratio, random_state=seed)
    elif sampling_name == "smote" and ratio is not None:
        features_tensor_train_val, y_tensor_train_val, train_mask_sampled = smote_mask(train_mask.bool(), features_tensor_train_val, y_tensor_train_val, ratio=ratio, random_state=seed)
    elif sampling_name == "targeted_neighbourhood_undersampling" and ratio is not None:
        sampler = TargetedNeighbourhoodUndersampling(remove_ratio=ratio, random_state=seed)
        train_mask_sampled = sampler(train_mask.bool(), features_tensor_train_val, y_tensor_train_val)
    elif sampling_name == "none" or ratio is None:
        train_mask_sampled = train_mask.bool()
    else:
        raise ValueError(f"Unrecognized sampling technique: {sampling_name!r}")

    # Extracted splits
    X_train = features_tensor_train_val[train_mask_sampled.cpu()].to(device_decoder)
    y_train = y_tensor_train_val[train_mask_sampled.cpu()].to(device_decoder)
    X_val = features_tensor_train_val[:val_mask.shape[0]][val_mask.bool().cpu()].to(device_decoder)
    y_val = y_tensor_train_val[:val_mask.shape[0]][val_mask.bool().cpu()].to(device_decoder)
    X_test = features_tensor_full[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)
    y_test = y_tensor_full[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder)

    decoder = Decoder_deep_norm(X_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = _build_loss_criterion(y_train, loss_name=loss, loss_kwargs=loss_kwargs or {}, device=device_decoder)
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor='val_ap'
    )

    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(X_train)
        loss_val = _compute_loss(criterion, output, y_train)
        loss_val.backward()
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
    ntw_torch, train_mask, val_mask, test_mask, embedding_dim, walk_length, context_size, walks_per_node, num_negative_samples, p, q, lr=0.01, n_epochs=1, n_epochs_decoder=1, ntw_nx=None, use_torch=False, use_intrinsic=True, percentile_q=99, ratio=None, sampling="none", loss="ce", loss_kwargs=None, seed=None, patience=10, checkpoint_path="res/checkpoints/best_model_node2vec.pt"
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
    sampling_name = _normalize_sampling_name(sampling)
    if sampling_name == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool(), y_tensor, ratio=ratio, random_state=seed)
    elif sampling_name == "smote" and ratio is not None:
        x, y_tensor, train_mask_sampled = smote_mask(train_mask.bool(), x, y_tensor, ratio=ratio, random_state=seed)
    elif sampling_name in ["graph_smote", "graphens"] and ratio is not None:
        x, y_tensor, train_mask_sampled = smote_mask(train_mask.bool(), x, y_tensor, ratio=ratio, random_state=seed)
    elif sampling_name == "targeted_neighbourhood_undersampling" and ratio is not None:
        sampler = TargetedNeighbourhoodUndersampling(remove_ratio=ratio, random_state=seed)
        train_mask_sampled = sampler(train_mask.bool(), x, y_tensor)
    elif sampling_name == "none" or ratio is None:
        train_mask_sampled = train_mask.bool()
    else:
        raise ValueError(f"Unrecognized sampling technique: {sampling_name!r}")

    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    x_train = x[train_mask_sampled].to(device_decoder).squeeze()
    x_val = x[:val_mask.shape[0]][val_mask.bool()].to(device_decoder).squeeze()
    x_test = x[:test_mask.shape[0]][test_mask.bool()].to(device_decoder).squeeze()
    y_train = y_tensor[train_mask_sampled].to(device_decoder).squeeze()
    y_val = ntw_torch.y[:val_mask.shape[0]][val_mask.bool()].to(device_decoder).squeeze()
    y_test = ntw_torch.y[:test_mask.shape[0]][test_mask.bool()].to(device_decoder).squeeze()

    decoder = Decoder_deep_norm(x_train.shape[1], 2, 10).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = _build_loss_criterion(y_train, loss_name=loss, loss_kwargs=loss_kwargs or {}, device=device_decoder)
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor='val_ap'
    )

    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(x_train)
        loss_val = _compute_loss(criterion, output, y_train)
        loss_val.backward()
        optimizer.step()

        decoder.eval()
        with torch.no_grad():
            val_output = decoder(x_val)
            val_output_softmax = val_output.softmax(dim=1)
            val_ap = average_precision_score(y_val.cpu().numpy(), val_output_softmax.cpu().numpy()[:, 1])
            early_stopping(val_ap, decoder)
        if early_stopping.early_stop:
            break

    if os.path.exists(checkpoint_path):
        decoder.load_state_dict(torch.load(checkpoint_path, map_location=device_decoder))

    decoder.eval()
    y_pred = decoder(x_test)
    y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    cutoff = np.percentile(y_pred.cpu().detach().numpy()[:,1], percentile_q)
    y_pred_hard = (y_pred.cpu().detach().numpy()[:,1] >= cutoff).astype(int)
    f1 = f1_score(y_test.cpu().detach().numpy(), y_pred_hard)
    return (ap_score, f1)

def node2vec_features_with_predictions(
    ntw_torch, train_mask, val_mask, test_mask, embedding_dim, walk_length, context_size, walks_per_node, num_negative_samples, p, q, lr=0.01, n_epochs=1, n_epochs_decoder=1, ntw_nx=None, use_torch=False, use_intrinsic=True, ratio=None, sampling="none", loss="ce", loss_kwargs=None, seed=None, patience=10, checkpoint_path="res/checkpoints/best_model_node2vec_tuned.pt"
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
    sampling_name = _normalize_sampling_name(sampling)
    if sampling_name == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool(), y_tensor, ratio=ratio, random_state=seed)
    elif sampling_name == "smote" and ratio is not None:
        x, y_tensor, train_mask_sampled = smote_mask(train_mask.bool(), x, y_tensor, ratio=ratio, random_state=seed)
    elif sampling_name in ["graph_smote", "graphens"] and ratio is not None:
        x, y_tensor, train_mask_sampled = smote_mask(train_mask.bool(), x, y_tensor, ratio=ratio, random_state=seed)
    elif sampling_name == "targeted_neighbourhood_undersampling" and ratio is not None:
        sampler = TargetedNeighbourhoodUndersampling(remove_ratio=ratio, random_state=seed)
        train_mask_sampled = sampler(train_mask.bool(), x, y_tensor)
    elif sampling_name == "none" or ratio is None:
        train_mask_sampled = train_mask.bool()
    else:
        raise ValueError(f"Unrecognized sampling technique: {sampling_name!r}")

    device_decoder = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    x_train = x[train_mask_sampled.cpu()].to(device_decoder).squeeze()
    x_val = x[:val_mask.shape[0]][val_mask.bool().cpu()].to(device_decoder).squeeze()
    x_test = x[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder).squeeze()
    y_train = y_tensor[train_mask_sampled.cpu()].to(device_decoder).squeeze()
    y_val = ntw_torch.y[:val_mask.shape[0]][val_mask.bool().cpu()].to(device_decoder).squeeze()
    y_test = ntw_torch.y[:test_mask.shape[0]][test_mask.bool().cpu()].to(device_decoder).squeeze()

    decoder = Decoder_deep_norm(x_train.shape[1], 2, 10).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = _build_loss_criterion(y_train, loss_name=loss, loss_kwargs=loss_kwargs or {}, device=device_decoder)
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor='val_ap'
    )

    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(x_train)
        loss_val = _compute_loss(criterion, output, y_train)
        loss_val.backward()
        optimizer.step()

        decoder.eval()
        with torch.no_grad():
            val_output = decoder(x_val)
            val_output_softmax = val_output.softmax(dim=1)
            val_ap = average_precision_score(y_val.cpu().numpy(), val_output_softmax.cpu().numpy()[:, 1])
            early_stopping(val_ap, decoder)
        if early_stopping.early_stop:
            break

    if os.path.exists(checkpoint_path):
        decoder.load_state_dict(torch.load(checkpoint_path, map_location=device_decoder))

    decoder.eval()
    y_pred = decoder(x_test)
    y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    return ap_score, y_pred.cpu().detach().numpy()[:,1], y_test.cpu().detach().numpy()

# =====================================================================
# 4. Standard GNN Methods (Validation, Slicing Protected)
# =====================================================================
def GNN_features(
    ntw_torch, model: nn.Module, lr: float, n_epochs: int, train_mask: torch.Tensor = None, val_mask: torch.Tensor = None, test_mask: torch.Tensor = None, use_intrinsic: bool = True, percentile_q: int = 99, patience: int = 10, checkpoint_path: str = "res/checkpoints/best_model.pt", monitor: str = 'val_ap', ratio=None, sampling="none", loss="ce", loss_kwargs=None, seed=None, sage_batch_size=1024, sage_num_neighbors=None
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor=monitor
    )

    y_tensor = ntw_torch.y.cpu()
    sampling_name = _normalize_sampling_name(sampling)

    if sampling_name == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool(), y_tensor, ratio=ratio, random_state=seed).to(device)
    elif sampling_name == "smote" and ratio is not None:
        x_np, y_np, train_mask_np = smote_mask(train_mask.bool(), ntw_torch.x, ntw_torch.y, ratio=ratio, random_state=seed)
        ntw_torch = ntw_torch.clone()
        ntw_torch.x = x_np.to(device)
        ntw_torch.y = y_np.to(device)
        train_mask_sampled = train_mask_np.to(device)
    elif sampling_name == "targeted_neighbourhood_undersampling" and ratio is not None:
        sampler = TargetedNeighbourhoodUndersampling(remove_ratio=ratio, random_state=seed)
        train_mask_np = sampler(train_mask.bool(), ntw_torch.x.cpu(), ntw_torch.y.cpu())
        train_mask_sampled = train_mask_np.to(device)
    elif sampling_name == "none" or ratio is None:
        train_mask_sampled = train_mask.bool().to(device)
    else:
        raise ValueError(f"Unrecognized sampling technique: {sampling_name!r}")

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
        return _build_loss_criterion(y_subset, loss_name=loss, loss_kwargs=loss_kwargs or {}, device=device)

    # GraphSAGE gets real neighbor-sampling mini-batch training instead of the full-graph
    # forward every other architecture uses -- see _build_neighbor_loader. Evaluation stays
    # full-batch (standard transductive-eval practice; the graphs here are small enough that
    # eval memory was never the bottleneck training was).
    use_neighbor_sampling = isinstance(model, GraphSAGE)
    if use_neighbor_sampling:
        num_neighbors = sage_num_neighbors or [15] * max(int(model.n_layers), 1)
        train_loader = _try_build_neighbor_loader(ntw_torch, train_mask_sampled, sage_batch_size, num_neighbors, shuffle=True)
        use_neighbor_sampling = train_loader is not None

    def train_epoch():
        model.train()
        if use_neighbor_sampling:
            total_loss, total_seeds = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                bx = batch.x if use_intrinsic else torch.ones((batch.x.shape[0], 1), dtype=torch.float32, device=device)
                out, _ = model(bx, batch.edge_index)
                seed_n = batch.batch_size
                y_batch = batch.y[:seed_n].long()
                criterion = _build_weighted_criterion(y_batch)
                loss_val = _compute_loss(criterion, out[:seed_n], y_batch)
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss_val.item() * seed_n
                total_seeds += seed_n
            return total_loss / max(total_seeds, 1)
        optimizer.zero_grad()
        out, _ = _forward(ntw_torch.x.to(device), ntw_torch.edge_index.to(device))
        y = ntw_torch.y.long().to(device)
        train_dev = train_mask_sampled
        y_train = y[train_dev]
        criterion = _build_weighted_criterion(y_train)
        loss_val = _compute_loss(criterion, out[train_dev], y_train)
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        return loss_val.item()

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
            loss_val = _compute_loss(criterion, out_filtered, y_filtered).item()
            y_hat = out_filtered.softmax(dim=1)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
            ap_score = average_precision_score(y_filtered.cpu().numpy(), y_hat.cpu().numpy()[:, 1])
            cutoff = np.percentile(y_hat.cpu().numpy()[:, 1], percentile_q)
            y_pred_hard = (y_hat.cpu().numpy()[:, 1] >= cutoff).astype(int)
            f1 = f1_score(y_filtered.cpu().numpy(), y_pred_hard)
            return {'loss': loss_val, 'ap': ap_score, 'f1': f1}

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
    ntw_torch, model: nn.Module, lr: float, n_epochs: int, train_mask: torch.Tensor = None, val_mask: torch.Tensor = None, test_mask: torch.Tensor = None, use_intrinsic: bool = True, patience: int = 10, checkpoint_path: str = "res/checkpoints/best_model.pt", monitor: str = 'val_ap', ratio=None, sampling="none", loss="ce", loss_kwargs=None, seed=None, sage_batch_size=1024, sage_num_neighbors=None
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor=monitor
    )

    y_tensor = ntw_torch.y.cpu()
    sampling_name = _normalize_sampling_name(sampling)

    if sampling_name == "random_undersample" and ratio is not None:
        train_mask_sampled = random_undersample_mask(train_mask.bool(), y_tensor, ratio=ratio, random_state=seed).to(device)
    elif sampling_name == "smote" and ratio is not None:
        x_np, y_np, train_mask_np = smote_mask(train_mask.bool(), ntw_torch.x, ntw_torch.y, ratio=ratio, random_state=seed)
        ntw_torch = ntw_torch.clone()
        ntw_torch.x = x_np.to(device)
        ntw_torch.y = y_np.to(device)
        train_mask_sampled = train_mask_np.to(device)
    elif sampling_name == "targeted_neighbourhood_undersampling" and ratio is not None:
        sampler = TargetedNeighbourhoodUndersampling(remove_ratio=ratio, random_state=seed)
        train_mask_np = sampler(train_mask.bool(), ntw_torch.x.cpu(), ntw_torch.y.cpu())
        train_mask_sampled = train_mask_np.to(device)
    elif sampling_name == "none" or ratio is None:
        train_mask_sampled = train_mask.bool().to(device)
    else:
        raise ValueError(f"Unrecognized sampling technique: {sampling_name!r}")

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
        return _build_loss_criterion(y_subset, loss_name=loss, loss_kwargs=loss_kwargs or {}, device=device)

    # See GNN_features for rationale: GraphSAGE gets real neighbor-sampling mini-batches,
    # every other architecture keeps the full-graph forward. Eval stays full-batch.
    use_neighbor_sampling = isinstance(model, GraphSAGE)
    if use_neighbor_sampling:
        num_neighbors = sage_num_neighbors or [15] * max(int(model.n_layers), 1)
        train_loader = _try_build_neighbor_loader(ntw_torch, train_mask_sampled, sage_batch_size, num_neighbors, shuffle=True)
        use_neighbor_sampling = train_loader is not None

    def train_epoch():
        model.train()
        if use_neighbor_sampling:
            total_loss, total_seeds = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                bx = batch.x if use_intrinsic else torch.ones((batch.x.shape[0], 1), dtype=torch.float32, device=device)
                out, _ = model(bx, batch.edge_index)
                seed_n = batch.batch_size
                y_batch = batch.y[:seed_n].long()
                criterion = _build_weighted_criterion(y_batch)
                loss_val = _compute_loss(criterion, out[:seed_n], y_batch)
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss_val.item() * seed_n
                total_seeds += seed_n
            return total_loss / max(total_seeds, 1)
        optimizer.zero_grad()
        out, _ = _forward(ntw_torch.x.to(device), ntw_torch.edge_index.to(device))
        y = ntw_torch.y.long().to(device)
        train_dev = train_mask_sampled
        y_train = y[train_dev]
        criterion = _build_weighted_criterion(y_train)
        loss_val = _compute_loss(criterion, out[train_dev], y_train)
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        return loss_val.item()

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
            loss_val = _compute_loss(criterion, out_filtered, y_filtered).item()
            y_hat = out_filtered.softmax(dim=1)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
            ap_score = average_precision_score(y_filtered.cpu().numpy(), y_hat.cpu().numpy()[:, 1])
            return {'loss': loss_val, 'ap': ap_score, 'output': y_hat, 'y': y_filtered}

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
# 5. GNN GraphSMOTE (Train-Evaluation Graph Match)
# =====================================================================
def _build_graphsmote_sampling(
    sampling_name, train_mask, ntw_torch, k_neighbors, ratio, random_state,
    gatsmote_k_neighbors, gatsmote_heads, gatsmote_edge_threshold, gatsmote_lambda1, gatsmote_lambda2,
    gatsmote_use_predicted_labels_for_homophily,
    tnu_k_neighbors, tnu_distance_metric, tnu_remove_ratio, tnu_noise_threshold,
    tnu_min_majority_keep, tnu_preserve_minority_neighbors,
    device,
):
    """Shared sampling dispatch for GNN_features_graphsmote(_with_predictions). Returns
    (x_smote, y_smote, train_mask_smote, edge_index_smote, gat_edge_gen, synthetic_pairs,
    pair_label_match) -- the last three are non-None only for sampling_name == 'gatsmote',
    since that is the only technique whose topology is recomputed every training step.
    """
    gat_edge_gen = None
    synthetic_pairs = None
    pair_label_match = None

    if sampling_name == "gatsmote":
        gat_edge_gen = GATEdgeGenerator(
            in_dim=ntw_torch.x.shape[1], k_neighbors=gatsmote_k_neighbors, attention_heads=gatsmote_heads,
            edge_threshold=gatsmote_edge_threshold, lambda_locality=gatsmote_lambda1, lambda_shortest=gatsmote_lambda2,
            ratio=ratio, use_predicted_labels_for_homophily=gatsmote_use_predicted_labels_for_homophily,
            random_state=random_state,
        ).to(device)
        prepared = gat_edge_gen.prepare_synthetic_nodes(train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index)
        x_smote, y_smote, train_mask_smote, edge_index_smote = (
            prepared['features'], prepared['labels'], prepared['mask'], prepared['edge_index']
        )
        synthetic_pairs = prepared['synthetic_pairs'].to(device)
        pair_label_match = prepared['pair_label_match'].to(device)
    elif sampling_name == "targeted_neighbourhood_undersampling":
        # --tnu-remove-ratio overrides the outer imbalance-ratio sweep only when explicitly
        # set; its default (None) preserves the existing behavior of following `ratio` so the
        # sweep grid still drives TNU by default.
        effective_remove_ratio = tnu_remove_ratio if tnu_remove_ratio is not None else ratio
        sampler = TargetedNeighbourhoodUndersampling(
            k_neighbors=tnu_k_neighbors, distance_metric=tnu_distance_metric, remove_ratio=effective_remove_ratio,
            preserve_minority_neighbors=tnu_preserve_minority_neighbors, noise_threshold=tnu_noise_threshold,
            min_majority_keep=tnu_min_majority_keep, random_state=random_state,
        )
        train_mask_smote = sampler(train_mask, ntw_torch.x, ntw_torch.y)
        x_smote, y_smote, edge_index_smote = ntw_torch.x, ntw_torch.y, ntw_torch.edge_index
    elif sampling_name == "graph_smote":
        x_smote, y_smote, train_mask_smote, edge_index_smote = graph_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, ratio=ratio, random_state=random_state
        )
    else:
        raise ValueError(f"Unrecognized sampling technique: {sampling_name!r}")

    return x_smote, y_smote, train_mask_smote, edge_index_smote, gat_edge_gen, synthetic_pairs, pair_label_match


def GNN_features_graphsmote(
    ntw_torch, model: nn.Module, lr: float, n_epochs: int, train_mask: torch.Tensor = None, val_mask: torch.Tensor = None, test_mask: torch.Tensor = None, use_intrinsic: bool = True, k_neighbors: int = 5, random_state: int = None, percentile_q: int = 99, sampling: str = "graph_smote", patience: int = 10, checkpoint_path: str = "res/checkpoints/best_model_graphsmote.pt", monitor: str = 'val_ap', ratio=None, loss="ce", loss_kwargs=None, gatsmote_k_neighbors=5, gatsmote_attention_heads=1, gatsmote_edge_threshold=0.5, gatsmote_homophily_weight=1.0, gatsmote_use_predicted_labels_for_homophily=False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    def _build_weighted_criterion(y_subset):
        return _build_loss_criterion(y_subset, loss_name=loss, loss_kwargs=loss_kwargs or {}, device=device)

    def _forward(x, edge_index, edge_attr=None):
        if use_intrinsic:
            return model(x, edge_index, edge_attr=edge_attr)
        ones = torch.ones((x.shape[0], 1), dtype=torch.float32, device=device)
        return model(ones, edge_index, edge_attr=edge_attr)

    sampling_name = _normalize_sampling_name(sampling)
    # ratio=None ("Original" row in the sweep) must mean "no resampling", matching the
    # semantics every other sampling family already uses (GNN_features/intrinsic_features
    # gate their sampling branches on `and ratio is not None`, falling back to the
    # untouched graph when ratio is None). Without this branch, graph_smote_mask/GATSMOTE
    # would still fully rebalance to ~1:1 and TNU would still remove every flagged noisy
    # node under ratio=None, silently contradicting the "original distribution" label.
    if ratio is None:
        x_smote, y_smote, train_mask_smote, edge_index_smote = ntw_torch.x, ntw_torch.y, train_mask.bool(), ntw_torch.edge_index
        edge_attr_smote = None
    elif sampling_name == "reweighted_graph_smote":
        x_smote, y_smote, train_mask_smote, edge_index_smote, edge_attr_smote = reweighted_graph_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, ratio=ratio, random_state=random_state
        )
    elif sampling_name == "unweighted_graph_smote":
        x_smote, y_smote, train_mask_smote, edge_index_smote = unweighted_graph_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, ratio=ratio, random_state=random_state
        )
        edge_attr_smote = None
    elif sampling_name == "gatsmote":
        sampler = GATSMOTE(k_neighbors=gatsmote_k_neighbors, attention_heads=gatsmote_attention_heads, edge_threshold=gatsmote_edge_threshold, homophily_weight=gatsmote_homophily_weight, ratio=ratio, use_predicted_labels_for_homophily=gatsmote_use_predicted_labels_for_homophily, random_state=random_state)
        x_smote, y_smote, train_mask_smote, edge_index_smote = sampler(train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index)
        edge_attr_smote = None
    elif sampling_name == "targeted_neighbourhood_undersampling":
        sampler = TargetedNeighbourhoodUndersampling(remove_ratio=ratio, random_state=random_state)
        train_mask_smote = sampler(train_mask, ntw_torch.x, ntw_torch.y)
        x_smote, y_smote, edge_index_smote = ntw_torch.x, ntw_torch.y, ntw_torch.edge_index
        edge_attr_smote = None
    elif sampling_name == "graph_smote":
        x_smote, y_smote, train_mask_smote, edge_index_smote = graph_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, ratio=ratio, random_state=random_state
        )
        edge_attr_smote = None
    else:
        raise ValueError(f"Unrecognized sampling technique: {sampling_name!r}")

    ntw_torch_smote = ntw_torch.clone()
    ntw_torch_smote.x = x_smote.to(device)
    ntw_torch_smote.y = y_smote.long().to(device)
    ntw_torch_smote.edge_index = edge_index_smote.long().to(device)
    train_mask_smote = train_mask_smote.bool().to(device)

    # Create padded masks for expanded graph size validation/test symmetry
    n_synthetic = int(x_smote.shape[0] - ntw_torch.x.shape[0])
    if val_mask is not None:
        val_mask_smote = torch.cat([val_mask.bool().cpu(), torch.zeros(n_synthetic, dtype=torch.bool)]).to(device)
    else:
        val_mask_smote = None
    test_mask_smote = torch.cat([test_mask.bool().cpu(), torch.zeros(n_synthetic, dtype=torch.bool)]).to(device)

    if gat_edge_gen is not None:
        joint_params = list(model.parameters()) + list(gat_edge_gen.parameters())
    else:
        joint_params = list(model.parameters())
    optimizer = torch.optim.Adam(joint_params, lr=lr, weight_decay=5e-4)

    def _current_graph():
        if gat_edge_gen is None:
            return ntw_torch_smote.edge_index, None, None, None
        return gat_edge_gen.build_epoch_graph(ntw_torch_smote.edge_index, ntw_torch_smote.x, synthetic_pairs, pair_label_match)

    def train_epoch():
        model.train()
        if gat_edge_gen is not None:
            gat_edge_gen.train()
        optimizer.zero_grad()
        edge_index_epoch, edge_attr_epoch, loss_locality, loss_shortest = _current_graph()
        out, _ = _forward(ntw_torch_smote.x, edge_index_epoch, edge_attr=edge_attr_epoch)
        y = ntw_torch_smote.y
        active_mask = train_mask_smote
        if active_mask.any():
            y_hat_filtered = out[active_mask]
            y_filtered = y[active_mask]
            criterion = _build_weighted_criterion(y_filtered)
            loss_node = _compute_loss(criterion, y_hat_filtered, y_filtered)
            if gat_edge_gen is not None:
                loss_val = loss_node + gat_edge_gen.lambda_locality * loss_locality + gat_edge_gen.lambda_shortest * loss_shortest
            else:
                loss_val = loss_node
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(joint_params, max_norm=clip_norm)
            optimizer.step()
            return loss_val.item()
        return 0.0

    def evaluate_split(mask_smote):
        model.eval()
        if gat_edge_gen is not None:
            gat_edge_gen.eval()
        with torch.no_grad():
            edge_index_epoch, edge_attr_epoch, _, _ = _current_graph()
            out, _ = _forward(ntw_torch_smote.x, edge_index_epoch, edge_attr=edge_attr_epoch)
            y = ntw_torch_smote.y
            mask_dev = mask_smote.bool().to(device)
            out_filtered = out[mask_dev]
            y_filtered = y[mask_dev]
            if out_filtered.shape[0] == 0:
                return None
            criterion = _build_weighted_criterion(y_filtered)
            loss_val = _compute_loss(criterion, out_filtered, y_filtered).item()
            y_hat = out_filtered.softmax(dim=1)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
            ap_score = average_precision_score(y_filtered.cpu().numpy(), y_hat.cpu().numpy()[:, 1])
            cutoff = np.percentile(y_hat.cpu().numpy()[:, 1], percentile_q)
            y_pred_hard = (y_hat.cpu().numpy()[:, 1] >= cutoff).astype(int)
            f1 = f1_score(y_filtered.cpu().numpy(), y_pred_hard)
            return {'loss': loss_val, 'ap': ap_score, 'f1': f1}

    for epoch in range(n_epochs):
        train_loss = train_epoch()
        if val_mask_smote is not None:
            val_result = evaluate_split(val_mask_smote)
            if val_result is not None:
                print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | val_loss={val_result['loss']:.6f} | val_ap={val_result['ap']:.6f}")
                metric_to_monitor = val_result['ap'] if monitor == 'val_ap' else val_result['loss']
                early_stopping(metric_to_monitor, checkpoint_target)
        if early_stopping.early_stop:
            print(f"[{sampling}] Early Stop triggered!")
            break

    if val_mask_smote is not None and os.path.exists(checkpoint_path):
        checkpoint_target.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_result = evaluate_split(test_mask_smote)
    return test_result['ap'], test_result['f1']

def GNN_features_graphsmote_with_predictions(
    ntw_torch, model: nn.Module, lr: float, n_epochs: int, train_mask: torch.Tensor = None, val_mask: torch.Tensor = None, test_mask: torch.Tensor = None, use_intrinsic: bool = True, k_neighbors: int = 5, random_state: int = None, sampling: str = "graph_smote", patience: int = 10, checkpoint_path: str = "res/checkpoints/best_model_graphsmote.pt", monitor: str = 'val_ap', ratio=None, loss="ce", loss_kwargs=None, gatsmote_k_neighbors=5, gatsmote_attention_heads=1, gatsmote_edge_threshold=0.5, gatsmote_homophily_weight=1.0, gatsmote_use_predicted_labels_for_homophily=False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    def _build_weighted_criterion(y_subset):
        return _build_loss_criterion(y_subset, loss_name=loss, loss_kwargs=loss_kwargs or {}, device=device)

    def _forward(x, edge_index, edge_attr=None):
        if use_intrinsic:
            return model(x, edge_index, edge_attr=edge_attr)
        ones = torch.ones((x.shape[0], 1), dtype=torch.float32, device=device)
        return model(ones, edge_index, edge_attr=edge_attr)

    sampling_name = _normalize_sampling_name(sampling)
    # ratio=None ("Original" row in the sweep) must mean "no resampling", matching the
    # semantics every other sampling family already uses (GNN_features/intrinsic_features
    # gate their sampling branches on `and ratio is not None`, falling back to the
    # untouched graph when ratio is None). Without this branch, graph_smote_mask/GATSMOTE
    # would still fully rebalance to ~1:1 and TNU would still remove every flagged noisy
    # node under ratio=None, silently contradicting the "original distribution" label.
    if ratio is None:
        x_smote, y_smote, train_mask_smote, edge_index_smote = ntw_torch.x, ntw_torch.y, train_mask.bool(), ntw_torch.edge_index
        edge_attr_smote = None
    elif sampling_name == "reweighted_graph_smote":
        x_smote, y_smote, train_mask_smote, edge_index_smote, edge_attr_smote = reweighted_graph_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, ratio=ratio, random_state=random_state
        )
    elif sampling_name == "unweighted_graph_smote":
        x_smote, y_smote, train_mask_smote, edge_index_smote = unweighted_graph_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, ratio=ratio, random_state=random_state
        )
        edge_attr_smote = None
    elif sampling_name == "gatsmote":
        sampler = GATSMOTE(k_neighbors=gatsmote_k_neighbors, attention_heads=gatsmote_attention_heads, edge_threshold=gatsmote_edge_threshold, homophily_weight=gatsmote_homophily_weight, ratio=ratio, use_predicted_labels_for_homophily=gatsmote_use_predicted_labels_for_homophily, random_state=random_state)
        x_smote, y_smote, train_mask_smote, edge_index_smote = sampler(train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index)
        edge_attr_smote = None
    elif sampling_name == "targeted_neighbourhood_undersampling":
        sampler = TargetedNeighbourhoodUndersampling(remove_ratio=ratio, random_state=random_state)
        train_mask_smote = sampler(train_mask, ntw_torch.x, ntw_torch.y)
        x_smote, y_smote, edge_index_smote = ntw_torch.x, ntw_torch.y, ntw_torch.edge_index
        edge_attr_smote = None
    elif sampling_name == "graph_smote":
        x_smote, y_smote, train_mask_smote, edge_index_smote = graph_smote_mask(
            train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index, k_neighbors=k_neighbors, ratio=ratio, random_state=random_state
        )
        edge_attr_smote = None
    else:
        raise ValueError(f"Unrecognized sampling technique: {sampling_name!r}")

    ntw_torch_smote = ntw_torch.clone()
    ntw_torch_smote.x = x_smote.to(device)
    ntw_torch_smote.y = y_smote.long().to(device)
    ntw_torch_smote.edge_index = edge_index_smote.long().to(device)
    train_mask_smote = train_mask_smote.bool().to(device)

    # Create padded masks for expanded graph size validation/test symmetry
    n_synthetic = int(x_smote.shape[0] - ntw_torch.x.shape[0])
    if val_mask is not None:
        val_mask_smote = torch.cat([val_mask.bool().cpu(), torch.zeros(n_synthetic, dtype=torch.bool)]).to(device)
    else:
        val_mask_smote = None
    test_mask_smote = torch.cat([test_mask.bool().cpu(), torch.zeros(n_synthetic, dtype=torch.bool)]).to(device)

    if gat_edge_gen is not None:
        joint_params = list(model.parameters()) + list(gat_edge_gen.parameters())
    else:
        joint_params = list(model.parameters())
    optimizer = torch.optim.Adam(joint_params, lr=lr, weight_decay=5e-4)

    def _current_graph():
        if gat_edge_gen is None:
            return ntw_torch_smote.edge_index, None, None, None
        return gat_edge_gen.build_epoch_graph(ntw_torch_smote.edge_index, ntw_torch_smote.x, synthetic_pairs, pair_label_match)

    def train_epoch():
        model.train()
        if gat_edge_gen is not None:
            gat_edge_gen.train()
        optimizer.zero_grad()
        edge_index_epoch, edge_attr_epoch, loss_locality, loss_shortest = _current_graph()
        out, _ = _forward(ntw_torch_smote.x, edge_index_epoch, edge_attr=edge_attr_epoch)
        y = ntw_torch_smote.y
        active_mask = train_mask_smote
        if active_mask.any():
            y_hat_filtered = out[active_mask]
            y_filtered = y[active_mask]
            criterion = _build_weighted_criterion(y_filtered)
            loss_node = _compute_loss(criterion, y_hat_filtered, y_filtered)
            if gat_edge_gen is not None:
                loss_val = loss_node + gat_edge_gen.lambda_locality * loss_locality + gat_edge_gen.lambda_shortest * loss_shortest
            else:
                loss_val = loss_node
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(joint_params, max_norm=clip_norm)
            optimizer.step()
            return loss_val.item()
        return 0.0

    def evaluate_split(mask_smote):
        model.eval()
        if gat_edge_gen is not None:
            gat_edge_gen.eval()
        with torch.no_grad():
            edge_index_epoch, edge_attr_epoch, _, _ = _current_graph()
            out, _ = _forward(ntw_torch_smote.x, edge_index_epoch, edge_attr=edge_attr_epoch)
            y = ntw_torch_smote.y
            mask_dev = mask_smote.bool().to(device)
            out_filtered = out[mask_dev]
            y_filtered = y[mask_dev]
            if out_filtered.shape[0] == 0:
                return None
            criterion = _build_weighted_criterion(y_filtered)
            loss_val = _compute_loss(criterion, out_filtered, y_filtered).item()
            y_hat = out_filtered.softmax(dim=1)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
            ap_score = average_precision_score(y_filtered.cpu().numpy(), y_hat.cpu().numpy()[:, 1])
            return {'loss': loss_val, 'ap': ap_score, 'output': y_hat, 'y': y_filtered}

    for epoch in range(n_epochs):
        train_loss = train_epoch()
        if val_mask_smote is not None:
            val_result = evaluate_split(val_mask_smote)
            if val_result is not None:
                print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | val_loss={val_result['loss']:.6f} | val_ap={val_result['ap']:.6f}")
                metric_to_monitor = val_result['ap'] if monitor == 'val_ap' else val_result['loss']
                early_stopping(metric_to_monitor, checkpoint_target)
        if early_stopping.early_stop:
            print(f"[{sampling}] Early Stop triggered!")
            break

    if val_mask_smote is not None and os.path.exists(checkpoint_path):
        checkpoint_target.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_result = evaluate_split(test_mask_smote)
    return test_result['ap'], test_result['output'].cpu().numpy()[:, 1], test_result['y'].cpu().numpy()

# =====================================================================
# 6. GNN GraphENS (True Algorithm 1 Implementation, Park et al. ICLR 2022)
# =====================================================================
def GNN_features_graphens_with_predictions(
    ntw_torch, model: nn.Module, lr: float, n_epochs: int,
    train_mask: torch.Tensor = None, val_mask: torch.Tensor = None, test_mask: torch.Tensor = None,
    use_intrinsic: bool = True, random_state: int = None, patience: int = 10,
    checkpoint_path: str = "res/checkpoints/best_model_graphens.pt", monitor: str = 'val_ap',
    ratio=None, loss="ce", loss_kwargs=None, clip_norm=1.0,
    graphens_warmup: int = 5, graphens_mask_k: float = 5.0, graphens_pred_temp: float = 1.0,
):
    """GraphENS (Park, Song & Yang, ICLR 2022), Algorithm 1.

    Unlike GraphSMOTE/GATSMOTE/TNU
    (GNN_features_graphsmote_with_predictions), which sample a static
    augmented graph once before the epoch loop, GraphENS resamples and
    remixes its synthetic ego-network nodes EVERY epoch, using confidence
    (o_hat) and saliency (S) state carried over from the previous epoch.
    That cross-epoch state is closured local state inside this function
    only. This is deliberately a separate function rather than another
    branch of GNN_features_graphsmote_with_predictions, so the four
    techniques that ARE legitimately one-shot stay physically unaffected by
    this change.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, checkpoint_path=checkpoint_path, monitor=monitor
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    def _build_weighted_criterion(y_subset):
        return _build_loss_criterion(y_subset, loss_name=loss, loss_kwargs=loss_kwargs or {}, device=device)

    def _forward(x, edge_index):
        if use_intrinsic:
            return model(x, edge_index)
        ones = torch.ones((x.shape[0], 1), dtype=torch.float32, device=device)
        return model(ones, edge_index)

    real_edge_index = ntw_torch.edge_index.long()
    real_edge_index_dev = real_edge_index.to(device)
    n_real = int(ntw_torch.x.shape[0])
    y_real_cpu = ntw_torch.y.long().cpu()
    train_mask_bool = train_mask.bool()

    # Class-conditional pools (Algorithm 1 item 1, binary reduction -- see
    # graphens.sample_minor_target_pairs docstring): the minority class is
    # oversampled by drawing v_minor from minor_pool and v_target from
    # target_pool (majority class), both restricted to the train split.
    train_labels = y_real_cpu[train_mask_bool.cpu()]
    unique_classes, class_counts = torch.unique(train_labels, return_counts=True)
    minority_class = None
    n_synthetic_total = 0
    if unique_classes.numel() >= 2:
        minority_class = unique_classes[torch.argmin(class_counts)].item()
        majority_class = unique_classes[torch.argmax(class_counts)].item()
        minority_count = int(class_counts.min().item())
        majority_count = int(class_counts.max().item())
        target_minority_count = int(round(majority_count / ratio)) if ratio is not None else majority_count
        n_synthetic_total = max(0, target_minority_count - minority_count)

    minor_pool = target_pool = None
    if n_synthetic_total > 0:
        minor_pool = torch.where(train_mask_bool.cpu() & (y_real_cpu == minority_class))[0]
        target_pool = torch.where(train_mask_bool.cpu() & (y_real_cpu == majority_class))[0]
        if minor_pool.numel() == 0 or target_pool.numel() == 0:
            n_synthetic_total = 0

    # Real-graph adjacency, built once -- real edges are static across
    # epochs; only the synthetic nodes/edges below are per-epoch.
    adjacency = graphens.build_adjacency_list(real_edge_index, n_real)
    degree_tensor = torch.tensor([adjacency[i].numel() for i in range(n_real)], dtype=torch.long)

    def _duplicate_minor_neighbor_edges(minor_idx_batch):
        # Warmup path (item 12): synthetic node v_mixed inherits v_minor's
        # real neighbors directly, no blending.
        src_parts, dst_parts = [], []
        for i, m in enumerate(minor_idx_batch.tolist()):
            neigh = adjacency[m]
            if neigh.numel() == 0:
                continue
            src_parts.append(neigh)
            dst_parts.append(torch.full((neigh.numel(),), n_real + i, dtype=torch.long))
        if not src_parts:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0)

    def _blended_neighbor_edges(minor_idx_batch, target_idx_batch, phi_hat):
        # Full path (items 8-9): Eq. 1 blended adjacency; neighbor count is
        # r ~ p_degree(D), the graph's own degree distribution, capped at
        # deg(v_minor) -- NOT simply deg(v_minor) directly (paper Section
        # 4.1 + reference gens.py:204-208). Directional: sampled real
        # neighbor -> synthetic node only (never symmetrized -- see
        # edge_index_epoch assembly below).
        minor_degrees = degree_tensor[minor_idx_batch]
        aug_degrees = graphens.sample_augmented_degree(degree_tensor, cap=minor_degrees)
        src_parts, dst_parts = [], []
        for i in range(minor_idx_batch.shape[0]):
            m = int(minor_idx_batch[i].item())
            t = int(target_idx_batch[i].item())
            minor_neighbors = adjacency[m]
            target_neighbors = adjacency[t]
            num_samples = int(aug_degrees[i].item())
            sampled = graphens.blended_neighbor_sampling(
                minor_neighbors, target_neighbors, float(phi_hat[i].item()), num_samples
            )
            if sampled.numel() == 0:
                continue
            src_parts.append(sampled)
            dst_parts.append(torch.full((sampled.numel(),), n_real + i, dtype=torch.long))
        if not src_parts:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0)

    # Cross-epoch state (Algorithm 1 items 3, 10, 11): o_hat^(t-1) and
    # S^(t-1), both computed at the end of the previous train_epoch() call
    # and consumed at the start of the next. None until the first epoch has
    # run once, which forces the warmup/simple path regardless of
    # --graphens-warmup (there is no "previous epoch" before epoch 0).
    state = {"prev_confidence": None, "prev_saliency": None, "last_path": None}

    def train_epoch(epoch_idx):
        model.train()
        optimizer.zero_grad()

        real_x = ntw_torch.x.to(device).detach().clone().requires_grad_(True)

        if n_synthetic_total > 0:
            minor_idx, target_idx = graphens.sample_minor_target_pairs(minor_pool, [target_pool], n_synthetic_total)
            minor_features = real_x[minor_idx.to(device)]
            target_features = real_x[target_idx.to(device)]

            use_warmup_path = (
                epoch_idx < graphens_warmup
                or state["prev_confidence"] is None
                or state["prev_saliency"] is None
            )
            if use_warmup_path:
                mixed_features, _lam = graphens.warmup_mixup(minor_features, target_features)
                new_edges = _duplicate_minor_neighbor_edges(minor_idx)
            else:
                minor_confidence = state["prev_confidence"][minor_idx]
                target_confidence = state["prev_confidence"][target_idx]
                phi_hat = graphens.compute_mixing_ratio(minor_confidence, target_confidence)
                target_saliency = state["prev_saliency"][target_idx]
                mixed_features, _lam = graphens.saliency_masked_mixup(
                    minor_features, target_features, target_saliency, phi_hat, graphens_mask_k
                )
                new_edges = _blended_neighbor_edges(minor_idx, target_idx, phi_hat.detach().cpu())

            x_epoch = torch.cat([real_x, mixed_features], dim=0)
            edge_index_epoch = torch.cat([real_edge_index_dev, new_edges.to(device)], dim=1)
            y_epoch = torch.cat([
                ntw_torch.y.long().to(device),
                torch.full((n_synthetic_total,), minority_class, dtype=torch.long, device=device)
            ])
            train_mask_epoch = torch.cat([
                train_mask_bool.to(device),
                torch.ones(n_synthetic_total, dtype=torch.bool, device=device)
            ])
            path_used = "warmup" if use_warmup_path else "full"
        else:
            x_epoch = real_x
            edge_index_epoch = real_edge_index_dev
            y_epoch = ntw_torch.y.long().to(device)
            train_mask_epoch = train_mask_bool.to(device)
            path_used = "no_augmentation"

        out, _ = _forward(x_epoch, edge_index_epoch)
        y_train = y_epoch[train_mask_epoch]
        criterion = _build_weighted_criterion(y_train)
        loss_val = _compute_loss(criterion, out[train_mask_epoch], y_train)
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        # Reuse this epoch's own backward pass for saliency (item 11): no
        # second backward. real_x.grad already reflects both nodes' own
        # loss AND any loss routed through synthetic nodes derived from
        # them (autograd chains mixed_features back through real_x).
        # Paper Section 4.2 defines saliency as the MAGNITUDE of the
        # gradient, s_(v,i) = |[dL/dX]_(v,i)| -- abs() here, not the raw
        # signed gradient (confirmed against the paper PDF; the reference
        # code's saliency_dst.abs() agrees).
        this_epoch_saliency = real_x.grad.detach().abs().clone()
        with torch.no_grad():
            this_epoch_confidence = graphens.aggregate_confidence(
                out[:n_real].detach(), real_edge_index_dev, temperature=graphens_pred_temp
            )
        state["prev_saliency"] = this_epoch_saliency
        state["prev_confidence"] = this_epoch_confidence
        state["last_path"] = path_used

        return loss_val.item()

    def evaluate_split(mask):
        # GraphENS only augments the TRAINING graph; val/test nodes were
        # never eligible as v_minor/v_target sources, so evaluation always
        # runs on the plain real graph, no padding needed.
        model.eval()
        with torch.no_grad():
            out, _ = _forward(ntw_torch.x.to(device), real_edge_index_dev)
            y = ntw_torch.y.long().to(device)
            mask_dev = mask.bool().to(device)
            out_filtered = out[:mask_dev.shape[0]][mask_dev]
            y_filtered = y[:mask_dev.shape[0]][mask_dev]
            if out_filtered.shape[0] == 0:
                return None
            criterion = _build_weighted_criterion(y_filtered)
            loss_val = _compute_loss(criterion, out_filtered, y_filtered).item()
            y_hat = out_filtered.softmax(dim=1)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
            ap_score = average_precision_score(y_filtered.cpu().numpy(), y_hat.cpu().numpy()[:, 1])
            return {'loss': loss_val, 'ap': ap_score, 'output': y_hat, 'y': y_filtered}

    for epoch in range(n_epochs):
        train_loss = train_epoch(epoch)
        if val_mask is not None:
            val_result = evaluate_split(val_mask)
            if val_result is not None:
                print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | val_loss={val_result['loss']:.6f} | val_ap={val_result['ap']:.6f} | graphens_path={state['last_path']}")
                metric_to_monitor = val_result['ap'] if monitor == 'val_ap' else val_result['loss']
                early_stopping(metric_to_monitor, model)
        if early_stopping.early_stop:
            print(f"[graphens] Early Stop triggered!")
            break

    if val_mask is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_result = evaluate_split(test_mask)
    return test_result['ap'], test_result['output'].cpu().numpy()[:, 1], test_result['y'].cpu().numpy()
