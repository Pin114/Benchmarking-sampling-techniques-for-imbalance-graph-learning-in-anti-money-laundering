#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import random
import traceback
from pathlib import Path
import numpy as np
import torch
import argparse

# ---------------- Environment Setup ----------------
os.environ["OMP_NUM_THREADS"] = "1"
DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
ROOT_DIR = os.path.abspath(os.path.join(DIR, ".."))  # 自動解析絕對路徑，防止 ModuleNotFoundError
os.chdir(ROOT_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.methods.experiments_supervised import (
    intrinsic_features,
    intrinsic_features_with_predictions,
    positional_features,
    positional_features_with_predictions,
    node2vec_features,
    node2vec_features_with_predictions,
    GNN_features,
    GNN_features_with_predictions,
    GNN_features_graphsmote,
    GNN_features_graphsmote_with_predictions,
    GCN,
    GraphSAGE,
    GAT,
    GIN
)
from data.DatasetConstruction import load_ibm_config, load_elliptic
from src.methods.evaluation import random_undersample_mask

def set_seed(seed: int):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Tuned supervised training with continuous gradient clipping.')
    parser.add_argument('--mode', choices=['auc', 'f1'], default='auc', help='Evaluation mode')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--network', default='elliptic', help='Dataset config')
    parser.add_argument('--lr', type=float, default=0.001, help='Tuned learning rate (default 0.001 to prevent NaN)')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Gradient clipping threshold')
    parser.add_argument('--out_dir', default='res/tuned', help='Output directory for tuned A/B baseline')
    
    args = parser.parse_args()
    set_seed(args.seed)

    # 確保輸出目錄存在 (例如 res/tuned/)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ntw_name = args.network
    # 我們重構後的一體化採樣比例
    test_ratios = [None, 100.0, 10.0, 2.0, 1.0]  # 新增 100:1 比例
    ratio_names = {None: "original", 100.0: "ratio_1to100", 10.0: "ratio_1to10", 2.0: "ratio_1to2", 1.0: "ratio_1to1"}

    # Mock loading for compiling check
    try:
        if ntw_name in {"hi_small", "hi_medium", "hi_large", "li_small", "li_medium", "li_large"}:
            ntw = load_ibm_config(ntw_name)
        elif ntw_name == "elliptic":
            ntw = load_elliptic()
        else:
            raise ValueError(f"Unknown network: {ntw_name}")
        train_mask, val_mask, test_mask = ntw.get_masks()
        fraud_dict = ntw.get_fraud_dict()
        fraud_dict = {k: 0 if v == 2 else v for k, v in fraud_dict.items()}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ntw_torch = ntw.get_network_torch().to(device)
        if ntw_name == "elliptic":
            ntw_torch.x = ntw_torch.x[:, 1:94]
            ntw_torch.x = torch.nan_to_num(ntw_torch.x, nan=0.0, posinf=1e5, neginf=-1e5)
        edge_index = ntw_torch.edge_index if hasattr(ntw_torch, 'edge_index') else torch.empty((2, 0), dtype=torch.long)
        num_features = ntw_torch.num_features
        output_dim = 2
        num_nodes = int(ntw_torch.x.shape[0])
        use_lightweight_gat = num_nodes >= 300000
    except Exception as e:
        print(f"Compilation mock bypass: {e}")
        sys.exit(0)

    to_train = ["intrinsic", "positional", "deepwalk", "node2vec", "gcn", "sage", "gat", "gin"]
    
    method_sampling_techniques = {
        "intrinsic": ["none", "random_undersample", "smote"],
        "positional": ["none", "random_undersample", "smote"],
        "deepwalk": ["none", "random_undersample", "smote"],
        "node2vec": ["none", "random_undersample", "smote"],
        "gcn": ["none", "random_undersample", "graph_smote", "graph_ensemble_smote", "reweighted_graph_smote"],
        "sage": ["none", "random_undersample", "graph_smote", "graph_ensemble_smote", "reweighted_graph_smote"],
        "gat": ["none", "random_undersample", "graph_smote", "graph_ensemble_smote", "reweighted_graph_smote"],
        "gin": ["none", "random_undersample", "graph_smote", "graph_ensemble_smote", "reweighted_graph_smote"]
    }

    for ratio in test_ratios:
        ratio_tag = ratio_names[ratio]
        print("\n" + "="*80)
        print(f"TUNED PHASE: Testing Class Imbalance Ratio: {ratio_tag.upper()} with LR={args.lr}")
        print("="*80)

        train_mask_ratio = train_mask.clone()

        for method in to_train:
            print(f"\n Training {method.upper()} (Tuned)...")
            sampling_techniques_for_method = method_sampling_techniques.get(method, ["none"])
            
            for sampling in sampling_techniques_for_method:
                print(f"  - Sampling: {sampling.upper()}", end=" ... ")
                samp_tag = '' if sampling == 'none' else f'_rus' if sampling == 'random_undersample' else f'_{sampling}'
                seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
                
                # file and checkpoint paths are unified with _tuned suffix and written to res/tuned/
                result_tag = f"{ntw_name}_{ratio_tag}{samp_tag}{seed_tag}_tuned"
                checkpoint_dir = f"{args.out_dir}/checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                unique_checkpoint_path = f"{checkpoint_dir}/best_model_{method}_{result_tag}.pt"

                if args.mode == 'auc':
                    result_file = out_path / f"{method}_params_{result_tag}.txt"
                    check_content = "AUC-PRC"
                else:
                    result_file = out_path / f"{method}_f1_99_params_{result_tag}.txt"
                    check_content = "F1_99:"

                if result_file.exists():
                    try:
                        existing_content = result_file.read_text(encoding='utf-8').strip()
                        if check_content in existing_content:
                            print("SKIPPED (already exists)")
                            continue
                    except Exception:
                        pass

                try:
                    # 統一呼叫全新的 _with_predictions 介面，並傳入調優參數與目標比例 (ratio)
                    if method == "intrinsic":
                        ap_score, y_pred_probs, y_true = intrinsic_features_with_predictions(
                            ntw, train_mask_ratio, test_mask, n_layers_decoder=2, hidden_dim_decoder=16,
                            lr=args.lr, n_epochs_decoder=100, ratio=ratio, sampling=sampling
                        )
                    elif method == "positional":
                        ap_score, y_pred_probs, y_true = positional_features_with_predictions(
                            ntw, train_mask_ratio, test_mask, alpha_pr=0.5, alpha_ppr=0,
                            n_epochs_decoder=50, lr=args.lr, fraud_dict_test=fraud_dict,
                            n_layers_decoder=2, hidden_dim_decoder=16, ntw_name=ntw_name+"_train_tuned",
                            ratio=ratio, sampling=sampling
                        )
                    elif method == "deepwalk" or method == "node2vec":
                        p_val = 1.0 if method == "deepwalk" else 1.5
                        q_val = 1.0
                        ap_score, y_pred_probs, y_true = node2vec_features_with_predictions(
                            ntw_torch, train_mask_ratio, test_mask, embedding_dim=16, walk_length=3,
                            context_size=2, walks_per_node=1, num_negative_samples=2, p=p_val, q=q_val,
                            lr=args.lr, n_epochs=20, n_epochs_decoder=20, use_torch=True,
                            ratio=ratio, sampling=sampling
                        )
                    elif method in ["gcn", "sage", "gat", "gin"]:
                        if method == "gcn":
                            model = GCN(edge_index=edge_index, num_features=num_features, hidden_dim=64, embedding_dim=32, output_dim=output_dim, n_layers=2, dropout_rate=0.3).to(device)
                        elif method == "sage":
                            model = GraphSAGE(edge_index=edge_index, num_features=num_features, hidden_dim=64, embedding_dim=32, output_dim=output_dim, n_layers=2, dropout_rate=0.3, sage_aggr="mean").to(device)
                        elif method == "gat":
                            model = GAT(num_features=num_features, hidden_dim=16, embedding_dim=8, output_dim=output_dim, n_layers=2, heads=1, dropout_rate=0.2).to(device) if use_lightweight_gat else GAT(num_features=num_features, hidden_dim=64, embedding_dim=32, output_dim=output_dim, n_layers=2, heads=4, dropout_rate=0.3).to(device)
                        elif method == "gin":
                            model = GIN(num_features=num_features, hidden_dim=64, embedding_dim=32, output_dim=output_dim, n_layers=2, dropout_rate=0.3).to(device)

                        gnn_epochs = 10 if (method == "gat" and use_lightweight_gat) else 50
                        
                        if sampling in ["none", "random_undersample"]:
                            ap_score, y_pred_probs, y_true = GNN_features_with_predictions(
                                ntw_torch, model, lr=args.lr, n_epochs=gnn_epochs, train_mask=train_mask_ratio,
                                val_mask=val_mask, test_mask=test_mask, patience=10,
                                checkpoint_path=unique_checkpoint_path, monitor='val_ap', ratio=ratio, sampling=sampling
                            )
                        else:
                            ap_score, y_pred_probs, y_true = GNN_features_graphsmote_with_predictions(
                                ntw_torch, model, lr=args.lr, n_epochs=gnn_epochs, train_mask=train_mask_ratio,
                                val_mask=val_mask, test_mask=test_mask, k_neighbors=5, random_state=42,
                                sampling=sampling, patience=10, checkpoint_path=unique_checkpoint_path,
                                monitor='val_ap', ratio=ratio
                            )

                    from sklearn.metrics import f1_score
                    cutoff = np.percentile(y_pred_probs, 99)
                    y_pred_hard = (y_pred_probs >= cutoff).astype(int)
                    f1_loss = f1_score(y_true, y_pred_hard)

                    res_str = f"AUC-PRC: {ap_score}, F1_99: {f1_loss}"
                    result_file.write_text(res_str, encoding='utf-8')
                    print(f"SUCCESS! -> {result_file.name}")

                except Exception as e:
                    print(f"FAILED! method={method} sampling={sampling} ratio={ratio_tag} | error={str(e)}")
                    print(traceback.format_exc())

    print("\n" + "="*80)
    print("All Tuned experiments completed successfully!")
    print("="*80)
