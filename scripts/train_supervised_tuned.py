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
os.chdir(DIR + "/../")
sys.path.append(DIR + "/../")
SRC_PATH = os.path.abspath(os.path.join(DIR, '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# 從我們自定義的 tuned 模組導入具有梯度剪裁與防退化重加權的實驗方法
from src.methods.experiments_supervised_tuned import (
    intrinsic_features,
    positional_features,
    intrinsic_features_with_predictions,
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
from src.methods.evaluation import random_undersample_mask, adjust_mask_to_ratio

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
    parser = argparse.ArgumentParser(description='Run tuned supervised training with custom LR and continuous gradient clipping.')
    parser.add_argument('--mode', choices=['auc', 'f1'], default='auc', help='Choose to run AUC or F1 evaluation')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for this run')
    parser.add_argument(
        '--network', 
        choices=['elliptic', 'hi_small', 'hi_medium', 'hi_large', 'li_small', 'li_medium', 'li_large'], 
        default='hi_small', 
        help='Dataset/configuration to run'
    )
    # ===== 新增超參數調優參數 =====
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for GNNs and decoders (default: 0.001 to prevent gradient explosion)')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Maximum gradient norm for clipping (default: 1.0)')
    parser.add_argument('--out_dir', type=str, default='res/tuned', help='Directory to save results (default: res/tuned)')

    args = parser.parse_args()
    set_seed(args.seed)

    # 確保輸出目錄與 Checkpoints 子目錄存在
    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    ntw_name = args.network
    test_ratios = [None, 10.0, 2.0, 1.0]
    ratio_names = {None: "original", 10.0: "ratio_1to10", 2.0: "ratio_1to2", 1.0: "ratio_1to1"}

    if ntw_name in {"hi_small", "hi_medium", "hi_large", "li_small", "li_medium", "li_large"}:
        ntw = load_ibm_config(ntw_name)
    elif ntw_name == "elliptic":
        ntw = load_elliptic()
    else:
        raise ValueError("Network not found")

    train_mask, val_mask, test_mask = ntw.get_masks()
    to_train = ["intrinsic", "positional", "deepwalk", "node2vec", "gcn", "sage", "gat", "gin"]
    
    method_sampling_techniques = {
        "intrinsic": ["none", "random_undersample", "smote"],
        "positional": ["none", "random_undersample", "smote"],
        "deepwalk": ["none", "random_undersample", "graph_smote", "graph_ensemble_smote", "reweighted_graph_smote"],
        "node2vec": ["none", "random_undersample", "graph_smote", "graph_ensemble_smote", "reweighted_graph_smote"],
        "gcn": ["none", "random_undersample", "graph_smote", "graph_ensemble_smote", "reweighted_graph_smote"],
        "sage": ["none", "random_undersample", "graph_smote", "graph_ensemble_smote", "reweighted_graph_smote"],
        "gat": ["none", "random_undersample", "graph_smote", "graph_ensemble_smote", "reweighted_graph_smote"],
        "gin": ["none", "random_undersample", "graph_smote", "graph_ensemble_smote", "reweighted_graph_smote"]
    }

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

    for ratio in test_ratios:
        ratio_tag = ratio_names[ratio]
        print("\n" + "="*80)
        print(f"PHASE: Testing Class Imbalance Ratio: {ratio_tag.upper()} with LR={args.lr}, CLIP={args.clip_norm}")
        print("="*80)

        if ratio is None:
            train_mask_ratio = train_mask.clone()
        else:
            train_mask_ratio, _, _, _ = adjust_mask_to_ratio(
                train_mask.clone(), ntw_torch.y.cpu(), target_ratio=ratio, random_state=42
            )
        train_mask_ratio = train_mask_ratio.to(train_mask.device)

        for method in to_train:
            print(f"\n Training {method.upper()}...")
            sampling_techniques_for_method = method_sampling_techniques.get(method, ["none"])
            for sampling in sampling_techniques_for_method:
                print(f" - Sampling: {sampling.upper()}", end=" ... ")
                samp_tag = '' if sampling == 'none' else f'_rus' if sampling == 'random_undersample' else f'_{sampling}'
                seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
                result_tag = f"{ntw_name}_{ratio_tag}{samp_tag}{seed_tag}"

                unique_checkpoint_path = f"{checkpoint_dir}/best_model_{method}_{result_tag}.pt"

                if args.mode == 'auc':
                    result_files = [f"{args.out_dir}/{method}_params_{result_tag}.txt"]
                    check_contents = ["AUC-PRC"]
                else:
                    result_files = [f"{args.out_dir}/{method}_f1_99_params_{result_tag}.txt"]
                    check_contents = ["F1_99:"]

                # ----- 防重複計算檢查 -----
                all_exist = True
                for rf, cc in zip(result_files, check_contents):
                    if not os.path.exists(rf):
                        all_exist = False
                        break
                    try:
                        with open(rf, 'r') as f:
                            existing_content = f.read().strip()
                            if not existing_content or cc not in existing_content:
                                all_exist = False
                                break
                    except Exception:
                        all_exist = False
                        break

                if all_exist:
                    print("SKIPPED (all results already exist)")
                    continue

                if sampling == "none":
                    train_mask_sampled = train_mask_ratio
                else:
                    train_mask_sampled = random_undersample_mask(
                        train_mask_ratio.contiguous().view(-1), ntw_torch.y.cpu(), target_ratio=1.0
                    )
                train_mask_sampled = train_mask_sampled.to(train_mask_ratio.device)

                try:
                    results = []
                    # ========== F1 MODE ==========
                    if args.mode == 'f1':
                        if method == "intrinsic":
                            ap_score, y_pred_probs, y_true = intrinsic_features_with_predictions(
                                ntw, train_mask_sampled, test_mask, n_layers_decoder=2, hidden_dim_decoder=16, 
                                lr=args.lr, n_epochs_decoder=100, clip_norm=args.clip_norm
                            )
                        elif method == "positional":
                            ap_score, y_pred_probs, y_true = positional_features_with_predictions(
                                ntw, train_mask_sampled, test_mask, alpha_pr=0.5, alpha_ppr=0, n_epochs_decoder=50, 
                                lr=args.lr, fraud_dict_test=fraud_dict, n_layers_decoder=2, hidden_dim_decoder=16, 
                                ntw_name=ntw_name+"_train", clip_norm=args.clip_norm
                            )
                        elif method == "deepwalk":
                            ap_score, y_pred_probs, y_true = node2vec_features_with_predictions(
                                ntw_torch, train_mask_sampled, test_mask, embedding_dim=16, walk_length=3, context_size=2, 
                                walks_per_node=1, num_negative_samples=2, p=1, q=1, lr=args.lr, n_epochs=30, n_epochs_decoder=30, 
                                use_torch=True, clip_norm=args.clip_norm
                            )
                        elif method == "node2vec":
                            ap_score, y_pred_probs, y_true = node2vec_features_with_predictions(
                                ntw_torch, train_mask_sampled, test_mask, embedding_dim=16, walk_length=3, context_size=2, 
                                walks_per_node=1, num_negative_samples=2, p=1.5, q=1.0, lr=args.lr, n_epochs=20, n_epochs_decoder=20, 
                                ntw_nx=ntw.get_network_nx(), use_torch=True, clip_norm=args.clip_norm
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
                                    ntw_torch, model, lr=args.lr, n_epochs=gnn_epochs, train_mask=train_mask_sampled, 
                                    val_mask=val_mask, test_mask=test_mask, patience=10, checkpoint_path=unique_checkpoint_path, 
                                    monitor='val_ap', clip_norm=args.clip_norm
                                )
                            else:
                                ap_score, y_pred_probs, y_true = GNN_features_graphsmote_with_predictions(
                                    ntw_torch, model, lr=args.lr, n_epochs=gnn_epochs, train_mask=train_mask_sampled, 
                                    val_mask=val_mask, test_mask=test_mask, k_neighbors=5, random_state=42, sampling=sampling, 
                                    patience=10, checkpoint_path=unique_checkpoint_path, monitor='val_ap', clip_norm=args.clip_norm
                                )
                        
                        cutoff = np.percentile(y_pred_probs, 99)
                        y_pred_hard = (y_pred_probs >= cutoff).astype(int)
                        f1_loss = f1_score(y_true, y_pred_hard)
                        results.append(f"AUC-PRC: {ap_score}, F1_99: {f1_loss}")

                    # ========== AUC MODE ==========
                    else:
                        if method == "intrinsic":
                            ap_loss, f1_loss = intrinsic_features(
                                ntw, train_mask_sampled, test_mask, n_layers_decoder=2, hidden_dim_decoder=16, 
                                lr=args.lr, n_epochs_decoder=100, percentile_q=99, clip_norm=args.clip_norm
                            )
                        elif method == "positional":
                            ap_loss, f1_loss = positional_features(
                                ntw, train_mask_sampled, test_mask, alpha_pr=0.5, alpha_ppr=0, n_epochs_decoder=50, 
                                lr=args.lr, fraud_dict_test=fraud_dict, n_layers_decoder=2, hidden_dim_decoder=16, 
                                ntw_name=ntw_name+"_train", percentile_q=99, clip_norm=args.clip_norm
                            )
                        elif method == "deepwalk":
                            ap_loss, f1_loss = node2vec_features(
                                ntw_torch, train_mask_sampled, test_mask, embedding_dim=16, walk_length=3, context_size=2, 
                                walks_per_node=1, num_negative_samples=2, p=1, q=1, lr=args.lr, n_epochs=30, n_epochs_decoder=30, 
                                use_torch=True, percentile_q=99, clip_norm=args.clip_norm
                            )
                        elif method == "node2vec":
                            ap_loss, f1_loss = node2vec_features(
                                ntw_torch, train_mask_sampled, test_mask, embedding_dim=16, walk_length=3, context_size=2, 
                                walks_per_node=1, num_negative_samples=2, p=1.5, q=1.0, lr=args.lr, n_epochs=20, n_epochs_decoder=20, 
                                ntw_nx=ntw.get_network_nx(), use_torch=True, percentile_q=99, clip_norm=args.clip_norm
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
                                ap_loss, f1_loss = GNN_features(
                                    ntw_torch, model, lr=args.lr, n_epochs=gnn_epochs, train_mask=train_mask_sampled, 
                                    val_mask=val_mask, test_mask=test_mask, percentile_q=99, patience=10, 
                                    checkpoint_path=unique_checkpoint_path, monitor='val_ap', clip_norm=args.clip_norm
                                )
                            else:
                                ap_loss, f1_loss = GNN_features_graphsmote(
                                    ntw_torch, model, lr=args.lr, n_epochs=gnn_epochs, train_mask=train_mask_sampled, 
                                    val_mask=val_mask, test_mask=test_mask, k_neighbors=5, random_state=42, percentile_q=99, 
                                    sampling=sampling, patience=10, checkpoint_path=unique_checkpoint_path, monitor='val_ap', 
                                    clip_norm=args.clip_norm
                                )
                        results.append(f"AUC-PRC: {ap_loss}, F1: {f1_loss}")

                    for rf, res in zip(result_files, results):
                        with open(Path(rf), "w") as f:
                            f.write(res)
                    print(f"Done! {res} -> {rf}")

                except Exception as e:
                    print(f"Error! method={method} sampling={sampling} ratio={ratio_tag} detail={str(e)}")
                    print(traceback.format_exc())

    print("\n" + "="*80)
    print("All tuned training phases completed!")
    print("="*80)
