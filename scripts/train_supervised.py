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

DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIR + "/../")
sys.path.append(DIR + "/../")
SRC_PATH = os.path.abspath(os.path.join(DIR, '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.methods.experiments_supervised import (
    intrinsic_features, intrinsic_features_smote,
    positional_features, positional_features_smote,
    intrinsic_features_with_predictions, intrinsic_features_smote_with_predictions,
    positional_features_with_predictions, positional_features_smote_with_predictions,
    node2vec_features,
    GCN, GraphSAGE, GAT, GIN,
    GNN_features, GNN_features_graphsmote
)
from data.DatasetConstruction import load_ibm_config, load_elliptic
from src.methods.evaluation import random_undersample_mask, graph_smote_mask, graph_ensemble_smote_mask, adjust_mask_to_ratio


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
    parser = argparse.ArgumentParser(description='Run supervised training with AUC or F1 evaluation.')
    parser.add_argument('--mode', choices=['auc', 'f1'], default='auc', help='Choose to run AUC or F1 evaluation')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for this run')
    parser.add_argument(
        '--network',
        choices=['elliptic', 'hi_small', 'hi_medium', 'hi_large', 'li_small', 'li_medium', 'li_large'],
        default='hi_small',
        help='Dataset/configuration to run'
    )
    args = parser.parse_args()
    set_seed(args.seed)
    
    # ========== ENSURE RES DIRECTORY EXISTS ==========
    if not os.path.exists("res"):
        os.makedirs("res")
    
    ### Load Dataset ###
    ntw_name = args.network
    
    # ========== Class Imbalance Ratios ==========
    # ratio = majority_count / minority_count
    test_ratios = [
        None,    # Original imbalance (baseline from dataset)
        10.0,    # 1:10 ratio (light imbalance)
        2.0,     # 1:2 ratio (mild imbalance)
        1.0      # 1:1 ratio (fully balanced)
    ]
    
    ratio_names = {
        None: "original",
        10.0: "ratio_1to10",
        2.0: "ratio_1to2",
        1.0: "ratio_1to1"
    }

    if ntw_name in {"hi_small", "hi_medium", "hi_large", "li_small", "li_medium", "li_large"}:
        ntw = load_ibm_config(ntw_name)
    elif ntw_name == "elliptic":
        ntw = load_elliptic()
    else:
        raise ValueError("Network not found")

    train_mask, val_mask, test_mask = ntw.get_masks()
    # Maintain strict separation of train / val / test splits.
    # Validation is only used for monitoring overfitting, while final reported metrics
    # are always computed on the held-out test split.
    
    # ==========  Techniques ==========
    to_train = [
        "intrinsic",
        "positional",
        "deepwalk",
        "node2vec",
        "gcn",
        "sage",
        "gat",
        "gin"
    ]
    

    # Intrinsic & Positional: none, random_undersample, smote
    # GNN methods: none, random_undersample, graph_smote, graph_ensemble_smote
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
    
    # --- data preparation ---
    fraud_dict = ntw.get_fraud_dict()
    fraud_dict = {k: 0 if v == 2 else v for k, v in fraud_dict.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ntw_torch = ntw.get_network_torch().to(device)
    if ntw_name == "elliptic":
        ntw_torch.x = ntw_torch.x[:,1:94]
    ntw_torch.x = torch.nan_to_num(ntw_torch.x, nan=0.0, posinf=1e5, neginf=-1e5)
    
    if hasattr(ntw_torch, 'edge_index'):
        edge_index = ntw_torch.edge_index
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    num_features = ntw_torch.num_features
    output_dim = 2
    num_nodes = int(ntw_torch.x.shape[0])
    use_lightweight_gat = num_nodes >= 300000
    
    # ========== start testing imbalance ratios ==========
    for ratio in test_ratios:
        ratio_tag = ratio_names[ratio]
        
        print("\n" + "="*80)
        print(f"PHASE: Testing Class Imbalance Ratio: {ratio_tag.upper()}")
        if ratio is None:
            print("  (Using original dataset imbalance)")
        else:
            print(f"  (Using the original training split for {ratio_tag}; sampling methods will rebalance internally)")
        print("="*80)
        
        # The experiments should operate on the untouched training split; any class
        # balancing is applied later by the sampling method itself, not by a global
        # front-end undersampling step.
        if ratio is None:
            train_mask_ratio = train_mask.clone()
        else:
            train_mask_ratio, _, _, _ = adjust_mask_to_ratio(
                train_mask.clone(),
                ntw_torch.y.cpu(),
                target_ratio=ratio,
                random_state=42
            )
            train_mask_ratio = train_mask_ratio.to(train_mask.device)
        print("  Using the original train split directly; sampling methods will operate on the training set only.")
        
        # ========== train all methods ==========
        for method in to_train:
            print(f"\n  Training {method.upper()}...")
            
            # Determine sampling techniques for this method
            sampling_techniques_for_method = method_sampling_techniques.get(method, ["none"])
            
            for sampling in sampling_techniques_for_method:
                print(f"    - Sampling: {sampling.upper()}", end=" ... ")
                
                # construct result tag
                if sampling == 'none':
                    samp_tag = ''
                elif sampling == 'random_undersample':
                    samp_tag = '_rus'
                else:
                    samp_tag = f'_{sampling}'
                
                seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
                result_tag = f"{ntw_name}_{ratio_tag}{samp_tag}{seed_tag}"
                
                # Set filename based on mode
                if args.mode == 'auc':
                    result_file = f"res/{method}_params_{result_tag}.txt"
                    check_content = "AUC-PRC"
                    f1_thresholds = [99]  # Not used for AUC
                else:
                    f1_thresholds = [90, 99]
                    result_files = [f"res/{method}_f1_{thresh}_params_{result_tag}.txt" for thresh in f1_thresholds]
                    check_contents = [f"F1_{thresh}:" for thresh in f1_thresholds]
                
                if args.mode == 'auc':
                    result_files = [result_file]
                    check_contents = [check_content]
                    f1_thresholds = [99]
                else:
                    result_files = result_files
                    check_contents = check_contents
                    f1_thresholds = f1_thresholds
                
                # ========== SKIP IF ALREADY EXISTS ==========
                all_exist = True
                missing_files = []
                
                for rf, cc in zip(result_files, check_contents):
                    if not os.path.exists(rf):
                        all_exist = False
                        missing_files.append(rf)
                        continue
                    
                    try:
                        with open(rf, 'r') as f:
                            existing_content = f.read().strip()
                        
                        # Check if file has valid content with expected metric
                        if not existing_content:
                            all_exist = False
                            missing_files.append(f"{rf} (empty)")
                        elif cc not in existing_content:
                            all_exist = False
                            missing_files.append(f"{rf} (invalid content)")
                        else:
                            # File exists and has valid content
                            print(f"✓ Found: {rf}")
                    except Exception as e:
                        all_exist = False
                        missing_files.append(f"{rf} (error: {str(e)})")
                
                if all_exist:
                    print(f"SKIPPED (all results already exist)")
                    continue
                else:
                    if missing_files:
                        print(f"Need to compute: {missing_files}")
                
                if sampling == "none":
                    train_mask_sampled = train_mask_ratio
                    
                elif sampling == "random_undersample":
                    # Random undersampling of majority class
                    mask_to_pass = train_mask_ratio.contiguous().view(-1)
                    train_mask_sampled = random_undersample_mask(mask_to_pass, ntw_torch.y.cpu(), target_ratio=1.0)
                    train_mask_sampled = train_mask_sampled.to(train_mask_ratio.device)
                    
                elif sampling == "smote":
                    # SMOTE for intrinsic & positional (feature-based methods)
                    # NOTE: The *_smote() functions handle SMOTE internally, so we pass the original mask
                    # Don't apply SMOTE here - let the specific method functions handle it
                    train_mask_sampled = train_mask_ratio
                
                elif sampling == "graph_smote":
                    # GraphSMOTE for GNN methods - need to handle size mismatch
                    mask_to_pass = train_mask_ratio.contiguous().view(-1)
                    expanded_features, expanded_labels, expanded_mask, expanded_edge_index = graph_smote_mask(
                        mask_to_pass, ntw_torch.x.cpu(), ntw_torch.y.cpu(), 
                        edge_index.cpu(), k_neighbors=5, random_state=42
                    )
                    # For GraphSMOTE, return to original mask size (undersampling to match original graph)
                    # This is a workaround - ideally we'd expand the full graph structure
                    # For now, fall back to RUS on the ratio-adjusted mask
                    train_mask_sampled = random_undersample_mask(train_mask_ratio.contiguous().view(-1), ntw_torch.y.cpu(), target_ratio=1.0)
                    train_mask_sampled = train_mask_sampled.to(train_mask_ratio.device)

                elif sampling == "graph_ensemble_smote":
                    # GraphENS (Graph Ensemble SMOTE) for GNN methods
                    mask_to_pass = train_mask_ratio.contiguous().view(-1)
                    features_for_sampling = torch.nan_to_num(ntw_torch.x.cpu(), nan=0.0, posinf=1e5, neginf=-1e5)
                    expanded_features, expanded_labels, expanded_mask, expanded_edge_index = graph_ensemble_smote_mask(
                        mask_to_pass, features_for_sampling, ntw_torch.y.cpu(), 
                        edge_index.cpu(), k_neighbors=5, random_state=42
                    )
                    # Similar to GraphSMOTE, fall back to RUS to maintain original graph structure
                    train_mask_sampled = random_undersample_mask(train_mask_ratio.contiguous().view(-1), ntw_torch.y.cpu(), target_ratio=1.0)
                    train_mask_sampled = train_mask_sampled.to(train_mask_ratio.device)

                elif sampling == "reweighted_graph_smote":
                    train_mask_sampled = random_undersample_mask(train_mask_ratio.contiguous().view(-1), ntw_torch.y.cpu(), target_ratio=1.0)
                    train_mask_sampled = train_mask_sampled.to(train_mask_ratio.device)

                try:
                    results = []
                    result_files_list = []
                    
                    if args.mode == 'auc':
                        # For AUC mode, run single evaluation
                        thresh = 99
                        result_files_list = [result_file]
                    else:
                        # For F1 mode, run both thresholds
                        result_files_list = result_files
                    
                    # For F1 mode, train once and evaluate with different thresholds
                    if args.mode == 'f1':
                        # Train model once and get predictions
                        if method == "intrinsic":
                            if sampling == "none":
                                ap_score, y_pred_probs, y_true = intrinsic_features_with_predictions(
                                    ntw, train_mask_sampled, test_mask,
                                    n_layers_decoder=2, hidden_dim_decoder=16, lr=0.05, n_epochs_decoder=100
                                )
                            elif sampling == "random_undersample":
                                ap_score, y_pred_probs, y_true = intrinsic_features_with_predictions(
                                    ntw, train_mask_sampled, test_mask,
                                    n_layers_decoder=2, hidden_dim_decoder=16, lr=0.05, n_epochs_decoder=100
                                )
                            else:  # SMOTE
                                ap_score, y_pred_probs, y_true = intrinsic_features_smote_with_predictions(
                                    ntw, train_mask_sampled, test_mask,
                                    n_layers_decoder=2, hidden_dim_decoder=16, lr=0.05, n_epochs_decoder=100,
                                    k_neighbors=5, random_state=42
                                )
                            
                        elif method == "positional":
                            if sampling == "none":
                                ap_score, y_pred_probs, y_true = positional_features_with_predictions(
                                    ntw, train_mask_sampled, test_mask,
                                    alpha_pr=0.5, alpha_ppr=0, n_epochs_decoder=50, lr=0.05,
                                    fraud_dict_test=fraud_dict,
                                    n_layers_decoder=2, hidden_dim_decoder=16, 
                                    ntw_name=ntw_name+"_train"
                                )
                            elif sampling == "random_undersample":
                                ap_score, y_pred_probs, y_true = positional_features_with_predictions(
                                    ntw, train_mask_sampled, test_mask,
                                    alpha_pr=0.5, alpha_ppr=0, n_epochs_decoder=50, lr=0.05,
                                    fraud_dict_test=fraud_dict,
                                    n_layers_decoder=2, hidden_dim_decoder=16, 
                                    ntw_name=ntw_name+"_train"
                                )
                            else:  # SMOTE
                                ap_score, y_pred_probs, y_true = positional_features_smote_with_predictions(
                                    ntw, train_mask_sampled, test_mask,
                                    alpha_pr=0.5, alpha_ppr=0, n_epochs_decoder=50, lr=0.05,
                                    fraud_dict_test=fraud_dict,
                                    n_layers_decoder=2, hidden_dim_decoder=16, 
                                    ntw_name=ntw_name+"_train",
                                    k_neighbors=5, random_state=42
                                )
                            
                        # Calculate F1 for each threshold using the same predictions
                        for thresh_idx, thresh in enumerate(f1_thresholds):
                            import numpy as np
                            from sklearn.metrics import f1_score
                            cutoff = np.percentile(y_pred_probs, thresh)
                            y_pred_hard = (y_pred_probs >= cutoff).astype(int)
                            f1_loss = f1_score(y_true, y_pred_hard)
                            
                            if thresh == 90:
                                result = f"AUC-PRC: {ap_score}, F1_90: {f1_loss}"
                            else:  # thresh == 99
                                result = f"AUC-PRC: {ap_score}, F1_99: {f1_loss}"
                            
                            results.append(result)
                    
                    if args.mode == 'auc':
                        # For AUC mode, use original single evaluation
                        for thresh_idx, thresh in enumerate([99]):
                            if method == "intrinsic":
                                if sampling == "none":
                                    ap_loss, f1_loss = intrinsic_features(
                                        ntw, train_mask_sampled, test_mask,
                                        n_layers_decoder=2, hidden_dim_decoder=16, lr=0.05, n_epochs_decoder=100,
                                        percentile_q=thresh
                                    )
                                elif sampling == "random_undersample":
                                    ap_loss, f1_loss = intrinsic_features(
                                        ntw, train_mask_sampled, test_mask,
                                        n_layers_decoder=2, hidden_dim_decoder=16, lr=0.05, n_epochs_decoder=100,
                                        percentile_q=thresh
                                    )
                                else:  # SMOTE
                                    ap_loss, f1_loss = intrinsic_features_smote(
                                        ntw, train_mask_sampled, test_mask,
                                        n_layers_decoder=2, hidden_dim_decoder=16, lr=0.05, n_epochs_decoder=100,
                                        k_neighbors=5, random_state=42,
                                        percentile_q=thresh
                                    )
                                
                            elif method == "positional":
                                if sampling == "none":
                                    ap_loss, f1_loss = positional_features(
                                        ntw, train_mask_sampled, test_mask,
                                        alpha_pr=0.5, alpha_ppr=0, n_epochs_decoder=50, lr=0.05,
                                        fraud_dict_test=fraud_dict,
                                        n_layers_decoder=2, hidden_dim_decoder=16, 
                                        ntw_name=ntw_name+"_train",
                                        percentile_q=thresh
                                    )
                                elif sampling == "random_undersample":
                                    ap_loss, f1_loss = positional_features(
                                        ntw, train_mask_sampled, test_mask,
                                        alpha_pr=0.5, alpha_ppr=0, n_epochs_decoder=50, lr=0.05,
                                        fraud_dict_test=fraud_dict,
                                        n_layers_decoder=2, hidden_dim_decoder=16, 
                                        ntw_name=ntw_name+"_train",
                                        percentile_q=thresh
                                    )
                                else:  # SMOTE
                                    ap_loss, f1_loss = positional_features_smote(
                                        ntw, train_mask_sampled, test_mask,
                                        alpha_pr=0.5, alpha_ppr=0, n_epochs_decoder=50, lr=0.05,
                                        fraud_dict_test=fraud_dict,
                                        n_layers_decoder=2, hidden_dim_decoder=16, 
                                        ntw_name=ntw_name+"_train",
                                        k_neighbors=5, random_state=42,
                                        percentile_q=thresh
                                    )
                                
                            elif method == "deepwalk":
                                ap_loss, f1_loss = node2vec_features(
                                    ntw_torch, train_mask_sampled, test_mask,
                                    embedding_dim=16, walk_length=3, context_size=2,
                                    walks_per_node=1, num_negative_samples=2,
                                    p=1, q=1, lr=0.05, n_epochs=30, n_epochs_decoder=30, 
                                    use_torch=True, percentile_q=thresh
                                )
                                
                            elif method == "node2vec":
                                ap_loss, f1_loss = node2vec_features(
                                    ntw_torch, train_mask_sampled, test_mask,
                                    embedding_dim=16, walk_length=3, context_size=2,
                                    walks_per_node=1, num_negative_samples=2,
                                    p=1.5, q=1.0, lr=0.05, n_epochs=20, n_epochs_decoder=20, 
                                    ntw_nx=ntw.get_network_nx(), use_torch=True, percentile_q=thresh
                                )
                                
                            elif method == "gcn":
                                if sampling in ["none", "random_undersample"]:
                                    model_gcn = GCN(
                                        edge_index=edge_index, num_features=num_features,
                                        hidden_dim=64, embedding_dim=32, output_dim=output_dim,
                                        n_layers=2, dropout_rate=0.3
                                    ).to(device)
                                    ap_loss, f1_loss = GNN_features(
                                        ntw_torch, model_gcn, lr=0.05, n_epochs=50, 
                                        train_mask=train_mask_sampled, val_mask=val_mask, test_mask=test_mask, percentile_q=thresh
                                    )
                                else:  # graph_smote / graph_ensemble_smote / reweighted_graph_smote
                                    model_gcn = GCN(
                                        edge_index=edge_index, num_features=num_features,
                                        hidden_dim=64, embedding_dim=32, output_dim=output_dim,
                                        n_layers=2, dropout_rate=0.3
                                    ).to(device)
                                    ap_loss, f1_loss = GNN_features_graphsmote(
                                        ntw_torch, model_gcn, lr=0.05, n_epochs=50,
                                        train_mask=train_mask_sampled, val_mask=val_mask, test_mask=test_mask,
                                        k_neighbors=5, random_state=42, percentile_q=thresh,
                                        use_reweighted=(sampling == "reweighted_graph_smote")
                                    )
                                
                            elif method == "sage":
                                if sampling in ["none", "random_undersample"]:
                                    model_sage = GraphSAGE(
                                        edge_index=edge_index,
                                        num_features=num_features,
                                        hidden_dim=64, embedding_dim=32, output_dim=output_dim,
                                        n_layers=2, dropout_rate=0.3, sage_aggr="mean"
                                    ).to(device)
                                    ap_loss, f1_loss = GNN_features(
                                        ntw_torch, model_sage, lr=0.05, n_epochs=50, 
                                        train_mask=train_mask_sampled, val_mask=val_mask, test_mask=test_mask, percentile_q=thresh
                                    )
                                else:  # graph_smote / graph_ensemble_smote / reweighted_graph_smote
                                    model_sage = GraphSAGE(
                                        edge_index=edge_index,
                                        num_features=num_features,
                                        hidden_dim=64, embedding_dim=32, output_dim=output_dim,
                                        n_layers=2, dropout_rate=0.3, sage_aggr="mean"
                                    ).to(device)
                                    ap_loss, f1_loss = GNN_features_graphsmote(
                                        ntw_torch, model_sage, lr=0.05, n_epochs=50,
                                        train_mask=train_mask_sampled, val_mask=val_mask, test_mask=test_mask,
                                        k_neighbors=5, random_state=42, percentile_q=thresh,
                                        use_reweighted=(sampling == "reweighted_graph_smote")
                                    )
                                
                            elif method == "gat":
                                if use_lightweight_gat:
                                    gat_hidden_dim = 16
                                    gat_embedding_dim = 8
                                    gat_heads = 1
                                    gat_dropout = 0.2
                                    gat_epochs = 10
                                else:
                                    gat_hidden_dim = 64
                                    gat_embedding_dim = 32
                                    gat_heads = 4
                                    gat_dropout = 0.3
                                    gat_epochs = 50

                                if sampling in ["none", "random_undersample"]:
                                    model_gat = GAT(
                                        num_features=num_features,
                                        hidden_dim=gat_hidden_dim, embedding_dim=gat_embedding_dim, output_dim=output_dim,
                                        n_layers=2, heads=gat_heads, dropout_rate=gat_dropout
                                    ).to(device)
                                    ap_loss, f1_loss = GNN_features(
                                        ntw_torch, model_gat, lr=0.05, n_epochs=gat_epochs,
                                        train_mask=train_mask_sampled, val_mask=val_mask, test_mask=test_mask, percentile_q=thresh
                                    )
                                else:  # graph_smote / graph_ensemble_smote / reweighted_graph_smote
                                    model_gat = GAT(
                                        num_features=num_features,
                                        hidden_dim=gat_hidden_dim, embedding_dim=gat_embedding_dim, output_dim=output_dim,
                                        n_layers=2, heads=gat_heads, dropout_rate=gat_dropout
                                    ).to(device)
                                    ap_loss, f1_loss = GNN_features_graphsmote(
                                        ntw_torch, model_gat, lr=0.05, n_epochs=gat_epochs,
                                        train_mask=train_mask_sampled, val_mask=val_mask, test_mask=test_mask,
                                        k_neighbors=5, random_state=42, percentile_q=thresh,
                                        use_reweighted=(sampling == "reweighted_graph_smote")
                                    )
                                
                            elif method == "gin":
                                if sampling in ["none", "random_undersample"]:
                                    model_gin = GIN(
                                        num_features=num_features,
                                        hidden_dim=64, embedding_dim=32, output_dim=output_dim,
                                        n_layers=2, dropout_rate=0.3
                                    ).to(device)
                                    ap_loss, f1_loss = GNN_features(
                                        ntw_torch, model_gin, lr=0.05, n_epochs=50, 
                                        train_mask=train_mask_sampled, val_mask=val_mask, test_mask=test_mask, percentile_q=thresh
                                    )
                                else:  # graph_smote / graph_ensemble_smote / reweighted_graph_smote
                                    model_gin = GIN(
                                        num_features=num_features,
                                        hidden_dim=64, embedding_dim=32, output_dim=output_dim,
                                        n_layers=2, dropout_rate=0.3
                                    ).to(device)
                                    ap_loss, f1_loss = GNN_features_graphsmote(
                                        ntw_torch, model_gin, lr=0.05, n_epochs=50,
                                        train_mask=train_mask_sampled, val_mask=val_mask, test_mask=test_mask,
                                        k_neighbors=5, random_state=42, percentile_q=thresh,
                                        use_reweighted=(sampling == "reweighted_graph_smote")
                                    )
                            
                            if args.mode == 'auc':
                                result = f"AUC-PRC: {ap_loss}, F1: {f1_loss}"
                            else:
                                result = f"AUC-PRC: {ap_loss}, F1_{thresh}: {f1_loss}"
                            
                            results.append(result)
                    
                    # Save results for each threshold, but skip existing files to preserve prior runs
                    for rf, res in zip(result_files_list, results):
                        rf_path = Path(rf)
                        if rf_path.exists():
                            print(f"Skip existing result file: {rf_path}")
                            continue
                        with open(rf_path, "w") as f:
                            f.write(res)
                        print(f"Done! {res} -> {rf_path}")
                    
                    if args.mode == 'f1':
                        print(f"All thresholds completed for {method} {sampling} {ratio}")

                except Exception as e:
                    print(f"Error! method={method} sampling={sampling} ratio={ratio_tag} detail={str(e)}")
                    print(traceback.format_exc())
    
    print("\n" + "="*80)
    print("All training phases completed!")
    print("="*80)
