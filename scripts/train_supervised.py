import os 
import sys
import torch

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
    node2vec_features,
    GCN, GraphSAGE, GAT, GIN,
    GNN_features, GNN_features_graphsmote
)
from data.DatasetConstruction import load_ibm, load_elliptic
from src.methods.evaluation import adjust_mask_to_ratio, random_undersample_mask, smote_mask, graph_smote_mask

if __name__ == "__main__":
    ### Load Dataset ###
    ntw_name = "ibm"
    n_trials = 5
    
    # ========== 第一步：定義要測試的 Class Imbalance Ratios ==========
    # ratio = majority_count / minority_count
    test_ratios = [
        None,    # Original imbalance (baseline from dataset)
        2.0,     # 1:2 ratio (APATE findings suggest this is optimal for AML)
        1.0      # 1:1 ratio (fully balanced)
    ]
    
    ratio_names = {
        None: "original",
        2.0: "ratio_1to2",
        1.0: "ratio_1to1"
    }

    if ntw_name == "ibm":
        ntw = load_ibm()
    elif ntw_name == "elliptic":
        ntw = load_elliptic()
    else:
        raise ValueError("Network not found")

    original_train_mask, val_mask, test_mask = ntw.get_masks()
    
    # ========== 第二步：定義方法與對應的採樣 Techniques ==========
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
    
    # 定義每個方法使用的採樣技術
    # Intrinsic & Positional: none, random_undersample, smote
    # GNN methods: none, random_undersample, graph_smote
    method_sampling_techniques = {
        "intrinsic": ["none", "random_undersample", "smote"],
        "positional": ["none", "random_undersample", "smote"],
        "deepwalk": ["none", "random_undersample", "graph_smote"],
        "node2vec": ["none", "random_undersample", "graph_smote"],
        "gcn": ["none", "random_undersample", "graph_smote"],
        "sage": ["none", "random_undersample", "graph_smote"],
        "gat": ["none", "random_undersample", "graph_smote"],
        "gin": ["none", "random_undersample", "graph_smote"]
    }
    
    # --- 數據準備 (在所有迴圈外) ---
    fraud_dict = ntw.get_fraud_dict()
    fraud_dict = {k: 0 if v == 2 else v for k, v in fraud_dict.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ntw_torch = ntw.get_network_torch().to(device)
    if ntw_name == "elliptic":
        ntw_torch.x = ntw_torch.x[:,1:94]
    
    if hasattr(ntw_torch, 'edge_index'):
        edge_index = ntw_torch.edge_index
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    num_features = ntw_torch.num_features
    output_dim = 2
    
    # ========== 第三步：開始 Ratio 迴圈 - 逐個測試各個 imbalance ratios ==========
    for ratio in test_ratios:
        ratio_tag = ratio_names[ratio]
        
        print("\n" + "="*80)
        print(f"PHASE: Testing Class Imbalance Ratio: {ratio_tag.upper()}")
        if ratio is None:
            print("  (Using original dataset imbalance)")
        else:
            print(f"  (Adjusting to {ratio} ratio - majority:minority)")
        print("="*80)
        
        # 根據 ratio 調整 training mask
        if ratio is None:
            train_mask_ratio = original_train_mask
        else:
            train_mask_ratio, orig_maj, orig_min, new_maj = adjust_mask_to_ratio(
                original_train_mask, ntw_torch.y.cpu(), 
                target_ratio=ratio, random_state=42
            )
            train_mask_ratio = train_mask_ratio.to(original_train_mask.device)
            print(f"  Adjusted training set: {new_maj} majority + {orig_min} minority = {new_maj + orig_min} samples")
            print(f"  (Original: {orig_maj} majority + {orig_min} minority = {orig_maj + orig_min} samples)")
        
        # ========== 第四步：訓練所有方法（每個方法用對應的採樣技術）==========
        for method in to_train:
            print(f"\n  Training {method.upper()}...")
            
            # 根據方法類型決定要使用的採樣技術
            sampling_techniques_for_method = method_sampling_techniques.get(method, ["none"])
            
            for sampling in sampling_techniques_for_method:
                print(f"    - Sampling: {sampling.upper()}", end=" ... ")
                
                # 設置tag用於結果檔案名稱
                if sampling == 'none':
                    samp_tag = ''
                else:
                    samp_tag = f'_{sampling}'
                
                result_tag = f"{ntw_name}_{ratio_tag}{samp_tag}"
                
                # 根據採樣類型決定 train_mask_sampled
                if sampling == "none":
                    train_mask_sampled = train_mask_ratio
                    
                elif sampling == "random_undersample":
                    # Random undersampling of majority class
                    mask_to_pass = train_mask_ratio.contiguous().view(-1)
                    train_mask_sampled = random_undersample_mask(mask_to_pass, ntw_torch.y.cpu(), target_ratio=1.0)
                    train_mask_sampled = train_mask_sampled.to(train_mask_ratio.device)
                    
                elif sampling == "smote":
                    # SMOTE for intrinsic & positional (feature-based methods)
                    mask_to_pass = train_mask_ratio.contiguous().view(-1)
                    expanded_features, expanded_labels, expanded_mask = smote_mask(
                        mask_to_pass, ntw_torch.x.cpu(), ntw_torch.y.cpu(), k_neighbors=5, random_state=42
                    )
                    train_mask_sampled = expanded_mask.to(train_mask_ratio.device) if isinstance(expanded_mask, torch.Tensor) else torch.from_numpy(expanded_mask).to(train_mask_ratio.device)
                
                elif sampling == "graph_smote":
                    # GraphSMOTE for GNN methods
                    mask_to_pass = train_mask_ratio.contiguous().view(-1)
                    expanded_features, expanded_labels, expanded_mask, expanded_edge_index = graph_smote_mask(
                        mask_to_pass, ntw_torch.x.cpu(), ntw_torch.y.cpu(), 
                        edge_index.cpu(), k_neighbors=5, random_state=42
                    )
                    train_mask_sampled = expanded_mask.to(train_mask_ratio.device) if isinstance(expanded_mask, torch.Tensor) else torch.from_numpy(expanded_mask).to(train_mask_ratio.device)
                
                # ========== 訓練該方法 - 使用 experiments_supervised.py 中的函數 ==========
                try:
                    if method == "intrinsic":
                        if sampling == "none":
                            ap_loss = intrinsic_features(
                                ntw, train_mask_sampled, val_mask,
                                n_layers_decoder=2, hidden_dim_decoder=16, lr=0.05, n_epochs_decoder=100
                            )
                        elif sampling == "random_undersample":
                            ap_loss = intrinsic_features(
                                ntw, train_mask_sampled, val_mask,
                                n_layers_decoder=2, hidden_dim_decoder=16, lr=0.05, n_epochs_decoder=100
                            )
                        else:  # SMOTE
                            ap_loss = intrinsic_features_smote(
                                ntw, train_mask_sampled, val_mask,
                                n_layers_decoder=2, hidden_dim_decoder=16, lr=0.05, n_epochs_decoder=100,
                                k_neighbors=5, random_state=42
                            )
                        result = f"AUC-PRC: {ap_loss}"

                    elif method == "positional":
                        if sampling == "none":
                            ap_loss = positional_features(
                                ntw, train_mask_sampled, val_mask,
                                alpha_pr=0.5, alpha_ppr=0, n_epochs_decoder=50, lr=0.05,
                                fraud_dict_test=fraud_dict,
                                n_layers_decoder=2, hidden_dim_decoder=16, 
                                ntw_name=ntw_name+"_train"
                            )
                        elif sampling == "random_undersample":
                            ap_loss = positional_features(
                                ntw, train_mask_sampled, val_mask,
                                alpha_pr=0.5, alpha_ppr=0, n_epochs_decoder=50, lr=0.05,
                                fraud_dict_test=fraud_dict,
                                n_layers_decoder=2, hidden_dim_decoder=16, 
                                ntw_name=ntw_name+"_train"
                            )
                        else:  # SMOTE
                            ap_loss = positional_features_smote(
                                ntw, train_mask_sampled, val_mask,
                                alpha_pr=0.5, alpha_ppr=0, n_epochs_decoder=50, lr=0.05,
                                fraud_dict_test=fraud_dict,
                                n_layers_decoder=2, hidden_dim_decoder=16, 
                                ntw_name=ntw_name+"_train",
                                k_neighbors=5, random_state=42
                            )
                        result = f"AUC-PRC: {ap_loss}"

                    elif method == "deepwalk":
                        ap_loss = node2vec_features(
                            ntw_torch, train_mask_sampled, val_mask,
                            embedding_dim=32, walk_length=5, context_size=3,
                            walks_per_node=2, num_negative_samples=3,
                            p=1, q=1, lr=0.05, n_epochs=50, n_epochs_decoder=50, 
                            use_torch=True
                        )
                        result = f"AUC-PRC: {ap_loss}"

                    elif method == "node2vec":
                        ap_loss = node2vec_features(
                            ntw_torch, train_mask_sampled, val_mask,
                            embedding_dim=32, walk_length=5, context_size=3,
                            walks_per_node=2, num_negative_samples=3,
                            p=1.5, q=1.0, lr=0.05, n_epochs=50, n_epochs_decoder=50, 
                            ntw_nx=ntw.get_network_nx(), use_torch=False
                        )
                        result = f"AUC-PRC: {ap_loss}"

                    elif method == "gcn":
                        if sampling in ["none", "random_undersample"]:
                            model_gcn = GCN(
                                edge_index=edge_index, num_features=num_features,
                                hidden_dim=128, embedding_dim=64, output_dim=output_dim,
                                n_layers=2, dropout_rate=0.3
                            ).to(device)
                            ap_loss = GNN_features(ntw_torch, model_gcn, lr=0.05, n_epochs=100, 
                                                   train_mask=train_mask_sampled, test_mask=val_mask)
                        else:  # GraphSMOTE
                            model_gcn = GCN(
                                edge_index=edge_index, num_features=num_features,
                                hidden_dim=128, embedding_dim=64, output_dim=output_dim,
                                n_layers=2, dropout_rate=0.3
                            ).to(device)
                            ap_loss = GNN_features_graphsmote(
                                ntw_torch, model_gcn, lr=0.05, n_epochs=100, 
                                train_mask=train_mask_sampled, test_mask=val_mask,
                                k_neighbors=5, random_state=42
                            )
                        result = f"AUC-PRC: {ap_loss}"

                    elif method == "sage":
                        if sampling in ["none", "random_undersample"]:
                            model_sage = GraphSAGE(
                                num_features=num_features,
                                hidden_dim=128, embedding_dim=64, output_dim=output_dim,
                                n_layers=2, dropout_rate=0.3, sage_aggr="mean"
                            ).to(device)
                            ap_loss = GNN_features(ntw_torch, model_sage, lr=0.05, n_epochs=100, 
                                                   train_mask=train_mask_sampled, test_mask=val_mask)
                        else:  # GraphSMOTE
                            model_sage = GraphSAGE(
                                num_features=num_features,
                                hidden_dim=128, embedding_dim=64, output_dim=output_dim,
                                n_layers=2, dropout_rate=0.3, sage_aggr="mean"
                            ).to(device)
                            ap_loss = GNN_features_graphsmote(
                                ntw_torch, model_sage, lr=0.05, n_epochs=100, 
                                train_mask=train_mask_sampled, test_mask=val_mask,
                                k_neighbors=5, random_state=42
                            )
                        result = f"AUC-PRC: {ap_loss}"

                    elif method == "gat":
                        if sampling in ["none", "random_undersample"]:
                            model_gat = GAT(
                                num_features=num_features,
                                hidden_dim=128, embedding_dim=64, output_dim=output_dim,
                                n_layers=2, heads=4, dropout_rate=0.3
                            ).to(device)
                            ap_loss = GNN_features(ntw_torch, model_gat, lr=0.05, n_epochs=100, 
                                                   train_mask=train_mask_sampled, test_mask=val_mask)
                        else:  # GraphSMOTE
                            model_gat = GAT(
                                num_features=num_features,
                                hidden_dim=128, embedding_dim=64, output_dim=output_dim,
                                n_layers=2, heads=4, dropout_rate=0.3
                            ).to(device)
                            ap_loss = GNN_features_graphsmote(
                                ntw_torch, model_gat, lr=0.05, n_epochs=100, 
                                train_mask=train_mask_sampled, test_mask=val_mask,
                                k_neighbors=5, random_state=42
                            )
                        result = f"AUC-PRC: {ap_loss}"

                    elif method == "gin":
                        if sampling in ["none", "random_undersample"]:
                            model_gin = GIN(
                                num_features=num_features,
                                hidden_dim=128, embedding_dim=64, output_dim=output_dim,
                                n_layers=2, dropout_rate=0.3
                            ).to(device)
                            ap_loss = GNN_features(ntw_torch, model_gin, lr=0.05, n_epochs=100, 
                                                   train_mask=train_mask_sampled, test_mask=val_mask)
                        else:  # GraphSMOTE
                            model_gin = GIN(
                                num_features=num_features,
                                hidden_dim=128, embedding_dim=64, output_dim=output_dim,
                                n_layers=2, dropout_rate=0.3
                            ).to(device)
                            ap_loss = GNN_features_graphsmote(
                                ntw_torch, model_gin, lr=0.05, n_epochs=100, 
                                train_mask=train_mask_sampled, test_mask=val_mask,
                                k_neighbors=5, random_state=42
                            )
                        result = f"AUC-PRC: {ap_loss}"

                    # 保存結果
                    with open(f"res/{method}_params_{result_tag}.txt", "w") as f:
                        f.write(result)
                    
                    print(f"Done! {result}")

                except Exception as e:
                    print(f"Error! {str(e)}")
    
    print("\n" + "="*80)
    print("All training phases completed!")
    print("="*80)
