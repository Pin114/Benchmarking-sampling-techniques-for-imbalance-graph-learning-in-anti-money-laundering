import torch
import os
import sys
import numpy as np
import ast

DIR = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(DIR + "/../")
sys.path.append(DIR + "/../")

# ---------------- 🎯 修正 1: 處理 SIGKILL/多核問題 ----------------
# Node2Vec的底層並行計算可能會導致記憶體超限。
# 這裡強制將 OpenMP 核心數設為 1，以限制並行度，減輕 SIGKILL 風險。
os.environ["OMP_NUM_THREADS"] = "1"

# N2V_WORKERS 變數雖然沒有在函數中使用，但保留作為單核意圖的標記
N2V_WORKERS = 1 

from sklearn.ensemble import IsolationForest

from src.methods.experiments_unsupervised import *
from data.DatasetConstruction import *
from src.methods.evaluation import *
from src.methods.evaluation import random_undersample_mask # 假設這是您的欠採樣函數

if __name__ == "__main__":
    
    # ------------------ 🎯 採樣技術列表 (與 train_unsupervised 保持一致) ------------------
    sampling_techniques = ["none", "random_undersample"] 

    use_intrinsic = True
    intrinsic_str = "_intrinsic" if use_intrinsic else "_no_intrinsic"

    if use_intrinsic:
        to_test = ["intrinsic", "positional", "deepwalk", "node2vec"]
    else:
        to_test = ["positional",  "node2vec"]

    ### Load Dataset ###
    ntw_name = "ibm"

    if ntw_name == "ibm":
        ntw = load_ibm()
    elif ntw_name == "elliptic":
        ntw = load_elliptic()
    else:
        raise ValueError("Network not found")
    
    # Masks, device and default sampling tag
    percentile_q_list = [90, 99, 99.9]
    train_mask, val_mask, test_mask = ntw.get_masks()
    # 儲存原始的 Train + Validation Mask
    original_train_mask = torch.logical_or(train_mask, val_mask).detach()

    device_decoder = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    # ---------- Positional ----------
    x_intrinsic = ntw.get_features_torch()

    fraud_dict = ntw.get_fraud_dict()
    fraud_dict = {k: 0 if v == 2 else v for k, v in fraud_dict.items()}


    # ------------------ 🎯 迴圈開始：迭代採樣技術 ------------------
    for sampling in sampling_techniques:
        print("="*50)
        print(f"Starting test with sampling technique: {sampling.upper()}")
        print("="*50)

        samp_tag = '' if sampling == 'none' else f'_{sampling}'
        
        # 根據採樣類型決定 train_mask_sampled
        if sampling == "none":
            # 不採樣，使用原始的 Train + Val mask
            train_mask_sampled = original_train_mask
        elif sampling == "random_undersample":
            # 執行欠採樣
            # 確保 mask 是一個明確的 1D Tensor，以避免 ValueError
            mask_to_pass = original_train_mask.contiguous().view(-1)
            
            # 假設 random_undersample_mask(ntw, mask_tensor) 返回採樣後的 mask
            # evaluation.py 中的修正應該確保這裡返回的是 PyTorch Tensor
            train_mask_sampled = random_undersample_mask(ntw, mask_to_pass)
            
        else:
             # 如果定義了新的採樣技術，需要在這裡增加處理邏輯
             raise ValueError(f"Unknown sampling technique: {sampling}. Please define its implementation.")




        skip_positional_test = False

        if "positional" in to_test:
            print("Positional features")
            param_dict = None # 確保 param_dict 始終有定義
            
            # 檔案路徑現在會包含採樣標籤
            params_path = f"res/positional_params_{ntw_name}_unsupervised{samp_tag}.txt"
            
            if not os.path.exists(params_path):
                print(f"Warning: positional params file not found: {params_path}. Using fallback parameters.")
                # 🎯 設定後備參數
                param_dict = {
                    "n_estimators": 100,
                    "max_samples": 0.5,
                    "max_features_dec%": 1, 
                    "bootstrap": True,
                    "alpha_pr": 0.5 # 必須為 positional 設置預設 alpha_pr
                }
            else:
                with open(params_path, "r") as f:
                    params = f.readlines()
                param_dict = eval(params[0].strip())
            
            # 🎯 修正點：在 try 區塊上方計算 max_features_dec，確保其作用域正確
            max_features_dec = param_dict.get("max_features_dec%", 1)

            # --- 🎯 執行特徵計算和評估 ---
            try:
                # 確保 alpha_pr 存在
                alpha_pr_val = param_dict.get("alpha_pr", 0.5) 
                
                features_df = positional_features_calc(
                    ntw,
                    alpha_pr=alpha_pr_val, 
                    alpha_ppr=None,
                    fraud_dict_train=None,
                    fraud_dict_test=fraud_dict,
                    ntw_name=ntw_name + "_test",
                    use_intrinsic=use_intrinsic,
                )

                # Safe column dropping
                cols_to_drop = [c for c in ["PSP", "fraud"] if c in features_df.columns]

                # 🎯 修正點 1：強制使用 'cpu' 進行 Tensor 轉換，以避免 MPS 錯誤
                x = torch.tensor(features_df.drop(cols_to_drop, axis=1).values, dtype=torch.float32).to('cpu')
                y = torch.tensor(features_df["fraud"].values, dtype=torch.long).to('cpu')

                # 🎯 修正點 2：確保 train_mask_sampled 在 'cpu' 上
                if not isinstance(train_mask_sampled, torch.Tensor):
                    train_mask_sampled = torch.tensor(train_mask_sampled, dtype=torch.bool, device='cpu')
                else:
                    train_mask_sampled = train_mask_sampled.to('cpu') # 確保設備一致


                # mask including sampled train mask
                # 確保 test_mask 也在 CPU 上
                mask_s = torch.logical_or(train_mask_sampled, test_mask.to('cpu'))
            
                # 🎯 修正點：計算 max_features
                max_features = int(np.ceil(max_features_dec * x.shape[1] / 10))

                model_pos = IsolationForest(
                    n_estimators=param_dict.get("n_estimators", 100),
                    max_samples=param_dict.get("max_samples", 0.5),
                    max_features=max_features,
                    bootstrap=param_dict.get("bootstrap", True),
                )

                AUC_list_pos, AP_list_pos, precision_dict_pos, recall_dict_pos, F1_dict_pos = evaluate_if(
                    model_pos, x[mask_s], y[mask_s], percentile_q_list=percentile_q_list
                )

                save_results_TI(AUC_list_pos, AP_list_pos, f"{ntw_name}_positional_unsupervised{intrinsic_str}{samp_tag}")
                save_results_TD(precision_dict_pos, recall_dict_pos, F1_dict_pos, f"{ntw_name}_positional_unsupervised{intrinsic_str}{samp_tag}")
            
            except Exception as e:
                print(f"[ERROR] Positional features testing failed for sampling '{sampling}'. Setting skip_positional_test=True.")
                print(f"Detailed Error: {e}")
                skip_positional_test = True # 如果特徵計算或模型評估失敗，則設定跳過標記

        # 🎯 在進入 DeepWalk/Node2Vec 之前，檢查是否應該跳過整個區塊 
        if skip_positional_test:
            continue # 跳到下一個採樣迴圈 (RANDOM_UNDERSAMPLE)

        # ---------- Torch Models (DeepWalk / Node2Vec) ----------
        # 由於我們在腳本開頭設定了 OMP_NUM_THREADS="1"，這裡無需額外動作，它會限制多核運算。
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ntw_torch = ntw.get_network_torch().to(device)

        # Match train side slicing (remove time_step, keep 1:94)
        if hasattr(ntw_torch, "x"):
            ntw_torch.x = ntw_torch.x[:, 1:94]

        # ---------- Common Mask ----------
        # 🎯 關鍵修正：確保 train_mask_sampled 是 Tensor
        if not isinstance(train_mask_sampled, torch.Tensor):
            train_mask_sampled = torch.tensor(train_mask_sampled, dtype=torch.bool, device=ntw_torch.y.device)
            
        mask_s = torch.logical_or(train_mask_sampled, test_mask)
        # ---------- DeepWalk ----------
        if "deepwalk" in to_test:
            print("Deepwalk")
            # prefer deepwalk params file; if missing, fall back to node2vec params and force p=1,q=1
            params_path = f"res/deepwalk_params_{ntw_name}_unsupervised{samp_tag}.txt"
            used_node2vec_as_deepwalk = False
            if not os.path.exists(params_path):
                node2v_path = f"res/node2vec_params_{ntw_name}_unsupervised{samp_tag}.txt"
                if os.path.exists(node2v_path):
                    params_path = node2v_path
                    used_node2vec_as_deepwalk = True
                    print(f"Info: deepwalk params not found, falling back to node2vec params: {node2v_path}")
                else:
                    print(f"Warning: deepwalk params file not found: {params_path}. Skipping deepwalk test.")
                    params_path = None

            if params_path is not None:
                with open(params_path, "r") as f:
                    params = f.readlines()
                try:
                    param_dict = ast.literal_eval(params[0].strip())
                except Exception:
                    param_dict = eval(params[0].strip())

                    # if we used node2vec params as a deepwalk proxy, force p/q to 1
                    if used_node2vec_as_deepwalk:
                        param_dict['p'] = 1
                        param_dict['q'] = 1
                    
                    # ---  DeepWalk/Node2Vec 參數後備 ---
                    max_features_dec = param_dict.get("max_features_dec%", 1)
                    

                    model_deepwalk = node2vec_representation_torch(
                        ntw_torch,
                        train_mask_sampled,
                        test_mask,
                        # 降低參數以節省記憶體
                        embedding_dim=param_dict.get("embedding_dim", 16), # 維持 16 或降至 8
                        walk_length=param_dict.get("walk_length", 3),      # 由 5 降至 3
                        context_size=param_dict.get("context_size", 2),    # 由 3 降至 2
                        walks_per_node=param_dict.get("walks_per_node", 1), # 維持 1
                        num_negative_samples=param_dict.get("num_negative_samples", 1), # 維持 1
                        p=1,
                        q=1,
                        lr=param_dict.get("lr", 0.025),
                        n_epochs=param_dict.get("n_epochs", 10),           # 由 20 降至 10
                    )

                    x = model_deepwalk().detach().cpu()

                    if use_intrinsic:
                        x_intrinsic = x_intrinsic.cpu()
                        x = torch.cat((x, x_intrinsic), dim=1)[mask_s]
                    else:
                        x = x[mask_s]

                    y = ntw_torch.y.clone().detach().cpu()[mask_s]

                    # 使用修正後的 max_features_dec
                    max_features = int(np.ceil(max_features_dec * x.shape[1] / 10))

                    model_deepwalk = IsolationForest(
                        n_estimators=param_dict.get("n_estimators", 100),
                        max_samples=param_dict.get("max_samples", 0.5),
                        max_features=max_features,
                        bootstrap=param_dict.get("bootstrap", True),
                    )

                    AUC_list_dw, AP_list_dw, precision_dict_dw, recall_dict_dw, F1_dict_dw = evaluate_if(
                        model_deepwalk, x, y, percentile_q_list=percentile_q_list
                    )

                    save_results_TI(AUC_list_dw, AP_list_dw, f"{ntw_name}_deepwalk_unsupervised{intrinsic_str}{samp_tag}")
                    save_results_TD(precision_dict_dw, recall_dict_dw, F1_dict_dw, f"{ntw_name}_deepwalk_unsupervised{intrinsic_str}{samp_tag}")

            # ---------- Node2Vec ----------
            if "node2vec" in to_test:
                print("Node2vec")
                params_path = f"res/node2vec_params_{ntw_name}_unsupervised{samp_tag}.txt"
                if not os.path.exists(params_path):
                    print(f"Warning: node2vec params file not found: {params_path}. Skipping node2vec test.")
                else:
                    with open(params_path, "r") as f:
                        params = f.readlines()

                    param_dict = eval(params[0].strip())
                    
                    # --- Node2Vec 參數後備 ---
                    max_features_dec = param_dict.get("max_features_dec%", 1)

                    model_node2vec = node2vec_representation_torch(
                        ntw_torch,
                        train_mask_sampled,
                        test_mask,
                        # 降低參數以節省記憶體
                        embedding_dim=param_dict.get("embedding_dim", 16), # 維持 16 或降至 8
                        walk_length=param_dict.get("walk_length", 3),      # 由 5 降至 3
                        context_size=param_dict.get("context_size", 2),    # 由 3 降至 2
                        walks_per_node=param_dict.get("walks_per_node", 1), # 維持 1
                        num_negative_samples=param_dict.get("num_negative_samples", 1), # 維持 1
                        p=param_dict.get("p", 1.0),
                        q=param_dict.get("q", 1.0),
                        lr=param_dict.get("lr", 0.025),
                        n_epochs=param_dict.get("n_epochs", 10),           # 由 20 降至 10
                    )

                    x = model_node2vec().detach().cpu()

                    if use_intrinsic:
                        x_intrinsic = x_intrinsic.cpu()
                        x = torch.cat((x, x_intrinsic), dim=1)[mask_s]
                    else:
                        x = x[mask_s]

                    y = ntw_torch.y.clone().detach().cpu()[mask_s]

                    # 使用修正後的 max_features_dec
                    max_features = int(np.ceil(max_features_dec * x.shape[1] / 10))

                    model_node2vec = IsolationForest(
                        n_estimators=param_dict.get("n_estimators", 100),
                        max_samples=param_dict.get("max_samples", 0.5),
                        max_features=max_features,
                        bootstrap=param_dict.get("bootstrap", True),
                    )

                    AUC_list_n2v, AP_list_n2v, precision_dict_n2v, recall_dict_n2v, F1_dict_n2v = evaluate_if(
                        model_node2vec, x, y, percentile_q_list=percentile_q_list
                    )

                    # Build an experiment-specific filename using the key node2vec params
                    exp_name = f"{ntw_name}_node2vec"
                    exp_name += f"_p{param_dict.get('p','NA')}_q{param_dict.get('q','NA')}_d{param_dict.get('embedding_dim','NA')}_w{param_dict.get('walks_per_node','NA')}_ep{param_dict.get('n_epochs','NA')}"
                    exp_name += intrinsic_str + samp_tag

                    save_results_TI(AUC_list_n2v, AP_list_n2v, exp_name)
                    save_results_TD(precision_dict_n2v, recall_dict_n2v, F1_dict_n2v, exp_name)