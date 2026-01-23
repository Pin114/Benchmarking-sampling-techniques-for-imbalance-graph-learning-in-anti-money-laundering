import torch
import os
import sys
import numpy as np

# 設置環境變數，確保路徑正確
DIR = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(DIR + "/../")
sys.path.append(DIR + "/../")

# 確保可以導入 DatasetConstruction 和 evaluation
# 假設 data 和 src 目錄已在正確的位置
from data.DatasetConstruction import load_ibm 
from src.methods.evaluation import random_undersample_mask 

def run_quick_test():
    """
    載入 IBM 資料集，模擬 train_unsupervised.py 中產生錯誤的隨機欠採樣步驟。
    """
    print("--- Starting Quick Test for random_undersample_mask ---")

    try:
        # 1. 載入資料集
        ntw = load_ibm()
        train_mask, val_mask, test_mask = ntw.get_masks()
        
        # 2. 準備輸入 Mask 和 Labels
        original_train_mask = torch.logical_or(train_mask, val_mask).detach()
        # 確保 Mask 是 1D Tensor
        mask_to_pass = original_train_mask.contiguous().view(-1)

        # 獲取 PyG 格式的 Data 物件以取得 labels (y)
        ntw_torch = ntw.get_network_torch()
        labels = ntw_torch.y
        
        print(f"Original mask sum (True values): {mask_to_pass.sum().item()}")

        # 3. 呼叫 random_undersample_mask 函式
        # 根據 evaluation.py 的簽名，這需要 mask 和 labels。
        # 雖然 test_unsupervised.py 傳了 ntw 和 mask_to_pass，但在 evaluation.py 修正後，
        # 我們假設該函式簽名是 (mask, labels)。這裡我們使用最可能導致錯誤的參數組合：
        
        # 由於 test_unsupervised.py 是這樣調用的: train_mask_sampled = random_undersample_mask(ntw, mask_to_pass)
        # 並且在 train_unsupervised.py 是這樣調用的: random_undersample_mask(ntw.train_mask, ntw_torch.y, ...)
        # 我們模擬最簡潔的調用，假設 ntw 物件會被忽略，而 mask_to_pass 是第一個參數。
        # 為了避免在 test_unsupervised.py 中出現 TypeError，我們模擬最壞情況：
        
        # 為了讓它運行，我們假設 evaluation.py 接受 mask 和 labels (y)
        # 由於您的 test_unsupervised.py 傳入 ntw 和 mask，這裡必須猜測 evaluation.py 內部如何處理。
        # 最安全的做法是使用 train_unsupervised.py 的參數簽名：(mask, labels)
        
        print("Executing random_undersample_mask(mask_to_pass, labels)...")
        # 模擬調用 (使用最可能有效的簽名: mask, labels)
        train_mask_sampled = random_undersample_mask(mask_to_pass, labels, target_ratio=1.0, random_state=42)
        
        # 4. 驗證輸出類型和內容
        is_tensor = isinstance(train_mask_sampled, torch.Tensor)
        is_numpy = isinstance(train_mask_sampled, np.ndarray)
        
        # 驗證數據類型是否正確 (應該是 Tensor 或 NumPy bool)
        if not is_tensor and not is_numpy:
             raise TypeError(f"Output type is incorrect: {type(train_mask_sampled)}")
        
        # 驗證內容 (確保沒有 object_ 類型，並可以轉換為 Tensor)
        if is_numpy:
             # 模擬 test_unsupervised.py 正在執行的強制轉換
             _ = train_mask_sampled.astype(np.bool_) 
             train_mask_sampled_final = torch.tensor(_, dtype=torch.bool, device='cpu')
             print(f"Output is NumPy array (Type: {train_mask_sampled.dtype}).")
        else:
             train_mask_sampled_final = train_mask_sampled
             print(f"Output is PyTorch Tensor (Type: {train_mask_sampled.dtype}).")
             
        # 驗證採樣後的大小
        sampled_sum = train_mask_sampled_final.sum().item()
        print(f"Sampling successful. Sampled mask sum: {sampled_sum}")
        print("--- Test PASSED ---")

    except Exception as e:
        print(f"--- Test FAILED with ERROR ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("------------------------------")
        
    
if __name__ == "__main__":
    # 設置環境以確保 PyTorch 可以找到 NumPy
    import torch
    import numpy as np
    
    run_quick_test()