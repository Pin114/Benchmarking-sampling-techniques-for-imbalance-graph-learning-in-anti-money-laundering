# 反洗錢圖機器學習程式碼庫技術審計報告

*基於當前 `pinyu` 分支（merge 自 `main`，commit `f10c235` 之後）的實際程式碼狀態撰寫。*

本報告由 7 個獨立章節組成，逐一針對數據流水線、切分機制、洩漏風險、採樣技術理論對照、聯合訓練機制、分類器架構與超參數、以及張量形狀風險做顆粒度極細的審計，並在每節末尾附上專業見解。所有引用均標明檔案與行號，可直接對照原始碼。

---

## 1. 數據流水線與圖結構建構（細粒度視角）

### 1.1 生命週期：從 CSV 到 `Data` 物件

兩條完全獨立的載入路徑，由 [`data/DatasetConstruction.py`](data/DatasetConstruction.py) 管理，兩者最終都餵入同一個 [`network_AML`](src/utils/Network.py#L16) 類別。

**(a) Elliptic — `load_elliptic()`（[DatasetConstruction.py:17-53](data/DatasetConstruction.py#L17-L53)）**
- 讀三個檔案：`elliptic_txs_features.csv`（欄 0=txId，欄 1=time_step，欄 2-166=165 個匿名特徵）、`elliptic_txs_edgelist.csv`、`elliptic_txs_classes.csv`。
- `x = feat_df.loc[:, 'time_step':].values`（L32）把 `time_step` 本身也當作 `x` 的第 0 欄；`scripts/train_supervised_tuned.py` 之後用 `ntw_torch.x[:, 1:94]` 手動切掉它，只留 93 個「local」特徵（丟棄 72 個「aggregated」特徵欄與 `time_step` 本身）。
- 類別映射 `{'unknown': 2, '1': 1, '2': 0}`（L36）：`'1'`=illicit→`y=1`；`'2'`=licit→`y=0`；`'unknown'`→`y=2`（之後在建 mask 時被排除）。
- **切分是嚴格的時間切分**：`train: time_step<30`、`val: 30<=t<40`、`test: t>=40`，並且都 `& (y != 2)`（L47-49）。

**(b) IBM 系列 — `load_ibm_config()`（[DatasetConstruction.py:134-199](data/DatasetConstruction.py#L134-L199)）**
- 依 `Timestamp` 升冪排序，丟自轉帳（`Account==Account.1`），只保留排序後**最後** 500,000 筆（L154-156，取尾段而非隨機抽樣）。
- 節點特徵最終集合：`Amount Received/Paid`（**未做任何標準化/縮放**，見 §6.4 已知限制）+ `Day/Hour/Minute`（僅行事曆分量）+ 三組 one-hot（Receiving Currency, Payment Currency, Payment Format）。**帳號、銀行代碼被整個丟棄**，模型完全看不到「哪個帳戶/銀行」，只能透過拓撲間接感知。
- 邊建構於 `preprocess_ibm()`（L75-132）：節點＝**交易本身**（不是帳戶），若交易 A 的收款帳戶 == 交易 B 的付款帳戶，且 `0<=Δt<=240min`，建有向邊 A→B（建模資金流路徑）。
- **切分**（L186-195）：時間排序後，前 60%=train、接下來 20%=val、最後 20%=test——等價的時間切分。

### 1.2 `network_AML` 與精確圖結構語意

[`src/utils/Network.py`](src/utils/Network.py)：

- **邊被無條件對稱化**（`_set_up_network_info`, L51-57）：`self.directed` 從未被任何呼叫端設為 `True`，每條有向邊都被複製反向邊再合併——**方向語意在進入 GNN 前就已被抹除**（IBM 的資金流方向、Elliptic 原始支付方向皆然）。
- **`get_network_torch()`**（L104-134）：一次性把**全部**（train+val+test）節點與邊組成單一 `torch_geometric.data.Data(x, y, edge_index)`；三個 mask 只是掛在同一物件上的布林張量。**整張圖只有一份**，這是純 transductive、全圖（full-batch）設定，train/val/test 從未被切成獨立子圖。
- **`get_features(full=False)`**：Elliptic 專屬分支——`full=False`→欄 2-94（93 欄，local），`full=True`→欄 2-166（165 欄，local+aggregated）。`intrinsic_features*` 用預設 `full=False`（93 欄），但 `positional_features*` 呼叫 `ntw.get_features(full=True)`（165 欄）——**兩條 baseline pipeline 的「intrinsic 特徵空間」定義不同**,比較兩者分數時要留意這不是同一組特徵。

### 1.3 全域拓撲：GCN/GAT/GIN 全圖前向，GraphSAGE 現已有真正的鄰域採樣

過去所有 `GNN_features*` 函式簽名都帶著從未被引用的 `train_loader`/`val_loader`/`test_loader: DataLoader = None` 死參數——**這些參數與 `from torch.utils.data import DataLoader` import 已被移除**（[experiments_supervised.py](src/methods/experiments_supervised.py) 目前 5 個 `GNN_features*` 函式簽名均已清空這三個參數）。

四個架構目前的前向傳播機制（[`src/methods/utils/GNN.py`](src/methods/utils/GNN.py)）：

| 架構 | 訓練時的圖規模 | edge_attr/edge_weight |
|---|---|---|
| **GCN**（L29-77） | 全圖一次前向 | 真正使用：`edge_weight = edge_attr.view(-1).to(torch.float32)`（L63-65） |
| **GraphSAGE**（L79-136） | **真正的鄰域採樣 mini-batch**（見下） | 不使用（架構本身無此概念,非 bug——見 L120-124 註解） |
| **GAT**（L139-196） | 全圖一次前向 | 真正使用：`GATv2Conv(..., edge_dim=1)`（L163-171）,`edge_attr` reshape 成 `[-1,1]`（L179）後真正進入注意力計算 |
| **GIN**（L198-268） | 全圖一次前向 | 不使用（原始 GIN 公式無邊特徵,這正是 `GINE`（L275+,本 benchmark 未使用）存在的原因——見 L207-210 註解） |

**GraphSAGE 的鄰域採樣**（[experiments_supervised.py:70-101](src/methods/experiments_supervised.py#L70-L101)、整合於 [`GNN_features`](src/methods/experiments_supervised.py#L704)/[`GNN_features_with_predictions`](src/methods/experiments_supervised.py#L826)）：
```python
use_neighbor_sampling = isinstance(model, GraphSAGE)
if use_neighbor_sampling:
    num_neighbors = sage_num_neighbors or [15] * max(int(model.n_layers), 1)
    train_loader = _try_build_neighbor_loader(ntw_torch, train_mask_sampled, sage_batch_size, num_neighbors, shuffle=True)
    use_neighbor_sampling = train_loader is not None
```
`_build_neighbor_loader` 用 `torch_geometric.loader.NeighborLoader` 在 CPU 副本圖上建構,每個 mini-batch 把種子節點（`batch.batch_size` 個）放在最前面,後面接上取樣到的多跳鄰居；loss 只對種子節點段（`out[:seed_n]`）計算。`_try_build_neighbor_loader`（L85-101）會先真的拉一個 batch 探測：若當前環境缺 `pyg-lib`/`torch-sparse` 編譯後端（`NeighborLoader` 的取樣器需要其一）,會印出警告並**優雅降級成全圖前向**,不會讓整個實驗因為環境缺依賴而失敗。評估（`evaluate_split`）**永遠在全圖上跑**,不受此影響。

**專業見解**：這個設計把「訓練用鄰域採樣、評估用全圖」明確分開,是文獻上常見且站得住腳的折衷（模型選擇仍然公平,因為所有架構的 val/test 都在同一張未經採樣的圖上比較）。但要注意：GCN/GAT/GIN 目前完全沒有為大圖 mini-batch 訓練留擴充點——若未來要在百萬節點級的 IBM 資料集上對這三個架構也做記憶體優化,需要一次不小的架構重構,而不只是比照 SAGE 打開一個開關。

---

## 2. 數據切分機制與訓練-評估對齊

### 2.1 確切實現

兩個資料集都是**時間切分**（見 §1.1）,非隨機。Mask 以 `torch.bool` 儲存在 `Data.train_mask/val_mask/test_mask`（[Network.py:127-132](src/utils/Network.py#L127-L132)）,長度＝全圖節點數,直接對應節點編號。

### 2.2 Mask 在訓練/評估迴圈的逐行套用

以 `GNN_features`（[experiments_supervised.py:704-824](src/methods/experiments_supervised.py#L704-L824)）為例：
```python
train_mask_sampled = train_mask.bool().to(device)                 # sampling=="none" 分支
def train_epoch():
    out, _ = _forward(ntw_torch.x.to(device), ntw_torch.edge_index.to(device))  # 整圖前向
    y_train = y[train_mask_sampled]                                  # 只取 train 節點標籤
    loss_val = _compute_loss(criterion, out[train_mask_sampled], y_train)  # 只取 train 節點 logits
```
GNN 每個 epoch 對**全圖**（含 val/test）做一次前向傳播（訊息傳遞讓 val/test 節點的特徵參與 train 節點 embedding 聚合——這是 transductive 學習的必然,結構上使用了 val/test 節點沒有問題,只要沒有把它們的**標籤**用進 loss）,而 loss 計算明確切片,只有 `train_mask` 為真的節點進入梯度——這一步是乾淨的。**注意：`_compute_loss` 呼叫不再重複傳 `mask=` 參數**（過去這裡曾有一個 bug,見 §6.3）。

`evaluate_split(mask)` 在 `torch.no_grad()` 下重新做一次全圖前向,用同樣切片手法算 `val_ap`/`test_f1`。Epoch 迴圈只在 `val_mask` 觸發 `early_stopping`；`test_mask` 只在訓練結束、載回 val-best checkpoint 之後被觸碰恰好一次——正確的隔離順序。

`GNN_features_graphsmote(_with_predictions)`（[L1009-1255](src/methods/experiments_supervised.py#L1009-L1255)）在有合成節點時,`val_mask`/`test_mask` 被**對稱 padding**（合成節點永遠對 val/test mask 補 `False`）,兩者用完全相同的寫法、緊鄰兩行程式碼,沒有不對稱。

### 2.3 驗證集使用的嚴謹性審計——全部方法已對齊

`node2vec_features`/`node2vec_features_with_predictions`（[L488-595](src/methods/experiments_supervised.py#L488-L595)、[L596-703](src/methods/experiments_supervised.py#L596-L703)）現在的 decoder 訓練迴圈：
```python
x_val = x[:val_mask.shape[0]][val_mask.bool()].to(device_decoder).squeeze()
early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path, monitor='val_ap')
for epoch in range(n_epochs_decoder):
    ...
    val_ap = average_precision_score(y_val.cpu().numpy(), val_output_softmax.cpu().numpy()[:, 1])
    early_stopping(val_ap, decoder)
    if early_stopping.early_stop: break
if os.path.exists(checkpoint_path):
    decoder.load_state_dict(torch.load(checkpoint_path, map_location=device_decoder))
```
與 `intrinsic_features_with_predictions`/`positional_features_with_predictions` 用的是完全同一套機制。`train_supervised_tuned.py` 呼叫時也已補上 `patience=10, checkpoint_path=unique_checkpoint_path`。

**刻意保留不動的部分**：Node2Vec 的隨機遊走預訓練本身（`node2vec_representation_torch`, [functionsTorch.py:42-172](src/methods/utils/functionsTorch.py#L42-L172)）依然沒有驗證集訊號、固定跑 `n_epochs` 次無 early stop——這是無監督 skip-gram 目標的合理設計選擇（沒有自然的監督驗證訊號,若要用下游 AP 做 early stop 得每個 embedding epoch 都跑一次 decoder,混淆了兩個訓練階段的職責）,不是遺漏。

**現況總結**：`intrinsic`、`positional`、`node2vec/deepwalk`（decoder 階段）、所有 `GNN_features*` 系列,現在**全部**使用 `val_mask` 做 `EarlyStopping(monitor='val_ap')` + 最佳權重回載——跨 pipeline 的驗證機制已對齊。

**專業見解**：Node2Vec 在 `use_torch=True` 時只保留 `train∪test` 節點子圖建圖（[experiments_supervised.py:466](src/methods/experiments_supervised.py) 附近：`active_nodes = train_mask.bool() | test_mask.bool()`,故意排除 val）,代表 embedding 訓練階段**結構上**看得到 test 節點與其邊（雖然看不到 test 標籤）。這與 GNN 每個 epoch 對全圖做訊息傳遞本質相同,是 transductive 表徵學習常見的「結構性接觸」,值得記錄但不算 bug。

---

## 3. 詳盡的數據洩漏漏洞審計

### 3.1 位置/結構特徵洩漏——`positional_features_with_predictions`：乾淨，且已被主動加固

逐行追蹤（[L360-487](src/methods/experiments_supervised.py#L360-L487)）：
```python
train_val_mask = train_mask.bool() | val_mask.bool()
train_val_nodes = set(torch.where(train_val_mask)[0].tolist())
ntw_nx_train_val = ntw_nx_full.subgraph(list(train_val_nodes))    # 誘導子圖,test 節點與邊完全不存在
features_nx_df_train_val = local_features_nx(ntw_nx_train_val, ...)   # PageRank/密度/RNC 用這個子圖算
features_nk_df_train_val = features_nk(ntw_nx_train_val, ...)         # Betweenness/Closeness/Eigenvector 也是
```
程式碼裡明確留有註解：`# Subgraph isolation to completely block test set structure leakage`。`nx.pagerank`、`nx.betweenness_centrality`/`closeness_centrality`/`eigenvector_centrality`（[functionsNetworKit.py:7-36](src/methods/utils/functionsNetworKit.py#L7-L36)）在計算「訓練用特徵」時,物理上看不到任何 test 節點或連到 test 節點的邊。

`features_df_full`（全圖算的版本）只用來抽取 `X_test`,從不進入 `loss_val.backward()`——test 特徵可以看到全圖拓撲（推論時的合理假設）,但這不構成訓練梯度洩漏。

**唯一值得記錄、非 bug 的細節**：train 特徵子圖混了 val 節點拓撲（`train_val_mask = train|val`）——train 節點的中心性數值會被 val 節點的存在輕微影響,但只涉及結構,從不涉及 val 的**標籤**（`fraud_dict_train` 已被 `train_supervised_tuned.py` 的 `fraud_dict_known` 過濾成純 train,且有 `assert` 把關）。

### 3.2 重採樣/SMOTE 洩漏——`graph_smote_mask`/`GATEdgeGenerator` 已修復並保持一致

現況（[evaluation.py:198-296](src/methods/evaluation.py#L198-L296)）：
```python
nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, features_masked.shape[0]), algorithm='ball_tree').fit(features_masked)
...
for neighbor_idx in indices[0][1:]:
    neighbor_global_idx = int(idx_mask[neighbor_idx])   # 換算回全域 index
```
`features_masked = features_np[idx_mask]`——k-NN 候選池嚴格限制在 train_mask 節點,與 `GATEdgeGenerator.prepare_synthetic_nodes`（[samplers.py:159-195](src/methods/samplers.py#L159-L195)：`nbrs = NearestNeighbors(...).fit(features_masked)`）、`TargetedNeighbourhoodUndersampling.__call__`（本來就正確）採用一致的做法。合成的少數類節點只能連到訓練集裡的真實節點。

`GraphENS`（[graphens.py](src/methods/graphens.py)）的鄰接表建構於全圖真實邊,`minor_pool`/`target_pool` 限制在 train,但一個 train 少數類節點的真實鄰居有可能是 val/test 節點；新邊被明確設計成**單向**（`neighbor -> synthetic`,見 `blended_neighbor_sampling` 文件字串「never symmetrized」）,只把 val/test 的**特徵**單向餵進合成節點,不會讓合成節點的資訊流回 val/test 節點自身的 embedding——與所有 transductive GNN 本來就有的「train 節點聚合真實鄰居（可能是 val/test）」現象同一等級,不是 GraphENS 專屬的新增漏洞。

**現況總結**：`graph_smote_mask`、`GATEdgeGenerator`、`TargetedNeighbourhoodUndersampling` 三者的 k-NN/同質性搜尋現在**全部**正確限制在 train_mask 節點內。已被移除的 `reweighted_graph_smote_mask`/`unweighted_graph_smote_mask`（過去都有同樣的洩漏,前者的權重公式還退化成數學常數 `exp(-1)≈0.368`）已從 `evaluation.py` 整個刪除,不再是攻擊面。

---

## 4. 採樣技術：理論 vs. 程式碼實際實現

### 4.1 基礎 SMOTE / RUS
`smote_mask`（[evaluation.py:116-196](src/methods/evaluation.py#L116-L196)）直接呼叫 `imblearn.over_sampling.SMOTE.fit_resample`,教科書級別實作。`random_undersample_mask`（L66-114）用 `RandomState.choice` 無放回抽樣——與理論完全一致。

### 4.2 Vanilla Graph SMOTE
- **理論**（Zhao, Zhang & Wang, WSDM 2021）：SMOTE 插值生成節點後,訓練一個**邊生成器/解碼器**（對觀測鄰接矩陣做重構損失）,輸出連接機率,與分類器聯合訓練。
- **實作**（`graph_smote_mask`）：無可訓練邊生成器,用 k-NN（現已正確限制在 train）接邊,權重恆為 1（無 `edge_attr`）。是抓住「插值+接邊」精神的**一次性、非參數化啟發式**,與論文的可訓練解碼器機制無關。

### 4.3 GATSMOTE——真·可訓練 PyTorch 模組，且已依論文精修

`samplers.py` 的 `GATEdgeGenerator(nn.Module)`（[L16-395](src/methods/samplers.py#L16-L395)）實作 Liu et al., *GATSMOTE*（Mathematics 2022, DOI 10.3390/math10111799）。逐項核對：

- **可訓練權重矩陣/注意力向量**（[L46-47](src/methods/samplers.py#L46-L47) 附近）：`self.W = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(heads)])`、`self.a = nn.ModuleList([nn.Linear(2*hidden_dim, 1, bias=False) for _ in range(heads)])`——真正的 `nn.Parameter`,`nn.init.xavier_uniform_` 初始化。
- **多頭注意力**（`forward`, [L260-361](src/methods/samplers.py#L260-L361)）：`e = LeakyReLU(0.2)(a_h([Wz_i‖Wz_j]))`,`alpha = segment_softmax(e, dst_idx)`——標準 GAT 公式,逐目的節點（合成節點）正規化。
- **多頭融合（依論文精修）**：`logit_stack = torch.stack(head_logits, dim=-1)`（**未經 softmax 的原始分數 `e^{tk}`**）→ `edge_logits = self.fusion(logit_stack)`（可學習 `nn.Linear(heads,1)`）→ `edge_probs = sigmoid(edge_logits)`。這是刻意的設計決策：程式碼註解明確記錄「paper fuses the raw pre-normalization score e^{tk} (Eq. 8/Algorithm 1) before the separate softmax normalization step」——融合發生在 softmax **之前**,而非融合已正規化的注意力權重（這是團隊成員在合併我先前實作後,依論文校正的地方）。
- **局部性輔助損失**（paper Eq. 10 / Hypothesis 1）：
  $$\mathcal{L}_{\text{locality}} = -\text{mean}\big(2 \cdot E^t \cdot (\text{sim}_{\cos} - 0.5)\big)$$
  雙線性「推向極端」設計：高相似度的配對被推向 `E^t→1`,低相似度的推向 `E^t→0`,而非回歸到漸變的相似度值本身（先前版本用 MSE 回歸,已被此雙線性形式取代）。
- **最短路徑輔助損失**（paper Eq. 11 / Hypothesis 2）：透過有上限的 BFS（`_bounded_hop_distance`, [L107-129](src/methods/samplers.py#L107-L129)，`max_hops=4`）從合成節點的「SMOTE 母節點」（其在特徵空間中最近的真實少數類節點,`indices[0][0]`）出發計算跳數,再依同標籤配對的跳數給予獎勵：
  ```python
  distance_weight = clamp(pair_hop_distance / max_hops, 0, 1)
  per_pair_coeff = mismatch - match * distance_weight   # 同標籤且結構遠 → 推向 E^t=1；標籤不符 → 推向 E^t=0
  loss_shortest = (per_pair_coeff * edge_probs).mean()
  ```
  這比用二元標籤不符懲罰更貼近論文「縮短同標籤節點間路徑」的核心思想——用有界 BFS 跳數近似論文的精確路徑計數/矩陣冪運算,在大型稀疏 AML 圖上維持可計算性。

**已知限制（程式碼內明確記錄）**：在真實 HI-Small 圖（50 萬節點、63.1 萬邊）上量測,只有 0.33% 的配對能在 `max_hops=4` 內解出真實跳數；其餘配對中 99.11%（佔全部配對）根本與其母節點在**不同連通分量**（該圖有 37.4 萬個連通分量,其中 35.1 萬個是單點）,即無論跳數上限多大都不可達。機制對可解析的少數配對仍正確分級,對其餘配對方向性仍正確（把訊息傳遞碰不到的連接往上推）,但在此規模下對「遠」配對的解析度偏低——這不是跳數上限調錯,而是這張交易圖本身極度碎片化的結構性結果。

**聯合訓練**：見第 5 節。**結論**：這是貨真價實的可訓練模組,參數更新、多頭注意力（含論文精確的融合順序）、雙輔助損失反向傳播三者具足,且已針對論文公式做過校正。

### 4.4 GraphENS / Graph Ensemble SMOTE——真實實作，無死代碼

`sampling="graph_ensemble_smote"` 對應**真正的 GraphENS**（Park, Song & Yang, ICLR 2022 Algorithm 1）,由 [`graphens.py`](src/methods/graphens.py)（純函式）+ `GNN_features_graphens_with_predictions`（[experiments_supervised.py:1256 起](src/methods/experiments_supervised.py#L1256)）組成：

- **度數分佈對齊**：`graphens.sample_augmented_degree` 從真圖度數直方圖抽樣、夾到 `deg(v_minor)` 上限——確實實作。
- **混合自我網絡**：`blended_neighbor_sampling` 依 Eq.1 `p(u|v_mixed)=φ̂·p(u|v_minor)+(1-φ̂)·p(u|v_target)` 做無放回抽樣——確實實作。
- **KL 混合比 φ̂**、**顯著性遮罩 mixup**、**信心聚合 ô**（mean-then-softmax,文件字串明確記錄這是刻意選擇參考碼公式而非論文字面公式）——全部在 `graphens.py` 有實作且被 `train_epoch()` 呼叫,**沒有找到任何未被執行的死代碼**。

### 4.5 Targeted Neighbourhood Undersampling (TNU)

**判定噪聲節點的邏輯**（[samplers.py:409 起](src/methods/samplers.py#L409)）：對每個少數類節點,用 `NearestNeighbors(metric='cosine')` 找 k 個最近鄰,若某鄰居是多數類**且**cosine 距離 `>= noise_threshold`,標記為移除候選。

**CLI 參數傳遞——現已修復,兩條線職責分明**：外層 sweep 的 `ratio` 透過 `effective_remove_ratio = tnu_remove_ratio if tnu_remove_ratio is not None else ratio`（[experiments_supervised.py:986-987](src/methods/experiments_supervised.py#L986-L987)）預設驅動;`--tnu-k-neighbors`、`--tnu-distance-metric`、`--tnu-remove-ratio`、`--tnu-noise-threshold`、`--tnu-min-majority-keep`、`--tnu-preserve-minority-neighbors` 六個旗標全部透過 `_build_graphsmote_sampling` 傳進 `TargetedNeighbourhoodUndersampling(...)` 建構子,不再被靜默忽略。

### 4.6「Original」（ratio=None）語意跨方法一致性——已修復

`_build_graphsmote_sampling`（[L946-1007](src/methods/experiments_supervised.py#L946-L1007)）現在把 `if ratio is None:` 放在 if/elif 鏈最前面：
```python
if ratio is None:
    x_smote, y_smote, train_mask_smote, edge_index_smote = ntw_torch.x, ntw_torch.y, train_mask.bool(), ntw_torch.edge_index
elif sampling_name == "gatsmote":
    ...
```
`ratio=None`（「Original」列）現在對 `graph_smote`/`gatsmote`/`tnu` 全部代表「不重採樣」,與 `GNN_features`/`intrinsic_features` 等其他家族的既有語意（`elif sampling_name == "none" or ratio is None:`）一致。修復前,`graph_smote_mask(..., ratio=None,...)` 內部會 fallback 成 `target_minority_count=majority_count`（完全 1:1 平衡）,TNU 的 `remove_ratio=None` 則會移除**所有**偵測到的噪聲候選——兩者都違背「Original」應代表原生分佈的預期,已用實測驗證修復後 `y_true` 長度與未增強的 test set 完全一致。

---

## 5. 聯合訓練與優化機制

`GATEdgeGenerator` 是這份程式碼庫裡唯一具備自身可訓練參數的採樣器,`GNN_features_graphsmote(_with_predictions)`（[L1009-1255](src/methods/experiments_supervised.py#L1009-L1255)）為它接上完整的聯合訓練管線：

**優化器**：
```python
joint_params = list(model.parameters()) + list(gat_edge_gen.parameters()) if gat_edge_gen is not None else list(model.parameters())
optimizer = torch.optim.Adam(joint_params, lr=lr, weight_decay=5e-4)
```

**每個 epoch 的總損失公式**（`train_epoch`）：
```python
edge_index_epoch, edge_attr_epoch, loss_locality, loss_shortest = gat_edge_gen.build_epoch_graph(...)  # 用當下 W/a/fusion 重算
loss_node = _compute_loss(criterion, out[train_mask], y[train_mask])
loss_val = loss_node + gat_edge_gen.lambda_locality * loss_locality + gat_edge_gen.lambda_shortest * loss_shortest
loss_val.backward()
```
即 $\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{node}}+\lambda_1\cdot\mathcal{L}_{\text{locality}}+\lambda_2\cdot\mathcal{L}_{\text{shortest}}$,對應 CLI 的 `--gatsmote-lambda1`（預設 `0.2`）/`--gatsmote-lambda2`（預設 `0.05`）。**邊機率不是一次性算完存起來,而是每個 epoch 用當下的注意力參數重新計算**（`build_epoch_graph`,[samplers.py:363-395](src/methods/samplers.py#L363-L395)）;合成拓撲（哪些候選配對存在）則固定（`synthetic_pairs` 只在訓練開始前算一次,原因：候選池若每 epoch 重抽,mask 尺寸會不穩定）——固定候選集合 + 每 epoch 重算注意力權重,是「聯合訓練」在工程上唯一站得住腳的折衷。

**Early Stopping 與 Checkpoint 同步**：
```python
checkpoint_target = nn.ModuleList([model, gat_edge_gen]) if gat_edge_gen is not None else model
early_stopping(metric_to_monitor, checkpoint_target)         # 存整組模組的 state_dict
checkpoint_target.load_state_dict(torch.load(checkpoint_path, ...))   # 兩者一起回載
```
**這是先前版本一個真實存在過的 bug 的修復**：若只 `early_stopping(metric_to_monitor, model)`（只存分類器）,那麼「val 上最好的那一刻」的 `gat_edge_gen` 權重永遠不會被存下來——最終回載時,分類器是 val-best 版本,但 `gat_edge_gen` 卻是訓練跑到最後一刻的版本,兩者權重不同步,意味著 test 階段的邊機率是用「跟分類器從未真正搭配訓練過的一組注意力參數」算出來的。用 `nn.ModuleList([model, gat_edge_gen])` 包起來、整組存/取,徹底消除了這個風險。

**GraphENS 的對照**：`GNN_features_graphens_with_predictions` 的優化器只有 `model.parameters()`,因為 GraphENS 沒有自己的可訓練參數——它每 epoch 用的 `saliency`/`confidence` 是重用分類器自己這一步的梯度與輸出算出來的統計量（跨 epoch 傳遞的閉包狀態,非被優化的參數）,不需要、也沒有第二組權重要同步。這是這份程式碼庫裡「真正聯合訓練」與「動態但非聯合訓練」兩種範式的清楚分野。

---

## 6. 分類器架構與超參數配置

### 6.1 結構拆解

| | GCN | GraphSAGE | GAT | GIN |
|---|---|---|---|---|
| 層數 | 2 | 2 | 2 | 2 |
| hidden_dim | 64 | 64 | 64（輕量模式 16） | 64 |
| embedding_dim | 32 | 32 | 32（輕量模式 8） | 32 |
| output_dim | 2 | 2 | 2 | 2 |
| heads | — | — | 4（輕量模式 1） | — |
| dropout_rate | 0.3 | 0.3 | 0.3（輕量模式 0.2） | 0.3 |
| 激活函式 | `F.relu` | `F.relu` | `F.relu` | 層內建 `ReLU` |
| 輸出頭 | `Decoder_linear`（單層 Linear） | 同左 | 同左 | 同左 |

「輕量模式」（`use_lightweight_gat = num_nodes >= 300000`,`train_supervised_tuned.py`）只影響 GAT,連帶把 `gnn_epochs` 從 50 降到 10——GCN/SAGE/GIN 在大圖上完全不受影響。

### 6.2 edge_attr 處理

見 §1.3 表格。GAT 現在是**唯一**在架構層面真正利用注意力重新加權邊資訊的模型（GCN 用邊權重做線性加權平均,語意較弱）。GraphSAGE/GIN 的「不支援」是原始論文公式本身沒有邊特徵概念,不是實作疏漏——`GNN.py` 對這兩處都有明確註解說明原因。

### 6.3 CLI 超參數完整清單（`train_supervised_tuned.py`）

**不平衡比例掃描網格**：
```python
test_ratios = [None, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
```
語意：`target_minority_count = round(majority_count / ratio)`,`ratio` 是「多數:少數」目標比。`ratio=None`（Original）現在對**所有**採樣技術家族一致代表「不重採樣」（見 §4.6）。

**梯度裁剪**：`--clip_norm`（預設 `1.0`）真正傳進所有 `GNN_features*` 函式的 `clip_norm` 參數,取代先前硬編碼的 `max_norm=1.0`。

**損失函式公式**：
- **Cross-Entropy**：`nn.CrossEntropyLoss(weight=[1.0, pos_weight])`,`pos_weight = num_neg/max(num_pos,1)`,依當下子集動態算。
- **Focal Loss**（[losses.py:39-79](src/methods/losses.py#L39-L79)）：
  $$FL(p_t)=-\alpha_t(1-p_t)^{\gamma}\log(p_t)$$
  `gamma` 預設 2.0,`alpha` 支援 `None`/`'balanced'`/純量/`list`。**已修復的崩潰 bug**：所有 GNN 相關函式呼叫 `_compute_loss` 時,`out`/`y` 早已被 `train_mask` 篩選過,舊版又把篩選前的同一個 mask 重複傳進 `FocalLoss.forward(..., mask=...)`,觸發其內部 shape 檢查、必定 `raise ValueError`。修法是不再對已篩選的張量重複傳 `mask`——現在 `--loss focal` 搭配任何 GNN 方法都能正常訓練（已實測驗證,修前必炸、修後產出正常 loss 曲線）。

**GATSMOTE CLI**（`train_supervised_tuned.py`）：`--gatsmote-k-neighbors`（5）、`--gatsmote-heads`（4）、`--gatsmote-edge-threshold`（0.5）、`--gatsmote-lambda1`（0.2）、`--gatsmote-lambda2`（0.05）、`--gatsmote-use-predicted-labels`——全部正確傳遞（過去 `--gatsmote-attention-heads`/`--gatsmote-homophily-weight` 兩個舊名稱已改成上述新名稱,對應真實可訓練模組的參數）。

**TNU/GraphENS CLI**：見 §4.5、README「GraphENS」章節,全部有效傳遞。

### 6.4 已知限制（程式碼內明確記錄，非本次審計臆測）

1. **`--gatsmote-lambda1`/`--gatsmote-lambda2` 的預設值是在小型、良好縮放的合成圖上調的**,未在真實 IBM/Elliptic 資料的生產規模上驗證過。對真實 `hi_small` 的驗證嘗試產生了不可用的結果（`loss_node` 爆炸到數百萬量級）——這個爆炸是 §1.1 提到的 `Amount Received`/`Amount Paid`（未縮放,最高可達 `~2.8e11`）造成的產物,不是損失項本身的問題。待 IBM 特徵標準化問題解決後,這組預設值需要重新調校。
2. **最短路徑輔助損失在真實稀疏 AML 圖上解析度有限**（見 §4.3 已知限制）。
3. **GraphSAGE 的鄰域採樣需要 `pyg-lib` 或 `torch-sparse`** 搭配當前平台可用的編譯後端。兩者都列在 `requirements.txt`,但若平台沒有預編譯 wheel,訓練會靜默降級成全圖前向（會印出警告）。

---

## 7. 特徵空間不匹配與陣列截斷風險

### 7.1 SMOTE 家族附加合成節點後的形狀變化

`graph_smote_mask`/`GATEdgeGenerator.prepare_synthetic_nodes` 都遵循同一模式：合成節點永遠 append 在原始特徵矩陣尾端（`vstack/cat([原始N筆, 合成M筆])`）,`expanded_mask[:N]=原mask`,`expanded_mask[N:]=True`（全部算進 train）。

### 7.2 `X_val`/`X_test` 的切片邏輯——兩種策略,經查證均對稱

**策略 A（圖結構方法,`GNN_features_graphsmote*`）**：padding。`n_synthetic = x_smote.shape[0] - ntw_torch.x.shape[0]`;`val_mask_smote`/`test_mask_smote` 用完全相同的寫法、同一個 `n_synthetic`、緊鄰兩行程式碼補 `torch.zeros(n_synthetic, dtype=torch.bool)`——沒有「test 有保護、val 沒有」的不對稱。

**策略 B（純特徵空間方法,`intrinsic/positional/node2vec_features*`）**：截斷。`features_tensor[:val_mask.shape[0]]`/`[:test_mask.shape[0]]` 先把可能變長的張量切回原始長度 N,再套用原長度的 mask——`X_val` 與 `X_test` 現在**都**採用這個寫法（node2vec 修復後,兩者寫法完全對稱）。

### 7.3 `GATEdgeGenerator.build_epoch_graph` 的形狀安全

```python
keep = edge_probs.detach() >= self.edge_threshold
kept_pairs = synthetic_pairs[:, keep]; kept_weights = edge_probs[keep]
dynamic_edge_index = torch.cat([kept_pairs, kept_pairs.flip(0)], dim=1)
full_edge_index = torch.cat([base_edge_index, dynamic_edge_index], dim=1)
full_edge_attr = torch.cat([base_weight, dynamic_edge_weight], dim=0)
```
`keep` 全 False 時（訓練極早期,注意力尚未學到有意義的信號）,`dynamic_edge_index`/`dynamic_edge_weight` 會是空張量,`torch.cat` 對空張量與非空張量串接是良定義操作——臨時退化成「沒有合成邊」的圖,分類損失仍可正常計算,`loss_locality`/`loss_shortest`（仍對全部 `synthetic_pairs` 算,不受 `keep` 篩選影響）繼續提供梯度訊號,把注意力參數推向產生更有意義的 `E^t`。這是一個對「訓練初期沒有任何邊通過閾值」這個邊界情況天然穩健的設計。

### 7.4 `generate_all_tables.py` 的多 seed 聚合形狀安全

`scripts/generate_all_tables.py` 現在把每個 (metric, dataset, sampling, method, ratio) cell 存成 `{seed_key: value}` 字典而非單一純量,`aggregate_cell()` 對 `values_by_seed.values()` 算 `statistics.mean`/`statistics.stdev`（樣本標準差,ddof=1,僅在 n>=2 時計算,n=1 時顯示裸值）。已用合成的 3-seed 檔案實測驗證：`[0.50, 0.52, 0.48]` 正確算出 `0.5000 ± 0.0200`,並在真實資料上重新產出過 `tables/tuned_only_ratio_comparison_*.md`（含新增的 `F1_90` 表）。

**現況總結**：本節先前稽核提出的「node2vec 缺少 X_val 保護」問題已隨第 2 節的修復一併消失（現在有 X_val,且截斷邏輯與 X_test 對稱）;圖結構方法的 padding 邏輯本來就對稱,沒有新增風險;新增的多 seed 聚合邏輯也已驗證形狀安全。

---

## 專業見解總結（跨章節觀察）

1. **transductive 全圖設定是這整份 benchmark 的根本前提**,§1.3/§2.2/§3.2 反覆出現的「train 節點的 embedding 會聚合到 val/test 真實鄰居特徵」現象,源自「一張圖、mask 切分」的架構選擇,而非任何取樣器的個別缺陷。
2. **GATSMOTE 現在是方法論上最貼近原論文的實作**,但也是訓練成本最高、超參數最多的技術——建議調參實驗設計上為它分配更多隨機種子才能得到穩定結論。
3. **IBM 資料集的特徵標準化缺口是一個橫跨多個模組的根源問題**：它不只讓 GATSMOTE 的損失爆炸（§6.4 第 1 點）,任何依賴梯度尺度穩定性的元件（例如 Focal Loss 的 `(1-p_t)^γ` 項、GATEdgeGenerator 的雙輔助損失權重）在 IBM 資料集上都可能有類似的未驗證風險,值得列為下一輪修復的優先項目,而不是只在 GATSMOTE 這個單點修。
4. **GraphSAGE 的 mini-batch 訓練與 GCN/GAT/GIN 的全圖訓練並存**,是目前跨架構比較時必須留意的計算路徑差異：SAGE 看到的「一個 epoch」定義（多個 mini-batch 梯度更新）與其他三者（單一全圖梯度更新）不同,在解讀「相同 epoch 數下哪個架構收斂較快」這類比較時需要謹慎。
