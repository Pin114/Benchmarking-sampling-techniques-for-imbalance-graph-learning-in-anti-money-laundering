# Resampling Ratio Impact Analysis Matrix (AUC-PRC)

## 📁 Dataset: HI_SMALL
### ⚙️ Sampling Technique: GRAPH_ENSEMBLE_SMOTE
橫向對比不同的不平衡重採樣比例（Resampling Ratios）對模型最終泛化表現的實質影響：

| Method / Baseline | Imbalance original | Imbalance ratio_1to10 | Imbalance ratio_1to2 | Imbalance ratio_1to1 |
| --- | --- | --- | --- | --- |
| **DEEPWALK** | N/A | N/A | 0.003530 ± 0.000618 | 0.005938 ± 0.001740 |
| **GAT** | 0.004770 ± 0.000232 | N/A | 0.004908 ± 0.000777 | 0.004445 ± 0.000073 |
| **GCN** | 0.004723 ± 0.000459 | N/A | 0.004243 ± 0.000339 | 0.004307 ± 0.000129 |
| **GIN** | 0.004348 ± 0.001045 | N/A | 0.003770 ± 0.000152 | 0.003766 ± 0.000078 |
| **NODE2VEC** | N/A | N/A | 0.005724 ± 0.000125 | 0.006680 ± 0.001123 |
| **SAGE** | 0.004388 ± 0.000362 | N/A | 0.004666 ± 0.000174 | 0.004651 ± 0.000110 |

---

### ⚙️ Sampling Technique: GRAPH_SMOTE
橫向對比不同的不平衡重採樣比例（Resampling Ratios）對模型最終泛化表現的實質影響：

| Method / Baseline | Imbalance original | Imbalance ratio_1to10 | Imbalance ratio_1to2 | Imbalance ratio_1to1 |
| --- | --- | --- | --- | --- |
| **DEEPWALK** | N/A | N/A | 0.005625 ± 0.001287 | 0.022907 ± 0.026274 |
| **GAT** | 0.004076 ± 0.000233 | N/A | 0.004426 ± 0.000351 | 0.004647 ± 0.000165 |
| **GCN** | 0.004145 ± 0.000193 | N/A | 0.004440 ± 0.000098 | 0.004451 ± 0.000088 |
| **GIN** | 0.068392 ± 0.091801 | N/A | 0.003717 ± 0.000053 | 0.003859 ± 0.000091 |
| **NODE2VEC** | N/A | N/A | 0.008936 ± 0.004968 | 0.005327 ± 0.000818 |
| **SAGE** | 0.004490 ± 0.000199 | N/A | 0.004114 ± 0.000341 | 0.004249 ± 0.000344 |

---

### ⚙️ Sampling Technique: NONE
橫向對比不同的不平衡重採樣比例（Resampling Ratios）對模型最終泛化表現的實質影響：

| Method / Baseline | Imbalance original | Imbalance ratio_1to10 | Imbalance ratio_1to2 | Imbalance ratio_1to1 |
| --- | --- | --- | --- | --- |
| **DEEPWALK** | N/A | N/A | 0.004843 ± 0.000370 | 0.005486 ± 0.000477 |
| **GAT** | 0.003772 ± 0.000006 | N/A | 0.003780 ± 0.000000 | 0.003780 ± 0.000000 |
| **GCN** | 0.004753 ± 0.000935 | N/A | 0.004491 ± 0.000315 | 0.004506 ± 0.000784 |
| **GIN** | 0.004733 ± 0.000191 | N/A | 0.004001 ± 0.000386 | 0.003604 ± 0.000018 |
| **INTRINSIC** | N/A | N/A | 0.005528 ± 0.000401 | 0.006191 ± 0.000761 |
| **NODE2VEC** | N/A | N/A | 0.004581 ± 0.000274 | 0.004845 ± 0.000218 |
| **POSITIONAL** | N/A | N/A | 0.003921 ± 0.000425 | 0.003701 ± 0.000016 |
| **SAGE** | 0.004315 ± 0.000228 | N/A | 0.004633 ± 0.000542 | 0.004186 ± 0.000049 |

---

### ⚙️ Sampling Technique: RUS
橫向對比不同的不平衡重採樣比例（Resampling Ratios）對模型最終泛化表現的實質影響：

| Method / Baseline | Imbalance original | Imbalance ratio_1to10 | Imbalance ratio_1to2 | Imbalance ratio_1to1 |
| --- | --- | --- | --- | --- |
| **DEEPWALK** | N/A | N/A | 0.004038 ± 0.000540 | 0.005871 ± 0.001029 |
| **GAT** | 0.004557 ± 0.000133 | N/A | 0.003618 ± 0.000437 | 0.004318 ± 0.000354 |
| **GCN** | 0.004729 ± 0.000199 | N/A | 0.004691 ± 0.000486 | 0.004452 ± 0.000088 |
| **GIN** | 0.004358 ± 0.000471 | N/A | 0.003868 ± 0.000064 | 0.003862 ± 0.000063 |
| **INTRINSIC** | N/A | N/A | 0.004518 ± 0.000553 | 0.004445 ± 0.000435 |
| **NODE2VEC** | N/A | N/A | 0.005512 ± 0.000755 | 0.004494 ± 0.000913 |
| **POSITIONAL** | N/A | N/A | 0.003689 ± 0.000048 | 0.003804 ± 0.000113 |
| **SAGE** | 0.004531 ± 0.000169 | N/A | 0.004711 ± 0.000019 | 0.004864 ± 0.000202 |

---

### ⚙️ Sampling Technique: SMOTE
橫向對比不同的不平衡重採樣比例（Resampling Ratios）對模型最終泛化表現的實質影響：

| Method / Baseline | Imbalance original | Imbalance ratio_1to10 | Imbalance ratio_1to2 | Imbalance ratio_1to1 |
| --- | --- | --- | --- | --- |
| **GIN** | 0.004479 ± 0.000613 | N/A | N/A | N/A |
| **INTRINSIC** | N/A | N/A | 0.004714 ± 0.000610 | 0.003932 ± 0.000017 |
| **POSITIONAL** | N/A | N/A | 0.004081 ± 0.000167 | 0.003927 ± 0.000051 |

---

