Method X Sampling Table (LR=0.001, Gradient Clipping=1.0)

Cells aggregated across multiple --seed runs are shown as mean ± sample std (ddof=1); cells backed by a single run show a bare value. Bold marks the best mean in each column.

## Original

| Method | Sampling | ELLIPTIC | IBM HI-SMALL | IBM HI-MEDIUM | IBM LI-SMALL | IBM LI-MEDIUM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **INTRINSIC** | None (Baseline) | 0.5766 | - | **0.0012** | - | **0.0007** |
|  | RUS | 0.5828 | - | **0.0012** | - | **0.0007** |
|  | SMOTE | **0.5871** | - | **0.0012** | - | **0.0007** |
| | | | | | | |
| **POSITIONAL** | None (Baseline) | 0.0569 | - | - | - | - |
|  | RUS | 0.0569 | - | - | - | - |
|  | SMOTE | **0.0752** | - | - | - | - |
| | | | | | | |
| **DEEPWALK** | None (Baseline) | **0.1780** | - | - | - | - |
|  | RUS | 0.1201 | - | - | - | - |
|  | SMOTE | 0.1281 | - | - | - | - |
| | | | | | | |
| **NODE2VEC** | None (Baseline) | **0.1529** | - | - | - | - |
|  | RUS | 0.1288 | - | - | - | - |
|  | SMOTE | 0.0822 | - | - | - | - |
| | | | | | | |
| **GCN** | None (Baseline) | 0.1963 | - | - | - | - |
|  | RUS | 0.2326 | - | - | - | - |
|  | GRAPH_SMOTE | 0.2000 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.1692 | - | - | - | - |
|  | GATSMOTE | 0.1362 | - | - | - | - |
|  | TNU | **0.2713** | - | - | - | - |
| | | | | | | |
| **SAGE** | None (Baseline) | 0.3735 | - | - | - | - |
|  | RUS | **0.4144** | - | - | - | - |
|  | GRAPH_SMOTE | 0.3892 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2688 | - | - | - | - |
|  | GATSMOTE | 0.3662 | - | - | - | - |
|  | TNU | 0.3772 | - | - | - | - |
| | | | | | | |
| **GAT** | None (Baseline) | **0.4795** | - | - | - | - |
|  | RUS | 0.4044 | - | - | - | - |
|  | GRAPH_SMOTE | 0.4518 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.4791 | - | - | - | - |
|  | GATSMOTE | 0.4101 | - | - | - | - |
|  | TNU | 0.4716 | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | 0.4146 | - | - | - | - |
|  | RUS | **0.4311** | - | - | - | - |
|  | GRAPH_SMOTE | 0.3854 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.4211 | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | 0.4230 | - | - | - | - |
| | | | | | | |

---

## 1:100 (Ratio)

| Method | Sampling | ELLIPTIC | IBM HI-SMALL | IBM HI-MEDIUM | IBM LI-SMALL | IBM LI-MEDIUM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **INTRINSIC** | None (Baseline) | - | - | - | - | - |
|  | RUS | - | - | - | - | - |
|  | SMOTE | - | - | - | - | - |
| | | | | | | |
| **POSITIONAL** | None (Baseline) | - | - | - | - | - |
|  | RUS | - | - | - | - | - |
|  | SMOTE | - | - | - | - | - |
| | | | | | | |
| **DEEPWALK** | None (Baseline) | - | - | - | - | - |
|  | RUS | - | - | - | - | - |
|  | SMOTE | - | - | - | - | - |
| | | | | | | |
| **NODE2VEC** | None (Baseline) | - | - | - | - | - |
|  | RUS | - | - | - | - | - |
|  | SMOTE | - | - | - | - | - |
| | | | | | | |
| **GCN** | None (Baseline) | - | - | - | - | - |
|  | RUS | - | - | - | - | - |
|  | GRAPH_SMOTE | - | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | - | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | - | - | - | - | - |
| | | | | | | |
| **SAGE** | None (Baseline) | - | - | - | - | - |
|  | RUS | - | - | - | - | - |
|  | GRAPH_SMOTE | - | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | - | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | - | - | - | - | - |
| | | | | | | |
| **GAT** | None (Baseline) | - | - | - | - | - |
|  | RUS | - | - | - | - | - |
|  | GRAPH_SMOTE | - | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | - | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | - | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | - | - | - | - | - |
|  | RUS | - | - | - | - | - |
|  | GRAPH_SMOTE | - | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | - | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | - | - | - | - | - |
| | | | | | | |

---

## 1:10 (Ratio)

| Method | Sampling | ELLIPTIC | IBM HI-SMALL | IBM HI-MEDIUM | IBM LI-SMALL | IBM LI-MEDIUM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **INTRINSIC** | None (Baseline) | 0.5884 | - | - | - | - |
|  | RUS | 0.5875 | - | - | - | - |
|  | SMOTE | **0.5933** | - | - | - | - |
| | | | | | | |
| **POSITIONAL** | None (Baseline) | **0.0569** | - | - | - | - |
|  | RUS | **0.0569** | - | - | - | - |
|  | SMOTE | **0.0569** | - | - | - | - |
| | | | | | | |
| **DEEPWALK** | None (Baseline) | **0.1480** | - | - | - | - |
|  | RUS | 0.1231 | - | - | - | - |
|  | SMOTE | 0.1265 | - | - | - | - |
| | | | | | | |
| **NODE2VEC** | None (Baseline) | **0.1690** | - | - | - | - |
|  | RUS | 0.1416 | - | - | - | - |
|  | SMOTE | 0.1649 | - | - | - | - |
| | | | | | | |
| **GCN** | None (Baseline) | 0.2080 | - | - | - | - |
|  | RUS | 0.2104 | - | - | - | - |
|  | GRAPH_SMOTE | 0.1599 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2097 | - | - | - | - |
|  | GATSMOTE | **0.2293** | - | - | - | - |
|  | TNU | 0.2212 | - | - | - | - |
| | | | | | | |
| **SAGE** | None (Baseline) | 0.4179 | - | - | - | - |
|  | RUS | 0.4070 | - | - | - | - |
|  | GRAPH_SMOTE | **0.4834** | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.3824 | - | - | - | - |
|  | GATSMOTE | 0.4391 | - | - | - | - |
|  | TNU | 0.4453 | - | - | - | - |
| | | | | | | |
| **GAT** | None (Baseline) | 0.2984 | - | - | - | - |
|  | RUS | **0.4982** | - | - | - | - |
|  | GRAPH_SMOTE | 0.4544 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.4507 | - | - | - | - |
|  | GATSMOTE | 0.4371 | - | - | - | - |
|  | TNU | 0.4735 | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | 0.4336 | - | - | - | - |
|  | RUS | 0.4039 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3452 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | **0.4440** | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | 0.4375 | - | - | - | - |
| | | | | | | |

---

## 1:2 (Ratio)

| Method | Sampling | ELLIPTIC | IBM HI-SMALL | IBM HI-MEDIUM | IBM LI-SMALL | IBM LI-MEDIUM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **INTRINSIC** | None (Baseline) | 0.5823 | - | - | - | - |
|  | RUS | 0.5732 | - | - | - | - |
|  | SMOTE | **0.5851** | - | - | - | - |
| | | | | | | |
| **POSITIONAL** | None (Baseline) | 0.0569 | - | - | - | - |
|  | RUS | 0.0569 | - | - | - | - |
|  | SMOTE | **0.0920** | - | - | - | - |
| | | | | | | |
| **DEEPWALK** | None (Baseline) | 0.1060 | - | - | - | - |
|  | RUS | **0.1430** | - | - | - | - |
|  | SMOTE | 0.1127 | - | - | - | - |
| | | | | | | |
| **NODE2VEC** | None (Baseline) | **0.2263** | - | - | - | - |
|  | RUS | 0.1265 | - | - | - | - |
|  | SMOTE | 0.1041 | - | - | - | - |
| | | | | | | |
| **GCN** | None (Baseline) | 0.1871 | - | - | - | - |
|  | RUS | **0.2228** | - | - | - | - |
|  | GRAPH_SMOTE | 0.1983 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.1890 | - | - | - | - |
|  | GATSMOTE | 0.2027 | - | - | - | - |
|  | TNU | 0.1720 | - | - | - | - |
| | | | | | | |
| **SAGE** | None (Baseline) | 0.4159 | - | - | - | - |
|  | RUS | **0.4343** | - | - | - | - |
|  | GRAPH_SMOTE | 0.2679 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2087 | - | - | - | - |
|  | GATSMOTE | 0.2474 | - | - | - | - |
|  | TNU | 0.3346 | - | - | - | - |
| | | | | | | |
| **GAT** | None (Baseline) | **0.5044** | - | - | - | - |
|  | RUS | 0.4871 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3362 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.4531 | - | - | - | - |
|  | GATSMOTE | 0.1993 | - | - | - | - |
|  | TNU | 0.4731 | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | 0.4131 | - | - | - | - |
|  | RUS | 0.4166 | - | - | - | - |
|  | GRAPH_SMOTE | 0.1986 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | **0.4868** | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | 0.4558 | - | - | - | - |
| | | | | | | |

---

## 1:1 (Ratio)

| Method | Sampling | ELLIPTIC | IBM HI-SMALL | IBM HI-MEDIUM | IBM LI-SMALL | IBM LI-MEDIUM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **INTRINSIC** | None (Baseline) | **0.5885** | - | - | - | - |
|  | RUS | 0.5882 | - | - | - | - |
|  | SMOTE | 0.5880 | - | - | - | - |
| | | | | | | |
| **POSITIONAL** | None (Baseline) | **0.0569** | - | - | - | - |
|  | RUS | **0.0569** | - | - | - | - |
|  | SMOTE | 0.0469 | - | - | - | - |
| | | | | | | |
| **DEEPWALK** | None (Baseline) | 0.1169 | - | - | - | - |
|  | RUS | **0.1532** | - | - | - | - |
|  | SMOTE | 0.1405 | - | - | - | - |
| | | | | | | |
| **NODE2VEC** | None (Baseline) | **0.2535** | - | - | - | - |
|  | RUS | 0.1613 | - | - | - | - |
|  | SMOTE | 0.1252 | - | - | - | - |
| | | | | | | |
| **GCN** | None (Baseline) | 0.2318 | - | - | - | - |
|  | RUS | 0.1974 | - | - | - | - |
|  | GRAPH_SMOTE | 0.1870 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.1257 | - | - | - | - |
|  | GATSMOTE | 0.1760 | - | - | - | - |
|  | TNU | **0.2802** | - | - | - | - |
| | | | | | | |
| **SAGE** | None (Baseline) | **0.4413** | - | - | - | - |
|  | RUS | 0.3958 | - | - | - | - |
|  | GRAPH_SMOTE | 0.2246 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2933 | - | - | - | - |
|  | GATSMOTE | 0.2520 | - | - | - | - |
|  | TNU | 0.3752 | - | - | - | - |
| | | | | | | |
| **GAT** | None (Baseline) | **0.4904** | - | - | - | - |
|  | RUS | 0.4860 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3317 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.3467 | - | - | - | - |
|  | GATSMOTE | 0.2108 | - | - | - | - |
|  | TNU | 0.4758 | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | **0.4592** | - | - | - | - |
|  | RUS | 0.3957 | - | - | - | - |
|  | GRAPH_SMOTE | 0.2002 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.3851 | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | 0.4059 | - | - | - | - |
| | | | | | | |

---
