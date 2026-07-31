Method X Sampling Table (LR=0.001, Gradient Clipping=1.0)

Cells aggregated across multiple --seed runs are shown as mean ± sample std (ddof=1); cells backed by a single run show a bare value. Bold marks the best mean in each column.

## Original

| Method | Sampling | ELLIPTIC | IBM HI-SMALL | IBM HI-MEDIUM | IBM LI-SMALL | IBM LI-MEDIUM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **INTRINSIC** | None (Baseline) | 0.2914 | - | **0.0025** | - | **0.0014** |
|  | RUS | 0.2914 | - | **0.0025** | - | **0.0014** |
|  | SMOTE | **0.2941** | - | **0.0025** | - | **0.0014** |
| | | | | | | |
| **POSITIONAL** | None (Baseline) | **0.1076** | - | - | - | - |
|  | RUS | **0.1076** | - | - | - | - |
|  | SMOTE | 0.0000 | - | - | - | - |
| | | | | | | |
| **DEEPWALK** | None (Baseline) | **0.1230** | - | - | - | - |
|  | RUS | 0.0535 | - | - | - | - |
|  | SMOTE | 0.0695 | - | - | - | - |
| | | | | | | |
| **NODE2VEC** | None (Baseline) | **0.0802** | - | - | - | - |
|  | RUS | **0.0802** | - | - | - | - |
|  | SMOTE | 0.0321 | - | - | - | - |
| | | | | | | |
| **GCN** | None (Baseline) | 0.0856 | - | - | - | - |
|  | RUS | 0.1257 | - | - | - | - |
|  | GRAPH_SMOTE | 0.0829 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.0401 | - | - | - | - |
|  | GATSMOTE | 0.0267 | - | - | - | - |
|  | TNU | **0.1337** | - | - | - | - |
| | | | | | | |
| **SAGE** | None (Baseline) | 0.2139 | - | - | - | - |
|  | RUS | 0.2647 | - | - | - | - |
|  | GRAPH_SMOTE | **0.2701** | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.1150 | - | - | - | - |
|  | GATSMOTE | 0.2326 | - | - | - | - |
|  | TNU | 0.1925 | - | - | - | - |
| | | | | | | |
| **GAT** | None (Baseline) | 0.2807 | - | - | - | - |
|  | RUS | 0.2487 | - | - | - | - |
|  | GRAPH_SMOTE | 0.2620 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2807 | - | - | - | - |
|  | GATSMOTE | 0.2674 | - | - | - | - |
|  | TNU | **0.2861** | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | 0.2727 | - | - | - | - |
|  | RUS | 0.2513 | - | - | - | - |
|  | GRAPH_SMOTE | 0.2781 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2380 | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | **0.2861** | - | - | - | - |
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
| **INTRINSIC** | None (Baseline) | **0.2941** | - | - | - | - |
|  | RUS | 0.2914 | - | - | - | - |
|  | SMOTE | **0.2941** | - | - | - | - |
| | | | | | | |
| **POSITIONAL** | None (Baseline) | **0.1076** | - | - | - | - |
|  | RUS | **0.1076** | - | - | - | - |
|  | SMOTE | **0.1076** | - | - | - | - |
| | | | | | | |
| **DEEPWALK** | None (Baseline) | **0.0963** | - | - | - | - |
|  | RUS | 0.0802 | - | - | - | - |
|  | SMOTE | 0.0428 | - | - | - | - |
| | | | | | | |
| **NODE2VEC** | None (Baseline) | **0.1096** | - | - | - | - |
|  | RUS | 0.0936 | - | - | - | - |
|  | SMOTE | 0.0989 | - | - | - | - |
| | | | | | | |
| **GCN** | None (Baseline) | 0.0882 | - | - | - | - |
|  | RUS | 0.0989 | - | - | - | - |
|  | GRAPH_SMOTE | 0.0348 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.1150 | - | - | - | - |
|  | GATSMOTE | **0.1203** | - | - | - | - |
|  | TNU | 0.1016 | - | - | - | - |
| | | | | | | |
| **SAGE** | None (Baseline) | 0.2647 | - | - | - | - |
|  | RUS | 0.2433 | - | - | - | - |
|  | GRAPH_SMOTE | 0.2781 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2353 | - | - | - | - |
|  | GATSMOTE | 0.2487 | - | - | - | - |
|  | TNU | **0.2807** | - | - | - | - |
| | | | | | | |
| **GAT** | None (Baseline) | 0.2032 | - | - | - | - |
|  | RUS | **0.2861** | - | - | - | - |
|  | GRAPH_SMOTE | 0.2754 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2567 | - | - | - | - |
|  | GATSMOTE | 0.2754 | - | - | - | - |
|  | TNU | 0.2754 | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | 0.2674 | - | - | - | - |
|  | RUS | 0.2594 | - | - | - | - |
|  | GRAPH_SMOTE | 0.2193 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | **0.2807** | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | 0.2754 | - | - | - | - |
| | | | | | | |

---

## 1:2 (Ratio)

| Method | Sampling | ELLIPTIC | IBM HI-SMALL | IBM HI-MEDIUM | IBM LI-SMALL | IBM LI-MEDIUM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **INTRINSIC** | None (Baseline) | **0.2914** | - | - | - | - |
|  | RUS | **0.2914** | - | - | - | - |
|  | SMOTE | **0.2914** | - | - | - | - |
| | | | | | | |
| **POSITIONAL** | None (Baseline) | **0.1076** | - | - | - | - |
|  | RUS | **0.1076** | - | - | - | - |
|  | SMOTE | 0.1019 | - | - | - | - |
| | | | | | | |
| **DEEPWALK** | None (Baseline) | 0.0428 | - | - | - | - |
|  | RUS | **0.0989** | - | - | - | - |
|  | SMOTE | 0.0588 | - | - | - | - |
| | | | | | | |
| **NODE2VEC** | None (Baseline) | **0.1471** | - | - | - | - |
|  | RUS | 0.0642 | - | - | - | - |
|  | SMOTE | 0.0535 | - | - | - | - |
| | | | | | | |
| **GCN** | None (Baseline) | 0.0775 | - | - | - | - |
|  | RUS | **0.1096** | - | - | - | - |
|  | GRAPH_SMOTE | 0.0963 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.0775 | - | - | - | - |
|  | GATSMOTE | 0.1070 | - | - | - | - |
|  | TNU | 0.0455 | - | - | - | - |
| | | | | | | |
| **SAGE** | None (Baseline) | **0.2620** | - | - | - | - |
|  | RUS | 0.2513 | - | - | - | - |
|  | GRAPH_SMOTE | 0.1845 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.1283 | - | - | - | - |
|  | GATSMOTE | 0.1818 | - | - | - | - |
|  | TNU | 0.1551 | - | - | - | - |
| | | | | | | |
| **GAT** | None (Baseline) | **0.2941** | - | - | - | - |
|  | RUS | **0.2941** | - | - | - | - |
|  | GRAPH_SMOTE | 0.2112 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2781 | - | - | - | - |
|  | GATSMOTE | 0.1203 | - | - | - | - |
|  | TNU | 0.2914 | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | 0.2701 | - | - | - | - |
|  | RUS | 0.2647 | - | - | - | - |
|  | GRAPH_SMOTE | 0.1283 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2674 | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | **0.2861** | - | - | - | - |
| | | | | | | |

---

## 1:1 (Ratio)

| Method | Sampling | ELLIPTIC | IBM HI-SMALL | IBM HI-MEDIUM | IBM LI-SMALL | IBM LI-MEDIUM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **INTRINSIC** | None (Baseline) | 0.2914 | - | - | - | - |
|  | RUS | **0.2941** | - | - | - | - |
|  | SMOTE | 0.2914 | - | - | - | - |
| | | | | | | |
| **POSITIONAL** | None (Baseline) | **0.1076** | - | - | - | - |
|  | RUS | **0.1076** | - | - | - | - |
|  | SMOTE | 0.0000 | - | - | - | - |
| | | | | | | |
| **DEEPWALK** | None (Baseline) | 0.0428 | - | - | - | - |
|  | RUS | 0.0936 | - | - | - | - |
|  | SMOTE | **0.1016** | - | - | - | - |
| | | | | | | |
| **NODE2VEC** | None (Baseline) | **0.1765** | - | - | - | - |
|  | RUS | 0.1043 | - | - | - | - |
|  | SMOTE | 0.0775 | - | - | - | - |
| | | | | | | |
| **GCN** | None (Baseline) | 0.0775 | - | - | - | - |
|  | RUS | 0.0882 | - | - | - | - |
|  | GRAPH_SMOTE | 0.0642 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.0348 | - | - | - | - |
|  | GATSMOTE | 0.0749 | - | - | - | - |
|  | TNU | **0.1578** | - | - | - | - |
| | | | | | | |
| **SAGE** | None (Baseline) | **0.2754** | - | - | - | - |
|  | RUS | 0.2674 | - | - | - | - |
|  | GRAPH_SMOTE | 0.1684 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.1337 | - | - | - | - |
|  | GATSMOTE | 0.1898 | - | - | - | - |
|  | TNU | 0.2005 | - | - | - | - |
| | | | | | | |
| **GAT** | None (Baseline) | **0.2888** | - | - | - | - |
|  | RUS | 0.2861 | - | - | - | - |
|  | GRAPH_SMOTE | 0.2273 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2540 | - | - | - | - |
|  | GATSMOTE | 0.1310 | - | - | - | - |
|  | TNU | 0.2727 | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | **0.2727** | - | - | - | - |
|  | RUS | 0.2620 | - | - | - | - |
|  | GRAPH_SMOTE | 0.1283 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2433 | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | 0.2674 | - | - | - | - |
| | | | | | | |

---
