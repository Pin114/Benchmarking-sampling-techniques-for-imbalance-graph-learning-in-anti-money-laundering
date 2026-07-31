Method X Sampling Table (LR=0.001, Gradient Clipping=1.0)

Cells aggregated across multiple --seed runs are shown as mean ± sample std (ddof=1); cells backed by a single run show a bare value. Bold marks the best mean in each column.

## Original

| Method | Sampling | ELLIPTIC | IBM HI-SMALL | IBM HI-MEDIUM | IBM LI-SMALL | IBM LI-MEDIUM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **INTRINSIC** | None (Baseline) | **0.5797** | - | **0.0012** | - | **0.0007** |
|  | RUS | 0.5756 | - | **0.0012** | - | **0.0007** |
|  | SMOTE | 0.5418 | - | **0.0012** | - | **0.0007** |
| | | | | | | |
| **POSITIONAL** | None (Baseline) | **0.0569** | - | - | - | - |
|  | RUS | **0.0569** | - | - | - | - |
|  | SMOTE | **0.0569** | - | - | - | - |
| | | | | | | |
| **DEEPWALK** | None (Baseline) | 0.1452 | - | - | - | - |
|  | RUS | **0.1546** | - | - | - | - |
|  | SMOTE | 0.1177 | - | - | - | - |
| | | | | | | |
| **NODE2VEC** | None (Baseline) | **0.2712** | - | - | - | - |
|  | RUS | 0.1638 | - | - | - | - |
|  | SMOTE | 0.1456 | - | - | - | - |
| | | | | | | |
| **GCN** | None (Baseline) | **0.2319** | - | - | - | - |
|  | RUS | 0.2027 | - | - | - | - |
|  | GRAPH_SMOTE | 0.1875 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2010 | - | - | - | - |
|  | GATSMOTE | 0.2065 | - | - | - | - |
|  | TNU | 0.1915 | - | - | - | - |
| | | | | | | |
| **SAGE** | None (Baseline) | 0.3456 | - | - | - | - |
|  | RUS | 0.3287 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3534 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2278 | - | - | - | - |
|  | GATSMOTE | **0.4527** | - | - | - | - |
|  | TNU | 0.4030 | - | - | - | - |
| | | | | | | |
| **GAT** | None (Baseline) | 0.4701 | - | - | - | - |
|  | RUS | 0.4567 | - | - | - | - |
|  | GRAPH_SMOTE | 0.4474 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.4754 | - | - | - | - |
|  | GATSMOTE | 0.4191 | - | - | - | - |
|  | TNU | **0.4922** | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | **0.4827** | - | - | - | - |
|  | RUS | 0.3997 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3988 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.3648 | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | 0.4494 | - | - | - | - |
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

## 1:2 (Ratio)

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
|  | GATSMOTE | 0.3065 | - | - | - | - |
|  | TNU | 0.4758 | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | **0.4592** | - | - | - | - |
|  | RUS | 0.3957 | - | - | - | - |
|  | GRAPH_SMOTE | 0.2002 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.3851 | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | - | - | - | - | - |
| | | | | | | |

---
