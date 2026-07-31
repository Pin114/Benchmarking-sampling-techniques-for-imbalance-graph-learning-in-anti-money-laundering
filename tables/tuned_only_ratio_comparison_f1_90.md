Method X Sampling Table (LR=0.001, Gradient Clipping=1.0)

Cells aggregated across multiple --seed runs are shown as mean ± sample std (ddof=1); cells backed by a single run show a bare value. Bold marks the best mean in each column.

## Original

| Method | Sampling | ELLIPTIC | IBM HI-SMALL | IBM HI-MEDIUM | IBM LI-SMALL | IBM LI-MEDIUM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **INTRINSIC** | None (Baseline) | 0.4319 | - | **0.0025** | - | **0.0014** |
|  | RUS | **0.4353** | - | **0.0025** | - | **0.0014** |
|  | SMOTE | 0.4046 | - | **0.0025** | - | **0.0014** |
| | | | | | | |
| **POSITIONAL** | None (Baseline) | **0.1076** | - | - | - | - |
|  | RUS | **0.1076** | - | - | - | - |
|  | SMOTE | **0.1076** | - | - | - | - |
| | | | | | | |
| **DEEPWALK** | None (Baseline) | **0.2154** | - | - | - | - |
|  | RUS | 0.1915 | - | - | - | - |
|  | SMOTE | 0.1823 | - | - | - | - |
| | | | | | | |
| **NODE2VEC** | None (Baseline) | **0.2974** | - | - | - | - |
|  | RUS | 0.2165 | - | - | - | - |
|  | SMOTE | 0.2268 | - | - | - | - |
| | | | | | | |
| **GCN** | None (Baseline) | 0.3054 | - | - | - | - |
|  | RUS | 0.3009 | - | - | - | - |
|  | GRAPH_SMOTE | 0.2952 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2849 | - | - | - | - |
|  | GATSMOTE | **0.3088** | - | - | - | - |
|  | TNU | 0.2986 | - | - | - | - |
| | | | | | | |
| **SAGE** | None (Baseline) | 0.3476 | - | - | - | - |
|  | RUS | 0.3761 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3715 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2553 | - | - | - | - |
|  | GATSMOTE | **0.3772** | - | - | - | - |
|  | TNU | 0.3567 | - | - | - | - |
| | | | | | | |
| **GAT** | None (Baseline) | 0.3772 | - | - | - | - |
|  | RUS | 0.3886 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3818 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | **0.4023** | - | - | - | - |
|  | GATSMOTE | 0.3533 | - | - | - | - |
|  | TNU | 0.3783 | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | **0.3897** | - | - | - | - |
|  | RUS | 0.3476 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3658 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.3761 | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | 0.3726 | - | - | - | - |
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
| **INTRINSIC** | None (Baseline) | 0.4376 | - | - | - | - |
|  | RUS | 0.4330 | - | - | - | - |
|  | SMOTE | **0.4387** | - | - | - | - |
| | | | | | | |
| **POSITIONAL** | None (Baseline) | **0.1076** | - | - | - | - |
|  | RUS | **0.1076** | - | - | - | - |
|  | SMOTE | 0.0251 | - | - | - | - |
| | | | | | | |
| **DEEPWALK** | None (Baseline) | 0.1744 | - | - | - | - |
|  | RUS | 0.2108 | - | - | - | - |
|  | SMOTE | **0.2142** | - | - | - | - |
| | | | | | | |
| **NODE2VEC** | None (Baseline) | **0.2781** | - | - | - | - |
|  | RUS | 0.2382 | - | - | - | - |
|  | SMOTE | 0.1652 | - | - | - | - |
| | | | | | | |
| **GCN** | None (Baseline) | **0.3556** | - | - | - | - |
|  | RUS | 0.2940 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3225 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.1994 | - | - | - | - |
|  | GATSMOTE | 0.2655 | - | - | - | - |
|  | TNU | 0.3236 | - | - | - | - |
| | | | | | | |
| **SAGE** | None (Baseline) | **0.3738** | - | - | - | - |
|  | RUS | 0.3521 | - | - | - | - |
|  | GRAPH_SMOTE | 0.2393 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.3544 | - | - | - | - |
|  | GATSMOTE | 0.2587 | - | - | - | - |
|  | TNU | 0.3704 | - | - | - | - |
| | | | | | | |
| **GAT** | None (Baseline) | **0.3920** | - | - | - | - |
|  | RUS | 0.3772 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3225 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.3054 | - | - | - | - |
|  | GATSMOTE | 0.3020 | - | - | - | - |
|  | TNU | 0.3783 | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | 0.3578 | - | - | - | - |
|  | RUS | 0.3556 | - | - | - | - |
|  | GRAPH_SMOTE | 0.2325 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | **0.3795** | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | - | - | - | - | - |
| | | | | | | |

---
