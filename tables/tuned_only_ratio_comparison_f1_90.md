Method X Sampling Table (LR=0.001, Gradient Clipping=1.0)

Cells aggregated across multiple --seed runs are shown as mean ± sample std (ddof=1); cells backed by a single run show a bare value. Bold marks the best mean in each column.

## Original

| Method | Sampling | ELLIPTIC | IBM HI-SMALL | IBM HI-MEDIUM | IBM LI-SMALL | IBM LI-MEDIUM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **INTRINSIC** | None (Baseline) | **0.4387** | - | **0.0025** | - | **0.0014** |
|  | RUS | 0.4353 | - | **0.0025** | - | **0.0014** |
|  | SMOTE | 0.4365 | - | **0.0025** | - | **0.0014** |
| | | | | | | |
| **POSITIONAL** | None (Baseline) | 0.1076 | - | - | - | - |
|  | RUS | 0.1076 | - | - | - | - |
|  | SMOTE | **0.1105** | - | - | - | - |
| | | | | | | |
| **DEEPWALK** | None (Baseline) | **0.2336** | - | - | - | - |
|  | RUS | 0.1778 | - | - | - | - |
|  | SMOTE | 0.1823 | - | - | - | - |
| | | | | | | |
| **NODE2VEC** | None (Baseline) | **0.2268** | - | - | - | - |
|  | RUS | 0.1960 | - | - | - | - |
|  | SMOTE | 0.1197 | - | - | - | - |
| | | | | | | |
| **GCN** | None (Baseline) | 0.3031 | - | - | - | - |
|  | RUS | 0.3066 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3031 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2678 | - | - | - | - |
|  | GATSMOTE | 0.2154 | - | - | - | - |
|  | TNU | **0.3578** | - | - | - | - |
| | | | | | | |
| **SAGE** | None (Baseline) | **0.3761** | - | - | - | - |
|  | RUS | 0.3670 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3259 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.3487 | - | - | - | - |
|  | GATSMOTE | 0.3282 | - | - | - | - |
|  | TNU | 0.3692 | - | - | - | - |
| | | | | | | |
| **GAT** | None (Baseline) | 0.3829 | - | - | - | - |
|  | RUS | 0.3749 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3772 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.3772 | - | - | - | - |
|  | GATSMOTE | 0.3282 | - | - | - | - |
|  | TNU | **0.3840** | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | 0.3407 | - | - | - | - |
|  | RUS | 0.3670 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3339 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | **0.3795** | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | 0.3499 | - | - | - | - |
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
| **INTRINSIC** | None (Baseline) | **0.4387** | - | - | - | - |
|  | RUS | 0.4376 | - | - | - | - |
|  | SMOTE | 0.4376 | - | - | - | - |
| | | | | | | |
| **POSITIONAL** | None (Baseline) | **0.1076** | - | - | - | - |
|  | RUS | **0.1076** | - | - | - | - |
|  | SMOTE | **0.1076** | - | - | - | - |
| | | | | | | |
| **DEEPWALK** | None (Baseline) | **0.2028** | - | - | - | - |
|  | RUS | 0.1835 | - | - | - | - |
|  | SMOTE | **0.2028** | - | - | - | - |
| | | | | | | |
| **NODE2VEC** | None (Baseline) | **0.2359** | - | - | - | - |
|  | RUS | 0.1869 | - | - | - | - |
|  | SMOTE | 0.2348 | - | - | - | - |
| | | | | | | |
| **GCN** | None (Baseline) | 0.3020 | - | - | - | - |
|  | RUS | 0.3123 | - | - | - | - |
|  | GRAPH_SMOTE | 0.2541 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2906 | - | - | - | - |
|  | GATSMOTE | 0.3111 | - | - | - | - |
|  | TNU | **0.3145** | - | - | - | - |
| | | | | | | |
| **SAGE** | None (Baseline) | 0.3658 | - | - | - | - |
|  | RUS | 0.3681 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3818 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.3624 | - | - | - | - |
|  | GATSMOTE | **0.4023** | - | - | - | - |
|  | TNU | 0.3738 | - | - | - | - |
| | | | | | | |
| **GAT** | None (Baseline) | 0.2883 | - | - | - | - |
|  | RUS | **0.3886** | - | - | - | - |
|  | GRAPH_SMOTE | 0.3795 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.3840 | - | - | - | - |
|  | GATSMOTE | 0.3715 | - | - | - | - |
|  | TNU | 0.3806 | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | 0.3556 | - | - | - | - |
|  | RUS | 0.3442 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3339 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.3521 | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | **0.3601** | - | - | - | - |
| | | | | | | |

---

## 1:2 (Ratio)

| Method | Sampling | ELLIPTIC | IBM HI-SMALL | IBM HI-MEDIUM | IBM LI-SMALL | IBM LI-MEDIUM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **INTRINSIC** | None (Baseline) | 0.4330 | - | - | - | - |
|  | RUS | 0.4319 | - | - | - | - |
|  | SMOTE | **0.4376** | - | - | - | - |
| | | | | | | |
| **POSITIONAL** | None (Baseline) | 0.1076 | - | - | - | - |
|  | RUS | 0.1076 | - | - | - | - |
|  | SMOTE | **0.1335** | - | - | - | - |
| | | | | | | |
| **DEEPWALK** | None (Baseline) | 0.1709 | - | - | - | - |
|  | RUS | **0.1880** | - | - | - | - |
|  | SMOTE | 0.1607 | - | - | - | - |
| | | | | | | |
| **NODE2VEC** | None (Baseline) | **0.2655** | - | - | - | - |
|  | RUS | 0.1823 | - | - | - | - |
|  | SMOTE | 0.1425 | - | - | - | - |
| | | | | | | |
| **GCN** | None (Baseline) | 0.2826 | - | - | - | - |
|  | RUS | **0.3077** | - | - | - | - |
|  | GRAPH_SMOTE | 0.2872 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2803 | - | - | - | - |
|  | GATSMOTE | 0.2906 | - | - | - | - |
|  | TNU | 0.2678 | - | - | - | - |
| | | | | | | |
| **SAGE** | None (Baseline) | 0.3681 | - | - | - | - |
|  | RUS | **0.3783** | - | - | - | - |
|  | GRAPH_SMOTE | 0.2849 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | 0.2575 | - | - | - | - |
|  | GATSMOTE | 0.2724 | - | - | - | - |
|  | TNU | 0.3726 | - | - | - | - |
| | | | | | | |
| **GAT** | None (Baseline) | 0.3886 | - | - | - | - |
|  | RUS | 0.3829 | - | - | - | - |
|  | GRAPH_SMOTE | 0.3442 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | **0.3920** | - | - | - | - |
|  | GATSMOTE | 0.2302 | - | - | - | - |
|  | TNU | 0.3772 | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | 0.3499 | - | - | - | - |
|  | RUS | 0.3556 | - | - | - | - |
|  | GRAPH_SMOTE | 0.2632 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | **0.4011** | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | 0.3704 | - | - | - | - |
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
|  | GATSMOTE | 0.2621 | - | - | - | - |
|  | TNU | 0.3783 | - | - | - | - |
| | | | | | | |
| **GIN** | None (Baseline) | 0.3578 | - | - | - | - |
|  | RUS | 0.3556 | - | - | - | - |
|  | GRAPH_SMOTE | 0.2325 | - | - | - | - |
|  | GRAPH_ENSEMBLE_SMOTE | **0.3795** | - | - | - | - |
|  | GATSMOTE | - | - | - | - | - |
|  | TNU | 0.3396 | - | - | - | - |
| | | | | | | |

---
