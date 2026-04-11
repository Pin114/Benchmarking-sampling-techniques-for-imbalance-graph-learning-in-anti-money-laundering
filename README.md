# Benchmarking Sampling Techniques for Imbalanced Graph Learning in Anti-Money Laundering

## Overview
This repository contains the comprehensive benchmarking framework for evaluating graph representation learning and resampling techniques on extreme imbalanced transaction networks in the context of Anti-Money Laundering (AML). 

Moving beyond traditional static evaluations, this project evaluates **144 distinct experimental configurations** across two datasets (Elliptic Bitcoin & IBM Transaction Network). It investigates the divergence between overall ranking capabilities (AUC-PRC) and extreme operational constraints using a novel **Dynamic Quantile-based F1 evaluation (90th and 99th percentiles)**.

---

## Key Findings: The Context-Aware Decision Framework

A core objective of this study was to test the **APATE Hypothesis** (the premise that a 2:1 majority-to-minority ratio is universally optimal). Our empirical results **conditionally validate** this hypothesis, proving that there is no universal "silver bullet." Optimal configurations depend entirely on the financial institution's operational capacity:

### 1. Overall Ranking & Separability (AUC-PRC)
For institutions aiming to maximize overall separability across the entire network:
* **The APATE Hypothesis (2:1 Ratio) wins:** The 2:1 ratio consistently outperforms both the original extreme imbalance and the aggressive 1:1 balance.
* **Real-World Networks (Elliptic):** The combination of **GAT + GraphSMOTE** achieved the highest performance (AUC-PRC: **0.9275**), proving that topology-aware synthetic generation works exceptionally well on high-variance networks.
* **Synthetic Networks (IBM):** Due to the extreme 1959:1 imbalance, the random-guess baseline is naturally suppressed ($\approx$ 0.0005). While the absolute AUC-PRC appears low (Peak at **0.000948** with Intrinsic + SMOTE), the 2:1 ratio successfully prevented topological noise. *To measure true business value, we must pivot to the operational F1 metric below.*

### 2. Strict Operational Constraints (Quantile F1_90 & F1_99)
For compliance teams with constrained resources, capable of reviewing only the **Top 1% most suspicious alerts** (F1_99):
* **Synthetic Balancing Fails:** Forcing a 2:1 ratio causes probability over-calibration and introduces massive false positives. Retaining the **Original Imbalance** is strictly required to maintain conservative probability calibration.
* **GNN Oversmoothing & Model Collapse:** Under the F1_99 threshold, all topology-dependent methods (GNNs, DeepWalk, Node2Vec) suffered from severe model prediction collapse, plateauing at the trivial baseline (0.0015 on IBM). The illicit signals were entirely diluted by the 99.9% licit neighbors during message-passing.
* **The Winner:** Purely feature-based methods (`Intrinsic`) paired with feature-space `SMOTE` evaluate transactions in strict isolation. This combination successfully escaped oversmoothing and achieved an operational F1_99 of **0.1128 (11.28%)** on the IBM dataset, proving high actionability for real-world AML deployment.

---

## Methodology & Technical Architecture

### Datasets & Splits
* **Elliptic Bitcoin Dataset:** 203,769 nodes (Real-world cryptocurrency).
* **IBM Transaction Network:** 500,000 nodes, 1959:1 ratio (Synthetic fiat).
* **Data Partitioning:** Nodes/edges were chronologically partitioned into a **70% Training, 15% Validation, and 15% Testing** split prior to any sampling, ensuring zero data leakage.

### Graph Representation Learning Methods (8 Total)
* **Feature-Based (Evaluated in Isolation):** `Intrinsic` (Centrality, degree, clustering) and `Positional`.
  * *Note on Classifier:* Features extracted by these methods are fed into a standard **Multi-Layer Perceptron (MLP)** downstream classifier to ensure a fair architectural comparison against the MLP classification heads of GNNs.
* **Network Embeddings:** `DeepWalk`, `Node2Vec` (64-dim).
* **Graph Neural Networks (Message-Passing):** `GCN`, `GraphSAGE`, `GAT`, `GIN` (2 layers, 128 hidden dim).

### The "Dual-Track" Topological Adaptation
To resolve standard tensor dimension mismatches in `GraphSMOTE`, we engineered a dual-track pipeline:
1. **For GNNs & Embeddings:** Implemented a $k$-NN heuristic to dynamically reconstruct relational edges for synthetic nodes, preventing pipeline crashes and ensuring mathematical rigor.
2. **For Feature-Based Methods:** Utilized standard feature-space SMOTE, explicitly bypassing topological reconstruction to prevent structural noise.

---

## Quick Start & Reproduction

### Installation
```bash
conda create -n aml python=3.10
conda activate aml
pip install -r requirements.txt
1. Run the Training Pipeline (144 Configurations)
The pipeline explores 2 datasets × 3 ratios (Original, 2:1, 1:1) × 8 methods × 3-4 sampling techniques.
Mode A: Standard AUC-PRC Evaluation
python scripts/train_supervised.py
(Results saved in res/ as {method}_params_{dataset}_{ratio}_{sampling}.txt)
Mode B: Dynamic Quantile F1 Evaluation (New) Evaluates models using 90th and 99th percentile probability thresholds.
python scripts/train_supervised.py --mode f1
(Results saved in res/ as {method}_f1_90_... and {method}_f1_99_...)
2. Run Comprehensive Analysis
Parse the generated results and output the statistical ranking tables:
python scripts/analyze_results.py
python scripts/detailed_analysis.py

--------------------------------------------------------------------------------
Project Structure
.
├── README.md                              # Project overview and findings
├── scripts/
│   ├── train_supervised.py                # Main pipeline (AUC & F1 modes)
│   ├── analyze_results.py                 # Core statistical analysis
│   └── detailed_analysis.py               # Comprehensive F1/AUC reporting
├── src/
│   ├── methods/
│   │   ├── experiments_supervised.py      # Implementations of 8 methods + MLP heads
│   │   ├── evaluation.py                  # Quantile F1 and AUC-PRC metrics logic
│   │   ├── feature_smote_heuristic.py     # Dual-track k-NN edge reconstruction
│   │   └── utils/
│   │       ├── GNN.py                     # PyTorch GNN Architectures
│   │       └── ...
│   └── utils/
│       └── Network.py                     # 70/15/15 Data splitting and graph masking
├── data/                                  # IBM & Elliptic raw/processed data
└── res/                                   # Output .txt reports

--------------------------------------------------------------------------------
Citation
If you utilize this benchmarking framework or the dual-track evaluation methodology, please cite our thesis:
@mastersthesis{aml-graph-benchmark-2026,
  author  = {Chen, Pinyu and Ranson, Mathijs},
  title   = {Benchmarking Sampling Techniques for Imbalanced Graph Learning in Anti-Money Laundering},
  school  = {KU Leuven, Research Centre for Information Systems Engineering (LIRIS)},
  year    = {2026}
}
License: MIT License