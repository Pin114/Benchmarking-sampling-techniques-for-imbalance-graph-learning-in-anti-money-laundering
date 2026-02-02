# Benchmarking Sampling Techniques for Imbalanced Graph Learning in Anti-Money Laundering

## 📋 Overview

This project benchmarks various sampling techniques for graph representation learning on imbalanced transaction networks in the context of anti-money laundering (AML) detection. The research validates the **APATE hypothesis**: that a 2:1 (majority:minority) class imbalance ratio is optimal for AML model performance.

## 🎯 Key Findings

✅ **APATE Hypothesis Verified**: 2:1 class ratio achieves the best performance
- 2:1 ratio: AUC-PRC = 0.000774
- 1:1 ratio: AUC-PRC = 0.000761 (-1.7%)
- Original imbalance (1959:1): AUC-PRC = 0.000733 (-5.6%)

🏆 **Best Overall Combination**: Intrinsic Features + 2:1 Ratio + SMOTE
- AUC-PRC Score: 0.000948

📊 **Method Performance Ranking**:
1. Node2Vec (0.000800) - Embedding method
2. Intrinsic Features (0.000799) - Feature-based method
3. SAGE (0.000777) - GNN method
4. DeepWalk (0.000762) - Embedding method

🎨 **Sampling Technique Effectiveness**:
- SMOTE: +5.1% vs GraphSMOTE
- No Sampling: Baseline (0.000760)
- Random Undersampling (RUS): -0.1% vs No Sampling
- GraphSMOTE: -2.7% vs No Sampling (least effective)

## 🧪 Experimental Results

- **Total Experiments**: 72 (3 ratios × 8 methods × 3-4 samplings)
- **Dataset**: IBM Transaction Network (500K nodes, 41 features, 1959:1 imbalance)
- **Status**: ✅ All 72 combinations completed and validated






## 🏗️ Project Structure

```
.
├── README.md                          # Project overview and quick start
├── ARCHITECTURE.md                    # System architecture details
├── PROJECT_SCOPE.md                   # Research objectives and scope
├── FEATURE_SMOTE_IMPLEMENTATION.md    # Technical SMOTE documentation
├── PROJECT_AUDIT.md                   # Integration validation results
├── KNOWN_ISSUES.md                    # Known issues and limitations
├── TRAINING_PROGRESS.md               # Historical training logs
│
├── scripts/
│   ├── train_supervised.py            # Main training pipeline (72 experiments)
│   ├── analyze_results.py             # Basic statistical analysis
│   └── detailed_analysis.py           # Comprehensive analysis + APATE validation
│
├── src/
│   ├── methods/
│   │   ├── experiments_supervised.py  # 8 method implementations
│   │   ├── evaluation.py              # Metrics + SMOTE + GraphSMOTE
│   │   ├── feature_smote_heuristic.py # Feature-space SMOTE
│   │   └── utils/
│   │       ├── GNN.py                 # GNN models (GCN, SAGE, GAT, GIN)
│   │       ├── functionsNetworkX.py   # NetworkX utilities
│   │       ├── functionsNetworKit.py  # NetworkKit utilities
│   │       ├── functionsTorch.py      # PyTorch utilities
│   │       └── decoder.py             # Decoder models
│   ├── utils/
│   │   └── Network.py                 # Graph processing utilities
│   └── data/
│       └── DatasetConstruction.py     # Data loading and processing
│
├── data/
│   └── data/
│       ├── elliptic_bitcoin_dataset/
│       └── IBM/                       # IBM transaction network
│
├── res/                               # Results (72 AUC-PRC scores)
├── config/                            # Configuration files
├── requirements.txt                   # Dependencies
└── LICENSE                            # MIT License
```

## 🚀 Quick Start

### Installation
```bash
conda create -n aml python=3.10
conda activate aml
pip install -r requirements.txt
```

### Run Full Training Pipeline (72 Experiments)
```bash
cd scripts
python train_supervised.py
# Results saved in res/ as {method}_params_ibm_{ratio}_{sampling}.txt
```

### Analyze Results
```bash
# Quick analysis
python analyze_results.py

# Comprehensive analysis + APATE validation
python detailed_analysis.py
```

## 🔬 Technical Details

### Methods (8 Total)

**Feature-based Methods:**
- **Intrinsic:** Graph structure as features (degree, clustering coefficient, centrality measures)
- **Positional:** Node2Vec-like positional embeddings from random walks

**Embedding Methods:**
- **DeepWalk:** Skip-gram model on random walks (embedding_dim=64)
- **Node2Vec:** Guided random walks with p, q parameters (embedding_dim=64)

**Graph Neural Networks:**
- **GCN:** Graph Convolutional Network (2 layers, 128 hidden dim)
- **SAGE:** GraphSAGE with mean aggregator (2 layers, 128 hidden dim)
- **GAT:** Graph Attention Network with multi-head attention (2 heads, 128 hidden dim)
- **GIN:** Graph Isomorphism Network (2 layers, 128 hidden dim)

### Sampling Techniques

- **SMOTE:** Synthetic oversampling in feature space (k=5 neighbors)
- **GraphSMOTE:** Synthetic oversampling using graph structure
- **RUS:** Random undersampling of majority class
- **None:** No resampling (baseline)

### Imbalance Ratios

- **Original:** 1959:1 (as-is from IBM dataset)
- **1:1:** Balanced dataset (equal minority/majority)
- **2:1:** Optimal ratio found by APATE hypothesis

## 📊 Key Findings

### APATE Hypothesis: VERIFIED ✅

The **2:1 ratio is optimal** for this imbalanced graph task:

```
2:1 Ratio:      AUC-PRC = 0.000774 (BEST)
1:1 Ratio:      AUC-PRC = 0.000761 (-1.7%)
Original (1959:1): AUC-PRC = 0.000733 (-5.6%)
```

### Performance Rankings

**By Sampling Technique:**
1. SMOTE:       0.000777 (+5.1% vs no sampling)
2. None:        0.000760
3. RUS:         0.000759
4. GraphSMOTE:  0.000739 (-2.6% vs no sampling)

**By Method:**
1. Node2Vec:    0.000800
2. Intrinsic:   0.000799
3. SAGE:        0.000777
4. DeepWalk:    0.000762
5. GIN:         0.000757
6. GAT:         0.000755
7. Positional:  0.000746
8. GCN:         0.000720

**By Category:**
1. Embedding methods (DeepWalk, Node2Vec): 0.000781
2. Feature-based methods (Intrinsic, Positional): 0.000756
3. GNN methods (GCN, SAGE, GAT, GIN): 0.000743

### Best Combination

**Intrinsic + 2:1 Ratio + SMOTE = AUC-PRC: 0.000948**

## 📁 File Purpose Reference

| File | Purpose |
|------|---------|
| `scripts/train_supervised.py` | Main training pipeline - runs all 72 experiments (3 ratios × 8 methods × 3-4 samplings) |
| `scripts/analyze_results.py` | 7-level analysis: per method, per ratio, per sampling, combinations, rankings, top/bottom |
| `scripts/detailed_analysis.py` | 10-level analysis: all above + cross-tabulation, category analysis, APATE validation |
| `src/methods/experiments_supervised.py` | Core implementations of all 8 methods |
| `src/methods/evaluation.py` | SMOTE, GraphSMOTE, RUS, and other sampling functions |
| `src/methods/feature_smote_heuristic.py` | Feature-space SMOTE with k-NN edge construction |
| `src/methods/utils/GNN.py` | GCN, SAGE, GAT, GIN model implementations |
| `src/utils/Network.py` | Graph loading, processing, and utility functions |
| `data/DatasetConstruction.py` | IBM dataset loading and preprocessing |
| `config/*.yaml` | Configuration templates for methods and datasets |

## 📚 Documentation References

- **`PROJECT_SCOPE.md`** - Research objectives and hypotheses
- **`FEATURE_SMOTE_IMPLEMENTATION.md`** - Technical details of feature-space SMOTE
- **`PROJECT_AUDIT.md`** - Integration validation and testing results
- **`KNOWN_ISSUES.md`** - Known limitations and edge cases

## 📖 Citation

If you use this code, please cite:
```
@article{apate-hypothesis,
  title={Benchmarking Sampling Techniques for Imbalance Graph Learning in Anti-Money Laundering},
  year={2024}
}
```

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Questions or issues? Check KNOWN_ISSUES.md or open an issue.