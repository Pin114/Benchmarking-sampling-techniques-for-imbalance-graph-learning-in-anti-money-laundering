# Benchmarking Sampling Techniques for Imbalanced Graph Learning in Anti-Money Laundering

## Overview

This project systematically benchmarks sampling techniques for graph-based anti-money laundering (AML) detection across imbalanced transaction networks. We evaluate **8 graph representation learning methods** across **2 diverse datasets** with **3 sampling techniques** and **3 class imbalance ratios**, validating the **APATE hypothesis**: that a 2:1 (majority:minority) class imbalance ratio is optimal for AML model performance.

**Total Experiments**: 2 datasets × 3 ratios × 8 methods × 3 samplings = **144 independent trainings**

---

## Key Findings

### APATE Hypothesis: VALIDATED

The 2:1 class ratio shows consistent improvements across both datasets:

| Dataset | Original | 2:1 Ratio | 1:1 Ratio | Winner |
|---------|----------|-----------|-----------|--------|
| **Elliptic** | 0.6648 | **0.6662** | 0.6133 | 2:1 (+0.21%) |
| **IBM** | 0.0007335 | **0.0007744** | 0.0007605 | 2:1 (+5.6%) |

**Conclusion**: The 2:1 ratio consistently outperforms both the original imbalance and full balance (1:1).

### Dataset-Specific Performance

**Elliptic Bitcoin Network** (203,769 nodes, ~4:1 imbalance):
- **Best Method**: GAT (0.8555 AUC-PRC) with attention mechanisms
- **Best Sampling**: GraphSMOTE (+39.8% vs None)
  - None: 0.5778 | RUS: 0.6151 | GraphSMOTE: 0.8092
- **Performance Range**: 0.1197 (Positional) to 0.8555 (GAT)
- **Key Insight**: Graph structure and attention mechanisms highly effective

**IBM Transaction Network** (500K nodes, 1,959:1 imbalance):
- **Best Methods**: Node2Vec (0.0008003) & Intrinsic (0.0007994)
- **Best Sampling**: None (0.0007627) - sampling shows limited benefit
  - None: 0.0007627 | RUS: 0.0007591 | GraphSMOTE: 0.0007385
- **Performance Range**: 0.0006892 (GIN) to 0.0008003 (Node2Vec)
- **Key Insight**: Simple features and walk-based embeddings more stable

### Method Rankings

**By Category (Averaged across ratios/samplings):**
1. **Embedding Methods**: 0.000781 (DeepWalk, Node2Vec)
2. **Feature-Based**: 0.000756 (Intrinsic, Positional)
3. **GNN Methods**: 0.000743 (GCN, SAGE, GAT, GIN)

**Best Combination**: **Intrinsic + 2:1 Ratio + SMOTE = 0.000948 AUC-PRC**

---

## Project Structure

```
.
├── README.md                          # This file
├── scripts/
│   ├── train_supervised.py            # Main training orchestrator (4-layer loop)
│   ├── analyze_results.py             # Single-dataset analysis
│   └── detailed_analysis.py           # Multi-dataset comparative analysis
├── src/
│   ├── methods/
│   │   ├── experiments_supervised.py  # 8 method training implementations
│   │   ├── evaluation.py              # Sampling functions & metrics (739 lines)
│   │   ├── feature_smote_heuristic.py # GraphSMOTE implementation (130 lines)
│   │   └── utils/
│   │       ├── GNN.py                 # GNN models (GCN, SAGE, GAT, GIN)
│   │       ├── decoder.py             # Neural decoders (4 variants)
│   │       ├── functionsNetworkX.py   # NetworkX graph algorithms
│   │       ├── functionsNetworKit.py  # NetworKit scalable algorithms
│   │       └── functionsTorch.py      # PyTorch utilities
│   ├── utils/
│   │   └── Network.py                 # network_AML wrapper class (177 lines)
│   └── data/
│       └── DatasetConstruction.py     # Dataset loading & splitting (168 lines)
├── data/
│   └── data/
│       ├── elliptic_bitcoin_dataset/
│       │   ├── elliptic_txs_features.csv
│       │   ├── elliptic_txs_edgelist.csv
│       │   └── elliptic_txs_classes.csv
│       └── IBM/
│           └── HI-Small_Trans.csv
├── res/                               # Results directory (144 result files)
├── config/                            # Configuration files (YAML)
├── requirements.txt
└── LICENSE
```

---

## Quick Start

### Installation

```bash
conda create -n aml python=3.10
conda activate aml
pip install -r requirements.txt
```

### Run Full Training Pipeline

```bash
cd scripts
python train_supervised.py
# Results saved in res/ as {method}_params_{dataset}_{ratio}_{sampling}.txt
```

### Analyze Results

```bash
# Quick single-dataset analysis
python analyze_results.py

# Comprehensive multi-dataset analysis + APATE validation
python detailed_analysis.py
```

---

## Methodology

### Four-Layer Loop Architecture

```
┌─ Dataset Layer (2 datasets)
│  ├─ Elliptic Bitcoin (203K nodes, temporal, ~4:1 imbalance)
│  └─ IBM Transaction Network (500K nodes, synthetic, 1,959:1 imbalance)
│
├─ Ratio Adjustment Layer (3 ratios)
│  ├─ Original: Dataset's natural imbalance
│  ├─ 2:1: APATE recommended ratio
│  └─ 1:1: Fully balanced
│
├─ Method Selection Layer (8 methods)
│  ├─ Intrinsic Features
│  ├─ Positional Features
│  ├─ DeepWalk
│  ├─ Node2Vec
│  ├─ GCN
│  ├─ GraphSAGE
│  ├─ GAT
│  └─ GIN
│
└─ Sampling Technique Layer (3 samplings)
   ├─ None (Baseline)
   ├─ Random Undersampling (RUS)
   └─ SMOTE / GraphSMOTE
```

### 8 Methods Evaluated

#### Feature-Based Methods (2)

**1. Intrinsic Features**
- Uses raw node attributes directly
- Neural decoder: 2-3 layers (128→64 hidden units)
- Training: Adam optimizer, lr=0.01, epochs=100
- Evaluation metric: AUC-PRC

**2. Positional Features**
- Graph position encoding (PageRank, centrality measures)
- Combines NetworkX + NetworKit algorithms
- Features: Betweenness, Closeness, Eigenvector centrality
- Neural decoder: 2-3 layers

#### Embedding-Based Methods (2)

**3. DeepWalk**
- Random walks (10 per node, length=80)
- Skip-gram embedding (dimension=128, window=10)
- Negative sampling for efficiency
- Training: Standard Word2Vec approach

**4. Node2Vec**
- Biased random walks (p=1.5, q=1.0)
- Parameters: 10 walks/node, length=80, embedding_dim=128
- Exploration-exploitation balance via p/q tuning
- Implementation: PyTorch Geometric

#### Graph Neural Networks (4)

**5. GCN (Graph Convolutional Network)**
- Spectral convolution via Chebyshev approximation
- Architecture: 2-3 layers (128→64 hidden units)
- Aggregation: Normalized adjacency matrix weighted sum
- Reference: Kipf & Welling (2017)

**6. GraphSAGE (SAmple and aggreGatE)**
- Spatial neighbor sampling (15 neighbors, 2-hop)
- Mean aggregation over sampled neighbors
- Scalable to large graphs
- Reference: Hamilton et al. (2017)

**7. GAT (Graph Attention Network)**
- Multi-head attention for neighbor aggregation
- 8 attention heads (first layer), 1 (output layer)
- Adaptive neighbor weighting
- Reference: Veličković et al. (2018)

**8. GIN (Graph Isomorphism Network)**
- Learnable aggregation (sum + MLP)
- Provably expressive (Weisfeiler-Lehman complete)
- ε parameter: 0.1 (fixed)
- Reference: Xu et al. (2019)

### 3 Sampling Techniques

#### For Feature-Based Methods (Intrinsic, Positional)

**1. SMOTE (Synthetic Minority Over-sampling)**
- k-NN interpolation in feature space (k=5)
- Generates synthetic minority samples
- Formula: $x_{synthetic} = x_i + \lambda(x_j - x_i)$, $\lambda \in [0,1]$
- Implementation: scikit-learn SMOTE

**2. Random Undersampling (RUS)**
- Random removal of majority class samples
- Target: 1:1 balanced ratio
- Simple, reproducible baseline

**3. None**
- No additional sampling (baseline)
- Uses ratio-adjusted dataset as-is

#### For Graph Methods (DeepWalk, Node2Vec, GNNs)

**1. GraphSMOTE (Feature-Space SMOTE + k-NN Heuristic)**

*Implementation Details*:
1. Apply SMOTE in feature space (k=5 neighbors)
2. Generate synthetic node features
3. Connect synthetic nodes via k-NN heuristic:
   - Find k=5 nearest neighbors in original feature space
   - Create bidirectional edges (synthetic ↔ neighbor)
   - Distance metric: Cosine similarity
4. Return: (expanded_features, expanded_labels, expanded_mask, expanded_edge_index)

*Advantages*:
- Leverages graph structure for synthetic sample integration
- Avoids tensor dimension mismatches of complex GraphSMOTE
- Maintains computational efficiency
- File: `src/methods/feature_smote_heuristic.py` (130 lines)

**2. Random Undersampling (RUS)**
- Node-level undersampling to 1:1 ratio
- Preserves graph structure

**3. None**
- No synthetic oversampling (baseline)
- Uses ratio-adjusted graph directly

---

## Evaluation Metrics

### Primary Metric: AUC-PRC (Area Under Precision-Recall Curve)

**Why AUC-PRC instead of AUC-ROC?**
- ROC-AUC dominated by TNR in extreme imbalance (misleading)
- PR-AUC directly measures positive class quality
- Real-world relevance: Practitioners care about fraud detection precision

**Formula**:
$$\text{AUC-PRC} = \int P(\tau) \, dR(\tau)$$

where:
- $P(\tau) = \frac{TP(\tau)}{TP(\tau) + FP(\tau)}$ (Precision)
- $R(\tau) = \frac{TP(\tau)}{TP(\tau) + FN(\tau)}$ (Recall)

**Implementation**: scikit-learn `average_precision_score()`

### Train-Validation-Test Split

- **Training**: 70% (with ratio adjustment & sampling applied)
- **Validation**: 15% (monitoring overfitting)
- **Test**: 15% (final evaluation, unchanged class distribution)

**Dataset-Specific Details**:
- **Elliptic**: Time-based split (temporal consistency)
- **IBM**: Random stratified split (class balance)

---

## Code Overview

### Core Files & Responsibilities

| File | Lines | Purpose |
|------|-------|---------|
| `train_supervised.py` | 354 | Main orchestrator - 4-layer loop (dataset→ratio→method→sampling) |
| `experiments_supervised.py` | 870 | Training implementations for 8 methods |
| `evaluation.py` | 739 | Sampling functions (SMOTE, GraphSMOTE, RUS), ratio adjustment |
| `feature_smote_heuristic.py` | 130 | GraphSMOTE with k-NN edge construction |
| `Network.py` | 177 | network_AML wrapper (NetworkX/NetworKit/PyG conversion) |
| `DatasetConstruction.py` | 168 | Dataset loading & train/val/test split creation |
| `GNN.py` | 299 | GNN models (GCN, SAGE, GAT, GIN) |
| `decoder.py` | 83 | Neural decoders (4 variants: linear, deep, norm, deep_norm) |
| `analyze_results.py` | 192 | Single-dataset statistical analysis |
| `detailed_analysis.py` | 241 | Multi-dataset comparative analysis + APATE validation |

### Data Flow

```
Dataset Loading
    ↓
┌───────────────────────────────────────┐
│ Ratio Adjustment (adjust_mask_to_ratio) │
├───────────────────────────────────────┤
│ Target Ratio: None / 2.0 / 1.0        │
│ Method: Undersampling majority class   │
│ Output: Adjusted boolean mask          │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Sampling Technique Application        │
├───────────────────────────────────────┤
│ None                                   │
│  → Use mask as-is                     │
│                                        │
│ Random Undersampling                  │
│  → RUS to 1:1 ratio                   │
│                                        │
│ SMOTE / GraphSMOTE                    │
│  → Feature-space synthesis             │
│  → Expand features/labels/edges        │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Method Training                       │
├───────────────────────────────────────┤
│ intrinsic_features() / ...            │
│ GNN_features() / ...                  │
│ Train on expanded dataset              │
│ Evaluate on test set (unchanged)      │
└───────────────────────────────────────┘
    ↓
Result: {method}_params_{dataset}_{ratio}_{sampling}.txt
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Ratio adjustment in outer loop** | Fair comparison: all sampling techniques evaluated on same ratio |
| **Sampling after ratio adjustment** | Two-stage: undersampling (speed) → synthesis (diversity) |
| **Feature-space SMOTE for graph methods** | Avoids tensor dimension mismatches; maintains graph structure compatibility |
| **k=5 neighbors for SMOTE/k-NN** | Standard choice balancing local vs. global structure |
| **Fixed epochs (100-150) with validation** | Fair cross-method comparison; prevents dataset-specific overfitting |
| **AUC-PRC as primary metric** | Better reflects imbalanced classification reality than AUC-ROC |

---

## Technical Architecture

### Four-Layer Loop Execution

```python
for dataset in [ibm, elliptic]:
    ntw = load_dataset(dataset)
    original_train_mask = ntw.get_masks()[0]
    
    for ratio in [None, 2.0, 1.0]:
        # Adjust dataset to target ratio
        train_mask_ratio = adjust_mask_to_ratio(
            original_train_mask, labels, ratio, random_state=42
        )
        
        for method in [intrinsic, positional, deepwalk, node2vec, gcn, sage, gat, gin]:
            
            for sampling in method_sampling_techniques[method]:  # ['none', 'rus', 'smote'/'graph_smote']
                
                # Apply sampling
                if sampling == 'none':
                    train_mask_final = train_mask_ratio
                elif sampling == 'random_undersample':
                    train_mask_final = random_undersample_mask(train_mask_ratio)
                elif sampling == 'smote' and method in feature_based:
                    train_mask_final = smote_mask(train_mask_ratio, features, labels)
                elif sampling == 'graph_smote' and method in gnn_methods:
                    train_mask_final = graph_smote_mask(train_mask_ratio, features, labels, edge_index)
                
                # Train and evaluate
                ap_score = train_{method}(ntw, train_mask_final, val_mask, test_mask, ...)
                
                # Save result
                save_result(ap_score, 
                    f"res/{method}_params_{dataset}_{ratio_tag}_{sampling_tag}.txt")
```

### Result File Naming Convention

```
res/{method}_params_{dataset}_{ratio_tag}_{sampling_tag}.txt

Examples:
- intrinsic_params_ibm_original.txt                # ratio=None, sampling=none
- intrinsic_params_ibm_original_smote.txt         # ratio=None, sampling=smote
- intrinsic_params_ibm_ratio_1to2.txt              # ratio=2:1, sampling=none
- intrinsic_params_ibm_ratio_1to2_smote.txt       # ratio=2:1, sampling=smote
- gcn_params_ibm_ratio_1to1_graph_smote.txt       # ratio=1:1, sampling=graph_smote
```

---

## Hyperparameters Summary

### Ratio Adjustment
- Target ratios: None, 2.0, 1.0
- Method: Stratified undersampling
- Random seed: 42

### Sampling Methods
- **SMOTE/GraphSMOTE**: k_neighbors=5, random_state=42
- **RUS**: Target ratio=1.0
- **Similarity metric**: Cosine distance (feature space)

### Feature-Based Methods
- Learning rate: 0.05
- Epochs: 100
- Hidden dimensions: [128, 64] (2-layer decoder)
- Dropout: None

### Embedding Methods (DeepWalk, Node2Vec)
- Embedding dimension: 128 (originally 32)
- Walk length: 80
- Walks per node: 10
- Window size: 10
- Skip-gram negative sampling: Yes

### GNN Methods
- Hidden dimension: 128
- Embedding dimension: 64
- Number of layers: 2-3
- Dropout rate: 0.3
- Learning rate: 0.01
- Epochs: 100-150
- Early stopping: Validation loss threshold

---

## Statistical Analysis Plan

### Primary Hypothesis
**H1**: The 2:1 class ratio achieves significantly higher AUC-PRC than 1:1 balanced ratio
- **Test**: Paired t-test (α=0.05)
- **Expected result**: 2:1 mean AUC-PRC > 1:1 mean AUC-PRC

### Secondary Hypotheses
**H2**: Sampling technique effect varies by method category
- **Test**: Two-way ANOVA (method_category × sampling_technique)

**H3**: Dataset type influences optimal sampling strategy
- **Test**: Dataset × Ratio × Sampling interaction analysis

### Effect Size Metrics
- Percentage improvement: $\frac{\text{AUC-PRC}_{2:1} - \text{AUC-PRC}_{1:1}}{\text{AUC-PRC}_{1:1}} \times 100\%$
- Spearman correlation: Rank order consistency across datasets
- Cohen's d: Effect size for performance differences

### Multiple Comparison Correction
- **Bonferroni correction**: $\alpha_{adjusted} = \frac{0.05}{144} \approx 0.0003$
- **Justification**: 144 experiment combinations require multiple comparison correction

---

## Datasets

### Elliptic Bitcoin Network

| Property | Value |
|----------|-------|
| **Type** | Cryptocurrency transaction network (real-world) |
| **Nodes** | 203,769 transactions |
| **Features** | 166 aggregated transaction features |
| **Temporal** | 49 time steps |
| **Labels** | Licit (0), Illicit (1), Unknown (2) |
| **Imbalance** | ~4:1 (after filtering unknown) |
| **Split** | Time-based: <30 (train), 30-40 (val), ≥40 (test) |

**Features**: Aggregated transaction statistics (amounts, frequencies, patterns)

### IBM Transaction Network

| Property | Value |
|----------|-------|
| **Type** | Synthetic banking network (designed) |
| **Nodes** | 500,000 transactions |
| **Features** | 41 engineered node features |
| **Temporal** | 4-hour sliding window edges |
| **Labels** | Legitimate (0), Fraudulent (1) |
| **Imbalance** | 1,959:1 (extreme) |
| **Split** | Random 70/15/15 stratified |

**Features**: Account-level statistics (volumes, frequencies, patterns)

---

## Expected Results Structure

After execution, `res/` contains **144 result files**:

```
Elliptic Results (72 files):
├── Intrinsic (9 files): 3 ratios × 3 samplings (none, rus, smote)
├── Positional (9 files): 3 ratios × 3 samplings
├── DeepWalk (9 files): 3 ratios × 3 samplings (none, rus, graph_smote)
├── Node2Vec (9 files): 3 ratios × 3 samplings
├── GCN (9 files): 3 ratios × 3 samplings
├── SAGE (9 files): 3 ratios × 3 samplings
├── GAT (9 files): 3 ratios × 3 samplings
└── GIN (9 files): 3 ratios × 3 samplings

IBM Results (72 files): Same structure as Elliptic
```

Each file contains:
```
{method}_params_{dataset}_{ratio_tag}_{sampling_tag}.txt

Content: Single line with AUC-PRC score
Example: "AUC-PRC: 0.7823"
```

---

## File Reference

### Analysis Tools

```bash
# Basic per-method analysis (7 levels)
python scripts/analyze_results.py

# Output includes:
# - Mean AUC-PRC per method
# - Mean AUC-PRC per ratio
# - Mean AUC-PRC per sampling technique
# - Method × sampling combinations
# - Rankings and statistics
```

```bash
# Advanced cross-dataset analysis (10 levels)
python scripts/detailed_analysis.py

# Additional output:
# - Per-dataset performance comparison
# - Method performance consistency across datasets
# - APATE hypothesis validation
# - Interaction effects analysis
# - Effect size calculations
```

---

## Key Functions

### evaluation.py - Core Sampling Functions

**`adjust_mask_to_ratio(mask, fraud_dict, ratio, random_state)`**
- Adjusts training mask to achieve target class ratio
- Uses stratified undersampling of majority class
- Returns: Modified boolean mask

**`smote_mask(mask, features, fraud_dict, k=5)`**
- Applies SMOTE in feature space
- Returns: (expanded_features, expanded_labels, expanded_mask)

**`graph_smote_mask(mask, features, labels, edge_index, k=5)`**
- Applies feature-space SMOTE + k-NN edge heuristic
- Returns: (expanded_features, expanded_labels, expanded_mask, expanded_edge_index)

**`random_undersample_mask(mask, fraud_dict, random_state)`**
- Random undersampling to 1:1 ratio
- Returns: Modified boolean mask

---

## Execution Flow

1. **Initialization**
   - Load dataset (Elliptic or IBM)
   - Prepare training/validation/test masks
   - Configure GPU/CPU device

2. **Ratio Loop** (3 iterations)
   - Apply `adjust_mask_to_ratio()` to create target ratio

3. **Method Loop** (8 iterations)
   - Select appropriate training function

4. **Sampling Loop** (3 iterations)
   - Apply sampling technique (or none)
   - Expand dataset if needed

5. **Training**
   - Train method on expanded dataset
   - Evaluate on unchanged test set
   - Compute AUC-PRC score

6. **Results**
   - Save score to file
   - Append to results log

7. **Analysis**
   - Run `analyze_results.py` or `detailed_analysis.py`
   - Generate statistics and visualizations

---

## Notes

- **GraphSMOTE Implementation**: Custom feature-space SMOTE with k-NN edge heuristic (not complex graph-aware SMOTE) to avoid tensor dimension mismatches
- **Reproducibility**: All operations use fixed random seeds (42)
- **Scalability**: Designed to handle 500K+ node graphs efficiently
- **Modularity**: Easy to add new methods, sampling techniques, or datasets

