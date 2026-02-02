# Project Scope: Multi-Dataset Graph Learning with Class Imbalance & Sampling Techniques

## Overview

This project systematically evaluates the impact of **class imbalance ratios** and **sampling techniques** on supervised graph-based methods for anti-money laundering (AML) detection across **two major datasets**:

1. **Elliptic Bitcoin Dataset** - Large-scale cryptocurrency transaction network
2. **IBM Transaction Network** - Synthetic banking transaction network

## Research Hypothesis

**APATE Finding**: A 2:1 majority:minority class ratio is optimal for AML fraud detection across different network types.

**Validation Approach**: Three-nested-loop evaluation on both datasets:
1. **Ratio Loop** (Outer): Test different imbalance levels (original, 2:1, 1:1)
2. **Method Loop** (Middle): Train all 8 supervised methods
3. **Sampling Loop** (Inner): Apply appropriate sampling techniques (3 variants per method)

**Total Combinations**: 2 datasets × 3 ratios × 8 methods × 3 sampling techniques = **144 independent trainings**

## Datasets

### Elliptic Bitcoin Dataset
- **Type**: Cryptocurrency transaction network
- **Size**: 203,769 nodes with time-series splits
- **Classes**: Licit (0), Illicit (1), Unknown (2)
- **Time-based splits**:
  - Training: time_step < 30 (labeled only)
  - Validation: 30 ≤ time_step < 40 (labeled only)
  - Test: time_step ≥ 40 (labeled only)
- **Characteristics**: High variance in method performance, GraphSMOTE very effective

### IBM Transaction Network
- **Type**: Banking transaction network (synthetic)
- **Size**: 500K nodes with 41 features
- **Imbalance**: 1959:1 (severe imbalance)
- **Classes**: Legitimate (0), Fraudulent (1)
- **Construction**: Sliding window time-series graph from transaction records
- **Characteristics**: Low variance, feature-based methods effective, GraphSMOTE less beneficial

## Supervised Methods (8 total)

### Feature-Based Methods (2)
1. **Intrinsic Features** - Uses node intrinsic properties with a neural decoder
2. **Positional Features** - Uses PageRank and other positional metrics with a neural decoder

### Embedding-Based Methods (2)
3. **DeepWalk** - Random walk-based graph embeddings (p=q=1)
4. **Node2Vec** - Optimized random walk embeddings (p=1.5, q=1.0)

### Graph Neural Networks (4)
5. **GCN** - Graph Convolutional Network
6. **GraphSAGE** - Graph SAmple and aggreGatE
7. **GAT** - Graph Attention Network
8. **GIN** - Graph Isomorphism Network

## Class Imbalance Ratios

The pipeline tests three different majority:minority ratios:

- **None (Original)**: Dataset's natural imbalance (baseline)
- **2.0 (2:1 ratio)**: 2 majority samples per 1 minority sample (APATE recommendation)
- **1.0 (1:1 ratio)**: Fully balanced dataset

**Implementation**: `adjust_mask_to_ratio()` undersamples majority class to achieve target ratio

## Sampling Techniques

### For Feature-Based Methods (Intrinsic & Positional)
1. **None** - Use the ratio-adjusted dataset directly
2. **Random Undersampling (RUS)** - Additional undersampling to 1:1 ratio
3. **SMOTE** - Synthetic Minority Over-sampling in feature space

### For GNN Methods (DeepWalk, Node2Vec, GCN, SAGE, GAT, GIN)
1. **None** - Use the ratio-adjusted dataset directly
2. **Random Undersampling (RUS)** - Additional undersampling to 1:1 ratio
3. **GraphSMOTE** - Synthetic Minority Over-sampling using graph structure

## Pipeline Architecture

### Main Training Script (`scripts/train_supervised.py`)

**Workflow**:
```
Load Dataset (IBM or Elliptic)
  ↓
For each Ratio in [None, 2.0, 1.0]:
  Adjust training mask to target ratio
  ↓
  For each Method in [intrinsic, positional, deepwalk, node2vec, gcn, sage, gat, gin]:
    ↓
    For each Sampling in [none, random_undersample, smote/graph_smote]:
      Apply sampling technique
      Train method with sampled data
      Save result: res/{method}_params_{dataset}_{ratio_tag}_{sampling_tag}.txt
```

**Key Components**:
- `adjust_mask_to_ratio()` - Ratio adjustment via undersampling
- `random_undersample_mask()` - RUS implementation
- `smote_mask()` - SMOTE for feature-based methods
- `graph_smote_mask()` - GraphSMOTE for GNN methods

### Result File Organization

Results saved with naming convention:
```
res/{method}_params_{dataset}_{ratio_tag}_{sampling_tag}.txt
```

**Example Results**:
- `intrinsic_params_ibm_original.txt` - Intrinsic, original ratio, no sampling
- `intrinsic_params_ibm_original_rus.txt` - Intrinsic, original ratio, RUS
- `intrinsic_params_ibm_original_smote.txt` - Intrinsic, original ratio, SMOTE
- `gcn_params_ibm_ratio_1to2.txt` - GCN, 2:1 ratio, no sampling
- `gcn_params_ibm_ratio_1to2_rus.txt` - GCN, 2:1 ratio, RUS
- `gcn_params_ibm_ratio_1to2_graph_smote.txt` - GCN, 2:1 ratio, GraphSMOTE

**Total Output**: 72 result files (3 ratios × 8 methods × 3 sampling techniques)

## Key Implementation Details

### Environment Setup
- Python 3.10 
- PyTorch 2.x with PyTorch Geometric
- macOS-specific: OMP_NUM_THREADS=1 to avoid OpenMP conflicts

### Data Handling
- Training set: Adjusted via `adjust_mask_to_ratio()` based on current ratio
- Validation/Test set: Unchanged (original split for fair comparison)
- Features: Node attributes + positional encodings (method-dependent)
- Edge structure: Preserved for GNN methods

### Hyperparameters

**Feature-Based Methods**:
- `n_layers_decoder`: 2
- `hidden_dim_decoder`: 16
- `lr`: 0.05
- `n_epochs_decoder`: 100

**Embedding-Based Methods**:
- `embedding_dim`: 32
- `walk_length`: 5
- `context_size`: 3
- `walks_per_node`: 2
- `n_epochs`: 50

**GNN Methods**:
- `hidden_dim`: 128
- `embedding_dim`: 64
- `n_layers`: 2
- `dropout_rate`: 0.3
- `n_epochs`: 100

**Sampling Methods**:
- `k_neighbors`: 5 (for SMOTE/GraphSMOTE)
- `random_state`: 42 (for reproducibility)

## Removed Components

- `isolation_forest.py` - Deleted (unused anomaly detection code)
- Unsupervised training scripts - Simplified to supervised-only

## Expected Outputs & Analysis

### Phase 1: Training (Current)
- Generate 72 result files with AUC-PRC scores
- Organize results by: ratio, method, sampling technique

### Phase 2: Analysis (Post-Training)
- Compare performance across ratios to validate APATE hypothesis
- Identify optimal ratio for AML fraud detection
- Analyze sampling technique effectiveness per method type
- Determine if feature-based and GNN methods respond differently to sampling

## Files Structure

```
scripts/
  └── train_supervised.py          # Main training orchestrator

src/
  ├── methods/
  │   ├── experiments_supervised.py # All training functions
  │   ├── evaluation.py            # Sampling & ratio adjustment
  │   └── utils/
  │       ├── GNN.py              # Model definitions
  │       ├── functionsTorch.py    # PyTorch utilities
  │       ├── functionsNetworkX.py # NetworkX utilities
  │       └── functionsNetworKit.py # NetworKit utilities
  └── utils/
      └── Network.py              # Network data structure

data/
  └── DatasetConstruction.py       # Dataset loaders (IBM, Elliptic)

res/
  └── {method}_params_{dataset}_{ratio}_{sampling}.txt  # Results

ARCHITECTURE.md                     # Detailed system architecture
PROJECT_SCOPE.md                    # This file
```
