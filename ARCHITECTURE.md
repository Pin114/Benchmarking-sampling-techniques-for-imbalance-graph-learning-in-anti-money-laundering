# Architecture: Multi-Dataset Three-Layer Loop Structure for Class Imbalance Evaluation

## Overview
The training pipeline implements a systematic approach to evaluate the impact of class imbalance ratios and sampling techniques on various graph-based methods for AML (Anti-Money Laundering) detection **across two diverse datasets**: Elliptic Bitcoin and IBM Transaction Network.

## Four-Layer Loop Structure

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Layer 0: DATASET LOOP (Outermost Layer)                                      │
│ Purpose: Evaluate on multiple network types                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ ├─ elliptic  → Cryptocurrency transaction network (203K nodes)              │
│ └─ ibm       → Banking transaction network (500K nodes)                     │
│                                                                              │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ Layer 1: RATIO LOOP (Outer Layer)                                        │ │
│ │ Purpose: Test different class imbalance ratios                           │ │
│ ├──────────────────────────────────────────────────────────────────────────┤ │
│ │ ├─ ratio = None    → Original dataset imbalance (baseline)              │ │
│ │ ├─ ratio = 2.0     → 2:1 majority:minority (APATE recommendation)      │ │
│ │ └─ ratio = 1.0     → 1:1 fully balanced                                 │ │
│ │                                                                          │ │
│ │ ┌────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ Layer 2: METHOD LOOP (Middle Layer)                               │ │ │
│ │ │ Purpose: Train all 8 supervised methods                           │ │ │
│ │ ├────────────────────────────────────────────────────────────────────┤ │ │
│ │ │ ├─ intrinsic    (feature-based)                                  │ │ │
│ │ │ ├─ positional   (feature-based)                                  │ │ │
│ │ │ ├─ deepwalk     (random walk-based)                              │ │ │
│ │ │ ├─ node2vec     (random walk-based)                              │ │ │
│ │ │ ├─ gcn          (graph neural network)                           │ │ │
│ │ │ ├─ sage         (graph neural network)                           │ │ │
│ │ │ ├─ gat          (graph neural network)                           │ │ │
│ │ │ └─ gin          (graph neural network)                           │ │ │
│ │ │                                                                  │ │ │
│ │ │ ┌──────────────────────────────────────────────────────────────┐ │ │ │
│ │ │ │ Layer 3: SAMPLING LOOP (Inner Layer)                        │ │ │ │
│ │ │ │ Purpose: Apply appropriate sampling technique               │ │ │ │
│ │ │ ├──────────────────────────────────────────────────────────────┤ │ │ │
│ │ │ │                                                              │ │ │ │
│ │ │ │ For Feature-based Methods:                                  │ │ │ │
│ │ │ │   ├─ intrinsic   → ["none", "RUS", "SMOTE"]                 │ │ │ │
│ │ │ │   └─ positional  → ["none", "RUS", "SMOTE"]                 │ │ │ │
│ │ │ │                                                              │ │ │ │
│ │ │ │ For Embedding & GNN Methods:                                │ │ │ │
│ │ │ │   ├─ deepwalk    → ["none", "RUS", "GraphSMOTE"]            │ │ │ │
│ │ │ │   ├─ node2vec    → ["none", "RUS", "GraphSMOTE"]            │ │ │ │
│ │ │ │   ├─ gcn         → ["none", "RUS", "GraphSMOTE"]            │ │ │ │
│ │ │ │   ├─ sage        → ["none", "RUS", "GraphSMOTE"]            │ │ │ │
│ │ │ │   ├─ gat         → ["none", "RUS", "GraphSMOTE"]            │ │ │ │
│ │ │ │   └─ gin         → ["none", "RUS", "GraphSMOTE"]            │ │ │ │
│ │ │ │                                                              │ │ │ │
│ │ │ └──────────────────────────────────────────────────────────────┘ │ │ │
│ │ └────────────────────────────────────────────────────────────────────┘ │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Total Experiments**: 2 datasets × 3 ratios × 8 methods × 3 samplings = **144 combinations**

## Component Responsibilities

### 1. evaluation.py
Central module for data sampling and mask adjustment.

**Key Functions:**
- `adjust_mask_to_ratio(mask, labels, target_ratio, random_state)` - Adjusts dataset to target class imbalance ratio via undersampling
- `smote_mask()` - Applies SMOTE oversampling to minority class
- `graph_smote_mask()` - Applies GraphSMOTE using graph structure for synthetic sample generation
- `random_undersample_mask()` - Simple random undersampling of majority class

### 2. experiments_supervised.py
Implements training logic for all methods with support for different sampling variants.

**Base Training Functions:**
- `intrinsic_features(ntw, train_mask, test_mask, ...)` - Feature extraction from node attributes
- `positional_features(ntw, train_mask, test_mask, ...)` - Positional encoding features
- `node2vec_features(ntw_torch, train_mask, test_mask, ...)` - Random walk embeddings (DeepWalk/Node2Vec)
- `GNN_features(ntw_torch, model, lr, n_epochs, ...)` - Generic GNN training (GCN, SAGE, GAT, GIN)

**SMOTE Variants (Feature-based Methods):**
- `intrinsic_features_smote()` - Intrinsic features with SMOTE oversampling
- `positional_features_smote()` - Positional features with SMOTE oversampling

**GraphSMOTE Variants (GNN Methods):**
- `GNN_features_graphsmote()` - GNN training with GraphSMOTE oversampling

**Model Definitions:**
- `GCN` - Graph Convolutional Network
- `GraphSAGE` - Graph Sample and Aggregate Network
- `GAT` - Graph Attention Network
- `GIN` - Graph Isomorphism Network

### 3. train_supervised.py
Main orchestration script implementing the three-layer loop.

**Workflow:**
1. Load dataset (IBM or Elliptic)
2. Prepare data and device configuration
3. **Iterate through ratios** (None, 2.0, 1.0)
   - Apply `adjust_mask_to_ratio()` to create training set with target ratio
   - **Iterate through methods** (intrinsic, positional, deepwalk, node2vec, gcn, sage, gat, gin)
     - **Iterate through sampling techniques** (none/smote for feature-based, none/graph_smote for GNN)
       - Select appropriate function variant based on sampling type
       - Train and save results

## Data Flow

```
Original Dataset (IBM/Elliptic)
    ↓
Ratio Adjustment (adjust_mask_to_ratio)
    ├─ ratio=None   → original_train_mask
    ├─ ratio=2.0    → adjusted_train_mask (2:1 majority:minority)
    └─ ratio=1.0    → adjusted_train_mask (balanced)
    ↓
Sampling Technique Application
    ├─ none          → use mask as-is
    ├─ Random Undersampling → Apply RUS
    ├─ smote         → apply SMOTE oversampling → smote_mask
    └─ graph_smote   → apply GraphSMOTE → graph_smote_mask
    ↓
Train Method
    ├─ Call appropriate function from experiments_supervised.py
    └─ Save result: res/{method}_params_{ntw_name}_{ratio_tag}_{sampling_tag}.txt
```

## Result File Naming Convention

Results are saved with the following naming pattern:
```
res/{method}_params_{dataset}_{ratio_tag}_{sampling_tag}.txt
```

**Examples:**
- `intrinsic_params_ibm_original.txt` - Intrinsic with original ratio, no sampling
- `intrinsic_params_ibm_original_smote.txt` - Intrinsic with original ratio, SMOTE applied
- `intrinsic_params_ibm_ratio_1to2.txt` - Intrinsic with 2:1 ratio, no sampling
- `intrinsic_params_ibm_ratio_1to2_smote.txt` - Intrinsic with 2:1 ratio, SMOTE applied
- `gcn_params_ibm_ratio_1to2_graph_smote.txt` - GCN with 2:1 ratio, GraphSMOTE applied

## Sampling Strategy

### Baseline (None)
- Uses the ratio-adjusted dataset as-is
- No additional sampling applied
- Establishes performance baseline for each ratio

### Random Undersampling (RUS)
- Randomly removes majority class samples to achieve 1:1 ratio
- Applied to both feature-based and GNN methods
- Faster than synthetic methods, useful for comparison
- Applied **after** ratio adjustment

### Feature-based Methods (Intrinsic & Positional)
Use **SMOTE (Synthetic Minority Over-sampling Technique)**
- Generates synthetic samples by interpolating in feature space
- Suitable for tabular/feature-based methods
- Applied to feature vectors extracted from graph
- **Applied after** ratio adjustment and RUS baseline

### Graph-based Methods (DeepWalk, Node2Vec, GCN, SAGE, GAT, GIN)
Use **GraphSMOTE**
- Generates synthetic samples using graph structure information
- Considers neighborhood relationships when creating synthetic nodes
- More suitable for methods that leverage graph topology
- **Applied after** ratio adjustment and RUS baseline

## Execution Sequence

```python
for ratio in [None, 2.0, 1.0]:
    # Adjust dataset to target ratio
    train_mask_ratio, orig_maj, orig_min, new_maj = adjust_mask_to_ratio(
        original_train_mask, labels, ratio
    )
    
    for method in ["intrinsic", "positional", "deepwalk", "node2vec", "gcn", "sage", "gat", "gin"]:
        # Get sampling techniques specific to this method
        for sampling in method_sampling_techniques[method]:  # ["none", "random_undersample", "smote"/"graph_smote"]
            
            if sampling == "none":
                # Use ratio-adjusted mask directly
                train_mask_sampled = train_mask_ratio
                
            elif sampling == "random_undersample":
                # Apply random undersampling
                train_mask_sampled = random_undersample_mask(train_mask_ratio, labels)
                
            elif sampling == "smote" and method in feature_based_methods:
                # Apply SMOTE (feature-based methods only)
                expanded_features, expanded_labels, train_mask_sampled = smote_mask(
                    train_mask_ratio, features, labels, k_neighbors=5
                )
                
            elif sampling == "graph_smote" and method in gnn_methods:
                # Apply GraphSMOTE (GNN methods only)
                expanded_features, expanded_labels, train_mask_sampled, expanded_edge_index = graph_smote_mask(
                    train_mask_ratio, features, labels, edge_index, k_neighbors=5
                )
            
            # Select appropriate training function based on method and sampling
            if sampling == "smote" and method in ["intrinsic", "positional"]:
                result = {method}_features_smote(ntw, train_mask_sampled, val_mask, ...)
            elif sampling == "graph_smote" and method in gnn_methods:
                result = GNN_features_graphsmote(ntw_torch, model, train_mask_sampled, val_mask, ...)
            else:
                result = {method}_features(ntw, train_mask_sampled, val_mask, ...)
            
            # Save result with naming convention
            save_result(result, f"res/{method}_params_{dataset}_{ratio_tag}_{sampling_tag}.txt")
```

**Naming Convention Details:**
- `ratio_tag`: "original" (None), "ratio_1to2" (2.0 = 2:1 majority:minority), "ratio_1to1" (1.0 = balanced)
- `sampling_tag`: "" (none), "rus" (random_undersample), "smote"/"graph_smote"
- Example: `gcn_params_ibm_ratio_1to2_graph_smote.txt`

## Key Hyperparameters

**Ratio Adjustment:**
- Target ratios: None (original), 2.0 (2:1 majority:minority), 1.0 (balanced)
- Adjustment method: Random undersampling of majority class
- Random state: 42 (for reproducibility)

**Sampling Methods:**
- Random Undersampling: Target ratio = 1.0 (balanced)
- SMOTE/GraphSMOTE:
  - `k_neighbors`: 5
  - `random_state`: 42
  - Similarity metric (GraphSMOTE): cosine distance

**Feature-based methods (Intrinsic & Positional):**
- `n_layers_decoder`: 2
- `hidden_dim_decoder`: 16
- `lr`: 0.05
- `n_epochs_decoder`: 100

- Random walk methods (DeepWalk, Node2Vec):
- `embedding_dim`: 32
- `walk_length`: 5
- `context_size`: 3
- `walks_per_node`: 2
- `n_epochs`: 50

- GNN methods (GCN, SAGE, GAT, GIN):
- `hidden_dim`: 128
- `embedding_dim`: 64
- `n_layers`: 2
- `dropout_rate`: 0.3
- `n_epochs`: 100

## Performance Metrics

All methods return **AUC-PRC** (Area Under the Precision-Recall Curve) as the primary evaluation metric.

## Expected Outputs

After execution, the `res/` directory will contain result files for:
- **3 ratios** × **2 feature-based methods** × **3 sampling variants each** = 18 files for feature methods
- **3 ratios** × **6 GNN methods** × **3 sampling variants each** = 54 files for GNN methods
- **Total: 72 result files** (3 ratios × 8 methods × 3 sampling techniques)

Example structure:
```
res/
├── intrinsic_params_ibm_original.txt              (ratio=None, sampling=none)
├── intrinsic_params_ibm_original_rus.txt          (ratio=None, sampling=random_undersample)
├── intrinsic_params_ibm_original_smote.txt        (ratio=None, sampling=smote)
├── intrinsic_params_ibm_ratio_1to2.txt            (ratio=2.0, sampling=none)
├── intrinsic_params_ibm_ratio_1to2_rus.txt        (ratio=2.0, sampling=random_undersample)
├── intrinsic_params_ibm_ratio_1to2_smote.txt      (ratio=2.0, sampling=smote)
├── intrinsic_params_ibm_ratio_1to1.txt            (ratio=1.0, sampling=none)
├── intrinsic_params_ibm_ratio_1to1_rus.txt        (ratio=1.0, sampling=random_undersample)
├── intrinsic_params_ibm_ratio_1to1_smote.txt      (ratio=1.0, sampling=smote)
├── positional_params_ibm_original.txt
├── positional_params_ibm_original_rus.txt
├── positional_params_ibm_original_smote.txt
├── ... (more positional variants with 3 ratios × 3 sampling)
├── deepwalk_params_ibm_original_graph_smote.txt
├── deepwalk_params_ibm_original_rus.txt
├── deepwalk_params_ibm_original.txt
├── ... (more GNN variants with 3 ratios × 3 sampling each)
└── gin_params_ibm_ratio_1to1_graph_smote.txt
```

## Research Hypothesis

Based on APATE (Anti-Money Laundering Pattern Detection) research:
- **Hypothesis**: A 2:1 majority:minority ratio is optimal for AML fraud detection
- **Validation**: Compare AUC-PRC across ratios to identify the optimal imbalance level
- **Next Phase**: Once optimal ratio is confirmed, apply advanced sampling techniques to further improve performance
