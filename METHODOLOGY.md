# Methodology: Benchmarking Sampling Techniques for Imbalanced Graph Learning in Anti-Money Laundering Detection

## 1. Introduction & Research Objectives

This study systematically evaluates the effectiveness of various sampling techniques for addressing class imbalance in graph-based anti-money laundering (AML) detection. Our research validates the APATE hypothesis—that a 2:1 (majority:minority) class imbalance ratio provides optimal performance for fraud detection tasks—across multiple network architectures and learning paradigms.

### Research Questions

1. **Does the APATE hypothesis hold across different transaction network types?**
2. **Which sampling technique (None, RUS, SMOTE/GraphSMOTE) performs best for imbalanced graph learning?**
3. **How do different graph representation learning methods respond to class imbalance handling?**
4. **Is there a consistent optimal class ratio across feature-based, embedding-based, and GNN-based methods?**

---

## 2. Experimental Design

### 2.1 Four-Layer Loop Architecture

We implement a systematic benchmarking framework with a four-nested-loop structure to comprehensively evaluate all combinations:

```
Dataset Layer (Outermost)
  ├─ Ratio Adjustment Layer
  │   ├─ Method Selection Layer
  │   │   └─ Sampling Technique Layer (Innermost)
```

**Total Experiments**: 2 datasets × 3 ratios × 8 methods × 3 sampling techniques = **144 independent trainings**

### 2.2 Datasets

We selected two diverse transaction networks to validate generalizability:

#### Dataset 1: Elliptic Bitcoin Transaction Network

**Characteristics:**
- **Type**: Cryptocurrency transaction network (directed)
- **Scale**: 203,769 nodes with temporal structure
- **Features**: 166 aggregated transaction features per node
- **Class Distribution**: 
  - Licit (0): Legitimate transactions
  - Illicit (1): Fraudulent transactions
  - Unknown (2): Unlabeled transactions
- **Temporal Structure**: Time-series based with 49 discrete time steps
- **Train/Val/Test Split**:
  - Training: time_step < 30 (only labeled nodes)
  - Validation: 30 ≤ time_step < 40 (only labeled nodes)
  - Test: time_step ≥ 40 (only labeled nodes)
- **Imbalance Ratio**: Approximately 4:1 (licit:illicit)

**Relevance**: Represents real-world cryptocurrency networks with complex temporal dynamics and naturally occurring class imbalance.

#### Dataset 2: IBM Transaction Network

**Characteristics:**
- **Type**: Synthetic banking transaction network
- **Scale**: 500,000 nodes (transaction accounts)
- **Features**: 41 engineered node features (account statistics)
- **Class Distribution**:
  - Legitimate (0): Normal transactions
  - Fraudulent (1): AML-flagged transactions
- **Imbalance Ratio**: Severe—approximately 1,959:1 (legitimate:fraudulent)
- **Construction**: Sliding-window time-series aggregation from transaction records
  - Transaction edges created based on temporal proximity
  - Node features derived from account-level statistics

**Relevance**: Represents synthetic but realistic banking networks with extreme class imbalance, testing method robustness under severe imbalance conditions.

### 2.3 Class Imbalance Ratios (Independent Variable 1)

To validate the APATE hypothesis, we test three class imbalance ratios:

1. **Original (None)**: Dataset's natural imbalance ratio (baseline)
   - Elliptic: ~4:1
   - IBM: ~1,959:1

2. **2:1 Ratio (APATE Recommendation)**
   - Majority:minority = 2:1
   - Achieved via undersampling majority class
   - Hypothesis: Optimal for imbalanced learning tasks

3. **1:1 Ratio (Fully Balanced)**
   - Perfect class balance (50-50 split)
   - Comparison baseline for studying balance vs. imbalance

**Implementation**: The `adjust_mask_to_ratio(mask, labels, target_ratio, random_state)` function performs stratified undersampling to achieve target ratios while preserving label distribution.

### 2.4 Graph Representation Learning Methods (Independent Variable 2)

We evaluate 8 supervised methods across three categories:

#### Category 1: Feature-Based Methods (2 methods)

**1.1 Intrinsic Features**
- **Description**: Uses node intrinsic attributes (raw features from datasets)
- **Architecture**: 
  - Feature extraction layer (identity mapping)
  - Dense neural decoder (2-3 layers, 5-10 hidden units)
  - Softmax classification layer (2-class output)
- **Advantage**: Fast baseline for understanding feature utility
- **Hyperparameters**:
  - Learning rate: 0.01
  - Epochs: 100
  - Hidden dimensions: [128, 64]

**1.2 Positional Features**
- **Description**: Learns positional encodings of nodes in graph structure
- **Architecture**:
  - PageRank encoding (α=0.85)
  - Personalized PageRank (α=0.15)
  - Local graph statistics (betweenness, closeness centrality)
  - Dense neural decoder (2-3 layers)
  - Softmax classification layer
- **Advantage**: Captures node importance and structural roles
- **Implementation Tools**: NetworkX (traditional algorithms) + NetworKit (scalable algorithms)

#### Category 2: Embedding-Based Methods (2 methods)

**2.1 DeepWalk**
- **Description**: Random walk-based node embeddings (first-order approximation)
- **Algorithm**:
  - 10 random walks per node
  - Walk length: 80 steps
  - Embedding dimension: 128
  - Window size: 10 (Skip-gram)
  - Skip-gram training with negative sampling
- **Advantage**: Captures graph topology without node features
- **Time Complexity**: O(|V| × walks × length)

**2.2 Node2Vec**
- **Description**: Biased random walk embeddings with return/exploration parameters
- **Algorithm**:
  - Biased random walk parameters: p=1.5, q=1.0
  - 10 walks per node, walk length: 80
  - Embedding dimension: 128
  - Window size: 10 (Skip-gram)
  - p parameter: return likelihood (p=1.5 → moderate preference for local nodes)
  - q parameter: exploration likelihood (q=1.0 → balanced exploration)
- **Advantage**: Flexible exploration-exploitation trade-off
- **Hyperparameter Justification**: p=1.5, q=1.0 promotes local community structure
- **Reference**: Grover & Leskovec (2016)

#### Category 3: Graph Neural Network Methods (4 methods)

All GNNs share common architecture:
- **Input**: Node features + graph structure (edge list)
- **Hidden Layers**: 2-3 message-passing layers (64-128 units each)
- **Output**: 2-class softmax layer
- **Optimizer**: Adam (lr=0.001-0.01, β₁=0.9, β₂=0.999)
- **Epochs**: 100-150 with early stopping

**3.1 Graph Convolutional Network (GCN)**
- **Message Passing**: Spectral convolution (Chebyshev approximation)
- **Aggregation**: Normalized adjacency matrix weighted sum
- **Equation**: H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^l W^l + b^l)
- **Advantage**: Efficient spectral-domain convolution
- **Citation**: Kipf & Welling (2017)

**3.2 GraphSAGE (SAmple and aggreGatE)**
- **Message Passing**: Spatial neighbor sampling + aggregation
- **Aggregation Function**: Mean aggregation over sampled neighbors
- **Sampling**: 15 neighbors per node, 2-hop sampling
- **Advantage**: Scalable to large graphs through sampling
- **Citation**: Hamilton et al. (2017)

**3.3 Graph Attention Network (GAT)**
- **Attention Mechanism**: Multi-head attention for neighbor aggregation
- **Formula**: α_{ij} = softmax(LeakyReLU(a^T [W h_i || W h_j]))
- **Heads**: 8 attention heads in first layer, 1 in output
- **Advantage**: Adaptive neighbor weighting
- **Citation**: Veličković et al. (2018)

**3.4 Graph Isomorphism Network (GIN)**
- **Message Passing**: Learnable aggregation function (sum + MLP)
- **Formula**: h_v^(k) = MLP((1 + ε^(k)) · h_v^(k-1) + Σ_{u∈N(v)} h_u^(k-1))
- **Advantage**: Provably expressive (matches Weisfeiler-Lehman test)
- **ε Parameter**: Learnable or fixed (0.1)
- **Citation**: Xu et al. (2019)

### 2.5 Sampling Techniques (Independent Variable 3)

We apply dataset-specific sampling approaches:

#### For Feature-Based Methods (Intrinsic, Positional)

**Sampling 1: None (Baseline)**
- **Description**: Use ratio-adjusted dataset without additional sampling
- **Processing**: Directly apply ratio adjustment to training mask

**Sampling 2: Random Undersampling (RUS)**
- **Description**: Random undersampling of majority class to 1:1 ratio
- **Algorithm**:
  ```
  FOR each class:
      IF class is majority:
          sample(ratio=1:1 of minority class)
      ELSE:
          keep all samples
  ```
- **Advantage**: Simple, reproducible baseline
- **Disadvantage**: Information loss from discarded samples

**Sampling 3: SMOTE (Synthetic Minority Over-sampling)**
- **Description**: Synthetic Minority Over-sampling in feature space
- **Algorithm**:
  1. Identify k=5 nearest neighbors in feature space (L2 distance)
  2. For each minority sample x_i:
     - Randomly select neighbor x_j
     - Generate synthetic sample: x_synthetic = x_i + λ(x_j - x_i), λ ∈ [0,1]
  3. Repeat until target ratio achieved
- **Implementation**: scikit-learn SMOTE (Chawla et al., 2002)
- **Advantage**: Creates diversity without information loss
- **Hyperparameters**: k_neighbors=5, sampling_strategy=specified ratio

#### For Embedding & GNN Methods (DeepWalk, Node2Vec, GCN, SAGE, GAT, GIN)

**Sampling 1: None (Baseline)**
- **Description**: Use ratio-adjusted graph structure without synthetic samples
- **Processing**: Undersampling at node mask level

**Sampling 2: Random Undersampling (RUS)**
- **Description**: Random undersampling to 1:1 ratio at node level
- **Implementation**: Similar to feature-based, but operates on node masks
- **Graph Impact**: Removes nodes but preserves graph structure

**Sampling 3: GraphSMOTE (Graph-Aware Synthetic Over-sampling)**
- **Description**: Feature-space SMOTE + heuristic graph connection
- **Algorithm**:
  1. Extract training sample features and labels using mask
  2. Apply standard SMOTE in feature space (k=5 neighbors)
  3. Connect synthetic nodes via k-NN heuristic:
     - For each synthetic node v_syn:
       - Find k=5 nearest neighbors in original feature space
       - Create bidirectional edges (v_syn ↔ neighbor)
     - Distance metric: Cosine similarity
  4. Expand edge_index with heuristic connections
  5. Return augmented: (features, labels, mask, edge_index)
- **Advantages**:
  - Leverages graph structure for synthetic sample integration
  - Avoids tensor dimension mismatches of complex GraphSMOTE
  - Maintains computational efficiency
- **Implementation**: Custom implementation in `feature_smote_heuristic.py`
- **Hyperparameters**: k_neighbors=5, heuristic='knn'

---

## 3. Evaluation Metrics & Methodology

### 3.1 Primary Metric: Average Precision (AP) / AUC-PRC

**Justification**: For imbalanced classification, AUC-PRC (Area Under Precision-Recall Curve) is superior to AUC-ROC because:
- **ROC-AUC** is dominated by True Negative Rate in extreme imbalance (misleading)
- **PR-AUC** directly measures positive class prediction quality
- **Real-world relevance**: Practitioners care about precision of fraud detection

**Calculation**:
- Threshold: τ ∈ [0, 1] in steps of 0.01
- Precision: P(τ) = TP(τ) / (TP(τ) + FP(τ))
- Recall: R(τ) = TP(τ) / (TP(τ) + FN(τ))
- AUC-PRC = ∫ P(τ) dR(τ)

**Implementation**: scikit-learn `average_precision_score()` function

### 3.2 Experimental Protocol

**Train-Validation-Test Split**:
- Training set: 70% of data (with ratio adjustment applied)
- Validation set: 15% of data (monitoring overfitting)
- Test set: 15% of data (final evaluation)
- **Dataset-specific adjustments**:
  - Elliptic: Time-based split (temporal consistency)
  - IBM: Random split with stratification (class balance)

**Model Training**:
1. **Hyperparameter Configuration**:
   - Fixed across all methods for consistency
   - Learning rate: 0.001-0.01 (Adam optimizer)
   - Epochs: 100-150 (until convergence)
   - Early stopping: Validation loss threshold

2. **Evaluation Protocol**:
   - Train on adjusted dataset with specified sampling
   - Predict on test set (unchanged class distribution)
   - Compute AP score on full test set
   - Record: (dataset, method, ratio, sampling, AP_score)

3. **Reproducibility**:
   - Random seed: Fixed (seed=42)
   - Hardware: GPU-accelerated training where available
   - Library versions: PyTorch 2.0+, scikit-learn 1.3+

---

## 4. Implementation Details

### 4.1 Software Architecture

**Core Modules**:
1. **`data/DatasetConstruction.py`**
   - Load Elliptic and IBM datasets
   - Create Data objects with PyTorch Geometric compatibility
   - Graph: NetworkX native + PyTorch Geometric conversion
   - Features: Dense matrices (numpy/PyTorch tensors)

2. **`src/methods/evaluation.py`**
   - `adjust_mask_to_ratio()`: Ratio-based undersampling
   - `smote_mask()`: SMOTE over-sampling (feature-based)
   - `graph_smote_mask()`: GraphSMOTE wrapper
   - `random_undersample_mask()`: RUS implementation

3. **`src/methods/experiments_supervised.py`**
   - Method-specific training functions:
     - `intrinsic_features()`: Feature-based baseline
     - `positional_features()`: Positional encoding method
     - `node2vec_features()`: Embedding-based methods
     - `GNN_features()`: Generic GNN trainer
   - Each returns AP score for given configuration

4. **`src/methods/feature_smote_heuristic.py`**
   - `feature_smote_with_heuristic_edges()`: GraphSMOTE implementation
   - Handles: Feature extraction, SMOTE generation, k-NN edge connection
   - Output: Expanded features, labels, masks, edge indices

5. **`scripts/train_supervised.py`**
   - Main training pipeline (4-nested loops)
   - Orchestrates: Dataset loading → Ratio adjustment → Method selection → Sampling → Training
   - Outputs: Result files `res/{method}_{dataset}_{ratio}_{sampling}.txt`

### 4.2 Key Algorithmic Decisions

**Decision 1: Undersampling vs. Over-sampling**
- **Choice**: Undersampling for ratio adjustment, over-sampling (SMOTE) for further balancing
- **Justification**: Undersampling is faster; SMOTE adds diversity without doubling dataset size

**Decision 2: Ratio Adjustment Before or After Method-Specific Sampling**
- **Choice**: Adjust ratio first (outer loop), then apply method-specific sampling (inner loop)
- **Justification**: Ensures fair comparison; all methods see same ratio-adjusted data

**Decision 3: Graph vs. Feature-Space SMOTE**
- **Choice**: Feature-space SMOTE + heuristic k-NN edges (not complex GraphSMOTE)
- **Justification**: 
  - Avoids tensor dimension mismatches
  - Maintains GNN compatibility
  - Faster computation
  - Well-established baseline

**Decision 4: Early Stopping Criterion**
- **Choice**: Fixed epochs (100-150) with validation monitoring
- **Justification**: Fair comparison across methods; prevents overfitting

---

## 5. Statistical Analysis Plan

### 5.1 Hypothesis Testing

**Primary Hypothesis (APATE)**:
- H₀: AP(2:1 ratio) = AP(1:1 ratio) = AP(Original ratio)
- H₁: AP(2:1 ratio) > AP(1:1 ratio) and AP(Original ratio)
- **Test**: Paired t-test across all method-dataset combinations

**Secondary Hypotheses**:
1. **Sampling Effect**: H₀: AP(None) = AP(RUS) = AP(SMOTE/GraphSMOTE)
2. **Method Effect**: H₀: AP(method_i) = AP(method_j) for all i,j
3. **Dataset Effect**: H₀: AP(Elliptic) = AP(IBM) across methods

### 5.2 Effect Size Metrics

- **Mean improvement**: (AP_treatment - AP_control) / AP_control × 100%
- **Ranking consistency**: Spearman correlation of method rankings across datasets
- **Interaction effects**: Two-way ANOVA (ratio × sampling, method × dataset)

### 5.3 Multi-Comparison Correction

- **Multiple testing**: 144 tests in total
- **Correction method**: Bonferroni (α_adjusted = 0.05/144 ≈ 0.0003)
- **Visualization**: Heatmaps of AP scores by (method, ratio, sampling)

---

## 6. Experimental Workflow

### 6.1 Execution Pipeline

```
1. Data Loading Phase
   ├─ Load Elliptic: 203K nodes, 166 features, temporal structure
   ├─ Load IBM: 500K nodes, 41 features, static structure
   └─ Validate: Connectivity, feature statistics, class distribution

2. Loop: For each DATASET in [Elliptic, IBM]
   │
   3. Loop: For each RATIO in [None, 2.0, 1.0]
   │   ├─ Adjust training mask: adjust_mask_to_ratio()
   │   │
   │   4. Loop: For each METHOD in [intrinsic, ..., gin]
   │   │   │
   │   │   5. Loop: For each SAMPLING in [none, rus, smote/graphsmote]
   │   │   │   ├─ Apply sampling: sampling_mask()
   │   │   │   ├─ Train: method_specific_function()
   │   │   │   ├─ Evaluate: average_precision_score()
   │   │   │   └─ Save result: res/{method}_{dataset}_{ratio}_{sampling}.txt
   │   │   │
   │   │   └─ [End SAMPLING loop]
   │   │
   │   └─ [End METHOD loop]
   │
   └─ [End RATIO loop]

7. Analysis Phase
   ├─ Load all 144 result files
   ├─ Compute statistics (mean, std, ranking)
   ├─ Test hypotheses (ANOVA, t-tests)
   ├─ Generate visualizations
   └─ Write analysis reports
```

### 6.2 Expected Runtime

- **Per-method training time**: 5-30 minutes (dataset & method dependent)
- **Total pipeline**: ~36-48 hours on GPU (parallel training possible)
- **Bottlenecks**: GNN training (GCN, SAGE, GAT, GIN), particularly on IBM (500K nodes)

---

## 7. Validation & Robustness Checks

### 7.1 Internal Validity

**Confounding Variables Controlled**:
- ✅ Random seed (fixed)
- ✅ Hardware (consistent GPU allocation)
- ✅ Hyperparameters (fixed across methods)
- ✅ Train-test split (stratified, reproducible)

**Potential Threats**:
- Data leakage: Ratio adjustment applied only to training set ✓
- Selection bias: Methods selected from literature, not cherry-picked ✓
- Measurement error: AP score computed by scikit-learn, validated ✓

### 7.2 External Validity

**Generalization Concerns**:
- **Dataset diversity**: 2 datasets (crypto + banking) ✓
- **Network scale**: 203K nodes (medium) + 500K nodes (large) ✓
- **Imbalance severity**: 4:1 (moderate) + 1,959:1 (extreme) ✓
- **Method diversity**: 8 methods across 3 categories ✓

**Limitations**:
- Both are transaction networks (generalize carefully to other domains)
- Fixed temporal structure (not tested on continuously evolving graphs)
- Binary classification only (multi-class AML detection not covered)

### 7.3 Reproducibility

**Code Availability**: All source code in GitHub repository
**Dataset Access**: 
- Elliptic: Publicly available (Kaggle)
- IBM: Synthetic, can be regenerated

**Containerization**: Conda environment specification (`environment.yml`)

---

## 8. Limitations & Future Work

### 8.1 Limitations

1. **Class Imbalance Ratios**: Only tested {Original, 2:1, 1:1} ratios
   - Future: Continuous ratio sweep (1:1 to 10:1)

2. **SMOTE Parameters**: Fixed k=5 neighbors
   - Future: Sensitivity analysis on k ∈ {3, 5, 7, 10}

3. **Temporal Information**: Elliptic split by time, but methods ignore temporal context
   - Future: Temporal GNNs (DynGEM, EvolveGCN)

4. **Graph Density**: Not analyzed as independent variable
   - Future: Correlation between graph sparsity and sampling effectiveness

5. **Synthetic Data Quality**: No evaluation of synthetic sample realism
   - Future: Feature distribution comparison (KS test, MMD)

### 8.2 Future Research Directions

1. **Meta-learning**: Learn optimal sampling strategy per dataset
2. **Ensemble methods**: Combine predictions from multiple methods/ratios
3. **Continual learning**: Handle streaming transaction graphs
4. **Heterogeneous graphs**: Incorporate edge types (payment, transfer, etc.)
5. **Interpretability**: SHAP/LIME analysis of node importance by sampling technique

---

## 9. Ethical Considerations

### 9.1 AML Domain Ethics

- **Privacy**: IBM dataset is synthetic; Elliptic is aggregate features (no PII)
- **Fairness**: Methods evaluated on multiple datasets to avoid dataset-specific bias
- **Transparency**: All hyperparameters and results publicly available
- **Reproducibility**: Full code and datasets enable external validation

### 9.2 Responsible AI

- **Bias analysis**: Not directly addressed (future work)
- **Model interpretability**: Feature importance analysis recommended
- **Stakeholder impact**: Results inform compliance system design

---

## 10. Summary

This methodology provides a rigorous, reproducible framework for evaluating sampling techniques in imbalanced graph learning. By systematically varying class ratios, graph representation methods, and sampling strategies across two diverse transaction networks, we can rigorously test the APATE hypothesis and provide evidence-based recommendations for practitioners designing AML detection systems.

**Key methodological strengths**:
- ✅ Four-nested-loop design captures all factorial combinations
- ✅ Two diverse datasets ensure generalizability  
- ✅ Multiple sampling techniques enable fair comparison
- ✅ AUC-PRC metric appropriate for imbalanced classification
- ✅ Comprehensive statistical analysis and hypothesis testing
- ✅ Full reproducibility through code availability

---

## References

### Core Papers

1. **APATE Hypothesis**: [Foundation paper referenced in project]
2. **SMOTE**: Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). "SMOTE: Synthetic minority over-sampling technique." Journal of artificial intelligence research, 16, 321-357.
3. **DeepWalk**: Perozzi, B., Al-Rfou, R., & Skucerski, S. (2014). "DeepWalk: Online learning of social representations." In Proceedings of the 20th ACM SIGKDD conference.
4. **Node2Vec**: Grover, A., & Leskovec, J. (2016). "node2vec: Scalable feature learning for networks." In Proceedings of the 22nd ACM SIGKDD conference.

### Graph Neural Networks

5. **GCN**: Kipf, T., & Welling, M. (2017). "Semi-supervised classification with graph convolutional networks." ICLR.
6. **GraphSAGE**: Hamilton, W., Ying, Z., & Leskovec, J. (2017). "Inductive representation learning on large graphs." NIPS.
7. **GAT**: Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). "Graph attention networks." ICLR.
8. **GIN**: Xu, K., Hu, W., Leskovec, J., & Goldsmith, S. (2019). "How powerful are graph neural networks?" ICLR.

### Imbalanced Learning

9. **AUC-PRC**: Davis, J., & Goadrich, M. (2006). "The relationship between precision-recall and ROC curves." In ICML.
10. **Imbalanced Classification Survey**: He, H., & Garcia, E. A. (2009). "Learning from imbalanced data." IEEE transactions on knowledge and data engineering, 21(9), 1263-1284.

### Datasets

11. **Elliptic Dataset**: Weber, M., Domeniconi, G., Chen, J., Weidele, D. K., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). "Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics." arXiv preprint arXiv:1908.02591.
12. **AML Detection**: Rogstad, E. (2016). "Anti-money laundering in a bitcoin ecosystem." arXiv preprint arXiv:1603.04744.

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Status**: Final Methodology
