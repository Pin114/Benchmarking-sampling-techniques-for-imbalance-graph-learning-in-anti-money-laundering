# Project Scope: Supervised Graph Learning with Sampling Techniques

## Overview
This project benchmarks sampling techniques for imbalanced graph learning in anti-money laundering (AML) detection, focusing exclusively on **supervised learning methods**.

## Supervised Methods (6 approaches)

### Feature-Based Methods
1. **Intrinsic Features** - Uses node intrinsic properties with a neural decoder
2. **Positional Features** - Uses PageRank and other positional metrics with a neural decoder
3. **Node2Vec Embeddings** - Random walk-based graph embeddings trained via Word2Vec
4. **DeepWalk** - Graph embeddings variant (similar to Node2Vec with p=q=1)

### Graph Neural Networks (GNNs)
5. **GCN** - Graph Convolutional Network
6. **GraphSAGE** - Graph SAmple and aggreGatE
7. **GAT** - Graph Attention Network (optional)
8. **GIN** - Graph Isomorphism Network (optional)

## Sampling Techniques

The pipeline applies **two sampling strategies** uniformly across all methods:

- **No Sampling** (baseline): Use original imbalanced training set
- **Random Undersample Majority**: Randomly subsample majority class to match minority class size

## Pipeline Architecture

### Training (`scripts/train_supervised.py`)
- Loads dataset (IBM or Elliptic Bitcoin)
- Applies sampling strategy to training masks
- Uses **Optuna** for hyperparameter optimization
- Trains each method with sampled training data
- Saves optimized hyperparameters to `res/` directory

### Evaluation (`scripts/test_supervised.py`)
- Loads trained model hyperparameters
- Evaluates on test set with the same sampling strategy used during training
- Computes Average Precision (AP) scores
- Aggregates results across all methods and sampling strategies

## Key Implementation Details

### Environment Setup
- Python 3.10 (conda environment)
- PyTorch 2.x with PyTorch Geometric
- Optuna for hyperparameter tuning
- macOS-specific: OMP_NUM_THREADS=1 to avoid OpenMP conflicts
- Optuna forced to single-process (OPTUNA_N_JOBS=1) to prevent joblib SIGKILL on Apple Silicon

### Sampling Implementation
- `random_undersample_mask()` in `src/methods/evaluation.py`
- Applied at training time via `random_undersample_mask(mask, y)` 
- Same mask used during test evaluation for consistency

### Node2Vec Robustness
- `node2vec_representation_torch()` in `src/methods/utils/functionsTorch.py`
- Includes fallback: retries with `workers=1` if multi-worker mode fails
- Handles macOS libomp conflicts gracefully

## Removed Components

The following unsupervised-only code has been removed to simplify the codebase:
- `scripts/train_unsupervised.py` - Deleted
- `scripts/test_unsupervised.py` - Deleted
- `src/methods/experiments_unsupervised.py` - Deleted
- All unsupervised result files in `res/` - Deleted

## Expected Output

Hyperparameter results stored as:
```
res/{dataset_name}_{method_name}_{sampling_strategy}.txt
```

Example:
```
res/IBM_gcn_random_undersample.txt
res/IBM_intrinsic_none.txt
res/elliptic_node2vec_random_undersample.txt
```

## Next Steps

1. Run `python scripts/train_supervised.py` to optimize hyperparameters
2. Run `python scripts/test_supervised.py` to evaluate all methods
3. Analyze results to determine which sampling technique benefits each method
