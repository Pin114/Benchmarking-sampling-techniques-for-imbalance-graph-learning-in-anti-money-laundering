#!/usr/bin/env bash
# ==============================================================================
# run_supervised_repeated.sh - Multi-seed Supervised Training Runner (Baseline)
# ==============================================================================
# This script runs the baseline supervised model across multiple seeds and
# automatically compiles statistics (Mean ± Std) for all metrics.
#
# Fixes applied:
# 1. Broad glob isolation: Prevents picking up files from other datasets.
# 2. Summary filename stitching: Prevents double-nesting of dataset names.
# ==============================================================================

set -euo pipefail

# Auto-resolve repository root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-auc}"
NETWORK_NAME="${NETWORK_NAME:-hi_small}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/aml/bin/python}"
SEEDS=(42 123 999)

# Ensure output directory exists
mkdir -p res


# ------------------------------------------------------------------------------
#  Run Supervised Baseline Across Seeds
# ------------------------------------------------------------------------------
for seed in "${SEEDS[@]}"; do
    echo "================================================================================"
    echo "▶ STARTING BASELINE RUN: Dataset=${NETWORK_NAME} | Seed=${seed} | Mode=${MODE}"
    echo "================================================================================"
    "$PYTHON_BIN" scripts/train_supervised_tuned.py --mode "$MODE" --network "$NETWORK_NAME" --seed "$seed"
done

# ------------------------------------------------------------------------------
#  Robust Multi-Seed Summary Generator (Mean ± Std)
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "▶ COMPILING MULTI-SEED STATISTICS (Mean ± Std)..."
echo "================================================================================"

"$PYTHON_BIN" - <<'PY'
import os
import re
import statistics
import sys
from pathlib import Path

root = Path('res')
target_network = os.environ.get("NETWORK_NAME", "hi_small")

# ✅ FIX 1: Filter files to only match the current dataset to prevent cross-dataset contamination
files = sorted([
    f for f in root.glob('*_seed*.txt') 
    if f'_{target_network}_' in f.name
])

if not files:
    print(f"⚠️  [Warning] No seed results found for network '{target_network}' in '{root}'. Skipping summarization.")
    sys.exit(0)

# Group files by experiment configuration
by_group = {}
for path in files:
    # Matches group name (e.g. gcn_params_hi_small_ratio_1to10_rus) and seed
    match = re.match(r'^(?P<name>.+?)_seed(?P<seed>\d+)\.txt$', path.name)
    if not match:
        continue
    group_name = match.group('name')
    by_group.setdefault(group_name, []).append(path)

# Calculate statistics and write summary files
for group_name, paths in sorted(by_group.items()):
    metrics = {}
    for path in paths:
        with path.open('r', encoding='utf-8') as fh:
            content = fh.read().strip()
            for token in ['AUC-PRC', 'F1', 'F1_90', 'F1_99']:
                # Find metrics with pattern like "AUC-PRC: 0.85" or "F1_99: 0.72"
                m = re.search(rf'{token}:\s*([0-9.]+)', content)
                if m:
                    value = float(m.group(1))
                    metrics.setdefault(token, []).append(value)
                    
    if not metrics:
        continue
    
    # ✅ FIX 2: Prevent double-nesting dataset names. group_name already contains the dataset name.
    summary_path = root / f'{group_name}_summary.txt'
    
    seeds_found = len(paths)
    with summary_path.open('w', encoding='utf-8') as fh:
        for token, values in sorted(metrics.items()):
            mean_value = statistics.mean(values)
            std_value = statistics.pstdev(values) if len(values) > 1 else 0.0
            fh.write(f'{token}: {mean_value:.6f} ± {std_value:.6f} (Count: {seeds_found})\n')
            
    print(f'✅ Summarized: {summary_path.name} (Seeds counted: {seeds_found})')

print("================================================================================"
      "\n🎉 All baseline summarizations completed successfully!"
      "\n================================================================================")
PY
