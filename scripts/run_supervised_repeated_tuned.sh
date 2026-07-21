#!/usr/bin/env bash
set -euo pipefail

# ⚙️ 專案根目錄自適應設定
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ⚙️ 預設參數設定
MODE="${1:-auc}"
NETWORK_NAME="${NETWORK_NAME:-hi_small}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEEDS=(42 123 999)
OUT_DIR="res/tuned"

# 確保輸出目錄存在
mkdir -p "$OUT_DIR"

echo "================================================================================"
echo "🎓 STARTING TUNED AML REPEATED EXPERIMENTS (LR=0.001, Clip=1.0)"
echo "================================================================================"
echo "- Mode: ${MODE}"
echo "- Dataset: ${NETWORK_NAME}"
echo "- Seeds: ${SEEDS[*]}"
echo "- Target Directory: ${OUT_DIR}"
echo "================================================================================"

# 🔄 遍歷多個隨機種子進行訓練
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "🚀 >>> Running dataset=${NETWORK_NAME} | seed=${seed} | mode=${MODE} <<<"
    "$PYTHON_BIN" scripts/train_supervised_tuned.py \
        --mode "$MODE" \
        --network "$NETWORK_NAME" \
        --seed "$seed" \
        --lr 0.001 \
        --clip_norm 1.0 \
        --out_dir "$OUT_DIR"
done

# 📊 執行 Python 在線腳本：自動在 res/tuned 中將 42, 123, 999 種子合併為 Mean ± Std Summary
echo ""
echo "=== Aggregating seed results and generating Mean ± Std summaries in ${OUT_DIR} ==="
NETWORK_NAME="$NETWORK_NAME" "$PYTHON_BIN" - <<'PY'
import os
import re
import statistics
from pathlib import Path

# 定位調優後的結果目錄
root = Path('res/tuned')
files = sorted(root.glob('*_seed*.txt'))

if not files:
    print(f"Warning: No per-seed result files (*_seed*.txt) found in '{root}'.")
    print("Please make sure training ran successfully and output files were generated.")
    SystemExit(0)

by_group = {}
for path in files:
    # 提取檔名前綴（作為群組）與種子號
    match = re.match(r'^(?P<name>.+?)_seed(?P<seed>\d+)\.txt$', path.name)
    if not match:
        continue
    group_name = match.group('name')
    by_group.setdefault(group_name, []).append(path)

print(f"Found {len(by_group)} experimental configurations to aggregate.")

for group_name, paths in sorted(by_group.items()):
    metrics = {}
    for path in paths:
        with path.open('r', encoding='utf-8') as fh:
            content = fh.read().strip()
        
        # 嚴格過濾、無視 F1_90，僅統計 AUC-PRC 與 F1_99
        for token in ['AUC-PRC', 'F1_99']:
            if token in content:
                # 提取數值
                match_val = re.search(rf'{token}:?\s*([0-9.]+)', content)
                if match_val:
                    value = float(match_val.group(1))
                    metrics.setdefault(token, []).append(value)
                    
    if not metrics:
        continue
        
    # 寫入學術 Summary 檔案 (Mean ± Std)
    network_env = os.environ.get("NETWORK_NAME", "hi_small")
    summary_path = root / f'{group_name}_{network_env}_summary.txt'
    
    with summary_path.open('w', encoding='utf-8') as fh:
        for token, values in sorted(metrics.items()):
            mean_value = statistics.mean(values)
            std_value = statistics.pstdev(values) if len(values) > 1 else 0.0
            fh.write(f'{token}: {mean_value:.6f} ± {std_value:.6f}\n')
            
    print(f'✅ Summarized: {summary_path.name} (Seeds counted: {len(values)})')
PY

echo "================================================================================"
echo "🎉 ALL TUNED EXPERIMENTS COMPLETED & AGGREGATED!"
echo "- Results saved in: ${OUT_DIR}/"
echo "================================================================================"
