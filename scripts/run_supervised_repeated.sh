#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-auc}"
NETWORK_NAME="${NETWORK_NAME:-ibm}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/aml/bin/python}"
SEEDS=(42 123 999)

# Preserve legacy result files by migrating old _ibm_ names to _hi_small_ before reruns.
python - <<'PY'
from pathlib import Path
res_dir = Path('res')
if res_dir.exists():
        migrated = []
        for f in res_dir.iterdir():
                if f.is_file() and '_ibm_' in f.name:
                        new_name = f.name.replace('_ibm_', '_hi_small_')
                        f.rename(res_dir / new_name)
                        migrated.append(new_name)
        if migrated:
                print(f"Migrated {len(migrated)} legacy result files to _hi_small_ naming")
PY

for seed in "${SEEDS[@]}"; do
  echo "=== Running dataset=${NETWORK_NAME} seed=${seed} ==="
  "$PYTHON_BIN" scripts/train_supervised.py --mode "$MODE" --network "$NETWORK_NAME" --seed "$seed"
done

"$PYTHON_BIN" - <<'PY'
import os
import re
import statistics
from pathlib import Path

root = Path('res')
files = sorted(root.glob('*_seed*.txt'))
if not files:
    raise SystemExit('No per-seed result files were produced.')

by_group = {}
for path in files:
    match = re.match(r'^(?P<name>.+?)_seed(?P<seed>\d+)\.txt$', path.name)
    if not match:
        continue
    group_name = match.group('name')
    by_group.setdefault(group_name, []).append(path)

for group_name, paths in sorted(by_group.items()):
    metrics = {}
    for path in paths:
        with path.open('r', encoding='utf-8') as fh:
            content = fh.read().strip()
        for token in ['AUC-PRC', 'F1_90', 'F1_99']:
            if token in content:
                value = float(re.search(rf'{token}:\s*([0-9.]+)', content).group(1))
                metrics.setdefault(token, []).append(value)
    if not metrics:
        continue
    summary_path = root / f'{group_name}_{os.environ["NETWORK_NAME"]}_summary.txt'
    with summary_path.open('w', encoding='utf-8') as fh:
        for token, values in sorted(metrics.items()):
            mean_value = statistics.mean(values)
            std_value = statistics.pstdev(values) if len(values) > 1 else 0.0
            fh.write(f'{token}: {mean_value:.6f} ± {std_value:.6f}\n')
    print(f'Wrote {summary_path}')
PY
