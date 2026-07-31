#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import statistics
from pathlib import Path
from collections import defaultdict

DATASET_MAP = {
    'hi_small': 'IBM HI-SMALL',
    'hi_medium': 'IBM HI-MEDIUM',
    'li_small': 'IBM LI-SMALL',
    'li_medium': 'IBM LI-MEDIUM',
    'elliptic': 'ELLIPTIC'
}

METHODS = ['INTRINSIC', 'POSITIONAL', 'DEEPWALK', 'NODE2VEC', 'GCN', 'SAGE', 'GAT', 'GIN']

SAMPLING_TECHNIQUES = {
    'NONE': 'NONE',
    'RUS': 'RUS',
    'SMOTE': 'SMOTE',
    'GRAPH_SMOTE': 'GRAPH_SMOTE',
    'GRAPH_ENSEMBLE_SMOTE': 'GRAPH_ENSEMBLE_SMOTE',
    'GATSMOTE': 'GATSMOTE',
    'TNU': 'TNU',
}

RATIOS = ['original', 'ratio_1to100', 'ratio_1to10', 'ratio_1to2', 'ratio_1to1']

RATIO_DISPLAY = {
    'original': 'Original',
    'ratio_1to100': '1:100 (Ratio)',
    'ratio_1to10': '1:10 (Ratio)',
    'ratio_1to2': '1:2 (Ratio)',
    'ratio_1to1': '1:1 (Ratio)'
}

METHOD_SAMPLING_MAP = {
    'INTRINSIC': ['NONE', 'RUS', 'SMOTE'],
    'POSITIONAL': ['NONE', 'RUS', 'SMOTE'],
    'DEEPWALK': ['NONE', 'RUS', 'SMOTE'],
    'NODE2VEC': ['NONE', 'RUS', 'SMOTE'],
    'GCN': ['NONE', 'RUS', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'GATSMOTE', 'TNU'],
    'SAGE': ['NONE', 'RUS', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'GATSMOTE', 'TNU'],
    'GAT': ['NONE', 'RUS', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'GATSMOTE', 'TNU'],
    'GIN': ['NONE', 'RUS', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'GATSMOTE', 'TNU'],
}

# Longest/most-specific sampling tags first: "gatsmote" and "graph_smote" both contain
# "smote" as a substring, so checking in the wrong order would misclassify them as plain
# SMOTE. Same reasoning applies to "graph_ensemble_smote" vs "graph_smote".
SAMPLING_TAG_ORDER = ['gatsmote', 'tnu', 'graph_ensemble_smote', 'graph_smote', 'smote', 'rus']

SEED_RE = re.compile(r'_seed(\d+)')


def infer_metadata(path: Path):
    name = path.name
    stem = name[:-4] if name.endswith('.txt') else name

    dataset = 'unknown'
    for d_tag in DATASET_MAP.keys():
        if d_tag in name:
            dataset = d_tag
            break

    method = 'UNKNOWN'
    stem_upper = stem.upper()
    for m in METHODS:
        if m in stem_upper:
            method = m
            break

    ratio = 'original'
    for r in sorted(RATIOS, key=len, reverse=True):
        if r in stem:
            ratio = r
            break

    sampling = 'NONE'
    for s in SAMPLING_TAG_ORDER:
        if s in stem:
            sampling = s.upper()
            break

    seed_match = SEED_RE.search(stem)
    seed = seed_match.group(1) if seed_match else None

    return method, dataset, ratio, sampling, seed


def parse_both_metrics(path: Path):
    """
    Extracts AUC-PRC, F1_99, and F1_90 scores simultaneously from a single log file content.
    Returns: (auc_val, f1_99_val, f1_90_val) as raw strings (or None if absent).
    """
    auc_val = None
    f1_val = None
    f1_90_val = None
    try:
        content = path.read_text(encoding='utf-8').strip()
        if not content:
            return None, None, None

        lines = content.splitlines()
        # 1. Parse line by line to support multi-line summary formats
        for line in lines:
            if ':' in line:
                key, val = line.split(':', 1)
                key_upper = key.strip().upper()
                if 'AUC-PRC' in key_upper:
                    auc_val = val.strip()
                elif 'F1_90' in key_upper:
                    f1_90_val = val.strip()
                elif 'F1_99' in key_upper or key_upper == 'F1':
                    f1_val = val.strip()

        # 2. Fallback: Parse inline tokens separated by commas
        if not auc_val or not f1_val or not f1_90_val:
            tokens = content.split(',')
            for token in tokens:
                if ':' in token:
                    key, val = token.split(':', 1)
                    key_clean = key.strip().upper()
                    if 'AUC-PRC' in key_clean:
                        auc_val = val.strip()
                    elif 'F1_90' in key_clean:
                        f1_90_val = val.strip()
                    elif 'F1_99' in key_clean or key_clean == 'F1':
                        f1_val = val.strip()
    except Exception:
        pass
    return auc_val, f1_val, f1_90_val


def clean_val(val_str):
    if not val_str or val_str in ('N/A', '-'):
        return None
    match = re.search(r'([0-9.]+)', str(val_str))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def aggregate_cell(values_by_seed):
    """values_by_seed: {seed_key: float}. Returns (mean, std_or_None, n) or None if empty.
    std uses the sample standard deviation (ddof=1, statistics.stdev) -- the standard
    convention for reporting variability across independent repeated runs/seeds -- and is
    only computed when there are at least 2 seeds; a single-seed cell reports a bare value.
    """
    if not values_by_seed:
        return None
    values = list(values_by_seed.values())
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if len(values) > 1 else None
    return mean_val, std_val, len(values)


def format_cell(agg):
    if agg is None:
        return '-'
    mean_val, std_val, _n = agg
    if std_val is None:
        return f"{mean_val:.4f}"
    return f"{mean_val:.4f} ± {std_val:.4f}"


def main():
    res_dir = Path('res')
    if not res_dir.exists():
        print("Error: 'res' directory not found.")
        return

    tables_dir = Path('tables')
    tables_dir.mkdir(exist_ok=True)

    # raw[metric_type][dataset][sampling][method][ratio] = {seed_key: value}
    # Keying by seed (rather than overwriting a single scalar) is what lets multiple
    # --seed runs of the same configuration get aggregated into mean +/- std below,
    # instead of the last file glob() happens to visit silently winning.
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))

    all_files = sorted(res_dir.glob('**/*.txt'))
    print(f"Total files found in res/: {len(all_files)}")

    no_seed_counter = 0
    seeds_seen = set()
    for path in all_files:
        if path.name.startswith('.') or path.name.endswith('_summary.txt'):
            continue
        method, dataset, ratio, sampling, seed = infer_metadata(path)
        if dataset not in DATASET_MAP or method not in METHODS:
            continue

        if seed is not None:
            seed_key = f"seed{seed}"
            seeds_seen.add(seed)
        else:
            # No --seed was passed for this run; give it its own slot instead of colliding
            # with (and silently overwriting) any other seedless run of the same cell.
            no_seed_counter += 1
            seed_key = f"noseed{no_seed_counter}"

        auc_val, f1_val, f1_90_val = parse_both_metrics(path)
        for metric_name, raw_str in (('AUC-PRC', auc_val), ('F1_99', f1_val), ('F1_90', f1_90_val)):
            f_val = clean_val(raw_str)
            if f_val is not None:
                raw[metric_name][dataset][sampling][method][ratio][seed_key] = f_val

    print(f"Distinct seeds found: {sorted(seeds_seen) if seeds_seen else 'none'}")

    sorted_datasets = ['elliptic', 'hi_small', 'hi_medium', 'li_small', 'li_medium']

    for m_type in ['AUC-PRC', 'F1_99', 'F1_90']:
        output_lines = []
        output_lines.append("Method X Sampling Table (LR=0.001, Gradient Clipping=1.0)\n")
        output_lines.append(
            "Cells aggregated across multiple --seed runs are shown as mean ± sample std "
            "(ddof=1); cells backed by a single run show a bare value. Bold marks the best "
            "mean in each column.\n"
        )

        for ratio in RATIOS:
            ratio_title = RATIO_DISPLAY[ratio]
            output_lines.append(f"## {ratio_title}\n")

            headers = ['Method', 'Sampling', 'ELLIPTIC', 'IBM HI-SMALL', 'IBM HI-MEDIUM', 'IBM LI-SMALL', 'IBM LI-MEDIUM']
            output_lines.append('| ' + ' | '.join(headers) + ' |')
            output_lines.append('| ' + ' | '.join([':---', ':---', ':---:', ':---:', ':---:', ':---:', ':---:']) + ' |')

            for method in METHODS:
                samplings = METHOD_SAMPLING_MAP.get(method, [])

                col_max_vals = {}
                for d_key in sorted_datasets:
                    vals_for_dataset = []
                    for s_key in samplings:
                        agg = aggregate_cell(raw[m_type][d_key][s_key][method].get(ratio, {}))
                        if agg is not None:
                            vals_for_dataset.append(agg[0])
                    col_max_vals[d_key] = max(vals_for_dataset) if vals_for_dataset else -1.0

                for idx, s_key in enumerate(samplings):
                    s_display = "None (Baseline)" if s_key == 'NONE' else SAMPLING_TECHNIQUES.get(s_key, s_key)

                    row_cells = []
                    if idx == 0:
                        row_cells.append(f"**{method}**")
                    else:
                        row_cells.append("")

                    row_cells.append(s_display)

                    for d_key in sorted_datasets:
                        agg = aggregate_cell(raw[m_type][d_key][s_key][method].get(ratio, {}))
                        formatted_val = format_cell(agg)

                        max_val_for_col = col_max_vals[d_key]
                        if agg is not None and max_val_for_col > 0 and abs(agg[0] - max_val_for_col) < 1e-7:
                            formatted_val = f"**{formatted_val}**"
                        row_cells.append(formatted_val)

                    output_lines.append('| ' + ' | '.join(row_cells) + ' |')

                output_lines.append('| | | | | | | |')

            output_lines.append("\n---\n")

        output_file = tables_dir / f"tuned_only_ratio_comparison_{m_type.lower().replace('-', '_')}.md"
        output_file.write_text('\n'.join(output_lines), encoding='utf-8')
        print(f"[Success] Saved Tuned-Only {m_type} table to: {output_file}")


if __name__ == "__main__":
    main()
