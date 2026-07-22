#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
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
    'REWEIGHTED_GRAPH_SMOTE': 'REWEIGHTED_GRAPH_SMOTE'
}

# Mapping of models to their applicable sampling methods (avoids invalid N/A rows in tables)
METHOD_SAMPLING_MAP = {
    'INTRINSIC': ['NONE', 'RUS', 'SMOTE'],
    'POSITIONAL': ['NONE', 'RUS', 'SMOTE'],
    'DEEPWALK': ['NONE', 'RUS', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'REWEIGHTED_GRAPH_SMOTE'],
    'NODE2VEC': ['NONE', 'RUS', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'REWEIGHTED_GRAPH_SMOTE'],
    'GCN': ['NONE', 'RUS', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'REWEIGHTED_GRAPH_SMOTE'],
    'SAGE': ['NONE', 'RUS', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'REWEIGHTED_GRAPH_SMOTE'],
    'GAT': ['NONE', 'RUS', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'REWEIGHTED_GRAPH_SMOTE'],
    'GIN': ['NONE', 'RUS', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'REWEIGHTED_GRAPH_SMOTE'],
}

RATIOS = ['original', 'ratio_1to10', 'ratio_1to2', 'ratio_1to1']

RATIO_DISPLAY = {
    'original': 'Original ratio (Original)',
    'ratio_1to10': 'Ratio 1:10 (Ratio)',
    'ratio_1to2': 'Ratio 1:2 (Ratio)',
    'ratio_1to1': 'Ratio 1:1 (Ratio)'
}

def infer_metadata(path: Path):
    name = path.name
    stem = name[:-12] if name.endswith('_summary.txt') else (name[:-4] if name.endswith('.txt') else name)

    # 1. Identify dataset
    dataset = 'unknown'
    for d_tag in DATASET_MAP.keys():
        if d_tag in path.name:
            dataset = d_tag
            break

    # 2. Identify model method
    method = 'UNKNOWN'
    stem_upper = stem.upper()
    for m in METHODS:
        if m in stem_upper:
            method = m
            break

    # 3. Identify ratio
    ratio = 'original'
    for r in RATIOS:
        if r in stem:
            ratio = r
            break

    # 4. Identify sampling technique
    sampling = 'NONE'
    for s in ['reweighted_graph_smote', 'graph_ensemble_smote', 'graph_smote', 'smote', 'rus']:
        if s in stem:
            sampling = s.upper()
            break

    # 5. Identify metric type
    if 'F1_99' in stem_upper:
        metric_type = 'F1_99'
    else:
        metric_type = 'AUC-PRC'

    # 6. Identify whether it's tuned
    is_tuned = 'tuned' in str(path.parent) or 'tuned' in path.name
    return method, dataset, ratio, sampling, metric_type, is_tuned

def parse_metrics(path: Path, metric_type: str):
    try:
        content = path.read_text(encoding='utf-8').strip()
        if not content:
            return None

        # 1. If it's a summary file, extract the metric's "0.XXXXXX ± 0.YYYYYY"
        if path.name.endswith('_summary.txt'):
            for line in content.splitlines():
                if ':' in line:
                    key, val = line.split(':', 1)
                    if metric_type == 'AUC-PRC' and 'AUC-PRC' in key.strip().upper():
                        return val.strip()
                    elif metric_type == 'F1_99' and 'F1_99' in key.strip().upper():
                        return val.strip()
            # Fallback: if the file has only a single line
            if ':' not in content and '±' in content:
                return content.strip()

        # 2. If it's a single-run file, parse "AUC-PRC: 0.XXXX, F1_99: 0.YYYY"
        tokens = content.split(',')
        for token in tokens:
            if ':' in token:
                key, val = token.split(':', 1)
                key_clean = key.strip().upper()
                if metric_type == 'AUC-PRC' and 'AUC-PRC' in key_clean:
                    return val.strip()
                elif metric_type == 'F1_99' and 'F1_99' in key_clean:
                    return val.strip()

        # Fallback: if the file has only a single value
        if len(content) < 30 and ':' not in content:
            try:
                float(content)
                return content
            except ValueError:
                pass
    except Exception:
        pass
    return None

def clean_val(val_str):
    """Extract the float within a value; used to find the best value and compute lift."""
    if not val_str or val_str == 'N/A' or val_str == '-':
        return None
    match = re.search(r'([0-9\.]+)', str(val_str))
    if match:
        return float(match.group(1))
    return None

def format_cell_value(val):
    """Format the output value, keeping four decimal places."""
    if not val or val == 'N/A' or val == '-':
        return '-'
    match = re.match(r'^\s*([0-9\.]+)\s*±\s*([0-9\.]+)\s*$', str(val))
    if match:
        mean_val = float(match.group(1))
        std_val = float(match.group(2))
        return f"{mean_val:.4f} ± {std_val:.4f}"
    try:
        f_val = float(val)
        return f"{f_val:.4f}"
    except ValueError:
        pass
    return str(val)

def main():
    res_dir = Path('res')
    if not res_dir.exists():
        print("Error: 'res' directory not found.")
        return

    tables_dir = Path('tables')
    tables_dir.mkdir(exist_ok=True)

    # matrix[is_tuned][metric_type][dataset][sampling][method][ratio] = value
    matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))

    all_files = list(res_dir.glob('**/*.txt'))
    print(f"Total files found in res/: {len(all_files)}")

    for path in all_files:
        if path.name.startswith('.'):
            continue
        method, dataset, ratio, sampling, metric_type, is_tuned = infer_metadata(path)
        if dataset in DATASET_MAP and method in METHODS:
            val = parse_metrics(path, metric_type)
            if val:
                is_summary = path.name.endswith('_summary.txt')
                current_val = matrix[is_tuned][metric_type][dataset][sampling][method].get(ratio)
                # Prefer keeping summary.txt results (which include standard deviation)
                if not current_val or is_summary or ('±' not in str(current_val) and '±' in str(val)):
                    matrix[is_tuned][metric_type][dataset][sampling][method][ratio] = val

    # ========== 1. Output the combined vertical "Model x Sampling" table for Baseline and Tuned ==========
    for is_tuned, folder_name in [(False, "baseline"), (True, "tuned")]:
        for m_type in ['AUC-PRC', 'F1_99']:
            output_lines = []
            output_lines.append(f"# {folder_name.upper()} Model x Sampling Vertical Joint Analysis Matrix ({m_type})\n")
            output_lines.append("This table is organized by model, showing horizontally how each sampling technique's performance evolves across different imbalance ratios. **Bold** marks the best value for each model across all ratios.\n")

            for d_key in sorted(DATASET_MAP.keys()):
                d_name = DATASET_MAP[d_key]
                output_lines.append(f"## Dataset: {d_name}\n")

                headers = ['Model (Method)', 'Sampling Technique (Sampling)', 'Original ratio (Original)', 'Ratio 1:10 (Ratio)', 'Ratio 1:2 (Ratio)', 'Ratio 1:1 (Ratio)']
                output_lines.append('| ' + ' | '.join(headers) + ' |')
                output_lines.append('| ' + ' | '.join([':---', ':---', ':---:', ':---:', ':---:', ':---:']) + ' |')

                for method in METHODS:
                    samplings = METHOD_SAMPLING_MAP.get(method, [])

                    # Compute the max value for this method across all samplings/ratios in this dataset, for highlighting
                    max_val = -1.0
                    for s_key in samplings:
                        for ratio in RATIOS:
                            val = matrix[is_tuned][m_type][d_key][s_key][method].get(ratio)
                            f_val = clean_val(val)
                            if f_val is not None and f_val > max_val:
                                max_val = f_val

                    for idx, s_key in enumerate(samplings):
                        s_name = SAMPLING_TECHNIQUES.get(s_key, s_key)
                        s_display = "None (Baseline)" if s_key == 'NONE' else s_name

                        row_cells = []
                        # Model name is shown only on the model's first row; left blank otherwise
                        if idx == 0:
                            row_cells.append(f"**{method}**")
                        else:
                            row_cells.append("")

                        row_cells.append(s_display)

                        for ratio in RATIOS:
                            val = matrix[is_tuned][m_type][d_key][s_key][method].get(ratio)
                            formatted_val = format_cell_value(val)

                            # Highlight if this is the max value
                            f_val = clean_val(val)
                            if f_val is not None and max_val > 0 and abs(f_val - max_val) < 1e-7:
                                formatted_val = f"**{formatted_val}**"

                            row_cells.append(formatted_val)

                        output_lines.append('| ' + ' | '.join(row_cells) + ' |')

                    # Add a separator row (blank row) after each model to improve readability
                    output_lines.append('| | | | | | |')

                output_lines.append("\n---\n")

            output_file = tables_dir / f"ratio_comparison_tables_{folder_name}_{m_type.lower().replace('-', '_')}.md"
            output_file.write_text('\n'.join(output_lines), encoding='utf-8')
            print(f"[Success] Saved {folder_name} {m_type} table to: {output_file}")

    # ========== 2. Output additional comparison table (Baseline vs Tuned Ablation Study) ==========
    for m_type in ['AUC-PRC', 'F1_99']:
        comparison_lines = []
        comparison_lines.append(f"# Hyperparameter Tuning vs Ablation Comparison Table ({m_type})\n")
        comparison_lines.append("This table compares **Baseline (LR=0.05, No Clip)** against **Tuned (LR=0.001, Gradient Clip=1.0)** side by side, and computes the absolute performance improvement (Absolute Lift).\n")

        for d_key in sorted(DATASET_MAP.keys()):
            d_name = DATASET_MAP[d_key]
            comparison_lines.append(f"## Dataset: {d_name}\n")

            for ratio in RATIOS:
                ratio_name = RATIO_DISPLAY[ratio]
                comparison_lines.append(f"### Configured Ratio: {ratio_name}\n")

                headers = ['Model (Method)', 'Sampling Technique (Sampling)', 'Baseline (LR=0.05)', 'Tuned (LR=0.001)', 'Absolute Lift (Absolute Lift)']
                comparison_lines.append('| ' + ' | '.join(headers) + ' |')
                comparison_lines.append('| ' + ' | '.join([':---', ':---', ':---:', ':---:', ':---:']) + ' |')

                for method in METHODS:
                    samplings = METHOD_SAMPLING_MAP.get(method, [])
                    for idx, s_key in enumerate(samplings):
                        s_name = SAMPLING_TECHNIQUES.get(s_key, s_key)
                        s_display = "None (Baseline)" if s_key == 'NONE' else s_name

                        row_cells = []
                        if idx == 0:
                            row_cells.append(f"**{method}**")
                        else:
                            row_cells.append("")

                        row_cells.append(s_display)

                        val_base = matrix[False][m_type][d_key][s_key][method].get(ratio, '-')
                        val_tuned = matrix[True][m_type][d_key][s_key][method].get(ratio, '-')

                        formatted_base = format_cell_value(val_base)
                        formatted_tuned = format_cell_value(val_tuned)

                        float_base = clean_val(val_base)
                        float_tuned = clean_val(val_tuned)

                        if float_base is not None and float_tuned is not None:
                            diff = float_tuned - float_base
                            sign = "+" if diff >= 0 else ""
                            # Highlight positive improvement
                            if diff > 0:
                                lift_str = f"**{sign}{diff:.4f}**"
                            else:
                                lift_str = f"{sign}{diff:.4f}"
                        else:
                            lift_str = "-"

                        row_cells.extend([formatted_base, formatted_tuned, lift_str])
                        comparison_lines.append('| ' + ' | '.join(row_cells) + ' |')
                comparison_lines.append("\n")
            comparison_lines.append("\n---\n")

        output_file = tables_dir / f"ablation_comparison_{m_type.lower().replace('-', '_')}.md"
        output_file.write_text('\n'.join(comparison_lines), encoding='utf-8')
        print(f"[Success] Saved Ablation Comparison Table to: {output_file}")

if __name__ == "__main__":
    main()
