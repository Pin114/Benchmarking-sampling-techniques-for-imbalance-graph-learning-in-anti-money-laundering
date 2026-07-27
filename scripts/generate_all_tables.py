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
    'GCN': ['NONE', 'RUS', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'REWEIGHTED_GRAPH_SMOTE'],
    'SAGE': ['NONE', 'RUS', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'REWEIGHTED_GRAPH_SMOTE'],
    'GAT': ['NONE', 'RUS', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'REWEIGHTED_GRAPH_SMOTE'],
    'GIN': ['NONE', 'RUS', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'REWEIGHTED_GRAPH_SMOTE'],
}

def infer_metadata(path: Path):
    name = path.name
    stem = name[:-12] if name.endswith('_summary.txt') else (name[:-4] if name.endswith('.txt') else name)
    
    dataset = 'unknown'
    for d_tag in DATASET_MAP.keys():
        if d_tag in path.name:
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
    for s in ['reweighted_graph_smote', 'graph_ensemble_smote', 'graph_smote', 'smote', 'rus']:
        if s in stem:
            sampling = s.upper()
            break
            
    is_tuned = 'tuned' in str(path.parent) or 'tuned' in path.name
    return method, dataset, ratio, sampling, is_tuned

def parse_both_metrics(path: Path):
    """
    Extracts both AUC-PRC and F1 scores simultaneously from a single log file content.
    Returns: (auc_val, f1_val)
    """
    auc_val = None
    f1_val = None
    try:
        content = path.read_text(encoding='utf-8').strip()
        if not content:
            return None, None
        
        lines = content.splitlines()
        # 1. Parse line by line to support multi-line summary formats
        for line in lines:
            if ':' in line:
                key, val = line.split(':', 1)
                key_upper = key.strip().upper()
                if 'AUC-PRC' in key_upper:
                    auc_val = val.strip()
                elif 'F1_99' in key_upper or key_upper == 'F1':
                    f1_val = val.strip()
                    
        # 2. Fallback: Parse inline tokens separated by commas
        if not auc_val or not f1_val:
            tokens = content.split(',')
            for token in tokens:
                if ':' in token:
                    key, val = token.split(':', 1)
                    key_clean = key.strip().upper()
                    if 'AUC-PRC' in key_clean:
                        auc_val = val.strip()
                    elif 'F1_99' in key_clean or key_clean == 'F1':
                        f1_val = val.strip()
    except Exception:
        pass
    return auc_val, f1_val

def clean_val(val_str):
    if not val_str or val_str == 'N/A' or val_str == '-':
        return None
    match = re.search(r'([0-9\.]+)', str(val_str))
    if match:
        return float(match.group(1))
    return None

def format_cell_value(val):
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
    
    # matrix[metric_type][dataset][sampling][method][ratio] = value
    matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    all_files = list(res_dir.glob('**/*.txt'))
    print(f"Total files found in res/: {len(all_files)}")
    
    for path in all_files:
        if path.name.startswith('.'):
            continue
        method, dataset, ratio, sampling, is_tuned = infer_metadata(path)
            
        if dataset in DATASET_MAP and method in METHODS:
            auc_val, f1_val = parse_both_metrics(path)
            is_summary = path.name.endswith('_summary.txt')
            
            if auc_val:
                current_auc = matrix['AUC-PRC'][dataset][sampling][method].get(ratio)
                if not current_auc or is_summary or ('±' not in str(current_auc) and '±' in str(auc_val)):
                    matrix['AUC-PRC'][dataset][sampling][method][ratio] = auc_val
                    
            if f1_val:
                current_f1 = matrix['F1_99'][dataset][sampling][method].get(ratio)
                if not current_f1 or is_summary or ('±' not in str(current_f1) and '±' in str(f1_val)):
                    matrix['F1_99'][dataset][sampling][method][ratio] = f1_val

    sorted_datasets = ['elliptic', 'hi_small', 'hi_medium', 'li_small', 'li_medium']
    
    for m_type in ['AUC-PRC', 'F1_99']:
        output_lines = []
        output_lines.append("Method X Sampling Table (LR=0.001, Gradient Clipping=1.0)\n")
        
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
                        val = matrix[m_type][d_key][s_key][method].get(ratio)
                        f_val = clean_val(val)
                        if f_val is not None:
                            vals_for_dataset.append(f_val)
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
                        val = matrix[m_type][d_key][s_key][method].get(ratio)
                        formatted_val = format_cell_value(val)
                        
                        f_val = clean_val(val)
                        max_val_for_col = col_max_vals[d_key]
                        
                        if f_val is not None and max_val_for_col > 0 and abs(f_val - max_val_for_col) < 1e-7:
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