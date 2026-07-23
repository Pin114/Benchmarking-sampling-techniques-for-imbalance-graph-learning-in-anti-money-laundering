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
            
    if 'F1_99' in stem_upper:
        metric_type = 'F1_99'
    else:
        metric_type = 'AUC-PRC'
        
    is_tuned = 'tuned' in str(path.parent) or 'tuned' in path.name
    return method, dataset, ratio, sampling, metric_type, is_tuned

def parse_metrics(path: Path, metric_type: str):
    try:
        content = path.read_text(encoding='utf-8').strip()
        if not content:
            return None
        tokens = content.split(',')
        for token in tokens:
            if ':' in token:
                key, val = token.split(':', 1)
                key_clean = key.strip().upper()
                if metric_type == 'AUC-PRC' and 'AUC-PRC' in key_clean:
                    return val.strip()
                elif metric_type == 'F1_99' and 'F1_99' in key_clean:
                    return val.strip()
                    
        # Fallback: If no key-value pair is found, check if the content is a single numeric value
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
        method, dataset, ratio, sampling, metric_type, is_tuned = infer_metadata(path)
            
        if dataset in DATASET_MAP and method in METHODS:
            val = parse_metrics(path, metric_type)
            if val:
                is_summary = path.name.endswith('_summary.txt')
                current_val = matrix[metric_type][dataset][sampling][method].get(ratio)
                if not current_val or is_summary or ('±' not in str(current_val) and '±' in str(val)):
                    matrix[metric_type][dataset][sampling][method][ratio] = val

    sorted_datasets = ['elliptic', 'hi_small', 'hi_medium', 'li_small', 'li_medium']
    
    for m_type in ['AUC-PRC', 'F1_99']:
        output_lines = []
        output_lines.append("Method X Sampling Table (LR=0.001, Gradient Clipping=1.0)\n")
        
        for ratio in RATIOS:
            ratio_title = RATIO_DISPLAY[ratio]
            output_lines.append(f"## 評估設定比例: {ratio_title}\n")
            
            # build the table header
            headers = ['Method', 'Sampling', 'ELLIPTIC', 'IBM HI-SMALL', 'IBM HI-MEDIUM', 'IBM LI-SMALL', 'IBM LI-MEDIUM']
            output_lines.append('| ' + ' | '.join(headers) + ' |')
            output_lines.append('| ' + ' | '.join([':---', ':---', ':---:', ':---:', ':---:', ':---:', ':---:']) + ' |')
            
            for method in METHODS:
                samplings = METHOD_SAMPLING_MAP.get(method, [])
                
                # calculate the maximum value for each dataset column to highlight the best performance
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
                    s_name = SAMPLING_TECHNIQUES.get(s_key, s_key)
                    s_display = "None (Baseline)" if s_key == 'NONE' else s_name
                    
                    row_cells = []
                    # only display the method name for the first sampling technique row
                    if idx == 0:
                        row_cells.append(f"**{method}**")
                    else:
                        row_cells.append("")
                        
                    row_cells.append(s_display)
                    
                    # populate the performance values for each dataset
                    for d_key in sorted_datasets:
                        val = matrix[m_type][d_key][s_key][method].get(ratio)
                        formatted_val = format_cell_value(val)
                        
                        f_val = clean_val(val)
                        max_val_for_col = col_max_vals[d_key]
                        
                        # emphasize the best value in the column (dataset) for this method and ratio
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
