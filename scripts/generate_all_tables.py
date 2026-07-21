#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
from collections import defaultdict

# 5個實際執行的資料集分類對照
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

# 模型與其適用的採樣方法映射 (避免在表格中產生無效的 N/A 橫列)
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

# 比例顯示對照表
RATIO_DISPLAY = {
    'original': '原始比例 (Original)',
    'ratio_1to10': '比例 1:10 (Ratio)',
    'ratio_1to2': '比例 1:2 (Ratio)',
    'ratio_1to1': '比例 1:1 (Ratio)'
}

def infer_metadata(path: Path):
    name = path.name
    # 判斷是否為 summary 檔
    stem = name[:-12] if name.endswith('_summary.txt') else (name[:-4] if name.endswith('.txt') else name)
    
    # 1. 識別資料集
    dataset = 'unknown'
    for d_tag in DATASET_MAP.keys():
        if d_tag in path.name:
            dataset = d_tag
            break
            
    # 2. 識別模型方法
    method = 'UNKNOWN'
    stem_upper = stem.upper()
    for m in METHODS:
        if m in stem_upper:
            method = m
            break
            
    # 3. 識別比例
    ratio = 'original'
    for r in RATIOS:
        if r in stem:
            ratio = r
            break
            
    # 4. 識別採樣技術
    sampling = 'NONE'
    for s in ['reweighted_graph_smote', 'graph_ensemble_smote', 'graph_smote', 'smote', 'rus']:
        if s in stem:
            sampling = s.upper()
            break
            
    # 5. 識別指標類型
    if 'F1_99' in stem_upper:
        metric_type = 'F1_99'
    else:
        metric_type = 'AUC-PRC'
        
    # 6. 識別是否為 Tuned
    is_tuned = 'tuned' in str(path.parent) or 'tuned' in path.name
    return method, dataset, ratio, sampling, metric_type, is_tuned

def parse_metrics(path: Path, metric_type: str):
    try:
        content = path.read_text(encoding='utf-8').strip()
        if not content:
            return None
            
        # 1. 如果是 summary 檔案，提取對應指標的 "0.XXXXXX ± 0.YYYYYY"
        if path.name.endswith('_summary.txt'):
            for line in content.splitlines():
                if ':' in line:
                    key, val = line.split(':', 1)
                    if metric_type == 'AUC-PRC' and 'AUC-PRC' in key.strip().upper():
                        return val.strip()
                    elif metric_type == 'F1_99' and 'F1_99' in key.strip().upper():
                        return val.strip()
            # 兜底：如果檔案只有單行
            if ':' not in content and '±' in content:
                return content.strip()
                
        # 2. 如果是單次運行檔案，解析 "AUC-PRC: 0.XXXX, F1_99: 0.YYYY"
        tokens = content.split(',')
        for token in tokens:
            if ':' in token:
                key, val = token.split(':', 1)
                key_clean = key.strip().upper()
                if metric_type == 'AUC-PRC' and 'AUC-PRC' in key_clean:
                    return val.strip()
                elif metric_type == 'F1_99' and 'F1_99' in key_clean:
                    return val.strip()
                    
        # 兜底：如果檔案只有單個數值
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
    """提取數值中的 float，用於找出最優值與計算提升比例"""
    if not val_str or val_str == 'N/A' or val_str == '-':
        return None
    match = re.search(r'([0-9\.]+)', str(val_str))
    if match:
        return float(match.group(1))
    return None

def format_cell_value(val):
    """格式化輸出數值，保留小數點後四位"""
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
                # 優先保留 summary.txt 的結果 (帶有標準差)
                if not current_val or is_summary or ('±' not in str(current_val) and '±' in str(val)):
                    matrix[is_tuned][metric_type][dataset][sampling][method][ratio] = val
                    
    # ========== 1. 輸出 Baseline 與 Tuned 縱向「模型 × 採樣」聯立大表 ==========
    for is_tuned, folder_name in [(False, "baseline"), (True, "tuned")]:
        for m_type in ['AUC-PRC', 'F1_99']:
            output_lines = []
            output_lines.append(f"# {folder_name.upper()} 模型 × 採樣 縱向聯立分析矩陣 ({m_type})\n")
            output_lines.append("本表格以模型為主軸，橫向展示在不同不平衡比例下，各採樣技術的性能演變。**粗體高亮**代表各模型在所有比例下的最優值。\n")
            
            for d_key in sorted(DATASET_MAP.keys()):
                d_name = DATASET_MAP[d_key]
                output_lines.append(f"## Dataset: {d_name}\n")
                
                headers = ['模型 (Method)', '採樣技術 (Sampling)', '原始比例 (Original)', '比例 1:10 (Ratio)', '比例 1:2 (Ratio)', '比例 1:1 (Ratio)']
                output_lines.append('| ' + ' | '.join(headers) + ' |')
                output_lines.append('| ' + ' | '.join([':---', ':---', ':---:', ':---:', ':---:', ':---:']) + ' |')
                
                for method in METHODS:
                    samplings = METHOD_SAMPLING_MAP.get(method, [])
                    
                    # 計算該 method 在此 dataset 所有採樣和比例下的最大值，用於高亮
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
                        # 模型名稱僅在該模型的第一行顯示，其餘留白，維持學術美感
                        if idx == 0:
                            row_cells.append(f"**{method}**")
                        else:
                            row_cells.append("")
                            
                        row_cells.append(s_display)
                        
                        for ratio in RATIOS:
                            val = matrix[is_tuned][m_type][d_key][s_key][method].get(ratio)
                            formatted_val = format_cell_value(val)
                            
                            # 如果是最大值，進行高亮
                            f_val = clean_val(val)
                            if f_val is not None and max_val > 0 and abs(f_val - max_val) < 1e-7:
                                formatted_val = f"**{formatted_val}**"
                                
                            row_cells.append(formatted_val)
                            
                        output_lines.append('| ' + ' | '.join(row_cells) + ' |')
                    
                    # 每個模型結束後補一個表格分隔行 (空列)，增加易讀性
                    output_lines.append('| | | | | | |')
                    
                output_lines.append("\n---\n")
                
            output_file = tables_dir / f"ratio_comparison_tables_{folder_name}_{m_type.lower().replace('-', '_')}.md"
            output_file.write_text('\n'.join(output_lines), encoding='utf-8')
            print(f"[Success] Saved {folder_name} {m_type} table to: {output_file}")
            
    # ========== 2. 輸出 額外對照表格 (Baseline vs Tuned Ablation Study) ==========
    for m_type in ['AUC-PRC', 'F1_99']:
        comparison_lines = []
        comparison_lines.append(f"# 參數調優與消融對比大表 ({m_type})\n")
        comparison_lines.append("本表格橫向對比 **Baseline (LR=0.05, No Clip)** 與 **Tuned (LR=0.001, Gradient Clip=1.0)**，並計算絕對性能提升 (Absolute Lift)。\n")
        
        for d_key in sorted(DATASET_MAP.keys()):
            d_name = DATASET_MAP[d_key]
            comparison_lines.append(f"## Dataset: {d_name}\n")
            
            for ratio in RATIOS:
                ratio_name = RATIO_DISPLAY[ratio]
                comparison_lines.append(f"### 設定比例: {ratio_name}\n")
                
                headers = ['模型 (Method)', '採樣技術 (Sampling)', 'Baseline (LR=0.05)', 'Tuned (LR=0.001)', '絕對提升 (Absolute Lift)']
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
                            # 高亮正提升
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
