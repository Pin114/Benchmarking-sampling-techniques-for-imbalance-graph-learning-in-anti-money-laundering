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

RATIOS = ['original', 'ratio_1to10', 'ratio_1to2', 'ratio_1to1']

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
    """提取數值中的 float，用於計算提升比例"""
    if not val_str or val_str == 'N/A' or val_str == '-':
        return None
    # 提取第一個浮點數
    match = re.search(r'([0-9\.]+)', str(val_str))
    if match:
        return float(match.group(1))
    return None

def main():
    res_dir = Path('res')
    if not res_dir.exists():
        print("Error: 'res' directory not found.")
        return

    tables_dir = Path('tables')
    tables_dir.mkdir(exist_ok=True)

    # 巢狀字典結構:
    # matrix[is_tuned][metric_type][dataset][sampling][method][ratio] = value
    matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))

    # 遞迴尋找 res 目錄下所有的 .txt 檔案
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

    # ========== 1. 輸出 Baseline (原版) 與 Tuned (調優版) 獨立表格 ==========
    for is_tuned, folder_name in [(False, "baseline"), (True, "tuned")]:
        for m_type in ['AUC-PRC', 'F1_99']:
            output_lines = []
            output_lines.append(f"# {folder_name.upper()} Resampling Ratio Impact Analysis Matrix ({m_type})\n")
            
            for d_key in sorted(DATASET_MAP.keys()):
                d_name = DATASET_MAP[d_key]
                output_lines.append(f"## Dataset: {d_name}\n")
                
                for s_key in ['NONE', 'RUS', 'SMOTE', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'REWEIGHTED_GRAPH_SMOTE']:
                    s_name = SAMPLING_TECHNIQUES[s_key]
                    
                    for ratio in RATIOS:
                        has_any_data = False
                        for method in METHODS:
                            if ratio in matrix[is_tuned][m_type][d_key][s_key][method]:
                                has_any_data = True
                                break
                        if not has_any_data:
                            continue
                            
                        output_lines.append(f"### Sampling Technique: {s_name} ({ratio})\n")
                        headers = ['Method / Baseline', f'Value ({m_type})']
                        output_lines.append('| ' + ' | '.join(headers) + ' |')
                        output_lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
                        
                        for method in METHODS:
                            val = matrix[is_tuned][m_type][d_key][s_key][method].get(ratio, 'N/A')
                            if val != 'N/A':
                                output_lines.append(f"| **{method}** | {val} |")
                        output_lines.append("\n---\n")
            
            output_file = tables_dir / f"ratio_comparison_tables_{folder_name}_{m_type.lower().replace('-', '_')}.md"
            output_file.write_text('\n'.join(output_lines), encoding='utf-8')
            print(f"[Success] Saved {folder_name} {m_type} table to: {output_file}")


    # ========== 2. 輸出 額外對照表格 (Baseline vs Tuned Ablation Study) ==========
    for m_type in ['AUC-PRC', 'F1_99']:
        comparison_lines = []
        comparison_lines.append(f"# Ablation Study & Parameter Tuning Comparison ({m_type})\n")
        comparison_lines.append("This table contrasts the **Baseline (LR=0.05, No Clip)** vs **Tuned (LR=0.001, Gradient Clip)** settings.\n")
        
        for d_key in sorted(DATASET_MAP.keys()):
            d_name = DATASET_MAP[d_key]
            comparison_lines.append(f"## Dataset: {d_name}\n")
            
            for s_key in ['NONE', 'RUS', 'SMOTE', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'REWEIGHTED_GRAPH_SMOTE']:
                s_name = SAMPLING_TECHNIQUES[s_key]
                
                for ratio in RATIOS:
                    # 檢查在此組合下是否有任一邊的資料
                    has_data = False
                    for method in METHODS:
                        if (ratio in matrix[False][m_type][d_key][s_key][method]) or (ratio in matrix[True][m_type][d_key][s_key][method]):
                            has_data = True
                            break
                    if not has_data:
                        continue
                        
                    comparison_lines.append(f"### Comparison: {s_name} ({ratio})\n")
                    headers = ['Method', 'Baseline (No Clip, LR=0.05)', 'Tuned (Clip=1.0, LR=0.001)', 'Absolute Lift']
                    comparison_lines.append('| ' + ' | '.join(headers) + ' |')
                    comparison_lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
                    
                    for method in METHODS:
                        val_base = matrix[False][m_type][d_key][s_key][method].get(ratio, '-')
                        val_tuned = matrix[True][m_type][d_key][s_key][method].get(ratio, '-')
                        
                        if val_base == '-' and val_tuned == '-':
                            continue
                            
                        # 計算提升
                        float_base = clean_val(val_base)
                        float_tuned = clean_val(val_tuned)
                        
                        if float_base is not None and float_tuned is not None:
                            diff = float_tuned - float_base
                            # 格式化輸出
                            sign = "+" if diff >= 0 else ""
                            lift_str = f"**{sign}{diff:.4f}**"
                        else:
                            lift_str = "N/A"
                            
                        comparison_lines.append(f"| **{method}** | {val_base} | {val_tuned} | {lift_str} |")
                    comparison_lines.append("\n---\n")
                    
        output_file = tables_dir / f"ablation_comparison_{m_type.lower().replace('-', '_')}.md"
        output_file.write_text('\n'.join(comparison_lines), encoding='utf-8')
        print(f"[Success] Saved Ablation Comparison Table to: {output_file}")

if __name__ == "__main__":
    main()
