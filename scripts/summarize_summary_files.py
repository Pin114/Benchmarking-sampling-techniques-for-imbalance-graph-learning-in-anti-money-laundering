#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from collections import defaultdict

def parse_summary_file(path: Path):
    metrics = {}
    try:
        content = path.read_text(encoding='utf-8').strip().splitlines()
    except Exception:
        return None

    for line in content:
        if ':' not in line:
            continue
        key, value_part = line.split(':', 1)
        metrics[key.strip()] = value_part.strip()
    return metrics

def infer_metadata(path: Path):
    name = path.name
    stem = name[:-12] if name.endswith('_summary.txt') else (name[:-4] if name.endswith('.txt') else name)

    # 1. 識別模型方法
    method = 'UNKNOWN'
    for m in ['GCN', 'SAGE', 'GAT', 'GIN', 'INTRINSIC', 'POSITIONAL', 'DEEPWALK', 'NODE2VEC']:
        if m in stem.upper():
            method = m
            break
            
    if '_f1_90_' in stem:
        method = f"{method}_F1_90"
    elif '_f1_99_' in stem:
        method = f"{method}_F1_99"

    # 2. 辨識資料集標籤
    dataset = 'unknown'
    for d in ['hi_small', 'hi_medium', 'hi_large', 'li_small', 'li_medium', 'li_large', 'elliptic']:
        if d in stem:
            dataset = d
            break

    # 3. 識別採樣比例
    ratio = 'original'
    for r in ['original', 'ratio_1to10', 'ratio_1to2', 'ratio_1to1']:
        if r in stem:
            ratio = r
            break

    # 4. 識別採樣技術
    sampling = 'NONE'
    for s in ['reweighted_graph_smote', 'graph_ensemble_smote', 'graph_smote', 'smote', 'rus']:
        if s in stem:
            sampling = s.upper()
            break

    return {
        'method': method,
        'dataset': dataset,
        'ratio': ratio,
        'sampling': sampling,
    }

def main():
    parser = argparse.ArgumentParser(description='Generate one clear table per dataset.')
    parser.add_argument('--res-dir', default='res', help='Directory containing summary files')
    parser.add_argument('--output', default='res/ratio_comparison_tables.md', help='Output markdown file path')
    args = parser.parse_args()

    res_dir = Path(args.res_dir)
    if not res_dir.exists():
        print(f'No such directory: {res_dir}', file=sys.stderr)
        sys.exit(1)

    summary_files = list(res_dir.glob('*_summary.txt'))
    
    # 修正字典結構：改用 tuple 作為複合鍵，徹底平坦化，避免 defaultdict 嵌套覆蓋問題
    # 結構：(dataset, method) -> (sampling, ratio) -> auc_prc
    matrix = defaultdict(dict)
    
    # 用來收集所有實際存在、看過的 datasets 和 methods
    observed_datasets = set()
    observed_methods = defaultdict(set)

    # 統一定義我們想要顯示的所有（採樣技術, 比例）欄位組合
    config_columns = [
        ('NONE', 'original'),
        ('NONE', 'ratio_1to10'),
        ('NONE', 'ratio_1to2'),
        ('NONE', 'ratio_1to1'),
        ('RUS', 'original'),
        ('RUS', 'ratio_1to10'),
        ('RUS', 'ratio_1to2'),
        ('RUS', 'ratio_1to1'),
        ('SMOTE', 'original'),
        ('SMOTE', 'ratio_1to10'),
        ('SMOTE', 'ratio_1to2'),
        ('SMOTE', 'ratio_1to1'),
        ('GRAPH_SMOTE', 'original'),
        ('GRAPH_SMOTE', 'ratio_1to10'),
        ('GRAPH_SMOTE', 'ratio_1to2'),
        ('GRAPH_SMOTE', 'ratio_1to1'),
        ('GRAPH_ENSEMBLE_SMOTE', 'original'),
        ('GRAPH_ENSEMBLE_SMOTE', 'ratio_1to10'),
        ('GRAPH_ENSEMBLE_SMOTE', 'ratio_1to2'),
        ('GRAPH_ENSEMBLE_SMOTE', 'ratio_1to1'),
        ('REWEIGHTED_GRAPH_SMOTE', 'original'),
        ('REWEIGHTED_GRAPH_SMOTE', 'ratio_1to10'),
        ('REWEIGHTED_GRAPH_SMOTE', 'ratio_1to2'),
        ('REWEIGHTED_GRAPH_SMOTE', 'ratio_1to1'),
    ]

    for path in summary_files:
        meta = infer_metadata(path)
        metrics = parse_summary_file(path) or {}
        auc_prc = metrics.get('AUC-PRC', 'N/A')
        
        d_key = meta['dataset']
        m_key = meta['method']
        
        observed_datasets.add(d_key)
        observed_methods[d_key].add(m_key)
        
        config_key = (meta['sampling'], meta['ratio'])
        matrix[(d_key, m_key)][config_key] = auc_prc

    output_lines = []
    output_lines.append("# Resampling Ratio Impact Analysis Matrix (AUC-PRC)\n")

    # 依資料集迭代，一個資料集只生一個乾淨表格
    for dataset in sorted(observed_datasets):
        output_lines.append(f"## 📁 Dataset: {dataset.upper()}")
        output_lines.append(f"橫向對比不同的不平衡重採樣技術與比例對模型最終泛化表現的實質影響：\n")
        
        # 建立欄位表頭
        headers = ['Method / Baseline'] + [f"{s}<br>{r.replace('ratio_', '')}" for s, r in config_columns]
        output_lines.append('| ' + ' | '.join(headers) + ' |')
        output_lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
        
        # 填入該資料集底下的所有模型數據
        for method in sorted(observed_methods[dataset]):
            row_cells = [f"**{method}**"]
            for config_key in config_columns:
                cell_value = matrix[(dataset, method)].get(config_key, 'N/A')
                row_cells.append(cell_value)
            output_lines.append('| ' + ' | '.join(row_cells) + ' |')
        
        output_lines.append("\n" + "---" + "\n")

    final_content = '\n'.join(output_lines)
    print(final_content)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(final_content + '\n', encoding='utf-8')
    print(f'\n[SUCCESS] Unified per-dataset tables saved to: {out_path}')

if __name__ == '__main__':
    main