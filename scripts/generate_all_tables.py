#!/usr/bin/env python3
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

# 支援的所有模型方法
METHODS = ['INTRINSIC', 'POSITIONAL', 'DEEPWALK', 'NODE2VEC', 'GCN', 'SAGE', 'GAT', 'GIN']

# 採樣技術列表
SAMPLING_TECHNIQUES = {
    'NONE': '無採樣基準 (NONE)',
    'RUS': '隨機下採樣 (RUS)',
    'SMOTE': '合成少數類過採樣 (SMOTE)',
    'GRAPH_SMOTE': '圖合成過採樣 (GRAPH_SMOTE)',
    'GRAPH_ENSEMBLE_SMOTE': '圖集成過採樣 (GRAPH_ENSEMBLE_SMOTE)',
    'REWEIGHTED_GRAPH_SMOTE': '重加權圖過採樣 (REWEIGHTED_GRAPH_SMOTE)'
}

# 欄位定義 (重採樣目標比例)
RATIOS = ['original', 'ratio_1to10', 'ratio_1to2', 'ratio_1to1']

def infer_metadata(path: Path):
    """
    從檔案名稱推導模型、資料集、比例、採樣技術與指標類型
    """
    name = path.name
    stem = name[:-12] if name.endswith('_summary.txt') else (name[:-4] if name.endswith('.txt') else name)
    stem_upper = stem.upper()
    
    # 1. 識別模型方法
    method = 'UNKNOWN'
    for m in METHODS:
        if m in stem_upper:
            method = m
            break
            
    # 2. 識別資料集
    dataset = 'unknown'
    for d in DATASET_MAP.keys():
        if d in stem:
            dataset = d
            break
    # 兼容舊命名的 `_ibm_` 對應到 `hi_small`
    if '_ibm_' in stem:
        dataset = 'hi_small'
            
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
            
    # 5. 識別指標類型 (AUC-PRC, F1_90, F1_99)
    if '_F1_90_' in stem_upper:
        metric_type = 'F1_90'
    elif '_F1_99_' in stem_upper:
        metric_type = 'F1_99'
    else:
        metric_type = 'AUC-PRC'
        
    return method, dataset, ratio, sampling, metric_type

def parse_metrics(path: Path, metric_type: str):
    """
    從檔案內容解析指標數值。支援：
    1. 含有 "mean ± std" 結構的 summary 檔案： "AUC-PRC: 0.123456 ± 0.0012"
    2. 單次運行的 txt 檔案： "AUC-PRC: 0.123456, F1_90: 0.0312"
    """
    try:
        content = path.read_text(encoding='utf-8').strip()
        if not content:
            return None
            
        # 1. 如果是 summary 檔案，提取對應指標的 "0.XXXXXX ± 0.YYYYYY"
        if path.name.endswith('_summary.txt'):
            for line in content.splitlines():
                if ':' in line:
                    key, val = line.split(':', 1)
                    if key.strip() == metric_type:
                        return val.strip()
                    # 如果 metric_type 是 AUC-PRC 且檔案內容只有單行 "0.12345 ± 0.01" 則直接回傳
                    if metric_type == 'AUC-PRC' and 'AUC-PRC' in key:
                        return val.strip()
        
        # 2. 如果是單次運行檔案，解析 "AUC-PRC: 0.XXXX, F1_90: 0.YYYY" 等格式
        tokens = content.split(',')
        for token in tokens:
            if ':' in token:
                key, val = token.split(':', 1)
                key_clean = key.strip().upper()
                if metric_type == 'AUC-PRC' and 'AUC-PRC' in key_clean:
                    return val.strip()
                elif metric_type == 'F1_90' and 'F1_90' in key_clean:
                    return val.strip()
                elif metric_type == 'F1_99' and 'F1_99' in key_clean:
                    return val.strip()
                
        # 兜底：如果檔案只有單個數值，且請求的是 AUC-PRC
        if len(content) < 30 and metric_type == 'AUC-PRC' and ':' not in content:
            try:
                float(content)
                return content
            except ValueError:
                pass
    except Exception:
        pass
    return None

def main():
    res_dir = Path('res')
    if not res_dir.exists():
        print("Error: 'res' directory not found. Please run this script in the root of your workspace.")
        return

    # 巢狀字典結構: matrix[metric_type][dataset][sampling][method][ratio] = value
    matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    # 遞迴尋找 res 目錄下所有的 .txt 檔案
    all_files = list(res_dir.glob('**/*.txt'))
    print(f"Total files found in res/: {len(all_files)}")
    
    for path in all_files:
        if path.name == 'ratio_comparison_tables.md' or path.name.startswith('.'):
            continue
        
        method, dataset, ratio, sampling, metric_type = infer_metadata(path)
        
        if dataset in DATASET_MAP and method in METHODS:
            val = parse_metrics(path, metric_type)
            if val:
                # 優先保留 summary.txt 的結果 (帶有標準差)
                is_summary = path.name.endswith('_summary.txt')
                current_val = matrix[metric_type][dataset][sampling][method].get(ratio)
                
                if not current_val or is_summary or ('±' not in str(current_val) and '±' in str(val)):
                    matrix[metric_type][dataset][sampling][method][ratio] = val

    # 針對不同的指標分別輸出報告
    for m_type in ['AUC-PRC', 'F1_90', 'F1_99']:
        output_lines = []
        output_lines.append(f"# Resampling Ratio Impact Analysis Matrix ({m_type})\n")
        output_lines.append("> 橫向對比不同的不平衡重採樣技術與比例對模型最終泛化表現的實質影響：\n\n---\n")
        
        # 遍歷5個資料集
        for d_key in sorted(DATASET_MAP.keys()):
            d_name = DATASET_MAP[d_key]
            output_lines.append(f"## 📁 Dataset: {d_name}\n")
            
            # 遍歷所有的採樣技術，為每個技術產生一個對比表
            for s_key in ['NONE', 'RUS', 'SMOTE', 'GRAPH_SMOTE', 'GRAPH_ENSEMBLE_SMOTE', 'REWEIGHTED_GRAPH_SMOTE']:
                s_name = SAMPLING_TECHNIQUES[s_key]
                
                # 檢查這個資料集與技術下，是否有任何模型的數據。若完全沒數據則跳過不畫，保持排版乾淨。
                has_any_data = False
                for method in METHODS:
                    for ratio in RATIOS:
                        if ratio in matrix[m_type][d_key][s_key][method]:
                            has_any_data = True
                            break
                
                if not has_any_data:
                    continue
                    
                output_lines.append(f"### ⚙️ Sampling Technique: {s_name}\n")
                
                # 輸出表頭
                headers = ['Method / Baseline', 'Imbalance original', 'Imbalance ratio_1to10', 'Imbalance ratio_1to2', 'Imbalance ratio_1to1']
                output_lines.append('| ' + ' | '.join(headers) + ' |')
                output_lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
                
                # 輸出每一列模型數據
                for method in METHODS:
                    row_cells = [f"**{method}**"]
                    row_has_data = False
                    for ratio in RATIOS:
                        val = matrix[m_type][d_key][s_key][method].get(ratio, 'N/A')
                        if val != 'N/A':
                            row_has_data = True
                        row_cells.append(str(val))
                    
                    if row_has_data:
                        output_lines.append('| ' + ' | '.join(row_cells) + ' |')
                
                output_lines.append("\n---\n")
            
            output_lines.append("\n================================================================================\n")
            
        # 寫入對應指標的獨立 Markdown 表格中
        output_file = res_dir / f"ratio_comparison_tables_{m_type.lower().replace('-', '_')}.md"
        output_file.write_text('\n'.join(output_lines), encoding='utf-8')
        print(f"[Success] Unified tables for {m_type} saved to: {output_file}")

if __name__ == "__main__":
    main()
