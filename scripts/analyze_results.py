"""
分析 72 個訓練結果，驗證 APATE 假設和採樣技術效果
"""
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

# Setup paths
DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIR + "/../")
sys.path.append(DIR + "/../")

def parse_results():
    """解析所有 72 個訓練結果文件"""
    res_dir = "res"
    results = []
    
    for filename in os.listdir(res_dir):
        if not filename.endswith('.txt'):
            continue
        
        # Parse filename: {method}_params_ibm_{ratio}{sampling}.txt
        # Examples:
        # - intrinsic_params_ibm_original.txt
        # - intrinsic_params_ibm_original_smote.txt
        # - gcn_params_ibm_ratio_1to1_graph_smote.txt
        
        base = filename.replace("_params_ibm_", "|").replace(".txt", "") # ibm or elliptic
        parts = base.split("|")
        
        if len(parts) != 2:
            continue
        
        method = parts[0]
        rest = parts[1]
        
        # Determine ratio and sampling
        if "ratio_1to1" in rest:
            ratio = "1:1"
            rest = rest.replace("ratio_1to1_", "").replace("ratio_1to1", "")
        elif "ratio_1to2" in rest:
            ratio = "2:1"
            rest = rest.replace("ratio_1to2_", "").replace("ratio_1to2", "")
        else:
            ratio = "Original"
        
        # Determine sampling
        if "graph_smote" in rest:
            sampling = "GraphSMOTE"
        elif "smote" in rest:
            sampling = "SMOTE"
        elif "rus" in rest:
            sampling = "RUS"
        else:
            sampling = "None"
        
        # Read AUC-PRC score
        filepath = os.path.join(res_dir, filename)
        try:
            with open(filepath) as f:
                content = f.read().strip()
            if "AUC-PRC:" in content:
                score = float(content.split("AUC-PRC:")[1].strip())
                results.append({
                    'method': method,
                    'ratio': ratio,
                    'sampling': sampling,
                    'score': score,
                    'filename': filename
                })
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("=" * 100)
    print("分析 AML 不平衡圖學習採樣技術的 72 個訓練結果")
    print("=" * 100)
    
    # Parse results
    df = parse_results()
    
    print(f"\n✅ 成功加載 {len(df)}/72 個結果\n")
    
    if len(df) == 0:
        print("❌ 沒有找到結果文件")
        sys.exit(1)
    
    # ========== 分析 1: 按比例分析 ==========
    print("=" * 100)
    print("1️⃣  按類別不平衡比例分析（驗證 APATE 假設：2:1 是否最優）")
    print("=" * 100)
    
    ratio_analysis = df.groupby('ratio')['score'].agg(['count', 'mean', 'std', 'min', 'max'])
    ratio_analysis = ratio_analysis.sort_values('mean', ascending=False)
    print("\n", ratio_analysis)
    
    # Find best ratio
    best_ratio = ratio_analysis['mean'].idxmax()
    print(f"\n🏆 最佳比例: {best_ratio} (平均 AUC-PRC: {ratio_analysis.loc[best_ratio, 'mean']:.6f})")
    
    # ========== 分析 2: 按採樣技術分析 ==========
    print("\n" + "=" * 100)
    print("2️⃣  按採樣技術分析（評估 SMOTE/GraphSMOTE 效果）")
    print("=" * 100)
    
    sampling_analysis = df.groupby('sampling')['score'].agg(['count', 'mean', 'std', 'min', 'max'])
    sampling_analysis = sampling_analysis.sort_values('mean', ascending=False)
    print("\n", sampling_analysis)
    
    best_sampling = sampling_analysis['mean'].idxmax()
    worst_sampling = sampling_analysis['mean'].idxmin()
    improvement = (sampling_analysis.loc[best_sampling, 'mean'] - sampling_analysis.loc[worst_sampling, 'mean']) / sampling_analysis.loc[worst_sampling, 'mean'] * 100
    print(f"\n🏆 最佳採樣: {best_sampling} (平均 AUC-PRC: {sampling_analysis.loc[best_sampling, 'mean']:.6f})")
    print(f"📈 相比最差採樣 ({worst_sampling}): {improvement:+.1f}%")
    
    # ========== 分析 3: 按方法分析 ==========
    print("\n" + "=" * 100)
    print("3️⃣  按方法分析（比較 8 個方法的性能）")
    print("=" * 100)
    
    method_analysis = df.groupby('method')['score'].agg(['count', 'mean', 'std', 'min', 'max'])
    method_analysis = method_analysis.sort_values('mean', ascending=False)
    print("\n", method_analysis)
    
    best_method = method_analysis['mean'].idxmax()
    print(f"\n🏆 最佳方法: {best_method} (平均 AUC-PRC: {method_analysis.loc[best_method, 'mean']:.6f})")
    
    # ========== 分析 4: 交叉分析 (Ratio × Sampling) ==========
    print("\n" + "=" * 100)
    print("4️⃣  交叉分析：比例 × 採樣技術")
    print("=" * 100)
    
    cross_analysis = df.groupby(['ratio', 'sampling'])['score'].agg(['count', 'mean'])
    cross_pivot = df.pivot_table(values='score', index='ratio', columns='sampling', aggfunc='mean')
    print("\n", cross_pivot)
    
    # ========== 分析 5: 交叉分析 (Ratio × Method) ==========
    print("\n" + "=" * 100)
    print("5️⃣  交叉分析：比例 × 方法")
    print("=" * 100)
    
    method_ratio_pivot = df.pivot_table(values='score', index='method', columns='ratio', aggfunc='mean')
    print("\n", method_ratio_pivot)
    
    # ========== 分析 6: 交叉分析 (Method × Sampling) ==========
    print("\n" + "=" * 100)
    print("6️⃣  交叉分析：方法 × 採樣技術")
    print("=" * 100)
    
    method_sampling_pivot = df.pivot_table(values='score', index='method', columns='sampling', aggfunc='mean')
    print("\n", method_sampling_pivot)
    
    # ========== 分析 7: 最佳和最差組合 ==========
    print("\n" + "=" * 100)
    print("7️⃣  最佳和最差組合")
    print("=" * 100)
    
    top5 = df.nlargest(5, 'score')[['method', 'ratio', 'sampling', 'score']]
    print("\n🏆 Top 5 最佳組合:")
    for idx, row in top5.iterrows():
        print(f"   {row['method']:12} | {row['ratio']:8} | {row['sampling']:12} → {row['score']:.6f}")
    
    bottom5 = df.nsmallest(5, 'score')[['method', 'ratio', 'sampling', 'score']]
    print("\n❌ Bottom 5 最差組合:")
    for idx, row in bottom5.iterrows():
        print(f"   {row['method']:12} | {row['ratio']:8} | {row['sampling']:12} → {row['score']:.6f}")
    
    # ========== 統計摘要 ==========
    print("\n" + "=" * 100)
    print("📊 統計摘要")
    print("=" * 100)
    print(f"\n全體結果統計:")
    print(f"  • 平均 AUC-PRC:  {df['score'].mean():.6f}")
    print(f"  • 標準差:       {df['score'].std():.6f}")
    print(f"  • 最高分:       {df['score'].max():.6f}")
    print(f"  • 最低分:       {df['score'].min():.6f}")
    print(f"  • 中位數:       {df['score'].median():.6f}")
    
    # ========== APATE 假設驗證 ==========
    print("\n" + "=" * 100)
    print("🎯 APATE 假設驗證（2:1 比例是否為 AML 最優）")
    print("=" * 100)
    
    ratio_means = df.groupby('ratio')['score'].mean().sort_values(ascending=False)
    print("\n按平均 AUC-PRC 排序:")
    for i, (ratio, score) in enumerate(ratio_means.items(), 1):
        marker = "✅ 假設驗證" if ratio == "2:1" and i == 1 else "❌" if ratio == "2:1" and i != 1 else ""
        print(f"  {i}. {ratio:10} → {score:.6f} {marker}")
    
    if ratio_means.index[0] == "2:1":
        print("\n✅ APATE 假設得到驗證：2:1 比例確實表現最佳！")
    else:
        print(f"\n⚠️  APATE 假設需要修正：{ratio_means.index[0]} 表現最佳，而非 2:1")
    
    print("\n" + "=" * 100)
