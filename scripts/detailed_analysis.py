
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

# Setup
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
        
        base = filename.replace("_params_ibm_", "|").replace(".txt", "")
        parts = base.split("|")
        
        if len(parts) != 2:
            continue
        
        method = parts[0]
        rest = parts[1]
        
        # Determine ratio
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
        
        # Read score
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
            pass
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Parse results
    df = parse_results()
    
    print("=" * 100)
    print("📊 AML 不平衡圖學習採樣技術分析 - 詳細報告")
    print("=" * 100)
    print(f"\n✅ 成功加載 {len(df)}/72 個結果\n")
    
    # ========== 1. 按比例分析 ==========
    print("\n" + "=" * 100)
    print("1️⃣  類別不平衡比例分析（驗證 APATE 假設）")
    print("=" * 100)
    
    ratio_stats = df.groupby('ratio')['score'].agg(['count', 'mean', 'std', 'min', 'max']).round(6)
    ratio_stats = ratio_stats.sort_values('mean', ascending=False)
    print("\n", ratio_stats)
    
    # Calculate improvement
    best_ratio_score = ratio_stats['mean'].iloc[0]
    for idx, (ratio, row) in enumerate(ratio_stats.iterrows(), 1):
        improvement = (row['mean'] - best_ratio_score) / best_ratio_score * 100 if idx > 1 else 0
        symbol = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
        print(f"\n{symbol} {idx}. {ratio:8} → 平均: {row['mean']:.6f}, 標準差: {row['std']:.6f} {f'({improvement:+.1f}%)' if improvement != 0 else ''}")
    
    # ========== 2. 按採樣技術分析 ==========
    print("\n" + "=" * 100)
    print("2️⃣  採樣技術效果分析")
    print("=" * 100)
    
    sampling_stats = df.groupby('sampling')['score'].agg(['count', 'mean', 'std', 'min', 'max']).round(6)
    sampling_stats = sampling_stats.sort_values('mean', ascending=False)
    print("\n", sampling_stats)
    
    best_sampling_score = sampling_stats['mean'].iloc[0]
    worst_sampling = sampling_stats['mean'].idxmin()
    improvement = (sampling_stats['mean'].iloc[0] - sampling_stats['mean'].iloc[-1]) / sampling_stats['mean'].iloc[-1] * 100
    
    print(f"\n🏆 最佳採樣技術: {sampling_stats.index[0]} (AUC-PRC: {sampling_stats['mean'].iloc[0]:.6f})")
    print(f"📈 相比最差 ({worst_sampling}): {improvement:+.1f}%")
    
    # ========== 3. 按方法分析 ==========
    print("\n" + "=" * 100)
    print("3️⃣  方法性能對比")
    print("=" * 100)
    
    method_stats = df.groupby('method')['score'].agg(['count', 'mean', 'std', 'min', 'max']).round(6)
    method_stats = method_stats.sort_values('mean', ascending=False)
    print("\n", method_stats)
    
    print("\n方法分類:")
    feature_methods = ['intrinsic', 'positional']
    embedding_methods = ['deepwalk', 'node2vec']
    gnn_methods = ['gcn', 'sage', 'gat', 'gin']
    
    for method, row in method_stats.iterrows():
        if method in feature_methods:
            method_type = "📄 特徵方法"
        elif method in embedding_methods:
            method_type = "🧭 嵌入方法"
        else:
            method_type = "🌐 GNN 方法"
        print(f"  {method_type:15} {method:12} → 平均: {row['mean']:.6f}")
    
    # ========== 4. 最佳與最差組合 ==========
    print("\n" + "=" * 100)
    print("4️⃣  最佳和最差組合排名")
    print("=" * 100)
    
    print("\n🏆 Top 10 最佳組合:")
    top10 = df.nlargest(10, 'score')[['method', 'ratio', 'sampling', 'score']].reset_index(drop=True)
    for idx, row in top10.iterrows():
        print(f"  {idx+1:2}. {row['method']:12} | {row['ratio']:8} | {row['sampling']:12} → {row['score']:.6f}")
    
    print("\n❌ Bottom 10 最差組合:")
    bottom10 = df.nsmallest(10, 'score')[['method', 'ratio', 'sampling', 'score']].reset_index(drop=True)
    for idx, row in bottom10.iterrows():
        print(f"  {idx+1:2}. {row['method']:12} | {row['ratio']:8} | {row['sampling']:12} → {row['score']:.6f}")
    
    # ========== 5. 交叉分析 ==========
    print("\n" + "=" * 100)
    print("5️⃣  交叉分析：比例 × 採樣技術")
    print("=" * 100)
    
    cross_pivot = df.pivot_table(values='score', index='ratio', columns='sampling', aggfunc='mean')
    print("\n", cross_pivot.round(6))
    
    # ========== 6. 方法性能詳細對比 ==========
    print("\n" + "=" * 100)
    print("6️⃣  方法性能詳細對比：按比例分類")
    print("=" * 100)
    
    for ratio in ['2:1', '1:1', 'Original']:
        ratio_df = df[df['ratio'] == ratio]
        method_scores = ratio_df.groupby('method')['score'].mean().sort_values(ascending=False)
        print(f"\n{ratio} 比例下的方法排名:")
        for i, (method, score) in enumerate(method_scores.items(), 1):
            print(f"  {i}. {method:12} → {score:.6f}")
    
    # ========== 7. 統計摘要 ==========
    print("\n" + "=" * 100)
    print("📊 整體統計摘要")
    print("=" * 100)
    
    print(f"\n全體 72 個結果統計:")
    print(f"  • 平均 AUC-PRC:     {df['score'].mean():.6f}")
    print(f"  • 標準差:          {df['score'].std():.6f}")
    print(f"  • 最高分:          {df['score'].max():.6f} ({df[df['score'] == df['score'].max()].iloc[0]['filename']})")
    print(f"  • 最低分:          {df['score'].min():.6f} ({df[df['score'] == df['score'].min()].iloc[0]['filename']})")
    print(f"  • 中位數:          {df['score'].median():.6f}")
    print(f"  • 分數範圍:        {df['score'].max() - df['score'].min():.6f}")
    
    # ========== 8. APATE 假設驗證 ==========
    print("\n" + "=" * 100)
    print("🎯 APATE 假設驗證結果")
    print("=" * 100)
    
    ratio_means = df.groupby('ratio')['score'].mean().sort_values(ascending=False)
    print(f"\n類別不平衡比例效果排序 (按平均 AUC-PRC):")
    
    for i, (ratio, score) in enumerate(ratio_means.items(), 1):
        if ratio == "2:1":
            status = "✅ APATE 假設確認" if i == 1 else "❌ APATE 假設需要修正"
        else:
            status = ""
        
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"  {medal} {i}. {ratio:8} → {score:.6f} {status}")
    
    if ratio_means.index[0] == "2:1":
        improvement_vs_11 = (ratio_means['2:1'] - ratio_means['1:1']) / ratio_means['1:1'] * 100
        improvement_vs_orig = (ratio_means['2:1'] - ratio_means['Original']) / ratio_means['Original'] * 100
        print(f"\n✅ 結論：APATE 假設在本數據集上得到驗證！")
        print(f"   • 2:1 vs 1:1: {improvement_vs_11:+.1f}%")
        print(f"   • 2:1 vs Original: {improvement_vs_orig:+.1f}%")
    else:
        print(f"\n⚠️  結論：{ratio_means.index[0]} 表現最佳，與 APATE 假設不同")
    
    # ========== 9. 採樣技術效果分析 ==========
    print("\n" + "=" * 100)
    print("🔍 採樣技術效果細節分析")
    print("=" * 100)
    
    print("\n各採樣技術的效果:")
    sampling_means = df.groupby('sampling')['score'].mean().sort_values(ascending=False)
    
    for i, (sampling, score) in enumerate(sampling_means.items(), 1):
        change_vs_none = (score - sampling_means.get('None', score)) / sampling_means.get('None', score) * 100 if sampling != 'None' else 0
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"  {medal} {sampling:12} → {score:.6f} {f'({change_vs_none:+.1f}% vs None)' if sampling != 'None' else ''}")
    
    # ========== 10. 方法類別對比 ==========
    print("\n" + "=" * 100)
    print("📈 方法類別對比")
    print("=" * 100)
    
    feature_scores = df[df['method'].isin(feature_methods)]['score'].mean()
    embedding_scores = df[df['method'].isin(embedding_methods)]['score'].mean()
    gnn_scores = df[df['method'].isin(gnn_methods)]['score'].mean()
    
    print(f"\n方法類別平均性能:")
    print(f"  📄 特徵方法 (Intrinsic, Positional):  {feature_scores:.6f}")
    print(f"  🧭 嵌入方法 (DeepWalk, Node2Vec):    {embedding_scores:.6f}")
    print(f"  🌐 GNN 方法 (GCN, SAGE, GAT, GIN):   {gnn_scores:.6f}")
    
    best_category = max(
        [('特徵方法', feature_scores), ('嵌入方法', embedding_scores), ('GNN 方法', gnn_scores)],
        key=lambda x: x[1]
    )
    print(f"\n🏆 最佳方法類別: {best_category[0]} (AUC-PRC: {best_category[1]:.6f})")
    
    print("\n" + "=" * 100)
    print("分析完成！")
    print("=" * 100)
