
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
    """Parse all 72 training result files"""
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
    print(" AML Imbalanced Graph Learning Sampling Techniques Analysis - Detailed Report")
    print("=" * 100)
    print(f"\n Successfully loaded {len(df)}/72 results\n")
    
    # ========== 1. ratio analysis ==========
    print("\n" + "=" * 100)
    print(" Class Imbalance Ratio Analysis (Verify APATE Hypothesis)")
    print("=" * 100)
    
    ratio_stats = df.groupby('ratio')['score'].agg(['count', 'mean', 'std', 'min', 'max']).round(6)
    ratio_stats = ratio_stats.sort_values('mean', ascending=False)
    print("\n", ratio_stats)
    
    # Calculate improvement
    best_ratio_score = ratio_stats['mean'].iloc[0]
    for idx, (ratio, row) in enumerate(ratio_stats.iterrows(), 1):
        improvement = (row['mean'] - best_ratio_score) / best_ratio_score * 100 if idx > 1 else 0
        print(f"\n {idx}. {ratio:8} → Mean: {row['mean']:.6f}, Std Dev: {row['std']:.6f} {f'({improvement:+.1f}%)' if improvement != 0 else ''}")
    
    # ========== 2. sampling technique analysis ==========
    print("\n" + "=" * 100)
    print(" Sampling Technique Effectiveness Analysis")
    print("=" * 100)
    
    sampling_stats = df.groupby('sampling')['score'].agg(['count', 'mean', 'std', 'min', 'max']).round(6)
    sampling_stats = sampling_stats.sort_values('mean', ascending=False)
    print("\n", sampling_stats)
    
    best_sampling_score = sampling_stats['mean'].iloc[0]
    worst_sampling = sampling_stats['mean'].idxmin()
    improvement = (sampling_stats['mean'].iloc[0] - sampling_stats['mean'].iloc[-1]) / sampling_stats['mean'].iloc[-1] * 100
    
    print(f"\n Best Sampling Technique: {sampling_stats.index[0]} (AUC-PRC: {sampling_stats['mean'].iloc[0]:.6f})")
    print(f" Compared to Worst ({worst_sampling}): {improvement:+.1f}%")
    
    # ========== 3. method analysis ==========
    print("\n" + "=" * 100)
    print("  Method Performance Comparison")
    print("=" * 100)
    
    method_stats = df.groupby('method')['score'].agg(['count', 'mean', 'std', 'min', 'max']).round(6)
    method_stats = method_stats.sort_values('mean', ascending=False)
    print("\n", method_stats)
    
    print("\nMethod Classification:")
    feature_methods = ['intrinsic', 'positional']
    embedding_methods = ['deepwalk', 'node2vec']
    gnn_methods = ['gcn', 'sage', 'gat', 'gin']
    
    for method, row in method_stats.iterrows():
        if method in feature_methods:
            method_type = " Feature Methods"
        elif method in embedding_methods:
            method_type = " Embedding Methods"
        else:
            method_type = " GNN Methods"
        print(f"  {method_type:20} {method:12} → Mean: {row['mean']:.6f}")
    
    # ========== 4. best and worst combinations ==========
    print("\n" + "=" * 100)
    print(" Best and Worst Combinations Ranking")
    print("=" * 100)
    
    print("\n Top 10 Best Combinations:")
    top10 = df.nlargest(10, 'score')[['method', 'ratio', 'sampling', 'score']].reset_index(drop=True)
    for idx, row in top10.iterrows():
        print(f"  {idx+1:2}. {row['method']:12} | {row['ratio']:8} | {row['sampling']:12} → {row['score']:.6f}")
    
    print("\n Bottom 10 Worst Combinations:")
    bottom10 = df.nsmallest(10, 'score')[['method', 'ratio', 'sampling', 'score']].reset_index(drop=True)
    for idx, row in bottom10.iterrows():
        print(f"  {idx+1:2}. {row['method']:12} | {row['ratio']:8} | {row['sampling']:12} → {row['score']:.6f}")
    
    # ========== 5. cross analysis ==========
    print("\n" + "=" * 100)
    print("  Cross Analysis: Ratio × Sampling Technique")
    print("=" * 100)
    
    cross_pivot = df.pivot_table(values='score', index='ratio', columns='sampling', aggfunc='mean')
    print("\n", cross_pivot.round(6))
    
    # ========== 6. method detailed comparison ==========
    print("\n" + "=" * 100)
    print("  Method Performance Detailed Comparison: Classified by Ratio")
    print("=" * 100)
    
    for ratio in ['2:1', '1:1', 'Original']:
        ratio_df = df[df['ratio'] == ratio]
        method_scores = ratio_df.groupby('method')['score'].mean().sort_values(ascending=False)
        print(f"\nMethod Ranking under {ratio} Ratio:")
        for i, (method, score) in enumerate(method_scores.items(), 1):
            print(f"  {i}. {method:12} → {score:.6f}")
    
    # ========== 7. statistical summary ==========
    print("\n" + "=" * 100)
    print(" Overall Statistical Summary")
    print("=" * 100)
    
    print(f"\nStatistics for All 72 Results:")
    print(f"  • Mean AUC-PRC:         {df['score'].mean():.6f}")
    print(f"  • Standard Deviation:  {df['score'].std():.6f}")
    print(f"  • Highest Score:       {df['score'].max():.6f} ({df[df['score'] == df['score'].max()].iloc[0]['filename']})")
    print(f"  • Lowest Score:        {df['score'].min():.6f} ({df[df['score'] == df['score'].min()].iloc[0]['filename']})")
    print(f"  • Median:              {df['score'].median():.6f}")
    print(f"  • Score Range:         {df['score'].max() - df['score'].min():.6f}")
    
    # ========== 8. APATE hypothesis verification ==========
    print("\n" + "=" * 100)
    print(" APATE Hypothesis Verification Results")
    print("=" * 100)
    
    ratio_means = df.groupby('ratio')['score'].mean().sort_values(ascending=False)
    print(f"\nClass Imbalance Ratio Effect Ordering (by Mean AUC-PRC):")
    
    for i, (ratio, score) in enumerate(ratio_means.items(), 1):
        if ratio == "2:1":
            status = " APATE Hypothesis Confirmed" if i == 1 else " APATE Hypothesis Needs Revision"
        else:
            status = ""

        print(f" {i}. {ratio:8} → {score:.6f} {status}")
    
    if ratio_means.index[0] == "2:1":
        improvement_vs_11 = (ratio_means['2:1'] - ratio_means['1:1']) / ratio_means['1:1'] * 100
        improvement_vs_orig = (ratio_means['2:1'] - ratio_means['Original']) / ratio_means['Original'] * 100
        print(f"\n Conclusion: APATE Hypothesis Verified on This Dataset!")
        print(f"   • 2:1 vs 1:1: {improvement_vs_11:+.1f}%")
        print(f"   • 2:1 vs Original: {improvement_vs_orig:+.1f}%")
    else:
        print(f"\n Conclusion: {ratio_means.index[0]} Performs Best, Different from APATE Hypothesis")
    
    # ========== 9. sampling technique effect details ==========
    print("\n" + "=" * 100)
    print(" Sampling Technique Effect Detailed Analysis")
    print("=" * 100)
    
    print("\nEffectiveness of Each Sampling Technique:")
    sampling_means = df.groupby('sampling')['score'].mean().sort_values(ascending=False)
    
    for i, (sampling, score) in enumerate(sampling_means.items(), 1):
        change_vs_none = (score - sampling_means.get('None', score)) / sampling_means.get('None', score) * 100 if sampling != 'None' else 0
        print(f"  {sampling:12} → {score:.6f} {f'({change_vs_none:+.1f}% vs None)' if sampling != 'None' else ''}")
    
    # ========== 10. method category comparison ==========
    print("\n" + "=" * 100)
    print(" Method Category Comparison")
    print("=" * 100)
    
    feature_scores = df[df['method'].isin(feature_methods)]['score'].mean()
    embedding_scores = df[df['method'].isin(embedding_methods)]['score'].mean()
    gnn_scores = df[df['method'].isin(gnn_methods)]['score'].mean()
    
    print(f"\nAverage Performance by Method Category:")
    print(f"   Feature Methods (Intrinsic, Positional):  {feature_scores:.6f}")
    print(f"   Embedding Methods (DeepWalk, Node2Vec):    {embedding_scores:.6f}")
    print(f"   GNN Methods (GCN, SAGE, GAT, GIN):   {gnn_scores:.6f}")
    
    best_category = max(
        [('Feature Methods', feature_scores), ('Embedding Methods', embedding_scores), ('GNN Methods', gnn_scores)],
        key=lambda x: x[1]
    )
    print(f"\n Best Method Category: {best_category[0]} (AUC-PRC: {best_category[1]:.6f})")
    
    print("\n" + "=" * 100)
    print("Analysis Complete!")
    print("=" * 100)
