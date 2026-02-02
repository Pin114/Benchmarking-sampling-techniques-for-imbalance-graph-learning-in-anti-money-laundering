
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

    res_dir = "res"
    results = []
    
    for filename in os.listdir(res_dir):
        if not filename.endswith('.txt'):
            continue
        
        # Parse filename: {method}_params_ibm_{ratio}{sampling}.txt
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
    print("=" * 100)
    
    # Parse results
    df = parse_results()
    
    print(f"\n successfully loaded {len(df)}/72 results\n")
    
    if len(df) == 0:
        print("files not found or no valid results parsed.")
        sys.exit(1)
    
    # ========== analysis 1: according to ratios ==========
    print("=" * 100)
    print("  Analysis by Class Imbalance Ratio (Verify APATE Hypothesis: Is 2:1 Optimal?)")
    print("=" * 100)
    
    ratio_analysis = df.groupby('ratio')['score'].agg(['count', 'mean', 'std', 'min', 'max'])
    ratio_analysis = ratio_analysis.sort_values('mean', ascending=False)
    print("\n", ratio_analysis)
    
    # Find best ratio
    best_ratio = ratio_analysis['mean'].idxmax()
    print(f"\n🏆 Best Ratio: {best_ratio} (Mean AUC-PRC: {ratio_analysis.loc[best_ratio, 'mean']:.6f})")
    
    # ========== analysis 2: by sampling technique ==========
    print("\n" + "=" * 100)
    print("  Analysis by Sampling Technique (Evaluate SMOTE/GraphSMOTE Effectiveness)")
    print("=" * 100)
    
    sampling_analysis = df.groupby('sampling')['score'].agg(['count', 'mean', 'std', 'min', 'max'])
    sampling_analysis = sampling_analysis.sort_values('mean', ascending=False)
    print("\n", sampling_analysis)
    
    best_sampling = sampling_analysis['mean'].idxmax()
    worst_sampling = sampling_analysis['mean'].idxmin()
    improvement = (sampling_analysis.loc[best_sampling, 'mean'] - sampling_analysis.loc[worst_sampling, 'mean']) / sampling_analysis.loc[worst_sampling, 'mean'] * 100
    print(f"\n🏆 Best Sampling: {best_sampling} (Mean AUC-PRC: {sampling_analysis.loc[best_sampling, 'mean']:.6f})")
    print(f" Compared to Worst Sampling ({worst_sampling}): {improvement:+.1f}%")
    
    # ========== analysis 3: by methods ==========
    print("\n" + "=" * 100)
    print("  Analysis by Methods (Comparing 8 Methods Performance)")
    print("=" * 100)
    
    method_analysis = df.groupby('method')['score'].agg(['count', 'mean', 'std', 'min', 'max'])
    method_analysis = method_analysis.sort_values('mean', ascending=False)
    print("\n", method_analysis)
    
    best_method = method_analysis['mean'].idxmax()
    print(f"\n🏆 Best Method: {best_method} (Mean AUC-PRC: {method_analysis.loc[best_method, 'mean']:.6f})")
    
    # ========== analysis 4: cross analysis (Ratio × Sampling) ==========
    print("\n" + "=" * 100)
    print("  Cross Analysis: Ratio × Sampling Technique")
    print("=" * 100)
    
    cross_analysis = df.groupby(['ratio', 'sampling'])['score'].agg(['count', 'mean'])
    cross_pivot = df.pivot_table(values='score', index='ratio', columns='sampling', aggfunc='mean')
    print("\n", cross_pivot)
    
    # ========== analysis 5: cross analysis (Ratio × Method) ==========
    print("\n" + "=" * 100)
    print("  Cross Analysis: Ratio × Methods")
    print("=" * 100)
    
    method_ratio_pivot = df.pivot_table(values='score', index='method', columns='ratio', aggfunc='mean')
    print("\n", method_ratio_pivot)
    
    # ========== analysis 6: cross analysis (Method × Sampling) ==========
    print("\n" + "=" * 100)
    print("  Cross Analysis: Methods × Sampling Technique")
    print("=" * 100)
    
    method_sampling_pivot = df.pivot_table(values='score', index='method', columns='sampling', aggfunc='mean')
    print("\n", method_sampling_pivot)
    
    # ========== analysis 7: best and worst combinations ==========
    print("\n" + "=" * 100)
    print("  Best and Worst Combinations")
    print("=" * 100)
    
    top5 = df.nlargest(5, 'score')[['method', 'ratio', 'sampling', 'score']]
    print("\n Top 5 Best Combinations:")
    for idx, row in top5.iterrows():
        print(f"   {row['method']:12} | {row['ratio']:8} | {row['sampling']:12} → {row['score']:.6f}")
    
    bottom5 = df.nsmallest(5, 'score')[['method', 'ratio', 'sampling', 'score']]
    print("\n Bottom 5 Worst Combinations:")
    for idx, row in bottom5.iterrows():
        print(f"   {row['method']:12} | {row['ratio']:8} | {row['sampling']:12} → {row['score']:.6f}")
    
    # ========== statistical summary ==========
    print("\n" + "=" * 100)
    print(" Statistical Summary")  
    print("=" * 100)
    print("\nOverall Results Statistics:")
    print(f"  • Mean AUC-PRC:    {df['score'].mean():.6f}")
    print(f"  • Standard Deviation: {df['score'].std():.6f}")
    print(f"  • Maximum Score:   {df['score'].max():.6f}")
    print(f"  • Minimum Score:   {df['score'].min():.6f}")
    print(f"  • Median:          {df['score'].median():.6f}")
    
    # ========== APATE hypothesis verification ==========
    print("\n" + "=" * 100)
    print("🎯 APATE Hypothesis Verification (Is 2:1 Ratio Optimal for AML?)")
    print("=" * 100)
    
    ratio_means = df.groupby('ratio')['score'].mean().sort_values(ascending=False)
    print("\nSorted by Mean AUC-PRC:")
    for i, (ratio, score) in enumerate(ratio_means.items(), 1):
        marker = " Hypothesis Validated" if ratio == "2:1" and i == 1 else "❌" if ratio == "2:1" and i != 1 else ""
        print(f"  {i}. {ratio:10} → {score:.6f} {marker}")
    
    if ratio_means.index[0] == "2:1":
        print("\n APATE Hypothesis Confirmed: 2:1 Ratio Indeed Performs Best!")
    else:
        print(f"\n  APATE Hypothesis Needs Revision: {ratio_means.index[0]} Performs Best, Not 2:1")
    
    print("\n" + "=" * 100)
