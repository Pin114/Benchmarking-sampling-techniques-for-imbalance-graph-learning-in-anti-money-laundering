
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

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

        # Determine dataset
        if "_ibm_" in filename:
            dataset = "IBM"
        elif "_elliptic_" in filename:
            dataset = "Elliptic"
        else:
            continue  # Skip if not matching

        # Determine whether this file contains AUC-PRC or F1
        if "_f1_params_" in filename:
            metric = "F1"
            base = filename.replace("_f1_params_ibm_", "|").replace("_f1_params_elliptic_", "|").replace(".txt", "")
        else:
            metric = "AUC-PRC"
            base = filename.replace("_params_ibm_", "|").replace("_params_elliptic_", "|").replace(".txt", "")

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

            if metric == "AUC-PRC" and "AUC-PRC:" in content:
                score = float(content.split("AUC-PRC:")[1].strip())
            elif metric == "F1" and "F1:" in content:
                score = float(content.split("F1:")[1].strip())
            else:
                continue

            results.append({
                'dataset': dataset,
                'method': method,
                'ratio': ratio,
                'sampling': sampling,
                'metric': metric,
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
    
    def metric_analysis(df, metric_name):
        sub = df[df['metric'] == metric_name]
        if len(sub) == 0:
            print(f"No {metric_name} results found.")
            return

        percentile_note = "(F1 score uses quantile thresholding: 99% for intrinsic/positional/embedding, 90% for GNN)" if metric_name == "F1" else ""
        print("\n" + "=" * 100)
        print(f"  {metric_name} Analysis {percentile_note}")
        print("=" * 100)

        # ratio analysis
        ratio_analysis = sub.groupby('ratio')['score'].agg(['count', 'mean', 'std', 'min', 'max'])
        ratio_analysis = ratio_analysis.sort_values('mean', ascending=False)
        print("\n", ratio_analysis)

        best_ratio = ratio_analysis['mean'].idxmax()
        print(f"\n Best Ratio for {metric_name}: {best_ratio} (Mean {metric_name}: {ratio_analysis.loc[best_ratio, 'mean']:.6f})")

        # sampling analysis
        print("\n" + "=" * 100)
        print(f"  Sampling Technique Analysis for {metric_name}")
        print("=" * 100)
        sampling_analysis = sub.groupby('sampling')['score'].agg(['count', 'mean', 'std', 'min', 'max'])
        sampling_analysis = sampling_analysis.sort_values('mean', ascending=False)
        print("\n", sampling_analysis)

        best_sampling = sampling_analysis['mean'].idxmax()
        worst_sampling = sampling_analysis['mean'].idxmin()
        improvement = (sampling_analysis.loc[best_sampling, 'mean'] - sampling_analysis.loc[worst_sampling, 'mean']) / sampling_analysis.loc[worst_sampling, 'mean'] * 100
        print(f"\n Best Sampling: {best_sampling} (Mean {metric_name}: {sampling_analysis.loc[best_sampling, 'mean']:.6f})")
        print(f" Compared to Worst Sampling ({worst_sampling}): {improvement:+.1f}%")

        # method analysis
        print("\n" + "=" * 100)
        print(f"  Method Analysis for {metric_name}")
        print("=" * 100)
        method_analysis = sub.groupby('method')['score'].agg(['count', 'mean', 'std', 'min', 'max'])
        method_analysis = method_analysis.sort_values('mean', ascending=False)
        print("\n", method_analysis)

        best_method = method_analysis['mean'].idxmax()
        print(f"\n Best Method: {best_method} (Mean {metric_name}: {method_analysis.loc[best_method, 'mean']:.6f})")

        # cross analysis
        print("\n" + "=" * 100)
        print(f"  Cross Analysis (Ratio × Sampling) for {metric_name}")
        print("=" * 100)
        cross_pivot = sub.pivot_table(values='score', index='ratio', columns='sampling', aggfunc='mean')
        print("\n", cross_pivot)

        print("\n" + "=" * 100)
        print(f"  Cross Analysis (Method × Ratio) for {metric_name}")
        print("=" * 100)
        method_ratio_pivot = sub.pivot_table(values='score', index='method', columns='ratio', aggfunc='mean')
        print("\n", method_ratio_pivot)

        print("\n" + "=" * 100)
        print(f"  Cross Analysis (Method × Sampling) for {metric_name}")
        print("=" * 100)
        method_sampling_pivot = sub.pivot_table(values='score', index='method', columns='sampling', aggfunc='mean')
        print("\n", method_sampling_pivot)

        print("\n" + "=" * 100)
        print(f"  Statistical Summary for {metric_name}")
        print("=" * 100)
        print(f"\n  • Mean {metric_name}: {sub['score'].mean():.6f}")
        print(f"  • Std Dev: {sub['score'].std():.6f}")
        print(f"  • Max: {sub['score'].max():.6f}")
        print(f"  • Min: {sub['score'].min():.6f}")
        print(f"  • Median: {sub['score'].median():.6f}")

        print("\n" + "=" * 100)
        print(f"  APATE Hypothesis (for {metric_name})")
        print("=" * 100)
        ratio_means = sub.groupby('ratio')['score'].mean().sort_values(ascending=False)
        for i, (ratio, score) in enumerate(ratio_means.items(), 1):
            marker = " Hypothesis Validated" if ratio == "2:1" and i == 1 else "Wrong" if ratio == "2:1" and i != 1 else ""
            print(f"  {i}. {ratio:10} → {score:.6f}{marker}")
        if ratio_means.index[0] == "2:1":
            print("\n  APATE Hypothesis Confirmed (2:1 is best).")
        else:
            print(f"\n  APATE Hypothesis not confirmed ({ratio_means.index[0]} is best).")

def main():
    # Create analysis_reports folder if it doesn't exist
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'analysis_reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Parse all results
    results = parse_results()
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("files not found or no valid results parsed.")
        sys.exit(1)
    
    # Get unique datasets
    datasets = df['dataset'].unique()
    
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        report_file = os.path.join(reports_dir, f'{dataset.lower()}_analysis.txt')
        
        print(f"Generating analysis report for {dataset}...")
        
        # Redirect output to file
        with open(report_file, 'w') as f:
            # Save original stdout
            original_stdout = sys.stdout
            sys.stdout = f
            
            print(f"Analysis Report for {dataset} Dataset")
            print("=" * 100)
            print(f"Total results: {len(dataset_df)}")
            print(f"Date generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 100)
            
            # Execute metric analysis for both metrics
            metric_analysis(dataset_df, "AUC-PRC")
            metric_analysis(dataset_df, "F1")
            
            # ========== analysis 2: by sampling technique ==========
            print("\n" + "=" * 100)
            print("  Analysis by Sampling Technique (Evaluate SMOTE/GraphSMOTE Effectiveness)")
            print("=" * 100)
            
            sampling_analysis = dataset_df.groupby('sampling')['score'].agg(['count', 'mean', 'std', 'min', 'max'])
            sampling_analysis = sampling_analysis.sort_values('mean', ascending=False)
            print("\n", sampling_analysis)
            
            best_sampling = sampling_analysis['mean'].idxmax()
            worst_sampling = sampling_analysis['mean'].idxmin()
            improvement = (sampling_analysis.loc[best_sampling, 'mean'] - sampling_analysis.loc[worst_sampling, 'mean']) / sampling_analysis.loc[worst_sampling, 'mean'] * 100
            print(f"\n Best Sampling: {best_sampling} (Mean AUC-PRC: {sampling_analysis.loc[best_sampling, 'mean']:.6f})")
            print(f" Compared to Worst Sampling ({worst_sampling}): {improvement:+.1f}%")
            
            # ========== analysis 3: by methods ==========
            print("\n" + "=" * 100)
            print("  Analysis by Methods (Comparing 8 Methods Performance)")
            print("=" * 100)
            
            method_analysis = dataset_df.groupby('method')['score'].agg(['count', 'mean', 'std', 'min', 'max'])
            method_analysis = method_analysis.sort_values('mean', ascending=False)
            print("\n", method_analysis)
            
            best_method = method_analysis['mean'].idxmax()
            print(f"\n Best Method: {best_method} (Mean AUC-PRC: {method_analysis.loc[best_method, 'mean']:.6f})")
            
            # ========== analysis 4: cross analysis (Ratio × Sampling) ==========
            print("\n" + "=" * 100)
            print("  Cross Analysis: Ratio × Sampling Technique")
            print("=" * 100)
            
            cross_analysis = dataset_df.groupby(['ratio', 'sampling'])['score'].agg(['count', 'mean'])
            cross_pivot = dataset_df.pivot_table(values='score', index='ratio', columns='sampling', aggfunc='mean')
            print("\n", cross_pivot)
            
            # ========== analysis 5: cross analysis (Ratio × Method) ==========
            print("\n" + "=" * 100)
            print("  Cross Analysis: Ratio × Methods")
            print("=" * 100)
            
            method_ratio_pivot = dataset_df.pivot_table(values='score', index='method', columns='ratio', aggfunc='mean')
            print("\n", method_ratio_pivot)
            
            # ========== analysis 6: cross analysis (Method × Sampling) ==========
            print("\n" + "=" * 100)
            print("  Cross Analysis: Methods × Sampling Technique")
            print("=" * 100)
            
            method_sampling_pivot = dataset_df.pivot_table(values='score', index='method', columns='sampling', aggfunc='mean')
            print("\n", method_sampling_pivot)
            
            # ========== analysis 7: best and worst combinations ==========
            print("\n" + "=" * 100)
            print("  Best and Worst Combinations")
            print("=" * 100)
            
            top5 = dataset_df.nlargest(5, 'score')[['method', 'ratio', 'sampling', 'score']]
            print("\n Top 5 Best Combinations:")
            for idx, row in top5.iterrows():
                print(f"   {row['method']:12} | {row['ratio']:8} | {row['sampling']:12} → {row['score']:.6f}")
            
            bottom5 = dataset_df.nsmallest(5, 'score')[['method', 'ratio', 'sampling', 'score']]
            print("\n Bottom 5 Worst Combinations:")
            for idx, row in bottom5.iterrows():
                print(f"   {row['method']:12} | {row['ratio']:8} | {row['sampling']:12} → {row['score']:.6f}")
            
            # ========== statistical summary ==========
            print("\n" + "=" * 100)
            print(" Statistical Summary")  
            print("=" * 100)
            print("\nOverall Results Statistics:")
            print(f"  • Mean AUC-PRC:    {dataset_df['score'].mean():.6f}")
            print(f"  • Standard Deviation: {dataset_df['score'].std():.6f}")
            print(f"  • Maximum Score:   {dataset_df['score'].max():.6f}")
            print(f"  • Minimum Score:   {dataset_df['score'].min():.6f}")
            print(f"  • Median:          {dataset_df['score'].median():.6f}")
            
            # ========== APATE hypothesis verification ==========
            print("\n" + "=" * 100)
            print(" APATE Hypothesis Verification (Is 2:1 Ratio Optimal for AML?)")
            print("=" * 100)
            
            ratio_means = dataset_df.groupby('ratio')['score'].mean().sort_values(ascending=False)
            print("\nSorted by Mean AUC-PRC:")
            for i, (ratio, score) in enumerate(ratio_means.items(), 1):
                marker = " Hypothesis Validated" if ratio == "2:1" and i == 1 else "Wrong" if ratio == "2:1" and i != 1 else ""
                print(f"  {i}. {ratio:10} → {score:.6f} {marker}")
            
            if ratio_means.index[0] == "2:1":
                print("\n APATE Hypothesis Confirmed: 2:1 Ratio Indeed Performs Best!")
            else:
                print(f"\n  APATE Hypothesis Needs Revision: {ratio_means.index[0]} Performs Best, Not 2:1")
            
            print("\n" + "=" * 100)
            
            # Restore original stdout
            sys.stdout = original_stdout
        
        print(f"Report saved to: {report_file}")
    
    print("\nAll dataset analysis reports generated successfully!")

if __name__ == "__main__":
    main()
