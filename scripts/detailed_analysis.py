
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
    """Parse all training result files including AUC-PRC and F1"""
    res_dir = "res"
    results = []
    
    for filename in os.listdir(res_dir):
        if not filename.endswith('.txt'):
            continue

        if "_f1_" in filename and "_params_" in filename:
            # New format: method_f1_90_params_... or method_f1_99_params_...
            if "_f1_90_params_" in filename:
                metric = "F1_90"
            elif "_f1_99_params_" in filename:
                metric = "F1_99"
            else:
                continue
            base = filename.replace("_f1_90_params_ibm_", "|").replace("_f1_90_params_elliptic_", "|") \
                          .replace("_f1_99_params_ibm_", "|").replace("_f1_99_params_elliptic_", "|") \
                          .replace(".txt", "")
        elif "_f1_params_" in filename:
            # Old format: method_f1_params_...
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

            if metric == "AUC-PRC" and "AUC-PRC:" in content:
                score = float(content.split("AUC-PRC:")[1].strip())
            elif metric == "F1" and "F1:" in content:
                score = float(content.split("F1:")[1].strip())
            elif metric == "F1_90" and "F1_90:" in content:
                score = float(content.split("F1_90:")[1].strip())
            elif metric == "F1_99" and "F1_99:" in content:
                score = float(content.split("F1_99:")[1].strip())
            else:
                continue

            results.append({
                'method': method,
                'ratio': ratio,
                'sampling': sampling,
                'metric': metric,
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
    
    def metric_summary(df, metric_name):
        sub = df[df['metric'] == metric_name]
        if sub.empty:
            print(f"No {metric_name} results found.")
            return

        print("\n" + "=" * 100)
        print(f" Detailed {metric_name} Summary (F1 percentile: 99% intrinsic/embedding, 90% GNN data pipeline)")
        print("=" * 100)

        ratio_stats = sub.groupby('ratio')['score'].agg(['count', 'mean', 'std', 'min', 'max']).round(6).sort_values('mean', ascending=False)
        print("\n Ratio stats:\n", ratio_stats)

        sampling_stats = sub.groupby('sampling')['score'].agg(['count', 'mean', 'std', 'min', 'max']).round(6).sort_values('mean', ascending=False)
        print("\n Sampling stats:\n", sampling_stats)

        method_stats = sub.groupby('method')['score'].agg(['count', 'mean', 'std', 'min', 'max']).round(6).sort_values('mean', ascending=False)
        print("\n Method stats:\n", method_stats)

        print(f"\n{metric_name} overall mean: {sub['score'].mean():.6f}, std: {sub['score'].std():.6f}, max: {sub['score'].max():.6f}, min: {sub['score'].min():.6f}")

        ratio_means = sub.groupby('ratio')['score'].mean().sort_values(ascending=False)
        print("\n APATE check (ratio sorted):")
        for i, (ratio, score) in enumerate(ratio_means.items(), 1):
            marker = " (APATE best)" if ratio == '2:1' and i == 1 else ''
            print(f"  {i}. {ratio}: {score:.6f}{marker}")

    metric_summary(df, 'AUC-PRC')
    metric_summary(df, 'F1_90')
    metric_summary(df, 'F1_99')
