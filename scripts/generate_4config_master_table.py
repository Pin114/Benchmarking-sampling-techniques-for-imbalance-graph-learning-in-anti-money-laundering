#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

CONFIGS = ["hi_small", "hi_medium", "li_small", "li_medium"]


def parse_metric_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    out = {}
    for key in ["AUC-PRC", "F1_90", "F1_99", "F1"]:
        m = re.search(rf"{re.escape(key)}:\s*([0-9]*\.?[0-9]+)", text)
        if m:
            out[key] = float(m.group(1))
    return out


def parse_name(name: str):
    # Example: gcn_f1_90_params_hi_small_ratio_1to2_graph_smote.txt
    if not name.endswith(".txt"):
        return None
    stem = name[:-4]

    metric_mode = "auc"
    f1_target = None
    if "_f1_90_params_" in stem:
        method, rest = stem.split("_f1_90_params_", 1)
        metric_mode = "f1"
        f1_target = "90"
    elif "_f1_99_params_" in stem:
        method, rest = stem.split("_f1_99_params_", 1)
        metric_mode = "f1"
        f1_target = "99"
    elif "_params_" in stem:
        method, rest = stem.split("_params_", 1)
    else:
        return None

    config = None
    for c in CONFIGS:
        if rest.startswith(c + "_") or rest == c:
            config = c
            break
    if config is None:
        return None

    suffix = rest[len(config):].lstrip("_")
    ratio = "original"
    for r in ["ratio_1to10", "ratio_1to2", "ratio_1to1", "original"]:
        if suffix.startswith(r):
            ratio = r
            suffix = suffix[len(r):].lstrip("_")
            break

    sampling = "none"
    if suffix:
        sampling = suffix

    return {
        "method": method,
        "config": config,
        "ratio": ratio,
        "sampling": sampling,
        "metric_mode": metric_mode,
        "f1_target": f1_target,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate consolidated table for 4 IBM configs")
    parser.add_argument("--res-dir", default="res")
    parser.add_argument("--output", default="res/four_config_master_summary.md")
    args = parser.parse_args()

    res_dir = Path(args.res_dir)
    rows = {}

    for p in sorted(res_dir.glob("*.txt")):
        meta = parse_name(p.name)
        if meta is None:
            continue
        metrics = parse_metric_file(p)
        key = (meta["config"], meta["method"], meta["ratio"], meta["sampling"])
        row = rows.setdefault(
            key,
            {
                "config": meta["config"],
                "method": meta["method"],
                "ratio": meta["ratio"],
                "sampling": meta["sampling"],
                "AUC-PRC": "",
                "F1_90": "",
                "F1_99": "",
            },
        )
        if "AUC-PRC" in metrics:
            row["AUC-PRC"] = metrics["AUC-PRC"]
        if meta["metric_mode"] == "f1" and meta["f1_target"] == "90" and "F1_90" in metrics:
            row["F1_90"] = metrics["F1_90"]
        if meta["metric_mode"] == "f1" and meta["f1_target"] == "99" and "F1_99" in metrics:
            row["F1_99"] = metrics["F1_99"]
        if "F1_90" in metrics:
            row["F1_90"] = metrics["F1_90"]
        if "F1_99" in metrics:
            row["F1_99"] = metrics["F1_99"]

    ordered = sorted(rows.values(), key=lambda x: (CONFIGS.index(x["config"]), x["method"], x["ratio"], x["sampling"]))

    lines = []
    lines.append("| Config | Method | Ratio | Sampling | AUC-PRC | F1_90 | F1_99 |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: |")

    for r in ordered:
        def fmt(v):
            return f"{v:.6f}" if isinstance(v, float) else str(v)

        lines.append(
            "| {config} | {method} | {ratio} | {sampling} | {auc} | {f90} | {f99} |".format(
                config=r["config"],
                method=r["method"],
                ratio=r["ratio"],
                sampling=r["sampling"],
                auc=fmt(r["AUC-PRC"]),
                f90=fmt(r["F1_90"]),
                f99=fmt(r["F1_99"]),
            )
        )

    out = Path(args.output)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
