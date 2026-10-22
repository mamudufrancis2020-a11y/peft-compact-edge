#!/usr/bin/env python
"""Aggregate experiment results: means/stds, McNemar tests, chart data."""
import json, os, glob
import numpy as np
from scipy.stats import binomtest

import os
BASE = os.environ.get("PEFT_EDGE_BASE",
                      os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
RESULTS = os.path.join(BASE, "data", "results")
OUT = os.path.join(BASE, "data")

recs = [json.load(open(p)) for p in glob.glob(f"{RESULTS}/*.json")]
print(f"loaded {len(recs)} runs")

MODELS = ["BERT-Tiny", "BERT-Mini"]
DATASETS = ["SST-2", "AGNews"]
METHODS = ["FULL", "HEAD", "LORA-4", "LORA-16", "LORA-ALL8"]
SEEDS = [42, 123, 2026]

summary = {}
for mk in MODELS:
    for ds in DATASETS:
        for mt in METHODS:
            rs = [r for r in recs if r["model"] == mk and r["dataset"] == ds and r["method"] == mt]
            assert len(rs) == 3, f"{mk}/{ds}/{mt}: {len(rs)}"
            rs.sort(key=lambda r: r["seed"])
            def ms(key):
                v = np.array([r[key] for r in rs], dtype=float)
                return {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "values": v.tolist()}
            summary[f"{mk}|{ds}|{mt}"] = {
                "acc": ms("acc"), "macro_f1": ms("macro_f1"),
                "train_s": ms("train_s"), "eval_s": ms("eval_s"),
                "energy_wh": ms("energy_wh"), "co2_g": ms("co2_g"),
                "pct_trainable": ms("pct_trainable"),
                "trainable_params": rs[0]["trainable_params"],
                "total_params": rs[0]["total_params"],
                "peak_rss_mb": ms("peak_rss_mb"),
            }

# McNemar exact tests: FULL vs each method, per seed and pooled over seeds
mcnemar = {}
for mk in MODELS:
    for ds in DATASETS:
        full = {(r["seed"]): r for r in recs if r["model"] == mk and r["dataset"] == ds and r["method"] == "FULL"}
        for mt in METHODS:
            if mt == "FULL":
                continue
            other = {r["seed"]: r for r in recs if r["model"] == mk and r["dataset"] == ds and r["method"] == mt}
            per_seed, b_all, c_all = [], 0, 0
            for sd in SEEDS:
                yf, pf = full[sd]["labels"], full[sd]["preds"]
                yo, po = other[sd]["labels"], other[sd]["preds"]
                b = sum(1 for f_, o_ in zip(zip(yf, pf), zip(yo, po))
                        if f_[0] == f_[1] and o_[0] != o_[1])
                c = sum(1 for f_, o_ in zip(zip(yf, pf), zip(yo, po))
                        if f_[0] != f_[1] and o_[0] == o_[1])
                b_all += b; c_all += c
                if b + c > 0:
                    p = binomtest(min(b, c), b + c, 0.5).pvalue  # two-sided by default
                else:
                    p = 1.0
                per_seed.append({"seed": sd, "b": b, "c": c, "p": p})
            if b_all + c_all > 0:
                p_all = binomtest(min(b_all, c_all), b_all + c_all, 0.5).pvalue
            else:
                p_all = 1.0
            mcnemar[f"{mk}|{ds}|FULL_vs_{mt}"] = {"per_seed": per_seed,
                                                  "pooled_b": b_all, "pooled_c": c_all, "pooled_p": p_all}

# efficiency ratios vs FULL
ratios = {}
for mk in MODELS:
    for ds in DATASETS:
        f = summary[f"{mk}|{ds}|FULL"]
        for mt in METHODS:
            m = summary[f"{mk}|{ds}|{mt}"]
            ratios[f"{mk}|{ds}|{mt}"] = {
                "param_ratio": m["trainable_params"] / f["trainable_params"],
                "energy_ratio": m["energy_wh"]["mean"] / f["energy_wh"]["mean"],
                "time_ratio": m["train_s"]["mean"] / f["train_s"]["mean"],
                "acc_diff": m["acc"]["mean"] - f["acc"]["mean"],
            }

json.dump({"summary": summary, "mcnemar": mcnemar, "ratios": ratios},
          open(f"{OUT}/summary.json", "w"), indent=1)

# console report
for mk in MODELS:
    for ds in DATASETS:
        print(f"\n=== {mk} / {ds} ===")
        for mt in METHODS:
            s = summary[f"{mk}|{ds}|{mt}"]
            print(f"  {mt:9s} acc={s['acc']['mean']*100:5.2f}±{s['acc']['std']*100:4.2f} "
                  f"f1={s['macro_f1']['mean']*100:5.2f} params={s['trainable_params']:>8d} "
                  f"({s['pct_trainable']['mean']:6.3f}%) t={s['train_s']['mean']:6.1f}s "
                  f"E={s['energy_wh']['mean']:.3f}Wh CO2={s['co2_g']['mean']:.3f}g")
        for mt in METHODS[1:]:
            k = f"{mk}|{ds}|FULL_vs_{mt}"
            m = mcnemar[k]
            print(f"    McNemar FULL vs {mt:9s}: b={m['pooled_b']:4d} c={m['pooled_c']:4d} p={m['pooled_p']:.4f}")
print("\nsummary.json written")
