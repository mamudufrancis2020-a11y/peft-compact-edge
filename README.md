# PEFT under Edge Constraints — Compact Transformer Fine-Tuning Study

Code and data for the paper **"Parameter-Efficient Fine-Tuning of Compact
Transformer Language Models under Edge Constraints: A Controlled Empirical
Study of Accuracy, Energy, and Carbon Trade-Offs"** by Mamudu Francis Itanyi,
Francisca N. Oladipo, and Taiwo Timothy Oluwagbenga.

## What this is

A fully controlled, CPU-only factorial experiment comparing **full fine-tuning**
against **parameter-efficient fine-tuning (PEFT)** on *compact* (< 15M
parameter) pre-trained transformers under edge-like constraints:

| Factor | Levels |
|---|---|
| Backbones | BERT-Tiny (2L, H=128, 4.39M), BERT-Mini (4L, H=256, 11.17M) — `prajjwal1` checkpoints |
| Tasks | SST-2 (binary sentiment, 2000-example stratified train subset), AG News (4-class topic, 2400-example subset) |
| Protocols | FULL (full fine-tuning), HEAD (frozen backbone + linear head), LoRA r=4, LoRA r=16 (attention-only), LoRA-wide r=8 (all linear projections) |
| Seeds | 42, 123, 2026 |

**60 complete training runs** in total. Every run records accuracy, macro-F1,
trainable parameters, wall-clock training time, evaluation latency, average CPU
utilisation, peak process RSS, estimated training energy (Wh) and
carbon-equivalent (g CO2e).

Key finding: on the smallest backbone, **every PEFT configuration significantly
outperforms full fine-tuning** (up to +11.5 accuracy points with 118x-17,000x
fewer trainable parameters), inverting the large-model consensus; rank-16 LoRA
consumes 25-37% less energy than full fine-tuning. Exact McNemar tests pooled
over seeds support the differences.

## Repository layout

```
peft-compact-edge/
├── code/
│   ├── experiment.py    # full training pipeline (idempotent: skips completed runs)
│   └── aggregate.py     # means/stds + exact McNemar tests -> summary.json
├── data/
│   ├── results/         # 60 per-run JSON records (the released raw data)
│   └── summary.json     # aggregated statistics + McNemar test table
├── requirements.txt
├── CITATION.cff
└── LICENSE (MIT)
```

## Requirements

Tested with: Python 3.10+, torch 2.14.1 (CPU), transformers 4.57.6,
datasets 3.6.0, peft 0.21.2, scikit-learn 1.9.1, scipy 1.18.1,
numpy 2.5.4, psutil 7.2.2.

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A 2-core CPU / 4 GB RAM machine is sufficient (no GPU is used anywhere).

## Reproducing the study

```bash
# 1. Run all 60 training runs (several hours on a 2-core CPU;
#    completed runs are skipped automatically, so the script can be
#    interrupted and resumed at will)
python code/experiment.py

# 2. Aggregate results and recompute the McNemar tests
python code/aggregate.py
```

Both scripts read/write only inside the repository tree. To keep data
elsewhere, set `PEFT_EDGE_BASE=/path/to/dir`.

Datasets are downloaded automatically on first use via the Hugging Face
`datasets` library (`sst2` from GLUE; `ag_news`). The stratified subset
indices are deterministic given the seed, so the released per-run records
correspond exactly to a fresh re-run.

## Per-run record fields

| Field | Meaning |
|---|---|
| `model`, `dataset`, `method`, `seed` | experimental condition |
| `acc`, `macro_f1` | predictive performance on the fixed evaluation split |
| `trainable_params`, `total_params` | parameter counts |
| `train_s`, `eval_s` | wall-clock training / evaluation seconds |
| `avg_cpu_util` | mean sampled CPU utilisation in [0, 1] (10 Hz sampler) |
| `peak_rss_mb` | peak resident set size of the process (MB) |
| `energy_wh`, `co2_g` | estimated energy and carbon-equivalent (see model below) |

## Energy and carbon accounting model

Edge hardware rarely exposes power counters to user space, so a transparent
utilisation-based model is used (documented and critically discussed in the
manuscript, Equations 4-7):

```
P_avg = P_idle + u * (TDP - P_idle)      TDP = 15 W, P_idle = 3 W
E     = P_avg * t / 3600                 (Wh)
CO2e  = E * 480 / 1000                   (g, global-average grid intensity)
```

Absolute values are calibrated estimates for a 15 W-class processor; relative
comparisons between protocols running on identical hardware are robust to a
common multiplicative error in the power model.

## License

Released under the MIT License (see [LICENSE](LICENSE)).

## How to cite

See [CITATION.cff](CITATION.cff), or cite the manuscript directly:

> Itanyi, M.F.; Oladipo, F.N.; Oluwagbenga, T.T. Parameter-Efficient
> Fine-Tuning of Compact Transformer Language Models under Edge Constraints:
> A Controlled Empirical Study of Accuracy, Energy, and Carbon Trade-Offs.

Correspondence: m.f.itanyi@fuep.edu.ng (ORCID: 0009-0009-3380-4471)
