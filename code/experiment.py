#!/usr/bin/env python
"""
Controlled empirical study: parameter-efficient fine-tuning (PEFT) vs full
fine-tuning of compact transformer models under CPU-only edge constraints.

Design:
  Models  : prajjwal1/bert-tiny (L=2,H=128), prajjwal1/bert-mini (L=4,H=256)
  Tasks   : SST-2 (binary sentiment), AG News (4-class topic classification)
  Methods : FULL (full fine-tuning), HEAD (frozen backbone + linear head),
            LORA-4, LORA-16 (attention-only LoRA, r=4/16), LORA-ALL8
            (wider LoRA r=8 over all linear projections)
  Seeds   : 42, 123, 2026
  Metrics : accuracy, macro-F1, trainable params, wall-clock training time,
            eval latency, avg CPU utilization, estimated energy (Wh) and
            CO2e (g), peak process RSS.
  Energy model (documented in manuscript):
            P_avg = P_idle + u * (TDP - P_idle),  TDP = 15 W, P_idle = 3 W
            E = P_avg * t ;  CO2e = E * 480 g/kWh (global average grid)
Idempotent: completed runs (result JSON exists) are skipped.
"""
import os, json, time, random, threading, gc, sys
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# Repo root (override with the PEFT_EDGE_BASE environment variable if you
# keep data outside the repository tree).
BASE = os.environ.get("PEFT_EDGE_BASE",
                      os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
RESULTS = os.path.join(BASE, "data", "results")
os.makedirs(RESULTS, exist_ok=True)

SEEDS = [42, 123, 2026]
MODELS = {"BERT-Tiny": "prajjwal1/bert-tiny", "BERT-Mini": "prajjwal1/bert-mini"}
METHODS = ["FULL", "HEAD", "LORA-4", "LORA-16", "LORA-ALL8"]
MAX_LEN = 64
BATCH = 32
EPOCHS = 3
TRAIN_SUBSET = {"SST-2": 2000, "AGNews": 2400}
EVAL_SUBSET = {"SST-2": None, "AGNews": 1500}   # None = full dev (872)
LR = {"FULL": 2e-5, "HEAD": 1e-3, "LORA-4": 5e-4, "LORA-16": 5e-4, "LORA-ALL8": 5e-4}
TDP_W, P_IDLE_W, GRID_GCO2_KWH = 15.0, 3.0, 480.0

torch.set_num_threads(2)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def stratified_subset(ds, n, label_col, seed=0):
    if n is None or n >= len(ds):
        return list(range(len(ds)))
    rng = random.Random(seed)
    by_label = {}
    for i, lab in enumerate(ds[label_col]):
        by_label.setdefault(lab, []).append(i)
    per = n // len(by_label)
    picked = []
    for lab, idxs in by_label.items():
        rng.shuffle(idxs)
        picked.extend(idxs[:per])
    rng.shuffle(picked)
    return picked


def build_paired(model_name, dataset):
    """Return (train_texts, train_labels, eval_texts, eval_labels, n_classes)."""
    if dataset == "SST-2":
        d = load_dataset("nyu-mll/glue", "sst2")
        train, eval_ds = d["train"], d["validation"]
        tcol = "sentence"
        n_cls = 2
    else:
        d = load_dataset("fancyzhx/ag_news")
        train, eval_ds = d["train"], d["test"]
        tcol = "text"
        n_cls = 4
    tr_idx = stratified_subset(train, TRAIN_SUBSET[dataset], "label", seed=0)
    ev_idx = stratified_subset(eval_ds, EVAL_SUBSET[dataset], "label", seed=1)
    train_texts = [train[i][tcol] for i in tr_idx]
    train_labels = [train[i]["label"] for i in tr_idx]
    eval_texts = [eval_ds[i][tcol] for i in ev_idx]
    eval_labels = [eval_ds[i]["label"] for i in ev_idx]
    return train_texts, train_labels, eval_texts, eval_labels, n_cls


class PowerMeter(threading.Thread):
    """Samples CPU utilization + RSS while `active` is True."""
    def __init__(self):
        super().__init__(daemon=True)
        self.active = False
        self.utils, self.rss = [], []
        self.proc = psutil.Process()

    def run(self):
        while True:
            if self.active:
                self.utils.append(psutil.cpu_percent(interval=None))
                self.rss.append(self.proc.memory_info().rss / 1e6)
            time.sleep(0.1)

    def start_window(self):
        self.utils, self.rss = [], []
        psutil.cpu_percent(interval=None)  # prime
        self.active = True

    def stop_window(self):
        self.active = False
        u = np.mean(self.utils) / 100.0 if self.utils else 0.0
        rss = max(self.rss) if self.rss else 0.0
        return u, rss


def make_model(model_key, method, n_cls):
    name = MODELS[model_key]
    m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_cls)
    total = sum(p.numel() for p in m.parameters())
    if method == "FULL":
        pass
    elif method == "HEAD":
        for n, p in m.named_parameters():
            p.requires_grad = "classifier" in n
    else:
        cfgs = {
            "LORA-4": LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                                 target_modules=["query", "value"],
                                 task_type=TaskType.SEQ_CLS),
            "LORA-16": LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                  target_modules=["query", "value"],
                                  task_type=TaskType.SEQ_CLS),
            "LORA-ALL8": LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                    target_modules=["query", "key", "value",
                                                    "intermediate.dense",
                                                    "output.dense"],
                                    task_type=TaskType.SEQ_CLS),
        }
        m = get_peft_model(m, cfgs[method])
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return m, total, trainable


def encode(texts, tok):
    enc = tok(texts, truncation=True, padding="max_length",
              max_length=MAX_LEN, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def run_one(model_key, dataset, method, seed):
    tag = f"{model_key}_{dataset}_{method}_s{seed}".replace("/", "_")
    opath = f"{RESULTS}/{tag}.json"
    if os.path.exists(opath):
        print(f"[skip] {tag}", flush=True)
        return
    seed_all(seed)
    tok = AutoTokenizer.from_pretrained(MODELS[model_key])
    tr_x, tr_y, ev_x, ev_y, n_cls = build_paired(model_key, dataset)
    model, total_p, train_p = make_model(model_key, method, n_cls)
    model.train()

    ids, am = encode(tr_x, tok)
    labels = torch.tensor(tr_y)
    ev_ids, ev_am = encode(ev_x, tok)
    ev_labels = torch.tensor(ev_y)

    dl = DataLoader(list(zip(ids, am, labels)), batch_size=BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    steps_total = len(dl) * EPOCHS
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR[method], weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps_total), steps_total)

    meter = PowerMeter()
    if not meter.is_alive():
        meter.start()
    time.sleep(0.3)
    meter.start_window()
    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        for b_ids, b_am, b_y in dl:
            opt.zero_grad(set_to_none=True)
            logits_out = model(input_ids=b_ids, attention_mask=b_am, labels=b_y)
            logits_out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step()
    train_s = time.perf_counter() - t0
    util, peak_rss = meter.stop_window()

    # evaluation (inference latency measured separately)
    model.eval()
    te0 = time.perf_counter()
    preds = []
    with torch.no_grad():
        for i in range(0, len(ev_ids), BATCH):
            logits_out = model(input_ids=ev_ids[i:i+BATCH],
                               attention_mask=ev_am[i:i+BATCH])
            preds.extend(logits_out.logits.argmax(-1).tolist())
    eval_s = time.perf_counter() - te0

    acc = accuracy_score(ev_y, preds)
    f1m = f1_score(ev_y, preds, average="macro")
    p_avg = P_IDLE_W + util * (TDP_W - P_IDLE_W)
    energy_wh = p_avg * train_s / 3600.0
    co2_g = energy_wh / 1000.0 * GRID_GCO2_KWH

    rec = {"model": model_key, "dataset": dataset, "method": method, "seed": seed,
           "trainable_params": train_p, "total_params": total_p,
           "pct_trainable": 100.0 * train_p / total_p,
           "n_train": len(tr_y), "n_eval": len(ev_y), "epochs": EPOCHS,
           "batch": BATCH, "max_len": MAX_LEN, "lr": LR[method],
           "acc": acc, "macro_f1": f1m, "train_s": train_s, "eval_s": eval_s,
           "avg_cpu_util": util, "peak_rss_mb": peak_rss,
           "energy_wh": energy_wh, "co2_g": co2_g,
           "labels": ev_y, "preds": preds,
           "tdp_w": TDP_W, "p_idle_w": P_IDLE_W, "grid_gco2_kwh": GRID_GCO2_KWH}
    with open(opath, "w") as f:
        json.dump(rec, f)
    print(f"[done] {tag}: acc={acc:.4f} f1={f1m:.4f} train={train_s:.1f}s "
          f"E={energy_wh:.3f}Wh CO2={co2_g:.3f}g params={train_p}", flush=True)
    del model
    gc.collect()


def main():
    wanted = set()
    for a in sys.argv[1:]:
        wanted.update(a.split())
    order = wanted if wanted else None
    jobs = [(mk, ds, mt, sd) for mk in MODELS for ds in ("SST-2", "AGNews")
            for mt in METHODS for sd in SEEDS]
    if order:
        jobs = [j for j in jobs if j[1] in order or j[2] in order
                or f"{j[1]}_{j[2]}" in order]
    t0 = time.time()
    for i, (mk, ds, mt, sd) in enumerate(jobs):
        run_one(mk, ds, mt, sd)
        if i % 5 == 0:
            print(f"== {i+1}/{len(jobs)} jobs, elapsed {time.time()-t0:.0f}s ==",
                  flush=True)
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
