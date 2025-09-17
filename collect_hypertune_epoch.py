#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Path parsers
# -----------------------------
BASE_DIR_RE = re.compile(
    r"qualitative_(?P<p>\d+)_(?P<mlp_class>[A-Za-z0-9_]+)_(?P<num_neurons>\d+)_features_(?P<features>\d+)_k_(?P<k>\d+)$"
)
LEAF_DIR_RE = re.compile(
    r"p=(?P<p>\d+)_bs=(?P<bs>\d+)_nn=(?P<nn>\d+)_lr_(?P<lr>[0-9.eE+-]+)_wd_(?P<wd>[0-9.eE+-]+)_epochs=(?P<epochs>\d+)_training_set_size=(?P<tss>\d+)$"
)

def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float(eval(x))

def parse_base_dir(d: Path) -> Optional[Dict[str, Any]]:
    m = BASE_DIR_RE.search(d.name)
    if not m:
        return None
    out = m.groupdict()
    out = {k: (int(v) if v.isdigit() else v) for k, v in out.items()}
    mlp = out["mlp_class"]
    if "_" in mlp and mlp.split("_")[-1].isdigit():
        out["num_layers"] = int(mlp.split("_")[-1])
        out["MLP_class"] = "_".join(mlp.split("_")[:-1])
    else:
        out["num_layers"] = None
        out["MLP_class"] = mlp
    return out

def parse_leaf_dir(d: Path) -> Optional[Dict[str, Any]]:
    m = LEAF_DIR_RE.search(d.name)
    if not m:
        return None
    g = m.groupdict()
    return {
        "p": int(g["p"]),
        "batch_size": int(g["bs"]),
        "num_neurons": int(g["nn"]),
        "learning_rate": safe_float(g["lr"]),
        "weight_decay": safe_float(g["wd"]),
        "epochs": int(g["epochs"]),
        "training_set_size": int(g["tss"]),
    }

# -----------------------------
# JSON readers
# -----------------------------
def find_jsons(dirpath: Path) -> List[Path]:
    return [p for p in dirpath.glob("**/*.json") if p.is_file()]

def load_json(p: Path) -> Optional[Any]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

# -----------------------------
# Metric extraction helpers
# -----------------------------
ACC_KEYS = ["test_accuracy", "accuracy", "acc", "final_accuracy"]
LOSS_KEYS = ["test_loss", "loss", "final_loss"]

def get_val(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return None

def flatten_single_metrics(md: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract accuracy/loss/l2 from a single metrics dict.
    Also looks inside common wrappers like {'metrics': {...}} or {'test': {...}}.
    """
    # direct
    acc = get_val(md, ACC_KEYS)
    loss = get_val(md, LOSS_KEYS)
    l2 = md.get("l2_loss")
    # nested common fields
    for subkey in ("metrics", "test", "val", "eval"):
        if (acc is None or loss is None or l2 is None) and isinstance(md.get(subkey), dict):
            sub = md[subkey]
            acc = acc if acc is not None else get_val(sub, ACC_KEYS)
            loss = loss if loss is not None else get_val(sub, LOSS_KEYS)
            if l2 is None and isinstance(sub.get("l2_loss"), (int, float)):
                l2 = float(sub["l2_loss"])
    return {"accuracy": acc, "loss": loss, "l2_loss": (float(l2) if isinstance(l2, (int, float)) else None)}

def mean_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def m(key):
        vals = [r[key] for r in rows if r.get(key) is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
        return float(np.mean(vals)) if vals else None
    return {"final_accuracy": m("accuracy"), "final_loss": m("loss"), "l2_loss": m("l2_loss")}

# -----------------------------
# Epoch log extraction
# -----------------------------
def pick_epoch_from_obj(obj: Any, epoch: int) -> Optional[Dict[str, Any]]:
    """
    Robustly find metrics for a specific epoch across common shapes:

    A) list[dict] with per-epoch entries, each having 'epoch' or implicit index
    B) dict with keys -> seed, values -> list[dict] per-epoch
    C) dict {'epochs': list[dict]} or {'logs': list[dict]}
    D) dict {'seed': {...}} where inner has A/C
    """
    # A) list of entries
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        # prefer explicit 'epoch' match == epoch (1-based). Fallback to index epoch-1.
        by_epoch = [x for x in obj if isinstance(x.get("epoch"), (int, float)) and int(x["epoch"]) == int(epoch)]
        if by_epoch:
            return flatten_single_metrics(by_epoch[0])
        if len(obj) >= epoch:
            return flatten_single_metrics(obj[epoch - 1])
        return None

    # B) dict of seeds -> epoch lists
    if isinstance(obj, dict):
        # common wrappers
        for key in ("epochs", "logs", "history"):
            if isinstance(obj.get(key), list):
                return pick_epoch_from_obj(obj[key], epoch)

        # dict of seeds?
        values = list(obj.values())
        if values and all(isinstance(v, (list, dict)) for v in values):
            rows = []
            for v in values:
                r = pick_epoch_from_obj(v, epoch)
                if r:
                    rows.append(r)
            if rows:
                return mean_rows(rows)
        # fallback: maybe it's already a single metrics dict
        if any(k in obj for k in ACC_KEYS + LOSS_KEYS):
            return flatten_single_metrics(obj)

    return None

# -----------------------------
# Final log extraction (no epoch)
# -----------------------------
def pick_final_from_obj(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Accept a few shapes:
      1) {seed: {accuracy/loss/...}, ...} -> mean
      2) {"final"/"results"/"summary": {...}}
      3) single flat metrics dict
    """
    if isinstance(obj, dict):
        # 2)
        for key in ("final", "results", "summary"):
            if isinstance(obj.get(key), dict):
                return pick_final_from_obj(obj[key])
        # 1) seed dict?
        vals = list(obj.values())
        if vals and all(isinstance(v, dict) for v in vals):
            rows = [flatten_single_metrics(v) for v in vals]
            return mean_rows(rows)
        # 3) direct
        if any(k in obj for k in ACC_KEYS + LOSS_KEYS) or any(isinstance(obj.get(k), dict) for k in ("metrics","test","val","eval")):
            r = flatten_single_metrics(obj)
            return {"final_accuracy": r["accuracy"], "final_loss": r["loss"], "l2_loss": r["l2_loss"]}
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        # maybe last item is final
        r = flatten_single_metrics(obj[-1])
        return {"final_accuracy": r["accuracy"], "final_loss": r["loss"], "l2_loss": r["l2_loss"]}
    return None

# -----------------------------
# Plotting
# -----------------------------
def plot_heatmaps(df: pd.DataFrame, out_dir: Path, value_col: str) -> List[Path]:
    saved = []
    for (p, L), g in df.groupby(["p", "num_layers"], dropna=False):
        if g.empty: 
            continue
        pv = g.pivot_table(index="learning_rate", columns="weight_decay", values=value_col, aggfunc="mean")
        pv = pv.sort_index().sort_index(axis=1)
        Z = pv.to_numpy()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(Z, aspect="auto")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(value_col, rotation=270, labelpad=15)
        ax.set_yticks(np.arange(len(pv.index)))
        ax.set_yticklabels([f"{lr:.0e}" if (lr < 1e-3 or lr >= 1) else f"{lr:g}" for lr in pv.index])
        ax.set_xticks(np.arange(len(pv.columns)))
        ax.set_xticklabels([f"{wd:.0e}" if (wd < 1e-3 or wd >= 1) else f"{wd:g}" for wd in pv.columns], rotation=45, ha="right")
        ax.set_title(f"Heatmap — p={p}, L={L if L is not None else 'NA'} ({value_col})")
        ax.set_ylabel("learning_rate"); ax.set_xlabel("weight_decay")
        fig.tight_layout()
        png = out_dir / f"heatmap_{value_col}_p{p}_L{L if L is not None else 'NA'}.png"
        fig.savefig(png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(png)
        (out_dir / f"heatmap_{value_col}_p{p}_L{L if L is not None else 'NA'}.csv").write_text(pv.to_csv())
    return saved

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Collect hypertuning results; optionally take metrics from a specific epoch.")
    ap.add_argument("--root", type=Path, default=Path("/home/mila/w/weis/scratch/DL/MLP_dihedral/hypertuning1"))
    ap.add_argument("--outdir", type=Path, default=Path("./hypertune_out"))
    ap.add_argument("--metric", type=str, default="final_accuracy", choices=["final_accuracy","final_loss","l2_loss"],
                    help="Which column to plot.")
    ap.add_argument("--epoch", type=int, default=700, help="Pick metrics from this epoch if epoch logs are available.")
    ap.add_argument("--strict-epoch", action="store_true",
                    help="Only keep runs where the requested epoch exists; otherwise skip (no fallback).")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for base in args.root.rglob("qualitative_*"):
        if not base.is_dir(): 
            continue
        base_info = parse_base_dir(base)
        if not base_info:
            continue

        # search leaf model dirs
        for leaf in base.rglob("p=*_*"):
            if not leaf.is_dir():
                continue
            leaf_info = parse_leaf_dir(leaf)
            if not leaf_info:
                continue

            metrics = None
            # 1) try epoch-first
            if args.epoch is not None:
                for jf in find_jsons(leaf):
                    obj = load_json(jf)
                    if obj is None: 
                        continue
                    m = pick_epoch_from_obj(obj, args.epoch)
                    if m and any(v is not None for v in m.values()):
                        metrics = {"final_accuracy": m["final_accuracy"] if "final_accuracy" in m else m.get("accuracy"),
                                   "final_loss": m.get("final_loss") if "final_loss" in m else m.get("loss"),
                                   "l2_loss": m.get("l2_loss")}
                        break

            # 2) fallback to "final" logs unless strict
            if metrics is None and not args.strict_epoch:
                for jf in find_jsons(leaf):
                    obj = load_json(jf)
                    if obj is None: 
                        continue
                    m = pick_final_from_obj(obj)
                    if m and any(v is not None for v in m.values()):
                        metrics = m
                        break

            if metrics is None:
                continue

            row = {
                "p": base_info.get("p"),
                "num_layers": base_info.get("num_layers"),
                "MLP_class": base_info.get("MLP_class"),
                "features": base_info.get("features"),
                "k": base_info.get("k"),
                "base_num_neurons": base_info.get("num_neurons"),
                **leaf_info,
                **metrics,
                "metric_source": f"epoch_{args.epoch}" if args.epoch is not None else "final",
                "run_path": str(leaf),
            }
            rows.append(row)

    if not rows:
        print("No results found (did not locate epoch logs at requested epoch, and no fallback).")
        return

    df = pd.DataFrame(rows)

    # Save
    long_csv = args.outdir / "hypertune_results_long.csv"
    df.to_csv(long_csv, index=False)
    print(f"Wrote: {long_csv}")

    keep_cols = [
        "p","num_layers","features","k",
        "learning_rate","weight_decay",
        "batch_size","num_neurons","epochs","training_set_size",
        "final_accuracy","final_loss","l2_loss",
        "metric_source","run_path",
    ]
    compact = df[[c for c in keep_cols if c in df.columns]].copy()
    compact_csv = args.outdir / "hypertune_results_compact.csv"
    compact.to_csv(compact_csv, index=False)
    print(f"Wrote: {compact_csv}")

    # Plot
    saved = plot_heatmaps(df, args.outdir, value_col=args.metric)
    for p in saved:
        print(f"Wrote: {p}")

if __name__ == "__main__":
    main()
