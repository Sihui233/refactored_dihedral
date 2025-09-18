#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Path patterns (match your tree)
# -----------------------------
BASE_DIR_RE = re.compile(
    r"qualitative_(?P<p>\d+)_(?P<mlp_class>[A-Za-z0-9_]+)_(?P<num_neurons>\d+)_features_(?P<features>\d+)_k_(?P<k>\d+)$"
)
LEAF_DIR_RE = re.compile(
    r"p=(?P<p>\d+)_bs=(?P<bs>\d+)_nn=(?P<nn>\d+)_lr_(?P<lr>[0-9.eE+-]+)_wd_(?P<wd>[0-9.eE+-]+)_epochs=(?P<epochs>\d+)_training_set_size=(?P<tss>\d+)$"
)
SEED_DIR_RE = re.compile(r"^graphs_seed_(?P<seed>\d+)$")
APPROX_FILE_RE = re.compile(r"^approx_summary_layer(?P<layer>\d+)_p(?P<p>\d+)\.json$")


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
    # mlp_class like "two_embed_2" -> split to get num_layers
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
# JSON helpers
# -----------------------------
def load_json(p: Path) -> Optional[Any]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def epoch_gate_ok(log_obj: Any, epoch: int, acc_threshold: float, acc_key_candidates=("test_accuracy", "accuracy", "acc")) -> bool:
    """
    log_obj is expected to be a dict keyed by epoch strings: {"700": {...}}.
    Return True if epoch exists and accuracy >= threshold.
    """
    if not isinstance(log_obj, dict):
        return False
    key = str(epoch)
    if key not in log_obj or not isinstance(log_obj[key], dict):
        return False
    rec = log_obj[key]
    for k in acc_key_candidates:
        if k in rec and isinstance(rec[k], (int, float)):
            return float(rec[k]) >= acc_threshold - 1e-12
    return False


def extract_sum_all(approx_obj: Any) -> Optional[float]:
    """
    Expect structure like:
    {
      "consistency": {
        "ok": true,
        "sum_all": 4,
        ...
      },
      ...
    }
    """
    if isinstance(approx_obj, dict):
        c = approx_obj.get("consistency")
        if isinstance(c, dict) and isinstance(c.get("sum_all"), (int, float)):
            return float(c["sum_all"])
    return None


# -----------------------------
# Plot heatmaps (lr x wd) per (p, num_layers)
# -----------------------------
def plot_heatmaps(df_agg: pd.DataFrame, out_dir: Path, value_col: str = "sum_all_mean") -> List[Path]:
    saved: List[Path] = []
    if df_agg.empty:
        return saved

    # ensure numeric sort
    def fmt_tick(v: float) -> str:
        return f"{v:.0e}" if (v < 1e-3 or v >= 1) else f"{v:g}"

    for (p, L), g in df_agg.groupby(["p", "num_layers"], dropna=False):
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
        ax.set_yticklabels([fmt_tick(float(lr)) for lr in pv.index])
        ax.set_xticks(np.arange(len(pv.columns)))
        ax.set_xticklabels([fmt_tick(float(wd)) for wd in pv.columns], rotation=45, ha="right")

        ax.set_title(f"Frequencies learned (sum_all mean) — p={p}, L={L if L is not None else 'NA'}")
        ax.set_ylabel("learning_rate")
        ax.set_xlabel("weight_decay")
        fig.tight_layout()

        png = out_dir / f"heatmap_sum_all_p{p}_L{L if L is not None else 'NA'}.png"
        fig.savefig(png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(png)

        csv = out_dir / f"heatmap_sum_all_p{p}_L{L if L is not None else 'NA'}.csv"
        pv.to_csv(csv)
        saved.append(csv)

    return saved


# -----------------------------
# Main sweep
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Collect sum_all from approx summaries only for seeds whose epoch-700 test_accuracy==1.0.")
    ap.add_argument("--root", type=Path, default=Path("/home/mila/w/weis/scratch/DL/MLP_dihedral/hypertuning1"))
    ap.add_argument("--outdir", type=Path, default=Path("./freq_out"))
    ap.add_argument("--epoch", type=int, default=700)
    ap.add_argument("--acc-threshold", type=float, default=1.0, help="Gate on test accuracy >= this value at the chosen epoch.")
    ap.add_argument("--strict", action="store_true", help="If set, skip seeds with missing log files instead of trying mild fallbacks.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    # qualitative_* bases
    for base in args.root.rglob("qualitative_*"):
        if not base.is_dir():
            continue
        base_info = parse_base_dir(base)
        if not base_info:
            continue

        # leaf model dirs
        for leaf in base.rglob("p=*_*"):
            if not leaf.is_dir():
                continue
            leaf_info = parse_leaf_dir(leaf)
            if not leaf_info:
                continue

            p_path = leaf_info["p"]
            features = base_info.get("features")

            # enumerate seeds from graphs_seed_* dirs
            for child in leaf.iterdir():
                if not child.is_dir():
                    continue
                m = SEED_DIR_RE.match(child.name)
                if not m:
                    continue
                seed = int(m.group("seed"))

                # 1) find per-seed log file and check epoch gate
                #    canonical name: log_features_{features}_seed_{seed}.json
                log_paths = []
                if features is not None:
                    log_paths.append(leaf / f"log_features_{features}_seed_{seed}.json")
                # mild fallback: sometimes features might be omitted
                if not args.strict:
                    log_paths.append(leaf / f"log_seed_{seed}.json")
                    # any other common variant can be added here

                log_obj = None
                used_log = None
                for lp in log_paths:
                    if lp.is_file():
                        log_obj = load_json(lp)
                        used_log = lp
                        if log_obj is not None:
                            break
                if log_obj is None or not epoch_gate_ok(log_obj, args.epoch, args.acc_threshold):
                    # gate failed; skip this seed
                    continue

                # 2) within graphs_seed_{seed}, find approx_summary_layer*_p{p}.json
                for jf in child.iterdir():
                    if not jf.is_file():
                        continue
                    mm = APPROX_FILE_RE.match(jf.name)
                    if not mm:
                        continue
                    if int(mm.group("p")) != int(p_path):
                        continue
                    layer = int(mm.group("layer"))
                    approx_obj = load_json(jf)
                    if approx_obj is None:
                        continue
                    sum_all = extract_sum_all(approx_obj)
                    if sum_all is None:
                        continue

                    row = {
                        # base info
                        "p": base_info.get("p"),
                        "num_layers": base_info.get("num_layers"),
                        "MLP_class": base_info.get("MLP_class"),
                        "features": base_info.get("features"),
                        "k": base_info.get("k"),
                        "base_num_neurons": base_info.get("num_neurons"),
                        # leaf info
                        **leaf_info,
                        # seed/layer
                        "seed": seed,
                        "layer": layer,
                        # metric
                        "sum_all": float(sum_all),
                        # bookkeeping
                        "leaf_path": str(leaf),
                        "seed_dir": str(child),
                        "log_used": str(used_log) if used_log else None,
                        "approx_file": jf.name,
                        "epoch_gate": args.epoch,
                        "acc_threshold": args.acc_threshold,
                    }
                    rows.append(row)

    if not rows:
        print("No qualifying data found: either epoch gate failed or approx summaries missing.")
        return

    df = pd.DataFrame(rows)

    # Save long form (per seed×layer)
    long_csv = args.outdir / "freq_learned_long.csv"
    df.to_csv(long_csv, index=False)
    print(f"Wrote: {long_csv}")

    # Aggregate across seeds (mean/stdev/count) per (p, num_layers, lr, wd, layer)
    grp_cols = ["p", "num_layers", "learning_rate", "weight_decay", "layer"]
    agg = (
        df.groupby(grp_cols, dropna=False)
          .agg(sum_all_mean=("sum_all", "mean"),
               sum_all_std=("sum_all", "std"),
               n_seeds=("sum_all", "count"))
          .reset_index()
    )

    agg_csv = args.outdir / "freq_learned_agg.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"Wrote: {agg_csv}")

    # Heatmaps per (p, num_layers). If there are multiple layers, you can filter by layer first.
    # By default here we average over layers too (so one heatmap per p,L). If you want per-layer heatmaps, split below.
    agg_over_layers = (
        agg.groupby(["p", "num_layers", "learning_rate", "weight_decay"], dropna=False)
           .agg(sum_all_mean=("sum_all_mean", "mean"))
           .reset_index()
    )

    saved = plot_heatmaps(agg_over_layers, args.outdir, value_col="sum_all_mean")
    for pth in saved:
        print(f"Wrote: {pth}")


if __name__ == "__main__":
    main()
