#!/usr/bin/env python3
"""quantifying_pca_and_homology.py

Traverse experiment result folders, aggregate PCA variance‐explained data and persistent‑homology
summaries, and save comprehensive histogram plots for each feature subset.

Usage
-----
Simply run the script – no command‑line arguments are required. The list of root folders is
embedded inside the file (see `MAIN_DIRECTORIES`).

"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration   

# Root directories to analyse.
MAIN_DIRECTORIES: List[str] = [
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_no_embed_1_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_no_embed_2_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_no_embed_3_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_no_embed_cheating_1_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_no_embed_cheating_2_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_no_embed_cheating_3_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_one_embed_1_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_one_embed_2_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_one_embed_3_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_one_embed_cheating_1_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_one_embed_cheating_2_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_one_embed_cheating_3_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_two_embed_1_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_two_embed_2_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_two_embed_3_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_two_embed_cheating_1_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_two_embed_cheating_2_1024_features_128_k_58/pdf_plots",
    "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-6-sawtooth-finding-9/qualitative_59_two_embed_cheating_3_1024_features_128_k_58/pdf_plots",
]

# Histogram resolution (0 → 1 in 0.025 steps → 40 buckets)
NBINS: int = 40
BINS: np.ndarray = np.linspace(0.0, 1.0, NBINS + 1)
K_TERMS: int = 4  # Number of leading PCA terms to analyse

# Sub‑directory prefixes to recognise.
PREFIX_REGEXPS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^(cluster_contributions_to_logits)_freq=\d+$"), r"\1_freq="),
    (re.compile(r"^(cluster_weights_to_logits)_freq=\d+$"), r"\1_freq="),
    (re.compile(r"^embeds$"), r"embeds"),
    (re.compile(r"^(layer\d+)_freq=\d+$"), r"\1_freq="),
    (re.compile(r"^(layer\d+)_freq=\d+_weights$"), r"\1_freq=_weights"),
    (re.compile(r"^(layer\d+)_preacts$"), r"\1_preacts"),
    (re.compile(r"^(layer\d+)_weights$"), r"\1_weights"),
]

# Homology category mapping
HOMOLOGY_CATEGORIES = {
    (1, 0, 0): "disc",
    (1, 1, 0): "circle",
    (1, 2, 1): "torus",
}

###########################################################
# Utility helpers                                         #
###########################################################

def normalise_prefix(name: str) -> str | None:
    """Return a canonical prefix identifier (with layer index retained) or None if *name* is irrelevant."""
    for regex, repl in PREFIX_REGEXPS:
        if regex.match(name):
            return regex.sub(repl, name)
    return None  # Unrecognised sub‑directory


def hist_data_to_trace(values: List[float], title: str) -> go.Histogram:
    """Convert a raw vector of values into a Plotly histogram trace."""
    return go.Histogram(
        x=values,
        xbins=dict(start=0.0, end=1.0, size=1.0 / NBINS),
        name=title,
        marker=dict(line=dict(width=0)),
        showlegend=False,
    )

###########################################################
# Core aggregation logic                                  #
###########################################################

class PCABucketStore:
    """Collect variance ratios and cumulative variance ratios and expose histogram plotting."""

    def __init__(self) -> None:
        # mapping: (prefix, metric, term_idx) → list[float]
        self._values: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)

    def add(self, prefix: str, ratios: List[float], cumul: List[float]) -> None:
        """Add one observation (lists must each be >= *K_TERMS* long)."""
        for i in range(K_TERMS):
            self._values[(prefix, "ratio", i)].append(ratios[i])
            self._values[(prefix, "cumulative", i)].append(cumul[i])

    def prefixes(self) -> List[str]:
        """Return all unique prefixes collected so far."""
        return sorted({k[0] for k in self._values.keys()})

    def build_figure(self, prefix: str, display_name: str) -> go.Figure:
        """Produce a 2 × 4 subplot histogram figure for *prefix*."""
        fig = make_subplots(rows=2, cols=K_TERMS, subplot_titles=[f"Term {i+1}" for i in range(K_TERMS)] * 2)
        # First row: variance_ratio
        for i in range(K_TERMS):
            values = self._values.get((prefix, "ratio", i), [])
            fig.add_trace(hist_data_to_trace(values, f"VarRatio‑{i+1}"), row=1, col=i + 1)
        # Second row: cumulative_variance_ratio
        for i in range(K_TERMS):
            values = self._values.get((prefix, "cumulative", i), [])
            fig.add_trace(hist_data_to_trace(values, f"CumVar‑{i+1}"), row=2, col=i + 1)

        fig.update_layout(
            height=600,
            width=K_TERMS * 250,
            title_text=f"{display_name} PCA variance distributions – {prefix}",
            bargap=0.05,
        )
        fig.update_yaxes(title_text="counts")
        fig.update_xaxes(title_text="%var explained")
        return fig


class HomologyCounter:
    """Count disc/circle/torus occurrences for each prefix."""

    def __init__(self) -> None:
        # mapping: prefix → {category: int}
        self._counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.noise_adjustments: int = 0

    def add(self, prefix: str, b0: int, b1: int, b2: int) -> None:
        key = (b0, b1, b2)
        if b2 >= 2:  # Noise – assume void specimen
            self.noise_adjustments += 1
            return  # Discard noisy sample entirely (b0 set to 0 per spec)
        category = HOMOLOGY_CATEGORIES.get(key)
        if category:
            self._counts[prefix][category] += 1

    def prefixes(self) -> List[str]:
        return sorted(self._counts.keys())

    def build_figure(self, prefix: str) -> go.Figure:
        counts = self._counts[prefix]
        categories = ["disc", "circle", "torus"]
        values = [counts.get(cat, 0) for cat in categories]
        fig = go.Figure(
            data=[
                go.Bar(x=categories, y=values)
            ]
        )
        fig.update_layout(title_text=f"Homology counts – {prefix}")
        return fig

###########################################################
# Directory traversal                                      #
###########################################################

def process_3d_dir(directory: Path, pca_store: PCABucketStore) -> None:
    """Handle a single *…/3d* directory."""
    for sub in directory.iterdir():
        if not sub.is_dir():
            continue
        canonical = normalise_prefix(sub.name)
        if canonical is None:
            continue  # Irrelevant
        json_file = sub / "variance_explained.json"
        if not json_file.is_file():
            continue
        try:
            with json_file.open("r") as fp:
                data = json.load(fp)
            ratios = data["variance_ratio"]
            cumul = data["cumulative_variance_ratio"]
            if len(ratios) < K_TERMS or len(cumul) < K_TERMS:
                print(f"[WARN] {json_file} contains < {K_TERMS} terms; skipping.")
                continue
            pca_store.add(canonical, ratios, cumul)
        except Exception as exc:  # noqa: BLE001,E722
            print(f"[ERROR] Failed to parse {json_file}: {exc}")


def process_homology_dir(directory: Path, homology_counter: HomologyCounter) -> None:
    for sub in directory.iterdir():
        if not sub.is_dir():
            continue
        canonical = normalise_prefix(sub.name)
        if canonical is None:
            continue
        # Expect exactly one JSON file inside
        json_files = list(sub.glob("*.json"))
        if not json_files:
            continue
        if len(json_files) > 1:
            print(f"[WARN] Multiple JSONs in {sub}; using first.")
        jf = json_files[0]
        try:
            with jf.open("r") as fp:
                data = json.load(fp)
            b0 = int(data.get("b0", 0))
            b1 = int(data.get("b1", 0))
            b2 = int(data.get("b2", 0))
            homology_counter.add(canonical, b0, b1, b2)
        except Exception as exc:  # noqa: BLE001,E722
            print(f"[ERROR] Failed to parse {jf}: {exc}")

###########################################################
# Plot persistence                                         #
###########################################################

def save_pdf(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(path), engine="kaleido")
    except ValueError as err:
        # Fallback to HTML
        html_path = path.with_suffix(".html")
        print(f"[WARN] PDF export failed ({err}); writing HTML: {html_path}")
        fig.write_html(html_path)


def build_layerwise_homology_fig(
    counter: HomologyCounter,
    layer_numbers: list[int],
    column_prefixes: tuple[str, str],
    title: str,
    include_cluster_row: bool = False
) -> go.Figure:
    """
    Build an n-row × 2-col Plotly figure.

    *counter* – filled HomologyCounter
    *layer_numbers* – sorted list of layer indices, e.g. [1, 2, 3]
    *column_prefixes* – (prefix_for_col1, prefix_for_col2) such as
        ("freq=", "weights") or ("preacts", "weights")
    *title* – figure title
    """
    # If we are drawing the “frequency-clusters” figure, append one final row
    rows = len(layer_numbers) + (1 if include_cluster_row else 0)
    fig = make_subplots(
        rows=rows,
        cols=2,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        subplot_titles=[
            *(f"Layer {layer} – {col}"
              for layer in layer_numbers
              for col in column_prefixes),
            *(
                ["Cluster contributions", "Cluster weights"]
                if include_cluster_row else []
            ),
        ],
    )

    for r, layer in enumerate(layer_numbers, start=1):
        for c, col_prefix in enumerate(column_prefixes, start=1):
            prefix = f"layer{layer}_{col_prefix}"
            if prefix not in counter.prefixes():
                # leave blank if that particular plot is missing
                continue
            sub = counter.build_figure(prefix)
            for trace in sub.data:
                # copy each trace into the right subplot
                fig.add_trace(trace, row=r, col=c)

        # ── new final row: cluster-level contributions vs weights ────────────────
    if include_cluster_row:
        for c, pref in enumerate(
            ("cluster_contributions_to_logits_freq=",
             "cluster_weights_to_logits_freq="), start=1):
            if pref not in counter.prefixes():
                continue
            sub = counter.build_figure(pref)
            for tr in sub.data:
                fig.add_trace(tr, row=rows, col=c)

    height_per_row = 300
    fig.update_layout(
        height=height_per_row * rows,
        width=900,
        title_text=title,
        showlegend=False,
    )
    return fig


###########################################################
# Main entry‑point                                         #
###########################################################

def main() -> None:
    total_noise_adjustments = 0

    for main_dir_str in MAIN_DIRECTORIES:
        main_dir = Path(main_dir_str)
        if not main_dir.exists():
            print(f"[WARN] Skipping missing directory: {main_dir}")
            continue

        # ——— extract the class name from e.g. ".../qualitative_59_no_embed_cheating_2_1024_features_128_k_58/pdf_plots"
        parent_name = main_dir.parent.name
        m = re.match(r"qualitative_\d+_(?P<class>.+?)_\d+_1024_features_", parent_name)
        class_name = m.group("class") if m else parent_name

        class_map = {
            "no_embed":              "One Hot Concat",
            "no_embed_cheating":     "One Hot Vec Add",
            "one_embed":             "One Embed Concat",
            "one_embed_cheating":    "One Embed Vec Add",
            "two_embed":             "Two Embed Concat",
            "two_embed_cheating":    "Two Embed Vec Add",
        }
        display_name = class_map.get(class_name, class_name)

        print(f"\n=== Processing {main_dir} (class={class_name}) ===")

        pca_stores       = {"pca": PCABucketStore(), "diffusion": PCABucketStore()}
        homology_counters = {"pca": HomologyCounter(), "diffusion": HomologyCounter()}

        for seed_dir in [d for d in main_dir.iterdir() if d.is_dir()]:
            for family in ("pca_pdf_plots", "diffusion_pdf_plots"):
                base_dir   = seed_dir / family
                if not base_dir.is_dir():
                    continue

                key = "pca" if family.startswith("pca") else "diffusion"

                # 3-D PCA statistics --------------------------------------------------
                three_d = base_dir / "3d"
                if three_d.is_dir():
                    process_3d_dir(three_d, pca_stores[key])

                # Homology statistics -------------------------------------------------
                homology = base_dir / "homology"
                if homology.is_dir():
                    process_homology_dir(homology, homology_counters[key])

        # ── save PCA plots (unchanged apart from folder name) ------------------------
        for key, store in pca_stores.items():
            pca_dir = main_dir / f"plots-{key}"          # plots-pca / plots-diffusion
            for prefix in store.prefixes():
                fig = store.build_figure(prefix, display_name)
                save_pdf(fig, pca_dir / f"{prefix}.pdf")

        # ── save homology plots (new: per-family counters & folders) -----------------
        for key, counter in homology_counters.items():
            hom_dir = main_dir / f"plots-homology-{key}"  # plots-homology-pca / -diffusion

            # gather layer numbers present in *this* counter
            layer_nums = {
                int(m.group(1))
                for p in counter.prefixes()
                if (m := re.match(r"layer(\d+)_", p))
            }
            if not layer_nums:
                print(f"    [WARN] No layer-specific homology prefixes for {key}; skipping.")
                continue
            layers_sorted = sorted(layer_nums)

            # (1) frequency clusters vs cluster weights ------------------------------
            fig_freq = build_layerwise_homology_fig(
                counter, layers_sorted,
                ("freq=", "freq=_weights"),
                f"{display_name}: Homology of frequency clusters vs. cluster weights",
                True
            )
            fig_freq.update_yaxes(title_text="counts")
            save_pdf(fig_freq, hom_dir / "frequency_clusters_homology_layerwise.pdf")

            # (2) pre-activations vs cluster weights ---------------------------------
            fig_preacts = build_layerwise_homology_fig(
                counter, layers_sorted,
                ("preacts", "weights"),
                f"{display_name}: Homology of pre-activations vs. cluster weights",
            )
            fig_preacts.update_yaxes(title_text="counts")
            save_pdf(fig_preacts, hom_dir / "preacts_layerwise.pdf")

            print(f"    Homology noise adjustments ({key}): {counter.noise_adjustments}")
            total_noise_adjustments += counter.noise_adjustments


if __name__ == "__main__":
    sys.exit(main())
