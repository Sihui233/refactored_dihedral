import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.decomposition import PCA
import itertools
import json
import math, tempfile, uuid
from typing import List
from color_rules import colour_quad_mul_f        # ①  f·(a±b) mod p
from color_rules import colour_quad_mod_g, colour_quad_mod_g_no_fb      # ②  (a±b) mod g
from color_rules import colour_quad_a_only, colour_quad_b_only,lines_a_mod_g,lines_b_mod_g, lines_c_mod_g
# from mlp_models_multilayer import DonutMLP
from persistent_homology_gpu import run_ph_for_point_cloud
from pathlib import Path
from color_rules import build_ro_scale, build_vi_scale #, colour_pair_mod_g

try:
    from PyPDF2 import PdfMerger
except ImportError as e:
    PdfMerger = None
    _pdf2_err = e

FONT_SIZE = 18           # font size（title/axis title/ticks/legend）
CBAR_TICK_SIZE = 18      # colorbar ticks
CBAR_TITLE_SIZE = 18     # colorbar title
TICK_SIZE = 16
LEGEND_POS = dict(       # legend position
    x=1.12, y=1.02, xanchor="left", yanchor="top",
    orientation="v",
    bgcolor="rgba(255,255,255,0.65)",
    bordercolor="rgba(0,0,0,0.2)", borderwidth=1,
    font=dict(size=FONT_SIZE),
)

def _jitter_if_constant(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    if arr.size and np.allclose(arr, arr[0]):
        out = arr.astype(float).copy()
        out[0] = out[0] + eps
        return out
    return arr

# ---PCA Coordinates
from types import SimpleNamespace

def _sanitize_matrix(X: np.ndarray) -> np.ndarray:
    """Finite, centered, and with zero-variance columns removed."""
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # drop constant columns (zero variance)
    col_var = X.var(axis=0)
    keep = col_var > 0
    if keep.any():
        X = X[:, keep]
    else:
        # keep at least one column of zeros so shapes don't explode
        X = np.zeros((X.shape[0], 1), dtype=float)
    # center
    X = X - X.mean(axis=0, keepdims=True)
    return X

def _safe_pca_coords(X: np.ndarray, want_components: int):
    """
    PCA with guards:
      • remove NaN/Inf, drop constant cols, center
      • cap n_components by numeric rank and n-1
      • if rank==0, return zeros and a dummy pca object
    """
    X = _sanitize_matrix(X)
    n, d = X.shape
    if n < 2 or d == 0:
        k = 1 if d == 0 else min(1, want_components)
        return np.zeros((n, k), float), SimpleNamespace(explained_variance_ratio_=np.zeros(k, float))

    # numeric rank via SVD
    try:
        S = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        # extremely degenerate – fall back to zeros
        k = 1
        return np.zeros((n, k), float), SimpleNamespace(explained_variance_ratio_=np.zeros(k, float))

    tol = np.finfo(float).eps * max(n, d) * (S[0] if S.size else 0.0)
    rank = int((S > tol).sum())

    k = max(0, min(want_components, rank, n - 1))
    if k == 0:
        k = 1
        return np.zeros((n, k), float), SimpleNamespace(explained_variance_ratio_=np.zeros(k, float))

    from sklearn.decomposition import PCA
    pca = PCA(n_components=k, svd_solver="full")
    coords = pca.fit_transform(X)
    # guard against divide-by-zero in sklearn attrs
    if not np.isfinite(getattr(pca, "explained_variance_ratio_", np.array([1.0]))).all():
        pca.explained_variance_ratio_ = np.zeros(k, float)
    return coords, pca

def compute_pca_coords(embedding_weights, num_components=17):
    """
    Given embedding_weights (a NumPy array), compute and return the first num_components principal components.
    The data is centered and PCA is used so that the returned array has shape (n_points, num_components).
    Tries scikit-learn PCA first, falls back to manual SVD if sklearn is not available.
    """
    return _safe_pca_coords(np.asarray(embedding_weights), num_components)

# ---Diffusion Coordinates

def compute_diffusion_coords(
    embedding_weights: np.ndarray,
    num_coords: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the first *num_coords* non-trivial diffusion coordinates
    together with the full descending eigenvalue array (λ0…λn)."""
    # compute pairwise distances and kernel
    N = int(embedding_weights.shape[0])
    max_nontrivial = max(N - 1, 1)
    want = int(num_coords)
    k = min(want, max_nontrivial)

    d2 = squareform(pdist(embedding_weights, metric="euclidean")) ** 2
    eps = float(np.median(d2))
    if not np.isfinite(eps) or eps <= 0:
        pos = d2[d2 > 0]
        eps = float(pos.mean()) if pos.size else 1e-12

    A = np.exp(-d2 / eps)
    M = A / A.sum(axis=1, keepdims=True)

    # eigendecomposition
    eigenvalues, eigenvectors = eigh(M)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    if eigenvalues.shape[0] < num_coords + 1:
        raise ValueError(
            "Not enough eigenvalues to compute the requested diffusion coordinates."
        )

    # extract non-trivial coords
    coords = eigenvectors[:, 1 : num_coords + 1] * eigenvalues[1 : num_coords + 1]
    return coords, eigenvalues

def make_json(
    freq_list: list[int] | None,
    var_ratio: list[float],
    cum_ratio: list[float],
    save_dir: str
) -> None:
    """
    Dump frequency list and variance ratios as JSON into *save_dir*:
      {
        "freq_list": [...],
        "variance_ratio": [...],
        "cumulative_variance_ratio": [...]
      }
    The file is named "variance_explained.json" and is placed
    directly under *save_dir*.
    """
    os.makedirs(save_dir, exist_ok=True)
    data = {
        "freq_list": freq_list,
        "variance_ratio": var_ratio,
        "cumulative_variance_ratio": cum_ratio,
    }
    out_path = os.path.join(save_dir, "variance_explained.json")
    with open(out_path, "w") as fh:
        json.dump(data, fh, indent=4)

# ---------------- Plotting Functions ----------------
# ── helpers ────────────────────────────────────────────────────────────
def _make_hover(a_vals: np.ndarray, b_vals: np.ndarray) -> dict:
    """
    Return Plotly keyword args that inject the a,b coords into every point’s
    hover tooltip.  Usage:
        fig.add_trace(
            go.Scatter(...,
                       **_make_hover(a_vals, b_vals))
        )
    """
    custom = np.stack([a_vals, b_vals], axis=1)   # shape (N,2)
    return dict(
        customdata=custom,
        hovertemplate="a=%{customdata[0]}<br>b=%{customdata[1]}<extra></extra>"
    )

def generate_new_diffusion_plot(embedding_weights, output_file, p):
    """
    Compute the first 17 diffusion coordinates and plot 16 scatter plots of coordinate pairs:
    (Coord1 vs Coord2), (Coord2 vs Coord3), ..., (Coord16 vs Coord17) in a 4x4 grid.
    """
    diff_coords, _ = compute_diffusion_coords(embedding_weights, num_coords=17)
    num_plots = 16  # pairs: 1-2, 2-3, ..., 16-17
    fig = make_subplots(rows=4, cols=4,
                        subplot_titles=[f"Coord {i+1} vs {i+2}" for i in range(num_plots)])
    
    labels = np.arange(diff_coords.shape[0]) % p
    
    marker_args = dict(
        color=labels,
        colorscale=[(0.0, 'blue'), (0.5, 'red'), (1.0, 'blue')],
        cmin=0,
        cmax=p-1,
        size=6
    )
    
    plot_idx = 0
    for i in range(4):
        for j in range(4):
            x_coord = diff_coords[:, plot_idx]
            y_coord = diff_coords[:, plot_idx + 1]
            fig.add_trace(
                go.Scatter(x=x_coord, y=y_coord,
                           mode='markers', marker=marker_args),
                row=i+1, col=j+1
            )
            plot_idx += 1

    fig.update_layout(height=1000, width=1000,
                      title_text="New Diffusion Plot (16 coordinate pair plots)",
                      showlegend=False)
    fig.write_html(output_file, include_plotlyjs="cdn")
    print(f"New diffusion plot saved to {output_file}")

def create_2d_diffusion_figure(embedding_weights, color_values, title_text, p):
    """
    Compute the first 17 diffusion coordinates and create a 4x4 grid of 16 scatter plots
    (Coord1 vs Coord2, Coord2 vs Coord3, ..., Coord16 vs Coord17). Each point shows custom
    hover text "a={a}, b={b}, y={y}" computed from its index.
    The marker colors are set by the provided color_values.
    """
    diff_coords, _ = compute_diffusion_coords(embedding_weights, num_coords=17)
    num_plots = 16
    fig = make_subplots(
        rows=4, cols=4,
        subplot_titles=[f"Coord {i+1} vs {i+2}" for i in range(num_plots)]
    )
    
    n_points = diff_coords.shape[0]
    indices = np.arange(n_points)
    a_vals = indices // p
    b_vals = indices % p
    y_vals = (a_vals + b_vals) % p
    hover_texts = [f"a={a}, b={b}, y={y}" for a, b, y in zip(a_vals, b_vals, y_vals)]
    
    marker_args = dict(
        color=color_values,
        colorscale=[(0.0, 'blue'), (1.0, 'red')],
        cmin=0,
        cmax=p-1,
        size=6
    )
    
    plot_idx = 0
    for i in range(4):
        for j in range(4):
            x_coord = diff_coords[:, plot_idx]
            y_coord = diff_coords[:, plot_idx + 1]
            trace = go.Scatter(
                x=x_coord,
                y=y_coord,
                mode='markers',
                marker=marker_args,
                hovertext=hover_texts,
                hovertemplate='%{hovertext}<extra></extra>'
            )
            fig.add_trace(trace, row=i+1, col=j+1)
            plot_idx += 1

    fig.update_layout(
        height=1000,
        width=1000,
        title_text=title_text,
        showlegend=False
    )
    return fig

def create_3d_diffusion_figure(embedding_weights, color_values, title_text, p):
    """
    Compute the first 17 diffusion coordinates and create a 3x5 grid (15 subplots) of 3D scatter plots.
    Each subplot plots three consecutive coordinates:
      (Coord1, Coord2, Coord3), (Coord2, Coord3, Coord4), ..., (Coord15, Coord16, Coord17).
    """
    diff_coords, _ = compute_diffusion_coords(embedding_weights, num_coords=17)
    num_plots = 15
    rows, cols = 3, 5
    specs = [[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Coords {i+1}-{i+3}" for i in range(num_plots)],
        specs=specs,
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    n_points = diff_coords.shape[0]
    indices = np.arange(n_points)
    a_vals = indices // p
    b_vals = indices % p
    y_vals = (a_vals + b_vals) % p
    hover_texts = [f"a={a}, b={b}, y={y}" for a, b, y in zip(a_vals, b_vals, y_vals)]
    
    marker_args = dict(
        size=4,
        color=color_values,
        colorscale=[(0.0, 'blue'), (1.0, 'red')],
        cmin=0,
        cmax=p-1,
    )
    
    plot_idx = 0
    for i in range(rows):
        for j in range(cols):
            if plot_idx < num_plots:
                x_data = diff_coords[:, plot_idx]
                y_data = diff_coords[:, plot_idx + 1]
                z_data = diff_coords[:, plot_idx + 2]
                trace = go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode='markers',
                    marker=marker_args,
                    hovertext=hover_texts,
                    hovertemplate='%{hovertext}<extra></extra>'
                )
                fig.add_trace(trace, row=i+1, col=j+1)
                scene_id = f'scene{(i * cols + j + 1) if (i * cols + j + 1) > 1 else ""}'
                fig.layout[scene_id].xaxis.title = f"diff coord {plot_idx + 1}"
                fig.layout[scene_id].yaxis.title = f"diff coord {plot_idx + 2}"
                fig.layout[scene_id].zaxis.title = f"diff coord {plot_idx + 3}"
                plot_idx += 1

    fig.update_layout(
        height=1200,
        width=1800,
        title_text=title_text,
        showlegend=False
    )
    return fig


def create_3d_pca_figure(embedding_weights, color_values, title_text, p):
    """
    Compute the principal components and create a grid of 3D scatter plots.
    Each subplot plots three consecutive components:
      (PC1, PC2, PC3), (PC2, PC3, PC4), ..., (PC_(n-2), PC_(n-1), PC_n).
    """
    pca_coords,_ = compute_pca_coords(embedding_weights, num_components=17)
    available_components = pca_coords.shape[1]
    
    if available_components < 3:
        raise ValueError("Not enough PCA components to create a 3D plot.")

    num_plots = available_components - 2
    cols = min(5, num_plots)
    rows = (num_plots + cols - 1) // cols

    specs = [[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"PCs {i+1}-{i+3}" for i in range(num_plots)],
        specs=specs,
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    n_points = pca_coords.shape[0]
    indices = np.arange(n_points)
    a_vals = indices // p
    b_vals = indices % p
    y_vals = (a_vals + b_vals) % p
    hover_texts = [f"a={a}, b={b}, y={y}" for a, b, y in zip(a_vals, b_vals, y_vals)]
    
    marker_args = dict(
        size=4,
        color=color_values,
        colorscale=[(0.0, 'blue'), (1.0, 'red')],
        cmin=0,
        cmax=p-1,
    )
    
    plot_idx = 0
    for i in range(rows):
        for j in range(cols):
            if plot_idx < num_plots:
                x_data = pca_coords[:, plot_idx]
                y_data = pca_coords[:, plot_idx + 1]
                z_data = pca_coords[:, plot_idx + 2]
                trace = go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode='markers',
                    marker=marker_args,
                    hovertext=hover_texts,
                    hovertemplate='%{hovertext}<extra></extra>'
                )
                fig.add_trace(trace, row=i+1, col=j+1)
                scene_id = f'scene{(i * cols + j + 1) if (i * cols + j + 1) > 1 else ""}'
                fig.layout[scene_id].xaxis.title = f"PCA coord {plot_idx + 1}"
                fig.layout[scene_id].yaxis.title = f"PCA coord {plot_idx + 2}"
                fig.layout[scene_id].zaxis.title = f"PCA coord {plot_idx + 3}"
                plot_idx += 1

    fig.update_layout(
        height=1200,
        width=1800,
        title_text=title_text,
        showlegend=False
    )
    return fig

# ---Interactive embedding plot code

def generate_diffusion_map_figure(embedding_weights, epoch, p, f_multiplier=1, diffusion_coords=None):
    """
    Given embedding_weights and an epoch number, return a Plotly figure showing the diffusion map.
    If diffusion_coords is provided, use it instead of recomputing.
    """
    if diffusion_coords is None:
        diffusion_coords, _ = compute_diffusion_coords(embedding_weights)
    
    num_points = diffusion_coords.shape[0]
    if num_points == p:
        indices = np.arange(num_points)
        labels = (f_multiplier * indices) % p
    elif num_points == p*p:
        indices = np.arange(num_points)
        a = indices // p
        b = indices % p
        labels = (a + b) % p
    else:
        labels = np.zeros(num_points)
    
    custom_colorscale = [(0.0, 'blue'), (0.5, 'red'), (1.0, 'blue')]
    
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Coordinate 1 vs 2", "Coordinate 2 vs 3",
                                        "Coordinate 3 vs 4", "Coordinate 4 vs 5"))
    
    marker_args = dict(
        color=labels,
        colorscale=custom_colorscale,
        cmin=0,
        cmax=p-1,
        size=8,
        colorbar=dict(title="(f * index) mod p")
    )
    
    fig.add_trace(
        go.Scatter(x=diffusion_coords[:, 0], y=diffusion_coords[:, 1],
                   mode='markers', marker=marker_args),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=diffusion_coords[:, 1], y=diffusion_coords[:, 2],
                   mode='markers', marker=marker_args),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=diffusion_coords[:, 2], y=diffusion_coords[:, 3],
                   mode='markers', marker=marker_args),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=diffusion_coords[:, 3], y=diffusion_coords[:, 4],
                   mode='markers', marker=marker_args),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Diffusion Coordinate 1", row=1, col=1)
    fig.update_yaxes(title_text="Diffusion Coordinate 2", row=1, col=1)
    
    fig.update_xaxes(title_text="Diffusion Coordinate 2", row=1, col=2)
    fig.update_yaxes(title_text="Diffusion Coordinate 3", row=1, col=2)
    
    fig.update_xaxes(title_text="Diffusion Coordinate 3", row=2, col=1)
    fig.update_yaxes(title_text="Diffusion Coordinate 4", row=2, col=1)
    
    fig.update_xaxes(title_text="Diffusion Coordinate 4", row=2, col=2)
    fig.update_yaxes(title_text="Diffusion Coordinate 5", row=2, col=2)
    
    fig.update_layout(height=800, width=800,
                      title_text=f"Diffusion Map (Epoch {epoch}, f_multiplier={f_multiplier})",
                      showlegend=False)
    
    return fig

def generate_interactive_diffusion_map_html(epoch_embedding_log, output_file, p, f_multiplier=1):
    """
    Given a dictionary keyed by epoch with embedding weight matrices,
    create an interactive Plotly figure using precomputed diffusion coordinates per epoch.
    """
    sorted_epochs = sorted(epoch_embedding_log.keys())
    frames = []
    for idx, epoch in enumerate(sorted_epochs):
        emb_weights = np.array(epoch_embedding_log[epoch])
        diff_coords, _ = compute_diffusion_coords(emb_weights)
        fig_epoch = generate_diffusion_map_figure(emb_weights, epoch, p, f_multiplier=f_multiplier, diffusion_coords=diff_coords)
        frame = go.Frame(data=fig_epoch.data, name=str(epoch))
        frames.append(frame)
        print(f"Made diffusion plot for epoch {epoch} (f_multiplier={f_multiplier}).")
    
    base_epoch = sorted_epochs[0]
    base_emb_weights = np.array(epoch_embedding_log[base_epoch])
    base_diff_coords, _ = compute_diffusion_coords(base_emb_weights)
    base_fig = generate_diffusion_map_figure(base_emb_weights, base_epoch, p, f_multiplier=f_multiplier, diffusion_coords=base_diff_coords)
    
    slider_steps = []
    for epoch in sorted_epochs:
        step = dict(
            label=str(epoch),
            method="animate",
            args=[[str(epoch)],
                  {"mode": "immediate",
                   "frame": {"duration": 300, "redraw": True},
                   "transition": {"duration": 200}}]
        )
        slider_steps.append(step)
    
    base_fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1,
            x=1.1,
            xanchor="right",
            yanchor="top",
            pad={"t": 0, "r": 10},
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {
                    "frame": {"duration": 300, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 200}
                }]
            )]
        )],
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Epoch: "},
            pad={"t": 50},
            steps=slider_steps
        )]
    )
    
    base_fig.frames = frames
    base_fig.write_html(output_file, include_plotlyjs="cdn")
    print(f"Interactive diffusion map saved to {output_file}")

def _split_dualscale_views(coords: np.ndarray,
                           colour: np.ndarray,
                           p_cbar: int):
    """
    将“拼接色条”（长度=2*h，前半 Orange、后半 Viridis）切成两份：
      - viridis 半：colour ∈ [0, h)
      - orange 半：colour ∈ [h, 2h) → 重映射到 [0, h)
    返回：
      (coords_orange, colour_orange, h, scale_orange),
      (coords_viridis, colour_viridis, h, scale_viridis)
    其中 coords_* 对应未选中的点坐标置 NaN（Plotly 不会画出来）。
    """
    h = int(p_cbar // 2)

    # 掩蔽坐标（未选中点 → NaN），颜色未选中点 → NaN
    mask_orange  = (colour >= h)
    mask_viridis = ~mask_orange

    coords_orange  = coords.copy().astype(float)
    coords_orange[~mask_orange, :] = np.nan
    colour_orange  = np.where(mask_orange, colour - h, np.nan)

    coords_viridis  = coords.copy().astype(float)
    coords_viridis[~mask_viridis, :] = np.nan
    # 把后半段重新映射到 [0, h)
    colour_viridis  = np.where(mask_viridis, colour, np.nan)

    scale_orange  = build_ro_scale(h)  # 你的“红→橙”刻度
    scale_viridis = build_vi_scale(h)

    return (coords_orange,  colour_orange,  h, scale_orange), \
           (coords_viridis, colour_viridis, h, scale_viridis)


# Start of make .pdf code
def _write_multiplot_2d(coords: np.ndarray,
                        colour: np.ndarray,
                        ctitle: str,
                        out_path: str,
                        p: int,
                        p_cbar: int,
                        colorscale: str,
                        seed,
                        label: str,
                        tag: str) -> None:
    """
    Generate 2-D scatter plots for *all* coordinate pairs.
    Each PDF “page” contains 8 × 4 = 32 sub-plots; pages are merged
    into a single multi-page PDF written to ``out_path``.

    Parameters
    ----------
    coords : (n_points, n_dims) array
        Point coordinates.
    colour : (n_points,) array
        Per-point colour values.
    ctitle : str
        Title for the colour-bar.
    out_path : str
        Destination PDF (multi-page).
    p : int
        Modulus for colour bar scaling.
    seed, label, tag
        Passed straight through to titles / figure text.
    """
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coords must be 2-D with at least two columns.")
    if PdfMerger is None:
        raise ImportError(
            "PyPDF2 is required for PDF concatenation but could not be imported."
        ) from _pdf2_err

    # ── generate (i,j) pairs ────────────────────────────────────────────
    pairs = list(itertools.combinations(range(coords.shape[1]), 2))
    per_page = 32
    n_pages = math.ceil(len(pairs) / per_page)

    tmp_files: List[str] = []
    n_pts   = coords.shape[0]
    side  = int(math.isqrt(n_pts))   
    indices = np.arange(n_pts)
    a_vals = indices // side        
    b_vals = indices %  side         

    for page in range(n_pages):
        page_pairs = pairs[page * per_page:(page + 1) * per_page]
        n_cols, n_rows = 4, max(1, math.ceil(len(page_pairs) / 4))

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"{label}{i} vs {label}{j}" for i, j in page_pairs],
            horizontal_spacing=0.04,
            vertical_spacing=0.06,
        )

        hover_kw = _make_hover(a_vals, b_vals)         
        for k, (i, j) in enumerate(page_pairs, 1):
            r, c = 1 + (k - 1) // n_cols, 1 + (k - 1) % n_cols
          
            step      = max(1, p_cbar // 10)          
            tickvals  = list(range(0, p_cbar, step))  # 0, step, 2·step, …
            if tickvals[-1] != p_cbar - 1:      
                tickvals.append(p_cbar - 1)
            ticktext  = [str(v) for v in tickvals]   
            x = _jitter_if_constant(coords[:, i])
            y = _jitter_if_constant(coords[:, j]) 
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    name="",
                    showlegend=False,
                    marker=dict(
                    size=4,
                    color=colour,
                    colorscale=colorscale,
                    cmin=0,
                    cmax=p_cbar - 1,
                    line=dict(width=0),
                    showscale=(k == 1),
                    colorbar=dict(
                        title=dict(text=ctitle, side="right", font=dict(size=CBAR_TITLE_SIZE)),
                        tickvals=tickvals,
                        ticktext=ticktext,
                        tickfont=dict(size=CBAR_TICK_SIZE),
                        len=0.90,          # 可选：让色条更长一点
                    ),
                ),
                    **hover_kw                                  
                ),
                row=r,
                col=c,
            )

            fig.update_xaxes(title_text=f"{label}{i}", row=r, col=c)
            fig.update_yaxes(title_text=f"{label}{j}", row=r, col=c)

        fig.update_layout(
            width=1400,
            height=250 * n_rows + 100,  # dynamic vertical size
            title=f"{label} 2-D – seed {seed} – page {page + 1}/{n_pages} - {tag}",
            margin=dict(l=40, r=40, t=80, b=40),
        )
        fig.update_layout(font=dict(size=FONT_SIZE))

        # subplot_titles is annotations
        if hasattr(fig.layout, "annotations") and fig.layout.annotations:
            for ann in fig.layout.annotations:
                if ann is not None and hasattr(ann, "font") and ann.font is not None:
                    ann.font.size = FONT_SIZE

        # axis title and ticks font size
        fig.update_xaxes(title_font=dict(size=FONT_SIZE), tickfont=dict(size=TICK_SIZE))
        fig.update_yaxes(title_font=dict(size=FONT_SIZE), tickfont=dict(size=TICK_SIZE))

        # Write page to a temp file
        tmp_pdf = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.pdf")
        fig.write_image(tmp_pdf, format="pdf")
        html_name = os.path.basename(out_path).replace(".pdf", f"_page{page+1}.html")
        html_path = os.path.join(os.path.dirname(out_path), html_name)
        p = Path(html_path)                      # html_path 可能是 str，转成 Path 更稳
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(p), include_plotlyjs="cdn")

        tmp_files.append(tmp_pdf)

    # ── merge pages into one multi-page PDF ─────────────────────────────
    merger = PdfMerger()
    for pdf in tmp_files:
        merger.append(pdf)
    merger.write(out_path)
    merger.close()

    # clean-up
    for pdf in tmp_files:
        try:
            os.remove(pdf)
        except OSError:
            pass

    print(f"[{label} 2-D]  →  {out_path}")

# def _write_multiplot_3d(coords: np.ndarray,
#                         colour: np.ndarray,
#                         ctitle: str,
#                         out_path: str,
#                         p: int,
#                         p_cbar: int,
#                         colorscale: str,
#                         seed,
#                         label: str,
#                         tag: str):
#     """
#     Four-panel 3-D scatter (all 4 choose 3 combos) → *PDF*.
#     """
#     triplets = list(itertools.combinations(range(4), 3))
#     fig = make_subplots(
#         rows=2, cols=2,
#         specs=[[{'type': 'scene'}]*2]*2,
#         subplot_titles=[f"{label}{i} vs {label}{j} vs {label}{k}"
#                         for i, j, k in triplets],
#         horizontal_spacing=0.03, vertical_spacing=0.03
#     )
#     n_pts   = coords.shape[0]
#     indices = np.arange(n_pts)
#     side  = int(math.isqrt(n_pts))
#     a_vals = indices // side
#     b_vals = indices %  side
#     hover_kw = _make_hover(a_vals, b_vals)
#     for idx, (i, j, k) in enumerate(triplets, 1):
#         row, col = (1, idx) if idx <= 2 else (2, idx-2)
#         step      = max(1, p_cbar // 10)          
#         tickvals  = list(range(0, p_cbar, step))  
#         if tickvals[-1] != p_cbar - 1:            
#             tickvals.append(p_cbar - 1)
#         ticktext  = [str(v) for v in tickvals]    
#         fig.add_trace(
#             go.Scatter3d(
#                 x=coords[:, i], y=coords[:, j], z=coords[:, k],
#                 mode='markers',
#                 name='',  
#                 showlegend=False, 
#                 marker=dict(
#                     size=3,
#                     color=colour,
#                     colorscale=colorscale,
#                     cmin=0,
#                     cmax=p_cbar-1,
#                     showscale=(idx == 1),
#                     colorbar=dict(
#                         title=ctitle,
#                         tickvals=tickvals,
#                         ticktext=ticktext,
#                     ),
#                 ),
#                 **hover_kw   
#             ),
#             row=row, col=col
#         )
#         scene_id = f"scene{idx if idx > 1 else ''}"
#         fig.layout[scene_id].xaxis.title.text = f"{label}{i}"
#         fig.layout[scene_id].yaxis.title.text = f"{label}{j}"
#         fig.layout[scene_id].zaxis.title.text = f"{label}{k}"

#     fig.update_layout(
#         width=1000, height=900,
#         title=f"{label} 3-D (first 4) – seed {seed} - {tag}",
#         margin=dict(l=40, r=40, t=80, b=40),
#     )
#     fig.write_image(out_path, format="pdf")
#     html_name = os.path.basename(out_path).replace(".pdf", f".html")
#     html_path = os.path.join(os.path.dirname(out_path), html_name)
#     fig.write_html(html_path, include_plotlyjs="cdn")
#     print(f"[{label} 3-D]  →  {out_path}")


def _write_multiplot_3d(coords: np.ndarray,
                        colour: np.ndarray,
                        ctitle: str,
                        out_path: str,
                        p: int,
                        p_cbar: int,
                        colorscale: str,
                        seed,
                        label: str,
                        tag: str,
                        f: int,
                        mult: bool
                        ):
    """
    Four-panel 3-D scatter  ➜  PDF  (+HTML interactive)

    Parameters
    ----------
    coords : (n_points, n_dims) array
        Point coordinates.
    colour : (n_points,) array
        Per-point colour values.
    ctitle : str
        Title for the colour-bar.
    out_path : str
        Destination PDF (multi-page).
    p : int
        Modulus for colour bar scaling.
    seed, label, tag
        Passed straight through to titles / figure text.
    f : int
        Frequency
    """
    g       = p // math.gcd(p, f) or p
    n_pts   = coords.shape[0]
    side    = int(math.isqrt(n_pts))
    multi_view = (g != p) and (side == 2*p) and (mult)
    triplets   = list(itertools.combinations(range(4), 3))   # 4C3 = 4

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}]*2]*2,
        subplot_titles=[f"{label}{i} vs {label}{j} vs {label}{k}"
                        for i, j, k in triplets],
        horizontal_spacing=0.03, vertical_spacing=0.03
    )

    # ─── 共有数据预处理 ──────────────────────────────────────────────
    idxs    = np.arange(n_pts)
    a_vals  = idxs // side
    b_vals  = idxs %  side
    hover_kw = _make_hover(a_vals, b_vals)

    # 如果不是多视图：直接画一次散点（用函数输入的 colour / colorscale）
    if not multi_view:
        for s_idx, (i, j, k) in enumerate(triplets, 1):
            row, col = (1, s_idx) if s_idx <= 2 else (2, s_idx-2)
            step      = max(1, p_cbar // 10)
            ticks     = list(range(0, p_cbar, step))
            if ticks[-1] != p_cbar-1:
                ticks.append(p_cbar-1)

            x = _jitter_if_constant(coords[:, i])
            y = _jitter_if_constant(coords[:, j])
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=coords[:, k],
                    mode="markers",
                    marker=dict(size=3,
                                color=colour,
                                colorscale=colorscale,
                                cmin=0, cmax=p_cbar-1,
                                showscale=(s_idx == 1),
                                colorbar=dict(
                                    title=dict(text=ctitle, side="right", font=dict(size=CBAR_TITLE_SIZE)),
                                    tickfont=dict(size=CBAR_TICK_SIZE),
                                    len=0.90,
                                ),
                            ),
                    **hover_kw
                ),
                row=row, col=col
            )
            sid = f"scene{s_idx if s_idx>1 else ''}"
            fig.layout[sid].xaxis.title.text = f"{label}{i}"
            fig.layout[sid].yaxis.title.text = f"{label}{j}"
            fig.layout[sid].zaxis.title.text = f"{label}{k}"

    # ─── Multi ─────────────────────────────────────────
    else:

        # color vec
        col_a, _, _, cs_a = colour_quad_a_only(a_vals, b_vals, p, f, "full")
        col_b, _, _, cs_b = colour_quad_b_only(a_vals, b_vals, p, f, "full")
        col_c, _, pcbar_c, cs_c = colour_quad_mod_g(a_vals, b_vals, p, f, "full")

        # line idx
        h_pairs = lines_a_mod_g(a_vals, b_vals,p,g)
        # for b_fix in list(range(g)) + [p + i for i in range(g)]:
        #     # --- split according to a_vals ---
        #     solid_idx = np.where((a_vals < g) & (b_vals == b_fix))[0]
        #     dash_idx  = np.where((a_vals >= p) & (a_vals < p + g) & (b_vals == b_fix))[0]

        #     # --- color ---
        #     # b_fix < p blue，else red
        #     color = "blue" if b_fix < p else "red"
        #     line_id = f"b_{b_fix}" 
            
        #     if solid_idx.size > 1:
        #         h_pairs.append((solid_idx, "solid", color, line_id))
        #     if dash_idx.size  > 1:
        #         h_pairs.append((dash_idx,  "dash",  color, line_id))

        v_pairs = lines_b_mod_g(a_vals, b_vals,p,g)
        # for a_fix in list(range(g)) + [p + i for i in range(g)]:
        #     # --- split according to b_vals ---
        #     solid_idx = np.where((b_vals < g) & (a_vals == a_fix))[0]
        #     dash_idx  = np.where((b_vals >= p) & (b_vals < p + g) & (a_vals == a_fix))[0]

        #     # --- color ---
        #     # a_fix < p blue，else
        #     color = "blue" if a_fix < p else "red"
        #     line_id = f"a_{a_fix}" 
            
        #     if solid_idx.size > 1:
        #         v_pairs.append((solid_idx, "solid", color, line_id))
        #     if dash_idx.size  > 1:
        #         v_pairs.append((dash_idx,  "dash",  color, line_id))
        c_pairs = lines_c_mod_g(a_vals, b_vals,p,g)

        n_h, n_v, n_c = len(h_pairs), len(v_pairs), len(c_pairs)

        legend_shown_a, legend_shown_b,legend_shown_c = set(), set(), set()
        for s_idx, (i, j, k) in enumerate(triplets, 1):
            row, col = (1, s_idx) if s_idx <= 2 else (2, s_idx-2)

            x = _jitter_if_constant(coords[:, i])
            y = _jitter_if_constant(coords[:, j])
            # — a-view scatter (visible in default)
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=coords[:, k],
                    mode="markers",
                    marker=dict(size=3,
                                color=col_a, colorscale=cs_a,
                                cmin=0, cmax=2*g-1,
                                showscale=(s_idx == 1),
                                colorbar=dict(
                                    title=dict(text=ctitle, side="right", font=dict(size=CBAR_TITLE_SIZE)),
                                    tickfont=dict(size=CBAR_TICK_SIZE),
                                    len=0.90,
                                ),
                            ),
                    name="a mod g",
                    legendgroup="a",
                    visible=True,
                    **hover_kw
                ), row=row, col=col
            )

            # — b-view scatter (hidden)
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=coords[:, k],
                    mode="markers",
                    marker=dict(size=3,
                                color=col_b, colorscale=cs_b,
                                cmin=0, cmax=2*g-1,
                                showscale=(s_idx == 1),
                                colorbar=dict(
                                    title=dict(text=ctitle, side="right", font=dict(size=CBAR_TITLE_SIZE)),
                                    tickfont=dict(size=CBAR_TICK_SIZE),
                                    len=0.90,
                                ),
                            ),
                    name="b mod g",
                    legendgroup="b",
                    visible=False,
                    **hover_kw
                ), row=row, col=col
            )
            # — c-view scatter (hidden)
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=coords[:, k],
                    mode="markers",
                    marker=dict(size=3,
                                color=col_c, colorscale=cs_c,
                                cmin=0, cmax=pcbar_c-1,
                                showscale=(s_idx == 1),
                                colorbar=dict(
                                    title=dict(text=ctitle, side="right", font=dict(size=CBAR_TITLE_SIZE)),
                                    tickfont=dict(size=CBAR_TICK_SIZE),
                                    len=0.90,
                                ),
                            ),
                    name="c mod g",
                    legendgroup="c",
                    showlegend=(s_idx == 1),
                    visible=False,
                    **hover_kw
                ), row=row, col=col
            )

            # — (lines for graph a)
            for idx_arr, dash, color, gid in h_pairs:
                idx_sorted = idx_arr[np.argsort(a_vals[idx_arr])]

                if idx_sorted.size > 2:                 # at least 3 points to form a closed loop
                    idx_plot = np.concatenate([idx_sorted, idx_sorted[:1]])
                else:                                   # only 2 points
                    idx_plot = idx_sorted

                show_legend = gid not in legend_shown_a
                legend_shown_a.add(gid)
                fig.add_trace(
                    go.Scatter3d(
                        x=coords[idx_plot, i], y=coords[idx_plot, j], z=coords[idx_plot, k],
                        mode="lines",
                        name = gid,                  
                        legendgroup = gid,           # same group, visible/hidden the same time
                        showlegend = show_legend,
                        line=dict(color=color, dash=dash, width=1.2),
                        hoverinfo="skip",
                        visible=True
                    ), row=row, col=col
                )

            # — (lines for graph b)
            for idx_arr, dash, color, gid in v_pairs:
                idx_sorted = idx_arr[np.argsort(b_vals[idx_arr])]

                if idx_sorted.size > 2:
                    idx_plot = np.concatenate([idx_sorted, idx_sorted[:1]])
                else:                                   
                    idx_plot = idx_sorted

                show_legend = gid not in legend_shown_b
                legend_shown_b.add(gid)
                fig.add_trace(
                    go.Scatter3d(
                        x=coords[idx_plot, i], y=coords[idx_plot, j], z=coords[idx_plot, k],
                        mode="lines",
                        name = gid, 
                        legendgroup = gid, 
                        showlegend = show_legend,
                        line=dict(color=color, dash=dash, width=1.2),
                        hoverinfo="skip",
                        visible=False
                    ), row=row, col=col
                )

            # — (lines for graph c)
            for idx_arr, dash, color, gid in c_pairs:
                a_sub = a_vals[idx_arr]
                b_sub = b_vals[idx_arr]
                # lexsort 的第一个参数是次要关键字，第二个是主要关键字
                order = np.lexsort((b_sub, a_sub))
                idx_sorted = idx_arr[order]

                if idx_sorted.size > 2:
                    idx_plot = np.concatenate([idx_sorted, idx_sorted[:1]])
                else:                                   
                    idx_plot = idx_sorted

                show_legend = gid not in legend_shown_c
                legend_shown_c.add(gid)
                fig.add_trace(
                    go.Scatter3d(
                        x=coords[idx_plot, i], y=coords[idx_plot, j], z=coords[idx_plot, k],
                        mode="lines",
                        name = gid, 
                        legendgroup = gid, 
                        showlegend = show_legend,
                        line=dict(color=color, dash=dash, width=1.2),
                        hoverinfo="skip",
                        visible=False
                    ), row=row, col=col
                )

            sid = f"scene{s_idx if s_idx>1 else ''}"
            fig.layout[sid].xaxis.title.text = f"{label}{i}"
            fig.layout[sid].yaxis.title.text = f"{label}{j}"
            fig.layout[sid].zaxis.title.text = f"{label}{k}"

        # —— 按钮可见性向量
        vis_a = []
        vis_b = []
        vis_c = []
        for _ in range(len(triplets)):
            # 每个 subplot: [a-scatter, c-scatter, b-scatter, h-lines..., v-lines...]
            vis_a += [True,  False, False] + [True]*n_h + [False]*n_v + [False]*n_c
            vis_b += [False, True,  False] + [False]*n_h + [True]*n_v + [False]*n_c
            vis_c += [False, False, True ] + [False]*n_h + [False]*n_v + [True]*n_c
 
        fig.update_layout(
             updatemenus=[dict(
                buttons=[
                    dict(label="a mod g",
                         method="update",
                         args=[{"visible": vis_a},
                               {"title": "colour = a mod g"}]),
                    dict(label="b mod g",
                         method="update",
                         args=[{"visible": vis_b},
                               {"title": "colour = b mod g"}]),
                    dict(label="c mod g",
                         method="update",
                         args=[{"visible": vis_c},
                               {"title": "colour = c mod g"}]),
                ],
                 direction="down",
                 x=0.99, y=1.05, xanchor="left",
                 pad={"t": 0, "r": 6}
             )]
         )
        
        fig.update_layout(legend=LEGEND_POS)
        

    # 统一 3D 轴标题与刻度字号（所有子场景）
    for layout_key in fig.layout:
        if str(layout_key).startswith("scene"):
            scene = fig.layout[layout_key]
            scene.xaxis.title.font = dict(size=FONT_SIZE)
            scene.yaxis.title.font = dict(size=FONT_SIZE)
            scene.zaxis.title.font = dict(size=FONT_SIZE)
            scene.xaxis.tickfont = dict(size=TICK_SIZE)
            scene.yaxis.tickfont = dict(size=TICK_SIZE)
            scene.zaxis.tickfont = dict(size=TICK_SIZE)

    # ─── general Layout & output ───────────────────────────────────────────
    fig.update_layout(
        width=1600, height=1200,     # 原来 1000×900 -> 放大
        title=f"{label} 3-D (first 4) - seed {seed} - {tag}",
        margin=dict(l=40, r=40, t=80, b=40),
        font=dict(size=FONT_SIZE),
    )
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(out_path, format="pdf")                 # static PDF
    fig.write_html(os.path.splitext(out_path)[0] + ".html",
                   include_plotlyjs="cdn")                  # interactive HTML
    print(f"[{label} 3-D] → {out_path}")


def save_homology_artifacts(coords: np.ndarray,
                            root_dir: str,
                            tag: str,
                            seed,
                            label: str,
                            num_dims: int | None = 2) -> None:
    """
    Thin wrapper that decides the sub-folder and filename stem,
    then calls `run_ph_for_point_cloud` on the first `num_dims` coords (if set).

    Parameters
    ----------
    num_dims : int | None
        Number of leading dimensions to keep for PH; if None, uses all dims.
    """
    # decide sub-folder and filename stem
    subdir = os.path.join(root_dir, "homology", tag)
    stem = f"{label.lower()}_seed_{seed}"

    # select only the first num_dims dimensions (if requested)
    n_nbrs = 150
    if num_dims is not None:
        if num_dims < 1 or num_dims > coords.shape[1]:
            raise ValueError(f"num_dims must be between 1 and {coords.shape[1]}")
        coords_to_use = coords[:, :num_dims]
    else:
        coords_to_use = coords

    if num_dims == 2:
        n_nbrs = 300
    
    run_ph_for_point_cloud(
        coords_to_use,
        maxdim=2,
        ph_sparse=True,
        n_nbrs=n_nbrs,
        save_dir=subdir,
        filename_stem=stem,
        title=f"{label}  (seed={seed})"
    )

#  Helper: phase-scatter & vector plots for the equal-frequency case
def _make_single_freq_phase_plots(mat: np.ndarray,
                                  p: int,
                                  f: int,
                                  save_dir: str,
                                  *,
                                  seed: int | str = "",
                                  tag: str = "",
                                  colour_scale: str = "Viridis",
                                  eps: float = 0.16) -> None:
    """
    Build a 2×2 PDF figure with
       (1,1) raw (φ_a, φ_b) scatter               coloured by amplitude
       (1,2) raw vectors from (0,0) to each point
       (2,1) merged-point scatter  (fat circles, label = Σ amps)
       (2,2) merged vectors (fat endpoints)

    A torus distance ≤ *eps* merges points; merged coordinates are
    the amplitude-weighted circular mean, amplitude = Σ amplitudes.
    """
    f = int(f) % p
    if f == 0:
        print("[phase-plots] f ≡ 0 (mod p) – skipped.")
        return

    # -- 1.  FFT → amplitudes & phases ----------------------------------
    n_neurons = mat.shape[1]
    amps  = np.empty(n_neurons)
    phi_a = np.empty(n_neurons)
    phi_b = np.empty(n_neurons)

    for n in range(n_neurons):
        grid   = mat[:, n].reshape(p, p).T
        F      = np.fft.fft2(grid) / (p * p)
        ca, cb = F[f, 0], F[0, f]

        amps[n]  = np.hypot(2*np.abs(ca), 2*np.abs(cb))
        phi_a[n] = (-np.angle(ca)) % (2*np.pi)
        phi_b[n] = (-np.angle(cb)) % (2*np.pi)

    # -- 2.  cluster / merge close points on the torus ------------------
    unpicked   = set(range(n_neurons))
    m_phi_a, m_phi_b, m_amp = [], [], []

    def torus_dist(x1, y1, x2, y2):
        dx = np.abs(x1 - x2); dx = np.minimum(dx, 2*np.pi - dx)
        dy = np.abs(y1 - y2); dy = np.minimum(dy, 2*np.pi - dy)
        return np.sqrt(dx*dx + dy*dy)

    while unpicked:
        i      = unpicked.pop()
        group  = [i]
        for j in list(unpicked):
            if torus_dist(phi_a[i], phi_b[i], phi_a[j], phi_b[j]) <= eps:
                unpicked.remove(j)
                group.append(j)

        # amplitude-weighted *circular* mean ----------------------------
        A      = amps[group]
        w_sum  = A.sum()
        ang_ax = np.arctan2((A*np.sin(phi_a[group])).sum(),
                            (A*np.cos(phi_a[group])).sum()) % (2*np.pi)
        ang_bx = np.arctan2((A*np.sin(phi_b[group])).sum(),
                            (A*np.cos(phi_b[group])).sum()) % (2*np.pi)

        m_phi_a.append(ang_ax)
        m_phi_b.append(ang_bx)
        m_amp .append(w_sum)

    m_phi_a = np.asarray(m_phi_a)
    m_phi_b = np.asarray(m_phi_b)
    m_amp   = np.asarray(m_amp)

    # -- 3.  build 2×2 Plotly figure -----------------------------------
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "raw scatter", "raw vectors",
            "merged scatter", "merged vectors"
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.15
    )

    # row-1-col-1 : raw scatter
    fig.add_trace(
        go.Scatter(
            x=phi_a, y=phi_b, mode="markers",
            marker=dict(size=6, color=amps,
                        colorscale=colour_scale,
                        colorbar=dict(title="amplitude")),
            hovertemplate="φₐ=%{x:.2f}<br>φ_b=%{y:.2f}<br>|A|=%{marker.color:.3f}"
            "<extra></extra>"
        ), row=1, col=1
    )

    # row-1-col-2 : raw vectors
    for pa, pb in zip(phi_a, phi_b):
        fig.add_trace(
            go.Scatter(x=[0, pa], y=[0, pb],
                       mode="lines",
                       line=dict(width=1.5, color="rgba(0,0,0,0.5)"),
                       hoverinfo="skip"), row=1, col=2
        )
    fig.add_trace(
        go.Scatter(
            x=phi_a, y=phi_b, mode="markers",
            marker=dict(size=6, color=amps,
                        colorscale=colour_scale, showscale=False),
            hovertemplate="φₐ=%{x:.2f}<br>φ_b=%{y:.2f}<br>|A|=%{marker.color:.3f}"
            "<extra></extra>"
        ), row=1, col=2
    )

    # row-2-col-1 : merged scatter  (fat, annotated)
    fig.add_trace(
        go.Scatter(
            x=m_phi_a, y=m_phi_b,
            mode="markers+text",
            marker=dict(size=12, color=m_amp,
                        colorscale=colour_scale, showscale=False,
                        line=dict(width=1, color="black")),
            text=[f"{a:.1f}" for a in m_amp],
            textposition="top center",
            hovertemplate="[merged]<br>φₐ=%{x:.2f}<br>φ_b=%{y:.2f}<br>|A|=%{marker.color:.3f}"
            "<extra></extra>"
        ), row=2, col=1
    )

    # row-2-col-2 : merged vectors
    for pa, pb in zip(m_phi_a, m_phi_b):
        fig.add_trace(
            go.Scatter(
                x=[0, pa], y=[0, pb],
                mode="lines",
                line=dict(width=2, color="rgba(0,0,0,0.6)"),
                hoverinfo="skip"
            ), row=2, col=2
        )
    fig.add_trace(
        go.Scatter(
            x=m_phi_a, y=m_phi_b,
            mode="markers+text",
            marker=dict(size=12, color=m_amp,
                        colorscale=colour_scale, showscale=False,
                        line=dict(width=1, color="black")),
            text=[f"{a:.1f}" for a in m_amp],
            textposition="top center",
            hovertemplate="[merged]<br>φₐ=%{x:.2f}<br>φ_b=%{y:.2f}<br>|A|=%{marker.color:.3f}"
            "<extra></extra>"
        ), row=2, col=2
    )

    # common axes labels
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(title_text="φₐ (rad)", row=r, col=c)
            fig.update_yaxes(title_text="φ_b (rad)", row=r, col=c)

    fig.update_layout(
        width=1100, height=900,
        title=f"Seed {seed} – f = {f} – {tag}",
        margin=dict(l=60, r=60, t=80, b=60),
        showlegend=False
    )

    # -- 4.  save PDF ---------------------------------------------------
    phase_dir = os.path.join(save_dir, "phase_plots")
    os.makedirs(phase_dir, exist_ok=True)
    fname_pdf = f"seed_{seed}_f{f}{'_'+tag if tag else ''}.pdf"
    out_path  = os.path.join(phase_dir, fname_pdf)
    fig.write_image(out_path, format="pdf")   # requires Ka leido
    print(f"[phase-plots] wrote {out_path}")


    
def _generate_pdf_plots_for_matrix(mat: np.ndarray,
                                  p: int,
                                  save_dir: str,
                                  *,
                                  seed: int | str = "",
                                  freq_list: list[int] | None = None,
                                  tag,
                                  tag_q: str = "",
                                  class_string: str = "",
                                  colour_rule=None,
                                  num_principal_components=2) -> None:
    """
    Create the same PCA & diffusion-map PDF plots as the research notebook,
    **without** any persistent-homology calls.

    Parameters
    ----------
    mat : np.ndarray
        Data matrix whose rows are the points to embed.
    p : int
        Alphabet size for colour-coding.
    save_dir : str
        Root folder into which PDFs are written.
    seed : int | str, optional
        Identifying seed for titles / file names.
    freq_list : list[int] | None, optional
        Frequency multipliers for the extra colourings
        (pass whatever you used in `final_grouping`).
    tag : str, optional
        Extra string inserted in file names (e.g. "embeds").
    """
    n_samples, n_features = mat.shape
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to compute diffusion coordinates.")
    num_components = min(n_features, 8, n_samples - 1)
    if colour_rule in (colour_quad_a_only, colour_quad_b_only):
        mult = True
    elif colour_rule in (colour_quad_mul_f, colour_quad_mod_g, colour_quad_mod_g_no_fb):
        mult = True # False
    else:
        raise ValueError(f"Unsupported colour_rule: {colour_rule!r}")
    if num_components >= 4:
        append_to_title = f"{tag} & {class_string}"
        freq_list = sorted(freq_list or [])
        os.makedirs(save_dir, exist_ok=True)
        
        is_grid_pp  = (n_samples == p * p)
        is_grid_2pp = (n_samples == (2 * p) * (2 * p))

        if tag_q == "full" and is_grid_2pp:
            side = 2 * p
            indices = np.arange(side * side)
            a_vals  = indices // side
            b_vals  = indices %  side
        elif is_grid_pp:
            indices = np.arange(p * p)
            a_vals  = indices // p
            b_vals  = indices %  p
        else:
            a_vals = None
            b_vals = None

        # assert len(a_vals) == n_samples, (
        #     f"Colour vector ({len(a_vals)}) and matrix rows ({n_samples}) mismatch; "
        #     "check p or tag settings."
        # )
        
        # if colour_rule is None:
        #     colour_base = (a_vals + b_vals) % p
        # else:
        #     colour_base = colour_rule(a_vals, b_vals, p, tag) % p

        # ─── Directory tree ──────────────────────────────────────────────────
        pca_root = os.path.join(save_dir, "pca_pdf_plots")
        dif_root = os.path.join(save_dir, "diffusion_pdf_plots")
        for root in (pca_root, dif_root):
            for sub in ("2d", "3d"):
                os.makedirs(os.path.join(root, sub, tag), exist_ok=True)
        print("computing PCA")
        # ─── PCA (4 comps) ───────────────────────────────────────────────────
        pcs, pca = compute_pca_coords(mat, num_components=num_components)

        base_2d_dir = os.path.join(pca_root, "2d", tag)
        # base_2d = os.path.join(
        #     base_2d_dir,
        #     f"pca_seed_{seed}{('_'+tag) if tag else ''}_base.pdf"
        # )
        # _write_multiplot_2d(pcs, colour_base, "(a+b) mod p",
        #                     base_2d, p, seed, "PC", append_to_title)
        pca_var_ratio = pca.explained_variance_ratio_.tolist()
        pca_cum_ratio = np.cumsum(pca.explained_variance_ratio_).tolist()
        make_json(freq_list, pca_var_ratio, pca_cum_ratio, base_2d_dir)
        
        base_3d_dir = os.path.join(pca_root, "3d", tag)
        # base_3d = os.path.join(
        #     base_3d_dir,
        #     f"pca_seed_{seed}{('_'+tag) if tag else ''}_base_3d.pdf"
        # )
        # _write_multiplot_3d(pcs, colour_base, "(a+b) mod p",
        #                     base_3d, p, seed, "PC", append_to_title)
        make_json(freq_list, pca_var_ratio, pca_cum_ratio, base_3d_dir)
        
        # Extra plots per-frequency
        if (a_vals is not None) and (b_vals is not None):
            coords_ab = np.stack([a_vals, b_vals], axis=1)
            for f in freq_list:
                if f < 0:
                    f_abs = abs(f)
                else:
                    f_abs = f
                if f_abs % p == 0:
                    continue
                if colour_rule is None:
                    raise ValueError("Color rule empty.")
                else:
                    colour, caption, p_cbar, colorscale = colour_rule(
                        a_vals, b_vals, p, f_abs, tag_q
                    )
                name_stub = f"pca_seed_{seed}_freq_{f}.pdf"
                _write_multiplot_2d(
                    pcs,
                    colour,
                    caption,
                    os.path.join(pca_root, "2d", tag, tag_q, name_stub.replace(".pdf", "_2d.pdf")),
                    p,p_cbar, colorscale, seed, "PC", append_to_title
                )
                _write_multiplot_3d(
                    pcs,
                    colour,
                    caption,
                    os.path.join(pca_root, "3d", tag, tag_q, name_stub.replace(".pdf", "_3d.pdf")),
                    p,p_cbar, colorscale, seed, "PC", append_to_title,f=f,mult=mult
                )
                if colour_rule in (colour_quad_mod_g, colour_quad_mul_f, colour_quad_a_only, colour_quad_b_only) and tag_q == "full":
                    (pcs_orange,  col_orange,  h, scale_orange), \
                    (pcs_viridis, col_viridis, h2, scale_viridis) = _split_dualscale_views(pcs, colour, p_cbar)

                    # Orange（TL/BR）
                    _write_multiplot_2d(
                        pcs_orange, col_orange, f"{caption} - TL/BR (Orange only)",
                        os.path.join(pca_root, "2d", tag, tag_q, name_stub.replace(".pdf", "_2d_orange.pdf")),
                        p, h, scale_orange, seed, "PC", append_to_title
                    )
                    _write_multiplot_3d(
                        pcs_orange, col_orange, f"{caption} - TL/BR (Orange only)",
                        os.path.join(pca_root, "3d", tag, tag_q, name_stub.replace(".pdf", "_3d_orange.pdf")),
                        p, h, scale_orange, seed, "PC", append_to_title, f=f, mult=False
                    )

                    # Viridis（BL/TR）
                    _write_multiplot_2d(
                        pcs_viridis, col_viridis, f"{caption} – BL/TR (Viridis only)",
                        os.path.join(pca_root, "2d", tag, tag_q, name_stub.replace(".pdf", "_2d_viridis.pdf")),
                        p, h2, scale_viridis, seed, "PC", append_to_title
                    )
                    _write_multiplot_3d(
                        pcs_viridis, col_viridis, f"{caption} – BL/TR (Viridis only)",
                        os.path.join(pca_root, "3d", tag, tag_q, name_stub.replace(".pdf", "_3d_viridis.pdf")),
                        p, h2, scale_viridis, seed, "PC", append_to_title, f=f, mult=False
                    )

        save_homology_artifacts(
            pcs,
            root_dir=pca_root,
            tag=tag_q,
            seed=seed,
            label=f"PCA--{class_string}",
            num_dims=num_principal_components)

        # ─── Diffusion (first 4) ─────────────────────────────────────────────
        dmap, eigenvalues = compute_diffusion_coords(mat, num_coords=num_components)

        
        # base_2d_d = os.path.join(base_2d_d_dir,
        #                          f"diff_seed_{seed}{('_'+tag) if tag else ''}_base.pdf")
        
        # _write_multiplot_2d(dmap, colour_base, "(a+b) mod p",
        #                     base_2d_d, p, seed, "DM", append_to_title)
        
        nontriv = np.abs(eigenvalues[1:17])  # 1 to 16 (nontrivial)
        total = nontriv.sum()
        if total > 0:
            diff_var_ratio = (nontriv / total).tolist()
            diff_cum_ratio = np.cumsum(nontriv / total).tolist()
        else:
            diff_var_ratio = [0.0] * 16
            diff_cum_ratio = [0.0] * 16

        base_2d_d_dir = os.path.join(dif_root, "2d", tag)
        make_json(freq_list, diff_var_ratio, diff_cum_ratio, base_2d_d_dir)
        base_3d_d_dir = os.path.join(dif_root, "3d", tag)
        # base_3d_d = os.path.join(base_3d_d_dir,
        #                             f"diff_seed_{seed}{('_'+tag) if tag else ''}_base_3d.pdf")
        
        # _write_multiplot_3d(dmap, colour_base, "(a+b) mod p",
        #                     base_3d_d, p, seed, "DM", append_to_title)
        make_json(freq_list, diff_var_ratio, diff_cum_ratio, base_3d_d_dir)
        if (a_vals is not None) and (b_vals is not None):
            for f in freq_list:
                if f < 0:
                    f_abs = abs(f)
                else:
                    f_abs = f
                if f_abs % p == 0:
                    continue
                if colour_rule is None:
                    raise ValueError("Color rule empty.")
                else:
                    colour, caption, p_cbar, colorscale = colour_rule(
                        a_vals, b_vals, p, f_abs, tag_q
                    )
                name_stub = f"diff_seed_{seed}_freq_{f}.pdf"
                _write_multiplot_2d(
                    dmap,
                    colour,
                    caption,
                    os.path.join(dif_root, "2d", tag, tag_q, name_stub.replace(".pdf", "_2d.pdf")),
                    p, p_cbar, colorscale, seed, "DM", append_to_title
                )
                _write_multiplot_3d(
                    dmap,
                    colour,
                    caption,
                    os.path.join(dif_root, "3d", tag, tag_q, name_stub.replace(".pdf", "_3d.pdf")),
                    p, p_cbar, colorscale, seed, "DM", append_to_title,f,mult=mult
                )
                if colour_rule in (colour_quad_mod_g, colour_quad_mul_f, colour_quad_a_only, colour_quad_b_only) and tag_q == "full":
                    (dmap_orange,  col_orange,  h, scale_orange), \
                    (dmap_viridis, col_viridis, h2, scale_viridis) = _split_dualscale_views(dmap, colour, p_cbar)

                    # Orange（TL/BR）
                    _write_multiplot_2d(
                        dmap_orange, col_orange, f"{caption} - TL/BR (Orange only)",
                        os.path.join(dif_root, "2d", tag, tag_q, name_stub.replace(".pdf", "_2d_orange.pdf")),
                        p, h, scale_orange, seed, "DM", append_to_title
                    )
                    _write_multiplot_3d(
                        dmap_orange, col_orange, f"{caption} - TL/BR (Orange only)",
                        os.path.join(dif_root, "3d", tag, tag_q, name_stub.replace(".pdf", "_3d_orange.pdf")),
                        p, h, scale_orange, seed, "DM", append_to_title, f=f, mult=False
                    )

                    # Viridis（BL/TR）
                    _write_multiplot_2d(
                        dmap_viridis, col_viridis, f"{caption} - BL/TR (Viridis only)",
                        os.path.join(dif_root, "2d", tag, tag_q, name_stub.replace(".pdf", "_2d_viridis.pdf")),
                        p, h2, scale_viridis, seed, "DM", append_to_title
                    )
                    _write_multiplot_3d(
                        dmap_viridis, col_viridis, f"{caption} - BL/TR (Viridis only)",
                        os.path.join(dif_root, "3d", tag, tag_q, name_stub.replace(".pdf", "_3d_viridis.pdf")),
                        p, h2, scale_viridis, seed, "DM", append_to_title, f=f, mult=False
                    )
        
        save_homology_artifacts(
            dmap,
            root_dir=dif_root,
            tag=tag_q,
            seed=seed,
            label=f"Dif--{class_string}",
            num_dims=num_principal_components)

        print("✔️  All PCA / diffusion PDF plots written.")

        bundle_dir = os.path.join(save_dir, "bundles", tag)
        dump_embedding_bundle_json(
            bundle_dir,
            seed=seed, p=p,
            tag=tag,
            tag_q=tag_q,
            class_string=class_string,
            freq_list=freq_list,
            colour_rule_name=_rule_obj_to_name(colour_rule),
            pcs=pcs, pca_var_ratio=pca_var_ratio,
            dmap=dmap, diff_eigvals=eigenvalues,
            a_vals=(a_vals if a_vals is not None else []),
            b_vals=(b_vals if b_vals is not None else []),
            store_colour_vectors=False
        )

        print("PAC/Diffusion json written.")
        if len(freq_list) == 1 and (mat.shape[0] == p ** 2):
            # make a 2d scatterplot lattice of the phases and a vector plot of them
            _make_single_freq_phase_plots(mat, p, freq_list[0], save_dir,
                                    seed=seed, tag=tag)

        
def generate_pdf_plots_for_matrix(
        mat: np.ndarray,
        p: int,
        save_dir: str,
        *,
        seed: int | str = "",
        freq_list: list[int] | None = None,
        tag: str = "",
        tag_q: str = "",
        class_string: str = "",
        colour_rule=None,
        num_principal_components: int = 2,
        do_transposed: bool = False
) -> None:
    """
    Run all PCA / diffusion / homology plots for `mat`, and—if
    `do_transposed` is True—repeat on the transposed matrix.

    • The second run writes into exactly the same directory tree but with
      “_transposed” appended to every sub-folder via the `tag` argument.
    """
    # ---- first pass: original matrix ------------------------------------
    _generate_pdf_plots_for_matrix(
        mat, p, save_dir,
        seed=seed,
        freq_list=freq_list,
        tag=tag,
        tag_q=tag_q,
        class_string=class_string,
        colour_rule=colour_rule,
        num_principal_components=num_principal_components
    )

    # ---- optional second pass: transposed matrix ------------------------
    if do_transposed:
        new_tag = f"{tag}_transposed" if tag else "transposed"
        _generate_pdf_plots_for_matrix(
            mat.T, p, save_dir,
            seed=seed,
            freq_list=freq_list,
            tag=new_tag,
            class_string=class_string,
            colour_rule=colour_rule,
            num_principal_components=num_principal_components
        )

# def _generate_pdf_plots_for_matrix_gcd(mat: np.ndarray,
#                                   p: int,
#                                   save_dir: str,
#                                   *,
#                                   seed: int | str = "",
#                                   freq_list: list[int] | None = None,
#                                   tag: str = "",
#                                   class_string: str = "",
#                                   colour_rule=my_colour_rule,
#                                   num_principal_components=2) -> None:
#     """
#     Create the same PCA & diffusion-map PDF plots as the research notebook,
#     **without** any persistent-homology calls.

#     Parameters
#     ----------
#     mat : np.ndarray
#         Data matrix whose rows are the points to embed.
#     p : int
#         Alphabet size for colour-coding.
#     save_dir : str
#         Root folder into which PDFs are written.
#     seed : int | str, optional
#         Identifying seed for titles / file names.
#     freq_list : list[int] | None, optional
#         Frequency multipliers for the extra colourings
#         (pass whatever you used in `final_grouping`).
#     tag : str, optional
#         Extra string inserted in file names (e.g. "embeds").
#     """
#     n_samples, n_features = mat.shape
#     num_components = min(n_samples, n_features, 8)

#     if num_components >= 4:
#         append_to_title = f"{tag} & {class_string}"
#         freq_list = sorted(freq_list or [])
#         os.makedirs(save_dir, exist_ok=True)
        

#         # ─── Colour bases (same as example) ───────────────────────────────────
#         if tag == "full":
#             side = 2 * p                       # 总边长
#             indices = np.arange(side**2)
#             a_vals  = indices // side          # 全局行 0…2p-1
#             b_vals  = indices %  side          # 全局列 0…2p-1
#         else:
#             n_points = p**2
#             indices = np.arange(n_points)
#             a_vals = indices // p
#             b_vals = indices % p
#         assert len(a_vals) == n_samples, (
#             f"Colour vector ({len(a_vals)}) and matrix rows ({n_samples}) mismatch; "
#             "check p or tag settings."
#         )
#         coords_ab = np.stack([a_vals, b_vals], axis=1)
#         if colour_rule is None:
#             colour_base = (a_vals + b_vals) % p
#         else:
#             colour_base = colour_rule(a_vals, b_vals, p, tag) % p

#         # ─── Directory tree ──────────────────────────────────────────────────
#         pca_root = os.path.join(save_dir, "pca_pdf_plots")
#         dif_root = os.path.join(save_dir, "diffusion_pdf_plots")
#         for root in (pca_root, dif_root):
#             for sub in ("2d", "3d"):
#                 os.makedirs(os.path.join(root, sub, tag), exist_ok=True)
#         print("computing PCA")
#         # ─── PCA (4 comps) ───────────────────────────────────────────────────
#         pcs, pca = compute_pca_coords(mat, num_components=num_components)

#         base_2d_dir = os.path.join(pca_root, "2d", tag)
#         # base_2d = os.path.join(
#         #     base_2d_dir,
#         #     f"pca_seed_{seed}{('_'+tag) if tag else ''}_base.pdf"
#         # )
#         # _write_multiplot_2d(pcs, colour_base, "(a+b) mod p",
#         #                     base_2d, p, seed, "PC", append_to_title)
#         var_ratio = pca.explained_variance_ratio_.tolist()
#         cum_ratio = np.cumsum(pca.explained_variance_ratio_).tolist()
#         make_json(freq_list, var_ratio, cum_ratio, base_2d_dir)
        
#         base_3d_dir = os.path.join(pca_root, "3d", tag)
#         # base_3d = os.path.join(
#         #     base_3d_dir,
#         #     f"pca_seed_{seed}{('_'+tag) if tag else ''}_base_3d.pdf"
#         # )
#         # _write_multiplot_3d(pcs, colour_base, "(a+b) mod p",
#         #                     base_3d, p, seed, "PC", append_to_title)
#         make_json(freq_list, var_ratio, cum_ratio, base_3d_dir)
        
#         # Extra plots per-frequency
#         for f in freq_list:
#             if f % p == 0:
#                 continue
#             if colour_rule is None:
#                 colour_f = (f * (a_vals + b_vals)) % p
#             else:
#                 p_eff = p // math.gcd(p, f)
#                 colour_f = colour_rule(a_vals, b_vals, p, tag) % p_eff
#             name_stub = f"pca_seed_{seed}_freq_{f}.pdf"
#             _write_multiplot_2d(
#                 pcs,
#                 colour_f,
#                 f"(a+b) mod {p_eff}",
#                 os.path.join(pca_root, "2d", tag, name_stub.replace(".pdf", "_2d.pdf")),
#                 p,p_eff, seed, "PC", append_to_title
#             )
#             _write_multiplot_3d(
#                 pcs,
#                 colour_f,
#                 f"(a+b) mod {p_eff}",
#                 os.path.join(pca_root, "3d", tag, name_stub.replace(".pdf", "_3d.pdf")),
#                 p,p_eff, seed, "PC", append_to_title
#             )

#         save_homology_artifacts(
#             pcs,
#             root_dir=pca_root,
#             tag=tag,
#             seed=seed,
#             label=f"PCA--{class_string}",
#             num_dims=num_principal_components)

#         # ─── Diffusion (first 4) ─────────────────────────────────────────────
#         dmap, eigenvalues = compute_diffusion_coords(mat, num_coords=num_components)

#         base_2d_d_dir = os.path.join(dif_root, "2d", tag)
#         # base_2d_d = os.path.join(base_2d_d_dir,
#         #                          f"diff_seed_{seed}{('_'+tag) if tag else ''}_base.pdf")
        
#         # _write_multiplot_2d(dmap, colour_base, "(a+b) mod p",
#         #                     base_2d_d, p, seed, "DM", append_to_title)
#         make_json(freq_list, var_ratio, cum_ratio, base_2d_d_dir)
        
#         nontriv = np.abs(eigenvalues[1:17])  # 1 to 16 (nontrivial)
#         total = nontriv.sum()
#         if total > 0:
#             var_ratio = (nontriv / total).tolist()
#             cum_ratio = np.cumsum(nontriv / total).tolist()
#         else:
#             var_ratio = [0.0] * 16
#             cum_ratio = [0.0] * 16

#         base_3d_d_dir = os.path.join(dif_root, "3d", tag)
#         # base_3d_d = os.path.join(base_3d_d_dir,
#         #                             f"diff_seed_{seed}{('_'+tag) if tag else ''}_base_3d.pdf")
        
#         # _write_multiplot_3d(dmap, colour_base, "(a+b) mod p",
#         #                     base_3d_d, p, seed, "DM", append_to_title)
#         make_json(freq_list, var_ratio, cum_ratio, base_3d_d_dir)
#         for f in freq_list:
#             if f % p == 0:
#                 continue
#             if colour_rule is None:
#                 colour_f = (f * (a_vals + b_vals)) % p
#             else:
#                 p_eff = p // math.gcd(p, f)
#                 colour_f = colour_rule(a_vals, b_vals, p, tag) % p_eff
#             name_stub = f"diff_seed_{seed}_freq_{f}.pdf"
#             _write_multiplot_2d(
#                 dmap,
#                 colour_f,
#                 f"(a+b) mod {p_eff}",
#                 os.path.join(dif_root, "2d", tag, name_stub.replace(".pdf", "_2d.pdf")),
#                 p, p_eff, seed, "DM", append_to_title
#             )
#             _write_multiplot_3d(
#                 dmap,
#                 colour_f,
#                 f"(a+b) mod {p_eff}",
#                 os.path.join(dif_root, "3d", tag, name_stub.replace(".pdf", "_3d.pdf")),
#                 p, p_eff, seed, "DM", append_to_title
#             )
        
#         save_homology_artifacts(
#             dmap,
#             root_dir=dif_root,
#             tag=tag,
#             seed=seed,
#             label=f"Dif--{class_string}",
#             num_dims=num_principal_components)

#         print("✔️  All PCA / diffusion PDF plots written.")

#         if len(freq_list) == 1 and (mat.shape[0] == p ** 2):
#             # make a 2d scatterplot lattice of the phases and a vector plot of them
#             _make_single_freq_phase_plots(mat, p, freq_list[0], save_dir,
#                                     seed=seed, tag=tag)

# def generate_pdf_plots_for_matrix_gcd(
#         mat: np.ndarray,
#         p: int,
#         save_dir: str,
#         *,
#         seed: int | str = "",
#         freq_list: list[int] | None = None,
#         tag: str = "",
#         class_string: str = "",
#         colour_rule=None,
#         num_principal_components: int = 2,
#         do_transposed: bool = False
# ) -> None:
#     """
#     Run all PCA / diffusion / homology plots for `mat`, and—if
#     `do_transposed` is True—repeat on the transposed matrix.

#     • The second run writes into exactly the same directory tree but with
#       “_transposed” appended to every sub-folder via the `tag` argument.
#     """
#     # ---- first pass: original matrix ------------------------------------
#     _generate_pdf_plots_for_matrix_gcd(
#         mat, p, save_dir,
#         seed=seed,
#         freq_list=freq_list,
#         tag=tag,
#         class_string=class_string,
#         colour_rule=my_colour_rule,
#         num_principal_components=num_principal_components
#     )

#     # ---- optional second pass: transposed matrix ------------------------
#     if do_transposed:
#         new_tag = f"{tag}_transposed" if tag else "transposed"
#         _generate_pdf_plots_for_matrix(
#             mat.T, p, save_dir,
#             seed=seed,
#             freq_list=freq_list,
#             tag=new_tag,
#             class_string=class_string,
#             colour_rule=my_colour_rule,
#             num_principal_components=num_principal_components
#         )           
def generate_pca_information_scaling_experiment(mat: np.ndarray,
                                                p: int,
                                                save_dir: str,
                                                *,
                                                seed: int | str = "",
                                                freq_list: list[int] | None = None,
                                                tag: str = "") -> None:
    """
    For a given data matrix mat, compute:
      - cumulative PCA variance ratios for components 1–4
      - cumulative diffusion 'variance' ratios (via eigenvalues) for coords 1–4
    and save them as JSON.

    Parameters
    ----------
    mat : np.ndarray
        Data matrix whose rows are the points to embed.
    p : int
        Alphabet size (only recorded in JSON for provenance).
    save_dir : str
        Directory into which the JSON file will be written.
    seed : int | str, optional
        Identifier for this run, used in the filename and in the JSON.
    freq_list : list[int] | None, optional
        Ignored here (present only to mirror generate_pdf_plots_for_matrix).
    tag : str, optional
        Extra string inserted in the filename (e.g. "embeds").
    """
    # ensure output dir exists
    os.makedirs(save_dir, exist_ok=True)

    # --- PCA part (up to 4 components) ---
    X = _sanitize_matrix(mat)
    n, d = X.shape
    n_comp = min(4, d, max(1, n - 1))
    coords, pca = _safe_pca_coords(X, n_comp)
    var_ratio = getattr(pca, "explained_variance_ratio_", np.zeros(n_comp))
    cum_var_ratio = np.cumsum(var_ratio).tolist()

    # Diffusion part (robust ε)
    d2 = squareform(pdist(X, metric="euclidean")) ** 2
    eps = float(np.median(d2))
    if not np.isfinite(eps) or eps <= 0:
        pos = d2[d2 > 0]
        eps = float(pos.mean()) if pos.size else 1e-12
    A = np.exp(-d2 / eps)
    M = A / A.sum(axis=1, keepdims=True)

    eigvals, _ = eigh(M)
    eigvals = eigvals[::-1]
    nontrivial = eigvals[1:1 + n_comp]
    total = float(np.sum(nontrivial))
    diff_ratios = (nontrivial / total) if total > 0 else np.zeros_like(nontrivial)
    cum_diff_ratio = np.cumsum(diff_ratios).tolist()

    info = {
        "seed": seed,
        "p": p,
        "num_pca_components": int(len(var_ratio)),
        "cumulative_pca_variance_ratio": cum_var_ratio,
        "num_diffusion_components": int(len(nontrivial)),
        "cumulative_diffusion_eigenvalue_ratio": cum_diff_ratio,
    }
    fname = f"pca_info_seed_{seed}" + (f"_{tag}" if tag else "") + ".json"
    with open(os.path.join(save_dir, fname), "w") as f:
        json.dump(info, f, indent=4)
    print(f"✔️  PCA & diffusion scaling info saved to {os.path.join(save_dir, fname)}")

# =========================
# JSON 打包 & 重建
# =========================
def _rule_obj_to_name(rule_fn) -> str:
    if rule_fn is None:
        return "none"
    mapping = {
        "colour_quad_mul_f": "mul_f",
        "colour_quad_mod_g": "mod_g",
        "colour_quad_a_only": "a_only",
        "colour_quad_b_only": "b_only",
        "colour_quad_mod_g_no_fb":"mod_g_no_fb"
    }
    name = getattr(rule_fn, "__name__", "")
    return mapping.get(name, name or "custom")

def _rule_name_to_obj(name: str):
    if name in (None, "", "none"):
        return None
    mapping = {
        "mul_f": colour_quad_mul_f,
        "mod_g": colour_quad_mod_g,
        "mod_g_no_fb": colour_quad_mod_g_no_fb,
        "a_only": colour_quad_a_only,
        "b_only": colour_quad_b_only,
        "colour_quad_mul_f": colour_quad_mul_f,
        "colour_quad_mod_g": colour_quad_mod_g,
        "colour_quad_a_only": colour_quad_a_only,
        "colour_quad_b_only": colour_quad_b_only,
        "colour_quad_mod_g_no_fb":colour_quad_mod_g_no_fb,
    }
    return mapping.get(name, None)


# =========================
# JSON bundling & reconstruction utilities
# =========================
def _to_list(x):
    if x is None:
        return None
    if isinstance(x, list):
        return x
    import numpy as _np
    if isinstance(x, _np.ndarray):
        return x.tolist()
    return list(x)

def dump_embedding_bundle_json(
    bundle_dir: str,
    *,
    seed,
    p: int,
    tag: str,
    class_string: str,
    freq_list: list[int],
    colour_rule_name: str,
    pcs: np.ndarray,
    pca_var_ratio: list[float],
    dmap: np.ndarray,
    diff_eigvals: np.ndarray,
    a_vals: list[int] | np.ndarray,
    b_vals: list[int] | np.ndarray,
    store_colour_vectors: bool = False,
    tag_q: str | None = None,
) -> str:
    """
    将用于重建的全部关键信息打包为一个 JSON 文件并写入 bundle_dir。
    返回写入的 json 路径。
    """
    Path(bundle_dir).mkdir(parents=True, exist_ok=True)

    N = int(pcs.shape[0])
    meta = {
        "version": 1,
        "seed": seed,
        "p": int(p),
        "tag": tag,
        "tag_q": tag_q if tag_q is not None else "",
        "class_string": class_string,
        "num_points": N,
        "pca_dims": int(pcs.shape[1]),
        "diff_dims": int(dmap.shape[1]),
        "freq_list": [int(f) for f in (freq_list or [])],
        "colour_rule_name": colour_rule_name,
        
        "plot_defaults": {
            "FONT_SIZE": FONT_SIZE,
            "CBAR_TICK_SIZE": CBAR_TICK_SIZE,
            "CBAR_TITLE_SIZE": CBAR_TITLE_SIZE,
            "TICK_SIZE": TICK_SIZE,
            "LEGEND_POS": LEGEND_POS,
        },
    }

    payload = {
        "meta": meta,
        "a_vals": _to_list(a_vals) if a_vals is not None else [],
        "b_vals": _to_list(b_vals) if b_vals is not None else [],
        "pca": {
            "coords": _to_list(pcs),
            "explained_variance_ratio": _to_list(pca_var_ratio),
            "cumulative_variance_ratio": _to_list(np.cumsum(pca_var_ratio)),
        },
        "diffusion": {
            "coords": _to_list(dmap),
            "eigenvalues": _to_list(diff_eigvals),
        },
    }

    # save the color vector for every f in JSON if needed
    if store_colour_vectors:
        colours = {}
        rule_fn = _rule_name_to_obj(colour_rule_name)
        if (rule_fn is not None) and len(payload["a_vals"]) == N and len(payload["b_vals"]) == N:
            A = np.asarray(payload["a_vals"])
            B = np.asarray(payload["b_vals"])
            for f in meta["freq_list"]:
                f_abs = abs(int(f))
                if f_abs % p == 0:
                    continue
                col, caption, p_cbar, colorscale = rule_fn(A, B, p, f_abs, meta["tag_q"])
                colours[str(f)] = {
                    "values": _to_list(col),
                    "caption": caption,
                    "p_cbar": int(p_cbar),
                    "colorscale": colorscale,
                }
        payload["precomputed_colours"] = colours

    fname = f"bundle_seed_{seed}"
    if tag:
        fname += f"_{tag}"
    out_path = str(Path(bundle_dir) / (fname + ".json"))
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return out_path


def load_embedding_bundle_json(path_or_obj):
    """Load bundle json and normalize to dict, try reconstruct a,b according to N、p、tag_q if a,b info is missing."""
    if isinstance(path_or_obj, (str, Path)):
        with open(path_or_obj, "r") as fh:
            bundle = json.load(fh)
    else:
        bundle = path_or_obj

    PCS  = np.asarray(bundle["pca"]["coords"])
    DMAP = np.asarray(bundle["diffusion"]["coords"])
    meta = bundle.get("meta", {})
    p    = int(meta.get("p", 0))
    tag_q= meta.get("tag_q", "") or ""
    N    = int(PCS.shape[0])

    A = np.asarray(bundle.get("a_vals", [])) if bundle.get("a_vals") else None
    B = np.asarray(bundle.get("b_vals", [])) if bundle.get("b_vals") else None

    # if a,b is missing and could be recontruct according to N、p、tag_q, run the recontrunction
    if (A is None or B is None) and p > 0 and N > 0:
        if N == p * p:
            idx = np.arange(N)
            A = idx // p
            B = idx %  p
        elif tag_q == "full" and N == (2 * p) * (2 * p):
            side = 2 * p
            idx  = np.arange(N)
            A = idx // side
            B = idx %  side
        # otherwise keep as None (later will degrade to 0/0)

    bundle["_PCS"] = PCS
    bundle["_DMAP"] = DMAP
    bundle["_A"] = A
    bundle["_B"] = B
    return bundle



def _get_colour_for_f(bundle: dict, f: int):
    """
    给定 bundle 和某个 f，返回 (colour, caption, p_cbar, colorscale)。
    先查 precomputed，否则用规则函数即时计算。
    """
    meta = bundle["meta"]
    p = int(meta["p"])
    tag_q = meta.get("tag_q", "")
    A = bundle.get("_A", None)
    B = bundle.get("_B", None)

    # 没有 a,b 就给个默认颜色（全 0）
    if A is None or B is None:
        N = bundle["_PCS"].shape[0]
        return (np.zeros(N, dtype=int), "index", p, "Viridis")

    # 预计算缓存
    pre = bundle.get("precomputed_colours", {})
    if pre and str(f) in pre:
        cobj = pre[str(f)]
        return (np.asarray(cobj["values"]),
                cobj.get("caption", ""),
                int(cobj.get("p_cbar", p)),
                cobj.get("colorscale", "Viridis"))

    # 动态计算
    rule_fn = _rule_name_to_obj(meta.get("colour_rule_name"))
    if rule_fn is None:
        N = bundle["_PCS"].shape[0]
        return (np.zeros(N, dtype=int), "index", p, "Viridis")

    f_abs = abs(int(f))
    if f_abs % p == 0:
        # 退化，返回常数色（避免 plotly 报错），色条长度仍用 p
        N = bundle["_PCS"].shape[0]
        return (np.zeros(N, dtype=int), f"degenerate f ({f_abs} mod {p} == 0)", p, "Viridis")

    col, caption, p_cbar, colorscale = rule_fn(A, B, p, f_abs, tag_q)
    return (np.asarray(col), caption, int(p_cbar), colorscale)


def _make_multiplot_3d_figure_html_only(
    coords: np.ndarray,
    colour: np.ndarray,
    caption: str,
    p: int,
    p_cbar: int,
    colorscale: str,
    *,
    seed,
    label: str,
    title_tag: str,
    f: int,
    mult: bool,
    a_vals: np.ndarray | None,
    b_vals: np.ndarray | None,
    tag_q: str,
) -> go.Figure:
    """
    复用 _write_multiplot_3d 的结构，但只构建 Figure（不写 PDF），方便 HTML 重建。
    """
    n_pts = coords.shape[0]
    side = int(math.isqrt(n_pts)) if n_pts > 0 else 0
    g = p // math.gcd(p, abs(int(f))) or p
    multi_view = (g != p) and (side == 2 * p) and mult

    triplets = list(itertools.combinations(range(4), 3))
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}] * 2] * 2,
        subplot_titles=[f"{label}{i} vs {label}{j} vs {label}{k}" for i, j, k in triplets],
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
    )

    if a_vals is None or b_vals is None:
        a_vals = np.zeros(n_pts, dtype=int)
        b_vals = np.zeros(n_pts, dtype=int)

    hover_kw = _make_hover(a_vals, b_vals)

    if not multi_view:
        # 单视图：直接画颜色 = colour
        for s_idx, (i, j, k) in enumerate(triplets, 1):
            row, col = (1, s_idx) if s_idx <= 2 else (2, s_idx - 2)
            step = max(1, p_cbar // 10)
            ticks = list(range(0, p_cbar, step))
            if ticks[-1] != p_cbar - 1:
                ticks.append(p_cbar - 1)

            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, i],
                    y=coords[:, j],
                    z=coords[:, k],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=colour,
                        colorscale=colorscale,
                        cmin=0,
                        cmax=p_cbar - 1,
                        showscale=(s_idx == 1),
                        colorbar=dict(
                            title=dict(text=caption, side="right", font=dict(size=CBAR_TITLE_SIZE)),
                            tickfont=dict(size=CBAR_TICK_SIZE),
                            len=0.90,
                        ),
                    ),
                    **hover_kw,
                    name="",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            sid = f"scene{s_idx if s_idx > 1 else ''}"
            fig.layout[sid].xaxis.title.text = f"{label}{i}"
            fig.layout[sid].yaxis.title.text = f"{label}{j}"
            fig.layout[sid].zaxis.title.text = f"{label}{k}"
    else:
        # 多视图：与 _write_multiplot_3d 的按钮逻辑一致（a/b/c 三套）
        A = a_vals
        B = b_vals
        col_a, _, _, cs_a = colour_quad_a_only(A, B, p, f, "full")
        col_b, _, _, cs_b = colour_quad_b_only(A, B, p, f, "full")
        # col_c, _, pcbar_c, cs_c = colour_quad_mod_g(A, B, p, f, "full")
        col_c, _, pcbar_c, cs_c = colour_quad_mod_g_no_fb(A, B, p, f, "full")


        h_pairs = lines_a_mod_g(A, B, p, g)
        v_pairs = lines_b_mod_g(A, B, p, g)
        #c_pairs = lines_c_mod_g(A, B, p, g)
        c_pairs = lines_a_mod_g(A, B, p, g)

        n_h, n_v, n_c = len(h_pairs), len(v_pairs), len(c_pairs)

        legend_shown_a, legend_shown_b, legend_shown_c = set(), set(), set()
        for s_idx, (i, j, k) in enumerate(triplets, 1):
            row, col = (1, s_idx) if s_idx <= 2 else (2, s_idx - 2)

            # a-view
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, i],
                    y=coords[:, j],
                    z=coords[:, k],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=col_a,
                        colorscale=cs_a,
                        cmin=0,
                        cmax=2 * g - 1,
                        showscale=(s_idx == 1),
                        colorbar=dict(
                            title=dict(text=caption, side="right", font=dict(size=CBAR_TITLE_SIZE)),
                            tickfont=dict(size=CBAR_TICK_SIZE),
                            len=0.90,
                        ),
                    ),
                    name="a mod g",
                    legendgroup="a",
                    visible=True,
                    **hover_kw,
                ),
                row=row,
                col=col,
            )

            # b-view
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, i],
                    y=coords[:, j],
                    z=coords[:, k],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=col_b,
                        colorscale=cs_b,
                        cmin=0,
                        cmax=2 * g - 1,
                        showscale=(s_idx == 1),
                        colorbar=dict(
                            title=dict(text=caption, side="right", font=dict(size=CBAR_TITLE_SIZE)),
                            tickfont=dict(size=CBAR_TICK_SIZE),
                            len=0.90,
                        ),
                    ),
                    name="b mod g",
                    legendgroup="b",
                    visible=False,
                    **hover_kw,
                ),
                row=row,
                col=col,
            )

            # c-view
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, i],
                    y=coords[:, j],
                    z=coords[:, k],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=col_c,
                        colorscale=cs_c,
                        cmin=0,
                        cmax=pcbar_c - 1,
                        showscale=(s_idx == 1),
                        colorbar=dict(
                            title=dict(text=caption, side="right", font=dict(size=CBAR_TITLE_SIZE)),
                            tickfont=dict(size=CBAR_TICK_SIZE),
                            len=0.90,
                        ),
                    ),
                    name="c mod g",
                    legendgroup="c",
                    showlegend=(s_idx == 1),
                    visible=False,
                    **hover_kw,
                ),
                row=row,
                col=col,
            )

            # a-lines
            for idx_arr, dash, color, gid in h_pairs:
                idx_sorted = idx_arr[np.argsort(A[idx_arr])]
                idx_plot = np.concatenate([idx_sorted, idx_sorted[:1]]) if idx_sorted.size > 2 else idx_sorted
                show_legend = gid not in legend_shown_a
                legend_shown_a.add(gid)
                fig.add_trace(
                    go.Scatter3d(
                        x=coords[idx_plot, i],
                        y=coords[idx_plot, j],
                        z=coords[idx_plot, k],
                        mode="lines",
                        name=gid,
                        legendgroup=gid,
                        showlegend=show_legend,
                        line=dict(color=color, dash=dash, width=1.2),
                        hoverinfo="skip",
                        visible=True,
                    ),
                    row=row,
                    col=col,
                )

            # b-lines
            for idx_arr, dash, color, gid in v_pairs:
                idx_sorted = idx_arr[np.argsort(B[idx_arr])]
                idx_plot = np.concatenate([idx_sorted, idx_sorted[:1]]) if idx_sorted.size > 2 else idx_sorted
                show_legend = gid not in legend_shown_b
                legend_shown_b.add(gid)
                fig.add_trace(
                    go.Scatter3d(
                        x=coords[idx_plot, i],
                        y=coords[idx_plot, j],
                        z=coords[idx_plot, k],
                        mode="lines",
                        name=gid,
                        legendgroup=gid,
                        showlegend=show_legend,
                        line=dict(color=color, dash=dash, width=1.2),
                        hoverinfo="skip",
                        visible=False,
                    ),
                    row=row,
                    col=col,
                )

            # c-lines
            for idx_arr, dash, color, gid in c_pairs:
                a_sub = A[idx_arr]
                b_sub = B[idx_arr]
                order = np.lexsort((b_sub, a_sub))
                idx_sorted = idx_arr[order]
                idx_plot = np.concatenate([idx_sorted, idx_sorted[:1]]) if idx_sorted.size > 2 else idx_sorted
                show_legend = gid not in legend_shown_c
                legend_shown_c.add(gid)
                fig.add_trace(
                    go.Scatter3d(
                        x=coords[idx_plot, i],
                        y=coords[idx_plot, j],
                        z=coords[idx_plot, k],
                        mode="lines",
                        name=gid,
                        legendgroup=gid,
                        showlegend=show_legend,
                        line=dict(color=color, dash=dash, width=1.2),
                        hoverinfo="skip",
                        visible=False,
                    ),
                    row=row,
                    col=col,
                )

            sid = f"scene{s_idx if s_idx > 1 else ''}"
            fig.layout[sid].xaxis.title.text = f"{label}{i}"
            fig.layout[sid].yaxis.title.text = f"{label}{j}"
            fig.layout[sid].zaxis.title.text = f"{label}{k}"

        # 三个视图的可见性切换按钮
        vis_a, vis_b, vis_c = [], [], []
        for _ in range(len(triplets)):
            vis_a += [True, False, False] + [True] * n_h + [False] * n_v + [False] * n_c
            vis_b += [False, True, False] + [False] * n_h + [True] * n_v + [False] * n_c
            vis_c += [False, False, True] + [False] * n_h + [False] * n_v + [True] * n_c

        fig.update_layout(
            updatemenus=[dict(
                buttons=[
                    dict(label="a mod g", method="update",
                         args=[{"visible": vis_a}, {"title": {"text": "colour = a mod g"}}]),
                    dict(label="b mod g", method="update",
                         args=[{"visible": vis_b}, {"title": {"text": "colour = b mod g"}}]),
                    dict(label="c mod g", method="update",
                         args=[{"visible": vis_c}, {"title": {"text": "colour = c mod g"}}]),
                ],
                direction="down",
                x=0.99, y=1.05, xanchor="left",
                pad={"t": 0, "r": 6},
            )]
        )
        fig.update_layout(legend=LEGEND_POS)

    # 统一 3D 轴/字号
    for layout_key in fig.layout:
        if str(layout_key).startswith("scene"):
            scene = fig.layout[layout_key]
            scene.xaxis.title.font = dict(size=FONT_SIZE)
            scene.yaxis.title.font = dict(size=FONT_SIZE)
            scene.zaxis.title.font = dict(size=FONT_SIZE)
            scene.xaxis.tickfont = dict(size=TICK_SIZE)
            scene.yaxis.tickfont = dict(size=TICK_SIZE)
            scene.zaxis.tickfont = dict(size=TICK_SIZE)

    fig.update_layout(
        width=1600,
        height=1200,
        title=f"{label} 3-D (first 4) - seed {seed} - {title_tag}",
        margin=dict(l=40, r=40, t=80, b=40),
        font=dict(size=FONT_SIZE),
        showlegend=True,
    )
    return fig


def rebuild_embedding_html_from_bundle(
    bundle_json_path: str | dict,
    *,
    kind: str = "pca",          # "pca" | "diffusion"
    f: int | None = None,       # 选择某个频率来着色；None 则给默认颜色
    out_html: str | None = None # 若提供路径则写 HTML，否则仅返回 fig
) -> go.Figure:
    """
    从 JSON bundle 重建一个与 _write_multiplot_3d 相同布局的交互式 3D 图（只写 HTML，不写 PDF）。
    """
    bundle = load_embedding_bundle_json(bundle_json_path)
    meta = bundle["meta"]
    p = int(meta["p"])
    tag_q = meta.get("tag_q", "")
    seed = meta.get("seed", "")
    class_string = meta.get("class_string", "")
    label = "PC" if kind == "pca" else "DM"
    coords = bundle["_PCS"] if kind == "pca" else bundle["_DMAP"]

    # 颜色
    if f is None:
        N = coords.shape[0]
        colour = np.zeros(N, dtype=int)
        caption = "(index)"
        p_cbar = p
        colorscale = "Viridis"
        mult = False
    else:
        colour, caption, p_cbar, colorscale = _get_colour_for_f(bundle, f)
        # 与生成时一致的开关：a_only/b_only/mul_f/mod_g 这些允许 multi-view
        rule_name = meta.get("colour_rule_name", "")
        mult = rule_name in ("a_only", "b_only", "mul_f", "mod_g", "mod_g_no_fb",
                             "colour_quad_a_only", "colour_quad_b_only",
                             "colour_quad_mul_f", "colour_quad_mod_g","colour_quad_mod_g_no_fb")

    fig = _make_multiplot_3d_figure_html_only(
        coords=coords,
        colour=colour,
        caption=caption,
        p=p,
        p_cbar=p_cbar,
        colorscale=colorscale,
        seed=seed,
        label=label,
        title_tag=f"{meta.get('tag','')} & {class_string}",
        f=(f if f is not None else 1),
        mult=mult,
        a_vals=bundle.get("_A", None),
        b_vals=bundle.get("_B", None),
        tag_q=tag_q,
    )

    if out_html:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html, include_plotlyjs="cdn")
    return fig
