import os
import glob
import json
import collections
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
from itertools import combinations
import numpy as np
rng = np.random.default_rng()
import math
from math import gcd

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

# ---------------------------------------------
# 2)  MMD² with an RBF kernel, JAX version
# ---------------------------------------------

# ---------------------------------------------
# Helpers: turn a histogram into i.i.d. samples
# ---------------------------------------------
import numpy as np
from numpy.random import default_rng

def draw_samples(dist_dict, *, n_samples=10_000, seed=0):
    """
    Return an (n_samples, 2) float32 array drawn *with replacement*
    from the 2-D histogram `dist_dict` (maps "a,b" -> count).

    Sampling instead of full expansion keeps memory bounded and
    preserves exchangeability for the permutation test.
    """
    rng = default_rng(seed)
    coords = np.array([tuple(map(int, k.split(','))) for k in dist_dict],
                      dtype=np.float32)
    counts = np.array(list(dist_dict.values()), dtype=np.float64)
    p = counts / counts.sum()
    idx = rng.choice(len(coords), size=n_samples, replace=True, p=p)
    return coords[idx]


# ---------------------------------------------
# Unbiased Gaussian-kernel MMD in JAX
# ---------------------------------------------
import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def _sq_dists(A, B):
    return jnp.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)

@partial(jax.jit, static_argnames='sigma')
def mmd2_u_gaussian(X, Y, *, sigma):
    """Unbiased MMD² with an RBF kernel of bandwidth `sigma`."""
    σ2 = sigma ** 2

    Kxx = jnp.exp(-_sq_dists(X, X) / (2 * σ2))
    Kyy = jnp.exp(-_sq_dists(Y, Y) / (2 * σ2))
    Kxy = jnp.exp(-_sq_dists(X, Y) / (2 * σ2))

    m = X.shape[0]
    n = Y.shape[0]

    term_xx = (jnp.sum(Kxx) - jnp.trace(Kxx)) / (m * (m - 1))
    term_yy = (jnp.sum(Kyy) - jnp.trace(Kyy)) / (n * (n - 1))
    term_xy = jnp.sum(Kxy) / (m * n)

    return term_xx + term_yy - 2.0 * term_xy


# ---------------------------------------------
# Permutation test on *samples* (no weights!)
# ---------------------------------------------
def mmd_permutation_test_samples(
        X, Y,
        *, n_permutations=10_000,
        sigma=None,
        seed=0,
        batch_size=1_000,
):
    """
    Two-sample permutation test for the *unweighted* unbiased MMD².
    Returns (sqrt(MMD²_obs), p-value).
    """
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)
    Z = jnp.vstack([X, Y])
    m, n = len(X), len(Y)
    N    = m + n

    # bandwidth: pooled median heuristic unless caller fixes it
    if sigma is None:
        d2  = _sq_dists(Z, Z)
        σ   = float(jnp.sqrt(0.5 * jnp.median(
                     d2[jnp.triu_indices(N, k=1)])))
    else:
        σ = float(sigma)

    mmd2_obs = mmd2_u_gaussian(X, Y, sigma=σ)
    mmd_obs  = jnp.sqrt(jnp.maximum(mmd2_obs, 0.0))

    key0 = jax.random.PRNGKey(seed)

    @jax.jit
    def _one_perm(key_single):
        perm = jax.random.permutation(key_single, N)
        Xp   = Z[perm[:m]]
        Yp   = Z[perm[m:]]
        return jnp.sqrt(jnp.maximum(
            mmd2_u_gaussian(Xp, Yp, sigma=σ), 0.0))

    @jax.jit
    def _one_batch(carry, batch_keys):
        stats = jax.vmap(_one_perm)(batch_keys)
        exceed = jnp.sum(stats >= mmd_obs)
        return carry + exceed, None

    n_full, tail = divmod(n_permutations, batch_size)
    total = 0

    if n_full:
        keys = jax.random.split(key0, n_full * batch_size)\
                         .reshape(n_full, batch_size, 2)
        total, _ = jax.lax.scan(_one_batch, 0, keys)

    if tail:
        key_tail = jax.random.split(jax.random.fold_in(key0, 9999), tail)
        extra, _ = _one_batch(0, key_tail)
        total += extra

    p_val = float(total) / n_permutations
    return float(mmd_obs), p_val


# ---------------------------------------------
# Convenience wrapper: from histogram → p-value
# ---------------------------------------------
def mmd_test_from_hists(
        dist1, dist2,
        *, n_draws=10_000,
        n_permutations=10_000,
        seed=0,
):
    X = draw_samples(dist1, n_samples=n_draws, seed=seed)
    Y = draw_samples(dist2, n_samples=n_draws, seed=seed + 123)
    return mmd_permutation_test_samples(
        X, Y,
        n_permutations=n_permutations,
        seed=seed
    )


@jit
def _pairwise_sq_dists(A, B):
    # A: (n, d)  B: (m, d)   →   (n, m)
    return jnp.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)



def dict_to_weighted_array(dist_dict):
    """
    dist_dict maps 'a,b' -> count.
    Returns: coords  (N×2 ndarray),
             weights (N-vector, sums to 1.0)
    """
    coords  = []
    weights = []
    total   = sum(dist_dict.values())
    for k, cnt in dist_dict.items():
        a, b = map(int, k.split(','))
        coords.append((a, b))
        weights.append(cnt / total)       # normalise to prob-mass
    return np.asarray(coords, dtype=float), np.asarray(weights)


# ------------------------------------------------------------
# 3)  build weighted arrays for the *max-activation* heatmaps
# ------------------------------------------------------------
titles_row1 = ['MLP vec add', 'Attention 0.0', 'Attention 1.0', 'MLP concat']
# weighted = {t: dict_to_weighted_array(dist_max_all[t]) for t in titles_row1}


def make_heatmap_subtitles(dist_dicts, titles, offset1=22, offset2=40):
    """
    For each (title, dist), compute the avg log(count) strictly between
    b=a+offset1 and b=a+offset2, and return titles like:
      "MLP concatenated\nAvg between 22+x & 40+x: 0.123"
    """
    subtitles = []
    for t, d in zip(titles, dist_dicts):
        stats = compute_avg_log_counts(d, offset1, offset2)
        subtitles.append(
            f"{t}\n: off diagonal avg {stats['between']:.2f}"
        )
    return subtitles


def compute_avg_log_counts(dist, offset1, offset2):
    """
    Given a dict mapping "a,b" → count, build the p×p matrix,
    take log(count) (0→0), and compute three averages:
      1) on b = a + offset1
      2) on b = a + offset2
      3) for offset1 < b − a < offset2
    """
    # figure out matrix size p
    coords = [tuple(map(int, k.split(','))) for k in dist.keys()]
    p = max(max(a, b) for a, b in coords) + 1

    # build and log-transform matrix
    matrix = [[0]*p for _ in range(p)]
    for (a, b), cnt in [(c, dist[f"{c[0]},{c[1]}"]) for c in coords]:
        matrix[a][b] = cnt
    logm = [[math.log(v) if v > 0 else 0 for v in row] for row in matrix]

    # accumulate sums & counts
    sum1 = n1 = sum2 = n2 = sum_between = n_between = 0
    for a in range(p):
        b1 = a + offset1
        b2 = a + offset2
        # line 1
        if 0 <= b1 < p:
            sum1 += logm[a][b1]
            n1   += 1
        # line 2
        if 0 <= b2 < p:
            sum2 += logm[a][b2]
            n2   += 1
        # strictly between
        for b in range(p):
            if offset1 + a < b < offset2 + a:
                sum_between += logm[a][b]
                n_between  += 1

    return {
        'on_22+x':     sum1 / n1        if n1        else float('nan'),
        'on_40+x':     sum2 / n2        if n2        else float('nan'),
        'between':     sum_between / n_between if n_between else float('nan')
    }

# Define your base directories and corresponding titles
base_dirs = [
    '/home/mila/m/moisescg/scratch/neurips_2025_geometry_run-2/quantitative_metrics_multilayer_heatmaps_log59_one_embed_1_k_50/59_distributions_equivariantness/mlp=one_embed_1_p=59_bs=59_k=50_nn=512_wd=0.0001_lr=0.0005',
    '/home/mila/m/moisescg/scratch/neurips_2025_geometry_run-1/quantitative_metrics_multilayer_heatmaps_log59_one_embed_cheating_1_k_50/59_distributions_equivariantness/mlp=one_embed_cheating_1_p=59_bs=59_k=50_nn=512_wd=0.0001_lr=0.0005',
    '/home/mila/m/moisescg/scratch/neurips_2025_geometry_run-1/quantitative_metrics_transformer_1_heatmaps_log59_0.0_k_50/59_distributions_equivariantness/transformer_p=59_bs=59_k=50_dm=128_wd=5e-05_lr=0.0001',
    '/home/mila/m/moisescg/scratch/neurips_2025_geometry_run-1/quantitative_metrics_transformer_1_heatmaps_log59_1.0_k_50/59_distributions_equivariantness/transformer_p=59_bs=59_k=50_dm=128_wd=5e-05_lr=0.0001'
]



titles = [
    'MLP concat',
    'MLP vec add',
    'Attention 0.0',
    'Attention 1.0'
]

# true color scheme by directory index
colors = ['darkblue', 'teal', 'red', 'orange']
# map each title to its color
color_map = dict(zip(titles, colors))

# map original titles → base_dirs
dir_map   = dict(zip(titles, base_dirs))
color_map = dict(zip(titles, colors))

# now pick the new order you want:
titles    = ['MLP vec add', 'Attention 0.0', 'Attention 1.0', 'MLP concat']
base_dirs = [dir_map[t] for t in titles]
colors    = [color_map[t] for t in titles]
color_map = dict(zip(titles, colors))

# Containers for aggregated data
dist_max_all = {}
dist_center_all = {}
dist_phase_all = {}
dist_phase_eq_all = {}

dist_freq_equal_agg = {}
dist_freq_triplet_agg = {}

dist_freq_equal_all = {}
dist_freq_triplet_all = {}

equiv_records = []
margin_records = []
loss_records = []
grad_records = []
dirrel_records = []
cluster_counts_all = {}

# Parameters
MAX_FILES_PER_DIR = 703
FONT_SIZE = 24
per_file_data = {t: [] for t in titles}

# Iterate through each directory
# Process each directory
MAX_FILES = MAX_FILES_PER_DIR
# Process each directory and collect per-file data
for base_dir, title in zip(base_dirs, titles):
    # aggregated counters
    cnt_max = collections.Counter()
    cnt_center = collections.Counter()
    cnt_phase = collections.Counter()
    cnt_phase_eq = collections.Counter()
    cnt_freq = collections.Counter()
    cnt_freq3 = collections.Counter()

    # stats lists
    worst_equiv, equiv_means, equiv_stds = [], [], []
    min_margins, max_margins, avg_margins, std_margins = [], [], [], []
    min_losses, max_losses, avg_losses, std_losses = [], [], [], []
    avg_grads, std_grads = [], []
    avg_dirrels, std_dirrels = [], []
    cluster_counts = []

    all_files = glob.glob(os.path.join(base_dir, 'quantities_*.json'))
    processed = 0

    for fname in all_files:
        if processed >= MAX_FILES:
            break
        with open(fname) as f:
            data = json.load(f)
        try:
            # aggregate for heatmaps
            cnt_max.update(data['distribution_of_max_preactivations'])
            cnt_center.update(data['distribution_of_center_mass'])
            cnt_phase.update(data['distribution_of_phases'])
            cnt_phase_eq.update(data['distribution_of_phases_f_a=f_b'])
            cnt_freq.update(data['frequencies_equal'])
            cnt_freq3.update(data['frequencies_equal_triplets'])

            # stats
            e = data['networks_equivariantness_stats']
            worst_equiv.append(e['max']-e['min'])
            equiv_means.append(e['mean'])
            equiv_stds.append(e['std'])

            m = data['network_margin_stats']
            min_margins.append(m['min_margin'])
            max_margins.append(m['max_margin'])
            avg_margins.append(m['avg_margin'])
            std_margins.append(m['std_dev_margin'])

            l = data['network_loss_stats']
            min_losses.append(l['min_loss'])
            max_losses.append(l['max_loss'])
            avg_losses.append(l['avg_loss'])
            std_losses.append(l['std_dev_loss'])

            avg_grads.append(data['average_gradient_symmetricity'])
            std_grads.append(data['std_dev_gradient_symmetricity'])
            avg_dirrels.append(data['average_distance_irrelevance'])
            std_dirrels.append(data['std_dev_distance_irrelevance'])

            cluster_counts.append(len(data['clusters_equivariantness_stats']))

            # compute per-file neuron count and % harmonics
            total_neurons = sum(data['frequencies_equal'].values())
            harm_neurons = sum(cnt for k, cnt in data['frequencies_equal'].items()
                                if gcd(*map(int, k.split(','))) == min(*map(int, k.split(','))))
            harm_pct = harm_neurons / total_neurons if total_neurons else 0

            # triplet harmonics %
            total3 = sum(data['frequencies_equal_triplets'].values())
            harm3 = 0
            for k, cnt in data['frequencies_equal_triplets'].items():
                a, b, c = sorted(map(int, k.split(',')))
                if gcd(a, b) == gcd(a, c):
                    harm3 += cnt
            harm3_pct = harm3 / total3 if total3 else 0

            # save per-file record
            per_file_data[title].append({
                'total_neurons': total_neurons,
                'harm_pct': harm_pct,
                'harm3_pct': harm3_pct,
                'avg_grad': data['average_gradient_symmetricity'],
                'avg_dirrel': data['average_distance_irrelevance']
            })

            processed += 1
        except KeyError:
            print(KeyError)
            print(fname)
            continue
    print(f"{title}: found {len(all_files)} files, loaded {processed} files")
    # store aggregated for heatmaps
    dist_max_all[title] = dict(cnt_max)
    dist_center_all[title] = dict(cnt_center)
    dist_phase_all[title] = dict(cnt_phase)
    dist_phase_eq_all[title] = dict(cnt_phase_eq)
    dist_freq_equal_agg[title] = dict(cnt_freq)
    dist_freq_triplet_agg[title] = dict(cnt_freq3)

    # store stats
    equiv_records.append({'title':title,'worst':worst_equiv,'mean':equiv_means,'std':equiv_stds})
    margin_records.append({'title':title,'min':min_margins,'max':max_margins,'avg':avg_margins,'std':std_margins})
    loss_records.append({'title':title,'min':min_losses,'max':max_losses,'avg':avg_losses,'std':std_losses})
    grad_records.append({'title':title,'avg':avg_grads,'std':std_grads})
    dirrel_records.append({'title':title,'avg':avg_dirrels,'std':std_dirrels})
    cluster_counts_all[title] = cluster_counts



    
# Helper to build a subplot heatmap row with individual colorbars
import math
# … keep your other imports …

def make_heatmap_row(dist_dicts, subplot_titles, main_title,
                     offset1=None, offset2=None):
    """
    Build a 1×N row of heatmaps (log(count)), overlay lines
    b = a+offset1 and b = a+offset2 in EVERY panel if offsets given.
    """
    cols = len(dist_dicts)

    # infer p from first dist
    first = dist_dicts[0]
    coords = [(int(k.split(',')[0]), int(k.split(',')[1])) for k in first]
    p = max(max(a, b) for a, b in coords) + 1

    fig = make_subplots(
        rows=1, cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.065
    )

    for i, dist in enumerate(dist_dicts, start=1):
        # build log‐matrix
        matrix = [[0]*p for _ in range(p)]
        for k, v in dist.items():
            a, b = map(int, k.split(','))
            matrix[a][b] = v
        logm = [[math.log(v) if v>0 else 0 for v in row] for row in matrix]
        max_log = max(cell for row in logm for cell in row)

        # add heatmap
        ax_key = f"xaxis{i}" if i>1 else "xaxis"
        domain_end = getattr(fig.layout, ax_key).domain[1]
        cb_x = domain_end - 0.0036

        fig.add_trace(
            go.Heatmap(
                z=logm,
                x=list(range(p)), y=list(range(p)),
                zmin=0, zmax=max_log,
                colorscale='Inferno',
                showscale=True,
                colorbar=dict(
                    title="log(count)",
                    titlefont=dict(size=FONT_SIZE-5),
                    len=0.8, x=cb_x, xanchor='left',
                    tickmode='array',
                    tickvals=[0, max_log/2, max_log],
                    ticktext=["0", f"{max_log/2:.1f}", f"{max_log:.1f}"],
                )
            ), row=1, col=i
        )

        # overlay both lines everywhere
        if offset1 is not None and offset2 is not None:
            for off in (offset1, offset2):
                if i == 1:
                    line_color = 'white'
                elif i == cols:
                    line_color = 'black'
                else:
                    line_color = 'white'
                fig.add_shape(
                    type='line',
                    x0=off, y0=0,
                    x1=p-1, y1=p-1-off,
                    line=dict(color=line_color, width=1),
                    opacity=1.0,
                    xref=f"x{i}", yref=f"y{i}"
                )

    # layout tweaks
    fig.update_layout(
        title_text=main_title,
        font=dict(size=FONT_SIZE),
        height=int(400 * 1.2)
    )
    fig_combo.update_layout(
    margin=dict(
        t=0,   # top
        l=0,   # left
        r=0,   # right
        b=0    # bottom
    )
)
    fig.update_annotations(font_size=24)
    for idx in range(1, cols+1):
        fig.update_xaxes(title_text='b', row=1, col=idx, title_standoff=0)
        if idx == 1:
            fig.update_yaxes(title_text='a', row=1, col=idx, title_standoff=0)
        else:
            fig.update_yaxes(showticklabels=False, row=1, col=idx, title_standoff=0)

    return fig

offset1, offset2 = 22, 40

mlp_key = 'MLP vec add'
filtered = {}
for k, v in dist_max_all[mlp_key].items():
    a, b = map(int, k.split(','))
    filtered[k] = v if a == b else 0
dist_max_all[mlp_key] = filtered

# Build section figures
sections = []

# Build a 2×4 grid, row 1 = max‐activation, row 2 = center‐of‐mass
fig_combo = make_subplots(
    rows=2, cols=4,
    subplot_titles=[
        *(f"{t}\noff diag avg: {compute_avg_log_counts(dist_max_all[t], offset1, offset2)['between']:.2f}"
          for t in titles),
        *(f"{t}\noff diag avg: {compute_avg_log_counts(dist_center_all[t], offset1, offset2)['between']:.2f}"
          for t in titles)
    ],
    horizontal_spacing=0.09,
    vertical_spacing=0.12
)

for row_idx, dist_source in enumerate((dist_max_all, dist_center_all), start=1):
    y_center = 0.75 if row_idx == 1 else 0.19
    for col_idx, title in enumerate(titles, start=1):
        dist = dist_source[title]
        coords = [tuple(map(int, k.split(','))) for k in dist]
        p = max(max(a, b) for a, b in coords) + 1
        matrix = [[0]*p for _ in range(p)]
        for k, v in dist.items():
            a, b = map(int, k.split(','))
            matrix[a][b] = v
        logm = [[math.log(v) if v > 0 else 0 for v in rowm] for rowm in matrix]
        max_log = max(cell for rowm in logm for cell in rowm)

        axis_id = (row_idx - 1)*4 + col_idx
        domain = fig_combo.layout[f"xaxis{axis_id}"].domain
        cb_x = domain[1] - 0.0036

        fig_combo.add_trace(
            go.Heatmap(
                z=logm, x=list(range(p)), y=list(range(p)),
                zmin=0, zmax=max_log,
                colorscale='Inferno',
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text='log(count)',
                        font=dict(size=FONT_SIZE-4)
                    ),
                    titlefont=dict(size=FONT_SIZE-4),
                    len=0.4,
                    y=y_center,
                    yanchor='middle',
                    x=cb_x,
                    xanchor='left',
                    tickmode='array',
                    tickvals=[0, max_log/2, max_log],
                    ticktext=["0", f"{max_log/2:.1f}", f"{max_log:.1f}"],
                )
            ),
            row=row_idx, col=col_idx
        )

        # overlay the two off-diagonal lines
        for off in (offset1, offset2):
            line_color = 'black' if col_idx == 4 else 'white'
            fig_combo.add_shape(
                type='line',
                x0=off, y0=0,
                x1=p-1, y1=p-1-off,
                line=dict(color=line_color, width=1),
                xref=f"x{axis_id}", yref=f"y{axis_id}"
            )

fig_combo.update_layout(
    title_text="Distributions of: neuron max activation (top); neuron activation center of mass (bottom)",
    font=dict(size=FONT_SIZE),
    height=820,
    showlegend=False
)
fig_combo.update_annotations(font_size=24)

# only x‑axis labels on row 2
for col in range(1,5):
    fig_combo.update_xaxes(ticklabelstandoff=0, title_text='b', row=2, col=col)
# only y‑axis labels on column 1
fig_combo.update_yaxes(title_text='a', row=1, col=1, tickfont=dict(size=23))
fig_combo.update_yaxes(title_text='a', row=2, col=1, tickfont=dict(size=23))

# hide all other ticklabels
for row in (1,2):
    for col in (2,3,4):
        fig_combo.update_yaxes(showticklabels=False, row=row, col=col)
for col in (1,2,3,4):
    fig_combo.update_xaxes(showticklabels=False, row=1, col=col)

fig_combo.write_image(
    "heatmaps_maxpre_com.pdf",
    format="pdf",
    width=1660,    # adjust as needed for Overleaf 
    height=800,    # should match the height you used in the layout
    scale=4        # bump up resolution if you like
)

print("made heatmaps_maxpre_com.pdf")
sections.insert(0, ("Combined Activation & COM Heatmaps", [fig_combo]))


titles_row = ['MLP vec add', 'Attention 0.0', 'Attention 1.0', 'MLP concat']
pairs = list(combinations(enumerate(titles_row, 1), 2))   # ( (idx1,t1), (idx2,t2) )

titles = ['MLP vec add', 'Attention 0.0', 'Attention 1.0', 'MLP concat']
pairs_sampling = list(combinations(titles_row, 2))

# for label, dist_source in [
#     ("row 1  (max activation)", dist_max_all),
#     ("row 2  (center of mass)", dist_center_all),
# ]:
#     print(f"\nGaussian-kernel MMDs with *proper* permutation p‑values for {label}:")
#     for t1, t2 in pairs_sampling:
#         mmd, p = mmd_test_from_hists(
#             dist_source[t1],
#             dist_source[t2],
#             n_draws=8_000,
#             n_permutations=50_000,
#             seed=42
#         )
#         print(f"  {t1:15s} ↔ {t2:15s} :  MMD = {mmd:6.4f}   p = {p:7.5f}")

# 1) Max preactivation heatmaps row
max_dist_list = [dist_max_all[t] for t in titles]
max_subtitles = make_heatmap_subtitles(max_dist_list, titles)
max_subs = make_heatmap_subtitles(
    [dist_max_all[t] for t in titles],
    titles,
    offset1, offset2
)
fig_max_row = make_heatmap_row(
    [dist_max_all[t] for t in titles],
    max_subs,
    "The distribution of where the maximum activation of a neuron occurs across architectures",
    offset1, offset2
)
sections.append(("Max activation heatmaps", [fig_max_row]))

center_dist_list = [dist_center_all[t] for t in titles]
center_subtitles = make_heatmap_subtitles(center_dist_list, titles)


center_subs = make_heatmap_subtitles(
    [dist_center_all[t] for t in titles],
    titles,
    offset1, offset2
)
fig_center_row = make_heatmap_row(
    [dist_center_all[t] for t in titles],
    center_subs,
    "The distribution of the center of mass of a neurons activations across various architectures",
    offset1, offset2
)
sections.append(("Center of Mass Heatmaps", [fig_center_row]))


# 2) Center of mass heatmaps row
fig_center_row = make_heatmap_row(
    [dist_center_all[t] for t in titles],
    titles,
    "Center of Mass Across Architectures"
)

# 3) Worst-case equivariance histograms
fig_worst = make_subplots(rows=1, cols=len(equiv_records), subplot_titles=[r['title'] for r in equiv_records])
for i, r in enumerate(equiv_records, 1):
    fig_worst.add_trace(go.Histogram(
        x=r['worst'],
        name=r['title'],
        marker_color=colors[i-1]
    ), row=1, col=i)
fig_worst.update_layout(title_text='Worst-case Equivariance by Architecture', showlegend=False, font=dict(size=FONT_SIZE))
sections.append(('Worst-case Equivariance Histograms', [fig_worst]))

# 4) Generic scatter helper
import math

def make_scatter(rows, x, y, title, x_label=None, y_label=None):
    df = pd.DataFrame(rows)
    fig = px.scatter(
        df, x=x, y=y, color='title', title=title,
        color_discrete_map=color_map
    )
    fig.update_layout(font=dict(size=FONT_SIZE), height=400)
    if x_label:
        fig.update_xaxes(title_text=x_label)
    if y_label:
        fig.update_yaxes(title_text=y_label)
    return fig

# Scatter plots
equiv_rows = [{'title':r['title'],'mean':m,'std':s} for r in equiv_records for m,s in zip(r['mean'],r['std'])]
sections.append(('Equivariance Scatter', [make_scatter(equiv_rows,'mean','std','Equivariance (mean vs std)')]))
margin_rows = [{'title':r['title'],'avg_margin':a,'std_dev_margin':s} for r in margin_records for a,s in zip(r['avg'],r['std'])]
sections.append(('Margin Scatter', [make_scatter(margin_rows,'avg_margin','std_dev_margin','Margin (avg vs std)')]))
loss_rows = [{'title':r['title'],'avg_loss':a,'std_dev_loss':s} for r in loss_records for a,s in zip(r['avg'],r['std'])]
sections.append(('Loss Scatter', [make_scatter(loss_rows,'avg_loss','std_dev_loss','Loss (avg vs std)')]))
# --- Gradient Symmetricity Scatter (fixed) ---
grad_rows = [
    {'title': r['title'], 'avg_grad': a, 'std_grad': s}
    for r in grad_records
    for a, s in zip(r['avg'], r['std'])
]
sections.append(('Gradient Symmetricity Scatter', [
    make_scatter(
        grad_rows,
        'avg_grad', 'std_grad',
        'Gradient Symmetricity (avg vs std)',
        x_label='avg gradient symmetricity',
        y_label='std gradient symmetricity'
    )
]))

# --- Distance Irrelevance Scatter (fixed) ---
dirrel_rows = [
    {'title': r['title'], 'avg_dirrel': a, 'std_dirrel': s}
    for r in dirrel_records
    for a, s in zip(r['avg'], r['std'])
]
fig_dirrel = make_scatter(
    dirrel_rows,
    'avg_dirrel', 'std_dirrel',
    'Distance Irrelevance (avg vs std)',
    x_label='avg distance irrelevance',
    y_label='std distance irrelevance'
)
sections.append(('Distance Irrelevance Scatter', [fig_dirrel]))
# now relabel the axes
fig_dirrel.update_xaxes(title_text='avg distance irrelevance')
fig_dirrel.update_yaxes(title_text='std dist irrel')

# append into your sections list
sections.append(('Distance Irrelevance Scatter', [fig_dirrel]))

fig_clusters = make_subplots(
    rows=1, cols=len(titles),
    subplot_titles=[
        f"{t} (avg={sum(cluster_counts_all[t])/len(cluster_counts_all[t]):.2f})"
        for t in titles
    ],
    horizontal_spacing=0.05
)
for i, t in enumerate(titles, start=1):
    fig_clusters.add_trace(go.Histogram(
        x=cluster_counts_all[t],
        nbinsx=20,
        name=t,
        marker_color=colors[i-1],
        marker_line_width=0
    ),
        row=1, col=i
    )
fig_clusters.update_layout(
    title_text="Distribution of Cluster Counts per File",
    showlegend=False,
    font=dict(size=FONT_SIZE),
    height=400
)
sections.append(("Cluster Count Histograms", [fig_clusters]))



def torus_diag_distance(a: int, b: int, p: int) -> int:
    """
    Shortest wrap-around distance from (a,b) to the diagonal a=b on a p×p torus.
    Δ = min(|b-a|, p-|b-a|).
    """
    d = abs(b - a)
    return d if d <= p // 2 else p - d

# Precompute distances and averages for each architecture
distances_max_list = []
distances_center_list = []
avg_max_list = []
avg_center_list = []
psize_list = []

for title in titles:
    # infer torus size
    coords = [tuple(map(int, k.split(","))) for k in dist_max_all[title]]
    p = max(max(a, b) for a, b in coords) + 1
    psize_list.append(p)

    # build flattened-distance lists (count‑weighted)
    dists_max = []
    for k, cnt in dist_max_all[title].items():
        a, b = map(int, k.split(","))
        d = torus_diag_distance(a, b, p)
        dists_max.extend([d] * cnt)
    distances_max_list.append(dists_max)

    dists_center = []
    for k, cnt in dist_center_all[title].items():
        a, b = map(int, k.split(","))
        d = torus_diag_distance(a, b, p)
        dists_center.extend([d] * cnt)
    distances_center_list.append(dists_center)

    # compute averages
    avg_max = sum(dists_max) / len(dists_max) if dists_max else 0.0
    avg_center = sum(dists_center) / len(dists_center) if dists_center else 0.0
    avg_max_list.append(avg_max)
    avg_center_list.append(avg_center)

# build subplot_titles: first row then second row
subplot_titles = []
for title, avg in zip(titles, avg_max_list):
    subplot_titles.append(f"{title}\n avg: {avg:.2f}")
for title, avg in zip(titles, avg_center_list):
    subplot_titles.append(f"{title}\n avg: {avg:.2f}")

# create the figure with titles on both rows
fig_torus = make_subplots(
    rows=2, cols=len(titles),
    subplot_titles=subplot_titles,
    row_titles=["Max activation", "Center of mass"],
    horizontal_spacing=0.05
)

# add the histograms
for col in range(len(titles)):
    p = psize_list[col]
    fig_torus.add_trace(go.Histogram(
       x=distances_max_list[col],
       nbinsx=p//2,
       marker_color=colors[col],
       marker_line_width=0
   ),
        row=1, col=col+1
    )
    fig_torus.add_trace(go.Histogram(
       x=distances_center_list[col],
       nbinsx=p//2,
       marker_color=colors[col],
       marker_line_width=0
   ),
        row=2, col=col+1
    )

fig_torus.update_layout(
    title_text="Shortest torus‑distance from diagonal (a=b) to: max activation (row 1); center of mass (row 2)",
    font=dict(size=FONT_SIZE-2),
    showlegend=False,
    height=650
)
# — force all x‑axes to span 0…29 with ticks at 0,5,10,…,29
fig_torus.update_xaxes(
    range=[0, 29],
    tick0=0,
    dtick=5
)
# label every x‑axis
fig_torus.update_xaxes(title_text="distance")

# label only the first column’s y‑axes
fig_torus.update_yaxes(title_text="counts", col=1)

for ann in fig_torus.layout.annotations:
    ann.font = dict(size=24)
# append to the end of your sections list so it renders last
fig_torus.write_image(
    "histograms_torus_distance.pdf",  # output filename
    format="pdf",
    width=1600,   # match your layout needs
    height=650,   # should match the height you set above
    scale=4       # bump up resolution if you like
)
print("made histograms_torus_distance.pdf")
sections.append(("Torus Distance Histograms", [fig_torus]))

# --- Build combined 1×2 figure for Gradient Symmetricity & Distance Irrelevance ---
fig_combined = make_subplots(
    rows=1, cols=2,
    subplot_titles=("", ""),
    horizontal_spacing=0.10
)

# --- randomly split each cluster into 10 “layers”  ---
grad_layers   = {}   # key = (title, layer), value = list of row‑dicts
dirrel_layers = {}

for title in titles:
    # gradient symmetricity rows for this title
    rows_g = [r for r in grad_rows   if r['title'] == title]
    layers_g = [random.randint(0,99) for _ in rows_g]
    for layer in range(100):
        grad_layers[(title,layer)] = [
            r for r, l in zip(rows_g, layers_g) if l == layer
        ]

    # distance irrelevance rows for this title
    rows_d = [r for r in dirrel_rows if r['title'] == title]
    layers_d = [random.randint(0,99) for _ in rows_d]
    for layer in range(100):
        dirrel_layers[(title,layer)] = [
            r for r, l in zip(rows_d, layers_d) if l == layer
        ]


# --- add traces in interleaved “layer” order  ---
for layer in range(100):
    for i, title in enumerate(titles):
        color = color_map[title]
        legend = (layer == 0)   # only show legend entry on the first sub‑trace

        # Gradient Symmetricity panel (col=1)
        subset = grad_layers[(title, layer)]
        fig_combined.add_trace(
            go.Scatter(
                x=[r['avg_grad']  for r in subset],
                y=[r['std_grad']  for r in subset],
                mode='markers',
                marker=dict(color=color),
                name=title,
                legendgroup=title,
                showlegend=legend
            ),
            row=1, col=1
        )

        # Distance Irrelevance panel (col=2)
        subset = dirrel_layers[(title, layer)]
        fig_combined.add_trace(
            go.Scatter(
                x=[r['avg_dirrel'] for r in subset],
                y=[r['std_dirrel'] for r in subset],
                mode='markers',
                marker=dict(color=color),
                name=title,
                legendgroup=title,
                showlegend=False
            ),
            row=1, col=2
        )


# --- label axes ---
fig_combined.update_xaxes(title_text="avg gradient symmetricity", row=1, col=1)
fig_combined.update_yaxes(title_text="std gradient symmetricity", row=1, col=1)
fig_combined.update_xaxes(title_text="avg distance irrelevance", row=1, col=2)
fig_combined.update_yaxes(title_text="std distance irrelevance", row=1, col=2)

# --- overall layout tweaks ---
fig_combined.update_layout(
    title_text="Per‑architecture gradient symmetricity & distance irrelevance",
    font=dict(size=FONT_SIZE),
    height=520
)



for ann in fig_combined.layout.annotations:
    ann.font = dict(size=24)

fig_combined.write_image(
    "gradient_distance_irrelevance_2.pdf",  # output filename
    format="pdf",
    width=1200,   # adjust to suit your column widths
    height=520,   # matches the height you set in layout
    scale=4       # increase resolution if desired
)
print("made gradient_distance_irrelevance_2.pdf")
# append it last so it renders at the end
sections.append(("Gradient & Distance Irrelevance", [fig_combined]))

# 5) Phase distribution heatmaps
fig_phases = make_subplots(
    rows=2, cols=len(titles),
    subplot_titles=[f"{t}" for t in titles] + [f"{t}" for t in titles],
    row_titles=["distribution_of_phases","distribution_of_phases_f_a=f_b"],
    horizontal_spacing=0.05, vertical_spacing=0.1
)
for idx, t in enumerate(titles, start=1):
    # row 1: distribution_of_phases
    dist = dist_phase_all[t]
    coords = [tuple(map(int, k.split(','))) for k in dist]
    p = max(max(a,b) for a,b in coords) + 1
    matrix = [[0]*p for _ in range(p)]
    for k,v in dist.items():
        a,b = map(int, k.split(','))
        matrix[a][b] = v
    logm = [[math.log(v) if v>0 else 0 for v in row] for row in matrix]
    max_log = max(cell for row in logm for cell in row)
    fig_phases.add_trace(
        go.Heatmap(z=logm, x=list(range(p)), y=list(range(p)),
                   zmin=0, zmax=max_log, colorscale='Inferno'),
        row=1, col=idx
    )
    # row 2: distribution_of_phases_f_a=f_b
    dist2 = dist_phase_eq_all[t]
    matrix2 = [[0]*p for _ in range(p)]
    for k,v in dist2.items():
        a,b = map(int, k.split(','))
        matrix2[a][b] = v
    logm2 = [[math.log(v) if v>0 else 0 for v in row] for row in matrix2]
    fig_phases.add_trace(
        go.Heatmap(z=logm2, x=list(range(p)), y=list(range(p)),
                   zmin=0, zmax=max_log, colorscale='Inferno'),
        row=2, col=idx
    )
fig_phases.update_layout(title_text="phase distribution: not equal f's vs constrained to be equal f's", font=dict(size=24), height=800, showlegend=False)
sections.append(("Phase Distributions", [fig_phases]))

# 6) Percent of neurons learning harmonics
harm_scatter = go.Figure()
for title in titles:
    x = [rec['total_neurons'] for rec in per_file_data[title]]
    y = [rec['harm_pct'] for rec in per_file_data[title]]
    harm_scatter.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(color=color_map[title]),
        name=title
    ))
harm_scatter.update_layout(
    title_text="percent of neurons learning harmonics",
    xaxis_title="number of neurons",
    yaxis_title="% harmonics",
    font=dict(size=24),
    height=400
)
sections.append(("Percent Harmonics", [harm_scatter]))

# Triplet harmonics scatter (per-file)
harm3_scatter = go.Figure()
for title in titles:
    x3 = [rec['total_neurons'] for rec in per_file_data[title]]
    y3 = [rec['harm3_pct'] for rec in per_file_data[title]]
    harm3_scatter.add_trace(go.Scatter(
        x=x3, y=y3,
        mode='markers',
        marker=dict(color=color_map[title]),
        name=title
    ))
harm3_scatter.update_layout(
    title_text="percent triplet harmonics",
    xaxis_title="number of neurons",
    yaxis_title="% triplet harmonics",
    font=dict(size=24),
    height=400
)
sections.append(("Triplet Harmonics", [harm3_scatter]))

# 2d scatter plots of harmonics vs avg stats
hd_fig = make_subplots(rows=1, cols=2,
    subplot_titles=("% harmonics vs avg gradient symmetricity",
                    "% harmonics vs avg distance irrelevance"),
    horizontal_spacing=0.15
)
for i, title in enumerate(titles):
    xg = [rec['avg_grad'] for rec in per_file_data[title]]
    yh = [rec['harm_pct'] for rec in per_file_data[title]]
    hd_fig.add_trace(
        go.Scatter(x=xg, y=yh,
                   mode='markers', marker=dict(color=color_map[title]),
                   name=title), row=1, col=1
    )
    xd = [rec['avg_dirrel'] for rec in per_file_data[title]]
    yt = [rec['harm_pct'] for rec in per_file_data[title]]
    hd_fig.add_trace(
        go.Scatter(x=xd, y=yt,
                   mode='markers', marker=dict(color=color_map[title]),
                   name=title, showlegend=False), row=1, col=2
    )
hd_fig.update_xaxes(title_text="avg gradient symmetricity", row=1, col=1)
hd_fig.update_yaxes(title_text="% harmonics", row=1, col=1)
hd_fig.update_xaxes(title_text="avg distance irrelevance", row=1, col=2)
hd_fig.update_yaxes(title_text="% harmonics", row=1, col=2)
hd_fig.update_layout(
    title_text="2d scatter plots of harmonics for % harmonics vs avg gradient symmetricity and % harmonics vs avg distance irrelevance",
    font=dict(size=24), height=400
)
sections.append(("Harmonics vs Stats", [hd_fig]))


# Write out HTML
html_content = '<html><head><meta charset="utf-8" /><script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head><body>'
for heading, figs in sections:
    html_content += f"<h2 style='font-size:{FONT_SIZE}px'>{heading}</h2>"
    for fig in figs:
        html_content += fig.to_html(full_html=False, include_plotlyjs=False)
html_content += '</body></html>'

with open('paper_2_plots_final.html', 'w') as out:
    out.write(html_content)

print(f"Generated paper_2_plots_final.html (fonts={FONT_SIZE}, max files={MAX_FILES_PER_DIR}).")



import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def _sq_dists(a, b):                 # (p,d) × (q,d) → (p,q)
    return jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)

def mmd2_u_gaussian_batched(X, Y, *, sigma, block_size=4096):
    """
    Unbiased Gaussian-kernel MMD² computed in O(block_size²) memory.
    X, Y: JAX arrays of shape (m,d), (n,d)
    Returns a scalar jnp.float32 on the current device.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    Y = jnp.asarray(Y, dtype=jnp.float32)
    m, n = X.shape[0], Y.shape[0]
    sig2 = float(sigma) ** 2

    # running sums (on device)
    sum_xx = jnp.float32(0.)
    sum_yy = jnp.float32(0.)
    sum_xy = jnp.float32(0.)

    # --- XX -------------------------------------------------
    for i in range(0, m, block_size):
        Xi = X[i:i+block_size]
        for j in range(0, m, block_size):
            Xj = X[j:j+block_size]
            K = jnp.exp(-_sq_dists(Xi, Xj) / (2*sig2))
            if i == j:                         # subtract diagonal once
                sum_xx += jnp.sum(K) - jnp.sum(jnp.diag(K))
            else:
                sum_xx += jnp.sum(K)

    # --- YY -------------------------------------------------
    for i in range(0, n, block_size):
        Yi = Y[i:i+block_size]
        for j in range(0, n, block_size):
            Yj = Y[j:j+block_size]
            K = jnp.exp(-_sq_dists(Yi, Yj) / (2*sig2))
            if i == j:
                sum_yy += jnp.sum(K) - jnp.sum(jnp.diag(K))
            else:
                sum_yy += jnp.sum(K)

    # --- XY -------------------------------------------------
    for i in range(0, m, block_size):
        Xi = X[i:i+block_size]
        for j in range(0, n, block_size):
            Yj = Y[j:j+block_size]
            sum_xy += jnp.sum(jnp.exp(-_sq_dists(Xi, Yj) / (2*sig2)))

    term_xx = sum_xx / (m * (m - 1))
    term_yy = sum_yy / (n * (n - 1))
    term_xy = sum_xy / (m * n)
    return term_xx + term_yy - 2. * term_xy

def mmd_permutation_test_batched(
        X, Y,
        *,
        sigma=None,
        n_permutations=500,       # keep this modest!
        seed=0,
        block_size=4096,
):
    """
    Memory-safe two-sample test for large 1-D (or d-D) data.
    X, Y: np / jax arrays – they are NOT copied.
    Returns (sqrt(MMD²_obs), p-value).
    """
    X = jnp.asarray(X, dtype=jnp.float32).reshape(-1, X.shape[-1])
    Y = jnp.asarray(Y, dtype=jnp.float32).reshape(-1, Y.shape[-1])
    Z = jnp.concatenate([X, Y], axis=0)
    m, n, N = len(X), len(Y), len(X) + len(Y)

    # --- bandwidth: pooled median heuristic on a 10 000-subsample  ---
    if sigma is None:
        take = min(N, 10_000)
        idx  = jax.random.choice(jax.random.PRNGKey(seed), N, (take,), replace=False)
        d2   = _sq_dists(Z[idx], Z[idx])
        sigma = jnp.sqrt(0.5 * jnp.median(d2[jnp.triu_indices(take, k=1)]))

    mmd2_obs = mmd2_u_gaussian_batched(X, Y, sigma=sigma, block_size=block_size)
    mmd_obs  = jnp.sqrt(jnp.maximum(mmd2_obs, 0.0))

    # --- permutation test (works on host, each step uses GPU) ---
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_permutations):
        perm = rng.permutation(N)
        Xm   = Z[perm[:m]]
        Ym   = Z[perm[m:]]
        stat = jnp.sqrt(jnp.maximum(
            mmd2_u_gaussian_batched(Xm, Ym, sigma=sigma, block_size=block_size), 0.0))
        exceed += (float(stat) >= float(mmd_obs))
    p_val = exceed / n_permutations


    
    return float(mmd_obs), p_val

# ==========================================================
#  --------  GPU-accelerated MMD PERMUTATION TEST, 1-D  -----
#         for the torus-distance histograms you saved
# ==========================================================
import numpy as np
import jax, jax.numpy as jnp
from functools import partial
from numpy.random import default_rng

# ------------------------------------------------------------------
# 1.  DRAW i.i.d. SAMPLES  *robustly*
# ------------------------------------------------------------------
def draw_samples_1d(dist_dict, *, n_samples=2_000, seed=0, beta=None):
    """
    dist_dict maps distance -> count   (all counts ≥ 0)
    """
    # ---- keep only bins with positive mass -----------------------
    bins, counts = zip(*[(k, v) for k, v in dist_dict.items() if v > 0])
    bins    = np.asarray(bins,   dtype=np.float32)
    counts  = np.asarray(counts, dtype=np.float64)

    # ---- turn counts into probabilities -------------------------
    if beta is None:
        p = counts
    else:                       # Boltzmann / soft-max re-weighting
        logits = beta * counts
        logits = logits - logits.max()
        p = np.exp(logits)

    p = p / p.sum()             # **force exact normalisation**

    # ---- NumPy RNG needs NumPy arrays ---------------------------
    rng  = default_rng(seed)
    samp = rng.choice(bins, size=n_samples, replace=True, p=p)

    # return JAX array, shape (n,1)
    return jnp.asarray(samp, dtype=jnp.float32)[:, None]


# ------------------------------------------------------------------
# 2.  SAFE BANDWIDTH  (median-heuristic with a fallback)
# ------------------------------------------------------------------
@jax.jit
def _sq_dists_1d(A, B):
    return jnp.square(A - B.T)          # (n,m)

def _safe_bandwidth(Z):
    d2   = _sq_dists_1d(Z, Z)
    upper = d2[jnp.triu_indices(len(Z), k=1)]
    med   = jnp.median(upper)
    σ = jnp.sqrt(0.5 * med)
    # fallback: if σ == 0, use half the std-dev or 1.0
    return jnp.where(σ > 0, σ,
                     jnp.where(jnp.std(Z) > 0,
                               0.5 * jnp.std(Z),
                               1.0)).astype(jnp.float32)


# ------------------------------------------------------------------
# 3.  UNBIASED MMD²  (unchanged except σ → _safe_bandwidth)
# ------------------------------------------------------------------
@partial(jax.jit, static_argnames='sigma')
def mmd2_u_gaussian_1d(X, Y, *, sigma):
    σ2 = sigma ** 2
    Kxx = jnp.exp(-_sq_dists_1d(X, X) / (2*σ2))
    Kyy = jnp.exp(-_sq_dists_1d(Y, Y) / (2*σ2))
    Kxy = jnp.exp(-_sq_dists_1d(X, Y) / (2*σ2))

    m = X.shape[0]
    n = Y.shape[0]

    term_xx = (jnp.sum(Kxx) - jnp.trace(Kxx)) / (m*(m-1))
    term_yy = (jnp.sum(Kyy) - jnp.trace(Kyy)) / (n*(n-1))
    term_xy = jnp.sum(Kxy) / (m*n)
    return term_xx + term_yy - 2*term_xy


# ------------------------------------------------------------------
# 4.  PERMUTATION TEST  (bandwidth fallback + batch loop unchanged)
# ------------------------------------------------------------------
def mmd_permutation_test_samples_1d(X, Y,
        *, n_permutations=10_000, sigma=None,
        seed=0, batch_size=1_000):

    X = jnp.asarray(X, dtype=jnp.float32)
    Y = jnp.asarray(Y, dtype=jnp.float32)
    Z = jnp.vstack([X, Y])
    m, n = len(X), len(Y)
    N    = m + n

    sigma_safe = _safe_bandwidth(Z) if sigma is None else float(sigma)
    sigma_safe = float(sigma_safe)

    mmd2_obs = mmd2_u_gaussian_1d(X, Y, sigma=sigma_safe)
    mmd_obs  = jnp.sqrt(jnp.maximum(mmd2_obs, 0.0))

    key0 = jax.random.PRNGKey(seed)

    @jax.jit
    def _one_perm(k):
        perm = jax.random.permutation(k, N)
        Xp, Yp = Z[perm[:m]], Z[perm[m:]]
        return jnp.sqrt(jnp.maximum(
            mmd2_u_gaussian_1d(Xp, Yp, sigma=sigma_safe), 0.0))

    @jax.jit
    def _batch(carry, keys):
        stats  = jax.vmap(_one_perm)(keys)
        exceed = jnp.sum(stats >= mmd_obs)
        return carry + exceed, None

    n_full, tail = divmod(n_permutations, batch_size)
    total = 0
    if n_full:
        keys = jax.random.split(key0, n_full*batch_size)\
                       .reshape(n_full, batch_size, 2)
        total, _ = jax.lax.scan(_batch, 0, keys)
    if tail:
        keys_tail = jax.random.split(jax.random.fold_in(key0, 9999), tail)
        extra, _  = _batch(0, keys_tail)
        total += extra

    p_val = float(total) / n_permutations
    return float(mmd_obs), p_val


# ------------------------------------------------------------------
# 5.  CONVENIENCE WRAPPER  (signature unchanged)
# ------------------------------------------------------------------
def mmd_test_from_hists_1d(dist1, dist2,
        *, n_draws=2_000, n_permutations=10_000,
        beta=None, seed=0):
    X = draw_samples_1d(dist1, n_samples=n_draws, seed=seed,     beta=beta)
    Y = draw_samples_1d(dist2, n_samples=n_draws, seed=seed+123, beta=beta)
    return mmd_permutation_test_samples_1d(
        X, Y, n_permutations=n_permutations, seed=seed)

titles = ['MLP vec add', 'Attention 0.0', 'Attention 1.0', 'MLP concat']

# histograms you already built:
#   distances_max_list  → row-1 (max activation)
#   distances_center_list → row-2 (centre-of-mass)
#
# Convert those *flat* sample lists back to {distance:count} dicts.
def list_to_hist_dict(lst):
    h = collections.Counter(lst)
    return {int(k): int(v) for k,v in h.items()}

hist_max    = {t: list_to_hist_dict(lst) for t,lst in zip(titles, distances_max_list)}
hist_center = {t: list_to_hist_dict(lst) for t,lst in zip(titles, distances_center_list)}

pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]   # indices into titles

for label, source in [('max activation', hist_max),
                      ('centre of mass', hist_center)]:
    print(f"\nPermutation tests on torus-distance – {label}:")
    for i, j in pairs:
        t1, t2 = titles[i], titles[j]
        mmd, p = mmd_test_from_hists_1d(
            source[t1], source[t2],
            n_draws=2000,
            n_permutations=5000,
            beta=None,   # or beta=1.0 for Boltzmann re-weighting
            seed=42
        )
        print(f"{t1:15s} ↔ {t2:15s} :  MMD = {mmd:7.7f}   p = {p:8.7f}")



# ============================================================
#  FINAL:  GPU-accelerated MMD tests on the scatter-plot stats
#          (gradient symmetricity & distance irrelevance)
#          – once in 2-D   (avg , std)
#          – once in 1-D   (avg only)
#          → 4 printouts total
# ============================================================

print("\n==========   PERMUTATION-TEST MMDs   ==========")

rng = np.random.default_rng(12345)          # reproducibility

def _sample_rows(rows, *, n=MAX_FILES_PER_DIR):
    """Return *exactly* n rows (with replacement if needed)."""
    idx = rng.choice(len(rows), size=n, replace=True)
    return [rows[i] for i in idx]

# ------------------------------------------------------------------
# 1.  Build per-architecture sample arrays
# ------------------------------------------------------------------
grad_2d,  grad_1d  = {}, {}
dirr_2d,  dirr_1d  = {}, {}

for arch in titles:                               # titles already defined
    # ----- gradient symmetricity -----
    rows_g = [r for r in grad_rows if r['title'] == arch]
    samp_g = _sample_rows(rows_g)
    g_mat  = np.asarray([[r['avg_grad'],  r['std_grad']]  for r in samp_g],
                        dtype=np.float32)
    grad_2d[arch] = jnp.asarray(g_mat)          # (n,2)
    grad_1d[arch] = jnp.asarray(g_mat[:, :1])   # (n,1) – keep only avg

    # ----- distance irrelevance  -----
    rows_d = [r for r in dirrel_rows if r['title'] == arch]
    samp_d = _sample_rows(rows_d)
    d_mat  = np.asarray([[r['avg_dirrel'], r['std_dirrel']] for r in samp_d],
                        dtype=np.float32)
    dirr_2d[arch] = jnp.asarray(d_mat)          # (n,2)
    dirr_1d[arch] = jnp.asarray(d_mat[:, :1])   # (n,1)

pairs = list(combinations(titles, 2))            # 6 architecture pairs

def _run_suite(sample_dict) -> str:
    """Return a multi-line string with all pairwise (MMD, p) results."""
    out = []
    for a, b in pairs:
        mmd, p = mmd_permutation_test_samples(
            sample_dict[a], sample_dict[b],
            n_permutations=1_000,
            seed=42
        )
        out.append(f"{a:15s} ↔ {b:15s} :  MMD = {mmd:7.7f}   p = {p:8.7f}")
    return "\n".join(out)

# ------------------------------------------------------------------
# 2.  Four printouts (2 metrics × 2 dimensionalities)
# ------------------------------------------------------------------
print("\n▶ Gradient Symmetricity   (2-D : avg , std)\n" + _run_suite(grad_2d))
print("\n▶ Distance Irrelevance    (2-D : avg , std)\n" + _run_suite(dirr_2d))
print("\n▶ Gradient Symmetricity   (1-D : avg only) \n" + _run_suite(grad_1d))
print("\n▶ Distance Irrelevance    (1-D : avg only) \n" + _run_suite(dirr_1d))
