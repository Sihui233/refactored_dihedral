# plotting.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import extended_gcd, multiplicative_inverse, remap_with_frequency, compute_dft


def create_combined_preactivation_plot(contribution_a, contribution_b, neuron_indices, n_functions_per_plot=1):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Embedding A', f'Embedding B'))

    colors = ['blue', 'red', 'green', 'purple', 'orange']  # Add more colors if needed

    for idx, neuron_idx in enumerate(neuron_indices):
        neuron_idx_int = int(neuron_idx)

        # Plot for Embedding A
        fig.add_trace(
            go.Scatter(
                x=list(range(len(contribution_a[:, neuron_idx_int]))),
                y=contribution_a[:, neuron_idx_int],
                mode='lines+markers',
                name=f'Neuron {neuron_idx} embedding_a',
                marker=dict(color=colors[idx % len(colors)])
            ),
            row=1, col=1
        )

        # Plot for Embedding B
        fig.add_trace(
            go.Scatter(
                x=list(range(len(contribution_b[:, neuron_idx_int]))),
                y=contribution_b[:, neuron_idx_int],
                mode='lines+markers',
                name=f'Neuron {neuron_idx} embedding_b',
                marker=dict(color=colors[idx % len(colors)])
            ),
            row=1, col=2
        )

    fig.update_layout(
        title=f'Combined Pre-Activation Values for Neurons {neuron_indices}',
        height=400
    )
    fig.update_xaxes(title_text='Input Value')
    fig.update_yaxes(title_text='Pre-Activation Value')

    return fig


def create_combined_dft_plot(contribution_a, contribution_b, neuron_indices, simple_frequencies=None, n_functions_per_plot=1):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Embedding A (DFT)', f'Embedding B (DFT)'))

    colors = ['blue', 'red', 'green', 'purple', 'orange']  # Add more colors if needed

    for idx, neuron_idx in enumerate(neuron_indices):
        neuron_idx_int = int(neuron_idx)

        # Compute DFT for Embedding A
        dft_a = compute_dft(contribution_a[:, neuron_idx_int])
        dft_a_magnitude = np.abs(dft_a)
        x_values_a = list(range(len(dft_a)))

        # Assign colors: gold if frequency is in simple_frequencies, else use next available color
        if simple_frequencies is not None:
            colors_a = ['gold' if i in simple_frequencies else colors[idx % len(colors)] for i in x_values_a]
        else:
            colors_a = colors[idx % len(colors)]

        # Add labels for points with magnitude > 7.5
        labels_a = [str(x) if y > 7.5 else '' for x, y in zip(x_values_a, dft_a_magnitude)]

        fig.add_trace(
            go.Scatter(
                x=x_values_a,
                y=dft_a_magnitude,
                mode='markers+lines+text',
                marker=dict(color=colors_a, size=8),
                text=labels_a,
                textposition="top center",
                textfont=dict(color='blue'),
                name=f'Neuron {neuron_idx} Embedding A DFT'
            ),
            row=1, col=1
        )

        # Compute DFT for Embedding B
        dft_b = compute_dft(contribution_b[:, neuron_idx_int])
        dft_b_magnitude = np.abs(dft_b)
        x_values_b = list(range(len(dft_b)))

        # Assign colors: gold if frequency is in simple_frequencies, else use next available color
        if simple_frequencies is not None:
            colors_b = ['gold' if i in simple_frequencies else colors[idx % len(colors)] for i in x_values_b]
        else:
            colors_b = colors[idx % len(colors)]

        # Add labels for points with magnitude > 7.5
        labels_b = [str(x) if y > 7.5 else '' for x, y in zip(x_values_b, dft_b_magnitude)]

        fig.add_trace(
            go.Scatter(
                x=x_values_b,
                y=dft_b_magnitude,
                mode='markers+lines+text',
                marker=dict(color=colors_b, size=8),
                text=labels_b,
                textposition="top center",
                textfont=dict(color='blue'),
                name=f'Neuron {neuron_idx} Embedding B DFT'
            ),
            row=1, col=2
        )

    fig.update_layout(
        title=f'Combined DFT for Neurons {neuron_indices}',
        height=400
    )
    fig.update_xaxes(title_text='Frequency')
    max_y_value = max(max(dft_a_magnitude), max(dft_b_magnitude))
    fig.update_yaxes(title_text='Magnitude', range=[0, max_y_value * 1.15])  # Add some buffer on top
    fig.update_yaxes(title_text='Magnitude', range=[0, max_y_value * 1.15], row=1, col=2)  # Second subplot


    return fig


# ---- add this somewhere after you’ve built neuron_data and dominant_freq_clusters ----
import os, re
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

def _sanitize(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z\-]+", "-", str(s)).strip("-")

def make_cluster_html_pages(
    *,
    neuron_data: dict[int, dict[int, dict]],
    clusters,                 # dict[str, list[int]] or list[dict[str, list[int]]]
    layer_idx: int,           # 1-based layer index to use
    p: int,
    out_dir: str,
    show_full_fft: bool = False,  # if False, show only [0..p//2] on each axis
):
    os.makedirs(out_dir, exist_ok=True)
    clusters_at_layer = clusters if isinstance(clusters, dict) else clusters[layer_idx-1]
    if not clusters_at_layer:
        print(f"[make_cluster_html_pages] No clusters found for layer {layer_idx}.")
        return

    # --- helper: remap grid by frequency with wraparound + averaging ---
    def _remap_grid(grid: np.ndarray, fa: int, fb: int) -> np.ndarray:
        out = np.zeros_like(grid, dtype=float)
        cnt = np.zeros_like(grid, dtype=int)
        for a in range(p):
            na = (fa * a) % p
            for b in range(p):
                nb = (fb * b) % p
                out[na, nb] += grid[a, b]
                cnt[na, nb] += 1
        mask = cnt > 0
        out[mask] /= cnt[mask]
        return out

    # helper: build one neuron’s three figures -> HTML snippet
    def _neuron_sections(nid: int, grid: np.ndarray, include_js: bool) -> str:
        # ---------- preactivation heatmap (a on x, b on y) ----------
        fig_preact = go.Figure(
            go.Heatmap(
                x=np.arange(p), y=np.arange(p),
                z=grid.T,
                colorbar=dict(title="preact"),
            )
        )
        fig_preact.update_layout(
            title=f"preactivations (p×p grid)", xaxis_title="a", yaxis_title="b",
            yaxis=dict(autorange="reversed"), width=520, height=520, margin=dict(l=40,r=20,t=40,b=40)
        )

        # ---------- 2-D FFT & peak frequency detection ----------
        fft2 = np.fft.fft2(grid)
        mag  = np.abs(fft2)
        mag[0,0] = 0.0  # ignore DC

        if show_full_fft:
            z_fft = np.fft.fftshift(mag)  # full spectrum, centered
            # peak index in shifted space -> signed frequency
            pi, pj = np.unravel_index(np.argmax(z_fft), z_fft.shape)
            k_a = pi - (p // 2)
            k_b = pj - (p // 2)
            # map to canonical nonnegative reps in [0..floor(p/2)]
            f_a = int(min(abs(k_a), p - abs(k_a)))
            f_b = int(min(abs(k_b), p - abs(k_b)))
            x_fft = np.arange(-p//2, p//2) if p % 2 == 0 else np.arange(-(p//2), p//2+1)
            y_fft = x_fft
            title_fft = "DFT magnitude (fftshifted)"
        else:
            z_fft = mag[:p//2+1, :p//2+1]  # nonnegative quadrant
            pi, pj = np.unravel_index(np.argmax(z_fft), z_fft.shape)
            f_a, f_b = int(pi), int(pj)
            x_fft = np.arange(z_fft.shape[1]); y_fft = np.arange(z_fft.shape[0])
            title_fft = "DFT magnitude (0..⌊p/2⌋)"

        # decide remap factors per your rule
        if f_a == f_b or f_a == 0 or f_b == 0:
            f = max(f_a, f_b)
            if f == 0:  # extreme edge case: everything zero
                f = 1
            remap_fa = remap_fb = f
            remap_rule = f"axis/diag → scale by f={f}"
        else:
            remap_fa, remap_fb = f_a, f_b
            remap_rule = f"oblique → scale by (f_a,f_b)=({f_a},{f_b})"

        # ---------- remapped preactivations ----------
        remapped = _remap_grid(grid, remap_fa, remap_fb)
        fig_remap = go.Figure(
            go.Heatmap(
                x=np.arange(p), y=np.arange(p),
                z=remapped.T,
                colorbar=dict(title="preact"),
            )
        )
        fig_remap.update_layout(
            title=f"remapped preactivations (peak=({f_a},{f_b}); {remap_rule})",
            xaxis_title="a'", yaxis_title="b'",
            yaxis=dict(autorange="reversed"),
            width=520, height=520, margin=dict(l=40,r=20,t=40,b=40)
        )

        # ---------- FFT heatmap ----------
        fig_fft = go.Figure(go.Heatmap(x=x_fft, y=y_fft, z=z_fft, colorbar=dict(title="|FFT|")))
        fig_fft.update_layout(
            title=title_fft, xaxis_title="freq b", yaxis_title="freq a",
            width=520, height=520, margin=dict(l=40,r=20,t=40,b=40)
        )

        # only include PlotlyJS once per page (attach it to the first figure in the row)
        pre_div   = pio.to_html(fig_preact, include_plotlyjs="cdn" if include_js else False,
                                full_html=False, div_id=f"preact-{nid}")
        remap_div = pio.to_html(fig_remap, include_plotlyjs=False, full_html=False,
                                div_id=f"remap-{nid}")
        fft_div   = pio.to_html(fig_fft, include_plotlyjs=False, full_html=False,
                                div_id=f"dft-{nid}")

        # layout: first row = preactivations | remapped; second row = FFT
        return f"""
<section style="margin:24px 0; padding-bottom:16px; border-bottom:1px solid #eee">
  <h2 style="font:600 18px/1.2 ui-sans-serif,system-ui">neuron {nid}</h2>
  <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;align-items:start">
    {pre_div}
    {remap_div}
  </div>
  <div style="margin-top:16px">
    {fft_div}
  </div>
</section>
"""

    for freq_key, ids in sorted(clusters_at_layer.items()):
        if not ids:
            continue
        safe_key = _sanitize(str(freq_key).replace(",", "-"))
        out_path = os.path.join(out_dir, f"cluster_{safe_key}.html")

        sections = []
        include_js = True
        remapped_grids = []
        for nid in sorted(ids):
            grid = neuron_data[layer_idx][nid]["real_preactivations"]  # shape (p,p)

            # ---- get remapped for accumulation ----
            # reuse the inner helper for remapping directly
            fft2 = np.fft.fft2(grid)
            mag = np.abs(fft2); mag[0,0] = 0
            if show_full_fft:
                z_fft = np.fft.fftshift(mag)
                pi, pj = np.unravel_index(np.argmax(z_fft), z_fft.shape)
                k_a, k_b = pi - (p//2), pj - (p//2)
                f_a = int(min(abs(k_a), p - abs(k_a)))
                f_b = int(min(abs(k_b), p - abs(k_b)))
            else:
                z_fft = mag[:p//2+1, :p//2+1]
                pi, pj = np.unravel_index(np.argmax(z_fft), z_fft.shape)
                f_a, f_b = int(pi), int(pj)
            if f_a == f_b or f_a == 0 or f_b == 0:
                f = max(f_a, f_b) or 1
                remap_fa = remap_fb = f
            else:
                remap_fa, remap_fb = f_a, f_b

            remapped = _remap_grid(grid, remap_fa, remap_fb)
            remapped_grids.append(remapped)

            sections.append(_neuron_sections(nid, grid, include_js))
            include_js = False

        if remapped_grids:
            summed = np.sum(remapped_grids, axis=0)

            # ReLU-clip each neuron BEFORE summation
            clipped_grids = [np.where(r <= 0.09, 0, r) for r in remapped_grids]
            clipped_sum = np.sum(clipped_grids, axis=0)

            fig_sum = go.Figure(go.Heatmap(z=summed.T, colorbar=dict(title="sum")))
            fig_sum.update_layout(title="Sum of remapped", width=520, height=520, yaxis=dict(autorange="reversed"),
                                margin=dict(l=40,r=20,t=40,b=40))

            fig_relu = go.Figure(go.Heatmap(z=clipped_sum.T, colorbar=dict(title="sum")))
            fig_relu.update_layout(title="ReLU-clipped remapped sum", width=520, height=520, yaxis=dict(autorange="reversed"),
                                margin=dict(l=40,r=20,t=40,b=40))

            sum_div = pio.to_html(fig_sum, include_plotlyjs=False, full_html=False, div_id="sum-remapped")
            relu_div = pio.to_html(fig_relu, include_plotlyjs=False, full_html=False, div_id="relu-sum")


            sections.append(f"""
<section style="margin:24px 0; padding-bottom:16px; border-bottom:1px solid #eee">
  <h2 style="font:600 18px/1.2 ui-sans-serif,system-ui">sum of the above</h2>
  <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;align-items:start">
    {sum_div}
    {relu_div}
  </div>
</section>
""")

        # <-- build html *after* adding the sum section
        html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>cluster {freq_key}</title>
<style>
body{{max-width:1100px;margin:24px auto;padding:0 16px;font:14px/1.45 ui-sans-serif,system-ui}}
h1{{font:700 22px/1.2 ui-sans-serif,system-ui;margin:8px 0 16px}}
</style>
</head>
<body>
  <h1>cluster {freq_key}  —  layer {layer_idx}</h1>
  {"".join(sections)}
</body>
</html>"""
        with open(out_path, "w") as f:
            f.write(html)
        print(f"[make_cluster_html_pages] wrote {out_path}")