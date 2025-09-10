# plot_cluster_html.py
import os, re, json
from typing import Tuple, Dict, Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import dihedral
# -------------------------- helpers: remapping --------------------------

def remap_f(f: int) -> int:
    """
    Returns the remapping multiplier given f from 'cluster_{f}.json'.
    Replace the body with your desired mapping later.
    """
    return f

# def _remap_2d_matrix_by_mul_mod(z_img: np.ndarray, n: int, p:int, r: int) -> np.ndarray:
#     """
#     Returns z_remap where each original cell z[a,b] is moved to
#     (a', b') = (r*a mod n, r*b mod n).
#     """
#     assert z_img.shape == (n, n)
#     a = np.arange(n)[:, None]
#     b = np.arange(n)[None, :]
#     new_a = (r * a) % n  # shape (n,1)
#     new_b = (r * b) % n  # shape (1,n)

#     z_remap = np.empty_like(z_img)
#     # vectorized placement
#     z_remap[new_a, new_b] = z_img
#     return z_remap

# def _remap_1d_by_mul_mod(y: np.ndarray, n: int, p:int,  r: int) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Returns (x_remap, y_same_order) where x_remap[c] = (r*c mod n).
#     Useful for plotting original y values at remapped x positions.
#     """
#     c = np.arange(n)
#     x_remap = (r * c) % p
#     return x_remap, y

# -------------------------- helpers: fitting --------------------------
def _corr_index_dn(p: int) -> np.ndarray:
        """
        Build a length-(|G|^2) vector giving, for each pair (g_idx,h_idx),
        the class index of g*h in D_n under your enumeration.
        """
        G, idx = dihedral.dn_elements(p)
        q = 2 * p
        out = np.empty(q*q, dtype=int)
        t = 0
        for gi, g in enumerate(G):
            for hi, h in enumerate(G):
                prod = dihedral.mult(g, h, p)
                out[t] = idx[prod]
                t += 1
        return out

def _ang(n: int):
    a = np.arange(n)
    return (2.0 * np.pi / n) * a

def fit_shared_phase_2d(z_flat: np.ndarray, n: int) -> Tuple[float, float, float, np.ndarray]:
    """
    Fit: z[a,b] ≈ A_a cos(2π a / n + φ) + A_b cos(2π b / n + φ)  (shared φ)
    Returns (A_a, A_b, phi, z_hat_flat).
    The fit is done via a 4-term linear LS (cos_a, sin_a, cos_b, sin_b), then
    projecting both (α1, α2) and (β1, β2) onto a shared direction [cos φ, -sin φ].
    """
    assert z_flat.size == n * n
    a = np.repeat(np.arange(n), n)       # row-major (a outer, b inner)
    b = np.tile(np.arange(n), n)

    ang_a = _ang(n)[a]
    ang_b = _ang(n)[b]

    cos_a = np.cos(ang_a); sin_a = np.sin(ang_a)
    cos_b = np.cos(ang_b); sin_b = np.sin(ang_b)

    # Linear LS for: z ≈ α1 cos_a + α2 sin_a + β1 cos_b + β2 sin_b
    X = np.stack([cos_a, sin_a, cos_b, sin_b], axis=1)  # (n^2, 4)
    coef, *_ = np.linalg.lstsq(X, z_flat, rcond=None)
    α1, α2, β1, β2 = coef

    # Shared-phase direction v = [cos φ, -sin φ], estimate via sum of vectors
    va = np.array([α1, α2])
    vb = np.array([β1, β2])
    v  = va + vb
    if np.allclose(v, 0.0):
        # fallback: use va direction if vb cancels, else zero-φ
        v = va if np.linalg.norm(va) > 0 else np.array([1.0, 0.0])

    cos_phi, neg_sin_phi = v / np.linalg.norm(v)
    phi = np.arctan2(-neg_sin_phi, cos_phi)  # radians in (-π, π]

    # Amplitudes are projections onto the shared direction
    dir_vec = np.array([np.cos(phi), -np.sin(phi)])
    A_a = float(np.dot(va, dir_vec))
    A_b = float(np.dot(vb, dir_vec))

    # Reconstruct z_hat with shared φ
    z_hat_flat = A_a * np.cos(ang_a + phi) + A_b * np.cos(ang_b + phi)
    return A_a, A_b, float(phi), z_hat_flat

def fit_cosine_1d(y: np.ndarray, n: int) -> Tuple[float, float, np.ndarray]:
    """
    Fit: y[c] ≈ A cos(2π c / n + φ)
    Returns (A, φ, y_hat).
    """
    c = np.arange(n)
    ang = _ang(n)[c]
    X = np.stack([np.cos(ang), np.sin(ang)], axis=1)  # (n, 2)
    gamma, *_ = np.linalg.lstsq(X, y, rcond=None)
    g1, g2 = gamma
    A   = float(np.sqrt(g1 * g1 + g2 * g2))
    phi = float(np.arctan2(-g2, g1))
    y_hat = A * np.cos(ang + phi)
    return A, phi, y_hat

def wrap_phase(phi: float) -> float:
    """Wrap phase to (-π, π]."""
    out = (phi + np.pi) % (2 * np.pi) - np.pi
    # map -π -> π for nicer printing
    return np.pi if np.isclose(out, -np.pi) else out

# -------------------------- plotting per neuron --------------------------

def figure_for_neuron(
    neuron_idx: int,
    z_flat: np.ndarray,               # (n^2,)
    w_out: np.ndarray,                # (n,)
    contribs_to_logits: np.ndarray,   # (n^2, n)
    n: int,
    p:int,
    remap_mul: int = 1,
) -> go.Figure:
    # 1) 2D preactivations heatmap + fit (fit on original grid; display remapped)
    A_a, A_b, phi, _z_hat_flat = fit_shared_phase_2d(z_flat, n)
    z_img = z_flat.reshape(n, n)  # (a rows, b cols)
    # z_img_remap = _remap_2d_matrix_by_mul_mod(z_img, n, p, remap_mul)

    # 2) weights vs c + 1D cosine fit (x remapped)
    A_w, phi_w, w_hat = fit_cosine_1d(w_out, n)
    # x_remap, _ = _remap_1d_by_mul_mod(w_out, n, p, remap_mul)
    # x_remap_fit, _ = _remap_1d_by_mul_mod(w_hat, n, p, remap_mul)

    # 3) contributions to the correct logit c = (a + b) mod n, then remap (a,b)
    a_idx = np.repeat(np.arange(n), n)
    b_idx = np.tile(np.arange(n), n)
    
    assert z_flat.size == n*n and w_out.size == n and contribs_to_logits.shape == (n*n, n)

    c_corr = _corr_index_dn(p)             # 形状 (q^2,)
    corr_vals = contribs_to_logits[np.arange(n*n), c_corr].reshape(n, n)
    # corr_vals_remap = _remap_2d_matrix_by_mul_mod(corr_vals, n, p, remap_mul)

    # --- build subplot figure
    titles = (
        f"Preacts fit: A_a={A_a:.3f}, A_b={A_b:.3f}, φ={wrap_phase(phi):.3f} rad (remapped)",
        f"w_out fit: A={A_w:.3f}, φ={wrap_phase(phi_w):.3f} rad (x remapped)",
        "Neuron contribs to correct logit (remapped)"
    )
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles)

    # panel 1: heatmap of preactivations (remapped)
    fig.add_trace(
        go.Heatmap(
            z=z_img, x=list(range(n)), y=list(range(n)),
            colorscale="Viridis", colorbar=dict(title="preact"),
            hovertemplate="a'=%{x}, b'=%{y}<br>z=%{z:.4f}<extra></extra>"
        ),
        row=1, col=1
    )

    # panel 2: scatter of w_out and fitted cosine line (x remapped)
    c = np.arange(n)
    fig.add_trace(
        go.Scatter(
            x=list(range(n)), y=w_out, mode="markers", name="w_out",
            hovertemplate="c'=%{x}<br>w=%{y:.4f}<extra></extra>"
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(n)), y=w_hat, mode="lines", name="fit",
            hovertemplate="c'=%{x}<br>fit=%{y:.4f}<extra></extra>"
        ),
        row=1, col=2
    )

    # panel 3: heatmap of correct-logit contributions (remapped)
    fig.add_trace(
        go.Heatmap(
            z=corr_vals, x=list(range(n)), y=list(range(n)),
            colorscale="Viridis", colorbar=dict(title="contrib"),
            hovertemplate="a'=%{x}, b'=%{y}<br>C=%{z:.4f}<extra></extra>"
        ),
        row=1, col=3
    )

    fig.update_xaxes(title_text="a' (remapped)", row=1, col=1)
    fig.update_yaxes(title_text="b' (remapped)", row=1, col=1)
    fig.update_xaxes(title_text="c' (remapped)", row=1, col=2)
    fig.update_xaxes(title_text="a' (remapped)", row=1, col=3)
    fig.update_yaxes(title_text="b' (remapped)", row=1, col=3)

    fig.update_layout(
        showlegend=False,
        height=420,
        width=1320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

# -------------------------- IO & driver --------------------------

def load_cluster_json(path: str, n: int) -> Dict[int, Dict[str, Any]]:
    """
    Loads cluster_{f}.json and validates shapes.
    Returns dict mapping neuron_idx(int) -> {"preactivations": (n^2,), "w_out": (n,), "contribs_to_logits": (n^2, n)}
    """
    with open(path, "r") as f:
        raw = json.load(f)

    data: Dict[int, Dict[str, Any]] = {}
    for k, v in raw.items():
        nid = int(k)
        z  = np.asarray(v["preactivations"], dtype=float).reshape(-1)
        w  = np.asarray(v["w_out"],          dtype=float).reshape(-1)
        C  = np.asarray(v["contribs_to_logits"], dtype=float)
        # shape checks
        if z.size != n * n:
            raise ValueError(f"Neuron {nid}: expected preactivations len {n*n}, got {z.size}")
        if w.size != n:
            raise ValueError(f"Neuron {nid}: expected w_out len {n}, got {w.size}")
        if C.shape != (n * n, n):
            raise ValueError(f"Neuron {nid}: expected contribs_to_logits shape {(n*n, n)}, got {C.shape}")
        data[nid] = {"preactivations": z, "w_out": w, "contribs_to_logits": C}
    return dict(sorted(data.items(), key=lambda t: t[0]))

def write_cluster_html(json_path: str, n: int, p:int, out_html: str | None = None, remap_mul: int = 1) -> str:
    """
    Creates/overwrites cluster_{f}.html with <h2>neuron {idx}</h2> sections,
    each followed by a 1×3 subplot figure. Coordinates are remapped by:
      a' = (remap_mul * a) mod n, b' = (remap_mul * b) mod n, c' = (remap_mul * c) mod n.
    """
    if out_html is None:
        # derive cluster_{f}.html from json filename
        m = re.search(r"(cluster_[^./\\]+)\.json$", os.path.basename(json_path))
        base = m.group(1) if m else "cluster"
        out_html = os.path.join(os.path.dirname(json_path), f"{base}.html")

    neurons = load_cluster_json(json_path, n)

    # Write HTML scaffold once (include plotly.js CDN)
    with open(out_html, "w") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>\n")
        f.write("<title>Cluster plots</title>\n")
        f.write('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n')
        f.write("<style>body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}"
                "h2{margin:24px 8px 8px;} .figwrap{margin:6px 8px 28px;}</style>\n")
        f.write("</head><body>\n")
        f.write(f"<h1>{os.path.basename(json_path)}</h1>\n")
        f.write(f"<p><em>Remap multiplier: {remap_mul}</em></p>\n")

    # Append one figure per neuron
    for nid, obj in neurons.items():
        fig = figure_for_neuron(
            neuron_idx=nid,
            z_flat=obj["preactivations"],
            w_out=obj["w_out"],
            contribs_to_logits=obj["contribs_to_logits"],
            n=n,
            p=p,
            remap_mul=remap_mul,
        )
        html_fragment = pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=f"neuron-{nid}")
        with open(out_html, "a") as f:
            f.write(f"<h2>neuron {nid}</h2>\n<div class='figwrap'>\n{html_fragment}\n</div>\n")

    # ---- SPECIAL PLOT: sum of neuron 431 + 476 (+ 363) contributions ----
    SELECT_NEURONS = [0, 2, 21, 22, 23, 32, 34, 36, 41, 42, 47, 54,
                      60, 64, 65, 81, 87, 92, 101, 104, 110, 116,
                      117, 119, 122]
    present = [nid for nid in SELECT_NEURONS if nid in neurons]
    if present:
        c_corr = _corr_index_dn(p)
        acc = np.zeros((n, n), dtype=float)
        for nid in present:
            C = neurons[nid]["contribs_to_logits"]                    # (n*n, n)
            corr = C[np.arange(n*n), c_corr].reshape(n, n)            # (n, n)
            acc += corr

        title_ids = " ".join(map(str, present))
        fig_sel = go.Figure()
        fig_sel.add_trace(
            go.Heatmap(
                z=acc, x=list(range(n)), y=list(range(n)),
                colorscale="Viridis", colorbar=dict(title="contrib sum"),
                hovertemplate="a=%{x}, b=%{y}<br>Csum=%{z:.4f}<extra></extra>"
            )
        )
        fig_sel.update_layout(
            title=f"Sum of neuron contributions to correct logit — IDs: {title_ids}",
            height=600, width=600, margin=dict(l=10, r=10, t=40, b=10)
        )
        html_fragment = pio.to_html(fig_sel, include_plotlyjs=False, full_html=False, div_id="special-plot-selected")
        with open(out_html, "a") as f:
            f.write("<h2>special plot (selected neurons sum)</h2>\n<div class='figwrap'>\n")
            f.write(html_fragment)
            f.write("\n</div>\n")
    else:
        print("⚠️ None of the requested neuron IDs are present in this JSON.")

    with open(out_html, "a") as f:
        f.write("</body></html>")
    print(f"✔ wrote {out_html}")
    return out_html

# -------------------------- example usage --------------------------
if __name__ == "__main__":
    # Example: python plot_cluster_html.py  (adjust paths as needed)
    # CLUSTER_JSON = "/home/mila/m/moisescg/scratch/neurips_2025_crt-appendix-run-8-s-1-tiny-stuff/qualitative_59_one_embed_cheating_1_512_features_128_k_58/pdf_plots/seed_1/json/cluster_4.json"   # path to your file
    CLUSTER_JSON = "/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/pdf_plots/seed_0/json/cluster_1.json"
    N = 36
    p = 18

    # extract f from 'cluster_{f}.json' in the filename
    m = re.search(r"cluster_(\d+)\.json$", os.path.basename(CLUSTER_JSON))
    if not m:
        raise ValueError(f"Could not parse f from filename: {CLUSTER_JSON}")
    f_val = int(m.group(1))

    # get remap multiplier and pass it into write_cluster_html
    remap_mul = remap_f(f_val)

    # writes cluster_{f}.html next to the json, with remapped coordinates
    write_cluster_html(CLUSTER_JSON, N, p, remap_mul=1)
    base_html = re.sub(r"\.json$", ".html", CLUSTER_JSON)
    remapped_html = re.sub(r"\.html$", "_remapped.html", base_html)
    write_cluster_html(CLUSTER_JSON, N, p, out_html=remapped_html, remap_mul=remap_mul)
