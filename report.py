###############################################################################
# report.py  –  build a one-file PDF report for a single layer
###############################################################################
import numpy as np, jax.numpy as jnp, jax, plotly.express as px
from collections import defaultdict, OrderedDict
from itertools import islice                               
import io, base64
from PyPDF2 import PdfWriter, PdfReader
import tempfile, os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from collections import Counter
from sklearn.cluster import KMeans
from functools import reduce
from typing import Tuple,List, Iterable, Dict, Any
import plotly.io as pio
pio.kaleido.scope.default_timeout = 60 * 5
from pca_diffusion_plots_w_helpers import generate_pdf_plots_for_matrix
# from color_rules import colour_quad_mul_f        # ①  f·(a±b) mod p
# from color_rules import colour_quad_mod_g      # ②  (a±b) mod g
# from color_rules import colour_quad_a_only, colour_quad_b_only 
import uuid, time
from math import gcd
import math, json
import re
from math import gcd, pi, cos, sin
from pathlib import Path
import numpy as _np
import paper_plots
import dihedral
# ──────────────────────────────────────────────────────────────
#  DFT and Remapping helper
# ──────────────────────────────────────────────────────────────

def inverse_on_cosets(f: int, p: int) -> dict[int, int]:
    """
    Returns a dict mapping each coset-index k -> inverse of (f/g) mod (p/g).
    Even when gcd(f,p)=g>1 this is well-defined, since gcd(f/g, p/g)=1.
    """
    f, p = int(f), int(p)
    g = gcd(f, p)
    pg = p // g         # size of each coset
    f_prime = f // g    # the true multiplier inside each coset
    # sanity-check
    if gcd(f_prime, pg) != 1:
        raise ValueError(f"{f_prime} not invertible mod {pg}")
    inv = pow(f_prime, -1, pg)
    # every coset 0..g-1 uses the same inverse
    return {k: inv for k in range(g)}

def step_size(f: int, p: int) -> int:
    """
    Definition 4.1:  d := (f/g)^{-1} mod (p/g),  where g = gcd(f,p).
    """
    g = math.gcd(f, p)
    n = p // g
    # invert f/g modulo n
    return pow(f // g, -1, n)

def _remap_block(block: np.ndarray) -> tuple[np.ndarray,int,int]:
    """
    Definition 4.2 style remap:
      let h(a,b)=block[a,b], freq=(fa,fb).
      define step-sizes da,db.
      build g so that
        g(a2, b2) = h( (da * a2) % p, (db * b2) % p ).

    This fills every pixel exactly once as long as we use the step sizes
    from Def 4.1.
    """
    p = block.shape[0]
    # 1. find the dominant freq along two directions
    fb, fa = dominant_freqs_ab(block)

    # 2. calculate step sizes (didn't use them here though)
    db = step_size(fb, p)
    da = step_size(fa, p)

    # 3. remap
    out = np.zeros_like(block)
    for a2 in range(p):
        for b2 in range(p):
            # a0 = (da * a2) % p
            # b0 = (db * b2) % p
            #out[a2, b2] = block[a0, b0]
            a0 = (fa * a2) % p
            b0 = (fb * b2) % p
            out[a0, b0] = block[a2, b2]

    return out, fb, fa
# out[(fa * a) % p, (fb * b) % p] = block[a, b]
def dominant_freqs_ab(grid: np.ndarray) -> Tuple[int, int]:
    """Find the dominant single-frequency component along the horizontal (b)
    and vertical (a) axes **without summing the spectrum**.

    Parameters
    ----------
    grid : ndarray, shape = (p, p)
        Real-valued 2-D sample grid.

    Returns
    -------
    fb, fa : int, int
        Dominant frequency indices along *b* (horizontal / x) and *a*
        (vertical / y) **in this order**.
    """
    p = grid.shape[0]
    F = np.fft.fft2(grid)
    F_mag = np.abs(F) ** 2              # energy spectrum
    F_mag[0, 0] = 0                     # suppress DC

    # Row 0   → k_y = 0  (horizontal variations only)
    # Column 0→ k_x = 0  (vertical   variations only)
    half = p // 2
    row0 = F_mag[0, :half+1].copy()
    col0 = F_mag[:half+1, 0].copy()
    row0[0] = col0[0] = 0               # avoid picking DC twice

    fb = int(np.argmax(row0))           # horizontal (b)
    fa = int(np.argmax(col0))           # vertical   (a)
    return fb, fa

# ── helper: remap one p×p quadrant by (freq * a, freq * b) mod p ────────────

# def _remap_block(block: np.ndarray) -> tuple[np.ndarray,int,int]:
#     p = block.shape[0]
#     fb, fa = dominant_freqs_ab(block)
#     gb = gcd(fb, p)
#     ga = gcd(fa, p)
#     pgb = p // gb
#     pga = p // ga
#     invs_b = inverse_on_cosets(fb, p)
#     invs_a = inverse_on_cosets(fa, p)
#     out = np.zeros_like(block)
#     for a2 in range(p):
#         for b2 in range(p):
#             k_a = a2 // pga
#             k_b = b2 // pgb
#             a0, b0 = a2 % pga, b2 % pgb

#             # use the coset‐specific inverse:
#             a = (invs_a[k_a] * a0) % pga + k_a * pga
#             b = (invs_b[k_b] * b0) % pgb + k_b * pgb
#             out[a2, b2] = block[a, b]
#     return out, fb, fa

def _remapped_quadrants(tile: np.ndarray) -> tuple[np.ndarray,int,int]:
    """tile: (2p,2p) → return stitched (2p,2p) after per-quad remap"""
    p = tile.shape[0] // 2
    quads = [tile[:p, :p], tile[:p, p:], tile[p:, :p], tile[p:, p:]]
    remapped = []
    fb_set: set[int] = set()
    fa_set: set[int] = set()
    for q in quads:
        out, fb_q, fa_q = _remap_block(q)
        remapped.append(out)
        fb_set.add(fb_q)
        fa_set.add(fa_q)
    # stitch BL,BR,TL,TR back together
    stitched = np.zeros_like(tile)
    stitched[:p, :p] = remapped[0]          # BL
    stitched[:p, p:] = remapped[1]          # BR
    stitched[p:, :p] = remapped[2]          # TL
    stitched[p:, p:] = remapped[3]          # TR
    fa = 0
    fb = 0
    if len(fb_set) == 1 and len(fa_set) == 1:
        for freqb in fb_set:
            fb = freqb
        for freqa in fa_set:
            fa = freqa
    else:
        print("Quadrants are not mapped using the same fb/fa.")
    return stitched, fb, fa

# ──────────────────────────────────────────────────────────────
#  Quadrant‑phase helpers ‑‑ plain FFT + 1‑freq sine fit
# ──────────────────────────────────────────────────────────────

def single_freq_phase_shifts_ab(
    mat: np.ndarray,
    p:   int,
    fb:  int,
    fa:  int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract phase shifts of the *pure* (fb, 0) and (0, fa) components.

    Parameters
    ----------
    mat : ndarray, shape = (p*p, N) **or** (N, p*p)
        Flattened grids.
    p   : int
        Grid side length.
    fb  : int
        Horizontal frequency index (1 <= fb < p).
    fa  : int
        Vertical   frequency index (1 <= fa < p).

    Returns
    -------
    phi_b, phi_a : ndarray, shape = (N,)
        Phases (rad ∈ [0, 2π)) of the *b* and *a* components respectively.
    amps : ndarray, shape = (N,)
        Combined amplitude = 2(|c_b| + |c_a|).
    """

    if not (0 < fb < p) or not (0 < fa < p):
        raise ValueError("fb, fa must be in 1…p-1 (DC & Nyquist excluded)")

    # --- reshape: (N, p, p) -------------------------------------------
    if mat.shape[0] == p * p:
        grids = mat.T.reshape(-1, p, p)
    elif mat.shape[1] == p * p:
        grids = mat.reshape(-1, p, p)
    else:
        raise ValueError("mat must contain p*p pixels per sample")

    # --- 2‑D FFT -------------------------------------------------------
    F = np.fft.fft2(grids, axes=(-2, -1)) / (p * p)

    # Correct mapping:  (0, fb) ⇒ horizontal;  (fa, 0) ⇒ vertical
    c_b = F[:, 0,  fb]       # (k_y = 0 , k_x = fb)
    c_a = F[:, fa, 0]        # (k_y = fa, k_x = 0 )

    amps  = 2 * (np.abs(c_b) + np.abs(c_a))
    # Convert to phase in [0, 2π)
    phi_b = np.mod(np.angle(c_b), 2 * np.pi)
    phi_a = np.mod(np.angle(c_a), 2 * np.pi)

    # Map 2π back to 0 for numerical neatness
    phi_b[np.isclose(phi_b, 2 * np.pi, atol=1e-8)] = 0.0
    phi_a[np.isclose(phi_a, 2 * np.pi, atol=1e-8)] = 0.0

    return phi_b, phi_a, amps

# ---------------------------------------------------------------------
#  Quadrant helper (returns raw phases, not shifts)
# ---------------------------------------------------------------------

def _quadrant_ab_phases(grid: np.ndarray) -> List[Tuple[float, float]]:
    """Compute (phi_b, phi_a) for each quadrant of a 2p*2p grid.

    The grid is split into BL / BR / TL / TR.  For each quadrant we detect
    the dominant *b* and *a* frequencies separately, then call
    ``single_freq_phase_shifts_ab`` to get the raw phases.

    Returns
    -------
    List[(phi_b, phi_a)]
        List of phases for the four quadrants in visual order:
        BL → BR → TL → TR.
    """
    G = grid.shape[0]
    if grid.shape[1] != G:
        raise ValueError("grid must be square")
    p = G // 2

    # Visual order: BL, BR, TL, TR
    quads = [
        grid[:p, :p],   # BL
        grid[:p, p:],   # BR
        grid[p:, :p],   # TL
        grid[p:, p:],   # TR
    ]

    results: List[Tuple[float, float]] = []
    for q in quads:
        # --- 2‑D FFT ---------------------------------------------------
        F = np.fft.fft2(q) / (p * p)

        # Correct axis selection: k_y = 0 (row 0) for b, k_x = 0 (col 0) for a
        mags_b = np.abs(F[0, 1:p//2 + 1])     # (0 , k_x)
        mags_a = np.abs(F[1:p//2 + 1, 0])     # (k_y, 0)
        fb = int(np.argmax(mags_b)) + 1
        fa = int(np.argmax(mags_a)) + 1

        # --- Extract phases -------------------------------------------
        mat = q.reshape(1, -1)  # shape = (1, p*p); no transpose needed
        phi_b, phi_a, _ = single_freq_phase_shifts_ab(mat, p, fb, fa)
        results.append((phi_b[0], phi_a[0]))

    return results



# ---------------------------------------------------------------------------
# helper: turn Plotly fig → PDF bytes (raster – good enough for heat-maps)
# ---------------------------------------------------------------------------
def plotly_to_pdf_bytes(fig, scale=2.0):
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.to_image(tmp.name, scale=scale, width=600, height=450)
    with open(tmp.name, "rb") as f: png_bytes = f.read()
    return png_bytes

# ---------------------------------------------------------------------------
# dominant_irrep_axes_diag
#   Fhat_n : { (r,s,n) ->  M_{rs}^{(n)} } for a single neuron n
#   names  : ordered list of irrep labels, e.g. ["triv","sign","2D_1", …]
# Returns : (r*, s*), power, source ∈ {"row","col","diag"}
# ---------------------------------------------------------------------------
def dominant_irrep(Fhat_n, names):
    name2idx = {lab: i for i, lab in enumerate(names)}
    D = len(names)
    # ------------ 1. build power matrix P_{ij} = ‖M_{ij}‖ -------------------
    P = np.zeros((D, D))
    for (r, s, _), M in Fhat_n.items():
        M_np = np.array(M)
        P[name2idx[r], name2idx[s]] = np.linalg.norm(M)

    # ------------ 2. candidates --------------------------------------------
    P[0, 0] = -np.inf
    a_vals     = P[:, 0].copy()
    b_vals     = P[0, :].copy()
    diag_vals  = np.diagonal(P).copy()

    mx_a, ia   = a_vals.max(),  a_vals.argmax()
    mx_b, ib   = b_vals.max(),  b_vals.argmax()
    mx_d, idg  = diag_vals.max(), diag_vals.argmax()

    # ---------- selection rule ----------
    if mx_d > mx_a and mx_d > mx_b:          # pick diag
        fa, fb = idg, idg
    else:                                    # pick max on each axis
        fa, fb = ia,  ib

    return (names[fb], names[fa]) 

import numpy as np
from collections import defaultdict

def _power_matrix_from_Fhat(Fhat_n, names):
    name2idx = {lab: i for i, lab in enumerate(names)}
    D = len(names)
    P = np.zeros((D, D))
    for (r, s, _), M in Fhat_n.items():
        P[name2idx[r], name2idx[s]] = np.linalg.norm(np.array(M))
    return P

def _classify_by_gft(Fhat_n, names, freq_map, *, strict=True):
    """
    return:
      kind ∈ {"diag","axis"},
      r_star, s_star (irreps name),
      fa, fb (freq ints) if diag: fa==fb）
    """
    P = _power_matrix_from_Fhat(Fhat_n, names)
    P[0, 0] = -np.inf
    a_vals    = P[:, 0].copy()  
    b_vals    = P[0, :].copy()
    diag_vals = np.diagonal(P).copy()

    ia  = int(np.argmax(a_vals))
    ib  = int(np.argmax(b_vals))
    idg = int(np.argmax(diag_vals))
    mx_a, mx_b, mx_d = a_vals[ia], b_vals[ib], diag_vals[idg]

    if mx_d > mx_a and mx_d > mx_b:
        kind = "diag"
        r_star = s_star = names[idg]
        if r_star not in freq_map:
            if strict:
                raise KeyError(f"freq_map no mapping for irrep '{r_star}'")
            fa = fb = None
        else:
            fa = fb = int(freq_map[r_star])
    else:
        kind = "axis"
        r_star, s_star = names[ib], names[ia]
        fa = int(freq_map.get(s_star, -1)) if s_star in freq_map else None
        fb = int(freq_map.get(r_star, -1)) if r_star in freq_map else None
        if strict and (fa is None or fb is None):
            raise KeyError(f"freq_map missing for ({r_star},{s_star})")

    return {
        "kind": kind,
        "r_star": r_star, "s_star": s_star,
        "fa": fa, "fb": fb,
        "mx_a": float(mx_a), "mx_b": float(mx_b), "mx_d": float(mx_d),
        "ia": ia, "ib": ib, "idg": idg,
    }

def subgroup_scores(vec, coset_masks, skip_trivial=True, top_k=3):
    """
    Return a OrderedDict ranked by Cbar in ascending order:
      H -> {"CH":..., "Cbar":..., "K":...}
    if top_k=None, return all; otherwise only keep the top_k.
    """
    H2masks = defaultdict(list)
    for (H, cid), m in coset_masks.items():
        if skip_trivial and H == "C_1":
            continue
        H2masks[H].append(m)

    tot = vec.var()
    if tot < 1e-12:
        items = [(H, {"CH": 0.0, "Cbar": 0.0, "K": len(H2masks[H])}) for H in H2masks.keys()]
        items.sort(key=lambda t: t[1]["Cbar"])
        if isinstance(top_k, int):
            items = items[:top_k]
        return OrderedDict(items)

    scores = []
    for H, mask_list in H2masks.items():
        K = len(mask_list)
        sum_var = sum(vec[m].var() for m in mask_list)
        C = sum_var / tot
        Cbar = (sum_var / max(1, K)) / tot
        scores.append((H, {"CH": C, "Cbar": Cbar, "K": K}))

    scores.sort(key=lambda t: t[1]["Cbar"])
    if isinstance(top_k, int):
        scores = scores[:top_k]
    return OrderedDict(scores)

def merge_topk_sources(tag_scores_list, top_k: int = 3, tag_key: str = "origin"):
    """
    input:
      tag_scores_list: List[Tuple[str, OrderedDict]]
         e.g. [("Lcoset", scores_Lcoset), ("Rcoset", scores_Rcoset)]
         every scores_* : OrderedDict(H -> {"Cbar","CH","K"})
    return:
      top_mix: List[{"H","origin","Cbar","CH","K"}]  -- by Cbar ascending order returns top_k
    """
    merged = []
    for tag, scores in tag_scores_list:
        for H, s in scores.items():
            merged.append((s["Cbar"], H, tag, s["CH"], s["K"]))
    merged.sort(key=lambda t: t[0])
    top = merged[:top_k]
    return [{"H": H, tag_key: tag, "Cbar": cbar, "CH": CH, "K": K}
            for (cbar, H, tag, CH, K) in top]


def _quad_set_extrema_header(grid_2p: np.ndarray) -> tuple[str, bool]:
    """
    grid_2p: (G,G)
    """
    G = grid_2p.shape[0]
    assert grid_2p.shape[1] == G, "grid must be square"
    p = G // 2

    quads = {
        "BL": grid_2p[:p, :p],
        "BR": grid_2p[:p, p:],
        "TL": grid_2p[p:, :p],
        "TR": grid_2p[p:, p:],
    }
    # (b_base, a_base) for each quad
    bases = {"BL": (0, 0), "BR": (p, 0), "TL": (0, p), "TR": (p, p)}

    parts = []
    order = ["BL", "BR", "TL", "TR"]
    max_labels = {}

    for tag in order:
        q = quads[tag]
        fb, fa = dominant_freqs_ab(q)

        if fb != fa:
            return f"<b>skip: {tag} fb({fb}) != fa({fa})</b>", False

        f = int(fb)
        if (p % f) != 0:
            return f"<b>skip: {tag} p({p}) not divisible by f({f})</b>", False

        g = p // math.gcd(p, f)

        # extrema in quad
        amax_flat = int(np.argmax(q)); amin_flat = int(np.argmin(q))
        amax = np.unravel_index(amax_flat, q.shape)
        amin = np.unravel_index(amin_flat, q.shape)

        b_base, a_base= bases[tag]
        # shift based on quads
        max_label = (b_base + int(amax[1] % g), a_base + int(amax[0] % g))
        min_label = (b_base + int(amin[1] % g), a_base + int(amin[0] % g))
        
        max_labels[tag] = max_label
        parts.append(f"{tag} max:{max_label} min:{min_label}")
        
    cond1 = (max_labels["BL"][0] == max_labels["TL"][0])
    cond2 = (max_labels["BR"][0] == max_labels["TR"][0])
    cond3 = (max_labels["BL"][1] == max_labels["BR"][1])
    cond4 = (max_labels["TL"][1] == max_labels["TR"][1])
    all_true = cond1 and cond2 and cond3 and cond4

    if all_true:
        parts.append("ALL_TRUE")
    header = " | ".join(parts)
    return f"<b>{header}</b>", True

##### helpers for drawing neuron distribution by a,b mod g
def quad_set_extrema_records_strict_f(
    grid_2p: np.ndarray,
    n: int,
    cayley: np.ndarray | None = None
) -> dict:
    
    grid_2p = np.asarray(grid_2p, dtype=float)
    G = grid_2p.shape[0]
    assert grid_2p.shape[1] == G, "grid must be square"
    p = G // 2

    quads = {
        "BL": grid_2p[:p, :p],
        "BR": grid_2p[:p, p:],
        "TL": grid_2p[p:, :p],
        "TR": grid_2p[p:, p:],
    }
    order = ["BL", "BR", "TL", "TR"]

    # 1) check if freqs are consistent
    f_list = []
    for tag in order:
        q = quads[tag]
        fb, fa = dominant_freqs_ab(q)
        if fb != fa:
            return dict(ok=False, reason=f"{tag}: fb({fb}) != fa({fa})", f=None, g=None, records=[])
        f_list.append(int(fb))

    uniq = sorted(set(f_list))
    if len(uniq) != 1:
        return dict(ok=False, reason=f"f varies across quads: {f_list} (uniq={uniq})", f=None, g=None, records=[])

    f = uniq[0]
    if (p % f) != 0:
        return dict(ok=False, reason=f"p({p}) not divisible by f({f})", f=None, g=None, records=[])

    g = p // math.gcd(p, f)
    bases = {"BL": (0, 0), "BR": (0, p), "TL": (p, 0), "TR": (p, p)}

    if g <= 1:
        return dict(ok=False, reason=f"g={g} (p={p}, f={f}) leaves no second-best residue", f=f, g=g, records=[])

    def second_best_mod_row(row: np.ndarray, g: int, primary_mod: int):
        mods = np.arange(row.size) % g
        best_val = -np.inf; best_mod = None
        for r in range(g):
            if r == primary_mod:  # exclude the main residue
                continue
            mask = (mods == r)
            if not np.any(mask):
                continue
            v = row[mask].max()
            if v > best_val:
                best_val = v; best_mod = r
        return best_mod, float(best_val)

    def second_best_mod_col(col: np.ndarray, g: int, primary_mod: int):
        mods = np.arange(col.size) % g
        best_val = -np.inf; best_mod = None
        for r in range(g):
            if r == primary_mod:
                continue
            mask = (mods == r)
            if not np.any(mask):
                continue
            v = col[mask].max()
            if v > best_val:
                best_val = v; best_mod = r
        return best_mod, float(best_val)
    
    # 2) generate records
    records = []
    for tag in order:
        q = quads[tag]

        # extrema
        amax_flat = int(np.argmax(q))
        a_idx, b_idx = np.unravel_index(amax_flat, q.shape)
        max_val = float(q[a_idx, b_idx])

        a_mod0 = int(a_idx % g)
        b_mod0 = int(b_idx % g)

        rec = {
            "n": int(n),
            "quad": tag,
            "a_mod": a_mod0,
            "b_mod": b_mod0,
            "max_act": max_val,
        }

        if cayley is not None:
            a_base, b_base = bases[tag] # remap the index within quads to 0..2p-1
            a_abs = a_idx + a_base
            b_abs = b_idx + b_base
            c_idx = int(cayley[a_abs, b_abs])
            rec["c_idx"]   = c_idx
            rec["c_mod"]   = int(c_idx % g)
            rec["c_upper"] = bool(c_idx >= p) 

        # sec_max_a_fx：fix row a=a*, find second large b
        if q.shape[1] > 1:
            b2_mod, val2 = second_best_mod_row(q[a_idx, :], g, b_mod0)
            if b2_mod is not None:
                rec["sec_max_a_fx"] = {
                    "a_mod": a_mod0,
                    "b_mod": int(b2_mod), 
                    "val":  float(val2),
                }

        # sec_max_b_fx：fix column b=b*，find second large a
        if q.shape[0] > 1:
            a2_mod, val3 = second_best_mod_col(q[:, b_idx], g, a_mod0)
            if a2_mod is not None:
                rec["sec_max_b_fx"] = {
                    "a_mod": int(a2_mod),
                    "b_mod": b_mod0,
                    "val":  float(val3),
                }

        records.append(rec)

    return dict(ok=True, reason=None, f=f, g=g, records=records)

def pick_c_records(records):
    """
    extract records for c (only for extrema): return [{"n","quad","c_mod","c_upper","max_act"}, ...]
    """
    out = []
    for r in records:
        if "c_mod" in r:
            out.append({
                "n": r["n"], "quad": r["quad"],
                "c_mod": int(r["c_mod"]),
                "c_upper": bool(r.get("c_upper", False)),
                "max_act": float(r.get("max_act", 0.0)),
            })
    return out

import numpy as np
import plotly.graph_objects as go
from math import gcd, pi, cos, sin
from collections import defaultdict

# ---------- compute g-gon vertex  ----------
def _ngon_vertices(g: int, R: float, cx: float, cy: float, phi0: float):
    """
    (cx,cy) as center, radius R, starting angle phi0 (rad), gives g's coord counter-clockwisely.
    vertex order corresponds to residue 0..g-1's mapping
    """
    angs = [phi0 + 2 * pi * k / g for k in range(g)]
    xs = [cx + R * cos(a) for a in angs]
    ys = [cy + R * sin(a) for a in angs]
    return np.array(xs), np.array(ys)

def pick_records_for_mode(records, mode: str):
    """
    mode ∈ {"max","sec_a","sec_b"}
    output structure same as plot_coset_ngon_ring expected: including a_mod,b_mod,max_act,n,quad
    """
    out = []
    for r in records:
        if mode == "max":
            out.append({
                "n": r["n"], "quad": r["quad"],
                "a_mod": r["a_mod"], "b_mod": r["b_mod"], "max_act": r["max_act"],
            })
        elif mode == "sec_a":
            s = r.get("sec_max_a_fx")
            if s is not None:
                out.append({
                    "n": r["n"], "quad": r["quad"],
                    "a_mod": s["a_mod"], "b_mod": s["b_mod"], "max_act": s["val"],
                })
        elif mode == "sec_b":
            s = r.get("sec_max_b_fx")
            if s is not None:
                out.append({
                    "n": r["n"], "quad": r["quad"],
                    "a_mod": s["a_mod"], "b_mod": s["b_mod"], "max_act": s["val"],
                })
        else:
            raise ValueError("mode must be 'max' | 'sec_a' | 'sec_b'")
    return out

# ---------- ngon ring ----------
def plot_coset_ngon_ring(
    p: int,
    f: int,
    records: list,           # [{"n","quad","a_mod","b_mod","max_act"}, ...]  (a_mod,b_mod in 0..g-1)
    neuron_main: set,        # only plot these neurons
    title: str | None = None,
    R_outer: float = 1.0,
    r_inner: float = 0.26,
    ring_cap: int = 6,       # max point for each ring
    ring_step: float = 0.045,# ring distance（the bigger the farther between rings）
    text_size: int = 12,     # index size
    show_labels: bool = True,
    rotate_random: bool = False,
    seed: int | None = None
) -> go.Figure:
    """
    g-gon visualization:
      - outer layer: a<p (red) and a≥p (green) two g-gons, stagger by π/g;
      - inner layer: insert b<p and b≥p two g-gons for every vertex in the outer layer, stagger by π/g;
      - neurons with same (a_mod, a_upper, b_mod, b_upper) vertex layed in rings.
      - only show neuron index (pure text),no marker。
    """
    g = p // gcd(p, f)
    if g < 3:
        raise ValueError(f"g must be >=3; got g={g} (p={p}, f={f})")

    rng = np.random.default_rng(seed) if rotate_random else None

    if rotate_random:
        phi0_a_low = float(rng.uniform(0, 2*pi))
        phi0_b_low = float(rng.uniform(0, 2*pi))
    else:
        phi0_a_low = pi/2     # a<p starting angle
        phi0_b_low = 0.0      # b<p sarting angle

    phi0_a_up  = phi0_a_low + pi / g   # a≥p
    phi0_b_up  = phi0_b_low + pi / g   # b≥p

    XaL, YaL = _ngon_vertices(g, R_outer, 0.0, 0.0, phi0_a_low)  # a<p (red)
    XaU, YaU = _ngon_vertices(g, R_outer, 0.0, 0.0, phi0_a_up)   # a≥p (green)

    fig = go.Figure()

    # outer layer
    fig.add_trace(go.Scatter(
        x=np.r_[XaL, XaL[:1]], y=np.r_[YaL, YaL[:1]],
        mode="lines", line=dict(width=2, color="red"),
        hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=np.r_[XaU, XaU[:1]], y=np.r_[YaU, YaU[:1]],
        mode="lines", line=dict(width=2, color="green"),
        hoverinfo="skip", showlegend=False
    ))

    # labels for outer layer a vertices
    if show_labels:
        for r in range(g):
            fig.add_trace(go.Scatter(
                x=[XaL[r]], y=[YaL[r]], mode="text",
                text=[f"a≡{r} (mod {g})"],
                textfont=dict(size=12, color="red"),
                hoverinfo="skip", showlegend=False
            ))
        for r in range(g):
            fig.add_trace(go.Scatter(
                x=[XaU[r]], y=[YaU[r]], mode="text",
                text=[f"a≡{p+r} (≡{r})"],
                textfont=dict(size=12, color="green"),
                hoverinfo="skip", showlegend=False
            ))

    def _draw_inner_b_at(cx, cy):
        XbL, YbL = _ngon_vertices(g, r_inner, cx, cy, phi0_b_low)  # b<p
        XbU, YbU = _ngon_vertices(g, r_inner, cx, cy, phi0_b_up)   # b≥p
        fig.add_trace(go.Scatter(
            x=np.r_[XbL, XbL[:1]], y=np.r_[YbL, YbL[:1]],
            mode="lines", line=dict(width=1, color="rgba(0,0,0,0.25)"),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=np.r_[XbU, XbU[:1]], y=np.r_[YbU, YbU[:1]],
            mode="lines", line=dict(width=1, color="rgba(0,0,0,0.25)"),
            hoverinfo="skip", showlegend=False
        ))
        if show_labels:
            for r in range(g):
                fig.add_trace(go.Scatter(
                    x=[XbL[r]], y=[YbL[r]], mode="text",
                    text=[f"b≡{r}"], textfont=dict(size=10),
                    hoverinfo="skip", showlegend=False
                ))
            for r in range(g):
                fig.add_trace(go.Scatter(
                    x=[XbU[r]], y=[YbU[r]], mode="text",
                    text=[f"b≡{p+r}(≡{r})"], textfont=dict(size=10),
                    hoverinfo="skip", showlegend=False
                ))
        return (XbL, YbL, XbU, YbU)

    # key: ("aL"/"aU", r) → (XbL, YbL, XbU, YbU)
    inner_cache = {}
    for r in range(g):
        inner_cache[("aL", r)] = _draw_inner_b_at(XaL[r], YaL[r])
    for r in range(g):
        inner_cache[("aU", r)] = _draw_inner_b_at(XaU[r], YaU[r])

    clusters = defaultdict(list)  # key → list of {"n","max","center":(x,y)}
    def halves_from_quad(quad):
        quad = quad.upper()
        a_upper = quad in ("TL", "TR")
        b_upper = quad in ("BR", "TR")
        return a_upper, b_upper

    for rec in records:
        if rec["n"] not in neuron_main:
            continue
        n = int(rec["n"])
        a_mod = int(rec["a_mod"]) % g
        b_mod = int(rec["b_mod"]) % g
        max_act = float(rec.get("max_act", 0.0))
        a_upper, b_upper = halves_from_quad(rec["quad"])
        a_key = ("aU", a_mod) if a_upper else ("aL", a_mod)
        XbL, YbL, XbU, YbU = inner_cache[a_key]
        
        if b_upper:
            cx, cy = XbU[b_mod], YbU[b_mod]
        else:
            cx, cy = XbL[b_mod], YbL[b_mod]
        clusters[(a_key, ("bU" if b_upper else "bL"), b_mod)].append(
            {"n": n, "max": max_act, "center": (cx, cy)}
        )

    # —— ring arrangement
    Xs, Ys, texts = [], [], []
    for key, items in clusters.items():
        items_sorted = sorted(items, key=lambda d: (-d["max"], d["n"]))
        for i, d in enumerate(items_sorted):
            ring = i // ring_cap
            pos  = i % ring_cap
            ang  = 2 * pi * pos / max(1, ring_cap)
            rad  = ring_step * (ring + 1)
            cx, cy = d["center"]
            Xs.append(cx + rad * cos(ang))
            Ys.append(cy + rad * sin(ang))
            texts.append(str(d["n"]))  # only show index number

    if Xs:
        fig.add_trace(go.Scatter(
            x=Xs, y=Ys,
            mode="text",                    
            text=texts,
            textposition="top center",
            textfont=dict(size=text_size, color="#222"),
            hoverinfo="skip",               
            showlegend=False
        ))

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title= title or f"Coset n-gon (g={g}) — p={p}, f={f}, g={g}",
        width=900, height=900,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False
    )
    return fig

def plot_coset_ngon_c_single(
    p: int,
    f: int,
    c_records: list,        # [{"n","quad","c_mod","c_upper","max_act"}]
    neuron_main: set,
    title: str | None = None,
    R_outer: float = 1.0,
    ring_cap: int = 6,
    ring_step: float = 0.05,
    text_size: int = 12,
    show_labels: bool = True,
    rotate_random: bool = False,
    seed: int | None = None
) -> go.Figure:
    g = p // gcd(p, f)
    if g < 3:
        raise ValueError(f"g must be >=3; got g={g} (p={p}, f={f})")
    rng = np.random.default_rng(seed) if rotate_random else None
    phi0_low  = (rng.uniform(0, 2*np.pi) if rotate_random else np.pi/2)
    phi0_high = phi0_low + np.pi / g

    Xlow, Ylow = _ngon_vertices(g, R_outer, 0.0, 0.0, phi0_low)   # c<p
    Xhigh,Yhigh= _ngon_vertices(g, R_outer, 0.0, 0.0, phi0_high)  # c≥p

    fig = go.Figure()
    # two layers
    fig.add_trace(go.Scatter(x=np.r_[Xlow, Xlow[:1]],  y=np.r_[Ylow, Ylow[:1]],
                             mode="lines", line=dict(width=2, color="red"),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=np.r_[Xhigh, Xhigh[:1]],y=np.r_[Yhigh,Yhigh[:1]],
                             mode="lines", line=dict(width=2, color="green"),
                             hoverinfo="skip", showlegend=False))
    # labels for vertices
    if show_labels:
        for r in range(g):
            fig.add_trace(go.Scatter(x=[Xlow[r]],  y=[Ylow[r]],  mode="text",
                                     text=[f"c≡{r} (mod {g})"], textfont=dict(size=12, color="red"),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=[Xhigh[r]], y=[Yhigh[r]], mode="text",
                                     text=[f"c≡{p+r}(≡{r})"], textfont=dict(size=12, color="green"),
                                     hoverinfo="skip", showlegend=False))
    # cluster and ring arrangement
    clusters = defaultdict(list)   # key → list of {"n","max","center":(x,y)}
    for rec in c_records:
        if rec["n"] not in neuron_main:
            continue
        n = int(rec["n"])
        r = int(rec["c_mod"]) % g
        center = (Xhigh[r], Yhigh[r]) if rec["c_upper"] else (Xlow[r], Ylow[r])
        clusters[(r, rec["c_upper"])].append({"n": n, "max": float(rec["max_act"]), "center": center})

    Xs, Ys, texts = [], [], []
    for key, items in clusters.items():
        items_sorted = sorted(items, key=lambda d: (-d["max"], d["n"]))
        for i, d in enumerate(items_sorted):
            ring = i // ring_cap
            pos  = i % ring_cap
            ang  = 2*np.pi * pos / max(1, ring_cap)
            rad  = ring_step * (ring + 1)
            cx, cy = d["center"]
            Xs.append(cx + rad*np.cos(ang))
            Ys.append(cy + rad*np.sin(ang))
            texts.append(str(d["n"]))

    if Xs:
        fig.add_trace(go.Scatter(x=Xs, y=Ys, mode="text", text=texts,
                                 textposition="top center", textfont=dict(size=text_size, color="#222"),
                                 hoverinfo="skip", showlegend=False))

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title= title or f"c n-gon (single layer) — p={p}, f={f}, g={g}",
        width=900, height=900, margin=dict(l=40, r=40, t=60, b=40), showlegend=False
    )
    return fig


def single_neuron_figure(n,                      # ← neuron index
                     pre_grid,               # (G,G,N)       whole grid
                     left_vec, # (G,N), x, 1‑D branch
                     right_vec,    # (G,N),y, 1‑D branch
                     F_full, F_L, F_R,       # DFT dicts
                     names,                  # list of irrep labels
                     subgroup_info):         # dict {'L':…, 'R':…}
    G, D = pre_grid.shape[0], len(names)
    fig = make_subplots(
        rows=4, cols=4,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        specs=[
            [{"type":"heatmap"}, {"type":"xy"}, {"type":"xy"}, {"type":"xy"}],
            [{"type":"heatmap"}, {"type":"heatmap"}, {"type":"heatmap"}, None],
            [{"type":"xy"}, {"type":"xy"}, {"type":"xy"}, {"type":"xy"}],
            [{"type":"table","colspan":4}, None, None, None],
        ],
        subplot_titles=[
            # row‑1
            f"Whole-RAW (n{n})", f"a-RAW-n{n}, max={jnp.max(left_vec[:, n]):.2f}", f"b-RAW-n{n},max={jnp.max(right_vec[:, n]):.2f}", f"Quad-REMAP (n{n})",
            # row‑2
            f"Whole-DFT (n{n})", f"a-DFT (n{n})", f"b-DFT (n{n})",
            # row‑3
            "Quad-BL", "Quad-BR", "Quad-TL", "Quad-TR",
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )
    header, ok = _quad_set_extrema_header(pre_grid[:, :, n])
    if ok:
        fig.add_annotation(
            text=header,
            xref="paper", yref="paper",
            x=0.5, y=1.15,
            showarrow=False,
            font=dict(size=12),
            align="center"
        )
    # ───────────────────── ① Whole‐RAW heat‑map ────────────────────────────
    fig.add_trace(
        go.Heatmap(z=pre_grid[:, :, n], coloraxis="coloraxis1"),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Right input (b)", row=1, col=1)
    fig.update_yaxes(title_text="Left input (a)", row=1, col=1)
    # ───────────────────── ② Left / Right 1‑D lines ────────────────────────
    x_ticks = list(range(G))
    fig.add_trace(
        go.Scatter(x=x_ticks, y=left_vec[:, n],
                   mode="lines+markers",
                   line=dict(width=1), marker=dict(size=4),
                   showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=x_ticks, y=right_vec[:, n],
                   mode="lines+markers",
                   line=dict(width=1), marker=dict(size=4),
                   showlegend=False),
        row=1, col=3
    )
    for i in [2, 3]:
        fig.update_xaxes(showgrid=True, row=1, col=i)
        fig.update_yaxes(showgrid=True, row=1, col=i)
    
    remap_img, fb, fa = _remapped_quadrants(pre_grid[:, :, n])
    fig.add_trace(
        go.Heatmap(z=remap_img, colorscale="Viridis", showscale=False),
        row=1, col=4
    )
    fig.update_xaxes(title_text=f"b (remapped) by {fb}", row=1, col=4)
    fig.update_yaxes(title_text=f"a (remapped) by {fa}", row=1, col=4)
    # ───────────────────── ③ three DFT heat‑maps ───────────────────────────
    for col, Fhat in enumerate([F_full, F_L, F_R], start=1):
        P = np.zeros((D, D))
        for (r, s, idx), M in Fhat.items():
            if idx == n:
                P[names.index(r), names.index(s)] = np.linalg.norm(np.array(M))
        fig.add_trace(
            go.Heatmap(
                z=P, x=names, y=names,
                coloraxis="coloraxis2"
            ),
            row=2, col=col
        )

    quad_phase = _quadrant_ab_phases(pre_grid[:, :, n])
    axis_range = [0, 2*np.pi]
    tick_vals  = [0, np.pi, 2*np.pi]
    tick_text  = ["0", "π", "2π"]
    for idx, (phib, phia) in enumerate(quad_phase):
        fig.add_trace(
            go.Scatter(x=[phib], y=[phia],
                    mode="markers+text",
                    marker=dict(size=8, color="red"),
                    text=[f"{phib:.2f},{phia:.2f}"], 
                    textposition="top center",
                    textfont=dict(size=10),
                    cliponaxis=False,
                    showlegend=False),
            row=3, col=idx+1
        )
        fig.update_xaxes(range=axis_range,
                     tickvals=tick_vals, ticktext=tick_text,
                     title_text="φ_b(rad)",
                     row=3, col=idx+1)
        fig.update_yaxes(range=axis_range,
                        tickvals=tick_vals, ticktext=tick_text,
                        title_text="φ_a (rad)",
                        row=3, col=idx+1)
    # ───────────────────── ④ footer: subgroup scores ───────────────────────
    # --- Left vec two set of cosets + mix top-3 ---
    lineL_LC  = ", ".join(f"{h}:{s['Cbar']:.2e}" for h, s in subgroup_info['L']['Lcoset'].items())
    lineL_RC  = ", ".join(f"{h}:{s['Cbar']:.2e}" for h, s in subgroup_info['L']['Rcoset'].items())
    lineL_mix = ", ".join(f"{d['origin']}·{d['H']}:{d['Cbar']:.2e}" for d in subgroup_info['L'].get('mix3', []))

    # --- Right vec two set of cosets + mix top-3 ---
    lineR_LC  = ", ".join(f"{h}:{s['Cbar']:.2e}" for h, s in subgroup_info['R']['Lcoset'].items())
    lineR_RC  = ", ".join(f"{h}:{s['Cbar']:.2e}" for h, s in subgroup_info['R']['Rcoset'].items())
    lineR_mix = ", ".join(f"{d['origin']}·{d['H']}:{d['Cbar']:.2e}" for d in subgroup_info['R'].get('mix3', []))

    headers = ["L vec — L-coset", "L vec — R-coset", "L vec — mix-Top3",
           "R vec — L-coset", "R vec — R-coset", "R vec — mix-Top3"]
    cells = [[lineL_LC], [lineL_RC], [lineL_mix],
            [lineR_LC], [lineR_RC], [lineR_mix]]

    fig.add_trace(go.Table(
        header=dict(values=headers, align="center"),
        cells=dict(values=cells, align="left"),
        columnwidth=[0.16, 0.16, 0.18, 0.16, 0.16, 0.18]
    ), row=4, col=1)
    # lineL = ", ".join(f"{h}:{s['Cbar']:.2e}" for h, s in subgroup_info['L'].items())
    # lineR = ", ".join(f"{h}:{s['Cbar']:.2e}" for h, s in subgroup_info['R'].items())


    # caption = (f"<b>Neuron {n} - top-3 subgroup scores</b><br>"
    #            f"<span style='color:#1f77b4'>L</span> → {lineL}<br>"
    #            f"<span style='color:#ff7f0e'>R</span> → {lineR}")
    # fig.add_annotation(
    #     text=caption,
    #     showarrow=False,
    #     xref="paper", yref="paper",
    #     x=0.5, y=-0.14,
    #     font=dict(size=10),
    #     align="center"
    # )

    # ───────────────────── ⑤ layout / shared colorbars ─────────────────────

    for r in range(1, 3):        # 1‥2 row
        for c in range(1, 5):    # 1‥4 col
            # some units are None, which Plotly will ignore
            fig.update_xaxes(showgrid=False, zeroline=False, row=r, col=c)
            fig.update_yaxes(showgrid=False, zeroline=False, row=r, col=c)

    # ── 1. heatmap strictly square ────────────────────────────────
    for (r, c) in [(1,1), (1,4), (2,1), (2,2), (2,3),
                (3,1), (3,2), (3,3), (3,4)]:
        axis_name = f"x{(r-1)*4 + c}"
        fig.update_yaxes(scaleanchor=axis_name, scaleratio=1, row=r, col=c)

    # ── 2. colorbar ─────────────────────────────
    fig.update_layout(
        margin=dict(t=30, b=30, l=80, r=50),
        width=900, height=900,
        autosize=False,

        # colorbar for RAW (first row only)
        coloraxis1=dict(
            colorscale="Viridis",
            colorbar=dict(
                title="RAW",
                len=0.25,         
                y=0.88,         
                yanchor="middle",
                x=1.02, 
                xanchor="left"
            )
        ),

        # colorbar for DFT (second row only)
        coloraxis2=dict(
            colorscale="Viridis",
            colorbar=dict(
                title="GFT",
                len=0.25, 
                y=0.53,  
                yanchor="middle",
                x=1.02,
                xanchor="left"
            )
        )
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)

    return fig

def _two_stage_kmeans_prune(
    log_mp: np.ndarray,
    thresh1: float = 2.0,   
    thresh2: float = 2.0,  
    seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    
    assert log_mp.ndim == 1
    x = log_mp.reshape(-1, 1)

    n = x.shape[0]

    # ---------- Degenerate sizes ----------
    if n == 0:
        # empty set
        return np.array([], dtype=int), np.array([], dtype=int)
    if n == 1:
        # only one element
        return np.array([0], dtype=int), np.array([], dtype=int)

    # ---------- Stage 1 ----------
    km1 = KMeans(n_clusters=2, n_init='auto', random_state=seed)
    lab1 = km1.fit_predict(x)
    centers1 = km1.cluster_centers_.ravel()
    hi1 = int(np.argmax(centers1))
    gap1 = float(abs(centers1[0] - centers1[1]))

    if gap1 >= float(thresh1):
        keep1 = np.flatnonzero(lab1 == hi1)
    else:
        keep1 = np.arange(x.size, dtype=int)

    # ---------- Stage 2 ----------
    keep2 = keep1
    if keep1.size >= 2:
        x2 = x[keep1]
        km2 = KMeans(n_clusters=2, n_init='auto', random_state=seed+1)
        lab2 = km2.fit_predict(x2)
        centers2 = km2.cluster_centers_.ravel()
        hi2 = int(np.argmax(centers2))
        gap2 = float(abs(centers2[0] - centers2[1]))

        if gap2 >= float(thresh2):
            rel_keep2 = np.flatnonzero(lab2 == hi2)
            keep2 = keep1[rel_keep2]
        

    # ---------- Safety ----------
    if keep2.size == 0:
        keep2 = np.array([int(np.argmax(x))], dtype=int)

    # ---------- drop  ----------
    all_idx = np.arange(x.size, dtype=int)
    drop = np.setdiff1d(all_idx, keep2, assume_unique=False)
    return keep2, drop


def prepare_layer_artifacts(pre_grid, #(G, G, N)
                            left, right, #(G*G, N)
                            dft_2d, irreps, freq_map,
                            strict=True,
                            prune_cfg: dict | None = None,
                            store_full_neuron_grids: bool = False,
                            ):
    """
    Do DFT, and cluster based on dominant_irrep, return all the artifacts needed later.
    """
    G, N = pre_grid.shape[0], pre_grid.shape[-1]

    # 1) DFT once
    flat_all = pre_grid.reshape(G*G, N)
    F_full   = dft_2d(flat_all)
    F_L      = dft_2d(left)
    F_R      = dft_2d(right)

    # 2) cluster
    names = [lab for lab, _, _,_ in irreps]
    irrep2neurons = defaultdict(list)
    freq_cluster  = defaultdict(list) 
    diag_labels   = set()       
    neuron2pair   = {}

    freq_clust_2d = {
        "diag":      defaultdict(list),
        "row":       defaultdict(list),    
        "col":       defaultdict(list),    
        "pair_axis": defaultdict(list),    
    }

    neuron_data = {}
    for n in range(N):
        Fhat_n = {k: v for k, v in F_full.items() if k[2] == n}
        dom = _classify_by_gft(Fhat_n, names, freq_map, strict=strict)
        r_star, s_star = dom["r_star"], dom["s_star"]
        irrep2neurons[(r_star, s_star)].append(n)
        neuron2pair[n] = (r_star, s_star)
        if r_star == s_star:
            if r_star not in freq_map:
                if strict:
                    raise KeyError(f"freq_map no mapping for irrep '{r_star}'")
                else:
                    # skip unknown label
                    continue
            f = int(freq_map[r_star])
            freq_cluster[f].append(int(n))  
            
            diag_labels.add((r_star, dom["kind"]))
        
        if dom["kind"] == "diag" and dom["fa"] is not None:
            freq_clust_2d["diag"][dom["fa"]].append(int(n))
        else:
            if dom["fa"] is not None:
                freq_clust_2d["row"][dom["fa"]].append(int(n))
            if dom["fb"] is not None:
                freq_clust_2d["col"][dom["fb"]].append(int(n))
            if dom["fa"] is not None and dom["fb"] is not None:
                freq_clust_2d["pair_axis"][(int(dom["fa"]), int(dom["fb"]))].append(int(n))

        # 2d) neuron_data
        entry = {
            "a_values": np.arange(G, dtype=int),
            "b_values": np.arange(G, dtype=int),
            "dominant": dom,
        }
        if store_full_neuron_grids:
            grid = pre_grid[:, :, n]
            post = np.maximum(grid, 0.0)
            entry["real_preactivations"] = grid
            entry["postactivations"]     = post
        neuron_data[int(n)] = entry

    # remove duplicates
    for k, v in irrep2neurons.items():
        irrep2neurons[k] = list(dict.fromkeys(v))

    cluster_prune = {}   # (r,s) -> {"main": [global_n...], "drop": [global_n...], "per_neuron_log10": {n: val}}
    if prune_cfg is not None:
        t1 = float(prune_cfg.get("thresh1", 2.0))
        t2 = float(prune_cfg.get("thresh2", 2.0))
        seed = int(prune_cfg.get("seed", 0))

        G = pre_grid.shape[0]
        for (r, s), neuron_list in irrep2neurons.items():
            if r != s or len(neuron_list) == 0:
                continue
            # neuron's max preact（abs） in the cluster
            cluster_grid = pre_grid[:, :, neuron_list]    # (G,G,K)
            max_preacts = np.max(np.abs(cluster_grid), axis=(0, 1))  # (K,)
            log_mp = np.log10(max_preacts + 1e-20)
            keep_rel, drop_rel = _two_stage_kmeans_prune(log_mp, t1, t2, seed=seed)
            if keep_rel.size == 0:
                # fallback
                keep_rel = np.array([int(np.argmax(log_mp))], dtype=int)
                drop_rel = np.setdiff1d(np.arange(log_mp.size), keep_rel)

            main = [int(neuron_list[i]) for i in keep_rel]
            drop = [int(neuron_list[i]) for i in drop_rel]
            per_log = {int(neuron_list[i]): float(log_mp[i]) for i in range(log_mp.size)}
            cluster_prune[(r, s)] = {"main": main, "drop": drop, "per_neuron_log10": per_log}
        

    artifacts = {
        "F_full": F_full, "F_L": F_L, "F_R": F_R,
        "names": names,
        "irrep2neurons": irrep2neurons,
        "freq_cluster": freq_cluster,           
        "freq_clust_2d": freq_clust_2d, 
        "diag_labels": diag_labels,
        "neuron2pair": neuron2pair,
        "neuron_data": neuron_data, 
    }
    if prune_cfg is not None:
        artifacts["cluster_prune"] = cluster_prune
    return artifacts


def summarize_diag_labels(diag_labels: Iterable[str], p: int, names: List[str]) -> Dict[str, Any]:
    """
    group irreps based on the followings:
      - approx_coset:   2D_x and gcd(p, x) == 1
      - coset_2d:       2D_x and gcd(p, x)  > 1
      - coset_1d:       {sign, rp, srp}
      - kinds:          axis, diag
    return a JSON file, including details of grouping, counting, total number consistency, and approx_coset ratio.
    """
    diag_labels = set((str(label).strip(), str(kind).strip()) for label, kind in diag_labels)
    total_diag = len(diag_labels)
    label_kind_map = defaultdict(set)
    for label, kind in diag_labels:
        label_kind_map[label].add(kind)

    names_1d = {"sign", "rp", "srp"}

    pat_2d = re.compile(r"^2D[_\-]?(\d+)$", flags=re.IGNORECASE)

    approx_coset = []  # 2D_x & gcd(p,x)==1
    coset_2d     = []  # 2D_x & gcd(p,x)>1
    coset_1d     = []  # sign/rp/srp
    others       = [] 

    for label, kinds in label_kind_map.items():
        m = pat_2d.match(label)
        kinds_list = sorted(kinds)

        if m:
            f = int(m.group(1))
            gcd_pf = math.gcd(p, f)
            item = {"label": label, "kinds": kinds_list, "f": f, "gcd_pf": gcd_pf}
            if gcd_pf == 1:
                approx_coset.append(item)
            else:
                coset_2d.append(item)
        elif label.lower() in names_1d:
            coset_1d.append({"label": label, "kinds": kinds_list})
        else:
            others.append({"label": label, "kinds": kinds_list})

    total_diag = len(label_kind_map)

    counts = {
        "approx_coset": len(approx_coset),
        "coset_2d":     len(coset_2d),
        "coset_1d":     len(coset_1d),
        "others":       len(others),
        "total_diag":   total_diag,
    }

    # Consistency check
    sum_all = counts["approx_coset"] + counts["coset_2d"] + counts["coset_1d"] + counts["others"]
    consistency_ok = (sum_all == total_diag)

    # Collect all processed labels to check consistency
    processed_labels = set()
    for group in [approx_coset, coset_2d, coset_1d, others]:
        for item in group:
            processed_labels.add(item["label"])

    missing_labels = []
    if not consistency_ok:
        missing_labels = list(sorted(set(label_kind_map.keys()) - processed_labels))

    # Approx ratio
    approx_ratio = (counts["approx_coset"] / total_diag) if total_diag > 0 else 0.0

    summary = {
        "p": p,
        "names": names,
        "items": {
            "approx_coset": approx_coset,
            "coset_2d":     coset_2d,
            "coset_1d":     coset_1d,
            "others":       others,
        },
        "counts": counts,
        "consistency": {
            "ok": consistency_ok,
            "sum_all": sum_all,
            "expected_total": total_diag,
            "missing_when_not_ok": missing_labels,
        },
        "approx_ratio_in_diag": approx_ratio,
    }

    return summary
    
    # for lab in diag_labels:
    #     s = lab.strip()
    #     m = pat_2d.match(s)
    #     if m:
    #         f = int(m.group(1))
    #         if math.gcd(p, f) == 1:
    #             approx_coset.append({"label": s, "f": f, "gcd_pf": 1})
    #         else:
    #             coset_2d.append({"label": s, "f": f, "gcd_pf": math.gcd(p, f)})
    #     elif s.lower() in names_1d:
    #         coset_1d.append({"label": s})
    #     else:
    #         others.append({"label": s})

    # counts = {
    #     "approx_coset": len(approx_coset),
    #     "coset_2d":     len(coset_2d),
    #     "coset_1d":     len(coset_1d),
    #     "others":       len(others),
    #     "total_diag":   total_diag,
    # }

    # # consistency check：sum of 3 types + others == #diag_labels
    # sum_all = counts["approx_coset"] + counts["coset_2d"] + counts["coset_1d"] + counts["others"]
    # consistency_ok = (sum_all == total_diag)

    # # approx coset proportion
    # approx_ratio = (counts["approx_coset"] / total_diag) if total_diag > 0 else 0.0

    # summary = {
    #     "p": p,
    #     "names": names,
    #     "items": {
    #         "approx_coset": approx_coset,
    #         "coset_2d":     coset_2d,
    #         "coset_1d":     coset_1d,
    #         "others":       others,
    #     },
    #     "counts": counts,
    #     "consistency": {
    #         "ok": consistency_ok,
    #         "sum_all": sum_all,
    #         "expected_total": total_diag,
    #         "missing_when_not_ok": (
    #             [] if consistency_ok else
    #             list(sorted(diag_labels - set(x["label"] for grp in [approx_coset, coset_2d, coset_1d, others] for x in grp)))
    #         )
    #     },
    #     "approx_ratio_in_diag": approx_ratio,
    # }
    # return summary

# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def make_layer_report(pre_grid,                # (G,G,N)  float32
                      left, # (G*G,N)  float32
                      right, # (G*G,N)  float32
                      p, # p
                      dft_2d,    # callables from build_dft_wrappers
                      irreps, 
                      coset_masks_left,             # dict (H,cid) → bool(|G|,)
                      coset_masks_right,
                      save_dir   : str,
                      cluster_tau : float = 1e-3,
                      color_rule=None,
                      artifacts=None
                      ):
    G = pre_grid.shape[0]
    N   = pre_grid.shape[-1]
    p_rot = G // 2
    G_list, idx_map, cayley_table = dihedral.build_cayley_table(p_rot)

    # ===== 0.  prepare DFTs ==================================================
    if artifacts is None:
        artifacts = prepare_layer_artifacts(pre_grid, left, right, dft_2d, irreps)
    F_full = artifacts["F_full"]
    F_L    = artifacts["F_L"]
    F_R    = artifacts["F_R"]
    names  = artifacts["names"]
    irrep2neurons = artifacts["irrep2neurons"]

    cont_l = left.reshape(G, G, N) 
    left_vec = cont_l.mean(axis=1) 
    cont_r = right.reshape(G, G, N) 
    right_vec = cont_r.mean(axis=0) 
    coset_info = {}
    
    # ===== 2.  assert left/right give same grouping (optional) ===============
    # mismatch = []
    # for (r, s), group in irrep2neurons.items():
    #     left_ok, right_ok = [], []
    #     for n in group:
    #         Fhat_L_n = {k: v for k, v in F_L.items() if k[2] == n}
    #         Fhat_R_n = {k: v for k, v in F_R.items() if k[2] == n}
    #         rL, sL = dominant_irrep(Fhat_L_n, names)
    #         rR, sR = dominant_irrep(Fhat_R_n, names)
    #         if (rL, sL) == (r, 'triv'):
    #             left_ok.append(n)
    #         if (rR, sR) == ('triv', s):
    #             right_ok.append(n)

    #     if sorted(left_ok) != sorted(group) or sorted(right_ok) != sorted(group):
    #         mismatch.append((r, s))
    # if mismatch:
    #     print("⚠️  clusters disagree for", mismatch)

    # ===== 3.  top-3 subgroup concentration per neuron ==========================
    # for n in range(N):
    #     v_left  = left_vec[:, n]
    #     v_right = right_vec[:, n]
    #     v_whole = pre_grid[:, :, n].ravel()

    #     coset_info[n] = {
    #         "L": subgroup_scores(v_left,  coset_masks_left, top_k=3),
    #         "R": subgroup_scores(v_right, coset_masks_right, top_k=3),
    #     }

    #     if n < 10:   # print a preview
    #         print(f"neuron {n:3d} | "
    #               f"L {coset_info[n]['L']} | "
    #               f"R {coset_info[n]['R']}")
    for n in range(N):
        v_left  = left_vec[:, n]
        v_right = right_vec[:, n]

        # --- left_vec: left & right coset scoring ---
        L_LC_top = subgroup_scores(v_left,  coset_masks_left,  top_k=2)   # left_vec on left-coset
        L_RC_top = subgroup_scores(v_left,  coset_masks_right, top_k=2)   # left_vec on right-coset

        L_LC_all = subgroup_scores(v_left,  coset_masks_left,  top_k=None)
        L_RC_all = subgroup_scores(v_left,  coset_masks_right, top_k=None)
        L_mix3   = merge_topk_sources([("Lcoset", L_LC_all), ("Rcoset", L_RC_all)], top_k=3)

        # --- right_vec: same ---
        R_LC_top = subgroup_scores(v_right, coset_masks_left,  top_k=2)   # right_vec on left-coset
        R_RC_top = subgroup_scores(v_right, coset_masks_right, top_k=2)   # right_vec on right-coset

        R_LC_all = subgroup_scores(v_right, coset_masks_left,  top_k=None)
        R_RC_all = subgroup_scores(v_right, coset_masks_right, top_k=None)
        R_mix3   = merge_topk_sources([("Lcoset", R_LC_all), ("Rcoset", R_RC_all)], top_k=3)

        coset_info[n] = {
            "L": {              # ← left_vec
                "Lcoset": L_LC_top,
                "Rcoset": L_RC_top,
                "mix3":   L_mix3 
            },
            "R": {              # ← right_vec
                "Lcoset": R_LC_top,
                "Rcoset": R_RC_top,
                "mix3":   R_mix3
            }
        }

        if n < 10:
            print(f"neuron {n:3d} | "
                f"L[Lcoset]{L_LC_top} | L[Rcoset]{L_RC_top} | L-mix3 {L_mix3} | "
                f"R[Lcoset]{R_LC_top} | R[Rcoset]{R_RC_top} | R-mix3 {R_mix3}")

    # ===== 4.  build figures (Plotly) ========================================
    for (r, s), neuron_list in irrep2neurons.items():
        if r == s:
            writer = PdfWriter()

            cluster_acts = pre_grid[:, :, neuron_list]       # shape (G, G, len(neuron_list))
            max_act = cluster_acts.max()                    # max activation

            if artifacts is not None and "cluster_prune" in artifacts and (r, s) in artifacts["cluster_prune"]:
                _pack = artifacts["cluster_prune"][(r, s)]
                neuron_main = list(_pack["main"])
                neuron_drop = list(_pack["drop"])
                per_log = _pack["per_neuron_log10"]
            else:
                neuron_main = neuron_list
                neuron_drop = []
                
                cluster_grid = pre_grid[:, :, neuron_list]
                max_preacts = np.max(np.abs(cluster_grid), axis=(0, 1))
                per_log = {int(neuron_list[i]): float(np.log10(max_preacts[i] + 1e-20))
                           for i in range(len(neuron_list))}

            # ---- 1. cover page --------------------------------------------------
            cover = make_subplots(rows=1, cols=1)
            cover.add_annotation(
                text=(
                    f"<b>Cluster ({r},{s})</b><br>"
                    f"size = {len(neuron_list)}<br>"
                    f"max activation = {max_act:.2e}"
                ),
                xref="paper", yref="paper",
                x=0.5, y=0.6, showarrow=False,
                font=dict(size=24), align="center"
            )
            cover.update_xaxes(visible=False)
            cover.update_yaxes(visible=False)

            cover._uuid = uuid.uuid4().hex          
            pdf_cover = cover.to_image(format="pdf", engine="kaleido")
            reader    = PdfReader(io.BytesIO(pdf_cover))
            writer.add_page(reader.pages[0])
            # ---- 2. phase plot --------------------------------------------------
            quad_labels = ["Quad-BL", "Quad-BR", "Quad-TL", "Quad-TR"]
            fig_quads = make_subplots(rows=2, cols=2,
                                    subplot_titles=quad_labels)

            palette = (px.colors.qualitative.Light24 + px.colors.qualitative.Dark24)   # 10  distinct colors
            num_colors = len(palette)

            axis_range = [0, 2 * np.pi]
            tick_vals  = [0, np.pi, 2*np.pi]
            tick_text  = ["0", "π", "2π"]

            for n_idx, n in enumerate(neuron_list):
                # quad phase
                quad_phase = _quadrant_ab_phases(pre_grid[:, :, n])   # [(φx, φy), …]
                color = palette[n_idx % num_colors]

                for q_idx, (φx, φy) in enumerate(quad_phase):
                    r_0, c_0 = divmod(q_idx, 2)
                    # only show leagend in BL subplot, in case legend is too long
                    show_legend = (q_idx == 0)
                    fig_quads.add_trace(
                        go.Scatter(
                            x=[φx], y=[φy],
                            mode="markers",
                            marker=dict(size=5, color=color),
                            name=f"neuron {n}",
                            showlegend=show_legend,
                        ),
                        row=r_0 + 1, col=c_0 + 1
                    )
        
                    if n_idx == 0:
                        fig_quads.update_xaxes(
                            title_text="φ_a (rad)",
                            range=axis_range, tickvals=tick_vals, ticktext=tick_text,
                            row=r_0 + 1, col=c_0 + 1
                        )
                        fig_quads.update_yaxes(
                            title_text="φ_b (rad)",
                            range=axis_range, tickvals=tick_vals, ticktext=tick_text,
                            row=r_0 + 1, col=c_0 + 1
                        )

            fig_quads._uuid = uuid.uuid4().hex
            pdf_bytes = fig_quads.to_image(format="pdf", engine="kaleido")
            reader    = PdfReader(io.BytesIO(pdf_bytes))
            writer.add_page(reader.pages[0])
            # ---- 2. merged phase plot --------------------------------------------------
            quad_labels = ["Quad-BL", "Quad-BR", "Quad-TL", "Quad-TR"]
            fig_quads_mer = make_subplots(rows=2, cols=2, subplot_titles=quad_labels)

            axis_range = [0, 2 * np.pi]
            tick_vals  = [0, np.pi, 2*np.pi]
            tick_text  = ["0", "π", "2π"]

            color = "red"

            # merged quads：[(x,y)] → [amp, count]
            merged_quads = [defaultdict(list) for _ in range(4)]

            for n in neuron_main:
                quad_phase = _quadrant_ab_phases(pre_grid[:, :, n])   # list of 4 tuples
                max_amp = np.abs(pre_grid[:, :, n]).max()
                
                for q_idx, (φx, φy) in enumerate(quad_phase):
                    key = (round(φx, 4), round(φy, 4))  # could change to tolerance-based bin
                    merged_quads[q_idx][key].append(max_amp)

            # quadrant wise
            for q_idx, phase_dict in enumerate(merged_quads):
                r_0, c_0 = divmod(q_idx, 2)
                for (φx, φy), amps in phase_dict.items():
                    sum_amp = sum(amps)
                    count = len(amps)
                    size = 6 + 3 * np.log1p(count)  # point size with count
                    fig_quads_mer.add_trace(
                        go.Scatter(
                            x=[φx], y=[φy],
                            mode="markers+text",
                            marker=dict(size=size, color=color),
                            text=[f"{sum_amp:.2f}"],
                            textposition="top center",
                            showlegend=False
                        ),
                        row=r_0 + 1, col=c_0 + 1
                    )
                    fig_quads_mer.update_xaxes(
                        title_text="φ_a (rad)",
                        range=axis_range, tickvals=tick_vals, ticktext=tick_text,
                        row=r_0 + 1, col=c_0 + 1
                    )
                    fig_quads_mer.update_yaxes(
                        title_text="φ_b (rad)",
                        range=axis_range, tickvals=tick_vals, ticktext=tick_text,
                        row=r_0 + 1, col=c_0 + 1
                    )

            fig_quads_mer._uuid = uuid.uuid4().hex
            pdf_bytes = fig_quads_mer.to_image(format="pdf", engine="kaleido")
            reader    = PdfReader(io.BytesIO(pdf_bytes))
            writer.add_page(reader.pages[0])
        
            # ── 3. draw log(max_preact) scatter and insert into PDF ──
            # ── log(max preact) scatter：main vs drop ──
            cluster_grid = pre_grid[:, :, neuron_list]
            log_mp_all = np.log10(np.max(np.abs(cluster_grid), axis=(0, 1)) + 1e-20)
            color_tag = np.array(["drop"] * len(neuron_list))
            idx_map = {n: i for i, n in enumerate(neuron_list)}
            for n in neuron_main:
                if n in idx_map: color_tag[idx_map[n]] = "main"

            fig_mp = px.scatter(
                x=np.arange(len(neuron_list)),
                y=log_mp_all,
                color=color_tag,
                labels=dict(x="Neuron index in cluster", y="log10(max pre-act)"),
                title=f"Cluster ({r},{s}) – log10(max pre-act): main vs drop"
            )
            fig_mp._uuid = uuid.uuid4().hex
            pdf_bytes = fig_mp.to_image(format="pdf", engine="kaleido")
            writer.add_page(PdfReader(io.BytesIO(pdf_bytes)).pages[0])

            ### neuron distribution by a,b mod g graph
            by_f = {}   # f -> {"records":[], "neurons":set()}
            skipped = []
            for n in neuron_main:
                res = quad_set_extrema_records_strict_f(pre_grid[:, :, n], n, cayley=cayley_table)
                if not res["ok"]:
                    skipped.append((n, res["reason"]))
                    continue
                f = res["f"]
                by_f.setdefault(f, {"records": [], "neurons": set()})
                by_f[f]["records"].extend(res["records"])
                by_f[f]["neurons"].add(n)

            pages_made = 0
            for f, pack in by_f.items():
                if not pack["records"]:
                    continue
                rec_max   = pick_records_for_mode(pack["records"], "max")
                rec_sec_a = pick_records_for_mode(pack["records"], "sec_a")
                rec_sec_b = pick_records_for_mode(pack["records"], "sec_b")
                rec_c     = pick_c_records(pack["records"])

                for mode, recs in [("max", rec_max), ("sec_a", rec_sec_a), ("sec_b", rec_sec_b)]:
                    if not recs:
                        continue
                    try:
                        fig = plot_coset_ngon_ring(
                            p=p, f=f,
                            records=recs,
                            neuron_main=pack["neurons"],
                            title=f"coset n-gon — p={p}, f={f}, g={p//math.gcd(p,f)}  [{mode}]",
                            r_inner=0.32,
                            show_labels=True
                        )

                        if skipped:
                            fig.add_annotation(
                                text=f"Skipped {len(skipped)} neuron(s) (inconsistent f)",
                                xref="paper", yref="paper", x=1.0, y=1.08,
                                xanchor="right", yanchor="bottom", showarrow=False, font=dict(size=10)
                            )

                        fig._uuid = uuid.uuid4().hex 
                        pdf_hex = fig.to_image(format="pdf", engine="kaleido")
                        writer.add_page(PdfReader(io.BytesIO(pdf_hex)).pages[0])
                        pages_made += 1

                    except ValueError as e:
                        # e.g. if g < 3, plot_coset_ngon_ring will raise error, skip in this case
                        print(f"[skip] f={f}, mode={mode}: {e}")
                        continue
                
                if rec_c:
                    try:
                        fig_c = plot_coset_ngon_c_single(
                            p=p, f=f,
                            c_records=rec_c,
                            neuron_main=pack["neurons"],
                            title=f"coset n-gon (c) — p={p}, f={f}, g={p//math.gcd(p,f)}"
                        )
                        fig_c._uuid = uuid.uuid4().hex
                        pdf_hex_c = fig_c.to_image(format="pdf", engine="kaleido")
                        writer.add_page(PdfReader(io.BytesIO(pdf_hex_c)).pages[0])
                        pages_made += 1
                    except ValueError as e:
                        print(f"[skip c] f={f}: {e}")

            if pages_made == 0:
                print(f"[hexagram] no pages generated for Cluster ({r},{s}); skipped={len(skipped)}; by_f empty.")


            ## PCA & Diffusions
            mat = pre_grid[:, :, neuron_main].reshape(G*G, -1).astype(float)

            # where to put the PDFs that the helper will create
            embed_dir = os.path.join(save_dir, f"cluster_{r}_{s}_embeds")
            os.makedirs(embed_dir, exist_ok=True)
            p = G // 2
            quads = {
                "BL": pre_grid[:p, :p, neuron_main],
                "BR": pre_grid[:p, p:, neuron_main],
                "TL": pre_grid[p:, :p, neuron_main],
                "TR": pre_grid[p:, p:, neuron_main],
            }
            quad_freq_lists = {}

            for tag_q, quad in quads.items():
                dom_freqs = set()
                num_neurons = quad.shape[-1]

                for i in range(num_neurons):
                    grid = quad[:, :, i]  # shape = (p, p)
                    fb, fa = dominant_freqs_ab(grid)
                    if fa == fb:
                        dom_freqs.add(fb)

                quad_freq_lists[tag_q] = sorted(list(dom_freqs))

            lists = list(quad_freq_lists.values())

            common_freqs = list(reduce(lambda a, b: set(a) & set(b), lists))

            common_freqs.sort()
            color_rules = color_rule
            quad_freq_lists["full"] = common_freqs
            # full grid first
            generate_pdf_plots_for_matrix(
                mat,               # (G², |cluster|)
                p=p,               # alphabet size for colouring
                save_dir=embed_dir,
                seed=f"{r}_{s}",
                freq_list=quad_freq_lists["full"],     # same colouring convention you used elsewhere
                tag="full",        # sub‑folder names
                tag_q = "full",
                class_string=f"cluster_{r}_{s}",
                colour_rule=color_rules,
                num_principal_components=4
            )

            # --- four quadrants --------------------------------------------------
            
            for tag_q, quad in quads.items():
                qmat = quad.reshape(p*p, -1).astype(float)
                generate_pdf_plots_for_matrix(
                    qmat, p=p, save_dir=embed_dir,
                    seed=f"{r}_{s}", 
                    freq_list=quad_freq_lists[tag_q],
                    tag=tag_q, tag_q=tag_q, class_string=f"{tag_q}_{r}_{s}",
                    colour_rule=color_rules,
                    num_principal_components=4
                )

            # # ---- attach the first page of the 3‑D PCA & diffusion PDFs ----------
            # uuid_tag = uuid.uuid4().hex[:6]
            # pca_pdf = os.path.join(embed_dir, "pca_pdf_plots", "3d", "full",
            #                     f"pca_seed_{r}_{s}_{uuid_tag}.pdf")
            # diff_pdf = os.path.join(embed_dir, "diffusion_pdf_plots", "3d", "full",
            #                         f"diff_seed_{r}_{s}_{uuid_tag}.pdf")
            # for pdf_path in [pca_pdf, diff_pdf]:
            #     if os.path.isfile(pdf_path):
            #         writer.add_page(PdfReader(pdf_path).pages[0])

            # ---- 2. each fig_n directly → PDF → add page -----------------------
            for n in neuron_main:
                print(f"▶ Rendering neuron {n} in cluster ({r},{s})")
                fig_n = single_neuron_figure(
                    n, pre_grid, left_vec, right_vec,
                    F_full, F_L, F_R, names, coset_info[n])
                
                fig_n._uuid = uuid.uuid4().hex
                pdf_bytes = fig_n.to_image(format="pdf", engine="kaleido")
                reader = PdfReader(io.BytesIO(pdf_bytes))
                page0  = reader.pages[0]
                writer.add_page(page0)

            if len(neuron_drop) > 0:
                rows = []
                for n in sorted(neuron_drop):
                    logv = per_log.get(int(n), None)
                    maxv = float(np.max(np.abs(pre_grid[:, :, n])))
                    rows.append((int(n), maxv, (None if logv is None else float(logv))))
                rows.sort(key=lambda t: (t[2] if t[2] is not None else -1e9), reverse=True)  # ascending order according to log 

                tbl = go.Figure(data=[go.Table(
                    header=dict(values=["neuron index", "max |preact|", "log10(max)"]),
                    cells=dict(values=[ [r[0] for r in rows],
                                        [f"{r[1]:.3g}" for r in rows],
                                        [("—" if r[2] is None else f"{r[2]:.3f}") for r in rows] ])
                )])
                tbl.update_layout(title=f"Cluster ({r},{s}) - Dropped neurons")
                tbl._uuid = uuid.uuid4().hex
                pdf_bytes = tbl.to_image(format="pdf", engine="kaleido")
                writer.add_page(PdfReader(io.BytesIO(pdf_bytes)).pages[0])
            # ---- 3. write final PDF --------------------------------------------
            path = f"cluster_{r}_{s}.pdf"
            fin_path = os.path.join(save_dir,path)
            with open(fin_path, "wb") as f:
                writer.write(f)
            print("✅ saved", fin_path)
        else:
            cluster_acts = pre_grid[:, :, neuron_list]
            max_act = cluster_acts.max()
            print(f"Skip {r}_{s} irreps plot, max activation: {max_act}")

# ============= NEW: 2x4 per-neuron page (independent from make_layer_report) =============

def _line_segments_from_vec(y: np.ndarray, p: int):
    """把长度 G=2p 的 1D 向量拆成两段 [0..p-1], [p..2p-1]，用于“p-1 与 p 之间不连线”的折线。"""
    G = y.shape[0]
    assert G == 2 * p, f"expect len=2p; got {G}, p={p}"
    x = np.arange(G)
    seg1 = (x[:p], y[:p])
    seg2 = (x[p:], y[p:])
    return seg1, seg2

def _remap_1d_by_freq(y: np.ndarray, freq: int, p: int) -> np.ndarray:
    """
    1D 重映射：out[(freq * a) % p] = in[a]（下半段），
               out[p + (freq * (a-p)) % p] = in[a]（上半段）
    未被映射到的位置填 NaN（而不是 0），这样折线图会自动“断开”。
    """
    G = y.shape[0]; assert G == 2 * p
    y = np.asarray(y, dtype=float)
    out = np.full(G, np.nan, dtype=float)   # 用 NaN 标记“没有点”的位置

    # 下半段 0..p-1
    for a in range(p):
        out[(freq * a) % p] = y[a]

    # 上半段 p..2p-1
    for a in range(p, 2 * p):
        ap = a - p
        out[p + (freq * ap) % p] = y[a]

    return out


def _nice_limit(max_abs: float, scale: int = 10) -> float:
    """把上界向上取整到 1/scale 的刻度，例如 scale=10 → 保留 1 位小数。"""
    if max_abs <= 0:
        return 1.0 / scale
    return math.ceil(float(max_abs) * scale) / scale

def _compute_fig_size_for_square_cells(
    rows:int, cols:int,
    hspace:float, vspace:float,  # 与 make_subplots 的 horizontal_spacing / vertical_spacing 一致（域比例）
    cell_px:int = 280,           # 每个子图绘图区的目标边长（像素）
    margins=dict(l=50, r=100, t=80, b=50)  # 右边给色条多留一点
):
    """
    给定 rows/cols 和 spacing（域比例），计算使每个子图绘图区近似正方的 fig 宽高。
    公式基于 domain：每格的宽度比例 = (1 - (cols-1)*hspace)/cols；高度类似。
    像素宽 = (W - l - r) * frac_w；像素高 = (H - t - b) * frac_h。
    我们令 像素高 = cell_px，解出 H；再令像素宽=像素高，解出 W。
    """
    frac_w = (1.0 - (cols - 1) * hspace) / cols
    frac_h = (1.0 - (rows - 1) * vspace) / rows

    H = margins["t"] + margins["b"] + cell_px / frac_h
    # 注意右边有色条，右边距给大一些；W 依赖 H 以保证单元宽高相等
    W = margins["l"] + margins["r"] + (cell_px / frac_w)
    return int(round(W)), int(round(H)), margins

# —— y 轴范围：按 max activation 自适应到 1/rounding_scale —— 
def _yr(v, scale=10):
        vmax = float(np.nanmax(np.abs(np.asarray(v, dtype=float)))) if np.any(np.isfinite(v)) else 0.0
        lim = _nice_limit(vmax, scale)
        return [-lim - 0.1, lim + 0.1]

def _per_neuron_fig_2x4(
    n: int,
    pre_grid: np.ndarray,   # (G,G,N)
    left_vec: np.ndarray,   # (G,N)  (a-branch 1D)
    right_vec: np.ndarray,  # (G,N)  (b-branch 1D)
    F_full, F_L, F_R,       # DFT dicts
    names: list[str],       # irrep 名称
    *,
    rounding_scale: int = 10            # ← “scale=10”，用于 y 轴上界取整
) -> go.Figure:
    G = pre_grid.shape[0]
    p = G // 2
    grid = pre_grid[:, :, n]
    a1d  = left_vec[:, n]
    b1d  = right_vec[:, n]

    # 用 BL 象限估计 fa, fb（保持与你 _remap_block 的口径一致）
    fb_bl, fa_bl = dominant_freqs_ab(grid[:p, :p])
    fa_q, fb_q = fa_bl, fb_bl

    # 1D remap（与 2D 里“乘 freq 再取 mod p”的规则一致）
    a1d_remap = _remap_1d_by_freq(a1d, fa_q, p)
    b1d_remap = _remap_1d_by_freq(b1d, fb_q, p)

    # 2D DFT power 矩阵
    D = len(names); name2idx = {lab: i for i, lab in enumerate(names)}
    P_a = np.zeros((D, D)); P_b = np.zeros((D, D)); P_full = np.zeros((D, D))
    for (r, s, idx), M in F_L.items():
        if idx == n: P_a[name2idx[r], name2idx[s]] = np.linalg.norm(np.array(M))
    for (r, s, idx), M in F_R.items():
        if idx == n: P_b[name2idx[r], name2idx[s]] = np.linalg.norm(np.array(M))
    for (r, s, idx), M in F_full.items():
        if idx == n: P_full[name2idx[r], name2idx[s]] = np.linalg.norm(np.array(M))

    # 原来是统一的 y_range，这里改成 4 个面板各自的范围：
    y_a   = _yr(a1d,        rounding_scale)
    y_b   = _yr(b1d,        rounding_scale)
    y_ar  = _yr(a1d_remap,  rounding_scale)
    y_br  = _yr(b1d_remap,  rounding_scale)
    hspace = 0.05
    vspace = 0.14
    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{"type":"xy"}, {"type":"xy"}, {"type":"heatmap"}, {"type":"heatmap"}],
               [{"type":"xy"}, {"type":"xy"}, {"type":"heatmap"}, {"type":"heatmap"}]],
        horizontal_spacing=hspace,
        vertical_spacing=vspace,
        subplot_titles=[
            f"a-contribution, max={np.max(a1d):.2f}",
            f"b-contribution, max={np.max(b1d):.2f}",
            "a-GFT",
            "b-GFT",
            f"a-remap, f={fa_q}",
            f"b-remap, f={fb_q}",
            "Whole Preactivation",
            "Whole-GFT",
        ],
    )
    # 全局文字放大（刻度、轴名等默认都会变大）
    fig.update_layout(font=dict(size=18))

    # 放大所有 subplot title，并稍微上移一点
    for ann in fig.layout.annotations:
        if ann.text in (
            f"a-contribution, max={np.max(a1d):.2f}",
            f"b-contribution, max={np.max(b1d):.2f}",
            "a-GFT",
            "b-GFT",
            f"a-remap, f={fa_q}",
            f"b-remap, f={fb_q}",
            "Whole Preactivation",
            "Whole-GFT"
        ):
            ann.font.size = 18      # ← 比默认大两号
            ann.yshift = 5          # ← 让标题离图再远一点（正值向上）
            ann.xshift = 0

    # 轴标题更贴近轴，且字体更大；刻度字体也更大
    fig.update_xaxes(title_standoff=2, title_font=dict(size=18), tickfont=dict(size=16))
    fig.update_yaxes(title_standoff=2, title_font=dict(size=18), tickfont=dict(size=16))

    def _add_1d_line(xy, row, col, y_range, *, with_grid: bool):
        (x1,y1),(x2,y2) = xy
        # 段1
        fig.add_trace(go.Scatter(
            x=x1, y=y1,
            mode="markers+lines",
            line=dict(width=1.5, color="black"),        # 线统一黑色
            marker=dict(size=6, color=y1, coloraxis="coloraxis2"),  # 点颜色跟数值走，挂到 coloraxis2
            connectgaps=True,
            showlegend=False
        ), row=row, col=col)
        # 段2
        fig.add_trace(go.Scatter(
            x=x2, y=y2,
            mode="markers+lines",
            line=dict(width=1.5, color="black"),
            marker=dict(size=6, color=y2, coloraxis="coloraxis2"),
            connectgaps=True,
            showlegend=False
        ), row=row, col=col)

        fig.update_xaxes(showgrid=with_grid, row=row, col=col)
        fig.update_yaxes(showgrid=with_grid, row=row, col=col, range=y_range)

    _add_1d_line(_line_segments_from_vec(a1d, p),       row=1, col=1, y_range=y_a,  with_grid=True)
    _add_1d_line(_line_segments_from_vec(b1d, p),       row=1, col=2, y_range=y_b,  with_grid=True)

    
    # 第1行右侧：a/b 的 2D DFT（无 grid）
    fig.add_trace(go.Heatmap(z=P_a, x=names, y=names, showscale=False, coloraxis="coloraxis1"), row=1, col=3)
    fig.add_trace(go.Heatmap(z=P_b, x=names, y=names, showscale=False, coloraxis="coloraxis1"), row=1, col=4)
    # a-DFT
    fig.update_xaxes(tickangle=45, dtick=1, row=1, col=3)
    # b-DFT（row=1,col=4）
    fig.update_xaxes(tickangle=45, dtick=1, row=1, col=4)
    # Whole-DFT（row=2,col=4）
    fig.update_xaxes(tickangle=45, dtick=1, row=2, col=4)

    fig.update_yaxes(dtick=1, row=1, col=3)
    fig.update_yaxes(dtick=1, row=1, col=4)
    fig.update_yaxes(dtick=1, row=2, col=4)


    # 第2行：a/b remap 1D（含 NaN 断点）
    _add_1d_line(_line_segments_from_vec(a1d_remap, p), row=2, col=1, y_range=y_ar, with_grid=True)
    _add_1d_line(_line_segments_from_vec(b1d_remap, p), row=2, col=2, y_range=y_br, with_grid=True)


    # 第2行右侧：2D preact & whole DFT（无 grid）
    fig.add_trace(go.Heatmap(z=grid, showscale=False, coloraxis="coloraxis2"), row=2, col=3)
    fig.add_trace(go.Heatmap(z=P_full, x=names, y=names, showscale=False, coloraxis="coloraxis1"), row=2, col=4)

    pre_min = float(np.nanmin(grid))
    pre_max = float(np.nanmax(grid))
    if pre_min == pre_max:
        pre_min -= 1e-9
        pre_max += 1e-9
        
    fig.update_layout(
        coloraxis1=dict(
            colorscale="Viridis",
            colorbar=dict(
                title="DFT",
                len=0.36,       # 短一些
                y=0.80,         # 上半行中点（两行等高时 ~0.75）
                yanchor="middle",
                x=1.02, xanchor="left",
                thickness=12
            )
        ),
        coloraxis2=dict(
            colorscale="Viridis",
            cmin=pre_min, 
            cmax=pre_max,
            colorbar=dict(
                title="Preactivation",
                len=0.36,
                y=0.25,         # 下半行中点
                yanchor="middle",
                x=1.02, xanchor="left",
                thickness=12
            )
        )
    )
    for (r,c) in [(1,3),(1,4),(2,3),(2,4)]:
        fig.update_xaxes(showgrid=False, row=r, col=c)
        fig.update_yaxes(showgrid=False, row=r, col=c)
    for (r,c) in [(1,1),(1,2),(2,1),(2,2)]:
        fig.update_xaxes(showgrid=True,  row=r, col=c, automargin=True)
        fig.update_yaxes(showgrid=True,  row=r, col=c, automargin=True)

    # ========== 只对“数值-数值”的 2D preactivation 强制等比（像素方形）==========
    # DFT 矩阵一般是分类轴，不设 scaleanchor；否则会警告或无效
    axis_name = f"x{(2-1)*4 + 3}"  # row=2, col=3 → x7
    fig.update_xaxes(constrain="domain", row=2, col=3)
    fig.update_yaxes(constrain="domain", row=2, col=3)
    fig.update_yaxes(scaleanchor=axis_name, scaleratio=1, row=2, col=3)

    # ========== 统一白底 ==========
    fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white")

    # ========== 计算“方格排版”的宽高，并应用 ==========
    W, H, margins = _compute_fig_size_for_square_cells(
        rows=2, cols=4, hspace=hspace, vspace=vspace,
        cell_px=250,  # 调这个数就能统一变大/变小；子图区域会保持近似正方
        margins=dict(l=50, r=110, t=80, b=50)  # 右边给色条多 10px
    )
    fig.update_layout(width=W, height=H, margin=margins, showlegend=False)
    return fig

def _tighten_cols_and_colorbars(fig, want_gap12=0.45, cb_dx=0.0, right_margin=60):
    """
    - 缩小第 1、2 列之间的距离（可选）
    - 把两个 colorbar 往图靠一点（通过 x 平移）
    - 调整右边距，避免被截断
    """
    # 1) 手动收紧第 1、2 列的域（可按需改；不需要就注释掉）
    # 下面演示把 col1 和 col2 的 domain 往中间各推 1%（根据 need 调）
    for ax_id in (1, 2):  # xaxis1, xaxis2
        xa = getattr(fig.layout, f"xaxis{ax_id}", None)
        if xa and xa.domain:
            d0, d1 = list(xa.domain)
            mid = 0.5 * (d0 + d1)
            width = (d1 - d0) * (1.0 - want_gap12*0.0)  # 这里给了接口，但默认不改宽度
            new = [mid - width/2, mid + width/2]
            getattr(fig.layout, f"xaxis{ax_id}").domain = new

    # 2) 平移 colorbar：对 coloraxis1 和 coloraxis2 分别处理
    def _shift_cb(coloraxis_name, dx):
        ca = getattr(fig.layout, coloraxis_name, None)
        if not ca:
            return
        # 取出现有 colorbar 配置；可能是 None、对象或 dict
        cb = getattr(ca, "colorbar", None)
        if cb is None:
            cb_dict = {}
        else:
            # 转成纯 dict，避免重复关键字
            cb_dict = cb.to_plotly_json() if hasattr(cb, "to_plotly_json") else dict(cb)

        # 默认 x=1.02；在此基础上平移
        cur_x = cb_dict.get("x", 1.02)
        cb_dict["x"] = cur_x + dx

        # 把改好的 colorbar 回写；注意不要再额外传 x=...，只传一个 colorbar 映射
        ca_dict = ca.to_plotly_json() if hasattr(ca, "to_plotly_json") else dict(ca)
        ca_dict["colorbar"] = cb_dict
        fig.update_layout(**{coloraxis_name: ca_dict})

    # 根据参数把两个色条同时往左移一点（更靠近图）
    if cb_dx:
        _shift_cb("coloraxis1", -abs(cb_dx))
        _shift_cb("coloraxis2", -abs(cb_dx))

    # 3) 右边距
    fig.update_layout(margin=dict(r=right_margin))


def export_cluster_neuron_pages_2x4(
    pre_grid: np.ndarray,      # (G,G,N)
    left: np.ndarray,          # (G*G,N)  (与你现有 prepare 中一致；下面会转成 left_vec/right_vec)
    right: np.ndarray,         # (G*G,N)
    dft_2d,                    # 你的 DFT 包装器
    irreps,                    # [(label, …), ...] 只用 labels
    save_dir: str,             # 输出目录
    arts=None,
    rounding_scale: int = 10
):
    """
    聚类（沿用 dominant_irrep 规则），然后为每个 cluster 生成一个 PDF，
    其中每页 = 一个 neuron 的 2x4 正方形页面。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if arts == None:
        arts = prepare_layer_artifacts(pre_grid, left, right, dft_2d, irreps)
    F_full, F_L, F_R = arts["F_full"], arts["F_L"], arts["F_R"]
    names = arts["names"]; irrep2neurons = arts["irrep2neurons"]

    G, N = pre_grid.shape[0], pre_grid.shape[-1]
    cont_l = left.reshape(G, G, N);  left_vec  = cont_l.mean(axis=1)  # (G,N)
    cont_r = right.reshape(G, G, N); right_vec = cont_r.mean(axis=0)  # (G,N)

    for (r, s), neuron_list in irrep2neurons.items():
        if r == s:
            if arts is not None and "cluster_prune" in arts and (r, s) in arts["cluster_prune"]:
                pack = arts["cluster_prune"][(r, s)]
                keep = list(pack["main"]); drop = list(pack["drop"]); per_log = pack["per_neuron_log10"]
            else:
                keep = neuron_list; drop = []; per_log = {}

            cluster_dir = Path(save_dir) / f"cluster_{r}_{s}"
            cluster_dir.mkdir(parents=True, exist_ok=True)

            # 只导出 main
            for n in keep:
                fig = _per_neuron_fig_2x4(n, pre_grid, left_vec, right_vec, F_full, F_L, F_R, names)
                pdf_bytes = fig.to_image(format="pdf", engine="kaleido")
                n_dir = cluster_dir / str(n)
                n_dir.mkdir(exist_ok=True)
                with open(n_dir / "page.pdf", "wb") as f:
                    f.write(pdf_bytes)
                page = build_neuron_page_json(
                    n, pre_grid, left_vec, right_vec, F_full, F_L, F_R,
                    names, rounding_scale=rounding_scale
                )
                json_path = n_dir / "page.json"
                pdf_path  = n_dir / "page.pdf"
                recon_path = n_dir /"page_recon.pdf"

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(_tojson(page), f, ensure_ascii=False, indent=2)
                
                with open(pdf_path, "wb") as f: 
                    f.write(pdf_bytes)
                paper_plots.render_neuron_page_from_json(page, to_pdf_path=str(recon_path))
            
            if drop:
                with open(cluster_dir / "dropped.txt", "w", encoding="utf-8") as f:
                    f.write(f"Cluster ({r},{s}) dropped {len(drop)} neuron(s):\n")
                    for n in sorted(drop):
                        maxv = float(np.max(np.abs(pre_grid[:, :, n])))
                        logv = per_log.get(int(n), None)
                        f.write(f"n={n:4d}  max={maxv:.4g}  log10={('NA' if logv is None else f'{logv:.4f}')}\n")
        print(f"✅ done cluster ({r},{s})")

# ---------- 小工具：把 numpy 标量/数组安全转成 Python/JSON ----------
def _to_list(a):
    
    if isinstance(a, _np.ndarray):
        return a.tolist()
    return a

def _tojson(obj):
    
    # 1) ndarray → list
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    # 2) numpy 标量 → Python 标量
    if isinstance(obj, _np.generic):
        return _np.asarray(obj).item()
    # 3) JAX: ArrayImpl / DeviceArray / jnp 标量
    tname = type(obj).__name__
    if tname in ("ArrayImpl", "DeviceArray"):
        try:
            return _np.asarray(obj).tolist()
        except Exception:
            try:
                return obj.tolist()
            except Exception:
                return float(obj)  # 最后退路（标量）
    # 4) 容器递归
    if isinstance(obj, dict):
        return {k: _tojson(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tojson(v) for v in obj]
    # 5) 其他类型原样返回
    return obj

def _line_package(y: np.ndarray, p: int, y_range: list[float]):
    (x1,y1),(x2,y2) = _line_segments_from_vec(y, p)
    return {
        "x1": _to_list(x1), "y1": _to_list(y1),
        "x2": _to_list(x2), "y2": _to_list(y2),
        "y_range": y_range    # ← 直接存入“当时使用的范围”
    }

# ---------- 生成“单个 neuron 的 2×4 页面”的 JSON 描述 ----------
def build_neuron_page_json(
    n: int,
    pre_grid: np.ndarray,   # (G,G,N)
    left_vec: np.ndarray,   # (G,N)
    right_vec: np.ndarray,  # (G,N)
    F_full, F_L, F_R,       # DFT dicts
    names: list[str],
    *,
    rounding_scale: int = 10
) -> dict:
    G = pre_grid.shape[0]
    p = G // 2
    grid = pre_grid[:, :, n]
    a1d  = left_vec[:, n]
    b1d  = right_vec[:, n]

    # 与你页面里一致：用 BL 象限估 fa/fb
    fb_bl, fa_bl = dominant_freqs_ab(grid[:p, :p])
    fa_q, fb_q = int(fa_bl), int(fb_bl)

    # 1D remap：用 NaN 标无样本点
    a1d_remap = _remap_1d_by_freq(a1d, fa_q, p)
    b1d_remap = _remap_1d_by_freq(b1d, fb_q, p)

    # 三个 DFT 矩阵
    D = len(names); name2idx = {lab: i for i, lab in enumerate(names)}
    P_a = np.zeros((D, D)); P_b = np.zeros((D, D)); P_full = np.zeros((D, D))
    for (r, s, idx), M in F_L.items():
        if idx == n: P_a[name2idx[r], name2idx[s]] = np.linalg.norm(np.array(M))
    for (r, s, idx), M in F_R.items():
        if idx == n: P_b[name2idx[r], name2idx[s]] = np.linalg.norm(np.array(M))
    for (r, s, idx), M in F_full.items():
        if idx == n: P_full[name2idx[r], name2idx[s]] = np.linalg.norm(np.array(M))

    pre_min = float(np.nanmin(grid))
    pre_max = float(np.nanmax(grid))
    if pre_min == pre_max:
        pre_min -= 1e-9; pre_max += 1e-9

    # 四个折线面板打包（包含断开、y 范围等）
    y_a  = _yr(a1d,        rounding_scale)
    y_b  = _yr(b1d,        rounding_scale)
    y_ar = _yr(a1d_remap,  rounding_scale)
    y_br = _yr(b1d_remap,  rounding_scale)

    # 四个折线面板打包（包含断开、y 范围等）
    pack_a   = _line_package(a1d,        p, y_a)
    pack_b   = _line_package(b1d,        p, y_b)
    pack_ar  = _line_package(a1d_remap,  p, y_ar)
    pack_br  = _line_package(b1d_remap,  p, y_br)

    data = {
        "version": "2x4.v1",
        "n": int(n),
        "G": int(G),
        "p": int(p),
        "fa_q": int(fa_q),
        "fb_q": int(fb_q),
        "rounding_scale": int(rounding_scale),

        "titles": {
            "a":        f"a-contribution, max={np.max(a1d):.2f}",
            "b":        f"b-contribution, max={np.max(b1d):.2f}",
            "a_dft":    "a-GFT",
            "b_dft":    "b-GFT",
            "a_remap":  f"a-remap, f={fa_q}",
            "b_remap":  f"b-remap, f={fb_q}",
            "preact2d": "Whole Preactivation",
            "full_dft": "Whole-GFT",
        },

        "lines": {
            "a":       pack_a,
            "b":       pack_b,
            "a_remap": pack_ar,
            "b_remap": pack_br,
        },

        "heatmaps": {
            "a_dft":   {"z": _to_list(P_a),    "x": names, "y": names},
            "b_dft":   {"z": _to_list(P_b),    "x": names, "y": names},
            "preact2d":{"z": _to_list(grid)},
            "full_dft":{"z": _to_list(P_full), "x": names, "y": names},
        },

        # 颜色轴定义：和你页面逻辑一致
        "coloraxis": {
            "dft":     {"colorscale": "Inferno"},
            "preact":  {"colorscale": "Viridis", "cmin": pre_min, "cmax": pre_max},
        },

        # 版式参数（用于重建时保持一样的几何）
        "layout": {
            "hspace": 0.05, "vspace": 0.14,
            "cell_px": 250,
            "margins": {"l": 50, "r": 110, "t": 80, "b": 50},
            # 可随时在 JSON 里改，重建会精准复现
            "padding": {
                "title_yshift": 6,          # 子图标题往上抬多少像素
                "title_xshift": 0,
                "axis_title_standoff": 2,   # 轴标题距离坐标轴的贴近程度
            }
        },
    "style": {
        "font_size_base": 18,     # 全局字体
        "title_font_size": 18,    # 子图标题字体
        "tick_font_size": 16,     # 刻度字体
        "marker_size": 6,         # 折线图点大小
        "dft_tick_angle": 45,      # DFT 三个面板 x 轴刻度旋转角
        "dtick": 1,
    }
    }
    return data


    