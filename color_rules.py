from __future__ import annotations

import math
from typing import Literal, Tuple

import numpy as np
import plotly.colors as pc

# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------
Quadrant = Literal["full", "BL", "BR", "TL", "TR"]

# Which algebraic expression is used in each quadrant (un‑modded)
_QUAD_EXPR = {
    "BL": "a+b",
    "TL": "a+b",
    "BR": "b-a",
    "TR": "b-a",
}

__all__ = [
    "colour_quad_mul_f",
    "colour_quad_mod_g",
    "colour_quad_a_only",
    "colour_quad_b_only",
    "colour_pair_mod_g",
    # "COMBINED_COLORSCALE",
]

_DEFAULT = "Viridis"

def _base_val(a: np.ndarray, b: np.ndarray, tag: str) -> np.ndarray:
    """Return raw (a+b) or (b-a) according to quadrant."""
    if tag in ("BL", "TL"):
        return a + b
    elif tag in ("BR", "TR"):
        return b - a
    raise ValueError("tag must be one of BL/BR/TL/TR")

# =============================================================================
# colormap for Strategy ③  (Rainbow half 0‑0.5, Viridis half 0.5‑1)
# =============================================================================

def _resolve_scale(scale_name_or_list):
   
    if isinstance(scale_name_or_list, (list, tuple)):
        return scale_name_or_list

    name = str(scale_name_or_list)
    
    if name in pc.PLOTLY_SCALES:
        return pc.PLOTLY_SCALES[name]

    try:
        import plotly.express as px
        for ns in (px.colors.sequential, px.colors.diverging, px.colors.cyclical, px.colors.qualitative):
            if hasattr(ns, name):
                return getattr(ns, name)

        if name.lower() in ("oranges", "orrd"):
            return px.colors.sequential.Oranges
        if name.lower() in ("ylorrd",):
            return px.colors.sequential.YlOrRd
    except Exception:
        pass

    raise KeyError(f"Unknown colorscale name: {name}")

def _interp(scale_name_or_list, t: float) -> str:
    scale = _resolve_scale(scale_name_or_list)
    return pc.sample_colorscale(scale, [t])[0]



# def _interp_rainbow_red_orange(t: float) -> str:

#     return _interp("OrRd", 0.25 + 0.75 * t)

# def _interp_viridis(t: float) -> str:
    
#     return _interp("Viridis", 0.0 + 0.9 * t)

# def build_split_scale_red_orange(g: int) -> list[tuple[float,str]]:
#     if g <= 1:
#         return [(0.0, _interp("Viridis", 0.0)), (1.0, _interp("Viridis", 1.0))]
#     v = [(0.5 * i / (g - 1), _interp_viridis(i / (g - 1))) for i in range(g)]
#     r = [(0.5 + 0.5 * i / (g - 1), _interp_rainbow_red_orange(i / (g - 1))) for i in range(g)]
#     return v + r

# def build_vi_scale(g: int) -> list[tuple[float, str]]:
#     return [(i/(g-1), _interp("Viridis", 0.0 + 0.9*i/(g-1)))
#             for i in range(g)]

# def build_ro_scale(g: int) -> list[tuple[float, str]]:
#     """0‥g-1 remap to Rainbow """
#     return [(i/(g-1), _interp("OrRd", 0.25 + 0.75*i/(g-1)))
#             for i in range(g)]

############ ALL VIRIDIS ###############
def _interp_rainbow_red_orange(t: float) -> str:

    return _interp("Viridis", 0.55 + 0.45 * t)

def _interp_viridis(t: float) -> str:
    
    return _interp("Viridis", 0.0 + 0.45 * t)

def build_split_scale_red_orange(g: int) -> list[tuple[float,str]]:
    """
    Viridis half (upper)  +  Viridis half (lower)
    """
    if g <= 1:
        return [(0.0, _interp("Viridis", 0.0)), (1.0, _interp("Viridis", 1.0))]
    v = [(0.45 * i / (g - 1), _interp_viridis(i / (g - 1))) for i in range(g)]
    r = [(0.55 + 0.45 * i / (g - 1), _interp_rainbow_red_orange(i / (g - 1))) for i in range(g)]
    return v + r

def build_vi_scale(g: int) -> list[tuple[float, str]]:
    return [(i/(g-1), _interp("Viridis", 0.0 + 0.45*i/(g-1)))
            for i in range(g)]

def build_ro_scale(g: int) -> list[tuple[float, str]]:
    """0‥g-1 remap to viridis's 0.55‥1.00"""
    return [(i/(g-1), _interp("Viridis", 0.55 + 0.45*i/(g-1)))
            for i in range(g)]

########### Inverted Viridis ###########
# def _interp_viridis_low(t: float) -> str:
#     # Viridis 的下半段 [0.00, 0.45]
#     return _interp("Viridis", 0.00 + 0.45 * t)

# def _interp_viridis_high(t: float) -> str:
#     # Viridis 的上半段 [0.55, 1.00]
#     return _interp("Viridis", 0.55 + 0.45 * t)

# def build_split_scale_red_orange(g: int) -> list[tuple[float, str]]:
#     if g <= 1:
#         return [(0.0, _interp_viridis_high(0.0)), (1.0, _interp_viridis_low(1.0))]

#     # top → [0.00, 0.45]
#     high_first = [
#         (0.00 + 0.45 * (i / (g - 1)), _interp_viridis_high(i / (g - 1)))
#         for i in range(g)
#     ]
#     # bottom → [0.55, 1.00]
#     low_second = [
#         (0.55 + 0.45 * (i / (g - 1)), _interp_viridis_low(i / (g - 1)))
#         for i in range(g)
#     ]
#     return high_first + low_second

# def build_vi_scale(g: int) -> list[tuple[float, str]]:
#     """Viridis bottom [0.00, 0.45]"""
#     if g <= 1:
#         return [(0.0, _interp("Viridis", 0.00)), (1.0, _interp("Viridis", 0.45))]
#     return [(i / (g - 1), _interp("Viridis", 0.00 + 0.45 * i / (g - 1)))
#             for i in range(g)]

# def build_ro_scale(g: int) -> list[tuple[float, str]]:
#     """Viridis top [0.55, 1.00]"""
#     if g <= 1:
#         return [(0.0, _interp("Viridis", 0.55)), (1.0, _interp("Viridis", 1.00))]
#     return [(i / (g - 1), _interp("Viridis", 0.55 + 0.45 * i / (g - 1)))
#             for i in range(g)]


# =============================================================================
# Strategy ①  f · (a±b) mod p
# =============================================================================

# def colour_quad_mul_f(
#     a: np.ndarray | list[int],
#     b: np.ndarray | list[int],
#     p: int,
#     f: int,
#     tag: Quadrant,
# ) -> Tuple[np.ndarray, str, int]:
#     """Return (colour, caption, p_cbar)."""

#     a = np.asarray(a, int)
#     b = np.asarray(b, int)

#     if tag == "full":
#         # Handle each quadrant separately, then concat
#         side = 2 * p
#         A, B = a, b
#         top, right = A >= p, B >= p
#         out = np.empty_like(a)
#         for quad, m in {
#             "BL": (~top) & (~right),
#             "BR": (~top) & right,
#             "TL": top & (~right),
#             "TR": top & right,
#         }.items():
#             out[m], _, _, _ = colour_quad_mul_f(a[m] % p, b[m] % p, p, f, quad)
#         caption = f"f·(a±b) mod {p} [f={f}]"
#         return out, caption, int(out.max())+1, _DEFAULT

#     base = _base_val(a, b, tag)
#     colour = (f * base) % p
#     expr = _QUAD_EXPR[tag]
#     caption = f"{f}·({expr}) mod {p}"
#     return colour, caption, int(colour.max())+1, _DEFAULT


# # =============================================================================
# # Strategy ②  (a±b) mod g, g=gcd(f,p)
# # =============================================================================

def colour_quad_mod_g_no_fb(
    a: np.ndarray | list[int],
    b: np.ndarray | list[int],
    p: int,
    f: int,
    tag: Quadrant,
) -> Tuple[np.ndarray, str, int]:
    g = p // math.gcd(p, f) if math.gcd(p, f) != 0 else p
    a = np.asarray(a, int)
    b = np.asarray(b, int)

    if tag == "full":
        A, B = a, b
        side = 2 * p
        top, right = A >= p, B >= p
        out = np.empty_like(a)
        for quad, m in {
            "BL": (~top) & (~right),
            "BR": (~top) & right,
            "TL": top & (~right),
            "TR": top & right,
        }.items():
            out[m], _, _,_ = colour_quad_mod_g_no_fb(a[m] % p, b[m] % p, p, f, quad)
        caption = f"(a±b) mod {g} [g=p//gcd({f},{p})]"
        return out, caption, int(out.max())+1, _DEFAULT

    base = _base_val(a, b, tag)
    colour = base % g
    expr = _QUAD_EXPR[tag]
    caption = f"({expr}) mod {g}"
    return colour, caption, int(colour.max())+1, _DEFAULT

# ----------------------------------------------------------------------------
# Strategy ① f·(a±b) mod p with split BL/TR ↔ Viridis, others ↔ Orange
# ----------------------------------------------------------------------------
def colour_quad_mul_f(
    a: np.ndarray | list[int],
    b: np.ndarray | list[int],
    p: int,
    f: int,
    tag: Quadrant,
) -> Tuple[np.ndarray, str, int, list[tuple[float,str]]]:
    a = np.asarray(a, int)
    b = np.asarray(b, int)

    # full view: build one combined colour index array
    if tag == "full":
        A, B = a, b
        top = A >= p
        right = B >= p
        # compute raw base: a+b when right==False, else b-a
        base_raw = np.where(right, B - A, A + B)
        # modular colour values
        c = (f * base_raw) % p
        # front quadrants = BL or TR → Viridis; others → Orange
        mask_front = (~top & ~right) | (top & right)
        colour_idx = np.where(mask_front, c, p + c)
        caption = f"f·(b±a) mod {p} (BL/TR→Viridis dark; TL/BR→Viridis bright)"
        return colour_idx, caption, int(colour_idx.max()) + 1, build_split_scale_red_orange(p)

    # single quadrant unchanged
    base = _base_val(a, b, tag)
    is_rain = tag in ("BR", "TL")
    colour = (f * base) % p
    expr = _QUAD_EXPR[tag]
    caption = f"{f}·({expr}) mod {p}"
    cmap = build_ro_scale(p) if is_rain else build_vi_scale(p)
    return colour, caption, int(colour.max()) + 1, cmap

# ----------------------------------------------------------------------------
# Strategy ② (a±b) mod g with split BL/TR ↔ Viridis, others ↔ Orange
# ----------------------------------------------------------------------------
def colour_quad_mod_g(
    a: np.ndarray | list[int],
    b: np.ndarray | list[int],
    p: int,
    f: int,
    tag: Quadrant,
) -> Tuple[np.ndarray, str, int, list[tuple[float,str]]]:
    g = p // math.gcd(p, f) if math.gcd(p, f) != 0 else p
    a = np.asarray(a, int)
    b = np.asarray(b, int)

    if tag == "full":
        A, B = a, b
        top = A >= p
        right = B >= p
        base_raw = np.where(right, B - A, A + B)
        c = base_raw % g
        mask_front = (~top & ~right) | (top & right)
        colour_idx = np.where(mask_front, c, g + c)
        caption = f"(b±a) mod {g} (BL/TR→Viridis dark; TL/BR→Viridis bright)"
        return colour_idx, caption, int(colour_idx.max()) + 1, build_split_scale_red_orange(g)

    # single quadrant unchanged
    base = _base_val(a, b, tag)
    is_rain = tag in ("BR", "TL")
    colour = base % g
    expr = _QUAD_EXPR[tag]
    caption = f"({expr}) mod {g}"
    cmap = build_ro_scale(g) if is_rain else build_vi_scale(p)
    return colour, caption, int(colour.max()) + 1, cmap

# =============================================================================
# Strategy ③  a mod g  with Rainbow/Viridis split
# =============================================================================

def colour_quad_a_only(
    a: np.ndarray | list[int],
    b: np.ndarray | list[int],
    p: int,
    f: int,
    tag: Quadrant,
) -> Tuple[np.ndarray, str, int]:
    g = p // math.gcd(p, f) if math.gcd(p, f) != 0 else p

    a = np.asarray(a, int)
    b = np.asarray(b, int)

    if tag == "full":
        A, B = a, b
        top, right = A >= p, B >= p
        is_rain = (top & (~right)) | ((~top) & right)   # TL / BR

        # first bring to local 0…p‑1 *if* coordinates are global
        base_local = A % p
        base = base_local % g
        
        colour = np.where(is_rain, g + base, base)
        caption = (
            f"Viridis dark (BL/TR): a mod {g}; "
            f"Viridis bright (BR/TL): a mod {g}"
        )
        return colour, caption, 2 * g, build_split_scale_red_orange(g)

    # single quadrant – decide if this one is Rainbow or Viridis
    is_rain = tag in ("BR", "TL")
    base = np.asarray(a) % g  # gcd g divides p ⇒ direct mod suffices
    colour = base
    caption = f"a mod {g}  ({'Viridis bright' if is_rain else 'Viridis dark'})"
    cmap = build_ro_scale(g) if is_rain else build_vi_scale(p)
    return colour, caption, int(colour.max())+1, cmap

def colour_quad_b_only(
    a: np.ndarray | list[int],
    b: np.ndarray | list[int],
    p: int,
    f: int,
    tag: Quadrant,
) -> Tuple[np.ndarray, str, int]:
    g = p // math.gcd(p, f) if math.gcd(p, f) != 0 else p

    a = np.asarray(a, int)
    b = np.asarray(b, int)

    if tag == "full":
        A, B = a, b
        top, right = A >= p, B >= p
        is_rain = (top & (~right)) | ((~top) & right)   # TL / BR

        # first bring to local 0…p‑1 *if* coordinates are global
        base_local = B % p
        base = base_local % g
        
        colour = np.where(is_rain, g + base, base)
        caption = (
            f"Viridis dark (BL/TR): a mod {g}; "
            f"Viridis bright (BR/TL): a mod {g}"
        )
        return colour, caption, 2 * g, build_split_scale_red_orange(g)

    # single quadrant – decide if this one is Rainbow or Viridis
    is_rain = tag in ("BR", "TL")
    base = np.asarray(b) % g  # gcd g divides p ⇒ direct mod suffices
    colour = base
    caption = f"b mod {g}  ({'Viridis bright' if is_rain else 'Viridis dark'})"
    cmap = build_ro_scale(g) if is_rain else build_vi_scale(p)
    return colour, caption, int(colour.max())+1, cmap

def colour_quad_a_only_no_fb(
    a: np.ndarray | list[int],
    b: np.ndarray | list[int], 
    p: int,
    f: int,
    tag: str,                    # Quadrant | "full"
):
    g = p // math.gcd(p, f) if math.gcd(p, f) != 0 else p

    a = np.asarray(a, int)
    b = np.asarray(b, int)

    if tag == "full":
        A = a % p
        colour = A % g
    else:
        colour = a % g

    caption = f"a mod {g}"
    pbar = int(g)

    return colour, caption, pbar, _DEFAULT

def colour_quad_b_only(
    a: np.ndarray | list[int],
    b: np.ndarray | list[int],
    p: int,
    f: int,
    tag: Quadrant,
) -> Tuple[np.ndarray, str, int]:
    g = p // math.gcd(p, f) if math.gcd(p, f) != 0 else p

    a = np.asarray(a, int)
    b = np.asarray(b, int)

    if tag == "full":
        A, B = a, b
        top, right = A >= p, B >= p
        is_rain = (top & (~right)) | ((~top) & right)   # TL / BR

        # first bring to local 0…p‑1 *if* coordinates are global
        base_local = B % p
        base = base_local % g
        
        colour = np.where(is_rain, g + base, base)
        caption = (
            f"Viridis dark (BL/TR): a mod {g}; "
            f"Viridis bright (BR/TL): a mod {g}"
        )
        return colour, caption, 2 * g, build_split_scale_red_orange(g)

    # single quadrant – decide if this one is Rainbow or Viridis
    is_rain = tag in ("BR", "TL")
    base = np.asarray(b) % g  # gcd g divides p ⇒ direct mod suffices
    colour = base
    caption = f"b mod {g}  ({'Viridis bright' if is_rain else 'Viridis dark'})"
    cmap = build_ro_scale(g) if is_rain else build_vi_scale(p)
    return colour, caption, int(colour.max())+1, cmap

def colour_quad_b_only_no_fb(
    a: np.ndarray | list[int],
    b: np.ndarray | list[int], 
    p: int,
    f: int,
    tag: str,                    # Quadrant | "full"
):
    g = p // math.gcd(p, f) if math.gcd(p, f) != 0 else p

    a = np.asarray(a, int)
    b = np.asarray(b, int)

    if tag == "full":
        B = b % p
        colour = B % g
    else:
        colour = b % g

    caption = f"b mod {g}"
    pbar = int(g)

    return colour, caption, pbar, _DEFAULT


# def lines_a_mod_g(a_vals, b_vals,p,g):
#     h_pairs = []
#     for b_fix in list(range(g)) + [p + i for i in range(g)]:
#         # --- split according to a_vals ---
#         solid_idx = np.where((a_vals < g) & (b_vals == b_fix))[0]
#         dash_idx  = np.where((a_vals >= p) & (a_vals < p + g) & (b_vals == b_fix))[0]

#         # --- color ---
#         # b_fix < p blue，else red
#         color = "blue" if b_fix < p else "red"
#         line_id = f"b_{b_fix}" 
        
#         if solid_idx.size > 1:
#             h_pairs.append((solid_idx, "solid", color, line_id))
#         if dash_idx.size  > 1:
#             h_pairs.append((dash_idx,  "dash",  color, line_id))
#     return h_pairs

# def lines_b_mod_g(a_vals, b_vals,p,g):
#     v_pairs = []
#     for a_fix in list(range(g)) + [p + i for i in range(g)]:
#         # --- split according to b_vals ---
#         solid_idx = np.where((b_vals < g) & (a_vals == a_fix))[0]
#         dash_idx  = np.where((b_vals >= p) & (b_vals < p + g) & (a_vals == a_fix))[0]

#         # --- color ---
#         # a_fix < p blue，else
#         color = "blue" if a_fix < p else "red"
#         line_id = f"a_{a_fix}" 
        
#         if solid_idx.size > 1:
#             v_pairs.append((solid_idx, "solid", color, line_id))
#         if dash_idx.size  > 1:
#             v_pairs.append((dash_idx,  "dash",  color, line_id))
#     return v_pairs

# def lines_c_mod_g(a_vals, b_vals, p, g):
#     A, B = a_vals, b_vals
#     top, right = (A >= p), (B >= p)

#     # quads masking（constrain a and b's "local g×g" ranges）
#     mask_BL = (~top) & (~right) & (A < g)       & (B < g)         # a∈[0,g),   b∈[0,g)
#     mask_TR = ( top) &  (right) & (A >= p) & (A < p+g) & (B >= p) & (B < p+g)  # a∈[p,p+g), b∈[p,p+g)
#     mask_BR = (~top) &  (right) & (A < g)       & (B >= p) & (B < p+g)         # a∈[0,g),   b∈[p,p+g)
#     mask_TL = ( top) & (~right) & (A >= p) & (A < p+g) & (B < g)               # a∈[p,p+g), b∈[0,g)

#     pairs = []
#     for r in range(g):
#         # BL: (b + a) % g == r, blue
#         m_bl = mask_BL & (((B % g) + (A % g)) % g == r)
#         idxs = np.where(m_bl)[0]
#         if idxs.size > 1:
#             idxs = idxs[np.argsort(A[idxs])]
#             pairs.append((idxs, "solid", "blue",  f"BL_r{r}"))

#         # TR: (b - a) % g == r, red
#         m_tr = mask_TR & (((B - A) % g) == r)
#         idxs = np.where(m_tr)[0]
#         if idxs.size > 1:
#             idxs = idxs[np.argsort(A[idxs])]
#             pairs.append((idxs, "solid", "red",   f"TR_r{r}"))

#         # BR: (b - a) % g == r, green
#         m_br = mask_BR & (((B - A) % g) == r)
#         idxs = np.where(m_br)[0]
#         if idxs.size > 1:
#             idxs = idxs[np.argsort(A[idxs])]
#             pairs.append((idxs, "solid", "green", f"BR_r{r}"))

#         # TL: ((b%g) + ((a-p)%g)) % g == r, purple
#         m_tl = mask_TL & ((((B % g) + ((A - p) % g)) % g) == r)
#         idxs = np.where(m_tl)[0]
#         if idxs.size > 1:
#             idxs = idxs[np.argsort(A[idxs])]
#             pairs.append((idxs, "solid", "purple", f"TL_r{r}"))

#     print("c_pairs count:", len(pairs))
#     return pairs
### line by step size
import math
import numpy as np

# ---------- step size ----------
def step_size(f: int, p: int) -> int:
    g = math.gcd(f, p)
    n = p // g
    return pow(f // g, -1, n)   # inverse exists because gcd(f/g, p/g)=1

# ---------- internal: step-ordering utility ----------
def _order_by_step(local_vals: np.ndarray, g: int, d: int) -> np.ndarray:
    """
    Given local residues (0..g-1) for a set of points, return a permutation
    that orders them as r0, r0+d, r0+2d, ... (mod g). Start at the smallest residue
    present for determinism. Points with equal residue keep their original order.
    """
    resid = np.asarray(local_vals, int) % g
    if resid.size == 0:
        return np.arange(0, 0, dtype=int)

    # start residue = smallest present (deterministic)
    r0 = int(np.min(resid))
    seq = [(r0 + t * d) % g for t in range(g)]
    pos = {r: i for i, r in enumerate(seq)}  # residue -> order along cycle

    # stable argsort by 'pos[residue]'
    order_keys = np.vectorize(pos.get)(resid)
    return np.argsort(order_keys, kind="stable")

# ---------- A-lines: fix b, walk a with step d ----------
def lines_a_mod_g_step(a_vals, b_vals, p, g, d):
    """
    Return [(idx_ordered, dash, color, line_id), ...]
    For each fixed b (two stripes: BL and TL), order points by a in +d steps mod g.
    """
    a_vals = np.asarray(a_vals, int)
    b_vals = np.asarray(b_vals, int)
    out = []

    # BL: b in [0,g), a in [0,g)
    for b_fix in range(g):
        idx = np.where((b_vals == b_fix) & (a_vals >= 0) & (a_vals < g))[0]
        if idx.size >= 2:
            # local a = a
            order = _order_by_step(a_vals[idx] % g, g, d)
            out.append((idx[order], "solid", "blue", f"BL_b{b_fix}"))

    # TL: b in [0,g), a in [p,p+g)  (use a' = a - p)
    for b_fix in range(g):
        idx = np.where((b_vals == b_fix) & (a_vals >= p) & (a_vals < p + g))[0]
        if idx.size >= 2:
            order = _order_by_step((a_vals[idx] - p) % g, g, d)
            out.append((idx[order], "dash", "blue", f"TL_b{b_fix}"))

    # BR: b in [p,p+g), a in [0,g)
    for b_fix in range(p, p + g):
        idx = np.where((b_vals == b_fix) & (a_vals >= 0) & (a_vals < g))[0]
        if idx.size >= 2:
            order = _order_by_step(a_vals[idx] % g, g, d)
            out.append((idx[order], "solid", "red", f"BR_b{b_fix}"))

    # TR: b in [p,p+g), a in [p,p+g)  (use a' = a - p)
    for b_fix in range(p, p + g):
        idx = np.where((b_vals == b_fix) & (a_vals >= p) & (a_vals < p + g))[0]
        if idx.size >= 2:
            order = _order_by_step((a_vals[idx] - p) % g, g, d)
            out.append((idx[order], "dash", "red", f"TR_b{b_fix}"))

    return out

# ---------- B-lines: fix a, walk b with step d ----------
def lines_b_mod_g_step(a_vals, b_vals, p, g, d):
    """
    Symmetric to A-lines: for each fixed a (two stripes: BL and BR), step b by +d mod g.
    """
    a_vals = np.asarray(a_vals, int)
    b_vals = np.asarray(b_vals, int)
    out = []

    # BL: a in [0,g), b in [0,g)
    for a_fix in range(g):
        idx = np.where((a_vals == a_fix) & (b_vals >= 0) & (b_vals < g))[0]
        if idx.size >= 2:
            order = _order_by_step(b_vals[idx] % g, g, d)
            out.append((idx[order], "solid", "blue", f"BL_a{a_fix}"))

    # BR: a in [0,g), b in [p,p+g)  (use b' = b - p)
    for a_fix in range(g):
        idx = np.where((a_vals == a_fix) & (b_vals >= p) & (b_vals < p + g))[0]
        if idx.size >= 2:
            order = _order_by_step((b_vals[idx] - p) % g, g, d)
            out.append((idx[order], "dash", "red", f"BR_a{a_fix}"))

    # TL: a in [p,p+g), b in [0,g)  (use a' = a - p)
    for a_fix in range(p, p + g):
        idx = np.where((a_vals == a_fix) & (b_vals >= 0) & (b_vals < g))[0]
        if idx.size >= 2:
            order = _order_by_step(b_vals[idx] % g, g, d)
            out.append((idx[order], "solid", "blue", f"TL_a{a_fix}"))

    # TR: a in [p,p+g), b in [p,p+g)  (use a' = a - p, b' = b - p; stepping on b')
    for a_fix in range(p, p + g):
        idx = np.where((a_vals == a_fix) & (b_vals >= p) & (b_vals < p + g))[0]
        if idx.size >= 2:
            order = _order_by_step((b_vals[idx] - p) % g, g, d)
            out.append((idx[order], "dash", "red", f"TR_a{a_fix}"))

    return out

# ---------- C-lines: fix r ≡ (b±a) mod g on each quadrant, walk a (or a') by +d ----------
def lines_c_mod_g_step(a_vals, b_vals, p, g, d):
    """
    For each residue r in 0..g-1:
      BL: a∈[0,g), b∈[0,g),   (b+a)≡r mod g     (order by a)
      TR: a∈[p,p+g), b∈[p,p+g), (b-a)≡r mod g   (order by a' = a-p)
      BR: a∈[0,g),   b∈[p,p+g), (b-a)≡r mod g   (order by a)
      TL: a∈[p,p+g), b∈[0,g),   (b+a')≡r mod g  (order by a' = a-p)
    """
    A = np.asarray(a_vals, int)
    B = np.asarray(b_vals, int)
    top, right = (A >= p), (B >= p)

    mask_BL = (~top) & (~right) & (A < g)       & (B < g)
    mask_TR = ( top) &  (right) & (A >= p) & (A < p+g) & (B >= p) & (B < p+g)
    mask_BR = (~top) &  (right) & (A < g)       & (B >= p) & (B < p+g)
    mask_TL = ( top) & (~right) & (A >= p) & (A < p+g) & (B < g)

    out = []
    for r in range(g):
        # BL: (b+a) % g == r, order by a
        idx = np.where(mask_BL & (((B % g) + (A % g)) % g == r))[0]
        if idx.size >= 2:
            order = _order_by_step(A[idx] % g, g, d)
            out.append((idx[order], "solid", "blue",  f"BL_r{r}"))

        # TR: (b-a) % g == r, order by a' = a - p
        idx = np.where(mask_TR & (((B - A) % g) == r))[0]
        if idx.size >= 2:
            order = _order_by_step(((A[idx] - p) % g), g, d)
            out.append((idx[order], "solid", "red",   f"TR_r{r}"))

        # BR: (b-a) % g == r, order by a
        idx = np.where(mask_BR & (((B - A) % g) == r))[0]
        if idx.size >= 2:
            order = _order_by_step(A[idx] % g, g, d)
            out.append((idx[order], "solid", "green", f"BR_r{r}"))

        # TL: ((b%g) + (a' % g)) % g == r, order by a' = a - p
        idx = np.where(mask_TL & ((((B % g) + ((A - p) % g)) % g) == r))[0]
        if idx.size >= 2:
            order = _order_by_step(((A[idx] - p) % g), g, d)
            out.append((idx[order], "solid", "purple", f"TL_r{r}"))

    return out

# ==== helpers for discrete categorical colorbars ====
def _build_discrete_scale(colors: list[str]) -> list[tuple[float, str]]:
    """
    Build a Plotly colorscale for discrete classes 0..(k-1).
    Each class gets an interval [i/k, (i+1)/k].
    """
    k = len(colors)
    if k == 0:
        return [(0.0, "#000000"), (1.0, "#000000")]
    scale = []
    for i, c in enumerate(colors):
        lo = i / k
        hi = (i + 1) / k
        scale.append((lo, c))
        scale.append((hi, c))
    return scale

# sensible defaults (qualitative & readable on light bg)
# Colorblind-friendly discrete picks from Viridis
# (chosen for good spacing & contrast on white)

# (a%2, b%2) classes → 4 colors
_MOD2_COLORS = ["#3B528B",  # deep blue
                "#21918C",  # teal
                "#5EC962",  # green
                "#FDE725"]  # yellow

# front/back → 2 colors
_FB2_COLORS  = ["#2C728E",  # blue-teal (front)
                "#95D840"]  # yellow-green (back)


# ----------------------------------------------------------------------------
# NEW RULE A: colour by (a mod 2, b mod 2)
# classes: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
# ----------------------------------------------------------------------------
def colour_pair_mod2(
    a: np.ndarray | list[int],
    b: np.ndarray | list[int],
    p: int,
    f: int,
    tag: Quadrant,     # or "full"
) -> Tuple[np.ndarray, str, int, list[tuple[float, str]]]:
    a = np.asarray(a, int)
    b = np.asarray(b, int)

    # Always compute parity from the *actual indices* shown (global),
    # then reduce to {0,1}.
    if tag == "full":
        A, B = a, b
    else:
        # In a single quadrant function you typically pass local a,b already,
        # but this is safe even if they are global.
        A, B = a, b

    am2 = (A & 1)
    bm2 = (B & 1)

    # encode pair to 0..3
    colour = (am2 << 1) | bm2  # (a%2)*2 + (b%2)
    caption = "(a mod 2, b mod 2) in {00,01,10,11} → {0,1,2,3}"
    pbar = 4
    cmap = _build_discrete_scale(_MOD2_COLORS)
    return colour, caption, pbar, cmap

# ----------------------------------------------------------------------------
# NEW RULE B: colour by front/back quadrants (BL/TR = front, TL/BR = back)
# Uses your existing front/back definition used in split scales.
# ----------------------------------------------------------------------------
def colour_front_back_by_c(
    a: np.ndarray | list[int],
    b: np.ndarray | list[int],
    p: int,
    f: int,
    tag: Quadrant,     # or "full"
) -> Tuple[np.ndarray, str, int, list[tuple[float, str]]]:
    a = np.asarray(a, int)
    b = np.asarray(b, int)

    if tag == "full":
        A, B = a, b
        top   = (A >= p)
        right = (B >= p)
        # front = BL or TR; back = TL or BR
        is_front = ((~top) & (~right)) | (top & right)
        colour = np.where(is_front, 0, 1)  # 0:front, 1:back
        caption = "Sign +1 (BL/TR) vs Sign -1 (TL/BR) quadrants"
        pbar = 2
        cmap = _build_discrete_scale(_FB2_COLORS)
        return colour, caption, pbar, cmap

    # Single quadrant: just a constant mask per quadrant.
    # BL/TR → class 0 (front), TL/BR → class 1 (back).
    if tag in ("BL", "TR"):
        cls = 0
    elif tag in ("TL", "BR"):
        cls = 1
    else:
        raise ValueError("tag must be one of BL/BR/TL/TR or 'full'")
    colour = np.full_like(a, fill_value=cls, dtype=int)
    caption = f"{'Sign +1' if cls==0 else 'Sign -1'} quadrant ({tag})"
    pbar = 2
    cmap = _build_discrete_scale(_FB2_COLORS)
    return colour, caption, pbar, cmap
