# analysis/post_training_analysis_Transformer.py
import os, json, re
from typing import Dict, List, Tuple, Any
import numpy as np
import jax
import jax.numpy as jnp

import DFT, dihedral, report
from pca_diffusion_plots_w_helpers import generate_pdf_plots_for_matrix
from color_rules import colour_quad_a_only, colour_quad_b_only, colour_quad_mod_g

from transformer_class import (
    TransformerOneEmbed, TransformerTwoEmbed, HookPoint
)
import controllers.paths_Transformer as paths  
from controllers.training_prep_Transformer import eval_model 

def final_eval_all_models(*, states, x_eval_batches, y_eval_batches, init_metrics, random_seed_ints: List[int]):
    test_metrics = eval_model(states, x_eval_batches, y_eval_batches, init_metrics)
    results = {}
    for i, seed in enumerate(random_seed_ints):
        tm = jax.tree_util.tree_map(lambda x: x[i], test_metrics).compute()
        reached = float(tm["accuracy"]) ==1.0
        results[seed] = {
            "reach_100%_test": reached,
            "loss": float(tm["loss"]),
            "l2_loss": float(tm["l2_loss"]),
            "accuracy": float(tm["accuracy"]),
        }
    return results


def save_epoch_logs(logs_by_seed: Dict[int, Dict[int, Dict]], out_dir: str, features_or_dm: int):
    os.makedirs(out_dir, exist_ok=True)
    for seed, logs in logs_by_seed.items():
        path = os.path.join(out_dir, f"log_features_{features_or_dm}_seed_{seed}.json")
        with open(path, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"[Transformer] Epoch log for seed {seed} saved to {path}")


def save_final_logs(log_by_seed: Dict[int, Dict[int, Dict]], out_dir: str, features: int):
    """Dump epoch logs to JSON files per seed."""
    os.makedirs(out_dir, exist_ok=True)
    for seed, logs in log_by_seed.items():
        path = os.path.join(out_dir, f"final_log_features_{features}_seed_{seed}.json")
        with open(path, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"[Transformer] Final log for seed {seed} saved to {path}")

def save_prune_logs(log_by_seed: Dict[int, Dict[int, Dict]], out_dir: str, features: int):
    """Dump epoch logs to JSON files per seed."""
    os.makedirs(out_dir, exist_ok=True)
    for seed, logs in log_by_seed.items():
        path = os.path.join(out_dir, f"prune_log_features_{features}_seed_{seed}.json")
        with open(path, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"Prune log for seed {seed} saved to {path}")

def make_some_jsons(
    *,
    preacts: list[np.ndarray],
    group_size: int,
    clusters_by_layer: list[dict[int, list[int]]],
    cluster_weights_to_logits: dict[int, np.ndarray],
    cluster_weights_to_dmodel: dict[int, np.ndarray] | None = None,
    save_dir: str,
    subdir: str = "json",
    float_dtype=np.float32,
    sanity_check: bool = True,
    cluster_contribs_to_logits: dict[int, np.ndarray] | None = None,
    cluster_contribs_to_dmodel: dict[int, np.ndarray] | None = None, 
) -> str:
    """
    Writes one JSON per *last layer* cluster: cluster_{freq}.json
    For each neuron in the cluster (keyed by its neuron_idx as a string), stores:
      - "preactivations": (group_size^2,)
      - "w_out":          (group_size,)
      - "contribs_to_logits": (group_size^2, group_size) = ReLU(preacts)[:,None] * w_out[None,:]
      - "w_dmodel":       (d_model,)                 
      - "contribs_to_dmodel": (group_size^2, d_model) 

    Safety checks:
      • Ensures preacts[-1] is (group_size^2, width_last)
      • Ensures W_block is (|cluster|, group_size)
      • Ensures neuron_ids are within [0, width_last)
      • Optional exactness checks vs. cluster_contribs_to_logits[freq] and _to_dmodel[freq]
    """
    # ---- global shape checks
    if not preacts:
        raise ValueError("make_some_jsons: empty `preacts`.")
    Z_last = np.asarray(preacts[-1])  # (group_size^2, width_last)
    n_rows, width_last = Z_last.shape
    if n_rows != group_size * group_size:
        raise ValueError(f"make_some_jsons: expected group_size^2={group_size*group_size} rows, got {n_rows}.")
    if not clusters_by_layer:
        raise ValueError("make_some_jsons: empty `clusters_by_layer`.")

    last_layer_clusters = clusters_by_layer[-1] or {}
    if not isinstance(last_layer_clusters, dict):
        raise TypeError("make_some_jsons: clusters_by_layer[-1] must be a dict {freq -> [neuron_ids]}.")

    json_root = os.path.join(save_dir, subdir)
    os.makedirs(json_root, exist_ok=True)

    for freq, neuron_ids in last_layer_clusters.items():
        if not neuron_ids:
            continue

        # Pull the aligned output weights block (built with the SAME order as neuron_ids)
        W_block = cluster_weights_to_logits.get(freq, None)
        if W_block is None:
            # Nothing to write if we don't have this cluster's output weights
            continue
        W_block = np.asarray(W_block)  # (|cluster|, group_size)

        # also pull the W_dmodel rows if provided
        W_block_dmodel = None  
        if cluster_weights_to_dmodel is not None:  
            W_block_dmodel = cluster_weights_to_dmodel.get(freq)  # (|cluster|, d_model)  
            if W_block_dmodel is not None:  
                W_block_dmodel = np.asarray(W_block_dmodel)  

        # ---- index validation & alignment
        ids = np.asarray(neuron_ids, dtype=int)  # (|cluster|)
        valid_mask = (ids >= 0) & (ids < width_last)
        if np.any((ids < 0) | (ids >= width_last)):
            bad = ids[(ids < 0) | (ids >= width_last)]
            raise ValueError(f"cluster {freq}: invalid neuron ids {bad.tolist()} for width_last={width_last}")
        if W_block.shape[0] != ids.shape[0]:
            raise ValueError(f"cluster {freq}: W_block rows {W_block.shape[0]} != len(ids) {ids.shape[0]}")


        # ---- shape checks after filtering
        if W_block.shape[0] != ids.shape[0]:
            raise ValueError(
                f"make_some_jsons: for freq={freq}, W_block rows ({W_block.shape[0]}) "
                f"≠ number of neuron ids ({ids.shape[0]})."
            )
        # CHANGED: use group_size in message (was `p` which is undefined here)
        if W_block.shape[1] != group_size: 
            raise ValueError(  
                f"make_some_jsons: for freq={freq}, W_block has {W_block.shape[1]} columns, expected group_size={group_size}."  # CHANGED
            )

        # Gather per-neuron preacts and ReLU
        Z_cluster = Z_last[:, ids]                 # (group_size^2, |cluster|)
        H_cluster = np.maximum(Z_cluster, 0.0)     # (group_size^2, |cluster|)

        # Vectorized per-neuron contributions: (group_size^2, |cluster|, group_size)
        contribs_logits = H_cluster[:, :, None] * W_block[None, :, :]
        # --- DEBUG BEGIN ---
        Hf = np.asarray(H_cluster, dtype=np.float64)       # (p^2, |C|)
        Wf = np.asarray(W_block,  dtype=np.float64)        # (|C|, p)
        C_sum_broadcast = (Hf[:, :, None] * Wf[None, :, :]).sum(axis=1)  # (p^2, p)
        C_mm            = Hf @ Wf                                        # (p^2, p)

        if not np.allclose(C_sum_broadcast, C_mm, rtol=1e-9, atol=1e-12):
            diff  = np.max(np.abs(C_sum_broadcast - C_mm))
            where = np.unravel_index(np.argmax(np.abs(C_sum_broadcast - C_mm)), C_mm.shape)
            raise RuntimeError(
                f"[debug] freq={freq}: broadcast-sum vs matmul mismatch. "
                f"max_abs_diff={diff:.3e} at {where}; "
                f"dtypes H:{H_cluster.dtype} W:{W_block.dtype}"
            )
        # --- DEBUG END ---


        # per-neuron contribs to d_model if we have W_block_dmodel
        contribs_dmodel = None  
        if W_block_dmodel is not None:  
            contribs_dmodel = H_cluster[:, :, None] * W_block_dmodel[None, :, :]  # (group_size^2, |cluster|, d_model)  

        # Optional correctness check against provided cluster_contribs_to_logits
        if sanity_check and (cluster_contribs_to_logits is not None):
            C_freq_expected = cluster_contribs_to_logits.get(freq)
            if C_freq_expected is not None and np.size(C_freq_expected):
                C_exp = np.asarray(C_freq_expected, dtype=np.float64)
                if C_exp.shape != C_mm.shape:
                    raise ValueError(
                        f"[debug] freq={freq}: shape mismatch exp{C_exp.shape} vs mm{C_mm.shape}"
                    )
                scale = max(1.0, float(np.max(np.abs(C_exp))))
                if not np.allclose(C_mm, C_exp, rtol=1e-4, atol=1e-5*scale):
                    diff  = np.max(np.abs(C_mm - C_exp))
                    where = np.unravel_index(np.argmax(np.abs(C_mm - C_exp)), C_mm.shape)
                    raise ValueError(
                        f"make_some_jsons: contribution mismatch for freq={freq}. "
                        f"max_abs_diff={diff:.3e} at {where}. "
                        f"|C|={W_block.shape[0]}, group_size={group_size}"
            )

        # optional correctness check for d_model sums
        if sanity_check and (cluster_contribs_to_dmodel is not None) and (contribs_dmodel is not None):  # ADDED
            D_freq_expected = np.asarray(cluster_contribs_to_dmodel.get(freq))  # (group_size^2, d_model)  
            if D_freq_expected is not None and D_freq_expected.size:  
                D_sum = contribs_dmodel.sum(axis=1)  # (group_size^2, d_model)  
                if D_freq_expected.shape != D_sum.shape:  
                    raise ValueError(  
                        f"make_some_jsons: cluster_contribs_to_dmodel[{freq}] has shape {D_freq_expected.shape}, "  # ADDED
                        f"expected {D_sum.shape}."  
                    )
                if not np.allclose(D_sum, D_freq_expected, rtol=1e-5, atol=1e-6):  
                    raise ValueError(  
                        f"make_some_jsons: d_model contribution mismatch for freq={freq} " 
                        f"(sum of per-neuron ≠ cluster total)."  
                    )

        # Build JSON payload { "<neuron_idx>": {...}, ... } preserving original order
        payload = {}
        for j, nid in enumerate(ids.tolist()):
            entry = {
                "preactivations": Z_cluster[:, j].astype(float_dtype).tolist(),   # (group_size^2,)
                "w_out":          W_block[j, :].astype(float_dtype).tolist(),     # (group_size,)
                "contribs_to_logits": contribs_logits[:, j, :].astype(float_dtype).tolist(),  # (group_size^2, group_size)
            }
            # per-neuron d_model outputs if available
            if W_block_dmodel is not None:  
                entry["w_dmodel"] = W_block_dmodel[j, :].astype(float_dtype).tolist()  
                entry["contribs_to_dmodel"] = contribs_dmodel[:, j, :].astype(float_dtype).tolist()  
            payload[str(int(nid))] = entry  # keep original order

        out_path = os.path.join(json_root, f"cluster_{freq}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f)

    return json_root
# -------- prune helper ----------
def _load_alive_indices_for_seed(prune_dir: str, features_or_dm: int, seed: int,
                                *, num_layers: int, params_seed: dict) -> list[list[int]]:
    path = os.path.join(prune_dir, f"prune_log_features_{features_or_dm}_seed_{seed}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            rep = json.load(f)
        alive = rep.get("alive_final") or rep.get("stageB_alive") or rep.get("stageA_alive")
        if alive is not None:
            return [[int(x) for x in alive.get(str(li), [])] for li in range(num_layers)]
    # fallback = all original ids per layer (infer width from params)
    mlp = params_seed["blocks_0"]["mlp"]
    return [list(range(int(mlp[f"b_{li}"].shape[0]))) for li in range(num_layers)]

# ---------- utilities to read HookPoint pre-activations ----------
# ---------- pull preacts for all MLP layers (last token only) ----------
def _find_by_suffix(d, suffix):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(k, str) and k.endswith(suffix):
                if isinstance(v, list):   # flax intermediates store lists
                    return v[0]
                if isinstance(v, dict):
                    return next(iter(v.values()))
                return v
            out = _find_by_suffix(v, suffix)
            if out is not None:
                return out
    elif isinstance(d, list):
        for x in d:
            out = _find_by_suffix(x, suffix)
            if out is not None:
                return out
    return None


def _extract_hook_pre_all_layers(model, params, x_full, num_mlp_layers: int):
    _, inter = model.apply({"params": params}, x_full, training=False, mutable=["intermediates"])
    ints = inter["intermediates"]
    outs = []
    for l in range(1, num_mlp_layers + 1):
        suffix = f"blocks_0/mlp/hook_pre{l}"
        arr = _find_by_suffix(ints, suffix)
        if arr is None:
            arr = _find_by_suffix(ints, f"hook_pre{l}")
        outs.append(arr)  # (B, seq_len, width_l)
    return outs

def extract_preacts_last_token(
    *, model, params: dict, group_size: int, num_mlp_layers: int, last_token_index: int = 1
) -> List[np.ndarray]:
    """Returns list of (group_size^2, width_l) preact matrices for each MLP layer."""
    a = np.arange(group_size, dtype=np.int32); b = np.arange(group_size, dtype=np.int32)
    A, B = np.meshgrid(a, b, indexing="ij")
    x_full = jnp.stack([A.ravel(), B.ravel()], axis=-1).astype(jnp.int32)  # (p^2, 2)

    pre_list = _extract_hook_pre_all_layers(model, params, x_full, num_mlp_layers)
    preacts = [np.asarray(pre)[:, last_token_index, :] for pre in pre_list]  # (p^2, width_l)
    return preacts

# ---------- X_in and its split (for optional left/right display) ----------
def build_X_in_and_halves(model, params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X_in: (p^2, 2*D) = [E_a+pos0 || E_b+pos1]
    Xa, Xb: the two halves, each (p^2, D)
    """
    X_in = np.asarray(model.all_p_squared_embeddings(params))
    D2 = X_in.shape[1]; assert D2 % 2 == 0, "Expected even last dim for [Ea||Eb]."
    D = D2 // 2
    return X_in, X_in[:, :D], X_in[:, D:]

def compute_first_layer_ab_contribs_transformer(
    *,
    model,            # TransformerOneEmbed or TransformerTwoEmbed
    params: dict,
    group_size: int,           # group size
    last_token_index: int = 1,   # we read logits from the last token
    bias_mode: str = "b",        # "b" (default), "even", or "none"
):
    """
    Returns:
      pre_L1_from_a : (group_size:^2, d_mlp)  -- contribution from token 'a' via attention
      pre_L1_from_b : (group_size:^2, d_mlp)  -- residual (Eb+pos1) + attention from 'b' (+ bias per bias_mode)
      pre_L1_total  : (group_size:^2, d_mlp)  -- sum of the two (for sanity)
      pre_L1_hook   : (p^2, d_mlp)  -- actual hook_pre1 (for sanity)
    """

    # 1) Build full grid inputs (a,b)
    a = np.arange(group_size, dtype=np.int32); b = np.arange(group_size, dtype=np.int32)
    A, B = np.meshgrid(a, b, indexing="ij")
    x_full = jnp.stack([A.ravel(), B.ravel()], axis=-1).astype(jnp.int32)  # (p^2, 2)
    Bsize = x_full.shape[0]

    # 2) One forward pass with intermediates
    _, inter = model.apply({"params": params}, x_full, training=False, mutable=["intermediates"])
    ints = inter["intermediates"]

    # Attention pieces from block 0
    attn = _find_by_suffix(ints, "blocks_0/attn/hook_attn")  # (B, H, Q, P)
    v    = _find_by_suffix(ints, "blocks_0/attn/hook_v")     # (B, H, P, d_head)
    if attn is None:
        attn = _find_by_suffix(ints, "hook_attn")
    if v is None:
        v = _find_by_suffix(ints, "hook_v")
    if attn is None or v is None:
        raise KeyError("Could not find attention hooks (hook_attn, hook_v) in intermediates.")

    attn = jnp.asarray(attn)     # (B, H, 2, 2)
    v    = jnp.asarray(v)        # (B, H, 2, d_head)

    # 3) W_O (combine heads), and MLP layer-1 weights/bias
    W_O = jnp.asarray(params["blocks_0"]["attn"]["W_O"])  # (d_model, H*d_head)
    W0  = jnp.asarray(params["blocks_0"]["mlp"]["W_0"])   # (d_mlp, d_model)
    b0  = jnp.asarray(params["blocks_0"]["mlp"]["b_0"])   # (d_mlp,)

    # 4) Residual for token-1: (E_b + pos1)
    Wa, Wb = model.extract_embeddings_ab(params)  # (p,D), (p,D)
    pos0, pos1 = params["pos_embed"]["W_pos"][:2] # (D,), (D,)
    b_idx = x_full[:, 1]
    Eb_pos1 = jnp.asarray(Wb)[b_idx] + jnp.asarray(pos1)  # (B, D)

    # 5) Attention contributions for query token = last_token_index
    q = int(last_token_index)  # 1
    # from 'a' (src token 0) and from 'b' (src token 1)
    z_from_a = v[:, :, 0, :] * attn[:, :, q, 0][..., None]  # (B, H, d_head)
    z_from_b = v[:, :, 1, :] * attn[:, :, q, 1][..., None]  # (B, H, d_head)

    # flatten heads then map with W_O to model space
    zfa = z_from_a.reshape(Bsize, -1)  # (B, H*d_head)
    zfb = z_from_b.reshape(Bsize, -1)  # (B, H*d_head)
    attn_from_a = jnp.einsum("df,bf->bd", W_O, zfa)  # (B, d_model)
    attn_from_b = jnp.einsum("df,bf->bd", W_O, zfb)  # (B, d_model)

    # 6) Compose x_mid[1] contributions
    xmid_from_a = attn_from_a              # only via attention
    xmid_from_b = Eb_pos1 + attn_from_b    # residual + attention

    # 7) Map to MLP layer-1 preactivations
    pre_L1_from_a = jnp.einsum("md,bd->bm", W0, xmid_from_a)  # (B, d_mlp)
    pre_L1_from_b = jnp.einsum("md,bd->bm", W0, xmid_from_b)  # (B, d_mlp)

    # add bias according to your preference
    if bias_mode == "b":
        pre_L1_from_b = pre_L1_from_b + b0
    elif bias_mode == "even":
        half = 0.5 * b0
        pre_L1_from_a = pre_L1_from_a + half
        pre_L1_from_b = pre_L1_from_b + half
    elif bias_mode == "none":
        pass
    else:
        raise ValueError("bias_mode must be one of {'b','even','none'}")

    pre_L1_total = pre_L1_from_a + pre_L1_from_b

    # 8) (optional) grab true hook_pre1 for sanity
    pre1_hook = _find_by_suffix(ints, "blocks_0/mlp/hook_pre1")
    if pre1_hook is None:
        pre1_hook = _find_by_suffix(ints, "hook_pre1")
    if pre1_hook is None:
        raise KeyError("Could not find 'hook_pre1' in intermediates.")
    # slice last token and make (B, d_mlp)
    pre_L1_hook = jnp.asarray(pre1_hook)[:, q, :]

    # to numpy
    return (
        np.asarray(pre_L1_from_a),
        np.asarray(pre_L1_from_b),
        np.asarray(pre_L1_total),
        np.asarray(pre_L1_hook),
    )

# ---------- last-layer cluster → logits (Transformer effective map) ----------
def cluster_contribs_last_layer_transformer(
    *,
    preacts_last: np.ndarray,                       # (group_size^2, d_mlp)
    params: dict,
    clusters_last_layer: dict[int, list[int]],      # {freq -> [neuron_ids]} after pruning
):
    mlp_params = params["blocks_0"]["mlp"]
    W_out = np.asarray(mlp_params["W_out"])         # (d_model, d_mlp)
    W_U   = np.asarray(params["W_U"])               # (d_model, group_size)

    eff_W_dmodel = W_out.T                          # (d_mlp, d_model)
    eff_W_logits = W_out.T @ W_U                    # (d_mlp, group_size)

    Z_last = np.asarray(preacts_last)               # ensure ndarray
    H_last = np.maximum(Z_last, 0.0)                # (group_size^2, d_mlp)

    d_mlp = H_last.shape[1]

    contribs_to_dmodel: dict[int, np.ndarray] = {}
    contribs_to_logits: dict[int, np.ndarray]  = {}
    Wblocks_to_logits: dict[int, np.ndarray]   = {}
    Wblocks_to_dmodel: dict[int, np.ndarray]   = {}   

    for f, ids in (clusters_last_layer or {}).items():
        if not ids:
            continue
        ids = np.asarray(ids, dtype=int)

        # guard invalid indices but preserve order
        mask = (ids >= 0) & (ids < d_mlp)
        if not np.all(mask):
            ids = ids[mask]
            if ids.size == 0:
                continue

        Hc = H_last[:, ids]                         # (group_size^2, |C|)
        Wd = eff_W_dmodel[ids, :]                   # (|C|, d_model)
        Wl = eff_W_logits[ids, :]                   # (|C|, group_size)

        contribs_to_dmodel[f] = Hc @ Wd             # (group_size^2, d_model)
        contribs_to_logits[f] = Hc @ Wl             # (group_size^2, group_size)
        Wblocks_to_logits[f]  = Wl                  # keep for JSON parity
        Wblocks_to_dmodel[f]  = Wd                  

    # CHANGED: return the d_model blocks as well
    return contribs_to_dmodel, contribs_to_logits, Wblocks_to_logits, Wblocks_to_dmodel  # CHANGED


def run_post_training_analysis(
    *,
    model,
    states,
    random_seed_ints: List[int],
    p: int,
    group_size: int,             # == p, kept for symmetry with MLP signature
    num_layers: int,
    mdir: str,
    class_lower: str = "transformer",
    colour_rule= None,
    dmodel: int | None = None,
    alive_by_layer_override: dict[int, list[list[int]]] | None = None,
    write_json: bool = False,
    write_pdfs: bool = False,
):
    """Transformer post-hoc pipeline that mirrors MLP_pip:
       1) DFT prep, 2) per-layer artifacts (with pruning), 3) summaries,
       4) last-layer cluster → logits (and optional JSON/PDF outputs).
    """
    # ----- DFT prep (same as MLP) -----
    G, irreps = DFT.make_irreps_Dn(p)
    freq_map = {name: freq for (name, dim, R, freq) in irreps}
    rho_cache = DFT.build_rho_cache(G, irreps)
    dft_fn    = DFT.jit_wrap_group_dft(rho_cache, irreps, group_size)

    subgroups = dihedral.enumerate_subgroups_Dn(p)

    # coset mask (for make_layer_report)
    coset_masks_L = dihedral.build_coset_masks(G, subgroups, dihedral.mult, p, side="left")
    coset_masks_R = dihedral.build_coset_masks(G, subgroups, dihedral.mult, p, side="right")


    for seed_idx, seed in enumerate(random_seed_ints):
        print(f"\n=== Transformer post-training analysis for seed {seed} ===")
        gdir = paths.seed_graph_dir(mdir, seed)

        # single model params
        params_seed = jax.tree_util.tree_map(lambda x: x[seed_idx], states.params)
        prune_dir = mdir
        if alive_by_layer_override is not None and seed in alive_by_layer_override:
            alive_by_layer = alive_by_layer_override[seed]
        else:
            alive_by_layer = _load_alive_indices_for_seed(
                prune_dir=mdir, features_or_dm=dmodel,
                seed=seed, num_layers=num_layers, params_seed=params_seed
            )
        for li in range(num_layers):
            width_li = int(params_seed["blocks_0"]["mlp"][f"b_{li}"].shape[0])
            bad = [i for i in alive_by_layer[li] if i < 0 or i >= width_li]
            if bad:
                raise ValueError(f"[seed={seed}] alive_by_layer[{li}] includes bad index: {bad} (width={width_li})")

        # cluster + report generation
        layers_freq: List[Dict[int, list]] = []  # each layer：freq -> [neuron ids]
        # 1) preacts (last token), and X_in halves (optional)
        pre_acts_all = extract_preacts_last_token(
            model=model, params=params_seed, group_size=group_size, num_mlp_layers=num_layers, last_token_index=1
        )  # list of (p^2, width_l)
        X_in, Xa, Xb = build_X_in_and_halves(model, params_seed)  # (p^2, 2D), (p^2,D), (p^2,D)
        pre_a, pre_b, pre_sum, pre_hook = compute_first_layer_ab_contribs_transformer(
            model=model,
            params=params_seed,  # single-model pytree
            group_size=group_size,
            last_token_index=1,  # you train/eval on the last token
            bias_mode="b",       # sums exactly to hook preacts
        )
        # 2) artifacts per layer (with pruning like MLP)
        thresh_small = 2.0 if group_size < 50 else 3.0
        cluster_tau = 1e-3
        for layer_idx in range(num_layers):
            prei_full = pre_acts_all[layer_idx]                       # (p^2, width_full)
            alive_ids = alive_by_layer[layer_idx]                     # ORIGINAL ids
            prei      = prei_full[:, alive_ids]                       # (p^2, alive_count)
            prei_grid = prei.reshape(group_size, group_size, -1)
            pre_a_alive = pre_a[:, alive_ids]
            pre_b_alive = pre_b[:, alive_ids]
            artifacts = report.prepare_layer_artifacts(
                prei_grid, pre_a_alive, pre_b_alive, dft_fn, irreps, freq_map,
                prune_cfg={"thresh1": thresh_small, "thresh2": thresh_small, "seed": 0},
            )

            # per layer cluster based on freq
            local_clusters = artifacts.get("freq_cluster", {}) or {}
            clusters_layer = {freq: [alive_ids[j] for j in ids] for freq, ids in local_clusters.items()}  # {freq: [orig neuron ids]}
            layers_freq.append(clusters_layer)

            # approx summary per layer
            diag_labels = artifacts["diag_labels"]
            names = artifacts["names"]
            approx = report.summarize_diag_labels(diag_labels, p, names)
            with open(os.path.join(gdir, f"approx_summary_layer{layer_idx+1}_p{p}.json"), "w") as f:
                json.dump(approx, f, indent=2)

            # # commented out for scaling
            # report_dir = os.path.join(gdir, f"report_layer{layer_idx+1}")
            # os.makedirs(report_dir, exist_ok=True)
            # report.make_layer_report(
            #     prei_grid, pre_a_alive, pre_b_alive, p,
            #     dft_fn, irreps, coset_masks_L, coset_masks_R,
            #     report_dir, cluster_tau, colour_rule, artifacts,
            # )
            

        # 5) Last-layer cluster → logits using transformer effective map
        last_layer_clusters = layers_freq[-1]
        contribs_dmodel, contribs_logits, Wblocks_logits, Wblocks_dmodel = cluster_contribs_last_layer_transformer(
            preacts_last=pre_acts_all[-1],
            params=params_seed,
            clusters_last_layer=last_layer_clusters,
        )

        # 6) (optional) write per-cluster JSON compatible with MLP make_some_jsons
        if write_json:
            json_root = os.path.join(gdir, "json")
            os.makedirs(json_root, exist_ok=True)
            _ = make_some_jsons(
                preacts=pre_acts_all,
                group_size=group_size,
                clusters_by_layer=layers_freq,
                cluster_weights_to_logits=Wblocks_logits,
                cluster_weights_to_dmodel=Wblocks_dmodel,  
                cluster_contribs_to_logits=contribs_logits,
                cluster_contribs_to_dmodel=contribs_dmodel,
                save_dir=gdir,
                subdir="json",
                sanity_check=True,
            )
            print(f"[Transformer] cluster JSONs written → {os.path.join(gdir, 'json')}")

        # 7) (optional) PDFs of cluster→logits like MLP
        if write_pdfs and contribs_logits:  # CHANGED: was `contribs_to_logits`
            pdf_root = os.path.join(gdir, "pdf_plots", f"seed_{seed}")
            os.makedirs(pdf_root, exist_ok=True)
            num_pc = 4  # tune if desired
            for freq, C_freq in contribs_logits.items():
                generate_pdf_plots_for_matrix(
                    C_freq, p,
                    save_dir=pdf_root, seed=seed,
                    freq_list=[freq],
                    tag=f"cluster_contributions_to_logits_freq={freq}",
                    tag_q="full",
                    colour_rule=colour_quad_mod_g,
                    class_string=class_lower,
                    num_principal_components=num_pc,
                )
            print(f"[Transformer] PDF plots written → {pdf_root}")