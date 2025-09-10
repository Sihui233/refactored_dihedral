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


def final_eval_all_models(*, states, x_eval_batches, y_eval_batches, init_metrics, random_seed_ints: List[int]):
    from controllers.training_prep_MLP import eval_model # MLP version and Transformer version are identical
    test_metrics = eval_model(states, x_eval_batches, y_eval_batches, init_metrics)
    results = {}
    for i, seed in enumerate(random_seed_ints):
        tm = jax.tree_util.tree_map(lambda x: x[i], test_metrics).compute()
        results[seed] = {
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
        print(f"[Transformer] Final log for seed {seed} saved to {path}")


# ============== Internal helpers: fetch Transformer MLP's pre-acts ==============
##### BUGGY ####
def _find_num_mlp_layers(params_block: dict) -> int:
    """Count how many W_i / b_i under blocks_0/mlp"""
    mlp = params_block["mlp"]
    idxs = []
    for k in mlp.keys():
        m = re.match(r"W_(\d+)$", k)
        if m:
            idxs.append(int(m.group(1)))
    if not idxs:
        raise ValueError("Could not infer num_mlp_layers from params['blocks_0']['mlp']")
    return max(idxs) + 1

def _extract_hook_pre(model, params, x_batch, layer: int = 1):
    suffix = f"hook_pre{layer}"

    _, mut = model.apply({"params": params}, x_batch, mutable=["intermediates"], training=False)
    ints = mut.get("intermediates", {})

    def _find(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k.endswith(suffix):
                    if isinstance(v, list):  return v[0]
                    if isinstance(v, dict):  return next(iter(v.values()))
                    return v
                found = _find(v)
                if found is not None:
                    return found
        return None

    arr = _find(ints)
    if arr is None:
        raise KeyError(f"Couldn't find any key ending in '{suffix}'. \n"
                       f"(Top-level keys were: {list(ints.keys())}")
    arr = np.asarray(jax.device_get(arr))
    return arr  # full (N, seq_len, width), no token slicing



# ============== cluster → logits and embeddings ==============

def _effective_final_weights(params_block, params_top) -> np.ndarray:
    """
    params_block : the *last* TransformerBlock params  (e.g. params['blocks_0'])
    params_top   : the root-level params  (contains 'W_U')

    Returns
    -------
    eff_W : (d_mlp , p)   weight that multiplies the post-ReLU neuron
                        activations of the *deepest* MLP layer and goes
                        straight to the p logits.
    """
    W_out = np.array(params_block['mlp']['W_out'])         # (d_model , d_mlp)
    W_U   = np.array(params_top['W_U'])                    # (d_model, |G|)
    eff_W_logits  = W_out.T @ W_U                              # (d_mlp, |G|)                                   # (d_model, d_mlp_last)
    return eff_W_logits                    # (d_mlp_last, |G|)

def _embeddings_A_B(model, params: dict):
    """Return token embeddings of A and B (without position); also returns first 2 position vectors."""
    Wa, Wb = model.extract_embeddings_ab(params)           # (|G|, d_model) or (|G|, D)
    pos0, pos1 = params["pos_embed"]["W_pos"][:2]          # (d_model,)
    return np.asarray(Wa), np.asarray(Wb), np.asarray(pos0), np.asarray(pos1)

def _build_left_right_from_grid(pre_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a 3D array (|G|, |G|, width), return left/right broadcasted views
    of shape (|G|^2, width), constructed via averaging along axes.
    """
    G = pre_grid.shape[0]
    left_vec  = pre_grid.mean(axis=1)                             # (|G|, width)
    right_vec = pre_grid.mean(axis=0)                             # (|G|, width)
    left  = np.tile(left_vec[:, None, :],  (1, G, 1)).reshape(G*G, -1)
    right = np.tile(right_vec[None, :, :], (G, 1, 1)).reshape(G*G, -1)
    return left, right


def get_all_preacts_and_embeddings(
    *,
    model,
    neuron_data: dict[int, dict[int, dict]],
    dominant_freq_clusters,              # list[dict] or {freq: [ids]}
    params: dict,
):
    """
    Returns
    -------
    embeddings            : np.ndarray                         # (p , d_model)
    layer_preacts         : list[dict[str, np.ndarray]]        # as before
    cluster_contribs_logits: dict[str, np.ndarray]             # (p² , p) – after W_U
    cluster_contribs_dmodel: dict[str, np.ndarray]             # (p² , d_model) – before W_U
    """

    # ─────────────────────────── 0. constants & helpers ──────────────────────────
    last_layer_idx   = max(neuron_data)
    last_block_key   = f'blocks_0'            # we only have 1 block

    W_out = np.array(params[last_block_key]['mlp']['W_out'])   # (d_model , d_mlp)
    W_U   = np.array(params['W_U'])                            # (d_model , p)

    eff_W_logits  = W_out.T @ W_U                              # (d_mlp , p)
    eff_W_dmodel  = W_out.T                                    # (d_mlp , d_model)

    # ─────────────────────────── 1. token embeddings ────────────────────────────
    W_a, W_b = model.extract_embeddings_ab(params)               # (p , d_model)
    embeddings_a = np.asarray(W_a)
    embeddings_b = np.asarray(W_b)

    # ─────────────────────────── 2. layer-wise pre-acts ─────────────────────────
    layer_preacts: list[dict[str, np.ndarray]] = []

    # allow both the “single-dict” and the “list of dicts” formats you’re using
    if isinstance(dominant_freq_clusters, dict):
        clusters_by_layer = {1: dominant_freq_clusters}
    else:
        clusters_by_layer = {i + 1: d for i, d in enumerate(dominant_freq_clusters)}

    for layer_idx in sorted(neuron_data):
        pre_dict = {}
        for freq_key, ids in clusters_by_layer[layer_idx].items():
            if not ids:
                continue
            cols = []
            for nid in ids:
                v = neuron_data[layer_idx][nid]['real_preactivations'].reshape(-1)  # (p^2,)
                if float(v.max()) >= 1e-2:  # 0.01 threshold
                    cols.append(v)
            if cols:
                mat = np.stack(cols, axis=1)  # (p^2, kept_neurons)
                pre_dict[freq_key] = mat
        layer_preacts.append(pre_dict)

    # ─────────────────────────── 3. cluster → logits  &  cluster → d_model ──────
    cluster_contribs_logits  : dict[str, np.ndarray] = {}
    cluster_contribs_dmodel : dict[str, np.ndarray] = {}

    for freq_key, ids in clusters_by_layer[last_layer_idx].items():
        if not ids:
            continue
        kept_cols = []
        kept_ids  = []
        for nid in ids:
            post = neuron_data[last_layer_idx][nid].get(
                "postactivations",
                np.maximum(neuron_data[last_layer_idx][nid]["real_preactivations"], 0.0)
            ).reshape(-1)  # (p^2,)
            if float(post.max()) >= 1e-2:  # 0.01 threshold on *activations*
                kept_cols.append(post)
                kept_ids.append(nid)

        if kept_cols:
            post_mat = np.stack(kept_cols, axis=1)  # (p^2, kept_neurons)

            # IMPORTANT: index the effective weights with the *kept* neuron ids
            W_logits_sub = eff_W_logits[kept_ids, :]   # (kept_neurons, p)
            cluster_contribs_logits[freq_key]  = post_mat @ W_logits_sub

            W_dmodel_sub = eff_W_dmodel[kept_ids, :]   # (kept_neurons, d_model)
            cluster_contribs_dmodel[freq_key] = post_mat @ W_dmodel_sub

    return embeddings_a, embeddings_b, layer_preacts, cluster_contribs_logits, cluster_contribs_dmodel

# ============== Write per-cluster JSON (same format as MLP version) ==============

def _make_cluster_jsons(
    *,
    preacts_last: np.ndarray,
    cluster_W_blocks: Dict[Any, np.ndarray],
    save_dir: str,
    sanity_check: bool = True,
    cluster_contribs_to_logits: Dict[Any, np.ndarray] | None = None,
) -> str:
    json_root = os.path.join(save_dir, "cluster_jsons_lastlayer")
    os.makedirs(json_root, exist_ok=True)

    H_last = np.maximum(preacts_last, 0.0)

    for freq, W_blk in cluster_W_blocks.items():
        ids = np.arange(W_blk.shape[0])
        Z_cluster = preacts_last[:, ids]
        H_cluster = H_last[:, ids]
        contribs  = H_cluster[:, :, None] * W_blk[None, :, :]

        if sanity_check and (cluster_contribs_to_logits is not None):
            C_expected = np.asarray(cluster_contribs_to_logits.get(freq))
            if C_expected is not None and C_expected.size:
                C_sum = contribs.sum(axis=1)
                if C_expected.shape != C_sum.shape or not np.allclose(C_sum, C_expected, rtol=1e-5, atol=1e-6):
                    raise ValueError(f"[Sanity] cluster {freq} contrib mismatch.")

        payload = {}
        for j in range(len(ids)):
            payload[str(int(j))] = {
                "preactivations": Z_cluster[:, j].astype(np.float64).tolist(),
                "w_out":          W_blk[j, :].astype(np.float64).tolist(),
                "contribs_to_logits": contribs[:, j, :].astype(np.float64).tolist(),
            }
        out_path = os.path.join(json_root, f"cluster_{freq}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f)

    return json_root
