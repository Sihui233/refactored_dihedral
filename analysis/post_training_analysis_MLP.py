# analysis/post_training_analysis_MLP.py
import os
import json
from typing import Dict, List, Tuple, Union, Any
import numpy as np
import jax
import jax.numpy as jnp
import os, json
from collections import Counter
import DFT
import dihedral
import report
from mlp_models_multilayer import DonutMLP
import controllers.paths_MLP as paths
# Domain libs used in your monolith
import dihedral
import DFT
import report

# Plot helpers you already have
from pca_diffusion_plots_w_helpers import generate_pdf_plots_for_matrix, compute_pca_coords, _write_multiplot_3d, _write_multiplot_2d


# Colour rules (pick the one you want for PDFs)
from color_rules import colour_quad_a_only, colour_quad_mod_g

def final_eval_all_models(*, states, x_eval_batches, y_eval_batches, init_metrics, random_seed_ints: List[int]):
    """Compute final test metrics for each model over full group grid."""
    from controllers.training_prep_MLP import eval_model
    test_metrics = eval_model(states, x_eval_batches, y_eval_batches, init_metrics)
    results = {}
    for i, seed in enumerate(random_seed_ints):
        tm = jax.tree_util.tree_map(lambda x: x[i], test_metrics).compute()
        reached = float(tm["accuracy"]) ==1.0
        results[seed] = {
            "reach_100%_test": reached,
            "loss": float(tm["loss"]),
            "l2_loss": float(tm["l2_loss"]),
            "accuracy": float(tm["accuracy"])
        }
    return results


def save_epoch_logs(log_by_seed: Dict[int, Dict[int, Dict]], out_dir: str, features: int):
    """Dump epoch logs to JSON files per seed."""
    os.makedirs(out_dir, exist_ok=True)
    for seed, logs in log_by_seed.items():
        path = os.path.join(out_dir, f"log_features_{features}_seed_{seed}.json")
        with open(path, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"Epoch log for seed {seed} saved to {path}")

def save_final_logs(log_by_seed: Dict[int, Dict[int, Dict]], out_dir: str, features: int):
    """Dump epoch logs to JSON files per seed."""
    os.makedirs(out_dir, exist_ok=True)
    for seed, logs in log_by_seed.items():
        path = os.path.join(out_dir, f"final_log_features_{features}_seed_{seed}.json")
        with open(path, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"Final log for seed {seed} saved to {path}")

def save_prune_logs(log_by_seed: Dict[int, Dict[int, Dict]], out_dir: str, features: int):
    """Dump epoch logs to JSON files per seed."""
    os.makedirs(out_dir, exist_ok=True)
    for seed, logs in log_by_seed.items():
        path = os.path.join(out_dir, f"prune_log_features_{features}_seed_{seed}.json")
        with open(path, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"Prune log for seed {seed} saved to {path}")
        
        
def _load_alive_indices_for_seed(prune_dir: str, features: int, seed: int, *,
                                 num_layers: int,
                                 params_seed: dict) -> list[list[int]]:
    prune_path = os.path.join(prune_dir, f"prune_log_features_{features}_seed_{seed}.json")
    if os.path.exists(prune_path):
        with open(prune_path, "r") as f:
            rep = json.load(f)
        alive_map = rep.get("alive_final") or rep.get("stageB_alive") or rep.get("stageA_alive")
        if alive_map is not None:
            out = []
            for li in range(num_layers):
                out.append([int(x) for x in alive_map.get(str(li), [])])
            return out

    # fallback: all orig id
    out = []
    for li in range(1, num_layers + 1):
        width = int(params_seed[f"dense_{li}"]["bias"].shape[0])
        out.append(list(range(width)))
    return out

def mlp_W_between(params: dict, layer_from: int, layer_to: int) -> np.ndarray:
    """
    Return the kernel that maps activations of layer layer_from to preacts of layer_to.
    For DonutMLP naming: dense_1 maps input → layer1, dense_2 maps layer1 → layer2, etc.
    So 'between 1 and 2' is params['dense_2']['kernel'].
    """
    assert layer_to == layer_from + 1, "Only adjacent layers are supported here."
    key = f"dense_{layer_to}"
    if key not in params:
        raise KeyError(f"Missing {key} in params. Available keys: {list(params.keys())}")
    return np.asarray(params[key]["kernel"])

def plot_Wblock_pca_by_srcfreq_color_by_tgtfreq(
    *,
    params_seed: dict,
    layers_freq: list[dict[int, list[int]]],   # per-layer {freq -> [orig ids]}
    alive_by_layer: list[list[int]],           # per-layer [orig ids]
    gdir: str,
    p: int,
    seed: int | str,
    layer_from: int,                           # 1..num_layers-1
    min_src: int = 2,
    min_tgt: int = 2,
    colorscale: str = "Viridis"
):
    import os, json
    import numpy as np

    layer_to = layer_from + 1
    W = mlp_W_between(params_seed, layer_from, layer_to)          # (width_from, width_to)

    # clusters on source/target layers (kept in ORIGINAL ids)
    src_clusters = layers_freq[layer_from-1] or {}
    tgt_clusters = layers_freq[layer_to-1] or {}

    alive_src = set(int(i) for i in alive_by_layer[layer_from-1])
    tgt_alive = [int(i) for i in alive_by_layer[layer_to-1]
                 if 0 <= int(i) < W.shape[1]]                     # keep order

    # map each target id -> its freq label (unclustered -> -1)
    tgt_id2freq = {}
    for f_tgt, ids in (tgt_clusters or {}).items():
        for nid in ids:
            tgt_id2freq[int(nid)] = int(f_tgt)
    present_freqs = sorted({tgt_id2freq.get(j, -1) for j in tgt_alive})
    freq2code = {fval: k for k, fval in enumerate(present_freqs)}
    colour_vec = np.array([freq2code[tgt_id2freq.get(j, -1)] for j in tgt_alive], dtype=int)
    p_cbar = max(1, len(present_freqs))  # colour bar domain size

    # loop over source clusters
    for f_src, src_ids_raw in (src_clusters or {}).items():
        # rows = src-cluster (and alive), cols = alive target
        src_ids = [int(i) for i in src_ids_raw
                   if int(i) in alive_src and 0 <= int(i) < W.shape[0]]
        if len(src_ids) < min_src or len(tgt_alive) < min_tgt:
            continue

        # build feature matrix: each target neuron is a point (row)
        # features = connections from src-cluster neurons
        X = W[np.ix_(src_ids, tgt_alive)].T                       # (|tgt_alive|, |src_cluster|)
        # PCA coords (safe/centered/zero-var dropped)
        coords, pca = compute_pca_coords(X, num_components=min(8, X.shape[1], max(1, X.shape[0]-1)))

        # out paths + titles
        out_dir = os.path.join(gdir, "Wblock_pca", f"layer{layer_from}_to_{layer_to}", f"src_f={int(f_src)}")
        os.makedirs(out_dir, exist_ok=True)
        ctitle = f"target freq (layer {layer_to})"
        tag = f"W[{layer_from}->{layer_to}] src_f={int(f_src)} |src|={len(src_ids)} |tgt_alive|={len(tgt_alive)}"

        # 2D & 3D multi-plots (reuse your writers)
        _write_multiplot_2d(
            coords, colour_vec, ctitle,
            os.path.join(out_dir, f"pca_seed_{seed}_srcf_{int(f_src)}_2d.pdf"),
            p=p, p_cbar=p_cbar, colorscale=colorscale,
            seed=seed, label="PC", tag=tag
        )
        _write_multiplot_3d(
            coords, colour_vec, ctitle,
            os.path.join(out_dir, f"pca_seed_{seed}_srcf_{int(f_src)}_3d.pdf"),
            p=p, p_cbar=p_cbar, colorscale=colorscale,
            seed=seed, label="PC", tag=tag,
            f=int(f_src), mult=False  # no quadrant/multi-view for neurons
        )

        # save legend/mapping for the color codes
        with open(os.path.join(out_dir, "legend.json"), "w") as fh:
            json.dump({
                "layer_from": layer_from,
                "layer_to": layer_to,
                "src_freq": int(f_src),
                "present_target_freqs_in_order": present_freqs,   # index -> original freq
                "code_by_freq": {str(k): int(v) for k, v in freq2code.items()},
                "target_ids_in_order": tgt_alive
            }, fh, indent=2)


# def mlp_extract_all_weights(params: dict) -> dict:
#     """
#     Returns:
#       {
#         'hidden': [
#             {'name': 'dense_1', 'W': (in_dim, width1),  'b': (width1,)},
#             {'name': 'dense_2', 'W': (width1, width2),  'b': (width2,)},
#             ...
#         ],
#         'output': {'name': 'output_dense', 'W': (width_L, p), 'b': (p,) or None}
#       }
#     """
#     # collect hidden layers in numeric order: dense_1, dense_2, ...
#     hidden = []
#     for k in sorted([k for k in params if re.fullmatch(r"dense_\d+", k)],
#                     key=lambda s: int(s.split("_")[1])):
#         W = np.asarray(params[k]["kernel"])
#         b = np.asarray(params[k].get("bias")) if "bias" in params[k] else None
#         hidden.append({"name": k, "W": W, "b": b})

#     # output layer (if present)
#     out = None
#     if "output_dense" in params:
#         oW = np.asarray(params["output_dense"]["kernel"])
#         ob = np.asarray(params["output_dense"].get("bias")) if "bias" in params["output_dense"] else None
#         out = {"name": "output_dense", "W": oW, "b": ob}

#     return {"hidden": hidden, "output": out}

def get_all_preacts_and_embeddings(
        *,                        
        model: DonutMLP,
        params: dict,
        group_size: int | None = None,
        clusters_by_layer: list[dict[int, list[int]]] | None = None,
):
    
    """
    Build the (group_size², d_in) matrix that actually feeds the first Dense layer
    and return

    Returns
    -------
    preacts : list[np.ndarray]
        A list of length model.num_layers, where
        preacts[L] has shape (group_size², width_L) of the raw pre-ReLU activations.
    X_in : np.ndarray
        The (group_size², d_in) input matrix formed by all (a,b) embedding pairs.
    weights : list[np.ndarray]
        Hidden-layer weight kernels; weights[L] has shape
        (in_dim_L, width_L).
    cluster_contribs : dict[int, np.ndarray]
        For every frequency f in the **last** hidden layer this returns
        a matrix of shape (group_size², group_size):
    cluster_weights : dict[int, np.ndarray]
        For every frequency f in the last hidden layer, this is the slice
        of the output-layer kernel that feeds the logits from the neurons
        in cluster f.  Shape: (|cluster_f|, group_size).
    
            H_cluster @ W_block
        where
        - H_cluster is the ReLU’d activations of the neurons in cluster f
          at every of the group_size² inputs, and
        - W_block is the slice of the output layer’s weight matrix
          corresponding to those same neurons.
    """
    if clusters_by_layer is None:
        raise ValueError("clusters_by_layer cannot be None")

    group_size = group_size or model.group_size
    X_in = model.all_p_squared_embeddings(params)                # (group_size², d_in)

    # forward pass once to get *pre-activations*
    _, preacts = model.call_from_embedding(jnp.asarray(X_in), params)
    preacts_np = [np.asarray(layer) for layer in preacts]        # list[(p², width_L)]
    # convert last layer to *post-ReLU activations*
    H_last = np.maximum(preacts_np[-1], 0.0)                     # (p², width_{L})

    # collect hidden-layer kernels 
    weights_np = [np.asarray(params[f"dense_{l}"]["kernel"])
                  for l in range(1, model.num_layers + 1)]

    # build cluster-wise *logit contributions*
    W_out = np.asarray(params["output_dense"]["kernel"])         # (width_L, group_size)
    cluster_contribs: dict[int, np.ndarray] = {}
    cluster_weights : dict[int, np.ndarray] = {}
    last_layer_clusters = clusters_by_layer[-1]                  # freq → [ids]
    for freq, neuron_ids in last_layer_clusters.items():
        if not neuron_ids:                  # skip empty clusters
            continue
        H_cluster = H_last[:, neuron_ids]               # (p², |cluster|)
        W_block   = W_out[neuron_ids, :]                # (|cluster|, group_size)
        C_freq    = H_cluster @ W_block                 # (p², group_size)
        cluster_contribs[freq] = C_freq
        cluster_weights[freq]  = W_block  

    return preacts_np, X_in, weights_np, cluster_contribs, cluster_weights

def make_some_jsons(
    *,
    preacts: list[np.ndarray],
    group_size: int,
    clusters_by_layer: list[dict[int, list[int]]],
    cluster_weights_to_logits: dict[int, np.ndarray],
    save_dir: str,
    subdir: str = "json",
    float_dtype=np.float32,
    sanity_check: bool = True,
    cluster_contribs_to_logits: dict[int, np.ndarray] | None = None,
) -> str:
    """
    Writes one JSON per *last layer* cluster: cluster_{freq}.json
    For each neuron in the cluster (keyed by its neuron_idx as a string), stores:
      - "preactivations": (group_size^2,)
      - "w_out":          (group_size,)
      - "contribs_to_logits": (group_size^2, group_size) = ReLU(preacts)[:,None] * w_out[None,:]

    Safety checks:
      • Ensures preacts[-1] is (group_size^2, width_last)
      • Ensures W_block is (|cluster|, group_size)
      • Ensures neuron_ids are within [0, width_last)
      • Optional exactness check vs. cluster_contribs_to_logits[freq]
    """
    # ---- global shape checks
    if not preacts:
        raise ValueError("make_some_jsons: empty preacts.")
    Z_last = np.asarray(preacts[-1])  # (group_size^2, width_last)
    n_rows, width_last = Z_last.shape
    if n_rows != group_size * group_size:
        raise ValueError(f"make_some_jsons: expected group_size^2={group_size*group_size} rows, got {n_rows}.")
    if not clusters_by_layer:
        raise ValueError("make_some_jsons: empty clusters_by_layer.")

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

        # ---- index validation & alignment
        ids = np.asarray(neuron_ids, dtype=int)  # (|cluster|)
        valid_mask = (ids >= 0) & (ids < width_last)
        if not np.all(valid_mask):
            bad = ids[~valid_mask].tolist()
            # Filter both ids and W_block rows to keep alignment
            ids = ids[valid_mask]
            W_block = W_block[valid_mask, :]
            if ids.size == 0:
                # No valid neurons remain
                continue
            print(f"[make_some_jsons] freq={freq}: dropped invalid neuron ids {bad}")

        # ---- shape checks after filtering
        if W_block.shape[0] != ids.shape[0]:
            raise ValueError(
                f"make_some_jsons: for freq={freq}, W_block rows ({W_block.shape[0]}) "
                f"≠ number of neuron ids ({ids.shape[0]})."
            )
        if W_block.shape[1] != group_size:
            raise ValueError(
                f"make_some_jsons: for freq={freq}, W_block has {W_block.shape[1]} columns, expected p={p}."
            )

        # Gather per-neuron preacts and ReLU
        Z_cluster = Z_last[:, ids]                 # (group_size^2, |cluster|)
        H_cluster = np.maximum(Z_cluster, 0.0)     # (group_size^2, |cluster|)

        # Vectorized per-neuron contributions: (group_size^2, |cluster|, group_size)
        contribs = H_cluster[:, :, None] * W_block[None, :, :]

        # Optional correctness check against provided cluster_contribs_to_logits
        if sanity_check and (cluster_contribs_to_logits is not None):
            C_freq_expected = np.asarray(cluster_contribs_to_logits.get(freq))
            if C_freq_expected is not None and C_freq_expected.size:
                C_sum = contribs.sum(axis=1)  # (group_size^2, group_size)
                if C_freq_expected.shape != C_sum.shape:
                    raise ValueError(
                        f"make_some_jsons: cluster_contribs_to_logits[{freq}] has shape {C_freq_expected.shape}, "
                        f"expected {C_sum.shape}."
                    )
                if not np.allclose(C_sum, C_freq_expected, rtol=1e-5, atol=1e-6):
                    raise ValueError(
                        f"make_some_jsons: contribution mismatch for freq={freq} "
                        f"(sum of per-neuron ≠ cluster total)."
                    )

        # Build JSON payload { "<neuron_idx>": {...}, ... } preserving original order
        payload = {}
        for j, nid in enumerate(ids.tolist()):
            payload[str(int(nid))] = {
                "preactivations": Z_cluster[:, j].astype(float_dtype).tolist(),   # (group_size^2,)
                "w_out":          W_block[j, :].astype(float_dtype).tolist(),     # (group_size,)
                "contribs_to_logits": contribs[:, j, :].astype(float_dtype).tolist(),  # (group_size^2, group_size)
            }

        out_path = os.path.join(json_root, f"cluster_{freq}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f)

    return json_root

def apply_in_batches(model, params_seed, x_all, batch=4096):
    pre_acts_all_acc = None
    left_all, right_all = None, None
    for s in range(0, x_all.shape[0], batch):
        x = x_all[s:s+batch]
        _, preacts, left, right = model.apply({"params": params_seed}, x, training=False)
        preacts = [np.asarray(z) for z in preacts]
        if pre_acts_all_acc is None:
            pre_acts_all_acc = [p.copy() for p in preacts]
            left_all  = np.asarray(left)
            right_all = np.asarray(right)
        else:
            pre_acts_all_acc = [np.concatenate([a, b], axis=0) for a, b in zip(pre_acts_all_acc, preacts)]
            left_all  = np.concatenate([left_all,  np.asarray(left)],  axis=0)
            right_all = np.concatenate([right_all, np.asarray(right)], axis=0)
        jax.block_until_ready(left); jax.block_until_ready(right)
        del preacts, left, right, x
    return pre_acts_all_acc, left_all, right_all



def run_post_training_analysis(*,
    model,
    states,
    random_seed_ints: List[int],
    p: int,
    group_size: int,
    num_layers: int,
    mdir: str,
    mlp_class_lower: str,
    colour_rule = None,
    features: int | None = None,
    alive_by_layer_override: dict[int, list[list[int]]] | None = None,
):
    """
    Full post training analysis:
      1) prep DFT tools
      2) for every seed: cluster, generate report per layer
      3) generate cluster→logits' per-cluster JSON
      4) generate pdfs for cluster→logits
    write output in respective dir.
    """

    # DFT prep
    G, irreps = DFT.make_irreps_Dn(p)
    freq_map = {}
    for name, dim, R, freq in irreps:
        freq_map[name] = freq
        # print(f"Checking {name}...")
        
        # dihedral.check_representation_consistency(G, R, dihedral.mult, p)
    rho_cache = DFT.build_rho_cache(G, irreps)
    dft_fn    = DFT.jit_wrap_group_dft(rho_cache, irreps, group_size)
    subgroups = dihedral.enumerate_subgroups_Dn(p)

    # coset mask (for make_layer_report)
    coset_masks_L = dihedral.build_coset_masks(G, subgroups, dihedral.mult, p, side="left")
    coset_masks_R = dihedral.build_coset_masks(G, subgroups, dihedral.mult, p, side="right")
    prune_dir = mdir

    for seed_idx, seed in enumerate(random_seed_ints):
        print(f"\n=== Post-training analysis (alive-only, original IDs) for seed {seed} ===")
        gdir = paths.seed_graph_dir(mdir, seed)
        os.makedirs(gdir, exist_ok=True)

        # params for this seed
        params_seed = jax.tree_util.tree_map(lambda x: x[seed_idx], states.params)

        # load alive original ids for each layer (fallback = 0..width-1)
        if alive_by_layer_override is not None and seed in alive_by_layer_override:
            alive_by_layer = alive_by_layer_override[seed]
        else:
            alive_by_layer = _load_alive_indices_for_seed(
                prune_dir=mdir, features=features,
                seed=seed, num_layers=num_layers, params_seed=params_seed
            )

        # full-grid forward: get pre-activations (full width) and a/b contributions
        x_all = jnp.array([[g, h] for g in range(group_size) for h in range(group_size)],
                          dtype=jnp.int32)
        # _, pre_acts_all, left, right = model.apply({"params": params_seed}, x_all, training=False)
        pre_acts_all, left, right = apply_in_batches(model, params_seed, x_all, batch=4096)
        # tune batch to your GPU (2048–16384 works well)

        pre_acts_all = [np.asarray(z) for z in pre_acts_all]  # list[(p^2, width_L)]

        # ------ per-layer artifacts on alive-only columns, but keep ORIGINAL ids ------
        layers_freq: List[Dict[int, list]] = []  # {freq -> [original neuron ids]}
        cluster_tau = 1e-3
        thresh_small = 1.7 if group_size < 50 else 1.8

        for layer_idx in range(num_layers):
            prei_full = pre_acts_all[layer_idx]             # (p^2, width_full)
            alive_ids = alive_by_layer[layer_idx]           # original ids to keep

            if len(alive_ids) == 0:
                # Nothing alive in this layer — record empty info and continue
                layers_freq.append({})
                # still write an empty approx summary for consistency
                with open(os.path.join(gdir, f"approx_summary_layer{layer_idx+1}_p{p}.json"), "w") as f:
                    json.dump({}, f, indent=2)
                continue
            
            # slice to alive columns; keep variable names unchanged
            prei = prei_full[:, alive_ids]                  # (p^2, alive_count)
            prei_grid = prei.reshape(group_size, group_size, -1)
            left_alive  = left[:,  alive_ids]                        # (G^2, N_alive)
            right_alive = right[:, alive_ids]                        # (G^2, N_alive)
            assert prei_grid.shape[-1] == left_alive.shape[1] == right_alive.shape[1], (
                prei_grid.shape, left_alive.shape, right_alive.shape
            )
            artifacts = report.prepare_layer_artifacts(
                prei_grid, left_alive, right_alive, dft_fn, irreps, freq_map,
                prune_cfg={"thresh1": thresh_small, "thresh2": thresh_small, "seed": 0},
            )

            # map local alive indices back to ORIGINAL neuron ids
            local_clusters = artifacts.get("freq_cluster", {}) or {}
            clusters_layer = {
                freq: [alive_ids[j] for j in ids] for freq, ids in local_clusters.items()
            }
            layers_freq.append(clusters_layer)

            # approx summary (unchanged semantics)
            diag_labels = artifacts["diag_labels"]
            names = artifacts["names"]
            approx = report.summarize_diag_labels(diag_labels, p, names)
            with open(os.path.join(gdir, f"approx_summary_layer{layer_idx+1}_p{p}.json"), "w") as f:
                json.dump(approx, f, indent=2)

            # full report is also on alive-only activations
            report_dir = os.path.join(gdir, f"report_layer{layer_idx+1}")
            os.makedirs(report_dir, exist_ok=True)
            report.make_layer_report(
                prei_grid, left_alive, right_alive, p,
                dft_fn, irreps, coset_masks_L, coset_masks_R,
                report_dir, cluster_tau, colour_rule, artifacts,
            )

        # for layer_idx in range(num_layers-2):
        #     plot_Wblock_pca_by_srcfreq_color_by_tgtfreq(
        #         params_seed=params_seed,
        #         layers_freq=layers_freq,
        #         alive_by_layer=alive_by_layer,
        #         gdir=gdir,
        #         p=p,
        #         seed=seed,
        #         layer_from=layer_idx+1,
        #     )


        # ------ cluster → logits (use ORIGINAL ids to slice last layer & W_out) ------
        preacts, X_in, weights_by_layer, cluster_contribs, cluster_W_blocks = get_all_preacts_and_embeddings(
            model=model, params=params_seed, group_size=group_size, clusters_by_layer=layers_freq,
        )

        # write per-cluster JSON (keys are ORIGINAL ids)
        pdf_root = os.path.join(gdir, "pdf_plots", f"seed_{seed}")
        os.makedirs(pdf_root, exist_ok=True)
        json_root = make_some_jsons(
            preacts=preacts,
            group_size=group_size,
            clusters_by_layer=layers_freq,            # ORIGINAL ids
            cluster_weights_to_logits=cluster_W_blocks,
            cluster_contribs_to_logits=cluster_contribs,
            save_dir=pdf_root,
            sanity_check=True,
        )
        print(f"Wrote cluster JSONs to {json_root}")

        # PDF (optional)
        num_pc = 4 if "cheating" not in mlp_class_lower else 2
        for freq, C_freq in cluster_contribs.items():
            generate_pdf_plots_for_matrix(
                C_freq, p,
                save_dir=pdf_root, seed=seed,
                freq_list=[freq],
                tag=f"cluster_contributions_to_logits_freq={freq}",
                tag_q="full",
                colour_rule=colour_quad_mod_g,
                class_string=mlp_class_lower,
                num_principal_components=num_pc,
            )
        print(f"PDF plots written → {pdf_root}")