# analysis/post_training_analysis.py
import os
import json
from typing import Dict, List, Tuple
import numpy as np
import jax
import jax.numpy as jnp

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
from pca_diffusion_plots_w_helpers import generate_pdf_plots_for_matrix

# Colour rules (pick the one you want for PDFs)
from color_rules import colour_quad_a_only, colour_quad_mod_g

def final_eval_all_models(*, states, x_eval_batches, y_eval_batches, init_metrics, random_seed_ints: List[int]):
    """Compute final test metrics for each model over full group grid."""
    from controllers.training_prep_MLP import eval_model
    test_metrics = eval_model(states, x_eval_batches, y_eval_batches, init_metrics)
    results = {}
    for i, seed in enumerate(random_seed_ints):
        tm = jax.tree_util.tree_map(lambda x: x[i], test_metrics).compute()
        results[seed] = {
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
        print(f"Final log for seed {seed} saved to {path}")
        
        


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
        A list of length `model.num_layers`, where
        `preacts[L]` has shape `(group_size², width_L)` of the raw pre-ReLU activations.
    X_in : np.ndarray
        The `(group_size², d_in)` input matrix formed by all (a,b) embedding pairs.
    weights : list[np.ndarray]
        Hidden-layer weight kernels; `weights[L]` has shape
        `(in_dim_L, width_L)`.
    cluster_contribs : dict[int, np.ndarray]
        For every frequency `f` in the **last** hidden layer this returns
        a matrix of shape `(group_size², group_size)`:
    cluster_weights : dict[int, np.ndarray]
        For every frequency `f` in the last hidden layer, this is the slice
        of the output-layer kernel that feeds the logits from the neurons
        in cluster `f`.  Shape: `(|cluster_f|, group_size)`.
    
            H_cluster @ W_block
        where
        - `H_cluster` is the ReLU’d activations of the neurons in cluster `f`
          at every of the `group_size²` inputs, and
        - `W_block` is the slice of the output layer’s weight matrix
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
            # (optional) log: print(f"[make_some_jsons] freq={freq}: dropped invalid neuron ids {bad}")

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
        print(f"Checking {name}...")
        
        dihedral.check_representation_consistency(G, R, dihedral.mult, p)
    rho_cache = DFT.build_rho_cache(G, irreps)
    dft_fn    = DFT.jit_wrap_group_dft(rho_cache, irreps, group_size)
    subgroups = dihedral.enumerate_subgroups_Dn(p)

    # coset mask (for make_layer_report)
    coset_masks_L = dihedral.build_coset_masks(G, subgroups, dihedral.mult, p, side="left")
    coset_masks_R = dihedral.build_coset_masks(G, subgroups, dihedral.mult, p, side="right")

    for seed_idx, seed in enumerate(random_seed_ints):
        print(f"\n=== Post-training analysis for seed {seed} ===")
        gdir = paths.seed_graph_dir(mdir, seed)
        os.makedirs(gdir, exist_ok=True)

        # get params for seed
        params_seed = jax.tree_util.tree_map(lambda x: x[seed_idx], states.params)

        # full grid forwar to get pre-acts + left (a-contrib), right (b-contrib)
        x_all = jnp.array([[g, h] for g in range(group_size) for h in range(group_size)], dtype=jnp.int32)
        _, pre_acts_all, left, right = model.apply({"params": params_seed}, x_all, training=False)

        # cluster + report generation
        layers_freq: List[Dict[int, list]] = []  # each layer：freq -> [neuron ids]
        cluster_tau = 1e-3
        thresh_small = 2.0 if group_size < 50 else 3.0

        for layer_idx in range(num_layers):
            prei      = pre_acts_all[layer_idx]
            prei_grid = prei.reshape(group_size, group_size, -1)

            artifacts = report.prepare_layer_artifacts(
                prei_grid, left, right, dft_fn, irreps, freq_map,
                prune_cfg={"thresh1": thresh_small, "thresh2": thresh_small, "seed": 0},
            )
            report.make_layer_report(
                prei_grid, left, right, p,
                dft_fn, irreps, coset_masks_L, coset_masks_R,
                gdir, cluster_tau, colour_rule, artifacts,
            )

            # per layer cluster based on freq
            clusters_layer = artifacts["freq_cluster"]  # {freq: [neuron ids]}
            layers_freq.append(clusters_layer)

            # approx summary per layer
            diag_labels = artifacts["diag_labels"]
            names = artifacts["names"]
            approx = report.summarize_diag_labels(diag_labels, p, names)
            with open(os.path.join(gdir, f"approx_summary_layer{layer_idx+1}_p{p}.json"), "w") as f:
                json.dump(approx, f, indent=2)

        # preacts + cluster→logits
        preacts, X_in, weights_by_layer, cluster_contribs, cluster_W_blocks = get_all_preacts_and_embeddings(
            model=model, params=params_seed, group_size=group_size, clusters_by_layer=layers_freq,
        )

        # write per-cluster JSON
        pdf_root = os.path.join(gdir, "pdf_plots", f"seed_{seed}")
        os.makedirs(pdf_root, exist_ok=True)
        json_root = make_some_jsons(
            preacts=preacts,
            group_size=group_size,
            clusters_by_layer=layers_freq,
            cluster_weights_to_logits=cluster_W_blocks,
            cluster_contribs_to_logits=cluster_contribs,  # sanity check 可开关
            save_dir=pdf_root,
            sanity_check=True,
        )
        print(f"Wrote cluster JSONs to {json_root}")

        # draw cluster→logits PDF
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