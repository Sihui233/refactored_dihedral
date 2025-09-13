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
