# controllers/run_training_Transformer.py
from analysis.prune_Transformer import prune_two_stage_by_accuracy_batched_transformer
import os
import sys
import json
import gc
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from controllers.config_Transformer import Config
from controllers.paths_Transformer import base_dir, model_dir, seed_graph_dir
import controllers.training_prep_Transformer as prep
import analysis.while_training_analysis_gen as wta
import analysis.post_training_analysis_Transformer as pta
from color_rules import colour_quad_a_only


def main(argv):
    print("start args parsing")
    cfg = Config.from_argv(argv)
    print(
        f"args: lr={cfg.learning_rate}, wd={cfg.weight_decay}, "
        f"d_model={cfg.d_model}, heads={cfg.num_heads}x{cfg.d_head} "
        f"(ctx={cfg.n_ctx}), mlp_layers={cfg.num_mlp_layers}, "
        f"nn_mult={cfg.nn_multiplier}, seeds={cfg.random_seeds}"
    )

    training_set_size = cfg.k * cfg.batch_size
    group_size = 2 * cfg.p

    # ---- Paths (mirror MLP runner structure) ----
    model_tag = f"transformer_{cfg.num_mlp_layers}"
    base = base_dir(cfg.p, model_tag, cfg.d_model, cfg.k)
    mdir = model_dir(
        cfg.p,
        cfg.batch_size,
        cfg.weight_decay,
        cfg.epochs,
        training_set_size,
        base,
        cfg.d_model,
        cfg.nn_multiplier,
        cfg.num_mlp_layers,
        cfg.attn_coeff,
    )

    # ---- Utility: process seeds in small chunks (as in MLP) ----
    def chunks(lst: List[int], n: int):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    seed_chunk_size = 2  # keep small unless you’re sure about memory

    for seed_group in chunks(cfg.random_seeds, seed_chunk_size):
        print(f"\n=== Training seed group: {seed_group} ===")

        # ---- Prep: dataset / model+opt / states ----
        (
            train_ds_list,
            x_train_batches,
            y_train_batches,
            x_eval_batches,           # CHANGED: name aligned with MLP
            y_eval_batches,           # CHANGED: name aligned with MLP
        ) = prep.make_train_and_test_batches(
            p=cfg.p,
            batch_size=cfg.batch_size,
            k=cfg.k,
            random_seed_ints=seed_group,
            test_batch_size=None,
            shuffle_test=False,
            drop_remainder=False,
        )
        x_eval_full, y_eval_full = prep.make_full_eval_grid(cfg.p)
        print("made dataset")

        model = prep.build_model(cfg, group_size)
        print("model made")
        tx = prep.build_optimizer(cfg.optimizer, cfg.learning_rate)
        states, init_metrics = prep.create_states(
            model, tx, cfg.weight_decay, cfg.batch_size, seed_group
        )

        # ---- Train with periodic eval (same helper as MLP) ----
        states, logs_by_seed, first_100, first_loss, first_ce = wta.run_epochs_scaling(
            model=model,  # needed by wta for some analyses
            states=states,
            x_batches=x_train_batches,
            y_batches=y_train_batches,
            init_metrics=init_metrics,
            random_seed_ints=seed_group,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            eval_batches=(x_eval_batches, y_eval_batches),   # CHANGED: align with MLP path
            eval_every=1,
        )

        # ---- Persist epoch logs exactly like MLP does ----
        # (uses the Transformer pta.save_epoch_logs wrapper)
        pta.save_epoch_logs(logs_by_seed, mdir, cfg.d_model)

        # ---- Final eval summary (same API as MLP pta) ----
        final_results = pta.final_eval_all_models(             
            states=states,
            x_eval_batches=x_eval_batches,
            y_eval_batches=y_eval_batches,
            init_metrics=init_metrics,
            random_seed_ints=seed_group,
        )
        pta.save_final_logs(final_results, mdir, cfg.d_model)
        for seed, res in final_results.items():                
            print(
                f"[Seed {seed}] Final Test: loss={res['loss']:.6f} "
                f"acc={res['accuracy']*100:.2f}% l2={res['l2_loss']:.6f}"
            )

        # ---- Post-training analysis (Transformer version) ----
        
        # Only analyze seeds that reached 100% accuracy
        seed_to_idx = {s: i for i, s in enumerate(seed_group)}
        batched_params = states.params  # (stacked over seeds)
        prune_reports = {}              # optional: keep reports

        for seed in seed_group:
            res = final_results[seed]
            if not res.get("reach_100%_test", False):
                continue  # skip seeds that didn't reach ~100% on test

            i = seed_to_idx[seed]
            params_i = jax.tree_util.tree_map(lambda x: x[i], batched_params)

            pruned_params_i, report_i = prune_two_stage_by_accuracy_batched_transformer(
                model=model,
                params=params_i,
                full_x=x_eval_full,
                full_y=y_eval_full,
                num_mlp_layers=cfg.num_mlp_layers,
                batch_size=cfg.batch_size,
                abs_acc_th=0.005,    # same defaults as MLP
                hard_min_acc=1.0,
            )
            prune_reports[seed] = report_i
            batched_params = jax.tree_util.tree_map(lambda bp, sp: bp.at[i].set(sp),
                                            batched_params, pruned_params_i)

        states = states.replace(params=batched_params)
        # persist prune logs
        pta.save_prune_logs(prune_reports, mdir, cfg.d_model)

        good_seeds = [s for s, res in final_results.items() if res.get("reach_100%_test")]

        if not good_seeds:
            print("[PTA] Skipping post-training analysis: no seeds reached 100% accuracy.")
        else:
            # Subset states so index 0..len(good_seeds)-1 corresponds to good_seeds order
            idxs = np.array([seed_to_idx[s] for s in good_seeds], dtype=int)
            states_subset = jax.tree_util.tree_map(lambda x: x[idxs], states)

            print(f"[PTA] Running post-training analysis for seeds {good_seeds} "
                  f"(skipping {[s for s in seed_group if s not in good_seeds]}).")
            alive_by_layer_override = {s: rep["alive_final"] for s, rep in prune_reports.items()}

            pta.run_post_training_analysis(
                model=model,
                states=states_subset,
                random_seed_ints=good_seeds,
                p=cfg.p,
                group_size=group_size,
                num_layers=cfg.num_mlp_layers,
                mdir=mdir,
                class_lower=f"transformer_{cfg.num_mlp_layers}",
                colour_rule=colour_quad_a_only,
                write_json=True,
                write_pdfs=True,
                dmodel=cfg.d_model,
                alive_by_layer_override=alive_by_layer_override,
            )

        # ---- Free memory between chunks (as in MLP) ----
        del states, x_train_batches, y_train_batches, x_eval_batches, y_eval_batches
        gc.collect()
        jax.clear_caches()

    print("\nAll seed groups completed.")


if __name__ == "__main__":
    main(sys.argv)
