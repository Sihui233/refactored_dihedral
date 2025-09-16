# controllers/run_training.py
"""
High-level training runner that mirrors your monolithic script's flow, but uses the split modules:

• training_prep_MLP → data/model/optimizer/state utils
• while_training_analysis_MLP → train loop + periodic eval + logging
• post_training_analysis_MLP → final eval + post-training helpers

Pipeline:
  1) Prep data/model/optimizer/states
  2) Train with periodic eval via wta.run_epochs(eval_fn=...)
  3) Save per-epoch logs
  4) Run post training analysis
"""
import os
import sys
from typing import List, Dict

try:
    import jax
    if all(d.platform != "gpu" for d in jax.devices()):
        print("⚠️ No GPU detected — enabling multithreading for CPU.")
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=10"
except Exception:
    # If JAX isn't installed or fails to load, fall back safely
    pass

import jax.numpy as jnp

from controllers.config_MLP import Config
from controllers.paths_MLP import base_dir, model_dir, seed_graph_dir

# Prep / training / post-training modules
import controllers.training_prep_MLP as prep
import analysis.while_training_analysis_gen as wta
import analysis.post_training_analysis_MLP as pta
from analysis.prune_MLP import prune_two_stage_by_accuracy_batched
from color_rules import colour_quad_a_only

from itertools import islice
import gc


def main(argv):
    print("start args parsing")
    cfg = Config.from_argv(argv)
    print(
        f"args: lr: {cfg.learning_rate}, wd: {cfg.weight_decay},"
        f"nn: {cfg.num_neurons}, features: {cfg.features}, num_layer: {cfg.num_layers}"
    )
    print(f"Random seeds: {cfg.random_seeds}")

    mlp_class_lower = f"{cfg.MLP_class.lower()}_{cfg.num_layers}"
    training_set_size = cfg.k * cfg.batch_size
    group_size = 2 * cfg.p

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    chunk_size = 2  # try 1–2; increase if memory allows

    for seed_group in chunks(cfg.random_seeds, chunk_size):
        # ---- Prep: dataset / eval grid / model+opt / states ----
        ## eval not full grid
        (
            train_ds_list,
            x_batches,
            y_batches,
            x_eval_batches,
            y_eval_batches,
        ) = prep.make_train_and_test_batches(
            p=cfg.p,
            batch_size=cfg.batch_size,
            k=cfg.k,
            random_seed_ints=seed_group,
            test_batch_size=None,  # defaults to same as train B
            shuffle_test=False,
            drop_remainder=False,
        )

        ## if full grid eval
        x_eval_full, y_eval_full = prep.make_full_eval_grid(cfg.p)

        # x_eval_batches, y_eval_batches = prep.pad_to_batches(x_eval, y_eval, cfg.batch_size, len(cfg.random_seeds))
        # print("eval grid:", x_eval.shape, "batches:", x_eval_batches.shape, "\n")

        print("made dataset")

        model = prep.build_model(
            cfg.MLP_class,
            cfg.num_layers,
            cfg.num_neurons,
            cfg.features,
            group_size,
        )
        print("model made")

        tx = prep.build_optimizer(cfg.optimizer, cfg.learning_rate)
        states, init_metrics = prep.create_states(
            model, tx, cfg.weight_decay, cfg.batch_size, seed_group
        )

        # ---- Paths ----
        base = base_dir(cfg.p, mlp_class_lower, cfg.num_neurons, cfg.features, cfg.k)
        mdir = model_dir(
            cfg.p,
            cfg.batch_size,
            cfg.num_neurons,
            cfg.learning_rate,
            cfg.weight_decay,
            cfg.epochs,
            training_set_size,
            base,
            cfg.features,
        )

        def eval_fn(current_states):
            return prep.eval_model(current_states, x_eval_batches, y_eval_batches, init_metrics)

        # states, logs_by_seed, first_100, first_loss, first_ce = wta.run_epochs(
        #     states=states,
        #     x_batches=x_batches,
        #     y_batches=y_batches,
        #     init_metrics=init_metrics,
        #     random_seed_ints=cfg.random_seeds,
        #     weight_decay=cfg.weight_decay,
        #     epochs=cfg.epochs,
        #     eval_every=cfg.eval_every if hasattr(cfg, "eval_every") and cfg.eval_every else 5000,
        #     # eval_flat=(x_eval, y_eval),  # for margins
        #     # eval_fn=eval_fn,             # you could pass eval_batches=(x_eval_batches, y_eval_batches) instead
        # )

        states, logs_by_seed, first_100, first_loss, first_ce = wta.run_epochs_scaling(
            model=model,
            states=states,
            x_batches=x_batches,
            y_batches=y_batches,
            init_metrics=init_metrics,
            random_seed_ints=seed_group,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            eval_batches=(x_eval_batches, y_eval_batches),
            eval_every=1,
            # eval_flat=(x_eval, y_eval),  # for margins
            # eval_fn=eval_fn,             # you could pass eval_batches=(x_eval_batches, y_eval_batches) instead
        )

        # Persist epoch logs now (JSON per seed)
        pta.save_epoch_logs(logs_by_seed, mdir, cfg.features)

        # Final eval summary (optional if run_epochs already logged final test)
        final_results = pta.final_eval_all_models(
            states=states,
            x_eval_batches=x_eval_batches,
            y_eval_batches=y_eval_batches,
            init_metrics=init_metrics,
            random_seed_ints=seed_group,
        )
        pta.save_final_logs(final_results, mdir, cfg.features)

        for seed, res in final_results.items():
            print(
                f"[Seed {seed}] Final Test: loss={res['loss']:.6f} "
                f"acc={res['accuracy']*100:.2f}% l2={res['l2_loss']:.6f}"
            )

        # Map seed -> index in this seed_group
        seed_to_idx = {s: i for i, s in enumerate(seed_group)}
        batched_params = states.params  # (stacked over seeds)
        prune_reports: Dict[int, Dict] = {}

        # optional: keep reports
        for seed in seed_group:
            res = final_results[seed]
            if not res.get("reach_100%_test", False):
                continue  # skip seeds that didn't reach ~100% on test

            i = seed_to_idx[seed]
            params_i = jax.tree_util.tree_map(lambda x: x[i], batched_params)

            pruned_params_i, report_i = prune_two_stage_by_accuracy_batched(
                model=model,
                params=params_i,
                full_x=x_eval_full,
                full_y=y_eval_full,
                batch_size=cfg.batch_size,
                abs_acc_th=0.005,
                hard_min_acc=1.0,
            )
            prune_reports[seed] = report_i

            # Write pruned single-seed params back into the batched params
            batched_params = jax.tree_util.tree_map(
                lambda bp, sp: bp.at[i].set(sp), batched_params, pruned_params_i
            )

        pta.save_prune_logs(prune_reports, mdir, cfg.features)

        # Replace params in TrainState (flax struct dataclass supports .replace)
        states = states.replace(params=batched_params)

        # ---- Post-training analysis only on seeds that hit ~100% ----
        good_seeds = [s for s in seed_group if final_results[s].get("reach_100%_test", False)]
        if good_seeds:
            idxs = jnp.array([seed_to_idx[s] for s in good_seeds])
            states_subset = jax.tree_util.tree_map(lambda x: x[idxs], states)
            alive_by_layer_override = {s: rep["alive_final"] for s, rep in prune_reports.items()}
            
            pta.run_post_training_analysis(
                model=model,
                states=states_subset,
                random_seed_ints=good_seeds,
                p=cfg.p,
                group_size=group_size,
                num_layers=cfg.num_layers,
                mdir=mdir,
                mlp_class_lower=mlp_class_lower,
                colour_rule=colour_quad_a_only,
                features=cfg.features,
                alive_by_layer_override=alive_by_layer_override,
            )
        else:
            print("[PTA] Skipping post-training analysis: no seeds reached the accuracy gate.")

        # free memory between chunks
        del states, x_batches, y_batches, x_eval_batches, y_eval_batches
        gc.collect()
        jax.clear_caches()


if __name__ == "__main__":
    main(sys.argv)
