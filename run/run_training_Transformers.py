# controllers/run_training_Transformer.py
import os
import sys
from typing import List, Dict

import gc
import jax
import jax.numpy as jnp

from controllers.config_Transformer import Config
from controllers.paths_Transformer import base_dir, model_dir, seed_graph_dir
import controllers.training_prep_Transformer as prep
import analysis.while_training_analysis_gen as wta
import analysis.post_training_analysis_Transformer as pta


def main(argv):
    print("start args parsing")
    cfg = Config.from_argv(argv)

    print(
        f"args: lr={cfg.learning_rate}, wd={cfg.weight_decay}, "
        f"d_model={cfg.d_model}, heads={cfg.num_heads}x{cfg.d_head} "
        f"(ctx={cfg.n_ctx}), mlp_layers={cfg.num_mlp_layers}, "
        f"nn_mult={cfg.nn_multiplier}, optimizer={cfg.optimizer}, "
        f"seeds={cfg.random_seeds}"
    )

    training_set_size = cfg.k * cfg.batch_size
    group_size = 2 * cfg.p
    model_tag = f"transformer_{cfg.num_mlp_layers}"

    # ---------- small helper: chunk seeds to control memory ----------
    def chunks(lst: List[int], n: int):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    # tune if you want larger groups
    seed_chunk_size = 2

    for seed_group in chunks(cfg.random_seeds, seed_chunk_size):
        # ====== 1) Prep dataset / eval grid (test is complement) ======
        (
            train_ds_list,
            x_train_batches,
            y_train_batches,
            x_test_batches,
            y_test_batches,
        ) = prep.make_train_and_test_batches(
            p=cfg.p,
            batch_size=cfg.batch_size,
            k=cfg.k,
            random_seed_ints=seed_group,
            test_batch_size=None,  # default B_test == train B
            shuffle_test=False,
            drop_remainder=False,
        )

        # ====== 2) Build model / optimizer / per-seed states ======
        model = prep.build_model(cfg, group_size)
        tx = prep.build_optimizer(cfg.optimizer, cfg.learning_rate)
        states, init_metrics = prep.create_states(
            model, tx, cfg.weight_decay, cfg.batch_size, seed_group
        )

        # ====== 3) Paths (mirror MLP runner’s structure) ======
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

        # ====== 4) Train with periodic eval & logging (scaling loop) ======
        states, logs_by_seed, first_100, first_loss, first_ce = wta.run_epochs_scaling(
            model=model,
            states=states,
            x_batches=x_train_batches,
            y_batches=y_train_batches,
            init_metrics=init_metrics,
            random_seed_ints=seed_group,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            eval_every=1,  # like MLP runner; make configurable if desired
            eval_batches=(x_test_batches, y_test_batches),
            # or: eval_fn=lambda s: prep.eval_model(s, x_test_batches, y_test_batches, init_metrics)
        )

        # ====== 5) Persist per-epoch logs (same pattern as MLP) ======
        # (Transformer save helper mirrors MLP’s, filename uses d_model in place of features)
        pta.save_epoch_logs(logs_by_seed, mdir, cfg.d_model)

        # ====== 6) Final evaluation summary (per seed) ======
        final_results = pta.final_eval_all_models(
            states=states,
            x_eval_batches=x_test_batches,
            y_eval_batches=y_test_batches,
            init_metrics=init_metrics,
            random_seed_ints=seed_group,
        )
        for seed, res in final_results.items():
            print(
                f"[Seed {seed}] Final Test: loss={res['loss']:.6f} "
                f"acc={res['accuracy']*100:.2f}% l2={res['l2_loss']:.6f}"
            )

        # ====== 7) Free memory between seed chunks ======
        del (
            states,
            x_train_batches,
            y_train_batches,
            x_test_batches,
            y_test_batches,
            train_ds_list,
        )
        gc.collect()
        jax.clear_caches()


if __name__ == "__main__":
    main(sys.argv)
