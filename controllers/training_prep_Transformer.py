# controllers/run_training_Transformer.py
import os
import sys
import json
import gc
from typing import Dict, List

import jax
import jax.numpy as jnp

from controllers.config_Transformer import Config
from controllers.paths_Transformer import base_dir, model_dir, seed_graph_dir
import controllers.training_prep_Transformer as prep
import analysis.while_training_analysis_gen as wta


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

    # ---- Paths (match MLP runner’s idea of a stable base + model dir) ----
    # Keep transformer-specific tag, but directory “architecture” mirrors MLP:
    # base_dir(...), then a parameterized model_dir(...), then logs under mdir/logs_*
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

    # ---- Utility: process seeds in small chunks (like MLP runner) ----
    def chunks(lst: List[int], n: int):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    # You can tweak this based on memory; MLP runner used 1–2
    seed_chunk_size = 2

    for seed_group in chunks(cfg.random_seeds, seed_chunk_size):
        print(f"\n=== Training seed group: {seed_group} ===")

        # ---- Prep: dataset / model+opt / states ----
        # Use explicit train/test batch builders (transformer prep module)
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
            test_batch_size=None,     # default = train batch size
            shuffle_test=False,       # stable test ordering like MLP runner
            drop_remainder=False,     # keep all test samples
        )
        print("made dataset")

        model = prep.build_model(cfg, group_size)
        print("model made")
        tx = prep.build_optimizer(cfg.optimizer, cfg.learning_rate)
        states, init_metrics = prep.create_states(
            model, tx, cfg.weight_decay, cfg.batch_size, seed_group
        )

        # Eval fn if you ever want the callable path; here we use eval_batches below
        def eval_fn(current_states):
            return prep.eval_model(current_states, x_test_batches, y_test_batches, init_metrics)

        # ---- Train with periodic eval & extra logging (scaling runner) ----
        states, logs_by_seed, first_100, first_loss, first_ce = wta.run_epochs_scaling(
            model=model,                    # <- needed for Dirichlet auto-dispatch
            states=states,
            x_batches=x_train_batches,
            y_batches=y_train_batches,
            init_metrics=init_metrics,
            random_seed_ints=seed_group,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            eval_batches=(x_test_batches, y_test_batches),   # like MLP runner
            eval_every=1,
            # You could pass eval_fn=eval_fn instead of eval_batches if preferred
        )

        # ---- Save per-epoch logs (same architecture as MLP) ----
        # MLP uses pta.save_epoch_logs(mdir, ...). We mirror that structure:
        # mdir/logs_<key>=<value>/epochs_seed_<seed>.json + final_summary.json
        logs_root = os.path.join(mdir, f"logs_dmodel={cfg.d_model}")
        os.makedirs(logs_root, exist_ok=True)

        for seed, logs in logs_by_seed.items():
            out = os.path.join(logs_root, f"epochs_seed_{seed}.json")
            with open(out, "w") as f:
                json.dump(logs, f, indent=2)
            print(f"[saved] {out}")

        # ---- Final eval summary (per-seed), same idea as MLP’s pta.final_eval_all_models ----
        te = eval_fn(states)
        summary: Dict[int, Dict] = {}
        for i, seed in enumerate(seed_group):
            mi = jax.tree_util.tree_map(lambda x: x[i], te).compute()
            summary[seed] = dict(
                loss=float(mi["loss"]),
                l2_loss=float(mi["l2_loss"]),
                accuracy=float(mi["accuracy"]),
            )
            print(
                f"[Seed {seed}] Final Test: loss={summary[seed]['loss']:.6f} "
                f"acc={summary[seed]['accuracy']*100:.2f}% "
                f"l2={summary[seed]['l2_loss']:.6f}"
            )

        final_path = os.path.join(logs_root, "final_summary.json")
        # If multiple chunks run, merge/append results across chunks
        if os.path.exists(final_path):
            with open(final_path, "r") as f:
                prev = json.load(f)
        else:
            prev = {}
        prev.update(summary)
        with open(final_path, "w") as f:
            json.dump(prev, f, indent=2)
        print(f"[saved] {final_path}")

        # ---- Free memory between chunks (like MLP runner) ----
        del states, x_train_batches, y_train_batches, x_test_batches, y_test_batches
        gc.collect()
        jax.clear_caches()

    print("\nAll seed groups completed.")


if __name__ == "__main__":
    main(sys.argv)
