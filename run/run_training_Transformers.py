# controllers/run_training_Transformer.py
import os, sys, json
from typing import Dict
import jax
import jax.numpy as jnp

from controllers.config_Transformer import Config
from controllers.paths_Transformer import base_dir, model_dir, seed_graph_dir
import controllers.training_prep_Transformer as prep
import analysis.while_training_analysis_gen.py as wta

def main(argv):
    print("start args parsing")
    cfg = Config.from_argv(argv)
    print(f"args: lr={cfg.learning_rate}, wd={cfg.weight_decay}, d_model={cfg.d_model}, "
          f"heads={cfg.num_heads}x{cfg.d_head} (ctx={cfg.n_ctx}), mlp_layers={cfg.num_mlp_layers}, "
          f"nn_mult={cfg.nn_multiplier}, seeds={cfg.random_seeds}")
    training_set_size = cfg.k * cfg.batch_size
    group_size = 2 * cfg.p

    # ---- Prep data/eval/model/opt/states ----
    train_ds_list, x_batches, y_batches = prep.make_train_batches(cfg.p, cfg.batch_size, cfg.k, cfg.random_seeds)
    x_eval, y_eval = prep.make_full_eval_grid(cfg.p)
    x_eval_batches, y_eval_batches = prep.pad_to_batches(x_eval, y_eval, cfg.batch_size, len(cfg.random_seeds))

    model = prep.build_model(cfg, group_size)
    tx = prep.build_optimizer(cfg.optimizer, cfg.learning_rate)
    states, init_metrics = prep.create_states(model, tx, cfg.weight_decay, cfg.batch_size, cfg.random_seeds)

    # ---- Paths ----
    model_tag = f"transformer_{cfg.num_mlp_layers}"
    base = base_dir(cfg.p, model_tag, cfg.d_model, cfg.k)
    mdir = model_dir(cfg.p, cfg.batch_size, cfg.weight_decay, cfg.epochs, training_set_size,
                     base, cfg.d_model, cfg.nn_multiplier, cfg.num_mlp_layers, cfg.attn_coeff)

    def eval_fn(current_states):
        return prep.eval_model(current_states, x_eval_batches, y_eval_batches, init_metrics)

    # ---- Train ----
    states, logs_by_seed, first_100, first_loss, first_ce = wta.run_epochs(
        states=states,
        x_batches=x_batches,
        y_batches=y_batches,
        init_metrics=init_metrics,
        random_seed_ints=cfg.random_seeds,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
        eval_every=cfg.eval_every,
        eval_fn=eval_fn,
    )

    # ---- Save per-epoch logs (每个 seed 一个 json) ----
    logs_root = os.path.join(mdir, f"logs_dmodel={cfg.d_model}")
    os.makedirs(logs_root, exist_ok=True)
    for seed, logs in logs_by_seed.items():
        out = os.path.join(logs_root, f"epochs_seed_{seed}.json")
        with open(out, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"[saved] {out}")

    # ---- Final eval summary ----
    te = eval_fn(states)
    summary: Dict[int, Dict] = {}
    for i, seed in enumerate(cfg.random_seeds):
        mi = jax.tree_util.tree_map(lambda x: x[i], te).compute()
        summary[seed] = dict(loss=float(mi["loss"]),
                             l2_loss=float(mi["l2_loss"]),
                             accuracy=float(mi["accuracy"]))
        print(f"[Seed {seed}] Final Test: loss={summary[seed]['loss']:.6f} "
              f"acc={summary[seed]['accuracy']*100:.2f}% l2={summary[seed]['l2_loss']:.6f}")

    with open(os.path.join(logs_root, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main(sys.argv)
