# controllers/run_training.py
"""
High-level training runner that mirrors your monolithic script's flow,
but uses the split modules:
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


import jax
import jax.numpy as jnp


from controllers.config_MLP import Config
from controllers.paths_MLP import base_dir, model_dir, seed_graph_dir

# Prep / training / post-training modules
import controllers.training_prep_MLP as prep
import analysis.while_training_analysis_gen as wta
import analysis.post_training_analysis_MLP as pta

from color_rules import colour_quad_a_only


def main(argv):
    print("start args parsing")
    cfg = Config.from_argv(argv)
    print(f"args: lr: {cfg.learning_rate}, wd: {cfg.weight_decay},nn: {cfg.num_neurons}, features: {cfg.features}, num_layer: {cfg.num_layers}")
    print(f"Random seeds: {cfg.random_seeds}")
    mlp_class_lower = f"{cfg.MLP_class.lower()}_{cfg.num_layers}"
    training_set_size = cfg.k * cfg.batch_size
    group_size = 2* cfg.p

    # ---- Prep: dataset / eval grid / model+opt / states ----
    train_ds_list, x_batches, y_batches = prep.make_train_batches(cfg.p, cfg.batch_size, cfg.k, cfg.random_seeds)
    print("made dataset")
    x_eval, y_eval = prep.make_full_eval_grid(cfg.p)
    x_eval_batches, y_eval_batches = prep.pad_to_batches(x_eval, y_eval, cfg.batch_size, len(cfg.random_seeds))
    print("eval grid:", x_eval.shape, "batches:", x_eval_batches.shape, "\n")

    model = prep.build_model(cfg.MLP_class, cfg.num_layers, cfg.num_neurons, cfg.features, group_size)
    print("model made")
    tx = prep.build_optimizer(cfg.optimizer, cfg.learning_rate)
    states, init_metrics = prep.create_states(model, tx, cfg.weight_decay, cfg.batch_size, cfg.random_seeds)

    # ---- Paths ----
    base = base_dir(cfg.p, mlp_class_lower, cfg.num_neurons, cfg.features, cfg.k)
    mdir = model_dir(cfg.p, cfg.batch_size, cfg.num_neurons, cfg.weight_decay, cfg.epochs, training_set_size, base, cfg.features)

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
    #     eval_flat=(x_eval, y_eval), # for margins
    #     # eval_fn=eval_fn, # you could pass eval_batches=(x_eval_batches, y_eval_batches) instead
    # )

    
    states, logs_by_seed, first_100, first_loss, first_ce = wta.run_epochs_scaling(
        model=model,
        states=states,
        x_batches=x_batches,
        y_batches=y_batches,
        init_metrics=init_metrics,
        random_seed_ints=cfg.random_seeds,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
        eval_batches=(x_eval_batches, y_eval_batches),
        eval_every=1,
        eval_flat=(x_eval, y_eval), # for margins
        # eval_fn=eval_fn, # you could pass eval_batches=(x_eval_batches, y_eval_batches) instead
    )


    # Persist epoch logs now (JSON per seed)
    pta.save_epoch_logs(logs_by_seed, mdir, cfg.features)


    # Final eval summary (optional if run_epochs already logged final test)
    final_results = pta.final_eval_all_models(
                                            states=states,
                                            x_eval_batches=x_eval_batches,
                                            y_eval_batches=y_eval_batches,
                                            init_metrics=init_metrics,
                                            random_seed_ints=cfg.random_seeds,
                                            )
    for seed, res in final_results.items():
        print(f"[Seed {seed}] Final Test: loss={res['loss']:.6f} acc={res['accuracy']*100:.2f}% l2={res['l2_loss']:.6f}")
    
    pta.run_post_training_analysis(
        model=model,
        states=states,
        random_seed_ints=cfg.random_seeds,
        p=cfg.p,
        group_size=group_size,
        num_layers=cfg.num_layers,
        mdir=mdir,
        mlp_class_lower=mlp_class_lower,
        colour_rule=colour_quad_a_only
    )


if __name__ == "__main__":
    main(sys.argv)
