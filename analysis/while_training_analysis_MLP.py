# while_training_analysis_MLP.py
from typing import Dict, List, Tuple, Optional, Callable
import jax
import jax.numpy as jnp
from controllers.training_prep_MLP import train_epoch, eval_model

EvalFn = Callable[[any], any]  # eval_fn(states) -> metrics collection (per-model)

def _metrics_to_scalars(m):
    c = m.compute()
    return float(c["loss"]), float(c["l2_loss"]), float(c["accuracy"])

def run_epochs(*,
               states,
               x_batches, y_batches,
               init_metrics,
               random_seed_ints: List[int],
               weight_decay: float,
               epochs: int,
               eval_every: int = 5000,
               eval_batches: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
               eval_fn: Optional[EvalFn] = None,
               ):
    assert not (eval_batches is not None and eval_fn is not None), \
        "Pass either eval_batches or eval_fn, not both."
    num_models = len(random_seed_ints)

    logs_by_seed: Dict[int, Dict[int, Dict]] = {seed: {} for seed in random_seed_ints}
    first_100 = {seed: None for seed in random_seed_ints}
    first_loss = {seed: None for seed in random_seed_ints}
    first_ce   = {seed: None for seed in random_seed_ints}

    metrics_template = init_metrics

    def _run_eval(current_states):
        if eval_fn is not None:
            return eval_fn(current_states)
        if eval_batches is not None:
            xe, ye = eval_batches
            return eval_model(current_states, xe, ye, metrics_template)
        return None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # -------- train one epoch --------
        states, train_metrics = train_epoch(states, x_batches, y_batches, metrics_template)

        train_losses = []
        train_accuracies = []

        do_eval = (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs
        te_all = _run_eval(states) if do_eval else None
        test_losses = [] if do_eval else None
        test_accuracies = [] if do_eval else None
        if te_all is not None:
            print(f"\n--- Test Evaluation at Epoch {epoch + 1} ---")

        for i in range(num_models):
            seed = random_seed_ints[i]
            # Train metrics
            tm_i = jax.tree_util.tree_map(lambda x: x[i], train_metrics)
            train_loss, train_l2, train_acc = _metrics_to_scalars(tm_i)
            logs_by_seed[seed].setdefault(epoch + 1, {})
            logs_by_seed[seed][epoch + 1].update({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_ce_loss": train_loss - weight_decay * train_l2,
            })
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            print(f"Seed {seed}: Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.2%}")

            # Test metrics
            if do_eval:
                te_i = jax.tree_util.tree_map(lambda x: x[i], te_all)
                test_loss, test_l2, test_acc = _metrics_to_scalars(te_i)
                test_ce = test_loss - weight_decay * test_l2
                test_losses.append(test_loss)
                test_accuracies.append(test_acc)
                params_i = jax.tree_util.tree_map(lambda x: x[i], states.params)
                weight_norm = float(
                    sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params_i))
                )
                logs_by_seed[seed][epoch + 1].update({
                    "test_loss": test_loss,
                    "test_ce_loss": test_ce, 
                    "test_accuracy": test_acc,
                    "l2_weight_norm": weight_norm,
                })

                print(f"Seed {seed}: Test CE Loss: {test_ce:.6f}, Test Total Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.2%}")

                if first_100[seed] is None and test_acc >= 1.0:
                    first_100[seed] = epoch + 1
                    first_loss[seed] = test_loss
                    first_ce[seed]   = test_ce
                    logs_by_seed[seed][epoch + 1]["first_reach_100%"] = epoch + 1
                    print(
                        f"*** Seed {seed} first reached 100% accuracy at epoch {epoch + 1} "
                        f"with total loss {test_loss:.6f} and CE-only loss {test_ce:.6f} ***"
                    )
    return states, logs_by_seed, first_100, first_loss, first_ce
