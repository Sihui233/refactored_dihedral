# analysis/while_training_analysis_MLP.py
from typing import Dict, List, Tuple, Optional, Callable
import jax
import jax.numpy as jnp
from controllers.training_prep_MLP import train_epoch, eval_model,shuffle_batches_for_epoch

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
               eval_every: int = 1,
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
        xb, yb = shuffle_batches_for_epoch(x_batches, y_batches, epoch, random_seed_ints, True,debug=True)

        # states, train_metrics = train_epoch(states, x_batches, y_batches, metrics_template)
        states, train_metrics = train_epoch(states, xb, yb, metrics_template)

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

def run_epochs_scaling(*,
               states,
               x_batches, y_batches,
               init_metrics,
               random_seed_ints: List[int],
               weight_decay: float,
               epochs: int,
               eval_every: int = 1,
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
        xb, yb = shuffle_batches_for_epoch(x_batches, y_batches, epoch, random_seed_ints, True,debug=True)

        # states, train_metrics = train_epoch(states, x_batches, y_batches, metrics_template)
        states, train_metrics = train_epoch(states, xb, yb, metrics_template)

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
                    logs_by_seed[seed][epoch + 1].update({
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "train_ce_loss": train_loss - weight_decay * train_l2,
                        # "test_loss": test_loss,
                        # "test_ce_loss": test_ce, 
                        # "test_accuracy": test_acc,
                        # "l2_weight_norm": weight_norm,
                    })
                    print(
                        f"*** Seed {seed} first reached 100% accuracy at epoch {epoch + 1} "
                        f"with total loss {test_loss:.6f} and CE-only loss {test_ce:.6f} ***"
                    )
    return states, logs_by_seed, first_100, first_loss, first_ce


###### MLP #########


###### Trans ########

from functools import partial
# 1)  pure-JAX embedding extractor
def compute_embeddings_transformer(
    model,
    params: dict,
    x: jnp.ndarray,                       # (B , 2)  int32 tokens  (a , b)
    *,
    concat: bool = False,                 # False →  “Eₐ + E_b”    True →  “[Eₐ‖E_b]”
) -> jnp.ndarray:
    """
    Returns the *input* embedding vector that will be fed into the first
    Transformer block, **after** adding learnt position embeddings.

    • `concat == False`   →  shape  (B , D)
    • `concat == True`    →  shape  (B , 2 D)
    """
    # ---- 1. grab weights ----------------------------------------------------
    # shared token table (W_E)  &  first two learned position vectors
    Wa, Wb = model.extract_embeddings_ab(params)            # (p , D)
    pos0, pos1 = params["pos_embed"]["W_pos"][:2]                   # (D,)
    a_idx = x[:, 0]
    b_idx = x[:, 1]

    # ---- 2. look-up & add positions ----------------------------------------
    emb_a = Wa[a_idx] + pos0                              # (B, D)
    emb_b = Wb[b_idx] + pos1                                   # (B , D)

    if concat:
        return jnp.concatenate([emb_a, emb_b], axis=-1)   # (B, 2D)
    else:
        return emb_a + emb_b                                     # (B , D)

# 2)  make_energy_funcs_transformer
def make_energy_funcs_transformer(
    model,           # the *initialised* model instance
    params: dict,                   # its parameters
    *,
    concat: bool = False,
):
    """
    Returns two callables **emb_fn** and **batch_energy_sum** that exactly
    mirror the MLP helpers:

    • emb_fn(x_int)              →  embedding batch  (see above)
    • batch_energy_sum(e_batch)  →  Σ ‖J‖²_F  over that *batch*

    where J is the Jacobian  ∂ logits / ∂ embedding.
    """

    # ---------- f_embed : (D,) or (2D,)  →  (p,) logits ----------------------
    Wa, _ = model.extract_embeddings_ab(params)
    D = Wa.shape[1]

    def _to_seq(x_flat: jnp.ndarray) -> jnp.ndarray:
        if concat:
            ea, eb = jnp.split(x_flat, 2)
        else:
            ea = x_flat * 0.5
            eb = x_flat * 0.5
        return jnp.stack([ea, eb])[None, ...]             # (1, 2, D)

    def f_embed(x_flat: jnp.ndarray) -> jnp.ndarray:      # → (p,)
        seq_emb = _to_seq(x_flat)                         # (1, 2, D)
        logits  = model.call_from_embedding_sequence(seq_emb, params)[0]
        return logits[-1]                                  # last-token logits

    f_embed = jax.jit(f_embed)

    def _squared_frobenius_norm_of_jac(x_flat: jnp.ndarray) -> jnp.ndarray:
        J = jax.jacrev(f_embed)(x_flat)                   # (p, D | 2D)
        return jnp.sum(J * J)

    _squared_frobenius_norm_of_jac = jax.jit(_squared_frobenius_norm_of_jac)

    emb_fn = partial(compute_embeddings_transformer, model, params, concat=concat)

    def batch_energy_sum(batch_emb: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(_squared_frobenius_norm_of_jac)(batch_emb).sum()

    return emb_fn, batch_energy_sum

# 3)  driver that averages over an arbitrary input set
def compute_dirichlet_energy_embedding_transformer(
    model,
    params: dict,
    x_data: jnp.ndarray,                 # (N , 2)  all (a , b) pairs of interest
    *,
    batch_size: int = 1024,
    concat: bool = False,
) -> float:
    """
    Plain (non-JIT) wrapper that chunks `x_data` to keep memory modest.
    """
    emb_fn, batch_energy_sum = make_energy_funcs_transformer(
        model, params, concat=concat
    )

    total = 0.0
    n     = x_data.shape[0]

    for i in range(0, n, batch_size):
        x_batch   = x_data[i : i + batch_size]
        e_batch   = emb_fn(x_batch)                       # (B , D | 2D)
        total    += batch_energy_sum(e_batch)

    return float(total / n)
