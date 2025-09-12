# analysis/while_training_analysis_MLP.py
from typing import Dict, List, Tuple, Optional, Callable
import jax
import jax.numpy as jnp
from controllers.training_prep_MLP import train_epoch, eval_model,shuffle_batches_for_epoch

EvalFn = Callable[[any], any]  # eval_fn(states) -> metrics collection (per-model)

def _metrics_to_scalars(m):
    c = m.compute()
    return float(c["loss"]), float(c["l2_loss"]), float(c["accuracy"])

# ---------------- Margins ----------------
def build_margin_fn(model):
    @jax.jit
    def _margin(params, xs, ys):
        logits = model.apply({'params': params}, xs, training=False)[0]
        correct = logits[jnp.arange(xs.shape[0]), ys]
        masked = jnp.where(jax.nn.one_hot(ys, logits.shape[1], dtype=bool), -1e9, logits)
        runner = jnp.max(masked, axis=1)
        margins = correct - runner
        return jnp.min(margins), jnp.mean(margins)
    return _margin



# ---------------- Dirichlet energy (MLP) ----------------
def compute_embeddings_MLP(model, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """
    Build the first-layer input embedding for an MLP model.
    Tries to infer whether the first layer expects concat/add embeddings or one-hot.

    x: (B,2) int32 of indices (a,b)
    """
    a, b = x[:, 0], x[:, 1]

    def _find_first_dense_kernel(params_tree):
        for k, v in params_tree.items():
            if isinstance(v, dict):
                if "kernel" in v:
                    return v["kernel"]
                out = _find_first_dense_kernel(v)
                if out is not None:
                    return out
        return None

    if hasattr(model, "extract_embeddings_ab"):
        Wa, Wb = model.extract_embeddings_ab(params)  # (p, D_a), (p, D_b)
        Da, Db = Wa.shape[1], Wb.shape[1]
        k0 = _find_first_dense_kernel(params)
        if k0 is None:
            # Fallback: add embeddings if we cannot infer input size
            return Wa[a] + Wb[b]
        in_features = k0.shape[0]
        p_vocab = Wa.shape[0]

        # Concat learned embeddings
        if in_features == Da + Db:
            return jnp.concatenate([Wa[a], Wb[b]], axis=1)
        # Add learned embeddings
        if in_features == Da:
            return Wa[a] + Wb[b]
        # One-hot concat
        if in_features == 2 * p_vocab:
            return jnp.concatenate(
                [jax.nn.one_hot(a, p_vocab), jax.nn.one_hot(b, p_vocab)], axis=1
            ).astype(jnp.float32)
        # One-hot addition
        if in_features == p_vocab:
            return (jax.nn.one_hot(a, p_vocab) + jax.nn.one_hot(b, p_vocab)).astype(jnp.float32)

        # Last resort
        return Wa[a] + Wb[b]
    else:
        # No extract_embeddings_ab → assume one-hot style first layer
        k0 = _find_first_dense_kernel(params)
        if k0 is None:
            raise ValueError("Cannot infer first-layer input size for embeddings.")
        in_features = k0.shape[0]
        p_guess = int(jnp.max(jnp.concatenate([a, b])) + 1)
        if in_features == 2 * p_guess:
            return jnp.concatenate(
                [jax.nn.one_hot(a, p_guess), jax.nn.one_hot(b, p_guess)], axis=1
            ).astype(jnp.float32)
        if in_features == p_guess:
            return (jax.nn.one_hot(a, p_guess) + jax.nn.one_hot(b, p_guess)).astype(jnp.float32)
        raise ValueError("Unsupported first-layer input format.")

def make_energy_funcs_MLP(model, params: dict):
    """
    Returns:
      emb_fn(x_int)            -> first-layer embedding for a batch: (B, D or 2D)
      batch_energy_sum(emb_B)  -> sum over batch of ||J||_F^2
    where f_embed : embedding -> logits, and J = d logits / d embedding.
    Requires that `model.call_from_embedding(emb, params)` returns (logits, ...).
    """
    def f_embed(x_embed: jnp.ndarray) -> jnp.ndarray:
        logits, _ = model.call_from_embedding(x_embed, params)
        return logits  # (C,)

    grad_f = jax.jit(jax.jacrev(f_embed))

    @jax.jit
    def batch_energy_sum(batch_emb: jnp.ndarray) -> jnp.ndarray:
        J = jax.vmap(grad_f)(batch_emb)  # (B, C, D)
        return jnp.sum(J * J)

    def emb_fn(x_data: jnp.ndarray) -> jnp.ndarray:
        return compute_embeddings_MLP(model, params, x_data)

    return emb_fn, batch_energy_sum

def compute_dirichlet_energy_embedding_MLP(
    model,
    params: dict,
    x_data: jnp.ndarray,           # (N,2)
    *,
    chunk_size: int = 8192,
) -> float:
    """
    Average Dirichlet energy over x_data:
      (1/N) * Σ_x || d logits(x) / d emb(x) ||_F^2
    """
    emb_fn, batch_energy_sum = make_energy_funcs_MLP(model, params)
    emb = emb_fn(x_data)  # (N, D or 2D)
    N = emb.shape[0]
    total = 0.0
    for s in range(0, N, chunk_size):
        total += float(batch_energy_sum(emb[s:s+chunk_size]))
    return total / N

def run_epochs(*,
               model,
               states,
               x_batches, y_batches,
               init_metrics,
               random_seed_ints: List[int],
               weight_decay: float,
               epochs: int,
               eval_every: int = 1,
               eval_batches: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
               eval_flat:    Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
               eval_fn: Optional[EvalFn] = None,
               ):
    assert not (eval_batches is not None and eval_fn is not None), \
        "Pass either eval_batches or eval_fn, not both."
    num_models = len(random_seed_ints)
    margin_fn = build_margin_fn(model) 

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

    # Use 10x training batch size as a default chunk size for energy computation
    energy_chunk = int(x_batches.shape[2]) * 10
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
            xb_i = xb[i].reshape(-1, 2)
            yb_i = yb[i].reshape(-1)
            tr_min_m, tr_avg_m = margin_fn(params_i, xb_i, yb_i)
            logs_by_seed[seed][epoch + 1].update({
                "train_margin_min": float(tr_min_m),
                "train_margin_avg": float(tr_avg_m),
            })
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
                # margins
                if eval_flat is not None:
                    xe_flat, ye_flat = eval_flat              # shapes: (N,2), (N,)
                    min_m, avg_m = margin_fn(params_i, xe_flat, ye_flat)
                    xs_eval_for_energy = xe_flat
                elif eval_batches is not None:
                    xe_b, ye_b = eval_batches                 # (M, K, B, 2), (M, K, B)
                    xe_i = xe_b[i].reshape(-1, 2)             
                    ye_i = ye_b[i].reshape(-1)
                    # note: eval_batches is from pad_to_batches containing repeated paddings,
                    # not affecting min_margin, but is gonna affect avg_margin slightly.
                    min_m, avg_m = margin_fn(params_i, xe_i, ye_i)
                    xs_eval_for_energy = xe_i
                else:
                    min_m, avg_m = jnp.nan, jnp.nan  # if eval grid is none mark as nan
                    xs_eval_for_energy = None

                # Dirichlet energies
                if xs_eval_for_energy is not None:
                    de_test = compute_dirichlet_energy_embedding_MLP(
                        model, params_i, xs_eval_for_energy, chunk_size=energy_chunk
                    )
                else:
                    de_test = float("nan")
                de_train = compute_dirichlet_energy_embedding_MLP(
                    model, params_i, xb_i, chunk_size=energy_chunk
                )
                if xs_eval_for_energy is not None:
                    x_total = jnp.concatenate([xs_eval_for_energy, xb_i], axis=0)
                    de_total = compute_dirichlet_energy_embedding_MLP(
                        model, params_i, x_total, chunk_size=energy_chunk
                    )
                else:
                    de_total = de_train

                logs_by_seed[seed][epoch + 1].update({
                    "test_loss": test_loss,
                    "test_ce_loss": test_ce,
                    "test_accuracy": test_acc,
                    "l2_weight_norm": weight_norm,
                    "test_margin_min": float(min_m),
                    "test_margin_avg": float(avg_m),
                    "dirichlet_energy_test":  float(de_test),
                    "dirichlet_energy_train": float(de_train),
                    "dirichlet_energy_total": float(de_total),
                })
                print(
                    f"Seed {seed}: Test CE {test_ce:.6f}, Total {test_loss:.6f}, "
                    f"Acc {test_acc:.2%}, "
                    f"Margin[min/avg] {float(min_m):.4f}/{float(avg_m):.4f}, "
                    f"DE[test/train/total] {de_test:.3e}/{de_train:.3e}/{de_total:.3e}"
                )
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
                       model,
                       states,
                       x_batches, y_batches,
                       init_metrics,
                       random_seed_ints: List[int],
                       weight_decay: float,
                       epochs: int,
                       eval_every: int = 1,
                       eval_batches: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
                       eval_flat: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
                       eval_fn: Optional[EvalFn] = None,
                       ):
    """
    Same as run_epochs, but structured for your scaling experiments.
    Also logs margins and Dirichlet energy at evaluation points.
    """
    assert not (eval_batches is not None and eval_fn is not None), \
        "Pass either eval_batches or eval_fn, not both."
    num_models = len(random_seed_ints)
    margin_fn = build_margin_fn(model)

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

    energy_chunk = int(x_batches.shape[2]) * 10

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        xb, yb = shuffle_batches_for_epoch(
            x_batches, y_batches, epoch, random_seed_ints, True, debug=False
        )
        states, train_metrics = train_epoch(states, xb, yb, metrics_template)

        do_eval = (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs
        te_all = _run_eval(states) if do_eval else None
        if te_all is not None:
            print(f"\n--- Test Evaluation at Epoch {epoch + 1} ---")

        for i in range(num_models):
            seed = random_seed_ints[i]
            tm_i = jax.tree_util.tree_map(lambda x: x[i], train_metrics)
            train_loss, train_l2, train_acc = _metrics_to_scalars(tm_i)
            print(f"Seed {seed}: Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.2%}")

            if do_eval:
                te_i = jax.tree_util.tree_map(lambda x: x[i], te_all)
                test_loss, test_l2, test_acc = _metrics_to_scalars(te_i)
                test_ce = test_loss - weight_decay * test_l2
                params_i = jax.tree_util.tree_map(lambda x: x[i], states.params)
                weight_norm = float(
                    sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params_i))
                )

                # Train margins on current epoch's data
                xb_i = xb[i].reshape(-1, 2)
                yb_i = yb[i].reshape(-1)
                tr_min_m, tr_avg_m = margin_fn(params_i, xb_i, yb_i)

                # Eval margins
                if eval_flat is not None:
                    xe_flat, ye_flat = eval_flat
                    min_m, avg_m = margin_fn(params_i, xe_flat, ye_flat)
                    xs_eval_for_energy = xe_flat
                elif eval_batches is not None:
                    xe_b, ye_b = eval_batches
                    xe_i = xe_b[i].reshape(-1, 2)
                    ye_i = ye_b[i].reshape(-1)
                    min_m, avg_m = margin_fn(params_i, xe_i, ye_i)
                    xs_eval_for_energy = xe_i
                else:
                    min_m, avg_m = jnp.nan, jnp.nan
                    xs_eval_for_energy = None

                # Dirichlet energies
                if xs_eval_for_energy is not None:
                    de_test = compute_dirichlet_energy_embedding_MLP(
                        model, params_i, xs_eval_for_energy, chunk_size=energy_chunk
                    )
                else:
                    de_test = float("nan")
                de_train = compute_dirichlet_energy_embedding_MLP(
                    model, params_i, xb_i, chunk_size=energy_chunk
                )
                if xs_eval_for_energy is not None:
                    x_total = jnp.concatenate([xs_eval_for_energy, xb_i], axis=0)
                    de_total = compute_dirichlet_energy_embedding_MLP(
                        model, params_i, x_total, chunk_size=energy_chunk
                    )
                else:
                    de_total = de_train

                logs_by_seed[seed].setdefault(epoch + 1, {})
                logs_by_seed[seed][epoch + 1].update({
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_ce_loss": train_loss - weight_decay * train_l2,
                    "test_loss": test_loss,
                    "test_ce_loss": test_ce,
                    "test_accuracy": test_acc,
                    "l2_weight_norm": weight_norm,
                    "train_margin_min": float(tr_min_m),
                    "train_margin_avg": float(tr_avg_m),
                    "test_margin_min": float(min_m),
                    "test_margin_avg": float(avg_m),
                    "dirichlet_energy_test":  float(de_test),
                    "dirichlet_energy_train": float(de_train),
                    "dirichlet_energy_total": float(de_total),
                })

                print(
                    f"Seed {seed}: Test CE {test_ce:.6f}, Total {test_loss:.6f}, "
                    f"Acc {test_acc:.2%}, "
                    f"Margin[min/avg] {float(min_m):.4f}/{float(avg_m):.4f}, "
                    f"DE[test/train/total] {de_test:.3e}/{de_train:.3e}/{de_total:.3e}"
                )

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
