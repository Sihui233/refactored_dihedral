# analysis/while_training_analysis_gen.py
from typing import Dict, List, Tuple, Optional, Callable, Any
import jax
import jax.numpy as jnp
from controllers.training_prep_MLP import train_epoch, eval_model,shuffle_batches_for_epoch
from functools import partial

EvalFn = Callable[[Any], Any]  # eval_fn(states) -> metrics collection (per-model)

def _metrics_to_scalars(m):
    c = m.compute()
    return float(c["loss"]), float(c["l2_loss"]), float(c["accuracy"])

# ---------------- Margins ----------------
def build_margin_fn(model, *, chunk_size: int = 8192):
    """
    Returns a function margin_stats(params, xs, ys) that computes
    (min_margin, mean_margin) in small chunks to avoid OOM.

    xs: (N, 2) int32, ys: (N,) int32
    chunk_size: how many samples per chunk to run through the model.
    """
    @jax.jit
    def _batch_margins(params, xs_chunk, ys_chunk):
        out = model.apply({'params': params}, xs_chunk, training=False)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        # comaptible w transformer
        if logits.ndim == 3:  
            logits = logits[:, -1, :]   # (B,C)
        correct = logits[jnp.arange(xs_chunk.shape[0]), ys_chunk]
        masked  = jnp.where(jax.nn.one_hot(ys_chunk, logits.shape[1], dtype=bool), -1e30, logits)
        runner  = jnp.max(masked, axis=1)
        return correct - runner  # (b,)

    def margin_stats(params, xs, ys):
        N = xs.shape[0]
        # Use Python loop to avoid building a gigantic HLO / allocation.
        gmin = jnp.inf
        total = jnp.array(0.0)
        for start in range(0, int(N), int(chunk_size)):
            stop = start + int(chunk_size)
            m = _batch_margins(params, xs[start:stop], ys[start:stop])  # (b,)
            gmin = jnp.minimum(gmin, jnp.min(m))
            total = total + jnp.sum(m)
        return float(gmin), float(total / N)

    return margin_stats

# # ---------------- Dirichlet energy (MLP) ----------------
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
    # emb = emb_fn(x_data)  # (N, D or 2D)
    N = x_data.shape[0]
    total = jnp.array(0.0)
    for s in range(0, N, chunk_size):
        e = emb_fn(x_data[s:s+chunk_size]) 
        total = total + batch_energy_sum(e)
    return float(total / N)


###### Transformer Dirichlet (upgraded) ########

# 1) Pure-JAX embedding extractor — compatible with TransformerOneEmbed & TwoEmbed
def compute_embeddings_transformer(
    model,
    params: dict,
    x: jnp.ndarray,                      # (B, 2)  int32 tokens (a,b)
    *,
    concat: bool = False,                # False → “E_a + E_b”; True → “[E_a‖E_b]”
) -> jnp.ndarray:
    """
    Build the *first-block input embedding* (post position-embed) for the transformer.

    Returns the vector that feeds the first Transformer block *after* adding
    learned position embeddings:

      • concat == False  → shape (B, D)     representing (E_a + E_b) + (pos0 + pos1)
      • concat == True   → shape (B, 2D)    representing [E_a+pos0  ‖  E_b+pos1]

    Compatible with:
      - TransformerOneEmbed.extract_embeddings_ab -> (W_E, W_E)
      - TransformerTwoEmbed.extract_embeddings_ab -> (W_a, W_b)
    """
    a, b = x[:, 0], x[:, 1]
    Wa, Wb = model.extract_embeddings_ab(params)           # (p, D), (p, D)

    # learned pos of the first two positions
    pos0, pos1 = params["pos_embed"]["W_pos"][:2]         # (D,), (D,)

    emb_a = Wa[a] + pos0                                  # (B, D)
    emb_b = Wb[b] + pos1                                  # (B, D)

    if concat:
        return jnp.concatenate([emb_a, emb_b], axis=-1)   # (B, 2D)
    else:
        return emb_a + emb_b                              # (B, D)


# 2) Energy functions — match MLP interface/returns
def make_energy_funcs_transformer(
    model,                 # initialized TransformerOneEmbed or TransformerTwoEmbed
    params: dict,          # its parameters
    *,
    concat: bool = False,
):
    """
    Returns:
      emb_fn(x_int)            -> (B, D|2D)  batch input embeddings (already has pos)
      batch_energy_sum(emb_B)  -> scalar: Σ_b ||J||_F^2

    where f_embed : flat-embedding -> (last-token) logits, and
    J = ∂ logits / ∂ flat-embedding.
    """
    Wa, _ = model.extract_embeddings_ab(params)
    D = Wa.shape[1]  # d_model

    # Convert flat (D|2D,) into a (1, 2, D) sequence embedding for call_from_embedding_sequence
    def _to_seq(x_flat: jnp.ndarray) -> jnp.ndarray:
        if concat:
            ea, eb = jnp.split(x_flat, 2)                 # (D,), (D,)
        else:
            # Unbiased "split" of (E_a + E_b): treat both as 1/2 so gradients distribute
            ea = x_flat * 0.5
            eb = x_flat * 0.5
        return jnp.stack([ea, eb])[None, ...]             # (1, 2, D)

    # Single sample: flat embedding → last-token logits (p,)
    def f_embed(x_flat: jnp.ndarray) -> jnp.ndarray:
        seq_emb = _to_seq(x_flat)                                      # (1,2,D)
        logits  = model.call_from_embedding_sequence(seq_emb, params)  # (1,2,p)
        return logits[0, -1]                                           # (p,)

    # Jitted Jacobian of logits wrt flat embedding, computed once then vmapped
    grad_f = jax.jit(jax.jacrev(f_embed))  # (p,) wrt (D|2D,) → (p, D|2D)

    @jax.jit
    def batch_energy_sum(batch_emb: jnp.ndarray) -> jnp.ndarray:
        """
        batch_emb: (B, D|2D)
        returns:   scalar Σ_b ||J_b||_F^2
        """
        J = jax.vmap(grad_f)(batch_emb)  # (B, p, D|2D)
        return jnp.sum(J * J)

    # External: tokens (a,b) → flat embedding (already has pos added)
    emb_fn = partial(compute_embeddings_transformer, model, params, concat=concat)

    return emb_fn, batch_energy_sum


# 3) Driver: chunked average over an input set
def compute_dirichlet_energy_embedding_transformer(
    model,
    params: dict,
    x_data: jnp.ndarray,                 # (N, 2) token pairs (a,b)
    *,
    chunk_size: int = 8192,              # match MLP_dirichlet default
    concat: bool = False,
) -> float:
    """
    Average Dirichlet energy over x_data:
      (1/N) * Σ_x || ∂ logits(x) / ∂ emb(x) ||_F^2
    """
    emb_fn, batch_energy_sum = make_energy_funcs_transformer(model, params, concat=concat)

    N = int(x_data.shape[0])
    total = jnp.array(0.0)
    for s in range(0, N, chunk_size):
        e = emb_fn(x_data[s:s+chunk_size])     # (B, D|2D), already includes pos
        total = total + batch_energy_sum(e)    # scalar on device
    return float(total / N)

### dispatcher that auto-detects MLP or Transformer
def _is_transformer(model) -> bool:
    return hasattr(model, "call_from_embedding_sequence") and callable(getattr(model, "call_from_embedding_sequence"))

def compute_dirichlet_energy_embedding_auto(
    model,
    params: dict,
    x_data: jnp.ndarray,           # (N,2)
    *,
    chunk_size: int = 8192,
    # if you ever need to change Transformer flattening behavior:
    transformer_concat: bool = False,
) -> float:
    if _is_transformer(model):
        # Transformer Dirichlet
        return compute_dirichlet_energy_embedding_transformer(
            model, params, x_data, chunk_size=chunk_size, concat=transformer_concat
        )
    # MLP Dirichlet
    return compute_dirichlet_energy_embedding_MLP(
        model, params, x_data, chunk_size=chunk_size
    )

# def run_epochs(*,
#                model,
#                states,
#                x_batches, y_batches,
#                init_metrics,
#                random_seed_ints: List[int],
#                weight_decay: float,
#                epochs: int,
#                eval_every: int = 1,
#                eval_batches: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
#                eval_flat:    Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
#                eval_fn: Optional[EvalFn] = None,
#                ):
#     assert not (eval_batches is not None and eval_fn is not None), \
#         "Pass either eval_batches or eval_fn, not both."
#     num_models = len(random_seed_ints)
#     margin_fn = build_margin_fn(model) 

#     logs_by_seed: Dict[int, Dict[int, Dict]] = {seed: {} for seed in random_seed_ints}
#     first_100 = {seed: None for seed in random_seed_ints}
#     first_loss = {seed: None for seed in random_seed_ints}
#     first_ce   = {seed: None for seed in random_seed_ints}

#     metrics_template = init_metrics

#     def _run_eval(current_states):
#         if eval_fn is not None:
#             return eval_fn(current_states)
#         if eval_batches is not None:
#             xe, ye = eval_batches
#             return eval_model(current_states, xe, ye, metrics_template)
#         return None

#     # Use 10x training batch size as a default chunk size for energy computation
#     energy_chunk = int(min(2048, x_batches.shape[2] * 2))
#     for epoch in range(epochs):
#         print(f"Epoch {epoch + 1}/{epochs}")
#         # -------- train one epoch --------
#         xb, yb = shuffle_batches_for_epoch(x_batches, y_batches, epoch, random_seed_ints, True,debug=False)

#         # states, train_metrics = train_epoch(states, x_batches, y_batches, metrics_template)
#         states, train_metrics = train_epoch(states, xb, yb, metrics_template)

#         train_losses = []
#         train_accuracies = []

#         do_eval = (epoch + 1) % eval_every == 0 
#         te_all = _run_eval(states) if do_eval else None
#         test_losses = [] if do_eval else None
#         test_accuracies = [] if do_eval else None
#         if te_all is not None:
#             print(f"\n--- Test Evaluation at Epoch {epoch + 1} ---")

#         for i in range(num_models):
#             seed = random_seed_ints[i]
#             # Train metrics
#             tm_i = jax.tree_util.tree_map(lambda x: x[i], train_metrics)
#             train_loss, train_l2, train_acc = _metrics_to_scalars(tm_i)
#             logs_by_seed[seed].setdefault(epoch + 1, {})
#             logs_by_seed[seed][epoch + 1].update({
#                 "train_loss": train_loss,
#                 "train_accuracy": train_acc,
#                 "train_ce_loss": train_loss - weight_decay * train_l2,
#             })
#             train_losses.append(train_loss)
#             train_accuracies.append(train_acc)

#             print(f"Seed {seed}: Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.2%}")

#             # Test metrics
#             if do_eval or (epoch + 1) == epochs:
#                 te_i = jax.tree_util.tree_map(lambda x: x[i], te_all)
#                 test_loss, test_l2, test_acc = _metrics_to_scalars(te_i)
#                 test_ce = test_loss - weight_decay * test_l2
#                 test_losses.append(test_loss)
#                 test_accuracies.append(test_acc)
#                 params_i = jax.tree_util.tree_map(lambda x: x[i], states.params)
#                 weight_norm = float(
#                     sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params_i))
#                 )
#                 logs_by_seed[seed][epoch + 1].update({
#                     "test_loss": test_loss,
#                     "test_ce_loss": test_ce, 
#                     "test_accuracy": test_acc,
#                     "l2_weight_norm": weight_norm,
#                 })

#                 xb_i = xb[i].reshape(-1, 2)
#                 yb_i = yb[i].reshape(-1)
#                 tr_min_m, tr_avg_m = margin_fn(params_i, xb_i, yb_i)
#                 logs_by_seed[seed][epoch + 1].update({
#                     "train_margin_min": float(tr_min_m),
#                     "train_margin_avg": float(tr_avg_m),
#                 })

#                 print(f"Seed {seed}: Test CE Loss: {test_ce:.6f}, Test Total Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.2%}")
#                 # eval margins
#                 if eval_flat is not None:
#                     xe_flat, ye_flat = eval_flat              # shapes: (N,2), (N,)
#                     min_m, avg_m = margin_fn(params_i, xe_flat, ye_flat)
#                     xs_eval_for_energy = xe_flat
#                 elif eval_batches is not None:
#                     xe_b, ye_b = eval_batches                 # (M, K, B, 2), (M, K, B)
#                     xe_i = xe_b[i].reshape(-1, 2)             
#                     ye_i = ye_b[i].reshape(-1)
#                     # note: eval_batches is from pad_to_batches containing repeated paddings,
#                     # not affecting min_margin, but is gonna affect avg_margin slightly.
#                     min_m, avg_m = margin_fn(params_i, xe_i, ye_i)
#                     xs_eval_for_energy = xe_i
#                 else:
#                     min_m, avg_m = jnp.nan, jnp.nan  # if eval grid is none mark as nan
#                     xs_eval_for_energy = None

#                 # Dirichlet energies
#                 if xs_eval_for_energy is not None:
#                     de_test = compute_dirichlet_energy_embedding_MLP(
#                         model, params_i, xs_eval_for_energy, chunk_size=energy_chunk
#                     )
#                 else:
#                     de_test = float("nan")
#                 de_train = compute_dirichlet_energy_embedding_MLP(
#                     model, params_i, xb_i, chunk_size=energy_chunk
#                 )
                

#                 logs_by_seed[seed][epoch + 1].update({
#                     "test_loss": test_loss,
#                     "test_ce_loss": test_ce,
#                     "test_accuracy": test_acc,
#                     "l2_weight_norm": weight_norm,
#                     "test_margin_min": float(min_m),
#                     "test_margin_avg": float(avg_m),
#                     "dirichlet_energy_test":  float(de_test),
#                     "dirichlet_energy_train": float(de_train),
#                 })
#                 print(
#                     f"Seed {seed}: Test CE {test_ce:.6f}, Total {test_loss:.6f}, "
#                     f"Acc {test_acc:.2%}, "
#                     f"Margin[min/avg] {float(min_m):.4f}/{float(avg_m):.4f}, "
#                     f"DE[test/train/total] {de_test:.3e}/{de_train:.3e}"
#                 )
#                 if first_100[seed] is None and test_acc >= 1.0:
#                     first_100[seed] = epoch + 1
#                     first_loss[seed] = test_loss
#                     first_ce[seed]   = test_ce
#                     logs_by_seed[seed][epoch + 1]["first_reach_100%"] = epoch + 1
#                     print(
#                         f"*** Seed {seed} first reached 100% accuracy at epoch {epoch + 1} "
#                         f"with total loss {test_loss:.6f} and CE-only loss {test_ce:.6f} ***"
#                     )
                    
                
                
#     return states, logs_by_seed, first_100, first_loss, first_ce

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
    Same as run_epochs, but structured for scaling experiments.
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

    seeds_arr = jnp.asarray(random_seed_ints, dtype=jnp.uint32)
    energy_chunk = int(min(128, x_batches.shape[2] * 2))

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        xb, yb = shuffle_batches_for_epoch(x_batches, y_batches, epoch, seeds_arr,
                                        shuffle_within_batch=True, debug=False)

        states, train_metrics = train_epoch(states, xb, yb, metrics_template)

        do_eval = (epoch + 1) % eval_every == 0
        te_all = _run_eval(states) if do_eval else None

        jax.block_until_ready(states)

        if te_all is not None:
            print(f"\n--- Test Evaluation at Epoch {epoch + 1} ---")

        for i in range(num_models):
            seed = random_seed_ints[i]

            if (first_100[seed] is not None) and ((epoch + 1) != epochs):
                continue

            tm_i = jax.tree_util.tree_map(lambda x: x[i], train_metrics)
            train_loss, train_l2, train_acc = _metrics_to_scalars(tm_i)
            print(f"Seed {seed}: Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.2%}")

            if do_eval or (epoch + 1) == epochs:
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

                # # Dirichlet energies
                # if xs_eval_for_energy is not None:
                #     de_test = compute_dirichlet_energy_embedding_auto(
                #         model, params_i, xs_eval_for_energy, chunk_size=energy_chunk
                #     )
                # else:
                #     de_test = float("nan")

                # de_train = compute_dirichlet_energy_embedding_auto(
                #     model, params_i, xb_i, chunk_size=energy_chunk
                # )
                

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
                    # "dirichlet_energy_test":  float(de_test),
                    # "dirichlet_energy_train": float(de_train),
                })

                print(
                    f"Seed {seed}: Test CE {test_ce:.6f}, Total {test_loss:.6f}, "
                    f"Acc {test_acc:.2%}, "
                    f"Margin[min/avg] {float(min_m):.4f}/{float(avg_m):.4f}, "
                    # f"DE[test/train/total] {de_test:.3e}/{de_train:.3e}"
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
