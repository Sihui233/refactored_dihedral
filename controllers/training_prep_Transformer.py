# controllers/training_prep_Transformer.py
from typing import Dict, Any, Tuple, List
import jax
import jax.numpy as jnp
import numpy as np
import optax
from clu import metrics
from flax import struct
import jax.tree_util

import dihedral
import DFT
from utils import compute_pytree_size
import training

from transformer_class import TransformerOneEmbed, TransformerTwoEmbed

Params = Dict[str, Any]

# ---------------- Metrics ----------------
@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')
    l2_loss: metrics.Average.from_output('l2_loss')

# ---------------- small helpers ----------------
def logits_last_token(logits_3d: jnp.ndarray) -> jnp.ndarray:
    return logits_3d[:, -1, :]  # (B, vocab)

def cross_entropy_loss(y_pred_3d, y):
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits_last_token(y_pred_3d), labels=y
    ).mean()

def total_loss(y_pred_and_l2, y, weight_decay: float):
    y_pred, _, l2_loss = y_pred_and_l2
    return cross_entropy_loss(y_pred, y) + l2_loss * weight_decay

def apply_fn_builder(model):
    def apply(variables, x, training=False):
        params = variables["params"]
        outputs = model.apply({"params": params}, x, training=training)  # (B, seq, vocab)
        l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        return outputs, {}, l2_loss
    return apply

def compute_metrics(metrics_obj, *, loss, l2_loss, outputs, labels):
    logits_2d = logits_last_token(outputs)
    metric_updates = metrics_obj.single_from_model_output(
        logits=logits_2d, labels=labels, loss=loss, l2_loss=l2_loss
    )
    return metrics_obj.merge(metric_updates)

# ---------------- Model / Optimizer ----------------
def build_model(cfg, group_size: int):
    assert cfg.d_head * cfg.num_heads == cfg.d_model, "d_head * num_heads must equal d_model"
    # default TwoEmbed
    return TransformerTwoEmbed(
        num_layers=1,                      # block is 1
        num_mlp_layers=cfg.num_mlp_layers,
        d_vocab=group_size,
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        num_heads=cfg.num_heads,
        n_ctx=cfg.n_ctx,
        act_type=cfg.act_type,
        attn_coeff=cfg.attn_coeff,
        nn_multiplier=cfg.nn_multiplier,
    )

def build_optimizer(optimizer_name: str, lr: float, momentum: float = 0.0):
    if optimizer_name == "adam":
        return optax.adam(lr)
    if optimizer_name.startswith("SGD"):
        return optax.sgd(lr, momentum=momentum)
    raise ValueError(f"Unsupported optimizer type: {optimizer_name}")

# ---------------- Dataset / eval grid ----------------
def make_train_batches(p: int, batch_size: int, k: int, random_seed_ints: List[int]):
    train_ds_list = []
    for seed in random_seed_ints:
        x, y = dihedral.make_dihedral_dataset(p, batch_size, k, seed)
        train_ds_list.append((x, y))
    x_batches = jnp.stack([x for (x, _) in train_ds_list])
    y_batches = jnp.stack([y for (_, y) in train_ds_list])
    print("x_batches.shape =", x_batches.shape, "y_batches.shape =", y_batches.shape)
    print(f"Number of training batches: {x_batches.shape[1]}")
    dataset_size_mb = (x_batches.size * x_batches.dtype.itemsize) / (1024 ** 2)
    print(f"Dataset size per model: {dataset_size_mb:.2f} MB")
    return train_ds_list, x_batches, y_batches

def make_full_eval_grid(p: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    G, _ = DFT.make_irreps_Dn(p)
    idx = {g: i for i, g in enumerate(G)}
    group_size = len(G)
    x_eval = jnp.array([[idx[g], idx[h]] for g in G for h in G], dtype=jnp.int32)
    y_eval = jnp.array([idx[dihedral.mult(g, h, p)] for g in G for h in G], dtype=jnp.int32)
    return x_eval, y_eval

def pad_to_batches(x_eval, y_eval, batch_size: int, num_models: int):
    x_exp = jnp.tile(x_eval[None, :, :], (num_models, 1, 1))
    y_exp = jnp.tile(y_eval[None, :], (num_models, 1))
    total, bs = x_eval.shape[0], batch_size
    n_full, remain = total // bs, total % bs
    if remain > 0:
        pad = bs - remain
        x_pad = x_exp[:, :pad, :]
        y_pad = y_exp[:, :pad]
        x_eval_padded = jnp.concatenate([x_exp, x_pad], axis=1)
        y_eval_padded = jnp.concatenate([y_exp, y_pad], axis=1)
        n_batches = n_full + 1
    else:
        x_eval_padded, y_eval_padded, n_batches = x_exp, y_exp, n_full
    x_batches = x_eval_padded.reshape(num_models, n_batches, bs, 2)
    y_batches = y_eval_padded.reshape(num_models, n_batches, bs)
    return x_batches, y_batches

# ---------------- Create TrainState ----------------

def create_states(model, tx, weight_decay: float, batch_size: int, random_seed_ints: List[int]):
    dummy_x = jnp.zeros((batch_size, 2), dtype=jnp.int32)
    variables_list = [model.init(jax.random.PRNGKey(seed), dummy_x, training=False)
                      for seed in random_seed_ints]
    params_batch = jax.tree_util.tree_map(lambda *args: jnp.stack(args),
                                          *(v['params'] for v in variables_list))

    # independent init, every model's opt_state
    def init_opt(p): return tx.init(p)
    opt_states = []
    for i in range(len(random_seed_ints)):
        params_i = jax.tree_util.tree_map(lambda x: x[i], params_batch)
        opt_states.append(init_opt(params_i))
    opt_state_batch = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *opt_states)

    apply = apply_fn_builder(model)
    def loss_fn(y_pred_and_l2, y): return total_loss(y_pred_and_l2, y, weight_decay)

    states_list = []
    for i, seed in enumerate(random_seed_ints):
        rng_key = jax.random.PRNGKey(seed)
        params_i = jax.tree_util.tree_map(lambda x: x[i], params_batch)
        opt_state_i = jax.tree_util.tree_map(lambda x: x[i], opt_state_batch)
        state = training.TrainState(
            apply_fn=apply, params=params_i, tx=tx, opt_state=opt_state_i,
            loss_fn=loss_fn, loss_hessian_fn=lambda *_: (0.0, 0.0, 0.0),
            compute_metrics_fn=compute_metrics,
            rng_key=rng_key, initial_metrics=Metrics, batch_stats=None, injected_noise=0.0
        )
        states_list.append(state)

    states = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *states_list)
    init_metrics = jax.tree_util.tree_map(lambda *args: jnp.stack(args),
                                          *[s.initial_metrics.empty() for s in states_list])
    return states, init_metrics

# ---------------- JIT train/eval epoch ----------------
# same as MLP included for future references

@jax.jit
def train_epoch(states, x_batches, y_batches, initial_metrics):
    def train_step(state_metrics, batch):
        states, metrics_ = state_metrics
        x, y = batch
        new_states, new_metrics = jax.vmap(
            lambda st, m, xb, yb: st.train_step(m, (xb, yb)),
            in_axes=(0, 0, 0, 0)
        )(states, metrics_, x, y)
        return (new_states, new_metrics), None

    initial_state_metrics = (states, initial_metrics)
    transposed_x = x_batches.transpose(1, 0, 2, 3)
    transposed_y = y_batches.transpose(1, 0, 2)
    (new_states, new_metrics), _ = jax.lax.scan(
        train_step, 
        initial_state_metrics, 
        (transposed_x, transposed_y)
    )
    return new_states, new_metrics

@jax.jit
def eval_model(states, x_batches, y_batches, initial_metrics):
    def eval_step(metrics_, batch):
        x, y = batch
        new_metrics = jax.vmap(
            lambda st, m, xb, yb: st.eval_step(m, (xb, yb)),
            in_axes=(0, 0, 0, 0)
        )(states, metrics_, x, y)
        return new_metrics, None

    transposed_x = x_batches.transpose(1, 0, 2, 3)
    transposed_y = y_batches.transpose(1, 0, 2)
    final_metrics, _ = jax.lax.scan(
        eval_step, 
        initial_metrics, 
        (transposed_x, transposed_y)
    )
    return final_metrics
