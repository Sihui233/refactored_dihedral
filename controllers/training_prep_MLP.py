# training_prep_MLP.py
import os
import json
import copy
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
from mlp_models_multilayer import (
    DonutMLP, MLPOneEmbed, MLPOneHot, MLPTwoEmbed,
    MLPTwoEmbed_cheating, MLPOneEmbed_cheating, MLPOneHot_cheating, MLPOneEmbedResidual
)
import training

# ---------------- Types ----------------
Params = Dict[str, Any]

# ---------------- Metrics ----------------
@struct.dataclass
class Metrics(metrics.Collection):
    """Train/eval metrics collection."""
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')
    l2_loss: metrics.Average.from_output('l2_loss')

# ---------------- Loss / apply ----------------
def cross_entropy_loss(y_pred, y):
    """Average CE loss over batch."""
    return optax.softmax_cross_entropy_with_integer_labels(logits=y_pred, labels=y).mean()

def total_loss(y_pred_and_l2, y, weight_decay: float):
    """CE + weight decay."""
    y_pred, _, l2_loss = y_pred_and_l2
    return cross_entropy_loss(y_pred, y) + l2_loss * weight_decay

def apply_fn_builder(model):
    """Return apply function closed over model."""
    def apply(variables, x, training=False):
        params = variables['params']
        batch_stats = variables.get("batch_stats", None) or {}
        outputs, updates = model.apply({'params': params, 'batch_stats': batch_stats},
                                       x, training=training, mutable=['batch_stats'] if training else [])
        x_out, pre_activation, _, _ = outputs
        l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        return x_out, updates, l2_loss
    return apply


def sample_hessian(prediction, sample):
    """Dummy Hessian hook (kept for API compatibility)."""
    from optimizers import sample_crossentropy_hessian
    return (sample_crossentropy_hessian(prediction, sample[0]), 0.0, 0.0)

def compute_metrics(metrics_obj, *, loss, l2_loss, outputs, labels):
    """Update metrics with model outputs."""
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    metric_updates = metrics_obj.single_from_model_output(
        logits=logits, labels=labels, loss=loss, l2_loss=l2_loss)
    return metrics_obj.merge(metric_updates)

# ---------------- Model builders ----------------
MODEL_CLASS_MAP = {
    "no_embed": MLPOneHot,
    "one_embed": MLPOneEmbed,
    "one_embed_residual": MLPOneEmbedResidual,
    "two_embed": MLPTwoEmbed,
    "no_embed_cheating": MLPOneHot_cheating,
    "one_embed_cheating": MLPOneEmbed_cheating,
    "two_embed_cheating": MLPTwoEmbed_cheating,
}

def build_model(MLP_class: str, num_layers: int, num_neurons: int, features: int, group_size: int) -> DonutMLP:
    """Instantiate model from class name and hyperparams."""
    base = MLP_class.lower()
    if base not in MODEL_CLASS_MAP:
        raise ValueError(f"Unknown MLP_class: {MLP_class}")
    kwargs = dict(group_size=group_size, num_neurons=num_neurons, num_layers=num_layers)
    if "embed" in base:
        kwargs["features"] = features
    return MODEL_CLASS_MAP[base](**kwargs)

def build_optimizer(optimizer_name: str, lr: float):
    """Return Optax optimizer by name."""
    if optimizer_name == "adam":
        return optax.adam(lr)
    if optimizer_name.startswith("SGD"):
        return optax.sgd(lr, momentum=0.0)
    raise ValueError(f"Unsupported optimizer type: {optimizer_name}")

# ---------------- Dataset / eval grid ----------------
def make_train_batches(p: int, batch_size: int, k: int, random_seed_ints: List[int]):
    """Create per-seed dihedral dataset and stack to (num_models, k, batch, 2)."""
    train_ds_list = []
    for seed in random_seed_ints:
        x, y = dihedral.make_dihedral_dataset(p, batch_size, k, seed)
        train_ds_list.append((x, y))
    x_batches = jnp.stack([x for (x, _) in train_ds_list])
    y_batches = jnp.stack([y for (_, y) in train_ds_list])
    print("x_batches.shape =", x_batches.shape)
    print("y_batches.shape =", y_batches.shape)

    print(f"Number of training batches: {x_batches.shape[1]}")
    print("made dataset")

    dataset_size_bytes = (x_batches.size * x_batches.dtype.itemsize)
    dataset_size_mb = dataset_size_bytes / (1024 ** 2)
    print(f"Dataset size per model: {dataset_size_mb:.2f} MB")

    return train_ds_list, x_batches, y_batches

def make_full_eval_grid(p: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build (|G|^2, 2) inputs and labels for D_n group multiplication."""
    G, _ = DFT.make_irreps_Dn(p)
    idx = {g: i for i, g in enumerate(G)}
    group_size = len(G)
    x_eval = jnp.array([[idx[g], idx[h]] for g in G for h in G], dtype=jnp.int32)
    y_eval = jnp.array([idx[dihedral.mult(g, h, p)] for g in G for h in G], dtype=jnp.int32)
    return x_eval, y_eval  # shapes: (4p^2,2), (4p^2,)

def pad_to_batches(x_eval, y_eval, batch_size: int, num_models: int):
    """Tile eval to all models and pad to whole batches."""
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

# ---------------- TrainState init ----------------
def create_states(model, tx, weight_decay: float, batch_size: int, random_seed_ints: List[int]):
    """Initialize params/opt_state/TrainState for each seed and stack to batch."""
    dummy_x = jnp.zeros((batch_size, 2), dtype=jnp.int32)
    variables_list = [model.init(jax.random.PRNGKey(seed), dummy_x, training=False)
                      for seed in random_seed_ints]
    params_batch = jax.tree_util.tree_map(lambda *args: jnp.stack(args),
                                          *(v['params'] for v in variables_list))
    # opt_state per model
    def init_opt(p): return tx.init(p)
    opt_states = []
    for i in range(len(random_seed_ints)):
        params_i = jax.tree_util.tree_map(lambda x: x[i], params_batch)
        opt_states.append(init_opt(params_i))
    opt_state_batch = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *opt_states)

    # Build apply/loss fn
    apply = apply_fn_builder(model)

    def loss_fn(y_pred_and_l2, y): return total_loss(y_pred_and_l2, y, weight_decay)

    # TrainState objects per model
    states_list = []
    for i, seed in enumerate(random_seed_ints):
        rng_key = jax.random.PRNGKey(seed)
        params_i = jax.tree_util.tree_map(lambda x: x[i], params_batch)
        opt_state_i = jax.tree_util.tree_map(lambda x: x[i], opt_state_batch)
        state = training.TrainState(
            apply_fn=apply, params=params_i, tx=tx, opt_state=opt_state_i,
            loss_fn=loss_fn, loss_hessian_fn=sample_hessian, compute_metrics_fn=compute_metrics,
            rng_key=rng_key, initial_metrics=Metrics, batch_stats=None, injected_noise=0.0
        )
        states_list.append(state)

    states = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *states_list)
    init_metrics = jax.tree_util.tree_map(lambda *args: jnp.stack(args),
                                          *[s.initial_metrics.empty() for s in states_list])
    return states, init_metrics

# ---------------- JIT train/eval epoch ----------------
@jax.jit
def train_epoch(states, x_batches, y_batches, initial_metrics):
    """Run one full epoch over k batches for all models via vmap+scan."""
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

