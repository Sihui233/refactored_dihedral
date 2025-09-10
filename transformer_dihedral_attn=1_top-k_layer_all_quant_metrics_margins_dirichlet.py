import time
import operator
import jax
import jax.numpy as jnp  # JAX NumPy
from flax import linen as nn  # Linen API
from clu import metrics
from flax import struct
import optax  # Common loss functions and optimizers
import chex
import numpy as np
import sys
import flax.serialization as serialization
import jax.tree_util
import plotly.io as pio
import collections, json, os, functools
from collections import Counter
from flax.core.frozen_dict import freeze, unfreeze
from typing import Dict, Any, Tuple, Union, List
import re
from transformer_class import (
    HookPoint, Embed, PosEmbed, Attention, MLP, TransformerBlock,
    TransformerOneEmbed, TransformerTwoEmbed
)
import dihedral, DFT, report

from color_rules import colour_quad_mul_f        # ①  f·(a±b) mod p
from color_rules import colour_quad_mod_g      # ②  (a±b) mod g
from color_rules import colour_quad_a_only, colour_quad_b_only 

jax.config.update("jax_traceback_filtering", 'off')
# jax.config.update("jax_debug_nans", True)

import optimizers
import training
# from transformers_pca_diffusion_plots_w_helpers import generate_pdf_plots_for_matrix
from pca_diffusion_plots_w_helpers import generate_pdf_plots_for_matrix
# from trans_plotting import make_cluster_html_pages

#############################
### 1) DEFINE THE TRANSFORMER IN FLAX
#############################
loss_log_dir = ""

def check_unembed_rank(params, *, atol=1e-17, label="(unknown)") -> None:
    """Assert that params['W_U'] is full-rank and print its top-5 σ."""
    W = np.asarray(params["W_U"]).astype(np.float64)  # promote to f64
    s = np.linalg.svd(W, compute_uv=False)
    rel = s / s.max()
    eff_rank = int((rel > atol).sum())
    top5 = ", ".join(f"{v:8.3e}" for v in s[:9])
    print(f"[rank-check {label}] effective-rank={eff_rank:>2}   top-5 σ: {top5}")
    if eff_rank < 2:
        raise RuntimeError(
            f"W_U lost rank! ({eff_rank} < {W.shape[1]})  "
            "Did you mutate a shared buffer?"
        )


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

if len(sys.argv) < 15:
    print("Usage: script.py <learning_rate> <weight_decay> <p> <batch_size> <optimizer> <epochs> <k> <batch_experiment> <nn_multiplier> <zeta> <training_set_size> <momentum> <injected_noise> <num_mlp_layers> <random_seed_int_1> [<random_seed_int_2> ...]")
    sys.exit(1)

print("start args parsing")
# Parse command-line arguments
learning_rate = float(sys.argv[1])  # stepsize_
weight_decay = float(sys.argv[2])  # L2 norm
p = int(sys.argv[3])
batch_size = int(sys.argv[4])
optimizer = sys.argv[5]
epochs = int(sys.argv[6])
k = int(sys.argv[7])
batch_experiment = sys.argv[8]
# num_neurons = int(sys.argv[9])  # not used, but kept for consistency
nn_multiplier = int(sys.argv[9])
zeta = int(sys.argv[10])
training_set_size = k * batch_size
momentum = float(sys.argv[12])
injected_noise = float(sys.argv[13]) / float(k)
num_mlp_layers = int(sys.argv[14])

group_size = 2 * p

# Accept multiple random seeds
random_seed_ints = [int(seed) for seed in sys.argv[15:]]
num_models = len(random_seed_ints)

# def lr_schedule_fn(step):
#     total_steps = epochs * k
#     warmup_steps = total_steps // 4  # warmup over first 25% of training

#     def warmup_fn(step_):
#         return learning_rate * (step_ / warmup_steps)

#     def constant_fn(step_):
#         return learning_rate

#     return jax.lax.cond(
#         step < warmup_steps,
#         warmup_fn,
#         constant_fn,
#         operand=step
#     )

def lr_schedule_fn(step):
    total_steps = epochs * k
    warmup_steps = total_steps // 2
    cooldown_steps = total_steps - warmup_steps

    def warmup_fn(step_):
        return learning_rate * (step_ / warmup_steps)

    def cooldown_fn(step_):
        return learning_rate * (1 - (step_ - warmup_steps) / cooldown_steps)

    return jax.lax.cond(
        step < warmup_steps,
        warmup_fn,
        cooldown_fn,
        operand=step
    )

print("making dataset")

def make_dataset_for_seed(seed: int):
    x, y = dihedral.make_dihedral_dataset(p, batch_size, k, seed)
    return x, y


train_ds_list = []
for seed in random_seed_ints:
    x, y = make_dataset_for_seed(seed)
    train_ds_list.append((x, y))


train_x = jnp.stack([xy[0] for xy in train_ds_list])  # (num_models, k, batch, 2)
train_y = jnp.stack([xy[1] for xy in train_ds_list])  # (num_models, k, batch)

print("made dataset")

def compute_pytree_size(pytree):
    total_size = 0
    for array in jax.tree_util.tree_leaves(pytree):
        total_size += array.size * array.dtype.itemsize
    return total_size

dataset_size_bytes = (
    train_x.shape[1] * train_x.shape[2] * train_x.shape[3] * train_x.dtype.itemsize +  # x: (k,batch,2)
    train_y.shape[1] * train_y.shape[2] * train_y.dtype.itemsize                       # y: (k,batch)
)
dataset_size_mb = dataset_size_bytes / (1024 ** 2)
print(f"Dataset size per model: {dataset_size_mb:.2f} MB")

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')
    l2_loss: metrics.Average.from_output('l2_loss')

transformer_config = {
    "num_layers": 1,
    "num_mlp_layers": 1,    
    "d_vocab": group_size,
    "d_model": 128,
    "d_head": 32,
    "num_heads": 4,
    "n_ctx": 2,
    "act_type": "ReLU",
    "attn_coeff": 1.0,
    "nn_multiplier": nn_multiplier, 
}


transformer_config["num_mlp_layers"] = num_mlp_layers
NUM_MLP_LAYERS = transformer_config["num_mlp_layers"]
assert transformer_config["d_head"] * transformer_config["num_heads"] == transformer_config["d_model"]
BASE_DIR = f"/home/mila/w/weis/scratch/DL/dihedral_transformer/quantitative_metrics_transformer_{num_mlp_layers}_heatmaps_log{p}_{transformer_config['attn_coeff']}_k_{k}"
os.makedirs(BASE_DIR, exist_ok=True)
model_dir = os.path.join(
    BASE_DIR,
    f"{p}_models_embed",
    f"p={p}_bs={batch_size}_nn={nn_multiplier}_wd={weight_decay}_epochs={epochs}_training_set_size={training_set_size}"
)
os.makedirs(model_dir, exist_ok=True)
model = TransformerTwoEmbed(**transformer_config)
dummy_x = jnp.zeros((batch_size, 2), dtype=jnp.int32)


def cross_entropy_loss(y_pred, y):
    logits_last = logits_last_token(y_pred)
    log_probs = jax.nn.log_softmax(logits_last, axis=-1)
    nll = - log_probs[jnp.arange(y.shape[0]), y]
    return nll.mean()

# --- shared helper: get last-token logits -------------------------
def logits_last_token(logits_3d: jnp.ndarray) -> jnp.ndarray:
    """Return logits at the last position. Shape: (batch, d_vocab)."""
    return logits_3d[:, -1, :]
# ---------------------------------------------------------------------------

def total_loss(y_pred_and_l2, y):
    y_pred, _, l2_loss = y_pred_and_l2
    return cross_entropy_loss(y_pred, y) + l2_loss * weight_decay

# --- apply wrapper: MLP-compatible tuple signature ----------------
def apply(variables, x, training=False, *, capture_intermediates: bool = False):
    """
    MLP-compatible apply:
      returns (outputs, intermediates, l2_loss)
    `intermediates` is {} unless capture_intermediates=True.
    """
    params = variables["params"]

    if capture_intermediates:
        outputs, mut = model.apply(
            {"params": params}, x,
            training=training, mutable=["intermediates"]
        )
        inter = mut.get("intermediates", {})
    else:
        outputs = model.apply({"params": params}, x, training=training)
        inter = {}

    # L2 over all params, same as MLP
    l2_loss = sum(jnp.sum(jnp.square(p_)) for p_ in jax.tree_util.tree_leaves(params))
    return outputs, inter, l2_loss
# ---------------------------------------------------------------------------


def batched_apply(variables_batch, x_batch, training=False, *, capture_intermediates: bool = False):
    """Vectorised apply across models (axis 0)."""
    fn = lambda vars_, xx: apply(vars_, xx, training, capture_intermediates=capture_intermediates)
    return jax.vmap(fn, in_axes=(0, 0))(variables_batch, x_batch)

def sample_hessian(prediction, sample):
    logits_2d = prediction[0][:, -1, :]
    labels = sample[1]  # y_batch
    return (optimizers.sample_crossentropy_hessian(logits_2d, labels), prediction[1], 0.0)

def compute_metrics(metrics_, *, loss, l2_loss, outputs, labels):
    logits_last = logits_last_token(outputs[0])
    metric_updates = metrics_.single_from_model_output(
        logits=logits_last, labels=labels, loss=loss, l2_loss=l2_loss)
    return metrics_.merge(metric_updates)

def prepare_batches(batches_array):
    x = batches_array[:, :, :, :2].astype(jnp.int32)  # [num_models, k, batch_size, 2]
    y = batches_array[:, :, :, 2].astype(jnp.int32)   # [num_models, k, batch_size]
    return x, y

print("Transformer model created.")

G, irreps = DFT.make_irreps_Dn(p)
freq_map = {}
for name, dim, R, freq in irreps:
    freq_map[name] = freq
    print(f"Checking {name}...")
    
    dihedral.check_representation_consistency(G, R, dihedral.mult, p)

idx = {g: i for i, g in enumerate(G)}

variables_list = []
for seed in random_seed_ints:
    rng_key = jax.random.PRNGKey(seed)
    variables = model.init(rng_key, dummy_x, training=False)

    # Now print out every parameter name & its shape:
    def print_param_tree(tree, prefix=""):
        for name, subtree in tree.items():
            if isinstance(subtree, dict):
                print_param_tree(subtree, prefix + name + "/")
            else:
                # leaf array: print its full key and shape
                print(f"{prefix + name} — shape {tuple(subtree.shape)}")

    print("=== Transformer parameter names ===")
    print_param_tree(variables["params"])

    variables_list.append(variables)

model_size_bytes = compute_pytree_size(variables_list[0]["params"])
model_size_mb = model_size_bytes / (1024 ** 2)
print(f"Single model size: {model_size_mb:.2f} MB")

variables_batch = {
    "params": jax.tree_util.tree_map(
        lambda *args: jnp.stack(args),
        *(v["params"] for v in variables_list)
    ),
    "batch_stats": None
}

if optimizer == "adam":
    tx = optax.adam(lr_schedule_fn)
elif optimizer[:3] == "SGD":
    tx = optax.sgd(learning_rate, momentum)
else:
    raise ValueError("Unsupported optimizer type")

def init_opt_state(params):
    return tx.init(params)

opt_state_list = []
for i in range(num_models):
    params_i = jax.tree_map(lambda x: x[i].copy(), variables_batch["params"])
    opt_state_i = init_opt_state(params_i)
    opt_state_list.append(opt_state_i)

opt_state_batch = jax.tree_util.tree_map(lambda *arrs: jnp.stack(arrs), *opt_state_list)

def create_train_state(params, opt_state, rng_key):
    return training.TrainState(
        apply_fn=apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        loss_fn=total_loss,
        loss_hessian_fn=sample_hessian,
        compute_metrics_fn=compute_metrics,
        rng_key=rng_key,
        initial_metrics=Metrics,
        batch_stats=None,
        injected_noise=injected_noise
    )

states_list = []
for i in range(num_models):
    seed = random_seed_ints[i]
    rng_key = jax.random.PRNGKey(seed)
    params_i = jax.tree_map(lambda x: x[i].copy(), variables_batch["params"])
    opt_state_i = jax.tree_map(lambda x: x[i].copy(), opt_state_batch)
    st = create_train_state(params_i, opt_state_i, rng_key)
    states_list.append(st)

states = jax.tree_util.tree_map(lambda *arrs: jnp.stack(arrs), *states_list)

train_x = jax.device_put(train_x)  # [num_models, k, batch_size, 2]
train_y = jax.device_put(train_y)  # [num_models, k, batch_size]

initial_metrics_list = [st.initial_metrics.empty() for st in states_list]
initial_metrics = jax.tree_util.tree_map(lambda *arrs: jnp.stack(arrs), *initial_metrics_list)

# EVAL data
if training_set_size == group_size * group_size:
    print("Train set is the entire dataset. Using the training set for evaluation.")
    # Reshape training data: train_ds is [num_models, k, batch_size, 3].
    # We extract the first two columns for x and the third column for y.
    x_eval = train_x.reshape(num_models, -1, 2)
    y_eval = train_y.reshape(num_models, -1)
else:
    idx = {g: i for i, g in enumerate(G)}
    x_eval = jnp.array([[idx[g], idx[h]] for g in G for h in G], dtype=jnp.int32)
    y_eval = jnp.array([idx[dihedral.mult(g, h, p)] for g in G for h in G], dtype=jnp.int32)
    x_eval = jnp.tile(x_eval[None, :, :], (num_models, 1, 1))
    y_eval = jnp.tile(y_eval[None, :],       (num_models, 1))


x_eval = jax.device_put(x_eval)
y_eval = jax.device_put(y_eval)

eval_batch_size   = batch_size
total_eval_points = x_eval.shape[1]
num_full_batches  = total_eval_points // eval_batch_size
remain            = total_eval_points % eval_batch_size

if remain > 0:
    pad = eval_batch_size - remain
    x_pad = x_eval[:, :pad, :]   # (M, pad, 2)
    y_pad = y_eval[:, :pad]      # (M, pad)
    x_eval_padded = jnp.concatenate([x_eval, x_pad], axis=1)  # (M, N+pad, 2)
    y_eval_padded = jnp.concatenate([y_eval, y_pad], axis=1)  # (M, N+pad)
    num_eval_batches = num_full_batches + 1
else:
    x_eval_padded   = x_eval
    y_eval_padded   = y_eval
    num_eval_batches = num_full_batches

# (M, num_eval_batches, B, ...)
x_eval_batches = x_eval_padded.reshape(num_models, num_eval_batches, eval_batch_size, 2)
y_eval_batches = y_eval_padded.reshape(num_models, num_eval_batches, eval_batch_size)
print("eval grid:", x_eval.shape, "batches:", x_eval_batches.shape, "\n")

@jax.jit
def train_epoch(states_, x_batches, y_batches, init_metrics):
    """
    x_batches: [num_models, k2, batch_size2, 2]
    y_batches: [num_models, k2, batch_size2]
    We do a jax.lax.scan over the k2 dimension (the 'batch' dimension).
    """
    def train_step(carry, batch):
        (st, mets) = carry
        x_, y_ = batch
        new_states, new_metrics = jax.vmap(
            lambda s, m, xx, yy: s.train_step(m, (xx, yy)),
            in_axes=(0, 0, 0, 0)
        )(st, mets, x_, y_)
        return (new_states, new_metrics), None

    carry0 = (states_, init_metrics)
    transposed_x = x_batches.transpose(1, 0, 2, 3)  # shape [k2, num_models, batch_size2, 2]
    transposed_y = y_batches.transpose(1, 0, 2)     # shape [k2, num_models, batch_size2]
    (final_states, final_metrics), _ = jax.lax.scan(
        train_step, carry0, (transposed_x, transposed_y)
    )
    return final_states, final_metrics

@jax.jit
def eval_model(states_, x_batches, y_batches, init_metrics):
    def eval_step(mets, batch):
        x_, y_ = batch
        new_metrics = jax.vmap(
            lambda s, mm, xx, yy: s.eval_step(mm, (xx, yy)),
            in_axes=(0, 0, 0, 0)
        )(states_, mets, x_, y_)
        return new_metrics, None

    mets_ = init_metrics
    transposed_x = x_batches.transpose(1, 0, 2, 3)
    transposed_y = y_batches.transpose(1, 0, 2)
    final_metrics, _ = jax.lax.scan(eval_step, mets_, (transposed_x, transposed_y))
    return final_metrics


# Track the first epoch where each model hits 100% accuracy
first_100_test_loss = [None] * num_models
first_100_cross_entropy_loss = [None] * num_models 
first_100_epoch           = [None] * num_models          
first_100_summary         = [None] * num_models 
cross_entropies = {}

def as_mlp_batches(x: jnp.ndarray, y: jnp.ndarray):
    """
    Ensure shapes are exactly:
      x: (num_models, k, batch, 2)
      y: (num_models, k, batch)
    """
    assert x.ndim == 4 and x.shape[-1] == 2, "x must be (M, K, B, 2)"
    assert y.ndim == 3, "y must be (M, K, B)"
    return x.astype(jnp.int32), y.astype(jnp.int32)

######################################
# 1) Original Training Loop
######################################
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs} (original batch_size={batch_size})")
    train_x, train_y = as_mlp_batches(train_x, train_y)
    states, train_metrics = train_epoch(states, train_x, train_y, initial_metrics)
    # Print training metrics
    for i in range(num_models):
        train_metric_i = jax.tree_util.tree_map(lambda x: x[i], train_metrics)
        train_metric_i = train_metric_i.compute()
        print(f"Model {i + 1}/{num_models}: Train Loss: {train_metric_i['loss']:.6f}, "
              f"Train Accuracy: {train_metric_i['accuracy']:.2%}")

    # Evaluate on test
    print(f"\n--- Test Evaluation at Epoch {epoch + 1} ---")
    test_metrics = eval_model(states, x_eval_batches, y_eval_batches, initial_metrics)
    for i in range(num_models):
        test_metric_i = jax.tree_util.tree_map(lambda x: x[i], test_metrics)
        test_metric_i = test_metric_i.compute()
        test_loss = test_metric_i['loss']
        test_accuracy = test_metric_i['accuracy']
        print(f" Model {i + 1}/{num_models} -> Test Loss: {test_loss:.6f},  Test Acc: {test_accuracy:.2%}")
        cross_entropy_val = test_loss - weight_decay * test_metric_i['l2_loss']

        # Record first time hitting 100%
        if first_100_test_loss[i] is None and test_accuracy > 0.99999:
            first_100_epoch[i]              = epoch + 1            # 1-based
            first_100_test_loss[i]          = float(test_loss)
            first_100_cross_entropy_loss[i] = float(cross_entropy_val)

            # Build once so we can dump it later verbatim
            first_100_summary[i] = {
                "epoch":   first_100_epoch[i],
                "loss":    first_100_test_loss[i],
                "ce_loss": first_100_cross_entropy_loss[i],
            }
    check_unembed_rank(jax.tree_util.tree_map(lambda x: x[0], states.params), label=f"after epoch {epoch+1}")
    print("--- End of Test Evaluation ---\n")

test_metrics = eval_model(states, x_eval_batches, y_eval_batches, initial_metrics)
final_test_accuracies = []

def average_margin(logits_2d: jnp.ndarray, labels_1d: jnp.ndarray) -> float:
    """
    logits_2d : (N , p)   – logits for the last token
    labels_1d : (N,)      – correct class indices
    Returns the mean margin  (logit_correct − best_wrong_logit) over N samples.
    """
    # logit of the correct class
    corr = logits_2d[jnp.arange(logits_2d.shape[0]), labels_1d]

    # best wrong logit: mask the correct one with −∞, then row-max
    wrong = logits_2d.at[(jnp.arange(logits_2d.shape[0]), labels_1d)].set(-jnp.inf)
    best_wrong = wrong.max(axis=1)

    return float(jnp.mean(corr - best_wrong))

for i in range(num_models):
    test_metric = jax.tree_util.tree_map(lambda x: x[i], test_metrics)
    test_metric = test_metric.compute()
    test_accuracy = test_metric["accuracy"]
    final_test_accuracies.append(test_accuracy)
    print(f"Model {i + 1} final test accuracy: {test_accuracy:.2%}")
    cross_entropy = test_metric['loss'] - weight_decay * test_metric['l2_loss']
    seed = random_seed_ints[i]

    # ---------- single-model views ----------
    x_eval_i = x_eval[i]          # (p² , 2)
    y_eval_i = y_eval[i]          # (p² ,)
    params_i = jax.tree_util.tree_map(lambda x: x[i].copy(), states.params)

    # ---------- logits & margin ----------
    logits_full = model.apply({'params': params_i},
                              x_eval_i,
                              training=False)         # ✓ correct shape
    logits_last = logits_full[:, -1, :]               # (p² , p)
    avg_margin  = average_margin(logits_last, y_eval_i)

    cross_entropies[seed] = {
        "cross_entropy": float(cross_entropy),
        "avg_margin":    avg_margin
    }
    
    # (Saving or not saving logic the same as before)
    if test_accuracy >= 0.99999:
        experiment_name = batch_experiment
        optimizer_name = optimizer + str(momentum)
        num_neurons_transformer = transformer_config["d_model"] * nn_multiplier
        params_file_path = (
            f"{BASE_DIR}/params_transformer_r2_heatmap_k={k}_{epochs}/{p}_{k}_nn_{nn_multiplier}_fits_attn-co={transformer_config['attn_coeff']}_top-k_layers={num_mlp_layers}/"
            f"p={p}_bs={batch_size}_k={k}_nn={num_neurons_transformer}_lr={learning_rate}_wd={weight_decay}_epochs={epochs}_"
            f"training_set_size={training_set_size}/params_p_{p}_{batch_experiment}_"
            f"{optimizer_name}_ts_{training_set_size}_bs={batch_size}_nn={nn_multiplier}_"
            f"lr={learning_rate}_wd={weight_decay}_noise={injected_noise}_zeta={zeta}_k={k}_"
            f"rs_{random_seed_ints[i]}.params"
        )
        os.makedirs(os.path.dirname(params_file_path), exist_ok=True)
        with open(params_file_path, 'wb') as f:
            f.write(serialization.to_bytes(jax.tree_util.tree_map(lambda x: x[i], states.params)))
        print(f"Model {i + 1}: Parameters saved to {params_file_path}")
    else:
        print(f"Model {i + 1}: Test accuracy did not exceed 99.9%. Model parameters will not be saved.")
        # print(f"\n--- Misclassified Test Examples for Model {i + 1} ---")
        # single_params = jax.tree_map(lambda x: x[i], states.params)
        # logits = model.apply({'params': single_params}, x_eval, training=False)
        # predictions = jnp.argmax(logits[:, -1, :], axis=-1)
        # y_true = y_eval
        # incorrect_mask = predictions != y_true
        # incorrect_indices = jnp.where(incorrect_mask)[0]
        # if incorrect_indices.size > 0:
        #     print(f"  Total misclassifications: {len(incorrect_indices)}")
        #     for idx, (x_vals, true_label, pred_label) in enumerate(
        #         zip(x_eval[incorrect_indices],
        #             y_true[incorrect_indices],
        #             predictions[incorrect_indices]), 1
        #     ):
        #         a_val, b_val = x_vals
        #         print(f"    {idx}. a: {int(a_val)}, b: {int(b_val)}, True: {int(true_label)}, Predicted: {int(pred_label)}")
        # else:
        #     print("No misclassifications found. All predictions correct.")

    # Write cross-entropy to loss_log.txt
    num_neurons_transformer = transformer_config["d_model"] * nn_multiplier
    loss_log_dir = (
        f"{BASE_DIR}/transformer_r2_heatmap_k={k}_{epochs}/{p}_{k}_nn_{nn_multiplier}_fits_attn-co={transformer_config['attn_coeff']}_top-k_layers={num_mlp_layers}/"
        f"p={p}_bs={batch_size}_k={k}_nn={num_neurons_transformer}_lr={learning_rate}_wd={weight_decay}_epochs={epochs}_"
        f"training_set_size={training_set_size}/"
    )
    loss_log_path = os.path.join(loss_log_dir, "loss_log.txt")
    os.makedirs(loss_log_dir, exist_ok=True)
    with open(loss_log_path, "a") as log_file:
        log_file.write(f"{random_seed_ints[i]},{cross_entropy}\n")

# Finally, write the first_100% records
first_100_path = os.path.join(loss_log_dir, "first_100_acc_test_loss_records.txt")
with open(first_100_path, "a") as f:
    for i in range(num_models):
        if first_100_test_loss[i] is not None:
            f.write(f"{random_seed_ints[i]},{first_100_test_loss[i]},{first_100_cross_entropy_loss[i]}\n")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict


# DIHEDRAL: the grid runs over |G| = 2p, not over p
n = group_size
a_vals = np.arange(n, dtype=int)         # token-1 index ∈ [0, 2p)
b_vals = np.arange(n, dtype=int)         # token-2 index ∈ [0, 2p)
a_grid, b_grid = np.meshgrid(a_vals, b_vals, indexing="ij")
full_inputs = np.stack([a_grid.ravel(), b_grid.ravel()], axis=1).astype(np.int32)

def _extract_hook_pre(params, x_batch, layer: int = 1):
    suffix = f"hook_pre{layer}"

    # 1) Run and grab intermediates
    _, inter = model.apply(
        {"params": params},
        x_batch,
        mutable=["intermediates"],
        training=False,
    )
    ints = inter["intermediates"]

    # 2) DEBUG dump
    print(f"\n[debug] searching for '{suffix}' in your intermediates tree:")
    def _dump_tree(d, indent=0):
        for k, v in d.items():
            if isinstance(v, dict):
                print(" " * indent + f"{k}/")
                _dump_tree(v, indent + 2)
            elif isinstance(v, list):
                shape = getattr(v[0], "shape", None) if v else None
                print(" " * indent + f"{k} -> list(len={len(v)}), item.shape={shape}")
            else:
                shape = getattr(v, "shape", None)
                print(" " * indent + f"{k}: array.shape={shape}")
    _dump_tree(ints)

    # 3) Recursive finder that unwraps both lists and one‐key dicts
    def _find_hook(d):
        if isinstance(d, dict):
            for k, v in d.items():
                # if this key ends in our suffix, unwrap v:
                if k.endswith(suffix):
                    if isinstance(v, list):
                        return v[0]
                    if isinstance(v, dict):
                        # assume exactly one entry mapping full_name -> array
                        return next(iter(v.values()))
                    return v
                # else recurse deeper
                found = _find_hook(v)
                if found is not None:
                    return found
        return None

    arr = _find_hook(ints)
    if arr is None:
        raise KeyError(
            f"Couldn't find any key ending in '{suffix}'.\n"
            f"(Top-level keys were: {list(ints.keys())})"
        )

    # 4) Final sanity check / conversion
    if not hasattr(arr, "shape"):
        raise RuntimeError(f"Expected an array for '{suffix}', but got {type(arr)}")
    print(f"[debug] found '{suffix}' with shape {arr.shape}")
    return np.array(jax.device_get(arr))


def compute_injected_accuracy(params, fitted_pre, x_all, y_all, layer_idx):
    suffix = f"hook_pre{layer_idx}"

    # 1) Capture the current intermediates
    _, inter = model.apply(
        {"params": params},
        x_all,
        mutable=["intermediates"],
        training=False,
    )
    ints = inter["intermediates"]

    # 2) Find the full key-path to your hook_preX, handling dict/list/array leaves
    def find_hook_path(d, path):
        if isinstance(d, dict):
            for k, v in d.items():
                new_path = path + [k]
                # If this key ends in our suffix, dive one final level if needed
                if k.endswith(suffix):
                    if isinstance(v, list):
                        return new_path
                    if isinstance(v, dict):
                        # assume exactly one entry in that dict
                        inner = next(iter(v.keys()))
                        return new_path + [inner]
                    # v is already an array
                    return new_path
                # otherwise recurse
                res = find_hook_path(v, new_path)
                if res:
                    return res
        return None

    hook_path = find_hook_path(ints, [])
    if not hook_path:
        raise KeyError(f"No intermediate ending in '{suffix}' found. Keys seen:\n{list(ints.keys())}")

    print(f"[debug] will inject into path: {' ➔ '.join(hook_path)}")

    # 3) Build the minimal intermediates override dict
    new_int = fitted_pre
    # we want the final (deepest) key to map to a list-of-arrays
    for key in reversed(hook_path):
        if key == hook_path[-1]:
            new_int = { key: [new_int] }
        else:
            new_int = { key: new_int }
    new_intermediates = new_int

    # 4) Rerun with our override
    vars2 = freeze({"params": params, "intermediates": new_intermediates})
    logits_inj = model.apply(vars2, x_all, training=False)
    preds = jnp.argmax(logits_inj[:, -1, :], axis=-1)
    return jnp.mean(preds == y_all)


def batched_gradient_similarity(
        *, model, params,
        a_batch: jnp.ndarray,
        b_batch: jnp.ndarray,
        c_batch: jnp.ndarray
    ) -> jnp.ndarray:
    """
    Computes the cosine-similarity of (∂Q/∂E_a , ∂Q/∂E_b) for all triples in
    the three equal-length 1-D arrays a_batch, b_batch, c_batch.
    Gradients are taken w.r.t. the *sum* of token-embedding and
    positional-embedding, exactly like the authors’ PyTorch code.
    """

    # ── take gradient w.r.t. embedding *after* positional add ────────
    emba, embb = model.extract_embeddings_ab(params)           # (p , d_model)
    pos0, pos1 = params["pos_embed"]["W_pos"][:2]          # (d_model,)

    def scalar_logit(ea_plus_pos, eb_plus_pos, cls):
        seq = jnp.stack([ea_plus_pos, eb_plus_pos])[None, ...]   # (1,2,d)
        logits = model.call_from_embedding_sequence(seq, params)[0]
        return logits[-1, cls]                                   # scalar

    grad_a = jax.grad(scalar_logit, argnums=0)
    grad_b = jax.grad(scalar_logit, argnums=1)

    # add the position vectors now ↴
    vec_a = emba[a_batch]        # (N , d_model)
    vec_b = embb[b_batch]        # (N , d_model)

    g_a = jax.vmap(grad_a)(vec_a, vec_b, c_batch)          # (N , d_model)
    g_b = jax.vmap(grad_b)(vec_a, vec_b, c_batch)

    dot   = jnp.sum(g_a * g_b, axis=1)
    norms = (jnp.linalg.norm(g_a, axis=1) *
             jnp.linalg.norm(g_b, axis=1) + 1e-12)
    return dot / norms  

def _filter_neurons_by_max(mat: np.ndarray, thr: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only columns (neurons) whose max activation over the p^2 grid >= thr.
    mat: (p^2, nn_multiplier)
    returns: (filtered_mat, keep_mask)
    """
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix (p^2, N), got shape {mat.shape}")
    keep = (np.max(mat, axis=0) >= thr)
    if not np.any(keep):
        # Return a correctly shaped empty matrix and the mask
        return mat[:, :0], keep
    return mat[:, keep], keep


def get_all_preacts_and_embeddings(
    *,
    neuron_data: dict[int, dict[int, dict]],
    dominant_freq_clusters,              # list[dict] or {freq: [ids]}
    params: dict,
):
    """
    Returns
    -------
    embeddings            : np.ndarray                         # (p , d_model)
    layer_preacts         : list[dict[str, np.ndarray]]        # as before
    cluster_contribs_logits: dict[str, np.ndarray]             # (p² , p) – after W_U
    cluster_contribs_dmodel: dict[str, np.ndarray]             # (p² , d_model) – before W_U
    """

    # ─────────────────────────── 0. constants & helpers ──────────────────────────
    last_layer_idx   = max(neuron_data)
    last_block_key   = f'blocks_0'            # we only have 1 block

    W_out = np.array(params[last_block_key]['mlp']['W_out'])   # (d_model , d_mlp)
    W_U   = np.array(params['W_U'])                            # (d_model , p)

    eff_W_logits  = W_out.T @ W_U                              # (d_mlp , p)
    eff_W_dmodel  = W_out.T                                    # (d_mlp , d_model)

    # ─────────────────────────── 1. token embeddings ────────────────────────────
    W_a, W_b = model.extract_embeddings_ab(params)               # (p , d_model)
    embeddings_a = np.asarray(W_a)
    embeddings_b = np.asarray(W_b)

    # ─────────────────────────── 2. layer-wise pre-acts ─────────────────────────
    layer_preacts: list[dict[str, np.ndarray]] = []

    # allow both the “single-dict” and the “list of dicts” formats you’re using
    if isinstance(dominant_freq_clusters, dict):
        clusters_by_layer = {1: dominant_freq_clusters}
    else:
        clusters_by_layer = {i + 1: d for i, d in enumerate(dominant_freq_clusters)}

    for layer_idx in sorted(neuron_data):
        pre_dict = {}
        for freq_key, ids in clusters_by_layer[layer_idx].items():
            if not ids:
                continue
            cols = []
            for nid in ids:
                v = neuron_data[layer_idx][nid]['real_preactivations'].reshape(-1)  # (p^2,)
                if float(v.max()) >= 1e-2:  # 0.01 threshold
                    cols.append(v)
            if cols:
                mat = np.stack(cols, axis=1)  # (p^2, kept_neurons)
                pre_dict[freq_key] = mat
        layer_preacts.append(pre_dict)

    # ─────────────────────────── 3. cluster → logits  &  cluster → d_model ──────
    cluster_contribs_logits  : dict[str, np.ndarray] = {}
    cluster_contribs_dmodel : dict[str, np.ndarray] = {}

    for freq_key, ids in clusters_by_layer[last_layer_idx].items():
        if not ids:
            continue
        kept_cols = []
        kept_ids  = []
        for nid in ids:
            post = neuron_data[last_layer_idx][nid].get(
                "postactivations",
                np.maximum(neuron_data[last_layer_idx][nid]["real_preactivations"], 0.0)
            ).reshape(-1)  # (p^2,)
            if float(post.max()) >= 1e-2:  # 0.01 threshold on *activations*
                kept_cols.append(post)
                kept_ids.append(nid)

        if kept_cols:
            post_mat = np.stack(kept_cols, axis=1)  # (p^2, kept_neurons)

            # IMPORTANT: index the effective weights with the *kept* neuron ids
            W_logits_sub = eff_W_logits[kept_ids, :]   # (kept_neurons, p)
            cluster_contribs_logits[freq_key]  = post_mat @ W_logits_sub

            W_dmodel_sub = eff_W_dmodel[kept_ids, :]   # (kept_neurons, d_model)
            cluster_contribs_dmodel[freq_key] = post_mat @ W_dmodel_sub

    return embeddings_a, embeddings_b, layer_preacts, cluster_contribs_logits, cluster_contribs_dmodel

def detach_to_numpy(tree):
    """
    Return a genuine NumPy copy of every leaf.
    • jax.device_get()  → bring to host
    • np.array(..., copy=True)  → own the buffer
    """
    return jax.tree_util.tree_map(
        lambda x: np.array(jax.device_get(x), copy=True),
        tree,
    )

# ──────────────────────────────────────────────────────────────
# 1)  helper – effective weights of the **last MLP layer**
#     into the final logits   (W_outᵀ · W_U)
# ──────────────────────────────────────────────────────────────
def _effective_final_weights(params_block, params_top) -> np.ndarray:
    """
    params_block : the *last* TransformerBlock params  (e.g. params['blocks_0'])
    params_top   : the root-level params  (contains 'W_U')

    Returns
    -------
    eff_W : (d_mlp , p)   weight that multiplies the post-ReLU neuron
                        activations of the *deepest* MLP layer and goes
                        straight to the p logits.
    """
    W_out = np.array(params_block['mlp']['W_out'])         # (d_model , d_mlp)
    W_U   = np.array(params_top['W_U'])                    # (d_model , p)
    #   logits = W_Uᵀ · (resid + W_out · h)
    # incremental term for neuron n is  (W_Uᵀ · W_out[:, n]) · hₙ
    return W_out.T @ W_U                                   # (d_mlp , p)

def _mod_inverse(a: int, p: int) -> int:
    """Modular inverse (p is prime)."""
    return pow(a, p - 2, p)   # Fermat little theorem
# ──────────────────────────────────────────────────────────────
# 2)  helper – grab **all pre-activations** we need
# ──────────────────────────────────────────────────────────────
# def _collect_neuron_data(params_single,
#                         *, p: int,
#                         num_mlp_layers: int,
#                         model) -> (dict, dict):
#     """
#     Builds the neuron_data and the dominant_freq_clusters
#     exactly in the same format as the MLP code.

#     Returns
#     -------
#     neuron_data : { layer_idx -> { neuron_id -> {...} } }
#     clusters    : list[dict]  (one dict per layer) mapping freq -> [neuron_ids]
#     """
#     n = group_size                       # just a short alias
#     a_vals = np.arange(n)
#     b_vals = np.arange(n)
#     A, B = np.meshgrid(a_vals, b_vals, indexing="ij")
#     full_inputs = np.stack([A.ravel(), B.ravel()], axis=-1).astype(np.int32)

#     neuron_data  : dict = {}
#     freq_clusts  : list = []

#     for layer in range(1, num_mlp_layers + 1):
#         # ------------- pull out pre-acts for this layer -----------------
#         pre_all = _extract_hook_pre(params_single, full_inputs, layer)
#         # keep only token-1 (= b index); shape (p² , d_mlp)
#         pre_tok1 = np.array(pre_all[:, 1, :])               # to CPU
#         d_mlp    = pre_tok1.shape[1]

#         neuron_data[layer] = {}
#         freq2ids: Dict[str, List[int]] = collections.defaultdict(list)


#         for n_id in range(d_mlp):
#             grid = pre_tok1[:, n_id].reshape(n, n)          # (p,p)
#             post = np.maximum(grid, 0.0)

#             # ---- 2-D FFT dominant row/col frequencies -----------------------
#             fft  = np.fft.fft2(grid)
#             mag  = np.abs(fft);  mag[0, 0] = 0

#             fa = int(np.argmax(mag.sum(axis=1)[1:(n//2 + 1)]) + 1)  # row-freq
#             fb = int(np.argmax(mag.sum(axis=0)[1:(n//2 + 1)]) + 1)  # col-freq

#             # energy along each axis
#             energy_a = mag[fa, :].sum()
#             energy_b = mag[:, fb].sum()

#             # put the dominant one first
#             if energy_a >= energy_b:
#                 key = f"{fa},{fb}"
#             else:
#                 key = f"{fb},{fa}"

#             freq2ids[key].append(n_id)

#             neuron_data[layer][n_id] = {
#                 "a_values": a_vals,
#                 "b_values": b_vals,
#                 "real_preactivations": grid,
#                 "postactivations": post,
#             }

#         # prune empty freqs & store
#         freq_clusts.append({f: ids for f, ids in freq2ids.items() if ids})

#     return neuron_data, freq_clusts

def _parse_freq_key(freq_key: str) -> tuple[int, int]:
    
        nums = list(map(int, re.findall(r"\d+", str(freq_key))))
        if not nums:
            raise ValueError(f"Unrecognised freq_key: {freq_key!r}")
        if len(nums) == 1:
            return nums[0], nums[0]
        return nums[0], nums[1]
# --- Main analysis loop ---

print("starting main analysis loop")
rho_cache  = DFT.build_rho_cache(G, irreps)
dft_fn     = DFT.jit_wrap_group_dft(rho_cache, irreps, group_size)
subgroups = dihedral.enumerate_subgroups_Dn(p)   # n = group_size//2
x_all = jnp.array([[g, h]
                   for g in range(group_size)
                   for h in range(group_size)],
                  dtype=jnp.int32)

threshold = 0.00001
# collect layer‑1 fits per model
layer1_fits_all = [[] for _ in range(len(random_seed_ints))]

for m_i, seed in enumerate(random_seed_ints):
    trained_params_i = jax.tree_util.tree_map(lambda x: x[m_i].copy(), states.params)
    params_i_safe = detach_to_numpy(trained_params_i)
    check_unembed_rank(params_i_safe, label=f"seed {seed} – analysis-copy")
    print(f"\n===== Seed {seed} =====")
    graph_dir = os.path.join(model_dir, f"graphs_seed_{seed}_refined")
    paper_graph_dir = os.path.join(model_dir, f"p_graphs_seed_{seed}_refined")
    os.makedirs(graph_dir, exist_ok=True)
    tol = 6e-6
    layers_freq = []
    # extract this model’s params
    # params_i = jax.tree_util.tree_map(lambda x: x[m_i], states.params)
    params_i = params_i_safe
    W_U = np.array(params_i['W_U'])                   # or params_top['W_U'] if using block param split
    s = np.linalg.svd(W_U, compute_uv=False)
    rank = np.sum(s > 1e-6)
    W = np.array(params_i['W_U'])          # (128, 59)

    # don't trust Float32 SVD – promote to float64
    s  = np.linalg.svd(W.astype(np.float64), compute_uv=False)
    rel = s / s.max()                    # relative magnitudes

    print("top 10 σ:", s[:10])
    print("relative :", rel[:10])
    print("effective rank(1e-8):", (rel > 1e-8).sum())

    c0 = W[:, 0]
    c1 = W[:, 1]

    # angle between them
    angle = np.rad2deg(np.arccos( np.dot(c0, c1) /
                                (np.linalg.norm(c0)*np.linalg.norm(c1)) ))
    print(f"angle: {angle}")    

    print("old: W_U shape:", W_U.shape)
    print("Singular values:", s[:10])              # show top 10 for context
    print("Estimated rank(W_U):", rank)
    # np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
    print(W_U)


    # Analyze each layer
    neuron_data = {}
    for layer in range(1, NUM_MLP_LAYERS + 1):
        is_deepest = (layer == NUM_MLP_LAYERS)
        print(f"\n[analyzing] Seed {seed} · Layer {layer}")
        # 1) Extract true pre-activations
        pre_all = _extract_hook_pre(params_i, full_inputs, layer)
        pre_tok1 = pre_all[:, 1, :]
        d_mlp = pre_tok1.shape[-1]

        freq_r2s = defaultdict(list)
        freq_fit_type = {}

        pre_tok1 = pre_all[:, 1, :]                                        # (G², d_mlp)

        # ② reshape 成 (G, G, d_mlp) 的二维网格
        prei_grid = np.asarray(pre_tok1).reshape(group_size, group_size, -1)

        # ③ 从网格构造 left/right（沿 b 或 a 求平均，再平铺为 (G², N)）
        left_vec  = prei_grid.mean(axis=1)                                 # (G, d_mlp)
        right_vec = prei_grid.mean(axis=0)                                 # (G, d_mlp)
        left  = np.tile(left_vec[:, None, :],  (1, group_size, 1)).reshape(group_size*group_size, -1)
        right = np.tile(right_vec[None, :, :], (group_size, 1, 1)).reshape(group_size*group_size, -1)

        # ④ 跑你原来的报告代码
        cluster_tau = 1e-3
        color_rule = colour_quad_a_only
        t1 = 2.0 if group_size < 50 else 3
        t2 = 2.0 if group_size < 50 else 3

        artifacts = report.prepare_layer_artifacts(
            prei_grid, left, right, dft_fn, irreps, freq_map,
            prune_cfg={"thresh1": t1, "thresh2": t2, "seed": 0},
            store_full_neuron_grids=True
        )

        clusters_layer = artifacts["freq_cluster"]
        layers_freq.append(clusters_layer)
        layer_neuron_data = artifacts["neuron_data"]
        neuron_data[layer]=layer_neuron_data

        coset_masks_L = dihedral.build_coset_masks(G, subgroups, dihedral.mult, p, side="left")
        coset_masks_R = dihedral.build_coset_masks(G, subgroups, dihedral.mult, p, side="right")

        report.make_layer_report(
            prei_grid, left, right, p,
            dft_fn, irreps, 
            coset_masks_L, coset_masks_R,
            graph_dir, cluster_tau, color_rule,
            artifacts
        )


        report.export_cluster_neuron_pages_2x4(
            prei_grid, left, right,
            dft_fn, irreps, 
            paper_graph_dir,
            artifacts,
            rounding_scale=10
        )

        diag_labels = artifacts["diag_labels"]
        names = artifacts["names"]
        approx = report.summarize_diag_labels(diag_labels, p, names)

        filename = f"approx_summary_p{p}.json"
        with open(os.path.join(graph_dir, filename), "w", encoding="utf-8") as f:
            json.dump(approx, f, ensure_ascii=False, indent=2)



        # manual forward debug (unchanged) (two embed?? )
        # embed_mod = Embed(d_vocab=group_size, d_model=transformer_config['d_model'])
        # pos_mod = PosEmbed(max_ctx=2, d_model=transformer_config['d_model'])
        # attn_mod = Attention(
        #     d_model=transformer_config['d_model'], num_heads=transformer_config['num_heads'], d_head=transformer_config['d_head'],
        #     n_ctx=transformer_config['n_ctx'], attn_coeff=transformer_config['attn_coeff']
        # )

        # ---------- Dn manual forward debug ----------
        # x_debug = full_inputs[:16]  # shape (16,2), already built over [0, 2p)
        # y_debug = jnp.array([dihedral.idx_mul(int(i), int(j), G, idx, p) for i, j in x_debug], dtype=jnp.int32)

        # logits_ref = model.apply({'params': params_i}, x_debug)
        # preds_ref = jnp.argmax(logits_ref[:, -1, :], axis=-1)

        def manual_forward(params, x):
            # Same logic as before; only labels change.
            # x_emb  = Embed(d_vocab=group_size, d_model=transformer_config['d_model']).apply({'params': params['embed']}, x)
            # x_emb  = PosEmbed(max_ctx=2, d_model=transformer_config['d_model']).apply({'params': params['pos_embed']}, x_emb)
            x_emb_a = Embed(d_vocab=group_size, d_model=transformer_config['d_model']).apply({'params': params['embed_a']}, x[:, 0])
            x_emb_b = Embed(d_vocab=group_size, d_model=transformer_config['d_model']).apply({'params': params['embed_b']}, x[:, 1])
            x_emb   = jnp.stack([x_emb_a, x_emb_b], axis=1)
            x_emb   = PosEmbed(max_ctx=2, d_model=transformer_config['d_model']).apply({'params': params['pos_embed']}, x_emb)

            att    = Attention(
                d_model=transformer_config['d_model'],
                num_heads=transformer_config['num_heads'], d_head=transformer_config['d_head'],
                n_ctx=transformer_config['n_ctx'], attn_coeff=transformer_config['attn_coeff']
            ).apply({'params': params['blocks_0']['attn']}, x_emb)
            resid  = x_emb + att
            h = resid
            for i in range(NUM_MLP_LAYERS):
                W = params['blocks_0']['mlp'][f'W_{i}']
                b = params['blocks_0']['mlp'][f'b_{i}']
                pre = jnp.einsum("md,bpd->bpm", W, h) + b
                h   = jax.nn.relu(pre)
            out = jnp.einsum("dm,bpm->bpd", params['blocks_0']['mlp']['W_out'], h) + params['blocks_0']['mlp']['b_out']
            resid = resid + out
            return jnp.einsum("dm,bpd->bpm", params['W_U'], resid)

        # preds_man = jnp.argmax(manual_forward(params_i, x_debug)[:, -1, :], axis=-1)
        # print("manual == reference?", jnp.all(preds_man == preds_ref))
        # print("debug accuracy:", jnp.mean(preds_ref == y_debug))

        # # ---------- injected accuracy on full grid ----------
        # y_full = jnp.array([dihedral.idx_mul(int(i), int(j), G, idx, p) for i, j in full_inputs], dtype=jnp.int32)
        # real_pre = _extract_hook_pre(params_i, full_inputs, layer=1)[:, 1, :]  # token-2 preacts
        # fake_acc = compute_injected_accuracy(params_i, real_pre, full_inputs, y_full, layer_idx=1)
        # print(f"▶️ fake-injection acc: {float(fake_acc):.2%}")


    # ──────────────────────────────────────────────────────────────
    # build everything and call the generic tracker
    # ──────────────────────────────────────────────────────────────
    print(f"\n[metrics] collecting full metrics for seed {seed}")
    # single-model params → plain lil’ pytree on host
    params_i = jax.tree_util.tree_map(lambda x: x[m_i], states.params)

    # 3.1  neuron-level tensors & frequency clusters
    # neuron_data, dominant_freq_clusters = _collect_neuron_data(
    #     params_i,
    #     p=p,
    #     num_mlp_layers=NUM_MLP_LAYERS,
    #     model=model,
    # )


    # 3.2  effective weights of deepest MLP → logits
    last_block_key = f'blocks_{transformer_config["num_layers"]-1}'
    eff_W = _effective_final_weights(params_i[last_block_key], params_i)  # (d_mlp, p)
    def _phase_distribution(
        preacts: jnp.ndarray,      # (p , p , N)  real-valued
        threshold: float,
        p: int
    ) -> Counter:
        """
        Returns Counter mapping "phi_a,phi_b" -> count for all neurons whose
        max pre-activation > `threshold`.

        Model assumption for every eligible neuron n                           ⎧
            g_n(a,b) ≈ sin(2π f a/p + 2π φ_a/p)  +  sin(2π f b/p + 2π φ_b/p)  ⎩

        Phase is recovered from the **angle** of the 1-D FFT coefficients at
        the dominant row / column frequencies.  Everything is vectorised and
        runs on GPU –  no Python loops over neurons.
        """
        p_float = float(p)
        N = preacts.shape[-1]                       # number of neurons

        # ---------- keep only “strong” neurons ------------------------------
        max_per_neuron = jnp.max(preacts, axis=(0, 1))        # (N,)
        strong_mask = max_per_neuron > threshold
        if not bool(jnp.any(strong_mask)):
            return Counter()

        pre_strong = jnp.compress(strong_mask, preacts, axis=2)   # (p,p,N’)
        N_strong   = pre_strong.shape[-1]

        # ---------- row / column means  -------------------------------------
        row_mean = jnp.mean(pre_strong, axis=1)          # (p , N’)
        col_mean = jnp.mean(pre_strong, axis=0)          # (p , N’)

        # ---------- FFT along the appropriate axis --------------------------
        fft_row = jnp.fft.fft(row_mean, axis=0)          # (p , N’)
        fft_col = jnp.fft.fft(col_mean, axis=0)          # (p , N’)

        power_row = jnp.abs(fft_row) ** 2
        power_col = jnp.abs(fft_col) ** 2

        # ignore the DC component when searching for the dominant freq
        pos_freq_slice = slice(1, p // 2 + 1)            # 1 … ⌊p/2⌋
        row_slice = power_row[pos_freq_slice, :]
        col_slice = power_col[pos_freq_slice, :]

        fa = jnp.argmax(row_slice, axis=0) + 1           # (N’,)  1-based
        fb = jnp.argmax(col_slice, axis=0) + 1

        # gather the complex coefficients we need
        #    take_along_axis expects indices to have the gather dim present
        coeff_row = jnp.take_along_axis(
            fft_row, fa[None, :], axis=0).squeeze(0)      # (N’,)
        coeff_col = jnp.take_along_axis(
            fft_col, fb[None, :], axis=0).squeeze(0)

        # phase recovery  –  see derivation in the answer text
        phi_a = (-jnp.angle(coeff_row) * p_float) / (2 * jnp.pi * fa.astype(jnp.float32))
        phi_b = (-jnp.angle(coeff_col) * p_float) / (2 * jnp.pi * fb.astype(jnp.float32))

        phi_a_int = jnp.mod(jnp.rint(phi_a), p).astype(jnp.int32)   # 0 … p-1
        phi_b_int = jnp.mod(jnp.rint(phi_b), p).astype(jnp.int32)

        phi_pairs = jnp.stack([phi_a_int, phi_b_int], axis=1)       # (N’,2)
        phi_pairs_np = np.asarray(phi_pairs)

        ctr = Counter()
        for a, b in phi_pairs_np:
            ctr[f"{int(a)},{int(b)}"] += 1
        return ctr
    
    def _phase_distribution_equal_freq(
        preacts: jnp.ndarray,          # (p , p , N)
        threshold: float,
        p: int
    ) -> tuple[Counter, Counter, Counter, Counter, Counter]:
        """
        Three equal‑freq fits + two histogram counters.
        Returns:
        ctr_first:        Counter of phases from first fit
        ctr_second:       Counter of phases from second fit
        ctr_third:        Counter of phases from third fit
        freq_pairs_ctr:   Counter of (f1,f2) frequency‑pairs
        freq_triplets_ctr: Counter of (f1,f2,f3) frequency‑triplets
        """
        p_float = float(p)
        fft_lim = p // 2 + 1

        strong_mask = jnp.max(preacts, axis=(0, 1)) > threshold
        if not bool(jnp.any(strong_mask)):
            return Counter(), Counter(), Counter()

        pre_str = jnp.compress(strong_mask, preacts, axis=2)        # (p,p,N’)
        N_str   = pre_str.shape[-1]

        # ── helper: ONE equal-freq fit ───────────────────────────────────────
        def _single_equal_freq_fit(tensor, avoid_f=None):
            row_m = jnp.mean(tensor, axis=1)
            col_m = jnp.mean(tensor, axis=0)

            fft_r = jnp.fft.fft(row_m, axis=0)
            fft_c = jnp.fft.fft(col_m, axis=0)
            pow_r = jnp.abs(fft_r) ** 2
            pow_c = jnp.abs(fft_c) ** 2

            row_p = pow_r[1:fft_lim, :]
            col_p = pow_c[1:fft_lim, :]

            if avoid_f is not None:
                # turn avoid_f into a list of per-neuron arrays
                if avoid_f.ndim == 1:
                    avoids = [avoid_f]
                else:
                    avoids = [avoid_f[i] for i in range(avoid_f.shape[0])]

                # build one mask that bans *any* of the listed frequencies
                rows = jnp.arange(row_p.shape[0])[:, None]   # shape (fft_lim‑1, 1)
                mask = sum(rows == (af - 1)[None, :] for af in avoids) > 0

                # apply it
                row_p = jnp.where(mask, -1.0, row_p)
                col_p = jnp.where(mask, -1.0, col_p)

            f_sel = jnp.argmax(row_p + col_p, axis=0) + 1           # (N’,)

            coeff_r = jnp.take_along_axis(fft_r, f_sel[None, :], axis=0).squeeze(0)
            coeff_c = jnp.take_along_axis(fft_c, f_sel[None, :], axis=0).squeeze(0)

            phi_a = (-jnp.angle(coeff_r) * p_float) / (2 * jnp.pi * f_sel.astype(jnp.float32))
            phi_b = (-jnp.angle(coeff_c) * p_float) / (2 * jnp.pi * f_sel.astype(jnp.float32))

            phi_a_i = jnp.mod(jnp.rint(phi_a), p).astype(jnp.int32)
            phi_b_i = jnp.mod(jnp.rint(phi_b), p).astype(jnp.int32)

            ctr = Counter()
            for a, b in np.asarray(jnp.stack([phi_a_i, phi_b_i], axis=1)):
                ctr[f"{int(a)},{int(b)}"] += 1

            # build reconstruction for residual
            a_lin = jnp.arange(p)[:, None, None]
            b_lin = jnp.arange(p)[None, :, None]
            two_pi_over_p = 2 * jnp.pi / p
            recon = (jnp.sin(two_pi_over_p * f_sel * a_lin + two_pi_over_p * phi_a_i)
                + jnp.sin(two_pi_over_p * f_sel * b_lin + two_pi_over_p * phi_b_i))

            return ctr, f_sel, recon

        def build_freq_counter(*freq_arrays: jnp.ndarray) -> Counter[str]:
            """
            Count how often each tuple of frequencies occurs.
            E.g. build_freq_counter(f1, f2)  → Counter of "f1,f2"
                build_freq_counter(f1, f2, f3) → Counter of "f1,f2,f3"
            """
            ctr = Counter()
            # turn them into plain Python lists of ints
            lists = [np.asarray(arr).reshape(-1).tolist() for arr in freq_arrays]
            for freqs in zip(*lists):
                key = ",".join(str(int(f)) for f in freqs)
                ctr[key] += 1
            return ctr
        # ── first fit ────────────────────────────────────────────────────────
        ctr_first, f1, recon1 = _single_equal_freq_fit(pre_str)

        # ── second fit on residual ──────────────────────────────────────────
        residual1 = pre_str - recon1
        ctr_second, f2, recon2 = _single_equal_freq_fit(residual1, avoid_f=f1)
        residual2 = residual1 - recon2
        # avoid both f1 and f2 to force a new dominant freq
        avoid_both = jnp.stack([f1, f2], axis=0)
        ctr_third, f3, _ = _single_equal_freq_fit(residual2, avoid_f=avoid_both)

        # ── frequency-pair counter  -----------------------------------------
        freq_pairs_ctr    = build_freq_counter(f1, f2)
        freq_triplets_ctr = build_freq_counter(f1, f2, f3)

        return ctr_first, ctr_second, ctr_third, freq_pairs_ctr, freq_triplets_ctr
    
    def compute_and_track_quantities(
        *,
        seed: int,
        p: int,
        model,                        # trained DonutMLP (or subclass)
        params: dict,                 # parameters for this seed
        neuron_data: Dict[int, Dict[int, Dict[str, Any]]],
        cluster_groupings: Union[Dict[int, list], list],
        final_layer_weights: np.ndarray,     # shape (num_neurons_last, p)
        save_dir: str = ".",
    ) -> None:
        """
        Writes *quantities_{seed}.json* containing:

        • distribution_of_max_preactivations
        • networks_equivariantness_stats      (correct-logit stats)
        • network_margin_stats                (margin  stats)
        • network_loss_stats                  (per-sample CE-loss stats)   ← NEW
        • clusters_equivariantness_stats      (per-cluster correct-logit stats)
        • clusters_margin_stats               (per-cluster margin stats)
        """

        # ───────────── 1) where does each neuron reach its maximum? ─────────────
        dist_counter: collections.Counter[str] = collections.Counter()
        for layer_dict in neuron_data.values():
            for nd in layer_dict.values():
                real = np.asarray(nd.get("real_preactivations", []))
                if real.size:
                    a_idx, b_idx = np.unravel_index(real.argmax(), real.shape)
                    dist_counter[f"{a_idx},{b_idx}"] += 1
        distribution_of_max_preactivations = dict(dist_counter)

        # ───────────── 2) run the whole network on the complete p² grid ─────────
        a_grid, b_grid = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
        x_full = np.stack([a_grid.ravel(), b_grid.ravel()], axis=-1).astype(jnp.int32)

        logits_full = model.apply({"params": params}, x_full, training=False)
        logits = logits_full[:, -1, :]        # shape (p² , p)
        logits_np = np.asarray(logits)
        correct_idx = ((a_grid + b_grid) % p).ravel()

        correct_logits = logits_np[np.arange(p * p), correct_idx]                 # (p²,)

        # ----- margins -----
        tmp = logits_np.copy()
        tmp[np.arange(p * p), correct_idx] = -np.inf
        second_logits = tmp.max(axis=1)
        margins = correct_logits - second_logits

        # ----- per-sample CE loss (log-softmax trick, row-wise) -----
        row_max = logits_np.max(axis=1, keepdims=True)
        logsumexp = row_max + np.log(np.exp(logits_np - row_max).sum(axis=1, keepdims=True))
        ce_losses = (logsumexp.squeeze() - correct_logits)                         # (p²,)

        networks_equivariantness_stats = {
            "min":  float(correct_logits.min()),
            "max":  float(correct_logits.max()),
            "mean": float(correct_logits.mean()),
            "std":  float(correct_logits.std()),
        }
        network_margin_stats = {
            "avg_margin":     float(margins.mean()),
            "min_margin":     float(margins.min()),
            "max_margin":     float(margins.max()),
            "std_dev_margin": float(margins.std()),
        }
        network_loss_stats = {
            "avg_loss":  float(ce_losses.mean()),
            "min_loss":  float(ce_losses.min()),
            "max_loss":  float(ce_losses.max()),
            "std_dev_loss": float(ce_losses.std()),
        }

        # ───────────── 3) stats for frequency-clusters in last hidden layer ─────
        if isinstance(cluster_groupings, collections.abc.Mapping):
            last_clusters = cluster_groupings            # type: ignore
            last_layer_idx = max(neuron_data)
        else:
            last_clusters = cluster_groupings[-1]
            last_layer_idx = len(cluster_groupings)

        layer_nd = neuron_data[last_layer_idx]
        correct_idx_grid = (a_grid + b_grid) % p                                   # p×p

        clusters_equivariantness_stats = {}
        clusters_margin_stats = {}

        for freq, neuron_ids in last_clusters.items():
            if not neuron_ids:
                continue

            # build cluster logits: (p, p, p)
            cluster_logits = np.zeros((p, p, p), dtype=float)
            for n in neuron_ids:
                nd = layer_nd.get(n)
                if nd is None:
                    continue
                post = np.asarray(
                    nd.get("postactivations",
                        np.maximum(nd["real_preactivations"], 0.0))
                )                                           # p×p
                w_row = final_layer_weights[n]              # p,
                cluster_logits += post[..., None] * w_row

            # correct-logit stats
            corr = cluster_logits[np.arange(p)[:, None],
                                np.arange(p)[None, :],
                                correct_idx_grid]
            corr_flat = corr.ravel()
            clusters_equivariantness_stats[str(freq)] = {
                "min":  float(corr_flat.min()),
                "max":  float(corr_flat.max()),
                "mean": float(corr_flat.mean()),
                "std":  float(corr_flat.std()),
            }

            # margin stats (for the cluster contribution alone)
            logits_flat = cluster_logits.reshape(p * p, p)
            tmp = logits_flat.copy()
            tmp[np.arange(p * p), correct_idx] = -np.inf
            second = tmp.max(axis=1)
            cluster_margins = corr_flat - second
            clusters_margin_stats[str(freq)] = {
                "avg_margin":     float(cluster_margins.mean()),
                "min_margin":     float(cluster_margins.min()),
                "max_margin":     float(cluster_margins.max()),
                "std_dev_margin": float(cluster_margins.std()),
            }

        # ───────────── 4) dump everything to JSON ───────────────────────────────
        out = {
            "distribution_of_max_preactivations": distribution_of_max_preactivations,
            "networks_equivariantness_stats":     networks_equivariantness_stats,
            "network_margin_stats":               network_margin_stats,
            "network_loss_stats":                 network_loss_stats,   # ← NEW
            "clusters_equivariantness_stats":     clusters_equivariantness_stats,
            "clusters_margin_stats":              clusters_margin_stats,
        }

        # grad_stats, dist_stats = compute_useless_metrics(
        #     model=model,
        #     params=params,
        #     p=p,                    # 59
        #     rng_seed=42,
        #     max_samples=p*p         # use the full 59² = 3 481 triples
        # )
        # out.update(grad_stats)
        # out.update(dist_stats)

        distribution_of_center_mass = compute_center_mass_distribution(
            neuron_data=neuron_data,
            dominant_freq_clusters=cluster_groupings,
            p=p,
        )

        out["distribution_of_center_mass"] = distribution_of_center_mass

        # ─────────────  Phase & frequency histograms  ─────────────
        phases_free              = Counter()
        phases_equal_first       = Counter()
        phases_equal_second_fit  = Counter()
        phases_equal_third_fit   = Counter()
        freq_pairs_total         = Counter()
        freq_triplets_total      = Counter()

        for layer_dict in neuron_data.values():
            if not layer_dict:
                continue
            pre_layer = jnp.stack(
                [layer_dict[n]["real_preactivations"]
                    for n in sorted(layer_dict)],
                axis=-1                                   # (p,p,N)
            )

            phases_free += _phase_distribution(
                pre_layer, 0.01, p)

            ctr_first, ctr_second, ctr_third, ctr_pairs, ctr_triplets = _phase_distribution_equal_freq(
                pre_layer, 0.01, p)

            phases_equal_first       += ctr_first
            phases_equal_second_fit  += ctr_second
            phases_equal_third_fit   += ctr_third
            freq_pairs_total         += ctr_pairs
            freq_triplets_total      += ctr_triplets
            

        out["distribution_of_phases"]                       = dict(phases_free)
        out["distribution_of_phases_f_a=f_b"]               = dict(phases_equal_first)
        out["distribution_of_phases_f_a=f_b_second_fit"]    = dict(phases_equal_second_fit)
        out["distribution_of_phases_f_a=f_b_third_fit"]     = dict(phases_equal_third_fit)
        out["frequencies_equal"]                            = dict(freq_pairs_total)
        out["frequencies_equal_triplets"]                   = dict(freq_triplets_total)
        

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"quantities_{seed}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"[compute_and_track_quantities] wrote {path}")

    # 3.3  where to save
    transf_eqv_dir = os.path.join(
        BASE_DIR,
        f"{p}_distributions_equivariantness",
        f"transformer_p={p}_bs={batch_size}_k={k}_dm={transformer_config['d_model']}"
        f"_wd={weight_decay}_lr={learning_rate}"
    )
    os.makedirs(transf_eqv_dir, exist_ok=True)

    def _layer_centres_of_mass(
        preacts: jnp.ndarray,   # (p, p, N)
        freqs:   np.ndarray,    # (N,)
        p: int
    ) -> np.ndarray:            # → (N, 2)  [CoM_a , CoM_b]
        """GPU-optimised centre-of-mass in circular coordinates."""
        # --- modular inverses on CPU ------------------------------------
        invs = np.array([_mod_inverse(int(f), p) for f in freqs], dtype=np.int32)

        a_idx = jnp.arange(p, dtype=jnp.float32)           # (p,)
        b_idx = jnp.arange(p, dtype=jnp.float32)           # (p,)

        @jax.jit
        def _com(act_3d, invs_1d):
            invs_b = invs_1d.astype(jnp.float32)[None, None, :]    # (1,1,N)

            # linear indices → angles; then straighten by invs_b
            ang_a = (2 * jnp.pi * a_idx[:, None, None] / p) * invs_b   # (p,1,N)
            ang_b = (2 * jnp.pi * b_idx[None, :, None] / p) * invs_b   # (1,p,N)

            # use absolute activation as weight  (keeps both peaks)
            w   = jnp.abs(act_3d)                          # (p,p,N)  ≥0
            vec_a = jnp.sum(w * jnp.exp(1j * ang_a), axis=(0, 1))  # (N,)
            vec_b = jnp.sum(w * jnp.exp(1j * ang_b), axis=(0, 1))  # (N,)

            # circular mean → angle in [0, 2π)
            ang_com_a = (jnp.angle(vec_a) + 2 * jnp.pi) % (2 * jnp.pi)
            ang_com_b = (jnp.angle(vec_b) + 2 * jnp.pi) % (2 * jnp.pi)

            com_a = ang_com_a / (2 * jnp.pi) * p           # back to 0…p
            com_b = ang_com_b / (2 * jnp.pi) * p
            return jnp.stack([com_a, com_b], axis=1)       # (N,2)

        return np.asarray(_com(preacts, invs))   

    def compute_center_mass_distribution(
        *,
        neuron_data: Dict[int, Dict[int, Dict[str, Any]]],
        dominant_freq_clusters,      # same structure you already use
        p: int,
    ) -> Dict[str, int]:
        """
        Builds the distribution_of_center_mass counter across *all* layers.
        Keys are "a,b" strings with integer-rounded CoM coordinates.
        """
        counter = Counter()

        # iterate layer-wise ---------------------------------------------
        for layer_idx, layer_dict in neuron_data.items():
            # assemble (p,p,N) tensor and parallel freq list -------------
            neuron_ids      = sorted(layer_dict)
            if not neuron_ids:
                continue
            pre_list        = [layer_dict[n]["real_preactivations"] for n in neuron_ids]
            pre_layer       = np.stack(pre_list, axis=-1)           # (p,p,N)

            # frequencies: look them up from dominant_freq_clusters -----
            if isinstance(dominant_freq_clusters, dict):
                freq_map = dominant_freq_clusters                  # 1-layer case
            else:
                freq_map = dominant_freq_clusters[layer_idx - 1]   # list-of-dicts
            freqs = np.array([
                int(next((k.split(',')[0]          # dominant freq
                        for k, ids in freq_map.items() if n in ids), '1'))
                for n in neuron_ids
            ], dtype=int)

            coms = _layer_centres_of_mass(jnp.asarray(pre_layer), freqs, p)

            # round to nearest integer grid point ------------------------
            com_int = np.rint(coms).astype(int) % p                 # wrap to 0..p-1
            for a, b in com_int:
                counter[f"{a},{b}"] += 1

        return dict(counter)

    # # 3.4  compute & dump – ***same routine as before***
    # compute_and_track_quantities(
    #     seed=seed,
    #     p=p,
    #     model=model,
    #     params=params_i,
    #     neuron_data=neuron_data,
    #     cluster_groupings=,
    #     final_layer_weights=eff_W,
    #     save_dir=transf_eqv_dir,
    # )

    mlp_class_lower = f"transformer_{transformer_config['attn_coeff']}_{num_mlp_layers}"
    features = transformer_config['d_model']

    html_out_dir = os.path.join(BASE_DIR, "cluster_html", f"seed_{seed}")
    # make_cluster_html_pages(
    #     neuron_data=neuron_data,
    #     clusters=dominant_freq_clusters,   # list[dict] per layer in your code
    #     layer_idx=NUM_MLP_LAYERS,          # last hidden layer clusters
    #     p=p,
    #     out_dir=html_out_dir,
    #     show_full_fft=False,               # change to True if you want full fftshift view
    # )

    pdf_root = os.path.join(model_dir, "pdf_plots", f"seed_{seed}")
    embeddings_a, embeddings_b, layer_preacts, cluster_contribs, cluster_contribs_no_wu = get_all_preacts_and_embeddings(
        neuron_data=neuron_data,
        dominant_freq_clusters=layers_freq,
        params=params_i,                 # or any single-model params pytree
    )

    first_dir_path = os.path.join(
        model_dir,
        f"{p}_models_embed_{transformer_config['d_model']}"
        f"p={p}_bs={batch_size}_nn={nn_multiplier}"
        f"_wd={weight_decay}_epochs={epochs}"
        f"_training_set_size={training_set_size}",
    )
    os.makedirs(first_dir_path, exist_ok=True)

    first100_path = os.path.join(
        first_dir_path,
        f"first100_testacc_seed_{seed}.json",
    )

    info = first_100_summary[m_i]              # m_i is the index of this seed
    if info is not None:                       # wasn’t guaranteed to reach 100 %
        with open(first100_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"First-100-epoch summary for seed {seed} saved to {first100_path}")
    else:
        print(f"Seed {seed} never hit 100 % test accuracy – no summary written.")
    
    def plot_cluster_logit_heatmap(
        cluster_contribs_logits: dict[str, np.ndarray],
        cluster_key: str,
        logit_idx: int,
        p: int,
        *,
        title: str | None = None,
        colorscale: str = "RdBu",
        symmetric: bool = True,
        show: bool = True,
    ):
        """
        Visualise how one frequency-cluster contributes to a single logit
        over the (a, b) input grid.

        Parameters
        ----------
        cluster_contribs_logits : dict[str, np.ndarray]
            Output of `get_all_preacts_and_embeddings`.  
            Each value is shape `(p², p)` –  rows are flattened (a,b) pairs,
            columns are logits 0…p-1.
        cluster_key : str
            Which cluster to show (must be a key in `cluster_contribs_logits`).
        logit_idx : int
            The logit you care about (0 ≤ logit_idx < p).
        p : int
            Modulus – size of the a,b grid (so the heat-map will be p × p).
        title : str, optional
            Custom Plotly title.  If None, a default is generated.
        colorscale : str
            Any Plotly colourscale – e.g. "Viridis", "Cividis", "RdBu", …
        symmetric : bool
            If True, the colour range is centred on 0 (nice for positive/negative).
        show : bool
            If True (default) call `fig.show()`.  Otherwise just return the Figure.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        # ── 1. fetch & reshape ────────────────────────────────────────────
        if cluster_key not in cluster_contribs_logits:
            raise KeyError(f"{cluster_key!r} not found in cluster_contribs_logits")
        contrib_mat = cluster_contribs_logits[cluster_key]               # (p², p)
        if not (0 <= logit_idx < contrib_mat.shape[1]):
            raise IndexError(f"logit_idx {logit_idx} out of range 0–{contrib_mat.shape[1]-1}")

        flat = contrib_mat[:, logit_idx]                                 # (p²,)
        # The training code flattened with `np.meshgrid(..., indexing='ij')`
        # so rows iterate over *a* and columns over *b*.  We want x=a, y=b,
        # which is the transpose of that layout:
        heat = flat.reshape(p, p).T                                      # (b, a)

        # ── 2. colour limits (optional symmetric) ────────────────────────
        if symmetric:
            vmax = np.abs(heat).max()
            vmin = -vmax
        else:
            vmin, vmax = heat.min(), heat.max()

        # ── 3. build the Plotly figure ───────────────────────────────────
        fig = go.Figure(
            go.Heatmap(
                x=np.arange(p),          # a-values → x-axis
                y=np.arange(p),          # b-values → y-axis
                z=heat,
                colorscale=colorscale,
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(
                    title=f"Δ logit {logit_idx}",
                    title_side="right"
                ),
            )
        )

        fig.update_layout(
            title=title or f'Cluster "{cluster_key}" → logit {logit_idx}',
            xaxis_title="a",
            yaxis_title="b",
            yaxis=dict(autorange="reversed"),  # (0,0) bottom-left
            width=500, height=500,
        )

        if show:
            fig.show()

        return fig

    out_dir = os.path.join(pdf_root, "temp_logit_plots")
    os.makedirs(out_dir, exist_ok=True)

    # # 2) loop over clusters & desired logits
    # for cluster_key, mat in cluster_contribs.items():
    #     # extract just the numeric freq (assumes keys like "freq_3" or "3")
    #     # try:
    #     #     freq = int(cluster_key.split("_")[-1])
    #     # except ValueError or AttributeError:
    #     freq = cluster_key

    #     for logit_idx in (30, 31, 32):
    #         # 2a) plot (but don't auto-show)
    #         fig = plot_cluster_logit_heatmap(
    #             cluster_contribs,
    #             cluster_key=cluster_key,
    #             logit_idx=logit_idx,
    #             p=group_size,
    #             show=False
    #         )

    #         # 2b) save as PDF
    #         filename = f"f={freq}-logit={logit_idx}_cluster.pdf"
    #         out_path = os.path.join(out_dir, filename)
    #         fig.write_image(out_path, format="pdf")

    #         print(f"Saved {out_path}")

    def _best_line_and_freq(mat: np.ndarray, p: int) -> tuple[str, int] | None:
        """
        Look at average-over-columns grid, take 2D FFT, and find the strongest of:
        (0,f)  vertical   → 'axis'
        (f,0)  horizontal → 'axis'
        (f,f)  diagonal   → 'diag'
        Returns ('axis'|'diag', f) or None if everything is zero.
        """
        if mat.shape[0] != p * p:
            raise ValueError("The first dimension must be p².")
        grid = mat.mean(axis=1).reshape(p, p)
        fft2 = np.fft.fft2(grid)
        mag  = np.abs(fft2)
        mag[0, 0] = 0.0

        vert   = mag[0, 1:]          # (0,f)
        horiz  = mag[1:, 0]          # (f,0)
        diag   = np.diag(mag)[1:]    # (f,f)

        f_vert  = int(np.argmax(vert)  + 1); m_vert  = float(vert[f_vert  - 1])
        f_horiz = int(np.argmax(horiz) + 1); m_horiz = float(horiz[f_horiz - 1])
        f_diag  = int(np.argmax(diag)  + 1); m_diag  = float(diag[f_diag  - 1])

        # choose the biggest line; vertical/horizontal are both 'axis'
        candidates = [
            ("axis", m_vert,  f_vert),
            ("axis", m_horiz, f_horiz),
            ("diag", m_diag,  f_diag),
        ]
        kind, val, f = max(candidates, key=lambda t: t[1])
        if val <= 0.0:
            return None
        return kind, f


    def _concat_mats(mat_list: list[np.ndarray]) -> np.ndarray:
        """Horizontally concatenate a list of matrices (or return the single one)."""
        if not mat_list:
            raise ValueError("Empty matrix list.")
        if len(mat_list) == 1:
            return mat_list[0]
        return np.concatenate(mat_list, axis=1)

    # embeddings  →  freq_list_embeds
    freq_set = set()
    for layer_dict in layer_preacts:
        for freq_key in layer_dict:
            # fa, fb = map(int, freq_key.split(","))
            f = freq_key
            freq_set.update((f, f))

    freq_list_embeds = sorted(freq_set)

    generate_pdf_plots_for_matrix(
        embeddings_a,
        p,
        save_dir=pdf_root,
        seed=seed,
        freq_list=freq_list_embeds,
        tag="embeds_a",
        tag_q = "full", class_string=mlp_class_lower,
        colour_rule=colour_quad_a_only,
        num_principal_components=2
    )

    generate_pdf_plots_for_matrix(
        embeddings_b,
        p,
        save_dir=pdf_root,
        seed=seed,
        freq_list=freq_list_embeds,
        tag="embeds_b",
        tag_q = "full", class_string=mlp_class_lower,
        colour_rule=colour_quad_b_only,
        num_principal_components=2
    )

    # # each MLP layer pre-activations
    # for layer_idx, layer_dict in enumerate(layer_preacts, start=1):
    #     axis_by_f: dict[int, list[np.ndarray]] = {}
    #     diag_by_f: dict[int, list[np.ndarray]] = {}

    #     for freq_key, mat in layer_dict.items():
    #         #fa, fb = map(int, freq_key.split(","))
    #         f = freq_key
    #         fa = f
    #         fb = f

    #         # If fa == fb we treat it as a diagonal (a+b)–type cluster, else axis-aligned.
    #         if fa == fb:
    #             diag_by_f.setdefault(fa, []).append(mat)
    #         else:
    #             # You already ensured the first entry in the key is the dominant axis.
    #             axis_by_f.setdefault(fa, []).append(mat)

        # # (0,f) / (f,0) → concatenate & tag “…_second_order”
        # for best_f, mats in axis_by_f.items():
        #     merged = _concat_mats(mats)  # (p^2, total_neurons)
        #     merged, _ = _filter_neurons_by_max(merged, thr=1e-2)
        #     if merged.shape[1] < 2:
        #         continue 
        #     print(f"[debug] merged AXIS matrix shape: {merged.shape}")
        #     if np.linalg.matrix_rank(merged) < 2:
        #         continue
        #     tag = f"layer{layer_idx}_freq={best_f}_second_order"
        #     generate_pdf_plots_for_matrix(
        #         merged, p, save_dir=pdf_root, seed=seed,
        #         freq_list=[best_f], tag=tag, tag_q = "full", class_string=mlp_class_lower,
        #         colour_rule=colour_quad_mod_g,
        #         num_principal_components=3,
        #     )

        # # (f,f) → separate concatenation & base tag (no suffix)
        # for best_f, mats in diag_by_f.items():
        #     merged = _concat_mats(mats)  # (p^2, total_neurons)
        #     merged, _ = _filter_neurons_by_max(merged, thr=1e-2)
        #     if merged.shape[1] < 2:
        #         continue 
        #     print(f"[debug] merged DIAG matrix shape: {merged.shape}")
        #     if np.linalg.matrix_rank(merged) < 2:
        #         continue
        #     tag = f"layer{layer_idx}_freq={best_f}"
        #     generate_pdf_plots_for_matrix(
        #         merged, p, save_dir=pdf_root, seed=seed,
        #         freq_list=[best_f], tag=tag, tag_q = "full", class_string=mlp_class_lower,
        #         colour_rule=colour_quad_mod_g,
        #         num_principal_components=3,
        #     )

    # last-layer cluster-to-logit contributions

    for key, C_freq in cluster_contribs.items():
        # C_freq: (group_size**2, group_size)
        fa, fb = _parse_freq_key(key)
        freq = fa
        suffix = "" if fa == fb else "_second_order"

        generate_pdf_plots_for_matrix(
            C_freq, p,                 # ★ 第二个参数一定用 group_size = 2*p
            save_dir=pdf_root, seed=seed,
            freq_list=[freq],
            tag=f"cluster_contributions_to_logits_freq={freq}{suffix}",
            tag_q="full",
            colour_rule=colour_quad_mod_g,
            class_string=mlp_class_lower,
            num_principal_components=3,
        )

    # axis_cc: dict[int, list[np.ndarray]] = {}
    # diag_cc: dict[int, list[np.ndarray]] = {}

    # for freq_key, mat in cluster_contribs.items():
    #     fa, fb = _parse_freq_key(freq_key)

    #     # Your clustering guaranteed the *first* entry is the dominant axis
    #     # (see energy_a >= energy_b). Use that consistently here.
    #     if fa == fb:
    #         # diagonal (a+b)-type clusters
    #         diag_cc.setdefault(fa, []).append(mat)
    #     else:
    #         # axis-aligned (“second order”) clusters → group by the dominant axis freq
    #         axis_cc.setdefault(fa, []).append(mat)

    # # axis → “…_second_order”
    # for best_f, mats in axis_cc.items():
    #     merged = _concat_mats(mats)
    #     print(f"[debug] merged AXIS (cluster) shape: {merged.shape}")
    #     if np.linalg.matrix_rank(merged) < 2:
    #         continue
    #     tag = f"cluster_contributions_to_logits_freq={best_f}_second_order"
    #     generate_pdf_plots_for_matrix(
    #         merged, p, save_dir=pdf_root, seed=seed,
    #         freq_list=[best_f], tag=tag, tag_q = "full", class_string=mlp_class_lower,
    #             colour_rule=colour_quad_mod_g,
    #             num_principal_components=3,
    #     )

    # # diagonal → base tag
    # for best_f, mats in diag_cc.items():
    #     merged = _concat_mats(mats)
    #     print(f"[debug] merged DIAG (cluster) shape: {merged.shape}")
    #     if np.linalg.matrix_rank(merged) < 2:
    #         continue
    #     tag = f"cluster_contributions_to_logits_freq={best_f}"
    #     generate_pdf_plots_for_matrix(
    #         merged, p, save_dir=pdf_root, seed=seed,
    #         freq_list=[best_f], tag=tag, tag_q = "full", class_string=mlp_class_lower,
    #             colour_rule=colour_quad_mod_g,
    #             num_principal_components=3,
    #     )

    
    a_all, b_all = jnp.mgrid[0:p//2, 0:p//2] ## ? ##
    X_FULL_GRID  = jnp.stack([a_all.ravel(), b_all.ravel()], axis=-1).astype(jnp.int32)

    dirichlet_E = compute_dirichlet_energy_embedding_transformer(
        model, params_i_safe, X_FULL_GRID, concat=False)
    print(f"[Dirichlet] seed {seed}: {dirichlet_E:.6e}")

    # # optional: keep it in `layer_summaries` so you see it in the layer-JSONs
    # layer_summaries.setdefault(1, {})["dirichlet_energy_everything"] = float(dirichlet_E)

    # ----------------------------------------------------------------
    # 1)  append to the *reconstruction_metrics_…seed_*.json file
    # ----------------------------------------------------------------
    freq_json_dir = os.path.join(
        model_dir,
        f"{p}_freqs_distribution_r2_jsons",
        f"mlp=transformer_p={p}_bs={batch_size}_k={k}_nn={transformer_config['d_model']*nn_multiplier}"
        f"_wd={weight_decay}_lr={learning_rate}"
    )
    os.makedirs(freq_json_dir, exist_ok=True)

    out_path = os.path.join(
        freq_json_dir,
        f"reconstruction_metrics_top-k={k}_seed_{seed}.json"
    )

    if os.path.exists(out_path):
        with open(out_path) as f:
            rec_data = json.load(f)
    else:
        rec_data = {}

    rec_data.setdefault("model", {})["dirichlet_energy_everything"] = float(dirichlet_E)

    with open(out_path, "w") as f:
        json.dump(rec_data, f, indent=2)

    print(f"[Dirichlet] appended to → {out_path}")


    