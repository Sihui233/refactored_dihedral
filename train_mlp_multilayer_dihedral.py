#!/usr/bin/env python
# I commented out this files frequency .json log generation. 
import os
import numpy as np
try:
    import jax
    if all(d.platform != 'gpu' for d in jax.devices()):
        print("⚠️ No GPU detected — enabling multithreading for CPU.")
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=10"
except Exception:
    # If JAX isn't installed or fails to load, fall back safely
    pass
import jax
from jax import config
# config.update("jax_enable_x64", True)
import jax.numpy as jnp 
from clu import metrics
from flax import struct
import optax  
import sys
import json
import flax.serialization as serialization
import jax.tree_util
import copy
from typing import Dict, Any, Tuple, Union, List

from itertools import product
import plotly.io as pio
pio.kaleido.scope.default_timeout = 60 * 5


jax.config.update("jax_traceback_filtering", 'off')
print("Devices available:", jax.devices())
import optimizers
import training
import collections
from collections import Counter 
import plotly.graph_objs as go, plotly.io as pio
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import report
# from persistent_homology_gpu import run_ph_for_point_cloud
from pca_diffusion_plots_w_helpers import (
    generate_pdf_plots_for_matrix,
    generate_interactive_diffusion_map_html,
)
from utils import compute_pytree_size
import DFT
import dihedral
from color_rules import colour_quad_mul_f        # ①  f·(a±b) mod p
from color_rules import colour_quad_mod_g      # ②  (a±b) mod g
from color_rules import colour_quad_a_only, colour_quad_b_only 
# import model MLP classes
from mlp_models_multilayer import DonutMLP, MLPOneEmbed, MLPOneHot, MLPTwoEmbed, MLPTwoEmbed_cheating, MLPOneEmbed_cheating, MLPOneHot_cheating, MLPOneEmbedResidual

if len(sys.argv) < 14:
    print("Usage: script.py <learning_rate> <weight_decay> <p> <batch_size> <optimizer> <epochs> <k> <batch_experiment> <num_neurons> <MLP_class> <features> <num_layers> <random_seed_int_1> [<random_seed_int_2> ...]")
    sys.exit(1)

print("start args parsing")
learning_rate = float(sys.argv[1])  # stepsize
weight_decay = float(sys.argv[2])     # L2 regularization penalty
p = int(sys.argv[3])
batch_size = int(sys.argv[4])
optimizer = sys.argv[5]
epochs = int(sys.argv[6])
k = int(sys.argv[7])
batch_experiment = sys.argv[8]
num_neurons = int(sys.argv[9])
MLP_class = sys.argv[10]
training_set_size = k * batch_size
features = int(sys.argv[11])
num_layers = int(sys.argv[12])
top_k = [1]
random_seed_ints = [int(arg) for arg in sys.argv[13:]]
num_models = len(random_seed_ints)
print(f"args: lr: {learning_rate}, wd: {weight_decay},nn: {num_neurons}, features: {features}, num_layer: {num_layers}")
print(f"Random seeds: {random_seed_ints}")

group_size = 2 * p

# generate for each seed
train_ds_list = []
for seed in random_seed_ints:
    x, y = dihedral.make_dihedral_dataset(p, batch_size, k, seed)
    train_ds_list.append((x, y))

x_batches = jnp.stack([x for (x, _) in train_ds_list])  # (num_models, k, batch_size, 2)
y_batches = jnp.stack([y for (_, y) in train_ds_list])  # (num_models, k, batch_size)
print("x_batches.shape =", x_batches.shape)
print("y_batches.shape =", y_batches.shape)

print(f"Number of training batches: {x_batches.shape[1]}")

# ---------------- FOURIER TRANSFORM (ADDED) ----------------
# REASON: implement custom group Fourier transform for D_n preacts

# assume `irreps` and `G` constructed same as mult
G, irreps = DFT.make_irreps_Dn(p)
freq_map = {}
for name, dim, R, freq in irreps:
    freq_map[name] = freq
    print(f"Checking {name}...")
    
    dihedral.check_representation_consistency(G, R, dihedral.mult, p)

print("made dataset")

dataset_size_bytes = (x_batches.size * x_batches.dtype.itemsize)
dataset_size_mb = dataset_size_bytes / (1024 ** 2)
print(f"Dataset size per model: {dataset_size_mb:.2f} MB")

def positive_he_normal(key, shape, dtype=jnp.float32):
    init = jax.nn.initializers.he_normal()(key, shape, dtype)
    return jnp.abs(init)

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')
    l2_loss: metrics.Average.from_output('l2_loss')

model: DonutMLP
mlp_class_lower = f"{MLP_class.lower()}_{num_layers}"
model_class_map = {
    "no_embed": MLPOneHot,
    "one_embed": MLPOneEmbed,
    "one_embed_residual": MLPOneEmbedResidual,
    "two_embed": MLPTwoEmbed,
    "no_embed_cheating": MLPOneHot_cheating,
    "one_embed_cheating": MLPOneEmbed_cheating,
    "two_embed_cheating": MLPTwoEmbed_cheating,
    }
# Note these two maps can be replaced by better code checking if "cheating" in base_class_name, but for now I'm doing it this way cuz idk what I might add later
vector_addition_class_map = {
    "no_embed_cheating": MLPOneHot_cheating,
    "one_embed_cheating": MLPOneEmbed_cheating,
    "two_embed_cheating": MLPTwoEmbed_cheating,
    }
torus_class_map = {
    "no_embed": MLPOneHot,
    "one_embed": MLPOneEmbed,
    "two_embed": MLPTwoEmbed,
}
base_class_name = MLP_class.lower()

if base_class_name not in model_class_map:
    raise ValueError(f"Unknown MLP_class: {MLP_class}")
print(base_class_name)
if base_class_name not in torus_class_map and base_class_name not in vector_addition_class_map:
    raise ValueError(f"Unknown if MLP_class: {MLP_class} is vec add or torus")
if base_class_name in torus_class_map:
    num_principal_components = 4
if base_class_name in vector_addition_class_map:
    num_principal_components = 2

model_class = model_class_map[base_class_name]

kwargs = dict(p=group_size, num_neurons=num_neurons, num_layers=num_layers)
if "embed" in base_class_name:
    kwargs["features"] = features

model = model_class(**kwargs)
dummy_x = jnp.zeros(shape=(batch_size, 2), dtype=jnp.int32)

def cross_entropy_loss(y_pred, y):
    return optax.softmax_cross_entropy_with_integer_labels(logits=y_pred, labels=y).mean()

def total_loss(y_pred_and_l2, y):
    y_pred, pre_activation, l2_loss = y_pred_and_l2
    return cross_entropy_loss(y_pred, y) + l2_loss * weight_decay

def apply(variables, x, training=False):
    params = variables['params']
    batch_stats = variables.get("batch_stats", None)
    if batch_stats is None:
        batch_stats = {}
    outputs, updates = model.apply({'params': params, 'batch_stats': batch_stats}, x, training=training,
                                   mutable=['batch_stats'] if training else [])
    x_out, pre_activation, _, _ = outputs
    l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    return x_out, updates, l2_loss

def sample_hessian(prediction, sample):
    return (optimizers.sample_crossentropy_hessian(prediction, sample[0]), 0.0, 0.0)

def compute_metrics(metrics, *, loss, l2_loss, outputs, labels):
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    metric_updates = metrics.single_from_model_output(
        logits=logits, labels=labels, loss=loss, l2_loss=l2_loss)
    return metrics.merge(metric_updates)


print("model made")

def init_model(seed):
    rng_key = jax.random.PRNGKey(seed)
    variables = model.init(rng_key, dummy_x, training=False)
    return variables

variables_list = []
for seed in random_seed_ints:
    variables = init_model(seed)
    variables_list.append(variables)

compute_pytree_size(variables_list[0]['params'])

variables_batch = {}
variables_batch['params'] = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *(v['params'] for v in variables_list))
variables_batch['batch_stats'] = None

params_batch = variables_batch['params']

if optimizer == "adam":
    tx = optax.adam(learning_rate)
elif optimizer[:3] == "SGD":
    tx = optax.sgd(learning_rate, 0.0)
else:
    raise ValueError("Unsupported optimizer type")

def init_opt_state(params):
    return tx.init(params)

opt_state_list = []
for i in range(num_models):
    params_i = jax.tree_util.tree_map(lambda x: x[i], params_batch)
    opt_state = init_opt_state(params_i)
    opt_state_list.append(opt_state)

opt_state_batch = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *opt_state_list)

def create_train_state(params, opt_state, rng_key, batch_stats):
    state = training.TrainState(
        apply_fn=apply, params=params, tx=tx,
        opt_state=opt_state,
        loss_fn=total_loss,
        loss_hessian_fn=sample_hessian,
        compute_metrics_fn=compute_metrics,
        rng_key=rng_key,
        initial_metrics=Metrics,
        batch_stats=batch_stats,
        injected_noise=0.0
    )
    return state

states_list = []
for i in range(num_models):
    seed = random_seed_ints[i]
    rng_key = jax.random.PRNGKey(seed)
    params_i = jax.tree_util.tree_map(lambda x: x[i], params_batch)
    opt_state_i = jax.tree_util.tree_map(lambda x: x[i], opt_state_batch)
    batch_stats = None
    state = create_train_state(params_i, opt_state_i, rng_key, batch_stats)
    states_list.append(state)

states = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *states_list)

initial_metrics_list = [state.initial_metrics.empty() for state in states_list]
initial_metrics = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *initial_metrics_list)

### Added for test evaluation ###
# -----------------------------------------------------------
#  Build FULL evaluation grid for D_n  (all g,h ∈ D_n)
# -----------------------------------------------------------
# 1. build D_n elements list and index list
idx  = {g: i for i, g in enumerate(G)}

group_size = len(G)               # == 2 * p

# 2. build x_eval:  shape = (|G|², 2)   —— every row is (idx_g, idx_h)
x_eval = jnp.array(
    [[idx[g], idx[h]] for g in G for h in G],
    dtype=jnp.int32
)                                  # (4 p², 2)

# 3. build y_eval:  shape = (|G|²,)     —— every row is idx[g * h]
y_eval = jnp.array(
    [idx[dihedral.mult(g, h, p)] for g in G for h in G],
    dtype=jnp.int32
)                                  # (4 p²,)

# 4. duplicate eval data to every model (num_models) and do padding → batch structure
x_eval = jax.device_put(x_eval)
y_eval = jax.device_put(y_eval)

x_eval_expanded = jnp.tile(x_eval[None, :, :], (num_models, 1, 1))
y_eval_expanded = jnp.tile(y_eval[None, :],       (num_models, 1))

eval_batch_size   = batch_size 
total_eval_points = x_eval.shape[0]
num_full_batches  = total_eval_points // eval_batch_size
remain            = total_eval_points % eval_batch_size

if remain > 0:
    pad = eval_batch_size - remain
    x_pad = x_eval_expanded[:, :pad, :]
    y_pad = y_eval_expanded[:, :pad]
    x_eval_padded = jnp.concatenate([x_eval_expanded, x_pad], axis=1)
    y_eval_padded = jnp.concatenate([y_eval_expanded, y_pad], axis=1)
    num_eval_batches = num_full_batches + 1
else:
    x_eval_padded   = x_eval_expanded
    y_eval_padded   = y_eval_expanded
    num_eval_batches = num_full_batches

# → (num_models, num_eval_batches, eval_batch_size, …)
x_eval_batches = x_eval_padded.reshape(num_models, num_eval_batches,
                                       eval_batch_size, 2)
y_eval_batches = y_eval_padded.reshape(num_models, num_eval_batches,
                                       eval_batch_size)
print("eval grid:", x_eval.shape, "batches:", x_eval_batches.shape, "\n")

BASE_DIR = f"/home/mila/w/weis/scratch/DL/qualitative_{p}_{mlp_class_lower}_{num_neurons}_features_{features}_k_{k}"
os.makedirs(BASE_DIR, exist_ok=True)
model_dir = os.path.join(
    BASE_DIR,
    f"{p}_models_embed_{features}",
    f"p={p}_bs={batch_size}_nn={num_neurons}_wd={weight_decay}_epochs={epochs}_training_set_size={training_set_size}"
)
os.makedirs(model_dir, exist_ok=True)

# Logging dictionaries for metrics (per epoch)
log_by_seed = {seed: {} for seed in random_seed_ints}
epoch_dft_logs_by_seed = { seed: {} for seed in random_seed_ints }

# logs for effective embeddings, preactivations, and logits.
epoch_embedding_log = {}
epoch_preactivation_log = {}
epoch_logits_log = {}

# Build per-seed train vs test coords
coords_full = np.array(x_eval)   
train_coords_by_seed = {}
test_coords_by_seed  = {}

for i, seed in enumerate(random_seed_ints):
    x, y = train_ds_list[i]
    train_flat = np.array(x.reshape(-1, 2))
    seen = set(map(tuple, train_flat.tolist()))
    train_coords = np.array([xy for xy in coords_full if tuple(xy) in seen], dtype=int)
    test_coords  = np.array([xy for xy in coords_full if tuple(xy) not in seen], dtype=int)
    train_coords_by_seed[seed] = train_coords
    test_coords_by_seed[seed]  = test_coords


# Training and Evaluation Loops
@jax.jit
def train_epoch(states, x_batches, y_batches, initial_metrics):
    def train_step(state_metrics, batch):
        states, metrics = state_metrics
        x, y = batch
        new_states, new_metrics = jax.vmap(
            lambda state, metric, x, y: state.train_step(metric, (x, y)),
            in_axes=(0, 0, 0, 0)
        )(states, metrics, x, y)
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
    def eval_step(metrics, batch):
        x, y = batch
        new_metrics = jax.vmap(
            lambda state, metric, x, y: state.eval_step(metric, (x, y)),
            in_axes=(0, 0, 0, 0)
        )(states, metrics, x, y)
        return new_metrics, None
    metrics = initial_metrics
    transposed_x = x_batches.transpose(1, 0, 2, 3)
    transposed_y = y_batches.transpose(1, 0, 2)
    final_metrics, _ = jax.lax.scan(
        eval_step,
        metrics,
        (transposed_x, transposed_y)
    )
    return final_metrics

# # build all p² inputs [a, b]
# a_grid, b_grid = jnp.mgrid[0:p, 0:p]
# x_freq_all = jnp.stack([a_grid.ravel(), b_grid.ravel()], axis=-1).astype(jnp.int32)
# # num positive frequencies we care about
# max_freq = p // 2  
x_all = jnp.array(
    [[i, j] for i in range(group_size) for j in range(group_size)],
    dtype=jnp.int32
)

# @jax.jit
# def compute_dft_max_all_layers(params):
#     # run model on all inputs at once
#     _, pre_acts_all, _, _ = model.apply({'params': params}, x_freq_all, training=False)
#     # pre_acts_all is a list/tuple of arrays, one per layer: each of shape (p^2, num_neurons_layer)
#     all_mag_max = []
#     for pre in pre_acts_all:
#         # reshape to (p, p, num_neurons)
#         pre_grid = pre.reshape(p, p, pre.shape[-1])
#         # FFT along the 'a' axis
#         fft_grid = jnp.fft.fft(pre_grid, axis=0)
#         mag = jnp.abs(fft_grid)
#         # slice out positive frequencies 1…max_freq
#         mag_sub = mag[1 : max_freq+1, :, :]
#         # take max over b axis → (max_freq, num_neurons)
#         mag_max = jnp.max(mag_sub, axis=1)
#         all_mag_max.append(mag_max)
#     return all_mag_max  # list of (max_freq, num_neurons_layer)

# @jax.jit
# def compute_group_dft_energy_all_layers(params: dict) -> list[dict]:
#     """
#     For each hidden layer, compute the energy of pre-activations in each (irrep_r,irrep_s) basis.
#     Returns a list of dicts: one dict per layer mapping neuron_idx -> {(r,s): energy}
#     """
#     # build all (g,h) input pairs for D_n x D_n
#     x_all = jnp.array(
#         [[i, j] for i in range(group_size) for j in range(group_size)],
#         dtype=jnp.int32
#     )  # shape (4*n^2, 2)

#     # forward pass to get pre-activations for each layer: list of (group_size^2, num_neurons)
#     _, pre_acts_all, _, _ = model.apply({'params': params}, x_all, training=False)


#     energies_all = []
#     for pre in pre_acts_all:
#         # pre: jnp.ndarray of shape (group_size^2, num_neurons)
#         # compute group DFT coefficients and remap to energy
#         Fhat = DFT.group_dft_preacts(pre,G,irreps,group_size)
#         energy_map = DFT.remap_to_energy(Fhat)
#         # energy_map: {neuron_idx: {(r,s): float}}
#         energies_all.append(energy_map)
#     return energies_all

# @jax.jit
# def compute_margin_stats(params, xs, ys):
#     # xs: (N,2) int32; ys: (N,) int32
#     logits = model.apply({'params': params}, xs, training=False)[0]  # (N, C)
#     # correct‐class logits
#     correct = logits[jnp.arange(xs.shape[0]), ys]
#     # mask out correct class
#     one_hot = jax.nn.one_hot(ys, logits.shape[1], dtype=bool)
#     masked = jnp.where(one_hot, -1e9, logits)
#     runner = jnp.max(masked, axis=1)
#     margins = correct - runner  # shape (N,)
#     return jnp.min(margins), jnp.mean(margins)

energy_batch_size = 10 * batch_size

# # Dirichlet‐Energy Helpers
# def compute_embeddings(params, x):
#     a, b = x[:, 0], x[:, 1]
#     emb_a, emb_b = model.extract_embeddings_ab(params)   # (p,D)

#     in_features = params["dense_1"]["kernel"].shape[0]   # D (=128)

#     # special-case two-token Residual MLP 
#     if "V_proj" in params:               # present only in the new model
#         return jnp.concatenate([emb_a[a], emb_b[b]], axis=1)   # (B,2D)
    
#     # one-hot, concat (no_embed)
#     if in_features == 2 * p:
#         return jnp.concatenate(
#             [jax.nn.one_hot(a, p), jax.nn.one_hot(b, p)],
#             axis=1
#         ).astype(jnp.float32)
    
#     # one-hot, addition (no_embed_cheating)
#     if in_features == p:                  # «added» length is p
#         return (jax.nn.one_hot(a, p) + jax.nn.one_hot(b, p)).astype(jnp.float32)

#     if in_features == emb_a.shape[1] + emb_b.shape[1]:
#         return jnp.concatenate([emb_a[a], emb_b[b]], axis=1)
#     elif in_features == emb_a.shape[1]:
#         return emb_a[a] + emb_b[b]
#     else:
#         raise ValueError("Cannot build first-layer input ...")

# def make_energy_funcs(params):
#     # Build f_embed and its Jacobian once for these params
#     def f_embed(x_embed):
#         # x_embed: (2D,) → logits: (p,)
#         logits, _ = model.call_from_embedding(x_embed, params)
#         return logits
#     grad_f = jax.jit(jax.jacrev(f_embed))
#     @jax.jit
#     def batch_energy_sum(batch_emb):
#         jac = jax.vmap(grad_f)(batch_emb)      # (B,2D,p)
#         return jnp.sum(jac**2)                # scalar sum over B
#     # the local emb_fn that already knows `params`
#     def emb_fn(x_data):
#         return compute_embeddings(params, x_data)  # (N,2D)
#     return emb_fn, batch_energy_sum

# def compute_dirichlet_energy_embedding(params, x_data):
#     """
#     params: model parameters
#     x_data:       array (N,2) of input coords
#     energy_batch_size: global from above
#     """
#     compute_emb, batch_energy_sum = make_energy_funcs(params)
#     emb = compute_emb(x_data)                # (N,2D)
#     N = emb.shape[0]

#     total_energy = 0.0
#     # Python loop over chunks:
#     for start in range(0, N, energy_batch_size):
#         chunk = emb[start:start+energy_batch_size]
#         # this is a small JIT'ed call that runs entirely on device
#         total_energy += batch_energy_sum(chunk)

#     # average over N points
#     return total_energy / N                         # scalar

# # JIT-compile
# compute_dirichlet_energy_embedding_jit = jax.jit(compute_dirichlet_energy_embedding)

# === Training Loop ===
first_100_acc_epoch_by_seed = {seed: None for seed in random_seed_ints}
first_epoch_loss_by_seed = {seed: None for seed in random_seed_ints}
first_epoch_ce_loss_by_seed = {seed: None for seed in random_seed_ints} 

# for model_idx in range(num_models):
#     params_i = jax.tree_util.tree_map(lambda x: x[model_idx], states.params)
#     all_mag_max = compute_dft_max_all_layers(params_i)
#     mag_np_list  = [np.array(m) for m in all_mag_max]
#     seed = random_seed_ints[model_idx]
#     # record under epoch 0
#     epoch_dft_logs_by_seed[seed].setdefault(0, {})
#     for layer_idx, mag_np in enumerate(mag_np_list):
#         layer_dict = epoch_dft_logs_by_seed[seed][0].setdefault(layer_idx, {})
#         for neuron_idx in range(mag_np.shape[1]):
#             freq_dict = { str(f): float(mag_np[f-1, neuron_idx])
#                           for f in range(1, max_freq+1) }
#             layer_dict[neuron_idx] = freq_dict
# print("Logged initial (random‐init) DFT → epoch 0")
# for model_idx in range(num_models):
#     params_i = jax.tree_util.tree_map(lambda x: x[model_idx], states.params)
#     energies_all = compute_group_dft_energy_all_layers(params_i)
#     seed = random_seed_ints[model_idx]
#     epoch_dft_logs_by_seed[seed].setdefault(0, {})
#     for layer_idx, energy_map in enumerate(energies_all):
#         layer_dict = epoch_dft_logs_by_seed[seed][0].setdefault(layer_idx, {})
#         for neuron_idx, freq_eng in energy_map.items():
#             layer_dict[neuron_idx] = {k: float(v) for k, v in freq_eng.items()}
# print("Logged initial (random-init) group DFT energies → epoch 0")

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    states, train_metrics = train_epoch(states, x_batches, y_batches, initial_metrics)
    train_losses = []
    train_accuracies = []

    do_eval = (epoch + 1) % 5000 == 0 or (epoch + 1) == epochs
    if do_eval:
        print(f"\n--- Test Evaluation at Epoch {epoch + 1} ---")
        test_metrics = eval_model(states, x_eval_batches, y_eval_batches, initial_metrics)
        test_losses = []
        test_accuracies = []

    for i in range(num_models):
        seed = random_seed_ints[i]
        # Train metrics
        train_metric = jax.tree_util.tree_map(lambda x: x[i], train_metrics)
        train_metric = train_metric.compute()
        train_loss = float(train_metric['loss'])
        train_acc = float(train_metric['accuracy'])
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"Model {i + 1}/{num_models}: Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.2%}")

        # Test metrics
        if do_eval:
            test_metric = jax.tree_util.tree_map(lambda x: x[i], test_metrics)
            test_metric = test_metric.compute()
            test_loss = float(test_metric['loss'])
            test_accuracy = float(test_metric['accuracy'])
            test_l2_loss = float(test_metric['l2_loss'])
            test_ce_loss = test_loss - weight_decay * test_l2_loss
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            print(f"Model {i + 1}/{num_models}: Test CE Loss: {test_ce_loss:.6f}, Test Total Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.2%}")

            if first_100_acc_epoch_by_seed[seed] is None and test_accuracy >= 1.0:
                first_100_acc_epoch_by_seed[seed] = epoch + 1
                first_epoch_loss_by_seed[seed] = test_loss
                first_epoch_ce_loss_by_seed[seed] = test_ce_loss

                print(
                    f"*** Seed {seed} first reached 100% accuracy at epoch {epoch + 1} "
                    f"with total loss {test_loss:.6f} and CE-only loss {test_ce_loss:.6f} ***"
                )

            # Log to dictionary 
            params_i = jax.tree_util.tree_map(lambda x: x[i], states.params)
            weight_norm = float(sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params_i)))

            log_by_seed[seed][epoch + 1] = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_ce_loss": test_ce_loss, 
                "test_accuracy": test_accuracy,
                "l2_weight_norm": weight_norm,
                "first_reach_100%": epoch+1
            }

            # train margin commented out for scaling 
            # tc = train_coords_by_seed[seed]
            # ty = jnp.mod(tc[:, 0] + tc[:, 1], p)
            # train_min, train_avg = compute_margin_stats(params_i, jnp.array(tc), ty)

            # # test margin
            # if(k**2!=p**2):
            #     vc = test_coords_by_seed[seed]
            #     vy = jnp.mod(vc[:, 0] + vc[:, 1], p)
            # else:
            #     vc = tc
            #     vy = ty
            # test_min, test_avg = compute_margin_stats(params_i, jnp.array(vc), vy)

            # # total margin (use full eval grid)
            # total_min, total_avg = compute_margin_stats(params_i, x_eval, y_eval)

            # # update log
            # log_by_seed[seed][epoch + 1].update({
            #     "train_margin":      float(train_min),
            #     "train_avg_margin":  float(train_avg),
            #     "test_margin":       float(test_min),
            #     "test_avg_margin":   float(test_avg),
            #     "min_total_margin":      float(total_min),
            #     "total_avg_margin":  float(total_avg),
            # })

    #         # ------- Dirichlet energy  commented out for scaling
    #         # train_x_all = train_x[i].reshape(-1, train_x.shape[-1])
            
    #         # emb_fn, batch_sum = make_energy_funcs(params_i)

    #         # # Test set
    #         # N_test = x_eval.shape[0]
    #         # emb_test = emb_fn(x_eval)
    #         # sum_energy_test = batch_sum(emb_test)
    #         # de_test = float(sum_energy_test / N_test)

    #         # # Train set
    #         # N_train = train_x_all.shape[0]
    #         # emb_train = emb_fn(train_x_all)
    #         # sum_energy_train = batch_sum(emb_train)
    #         # de_train = float(sum_energy_train / N_train)

    #         # # Total = test + train
    #         # x_total = jnp.concatenate([x_eval, jnp.array(train_coords_by_seed[seed])], axis=0)
    #         # N_total = x_total.shape[0]
    #         # emb_total = emb_fn(x_total)
    #         # sum_energy_total = batch_sum(emb_total)
    #         # de_total = float(sum_energy_total / N_total)

    #         # # append to your log
    #         # log_by_seed[seed][epoch+1].update({
    #         #     "dirichlet_energy_test":  de_test,
    #         #     "dirichlet_energy_train": de_train,
    #         #     "dirichlet_energy_total": de_total,
    #         # })

    # # after your train/test logging…
    # # if do_eval:
    # #     for model_idx in range(num_models):
    # #         params_i = jax.tree_util.tree_map(lambda x: x[model_idx], states.params)
    # #         all_mag_max = compute_dft_max_all_layers(params_i)  # list of jnp arrays
    # #         mag_np_list = [np.array(m) for m in all_mag_max]
    # #         seed = random_seed_ints[model_idx]
    # #         e = epoch + 1
    # #         epoch_dft_logs_by_seed[seed].setdefault(e, {})

    # #         for layer_idx, mag_np in enumerate(mag_np_list):
    # #             layer_dict = epoch_dft_logs_by_seed[seed][e].setdefault(layer_idx, {})
    # #             num_neurons_layer = mag_np.shape[1]
    # #             for neuron_idx in range(num_neurons_layer):
    # #                 # build {freq: max_mag} dict
    # #                 freq_dict = {str(f): float(mag_np[f-1, neuron_idx])
    # #                             for f in range(1, max_freq+1)}
    # #                 layer_dict[neuron_idx] = freq_dict
    # if do_eval:
    #     for model_idx in range(num_models):
    #         params_i = jax.tree_util.tree_map(lambda x: x[model_idx], states.params)
    #         energies_all = compute_group_dft_energy_all_layers(params_i)
    #         seed = random_seed_ints[model_idx]
    #         e = epoch + 1
    #         epoch_dft_logs_by_seed[seed].setdefault(e, {})
    #         for layer_idx, energy_map in enumerate(energies_all):
    #             layer_dict = epoch_dft_logs_by_seed[seed][e].setdefault(layer_idx, {})
    #             for neuron_idx, freq_eng in energy_map.items():
    #                 layer_dict[neuron_idx] = {k: float(v) for k, v in freq_eng.items()}

    # # # === NEW: Log full DFT for every neuron using fixed frequency inputs ===
    # # if (epoch + 1) % 10000 == 0 or (epoch + 1) == epochs:
    # #     for i in range(num_models):
    # #         params_i = jax.tree_util.tree_map(lambda x: x[i], states.params)
    # #         x_freq_b2 = jnp.array([[a, 2] for a in range(p)], dtype=jnp.int32)
    # #         x_freq_b3 = jnp.array([[a, 3] for a in range(p)], dtype=jnp.int32)
    # #         _, pre_acts_b2, _, _ = model.apply({'params': params_i}, x_freq_b2, training=False)
    # #         _, pre_acts_b3, _, _ = model.apply({'params': params_i}, x_freq_b3, training=False)

    # #         # Use only the first hidden layer (index 0)
    # #         pre_act_b2_np = np.array(pre_acts_b2[0])
    # #         pre_act_b3_np = np.array(pre_acts_b3[0])
    # #         seed = random_seed_ints[i]
    # #         if (epoch + 1) not in epoch_dft_logs_by_seed[seed]:
    # #             epoch_dft_logs_by_seed[seed][epoch + 1] = {}
    # #         num_neurons_in_layer = pre_act_b2_np.shape[1]
    # #         for neuron_idx in range(num_neurons_in_layer):
    # #             neuron_pre_b2 = pre_act_b2_np[:, neuron_idx]
    # #             neuron_pre_b3 = pre_act_b3_np[:, neuron_idx]
    # #             fft_b2 = np.fft.fft(neuron_pre_b2)
    # #             fft_b3 = np.fft.fft(neuron_pre_b3)
    # #             max_b2 = np.max(np.abs(fft_b2))
    # #             max_b3 = np.max(np.abs(fft_b3))
    # #             if max_b2 >= max_b3:
    # #                 chosen_fft = fft_b2
    # #             else:
    # #                 chosen_fft = fft_b3
    # #             unique_range = range(1, (p // 2) + 1)
    # #             dft_dict = {str(freq): float(np.abs(chosen_fft[freq])) for freq in unique_range}
    # #             epoch_dft_logs_by_seed[seed][epoch + 1][neuron_idx] = dft_dict

    current_epoch = epoch + 1 

# === Final Evaluation on Test Set ===
print("Starting final evaluation...")
test_metrics = eval_model(states, x_eval_batches, y_eval_batches, initial_metrics)
network_metrics = {}  # To store loss and l2_loss for each seed.
for i in range(num_models):
    test_metric = jax.tree_util.tree_map(lambda x: x[i], test_metrics)
    test_metric = test_metric.compute()
    test_loss = float(test_metric["loss"])
    test_accuracy = float(test_metric["accuracy"])
    test_l2_loss = float(test_metric["l2_loss"])  # extract l2_loss from metrics
    print(f"Model {i + 1}/{num_models}: Final Test Loss: {test_loss:.6f}, Final Test Accuracy: {test_accuracy * 100:.2f}%")
    network_metrics[random_seed_ints[i]] = {"loss": test_loss, "l2_loss": test_l2_loss}
    # if test_accuracy >= 0.999:
    #     params_file_path = os.path.join(
    #         model_dir,
    #         f"params_p_{p}_{optimizer}_ts_{training_set_size}_"
    #         f"bs={batch_size}_nn={num_neurons}_lr={learning_rate}_wd={weight_decay}_"
    #         f"rs_{random_seed_ints[i]}.params"
    #     )
    #     os.makedirs(os.path.dirname(params_file_path), exist_ok=True)
    #     with open(params_file_path, 'wb') as f:
    #         f.write(serialization.to_bytes(jax.tree_util.tree_map(lambda x: x[i], states.params)))
    #     print(f"Model {i + 1}: Parameters saved to {params_file_path}")
    # else:
    #     print(f"Model {i + 1}: Test accuracy did not exceed 99.9%. Model parameters wont be saved")
    #     print(f"\n--- Misclassified Test Examples for Model {i + 1} ---")
    #     logits, _, _, _ = model.apply({'params': jax.tree_util.tree_map(lambda x: x[i], states.params)}, x_eval, training=False)
    #     predictions = jnp.argmax(logits, axis=-1)
    #     y_true = y_eval
    #     incorrect_mask = predictions != y_true
    #     incorrect_indices = jnp.where(incorrect_mask)[0]
    #     if incorrect_indices.size > 0:
    #         misclassified_x = x_eval[incorrect_indices]
    #         misclassified_y_true = y_true[incorrect_indices]
    #         misclassified_y_pred = predictions[incorrect_indices]
    #         print(f"Total Misclassifications: {len(incorrect_indices)}")
    #         for idx, (x_vals, true_label, pred_label) in enumerate(zip(misclassified_x, misclassified_y_true, misclassified_y_pred), 1):
    #             a_val, b_val = x_vals
    #             print(f"{idx}. a: {int(a_val)}, b: {int(b_val)}, True: {int(true_label)}, Predicted: {int(pred_label)}")

# Build new dictionaries based on final epoch grouping for DFT logs ===
final_epoch = epochs
seed_dict_freqs_list = {}
for seed in random_seed_ints:
    seed_dict_freqs_list[seed] = set()
    # Grab the per-layer logs at the final epoch
    final_epoch_log = epoch_dft_logs_by_seed[seed].get(final_epoch, {})
    # For each layer, group neurons by their strongest frequency
    for layer_idx, neuron_dict in final_epoch_log.items():
        grouping = {}
        for neuron_idx, dft_dict in neuron_dict.items():
            # find the freq with max magnitude
            max_key = max(dft_dict, key=dft_dict.get)
            grouping.setdefault(max_key, []).append(neuron_idx)
            seed_dict_freqs_list[seed].add(max_key)

        # For each frequency, build an epoch‐by‐epoch log for this layer
        for freq, neuron_list in grouping.items():
            new_dict = {}
            for epoch_num, layers_log in epoch_dft_logs_by_seed[seed].items():
                layer_logs = layers_log.get(layer_idx, {})
                filtered = {
                    str(n): layer_logs[n]
                    for n in neuron_list
                    if n in layer_logs
                }
                if filtered:
                    new_dict[epoch_num] = filtered

            # Write out JSON with layer and freq in the filename
            output_filepath = os.path.join(
                model_dir,
                f"layer_{layer_idx}_frequency_{freq}_log_seed_{seed}.json"
            )
            with open(output_filepath, 'w') as f:
                json.dump(new_dict, f, indent=2)
            print(
                f"Frequency log for layer {layer_idx}, freq {freq} "
                f"(seed {seed}) saved to {output_filepath}"
            )

for seed in random_seed_ints:
    # get the unique freqs from earlier for filename
    freq_set = seed_dict_freqs_list.get(seed, set())       # a Python set of ints
    if not freq_set:                                       # safety check
        freqs_str = "none"                                 # fallback 
    else:
        freqs_sorted = sorted(freq_set)                    # e.g. [1, 3, 5]
        freqs_str = ",".join(map(str, freqs_sorted))       # "1,3,5"
    # write the JSON
    log_file_path = os.path.join(
        model_dir,
        f"log_features_{features}_({freqs_str})_seed_{seed}.json"
    )
    with open(log_file_path, "w") as f:
        json.dump(log_by_seed[seed], f, indent=2)
    print(f"Final log for seed {seed} saved to {log_file_path}")

from plots_multilayer import (
    plot_cluster_preactivations,
    summed_preactivations,
    summed_postactivations,
    plot_cluster_to_logits,
    plot_all_clusters_to_logits,
    reconstruct_sine_fits_multilayer_logn_fits_layers_after_2,
    fit_sine_wave_multi_freq
)

def convert_to_builtin_type(obj):
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_builtin_type(v) for v in obj]
    else:
        return obj



def zero_dead_neurons_general(params, dead_neurons_by_layer):
    """
    Zero out the weights for dead neurons in a network of arbitrary depth.
    Works whether the first hidden layer is named "dense", "dense_1", or "input_dense".
    """
    new_params = copy.deepcopy(params)

    # 1. Identify the first‑layer key
    for cand in ("dense", "dense_1", "input_dense"):
        if cand in new_params:
            first_layer_key = cand
            break
    else:
        raise ValueError("Could not find first hidden layer in parameters.")

    # 2. Collect deeper hidden layers
    additional_keys = [k for k in new_params
                       if k.startswith("dense_") and k != first_layer_key and k != "output_dense"]
    additional_keys.sort(key=lambda k: int(k.split("_")[1]))  # e.g. dense_2 < dense_3 < …

    final_output_key = "output_dense"

    # 3. Sanity‑check the dead‑neuron list length
    total_layers = 1 + len(additional_keys)
    if len(dead_neurons_by_layer) != total_layers:
        raise ValueError(f"Expected {total_layers} dead‑neuron lists, got {len(dead_neurons_by_layer)}.")

    # 4. Zero first layer
    for idx in dead_neurons_by_layer[0]:
        new_params[first_layer_key]["kernel"] = (
            new_params[first_layer_key]["kernel"].at[:, idx].set(0.0)
        )
        new_params[first_layer_key]["bias"] = (
            new_params[first_layer_key]["bias"].at[idx].set(0.0)
        )

    # Also zero outgoing weights into the next layer, if any.
    if additional_keys:
        next_key = additional_keys[0]
        for idx in dead_neurons_by_layer[0]:
            new_params[next_key]["kernel"] = (
                new_params[next_key]["kernel"].at[idx, :].set(0.0)
            )

    # 5. Zero deeper layers.
    for i, key in enumerate(additional_keys):
        current_dead = dead_neurons_by_layer[i + 1]
        # Zero incoming weights and bias.
        for idx in current_dead:
            new_params[key]["kernel"] = new_params[key]["kernel"].at[:, idx].set(0.0)
            new_params[key]["bias"] = new_params[key]["bias"].at[idx].set(0.0)
        # Zero outgoing weights.
        if i < len(additional_keys) - 1:
            next_key = additional_keys[i + 1]
            for idx in current_dead:
                new_params[next_key]["kernel"] = (
                    new_params[next_key]["kernel"].at[idx, :].set(0.0)
                )
        elif final_output_key in new_params:
            for idx in current_dead:
                new_params[final_output_key]["kernel"] = (
                    new_params[final_output_key]["kernel"].at[idx, :].set(0.0)
                )

    return new_params

# def run_reconstruction(model, model_params, seed, top_k_val,
#                        p, batch_size, k, weight_decay, learning_rate,
#                        mlp_class_lower, contrib_a_np, contrib_b_np, bias_layer1,
#                        model_accuracy, test_total_loss, test_ce_loss):
#     """
#     Run the reconstruction process for a given top-k value.

#     Args:
#       model: The MLP model instance.
#       model_params: The parameters for this model (for seed `seed`).
#       seed: The current seed (for logging file names).
#       top_k_val: The number of key frequencies to use for the reconstruction.
#       p, batch_size, k, weight_decay, learning_rate: Hyperparameters.
#       mlp_class_lower: Lower-cased version of the model class name (used in file paths).
#       contrib_a_np, contrib_b_np: Precomputed contribution arrays from layer 1.
#       bias_layer1: Bias values extracted from layer 1.
#       model_accuracy: The final test accuracy of this model.
#       test_total_loss: The final test total loss of this model.
#       test_ce_loss: The final test cross-entropy loss of this model.
      
#     Returns:
#       A tuple containing:
#          - reconstruction_metrics: dict with loss/accuracy metrics.
#          - layer1_freq: frequency distribution for layer 1 (to learn new top-k).
#          - neuron_data: the dictionary containing per-neuron data.
#          - dominant_freq_clusters: the dominant frequency clusters from reconstruction.
#          - freq_json_dir: directory where reconstruction metrics are saved.
#     """
    # # Gather additional layer parameters
    # additional_layer_keys = [key for key in model_params
    #                          if key.startswith("dense_") and key not in ("dense_1", "output_dense")]
    # additional_layer_keys.sort(key=lambda k: int(k.split("_")[1]))
    # additional_layers_params = [model_params[k] for k in additional_layer_keys]

    # # Call the multilayer reconstruction function
    # (layer1_freq,
    #  additional_layers_freq,
    #  layer1_fits,
    #  additional_layers_fits_lookup,
    #  dead_neurons_layer1,
    #  additional_layers_dead_neurons,
    #  dominant_freq_clusters) = reconstruct_sine_fits_multilayer_logn_fits_layers_after_2(
    #         contrib_a_np,
    #         contrib_b_np,
    #         bias_layer1,
    #         additional_layers_params,
    #         p,
    #         top_k=top_k_val
    # )
    # num_neurons_layer1 = contrib_a_np.shape[1]

    # # Build neuron data (for saving reconstructed preactivations, fitted preactivations, etc)
    # neuron_data = {}
    # a_vals, b_vals = np.arange(p), np.arange(p)
    # a_grid, b_grid = np.meshgrid(a_vals, b_vals, indexing="ij")
    # ab_inputs = np.stack([a_grid.ravel(), b_grid.ravel()], axis=-1).astype(np.int32)
    # _, pre_acts_all, _, _ = model.apply({'params': model_params}, ab_inputs, training=False)
    # real_preacts = [np.array(act).reshape((p, p, -1)) for act in pre_acts_all]

    # # Layer 1 Reconstruction 
    # neuron_data[1] = {}
    # for neuron_idx, (fit_a, fit_b, bias_val) in enumerate(layer1_fits):
    #     fitted = np.zeros((p, p))
    #     for a in range(p):
    #         for b in range(p):
    #             fitted[a, b] = fit_a(a) + fit_b(b) + bias_val
    #     real = real_preacts[0][:, :, neuron_idx]
    #     postact = np.maximum(real, 0.0)
    #     neuron_data[1][neuron_idx] = {
    #         'a_values': np.arange(p),
    #         'b_values': np.arange(p),
    #         'real_preactivations': real,
    #         'fitted_preactivations': fitted,
    #         'postactivations': postact,
    #     }

    # # Additional Layers Reconstruction 
    # for layer_num, fit_lookup in enumerate(additional_layers_fits_lookup, start=2):
    #     neuron_data[layer_num] = {}
    #     real_layer = real_preacts[layer_num - 1]  # layer_num=2 corresponds to index 1.
    #     for neuron_idx, row_fns in enumerate(fit_lookup):
    #         fitted = np.zeros((p, p))
    #         for a in range(p):
    #             for b in range(p):
    #                 fitted[a, b] = row_fns[a](b)
    #         real = real_layer[:, :, neuron_idx]
    #         postact = np.maximum(real, 0.0)
    #         neuron_data[layer_num][neuron_idx] = {
    #             'a_values': np.arange(p),
    #             'b_values': np.arange(p),
    #             'real_preactivations': real,
    #             'fitted_preactivations': fitted,
    #             'postactivations': postact,
    #         }

    # # Set up the directory for logging frequency distributions 
    # freq_json_dir = os.path.join(
    #     BASE_DIR,
    #     f"{p}_freqs_distribution_r2_jsons",
    #     f"mlp={mlp_class_lower}_p={p}_bs={batch_size}_k={k}_nn={num_neurons_layer1}_wd={weight_decay}_lr={learning_rate}"
    # )
    # os.makedirs(freq_json_dir, exist_ok=True)
    
    # # --- Save frequency distributions ---
    # layer1_json_path = os.path.join(freq_json_dir, f"freq_distribution_layer_1_top-k_{top_k_val}_seed_{seed}.json")
    # with open(layer1_json_path, "w") as f:
    #     json.dump(convert_to_builtin_type({str(k): v for k, v in layer1_freq.items()}), f, indent=2)
    # for idx, layer_freq in enumerate(additional_layers_freq, start=2):
    #     layer_json_path = os.path.join(freq_json_dir, f"freq_distribution_layer_{idx}_top-k_{top_k_val}_seed_{seed}.json")
    #     with open(layer_json_path, "w") as f:
    #         json.dump(convert_to_builtin_type(layer_freq), f, indent=2)

    # # Reconstruct the network output 
    # # Reconstruct layer 1 using stored fits
    # h1_dead = np.zeros((p, p, num_neurons_layer1))
    # for n in range(num_neurons_layer1):
    #     fit_a, fit_b, bias_val = layer1_fits[n]
    #     for a in range(p):
    #         for b in range(p):
    #             h1_dead[a, b, n] = np.maximum(fit_a(a) + fit_b(b) + bias_val, 0.0)
    # # On-the-fly reconstruction for layer 1.
    # h1_sim = np.zeros((p, p, num_neurons_layer1))
    # for n in range(num_neurons_layer1):
    #     y_a = contrib_a_np[:, n]
    #     y_b = contrib_b_np[:, n]
    #     fit_a_sim, _ = fit_sine_wave_multi_freq(y_a, p, top_k=top_k_val)
    #     fit_b_sim, _ = fit_sine_wave_multi_freq(y_b, p, top_k=top_k_val)
    #     bias_val = bias_layer1[n]
    #     for a in range(p):
    #         for b in range(p):
    #             h1_sim[a, b, n] = np.maximum(fit_a_sim(a) + fit_b_sim(b) + bias_val, 0.0)
    # h_reconstructed = h1_dead.copy()
    # h_sim = h1_sim.copy()

    # # Process additional layers sequentially.
    # for layer_idx, key in enumerate(additional_layer_keys, start=2):
    #     current_weights = np.array(model_params[key]["kernel"])
    #     current_bias = np.array(model_params[key]["bias"])
    #     h_pre_sim = np.einsum('abn,nm->abm', h_sim, current_weights) + current_bias
    #     h_sim = np.maximum(h_pre_sim, 0)
    #     h_pre = np.einsum('abn,nm->abm', h_reconstructed, current_weights) + current_bias
    #     num_neurons_current = h_pre.shape[-1]
    #     h_reconstructed_new = np.zeros((p, p, num_neurons_current))
    #     lookup_table = additional_layers_fits_lookup[layer_idx - 2]
    #     for m in range(num_neurons_current):
    #         for a in range(p):
    #             for b in range(p):
    #                 h_reconstructed_new[a, b, m] = lookup_table[m][a](b)
    #     h_reconstructed = np.maximum(h_reconstructed_new, 0)

    # # Apply the final output layer
    # final_layer_weights = np.array(model_params["output_dense"]["kernel"])
    # output_bias = np.array(model_params["output_dense"].get("bias", np.zeros(p)))
    # logits_reconstructed_with_dead = np.dot(
    #     h_reconstructed.reshape(-1, h_reconstructed.shape[-1]),
    #     final_layer_weights
    # ) + output_bias
    # logits_reconstructed_with_dead = logits_reconstructed_with_dead.reshape(p, p, -1)
    # logits_reconstructed = np.dot(
    #     h_sim.reshape(-1, h_sim.shape[-1]),
    #     final_layer_weights
    # ) + output_bias
    # logits_reconstructed = logits_reconstructed.reshape(p, p, -1)

    # # Compute test accuracy and losses
    # a_vals = np.arange(p)
    # b_vals = np.arange(p)
    # a_grid, b_grid = np.meshgrid(a_vals, b_vals, indexing='ij')
    # labels = (a_grid + b_grid) % p

    # preds_dead = np.argmax(logits_reconstructed_with_dead, axis=-1)
    # fitted_accuracy_with_dead = np.mean(preds_dead == labels) * 100
    # preds_sim = np.argmax(logits_reconstructed, axis=-1)
    # fitted_accuracy_sim = np.mean(preds_sim == labels) * 100

    # def compute_loss_and_accuracy(logits, p):
    #     logits_flat = logits.reshape(-1, p)
    #     labels_flat = labels.reshape(-1)
    #     losses = []
    #     for i in range(logits_flat.shape[0]):
    #         assert labels_flat[i] < p, f"Invalid label {labels_flat[i]} at index {i} (p = {p})"
    #         logit_i = logits_flat[i]
    #         max_logit = np.max(logit_i)
    #         logsumexp = max_logit + np.log(np.sum(np.exp(logit_i - max_logit)))
    #         loss_i = -logit_i[labels_flat[i]] + logsumexp
    #         losses.append(loss_i)
    #     ce_loss = np.mean(losses)
    #     predictions = np.argmax(logits_flat, axis=1)
    #     accuracy = np.mean(predictions == labels_flat) * 100
    #     return ce_loss, accuracy

    # ce_loss_stored, acc_stored = compute_loss_and_accuracy(logits_reconstructed_with_dead, group_size)
    # ce_loss_onfly, acc_onfly = compute_loss_and_accuracy(logits_reconstructed, group_size)

    # # Update parameters by zeroing out dead neurons
    # dead_by_layer = [dead_neurons_layer1] + additional_layers_dead_neurons
    # updated_params = zero_dead_neurons_general(model_params, dead_by_layer)
    # l2_loss = 0.0
    # for key in additional_layer_keys:
    #     l2_loss += np.sum(np.square(np.array(updated_params[key]["kernel"])))
    # if "output_dense" in updated_params:
    #     l2_loss += np.sum(np.square(np.array(updated_params["output_dense"]["kernel"])))
    # total_loss_stored = ce_loss_stored + weight_decay * l2_loss
    # total_loss_onfly = ce_loss_onfly + weight_decay * l2_loss

    # # Package reconstruction metrics
    # reconstruction_metrics = {
    #     "model": {
    #         "cross_entropy_loss": float(test_ce_loss),
    #         "total_loss": float(test_total_loss),
    #         "accuracy": float(model_accuracy)
    #     },
    #     "stored_fits": {
    #         "cross_entropy_loss": float(ce_loss_stored),
    #         "total_loss": float(total_loss_stored),
    #         "accuracy": float(acc_stored)
    #     },
    #     "on_the_fly": {
    #         "cross_entropy_loss": float(ce_loss_onfly),
    #         "total_loss": float(total_loss_onfly),
    #         "accuracy": float(acc_onfly)
    #     }
    # }
#     # Compute average R² per layer using the frequency distributions
#     r2_per_layer = {}

#     # Layer 1 comes from layer1_freq: {freq: [count, R2, …?]}
#     total_neurons_1 = sum(count for count, r2, *_ in layer1_freq.values())
#     weighted_sum_1  = sum(count * r2 for count, r2, *_ in layer1_freq.values())
#     r2_per_layer['1'] = float(weighted_sum_1 / total_neurons_1) if total_neurons_1 > 0 else 0.0

#     # Additional layers come from additional_layers_freq, a list of dicts
#     for layer_idx, dist in enumerate(additional_layers_freq, start=2):
#         total_neurons = sum(count for count, r2, *_ in dist.values())
#         weighted_sum = sum(count * r2 for count, r2, *_ in dist.values())
#         r2_per_layer[str(layer_idx)] = float(weighted_sum / total_neurons) if total_neurons > 0 else 0.0

#     # merge into the metrics dict
#     reconstruction_metrics.update(r2_per_layer)
#     x_test = jnp.stack([a_grid.ravel(), b_grid.ravel()], axis=-1)       
#     de_test  = float( compute_dirichlet_energy_embedding_jit(model_params, x_test) )
#     reconstruction_metrics["model"].update({
#         "dirichlet_energy_everything":  de_test
#     })

#     output_json_path = os.path.join(freq_json_dir, f"reconstruction_metrics_top-k={top_k_val}_seed_{seed}.json")
#     with open(output_json_path, "w") as f:
#         json.dump(reconstruction_metrics, f, indent=2)
#     print(f"Reconstruction metrics saved to {output_json_path}")
#     return reconstruction_metrics, layer1_freq, neuron_data, dominant_freq_clusters, freq_json_dir

def get_all_preacts_and_embeddings(
        *,                        
        model: DonutMLP,
        params: dict,
        p: int | None = None,
        clusters_by_layer: list[dict[int, list[int]]] | None = None,
):
    
    """
    Build the (p², d_in) matrix that actually feeds the first Dense layer
    and return

    Returns
    -------
    preacts : list[np.ndarray]
        A list of length `model.num_layers`, where
        `preacts[L]` has shape `(p², width_L)` of the raw pre-ReLU activations.
    X_in : np.ndarray
        The `(p², d_in)` input matrix formed by all (a,b) embedding pairs.
    weights : list[np.ndarray]
        Hidden-layer weight kernels; `weights[L]` has shape
        `(in_dim_L, width_L)`.
    cluster_contribs : dict[int, np.ndarray]
        For every frequency `f` in the **last** hidden layer this returns
        a matrix of shape `(p², p)`:
    cluster_weights : dict[int, np.ndarray]
        For every frequency `f` in the last hidden layer, this is the slice
        of the output-layer kernel that feeds the logits from the neurons
        in cluster `f`.  Shape: `(|cluster_f|, p)`.
    
            H_cluster @ W_block
        where
        - `H_cluster` is the ReLU’d activations of the neurons in cluster `f`
          at every of the `p²` inputs, and
        - `W_block` is the slice of the output layer’s weight matrix
          corresponding to those same neurons.
    """
    if clusters_by_layer is None:
        raise ValueError("clusters_by_layer cannot be None")

    p = p or model.p
    X_in = model.all_p_squared_embeddings(params)                # (p², d_in)

    # forward pass once to get *pre-activations*
    _, preacts = model.call_from_embedding(jnp.asarray(X_in), params)
    preacts_np = [np.asarray(layer) for layer in preacts]        # list[(p², width_L)]
    # convert last layer to *post-ReLU activations*
    H_last = np.maximum(preacts_np[-1], 0.0)                     # (p², width_{L})

    # collect hidden-layer kernels 
    weights_np = [np.asarray(params[f"dense_{l}"]["kernel"])
                  for l in range(1, model.num_layers + 1)]

    # build cluster-wise *logit contributions*
    W_out = np.asarray(params["output_dense"]["kernel"])         # (width_L, p)
    cluster_contribs: dict[int, np.ndarray] = {}
    cluster_weights : dict[int, np.ndarray] = {}
    last_layer_clusters = clusters_by_layer[-1]                  # freq → [ids]
    for freq, neuron_ids in last_layer_clusters.items():
        if not neuron_ids:                  # skip empty clusters
            continue
        H_cluster = H_last[:, neuron_ids]               # (p², |cluster|)
        W_block   = W_out[neuron_ids, :]                # (|cluster|, p)
        C_freq    = H_cluster @ W_block                 # (p², p)
        cluster_contribs[freq] = C_freq
        cluster_weights[freq]  = W_block  

    return preacts_np, X_in, weights_np, cluster_contribs, cluster_weights

def make_some_jsons(
    *,
    preacts: list[np.ndarray],
    p: int,
    clusters_by_layer: list[dict[int, list[int]]],
    cluster_weights_to_logits: dict[int, np.ndarray],
    save_dir: str,
    subdir: str = "json",
    float_dtype=np.float32,
    sanity_check: bool = True,
    cluster_contribs_to_logits: dict[int, np.ndarray] | None = None,
) -> str:
    """
    Writes one JSON per *last layer* cluster: cluster_{freq}.json
    For each neuron in the cluster (keyed by its neuron_idx as a string), stores:
      - "preactivations": (p^2,)
      - "w_out":          (p,)
      - "contribs_to_logits": (p^2, p) = ReLU(preacts)[:,None] * w_out[None,:]

    Safety checks:
      • Ensures preacts[-1] is (p^2, width_last)
      • Ensures W_block is (|cluster|, p)
      • Ensures neuron_ids are within [0, width_last)
      • Optional exactness check vs. cluster_contribs_to_logits[freq]
    """
    # ---- global shape checks
    if not preacts:
        raise ValueError("make_some_jsons: empty `preacts`.")
    Z_last = np.asarray(preacts[-1])  # (p^2, width_last)
    n_rows, width_last = Z_last.shape
    if n_rows != p * p:
        raise ValueError(f"make_some_jsons: expected p^2={p*p} rows, got {n_rows}.")
    if not clusters_by_layer:
        raise ValueError("make_some_jsons: empty `clusters_by_layer`.")

    last_layer_clusters = clusters_by_layer[-1] or {}
    if not isinstance(last_layer_clusters, dict):
        raise TypeError("make_some_jsons: clusters_by_layer[-1] must be a dict {freq -> [neuron_ids]}.")

    json_root = os.path.join(save_dir, subdir)
    os.makedirs(json_root, exist_ok=True)

    for freq, neuron_ids in last_layer_clusters.items():
        if not neuron_ids:
            continue

        # Pull the aligned output weights block (built with the SAME order as neuron_ids)
        W_block = cluster_weights_to_logits.get(freq, None)
        if W_block is None:
            # Nothing to write if we don't have this cluster's output weights
            continue
        W_block = np.asarray(W_block)  # (|cluster|, p)

        # ---- index validation & alignment
        ids = np.asarray(neuron_ids, dtype=int)  # (|cluster|)
        valid_mask = (ids >= 0) & (ids < width_last)
        if not np.all(valid_mask):
            bad = ids[~valid_mask].tolist()
            # Filter both ids and W_block rows to keep alignment
            ids = ids[valid_mask]
            W_block = W_block[valid_mask, :]
            if ids.size == 0:
                # No valid neurons remain
                continue
            # (optional) log: print(f"[make_some_jsons] freq={freq}: dropped invalid neuron ids {bad}")

        # ---- shape checks after filtering
        if W_block.shape[0] != ids.shape[0]:
            raise ValueError(
                f"make_some_jsons: for freq={freq}, W_block rows ({W_block.shape[0]}) "
                f"≠ number of neuron ids ({ids.shape[0]})."
            )
        if W_block.shape[1] != p:
            raise ValueError(
                f"make_some_jsons: for freq={freq}, W_block has {W_block.shape[1]} columns, expected p={p}."
            )

        # Gather per-neuron preacts and ReLU
        Z_cluster = Z_last[:, ids]                 # (p^2, |cluster|)
        H_cluster = np.maximum(Z_cluster, 0.0)     # (p^2, |cluster|)

        # Vectorized per-neuron contributions: (p^2, |cluster|, p)
        contribs = H_cluster[:, :, None] * W_block[None, :, :]

        # Optional correctness check against provided cluster_contribs_to_logits
        if sanity_check and (cluster_contribs_to_logits is not None):
            C_freq_expected = np.asarray(cluster_contribs_to_logits.get(freq))
            if C_freq_expected is not None and C_freq_expected.size:
                C_sum = contribs.sum(axis=1)  # (p^2, p)
                if C_freq_expected.shape != C_sum.shape:
                    raise ValueError(
                        f"make_some_jsons: cluster_contribs_to_logits[{freq}] has shape {C_freq_expected.shape}, "
                        f"expected {C_sum.shape}."
                    )
                if not np.allclose(C_sum, C_freq_expected, rtol=1e-5, atol=1e-6):
                    raise ValueError(
                        f"make_some_jsons: contribution mismatch for freq={freq} "
                        f"(sum of per-neuron ≠ cluster total)."
                    )

        # Build JSON payload { "<neuron_idx>": {...}, ... } preserving original order
        payload = {}
        for j, nid in enumerate(ids.tolist()):
            payload[str(int(nid))] = {
                "preactivations": Z_cluster[:, j].astype(float_dtype).tolist(),   # (p^2,)
                "w_out":          W_block[j, :].astype(float_dtype).tolist(),     # (p,)
                "contribs_to_logits": contribs[:, j, :].astype(float_dtype).tolist(),  # (p^2, p)
            }

        out_path = os.path.join(json_root, f"cluster_{freq}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f)

    return json_root

print("starting main analysis loop")
rho_cache  = DFT.build_rho_cache(G, irreps)
dft_fn     = DFT.jit_wrap_group_dft(rho_cache, irreps, group_size)
subgroups = dihedral.enumerate_subgroups_Dn(p)   # n = group_size//2
for seed_idx, seed in enumerate(random_seed_ints):
    graph_dir = os.path.join(model_dir, f"graphs_seed_{seed}_refined")
    paper_graph_dir = os.path.join(model_dir, f"p_graphs_seed_{seed}_refined")

    os.makedirs(graph_dir, exist_ok=True)
    model_params_seed = jax.tree_util.tree_map(lambda x: x[seed_idx], states.params)
    x_all = jnp.array([[g, h]
                       for g in range(group_size)
                       for h in range(group_size)],
                      dtype=jnp.int32)
    _, pre_acts_all, left, right = model.apply({'params': model_params_seed},
                                        x_all, training=False)

    tol = 6e-6
    layers_freq = []
    for layer_idx in range(num_layers):
        prei      = pre_acts_all[layer_idx]
        prei_grid = prei.reshape(group_size, group_size, -1)
        # ### test DFT by verifying reconstruction
        # Fhat      = dft_fn(prei)
        # recon = DFT.inverse_group_dft(Fhat, rho_cache, irreps, group_size, prei_grid.shape[-1])
        # abs_err = jnp.max(jnp.abs(recon - prei_grid))
        # rel_err = (jnp.linalg.norm(recon - prei_grid) /
        #         (jnp.linalg.norm(prei_grid) + 1e-12))

        # print(f"max abs error = {abs_err:.2e} | rel error = {rel_err:.2e}")
        # assert abs_err < tol, f"ABS error too large: {abs_err}"
        
        
        cluster_tau = 1e-3
        color_rule = colour_quad_a_only
        # color_rule = colour_quad_mod_g
        t1 = 2.0 if group_size < 50 else 3
        t2 = 2.0 if group_size < 50 else 3
        artifacts = report.prepare_layer_artifacts(prei_grid, #(G, G, N)
                            left, right, #(G*G, N)
                            dft_fn, irreps, freq_map,
                            prune_cfg={"thresh1": t1, "thresh2": t2, "seed": 0})
        
        coset_masks_L = dihedral.build_coset_masks(G, subgroups, dihedral.mult, p, side="left")
        coset_masks_R = dihedral.build_coset_masks(G, subgroups, dihedral.mult, p, side="right")
        report.make_layer_report(prei_grid,left,right,p,
                                 dft_fn, irreps, 
                                 coset_masks_L, coset_masks_R,
                                 graph_dir, cluster_tau, color_rule,
                                 artifacts
                                 )
        clusters_layer = artifacts["freq_cluster"]
        layers_freq.append(clusters_layer)
        
        # report.export_cluster_neuron_pages_2x4(prei_grid,left,right,
        #                          dft_fn, irreps, 
        #                          paper_graph_dir,
        #                          artifacts,
        #                          rounding_scale=10
        #                          )
        
        diag_labels = artifacts["diag_labels"]
        names = artifacts["names"]
        approx = report.summarize_diag_labels(diag_labels,p,names)
        filename = f"approx_summary_p{p}.json"
        filepath = os.path.join(graph_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(approx, f, ensure_ascii=False, indent=2)

        print(f"Saved summary to {filepath}")
#######
#     # first‐layer bias (always named “dense_1”)
#     bias1 = np.array(model_params_seed["dense_1"]["bias"])

#     # deeper hidden layers
#     additional_layer_keys = [
#         k for k in model_params_seed
#         if k.startswith("dense_")
#         and k not in ("dense_1", "output_dense")
#     ]
#     additional_layer_keys.sort(key=lambda k: int(k.split("_")[1]))
#     biases_by_layer = [bias1] + [
#         np.array(model_params_seed[k]["bias"])
#         for k in additional_layer_keys
#     ]
#     bias_layer1 = np.array(model.bias(model_params_seed))
#     # Build input arrays for layer 1 contributions
#     x_freq_a = jnp.array([[a, 0] for a in range(p)], dtype=jnp.int32)
#     x_freq_b = jnp.array([[0, b] for b in range(p)], dtype=jnp.int32)
#     # Compute contributions for layer 1
#     _, _, contrib_a, _ = model.apply({'params': model_params_seed}, x_freq_a, training=False)
#     _, _, _, contrib_b = model.apply({'params': model_params_seed}, x_freq_b, training=False)
#     contrib_a_np = np.array(contrib_a)
#     contrib_b_np = np.array(contrib_b)    
#     mlp_class_plots = f"{MLP_class}_seed_{seed}_bs={batch_size}_k={k}"
    
#     m = log_by_seed[seed][epochs]
#     model_accuracy = m["test_accuracy"]
#     test_total_loss = m["test_loss"]
#     test_ce_loss = m["test_ce_loss"]
    
#     # First Reconstruction: top_k=1
#     print("starting first reconstruction")
#     rec_metrics1, layer1_freq, neuron_data, dominant_freq_clusters, freq_json_dir1 = run_reconstruction(
#         model, model_params_seed, seed, top_k_val=1, p=p, batch_size=batch_size, k=k,
#         weight_decay=weight_decay, learning_rate=learning_rate,
#         mlp_class_lower=mlp_class_lower,
#         contrib_a_np=contrib_a_np, contrib_b_np=contrib_b_np, bias_layer1=bias1,
#         model_accuracy=model_accuracy, test_total_loss=test_total_loss, test_ce_loss=test_ce_loss
#     )
    
#     # --- Determine new top_k based on the layer 1 frequency distribution.
#     new_top_k = len(layer1_freq)
#     print(f"For seed {seed}, top_k=1 yielded {new_top_k} key frequencies.")
    
#     # Only perform the second reconstruction for the specific MLP classes.
#     # if base_class_name in ["no_embed", "no_embed_cheating"]:
#     #     print("starting second reconstruction for one hot model to do logn frequencies for top k")
#     #     rec_metrics2, _, _, _, freq_json_dir2 = run_reconstruction(
#     #         model, model_params_seed, seed, top_k_val=new_top_k, p=p, batch_size=batch_size, k=k,
#     #         weight_decay=weight_decay, learning_rate=learning_rate,
#     #         mlp_class_lower=mlp_class_lower,
#     #         contrib_a_np=contrib_a_np, contrib_b_np=contrib_b_np, bias_layer1=bias1,
#     #         model_accuracy=model_accuracy, test_total_loss=test_total_loss, test_ce_loss=test_ce_loss
#     #     )
#     #     # --- Log training distributions for second reconstruction ---
#     #     os.makedirs(freq_json_dir2, exist_ok=True)
#     #     training_distributions_name2 = f"training_distributions_top-k={new_top_k}_{p}_{neuron_data[1].__len__()}_{weight_decay}_{learning_rate}.txt"
#     #     training_distributions_path2 = os.path.join(freq_json_dir2, training_distributions_name2)
#     #     with open(training_distributions_path2, "a") as f:
#     #         line = f"{seed},<fitted_accuracy_sim_run2>,<fitted_accuracy_with_dead_run2>\n"
#     #         f.write(line)
#     # else:
#     #     print(f"Skipping second reconstruction for model class: {base_class_name}")
    
#     # Log training distributions for first reconstruction
#     os.makedirs(freq_json_dir1, exist_ok=True)
#     training_distributions_name1 = f"training_distributions_top-k=1_{p}_{neuron_data[1].__len__()}_{weight_decay}_{learning_rate}.txt"
#     training_distributions_path1 = os.path.join(freq_json_dir1, training_distributions_name1)
#     with open(training_distributions_path1, "a") as f:
#         fitted_accuracy_sim_run1 = rec_metrics1["stored_fits"]["cross_entropy_loss"]
#         fitted_accuracy_with_dead_run1 = rec_metrics1["model"]["cross_entropy_loss"]
#         line = f"{seed},{fitted_accuracy_sim_run1},{fitted_accuracy_with_dead_run1}\n"
#         f.write(line)

#     cluster_grouping = dominant_freq_clusters

#     scaling_dir = f"/home/mila/w/weis/scratch/neurips_2025_crt-appendix-run-7-s-1/qualitative_{p}_{mlp_class_lower}_{num_neurons}_features_{features}_k_{k}/scaling"
#     os.makedirs(scaling_dir, exist_ok=True)

#     # Build the filename:  last_<lr>_<wd>_<train_size>_<seed>.txt
#     count_file_path = os.path.join(
#         scaling_dir,
#         f"last_{learning_rate}_{weight_decay}_{training_set_size}_{seed}_{p}.txt"
#     )

#     # Write the count
#     with open(count_file_path, "w") as f:
#         f.write(f"{len(cluster_grouping[0])}\n")
#     print(f"Count of key freqs for seed {seed}: {len(cluster_grouping)} → {count_file_path}")

#     plot_dir = os.path.join(
#         BASE_DIR,
#         f"{p}_plots",
#         f"bs={batch_size}_nn={num_neurons}_wd={weight_decay}_epochs={epochs}_training_set_size={training_set_size}")
#     os.makedirs(plot_dir, exist_ok=True)
#     print("start plotting cluster preactivations")
#     plot_cluster_preactivations(
#         cluster_groupings=dominant_freq_clusters,
#         neuron_data=neuron_data,
#         mlp_class=mlp_class_plots,
#         seed=seed,
#         features=features,
#         num_neurons=num_neurons,
#         base_dir=plot_dir,
#     )


#     # # 2) summed preactivations  (now per‐layer)
#     # summed_preactivations(
#     #     cluster_groupings=dominant_freq_clusters,
#     #     neuron_data=neuron_data,
#     #     biases_by_layer=biases_by_layer,
#     #     mlp_class=mlp_class_plots,
#     #     seed=seed,
#     #     features=features,
#     #     num_neurons=num_neurons,
#     #     base_dir=plot_dir,
#     # )

#     # # 3) summed postactivations
#     # summed_postactivations(
#     #     cluster_groupings=dominant_freq_clusters,
#     #     neuron_data=neuron_data,
#     #     biases_by_layer=biases_by_layer,
#     #     mlp_class=mlp_class_plots,
#     #     seed=seed,
#     #     features=features,
#     #     num_neurons=num_neurons,
#     #     base_dir=plot_dir,
#     # )

#     # # 4) cluster‐to‐logits (only needs last‐layer bias)
#     final_layer_weights = np.array(model_params_seed["output_dense"]["kernel"])
#     # plot_cluster_to_logits(
#     #     cluster_groupings=dominant_freq_clusters,
#     #     neuron_data=neuron_data,
#     #     biases_last_layer=biases_by_layer[-1],
#     #     final_layer_weights=final_layer_weights,
#     #     mlp_class=mlp_class_plots,
#     #     seed=seed,
#     #     features=features,
#     #     num_neurons=num_neurons,
#     #     base_dir=plot_dir,
#     # )

#     # # 5) all‐clusters‐to‐logits (doesn’t need biases)
#     # plot_all_clusters_to_logits(
#     #     neuron_data=neuron_data,
#     #     final_layer_weights=final_layer_weights,
#     #     mlp_class=mlp_class_plots,
#     #     seed=seed,
#     #     features=features,
#     #     num_neurons=num_neurons,
#     #     base_dir=plot_dir,
#     # )

# #     put_neuron_preaccs_on_torus(
# #         cluster_groupings=dominant_freq_clusters,
# #         neuron_data=neuron_data,
# #         p=p,
# #         base_dir=BASE_DIR,
# #         layer_idx=1
# # )
#     import functools
#         # 2)  GPU-friendly cosine-similarity of the two gradients

#     # def batched_gradient_similarity(
#     #     *,
#     #     model,
#     #     params: dict,
#     #     a_batch: jnp.ndarray,          # shape (N,)  int32
#     #     b_batch: jnp.ndarray,          # shape (N,)  int32
#     #     c_batch: jnp.ndarray,          # shape (N,)  int32 (target classes)
#     # ) -> jnp.ndarray:
#     #     """
#     #     Returns an array of cosine similarities (length N).

#     #     A single jit-compiled call performs:
#     #         • first-layer input construction
#     #         • forward + backward through the full network
#     #         • cosine-similarity   sim(∂Q/∂E_a , ∂Q/∂E_b)
#     #     """
#     #     emb_a, emb_b = model.extract_embeddings_ab(params)
#     #     D_a, D_b = emb_a.shape[1], emb_b.shape[1]
#     #     in_features = params["dense_1"]["kernel"].shape[0]
#     #     concat_case = ("V_proj" in params      # Residual model → must concatenate
#     #                     or in_features == D_a + D_b)               # bool

#     #     @functools.partial(jax.jit, static_argnums=0)
#     #     def _run(concat_flag: bool,
#     #             emb_a, emb_b, a_idx, b_idx, c_idx):

#     #         # ----- build batch of first-layer inputs ----------------------
#     #         vec_a = emb_a[a_idx]                                 # (N,D_a)
#     #         vec_b = emb_b[b_idx]                                 # (N,D_b)
#     #         x0    = jnp.concatenate([vec_a, vec_b], axis=1) if concat_flag else vec_a + vec_b

#     #         # ----- grad of scalar logit wrt x0 ---------------------------
#     #         def scalar_logit(x_emb, cls):
#     #             return model.call_from_embedding(x_emb, params)[0][cls]

#     #         grad_fn = jax.grad(scalar_logit, argnums=0)          # ∂/∂x_emb

#     #         # vmap over the whole batch → grads shape (N, dim)
#     #         grads = jax.vmap(grad_fn)(x0, c_idx)

#     #         # ----- split into g_a , g_b ----------------------------------
#     #         if concat_flag:
#     #             g_a, g_b = grads[:, :D_a], grads[:, D_a:]
#     #         else:                                                # addition path
#     #             g_a = g_b = grads

#     #         # cosine similarity (avoid /0 with eps)
#     #         eps = 1e-12
#     #         norms = jnp.linalg.norm(g_a, axis=1) * jnp.linalg.norm(g_b, axis=1) + eps
#     #         cos_sim = jnp.sum(g_a * g_b, axis=1) / norms         # (N,)

#     #         return cos_sim

#     #     return _run(concat_case, emb_a, emb_b, a_batch, b_batch, c_batch)
    
#     def full_gradient_symmetricity_in_p_batches(model, params):
#         """
#         Enumerate all p^3 triples in p chunks of size p^2 each,
#         calling the already-JITted batched_gradient_similarity
#         on each, and concatenating the results.
#         """
#         # 1) rebuild the full mesh
#         p = params["dense_1"]["kernel"].shape[0]
#         A, B, C = jnp.meshgrid(
#             jnp.arange(p, dtype=jnp.int32),
#             jnp.arange(p, dtype=jnp.int32),
#             jnp.arange(p, dtype=jnp.int32),
#             indexing="ij",
#         )
#         a_flat = A.reshape(-1)
#         b_flat = B.reshape(-1)
#         c_flat = C.reshape(-1)

#         # 2) chunk size = p^2
#         m = p * p
#         sims_chunks = []
#         for i in range(p):
#             start = i * m
#             end   = start + m
#             sims_chunks.append(
#                 batched_gradient_similarity(
#                     model=model,
#                     params=params,
#                     a_batch=a_flat[start:end],
#                     b_batch=b_flat[start:end],
#                     c_batch=c_flat[start:end],
#                 )
#             )

#         # 3) stitch back to length p^3
#         return jnp.concatenate(sims_chunks, axis=0)
#         # result lives on-device; convert to np as needed outside

#     def distance_irrelevance_stats(L: np.ndarray) -> dict:
#         """
#         Column version  = authors’ script   (may exceed 1)
#         Diagonal version = literal paper    (≤ 1)
#         """
#         p = L.shape[0]
#         global_std = L.std() + 1e-12

#         # column-wise
#         col_stds = L.std(axis=0)
#         q_col    = col_stds / global_std

#         # full wrap-around diagonals
#         diag_stds = np.empty(p)
#         for d in range(p):
#             diag_vals = L[np.arange(p), (np.arange(p) + d) % p]
#             diag_stds[d] = diag_vals.std()
#         q_diag = diag_stds / global_std

#         return {
#             "avg_dist_irrel_col":  float(q_col.mean()),
#             "std_dist_irrel_col":  float(q_col.std()),
#             "avg_dist_irrel_diag": float(q_diag.mean()),
#             "std_dist_irrel_diag": float(q_diag.std()),
#         }
#     def batched_gradient_similarity(
#         *,
#         model,
#         params: dict,
#         a_batch: jnp.ndarray,          # shape (N,)  int32
#         b_batch: jnp.ndarray,          # shape (N,)  int32
#         c_batch: jnp.ndarray,          # shape (N,)  int32 (target classes)
#     ) -> jnp.ndarray:
#         """
#         Returns an array of cosine similarities (length N).

#         A single jit-compiled call performs:
#             • first-layer input construction
#             • forward + backward through the full network
#             • cosine-similarity   sim(∂Q/∂E_a , ∂Q/∂E_b)
#         """
#         emb_a, emb_b = model.extract_embeddings_ab(params)
#         D_a, D_b = emb_a.shape[1], emb_b.shape[1]
#         in_features = params["dense_1"]["kernel"].shape[0]
#         concat_case = ("V_proj" in params      # Residual model → must concatenate
#                         or in_features == D_a + D_b)               # bool

#         @functools.partial(jax.jit, static_argnums=0)
#         def _run(concat_flag: bool,
#                 emb_a, emb_b, a_idx, b_idx, c_idx):

#             # build batch of first-layer inputs
#             vec_a = emb_a[a_idx]                                 # (N,D_a)
#             vec_b = emb_b[b_idx]                                 # (N,D_b)
#             x0    = jnp.concatenate([vec_a, vec_b], axis=1) if concat_flag else vec_a + vec_b

#             # grad of scalar logit wrt x0
#             def scalar_logit(x_emb, cls):
#                 return model.call_from_embedding(x_emb, params)[0][cls]

#             grad_fn = jax.grad(scalar_logit, argnums=0) # ∂/∂x_emb

#             # vmap over the whole batch → grads shape (N, dim)
#             grads = jax.vmap(grad_fn)(x0, c_idx)

#             # split into g_a , g_b
#             if concat_flag:
#                 g_a, g_b = grads[:, :D_a], grads[:, D_a:]
#             else:
#                 g_a = g_b = grads

#             # cosine similarity (avoid /0 with eps)
#             eps = 1e-12
#             norms = jnp.linalg.norm(g_a, axis=1) * jnp.linalg.norm(g_b, axis=1) + eps
#             cos_sim = jnp.sum(g_a * g_b, axis=1) / norms         # (N,)

#             return cos_sim

#         return _run(concat_case, emb_a, emb_b, a_batch, b_batch, c_batch)
#         # result lives on-device; convert to np as needed outside

#     def distance_irrelevance_stats(L: np.ndarray) -> dict:
#         """
#         Column version  = authors’ script   (may exceed 1)
#         Diagonal version = literal paper    (≤ 1)
#         """
#         p = L.shape[0]
#         global_std = L.std() + 1e-12

#         # column-wise
#         col_stds = L.std(axis=0)
#         q_col    = col_stds / global_std

#         # full wrap-around diagonals
#         diag_stds = np.empty(p)
#         for d in range(p):
#             diag_vals = L[np.arange(p), (np.arange(p) + d) % p]
#             diag_stds[d] = diag_vals.std()
#         q_diag = diag_stds / global_std

#         return {
#             "avg_dist_irrel_col":  float(q_col.mean()),
#             "std_dist_irrel_col":  float(q_col.std()),
#             "avg_dist_irrel_diag": float(q_diag.mean()),
#             "std_dist_irrel_diag": float(q_diag.std()),
#         }

#     # uses the batched routine
#     def compute_useless_metrics(
#         *,
#         model,
#         params: dict,
#         p: int,
#         rng_seed: int = 42,
#         max_samples: int = 205379,       # <= p² keeps memory small
#     ) -> Tuple[Dict[str, float], Dict[str, float]]:
#         """
#         Returns
#         -------
#         grad_stats : {"average_gradient_symmetricity", "std_dev_gradient_symmetricity"}
#         dist_stats : {"average_distance_irrelevance",  "std_dev_distance_irrelevance"}
#         """
#         # gradient-symmetricity
#         rng = np.random.default_rng(rng_seed)
#         all_triples = [(a, b, rng.integers(p))   # random c per (a,b)
#                     for a in range(p) for b in range(p)]
#         if len(all_triples) > max_samples:
#             triples = rng.choice(all_triples, size=max_samples, replace=False)
#         else:
#             triples = all_triples

#         a_arr = jnp.array([t[0] for t in triples], dtype=jnp.int32)
#         b_arr = jnp.array([t[1] for t in triples], dtype=jnp.int32)
#         c_arr = jnp.array([t[2] for t in triples], dtype=jnp.int32)

#         cos_sims = batched_gradient_similarity(
#             model=model, params=params,
#             a_batch=a_arr, b_batch=b_arr, c_batch=c_arr
#         )
#         cos_np = np.asarray(cos_sims)

#         grad_stats = {
#             "average_gradient_symmetricity": float(cos_np.mean()),
#             "std_dev_gradient_symmetricity": float(cos_np.std()),
#         }

#         # distance-irrelevance
#         a_grid, b_grid = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
#         x_full = jnp.stack([a_grid.ravel(), b_grid.ravel()], axis=-1).astype(jnp.int32)

#         logits = model.apply({"params": params}, x_full, training=False)[0]  # (p² , p)
#         logits_np   = np.asarray(logits)
#         correct_idx = ((a_grid + b_grid) % p).ravel()
#         correct_logits = logits_np[np.arange(p * p), correct_idx]            # (p²,)

#         # arrange into L[i,j] with i = (a+b) mod p , j = (a−b) mod p
#         L = np.empty((p, p), dtype=float)
#         i_mat, j_mat = (a_grid + b_grid) % p, (a_grid - b_grid) % p
#         L[i_mat, j_mat] = correct_logits.reshape(p, p)

#         # identical to the TRANSFORMER implementation
#         col_stds   = L.std(axis=0)
#         global_std = L.std() + 1e-12        # avoid divide-by-zero

#         q_vals = col_stds / global_std
#         dist_stats = {
#             "average_distance_irrelevance": float(q_vals.mean()),
#             "std_dev_distance_irrelevance": float(q_vals.std()),
#         }

#         return grad_stats, dist_stats

#     def _mod_inverse(a: int, p: int) -> int:
#         """Modular inverse (p is prime)."""
#         return pow(a, p - 2, p)   # Fermat little theorem

#     def _layer_centres_of_mass(
#         preacts: jnp.ndarray,   # (p, p, N)
#         freqs:   np.ndarray,    # (N,)
#         p: int
#     ) -> np.ndarray:            # → (N, 2)  [CoM_a , CoM_b]
#         """GPU-optimised centre-of-mass in circular coordinates."""
#         # modular inverses on CPU
#         invs = np.array([_mod_inverse(int(f), p) for f in freqs], dtype=np.int32)

#         a_idx = jnp.arange(p, dtype=jnp.float32)           # (p,)
#         b_idx = jnp.arange(p, dtype=jnp.float32)           # (p,)

#         @jax.jit
#         def _com(act_3d, invs_1d):
#             invs_b = invs_1d.astype(jnp.float32)[None, None, :]    # (1,1,N)

#             # linear indices → angles; then straighten by invs_b
#             ang_a = (2 * jnp.pi * a_idx[:, None, None] / p) * invs_b   # (p,1,N)
#             ang_b = (2 * jnp.pi * b_idx[None, :, None] / p) * invs_b   # (1,p,N)

#             # use absolute activation as weight  (keeps both peaks)
#             w   = jnp.abs(act_3d)                          # (p,p,N)  ≥0
#             vec_a = jnp.sum(w * jnp.exp(1j * ang_a), axis=(0, 1))  # (N,)
#             vec_b = jnp.sum(w * jnp.exp(1j * ang_b), axis=(0, 1))  # (N,)

#             # circular mean → angle in [0, 2π)
#             ang_com_a = (jnp.angle(vec_a) + 2 * jnp.pi) % (2 * jnp.pi)
#             ang_com_b = (jnp.angle(vec_b) + 2 * jnp.pi) % (2 * jnp.pi)

#             com_a = ang_com_a / (2 * jnp.pi) * p           # back to 0…p
#             com_b = ang_com_b / (2 * jnp.pi) * p
#             return jnp.stack([com_a, com_b], axis=1)       # (N,2)

#         return np.asarray(_com(preacts, invs))   

#     def _mod_inverse(a: int, p: int) -> int:
#         """Modular inverse (p is prime)."""
#         return pow(a, p - 2, p)   # Fermat little theorem


#     def _layer_centres_of_mass(
#         preacts: jnp.ndarray,   # (p, p, N)
#         freqs:   np.ndarray,    # (N,)
#         p: int
#     ) -> np.ndarray:            # → (N, 2)  [CoM_a , CoM_b]
#         """GPU-optimised centre-of-mass in circular coordinates."""
#         # --- modular inverses on CPU ------------------------------------
#         invs = np.array([_mod_inverse(int(f), p) for f in freqs], dtype=np.int32)

#         a_idx = jnp.arange(p, dtype=jnp.float32)           # (p,)
#         b_idx = jnp.arange(p, dtype=jnp.float32)           # (p,)

#         @jax.jit
#         def _com(act_3d, invs_1d):
#             invs_b = invs_1d.astype(jnp.float32)[None, None, :]    # (1,1,N)

#             # linear indices → angles; then straighten by invs_b
#             ang_a = (2 * jnp.pi * a_idx[:, None, None] / p) * invs_b   # (p,1,N)
#             ang_b = (2 * jnp.pi * b_idx[None, :, None] / p) * invs_b   # (1,p,N)

#             # use absolute activation as weight  (keeps both peaks)
#             w   = jnp.abs(act_3d)                          # (p,p,N)  ≥0
#             vec_a = jnp.sum(w * jnp.exp(1j * ang_a), axis=(0, 1))  # (N,)
#             vec_b = jnp.sum(w * jnp.exp(1j * ang_b), axis=(0, 1))  # (N,)

#             # circular mean → angle in [0, 2π)
#             ang_com_a = (jnp.angle(vec_a) + 2 * jnp.pi) % (2 * jnp.pi)
#             ang_com_b = (jnp.angle(vec_b) + 2 * jnp.pi) % (2 * jnp.pi)

#             com_a = ang_com_a / (2 * jnp.pi) * p           # back to 0…p
#             com_b = ang_com_b / (2 * jnp.pi) * p
#             return jnp.stack([com_a, com_b], axis=1)       # (N,2)

#         return np.asarray(_com(preacts, invs))   
#         # ─────────────────────────────────────────────────────────────

#     def _phase_distribution(preacts: jnp.ndarray, threshold: float, p: int) -> Counter:
#         # as in train_transformer
#         p_float = float(p)
#         # keep only “strong” neurons
#         max_per_neuron = jnp.max(preacts, axis=(0,1))
#         strong = max_per_neuron > threshold
#         if not bool(jnp.any(strong)):
#             return Counter()
#         pr = jnp.compress(strong, preacts, axis=2)  # (p,p,N')
#         # mean over columns/rows
#         row_mean = jnp.mean(pr, axis=1)
#         col_mean = jnp.mean(pr, axis=0)
#         # FFT
#         fft_row = jnp.fft.fft(row_mean, axis=0)
#         fft_col = jnp.fft.fft(col_mean, axis=0)
#         # power spectrum, ignore DC
#         pow_r = jnp.abs(fft_row)**2; pow_r = pow_r.at[0].set(0)
#         pow_c = jnp.abs(fft_col)**2; pow_c = pow_c.at[0].set(0)
#         # dominant freqs
#         fa = jnp.argmax(pow_r[1:p//2+1], axis=0) + 1
#         fb = jnp.argmax(pow_c[1:p//2+1], axis=0) + 1
#         # recover phases
#         coeff_r = jnp.take_along_axis(fft_row, fa[None], axis=0).squeeze(0)
#         coeff_c = jnp.take_along_axis(fft_col, fb[None], axis=0).squeeze(0)
#         phi_a = (-jnp.angle(coeff_r) * p_float) / (2*jnp.pi*fa.astype(jnp.float32))
#         phi_b = (-jnp.angle(coeff_c) * p_float) / (2*jnp.pi*fb.astype(jnp.float32))
#         phi_a_i = jnp.mod(jnp.rint(phi_a), p).astype(jnp.int32)
#         phi_b_i = jnp.mod(jnp.rint(phi_b), p).astype(jnp.int32)
#         ctr = Counter()
#         for a,b in zip(phi_a_i.tolist(), phi_b_i.tolist()):
#             ctr[f"{a},{b}"] += 1
#         return ctr
    
#     def _phase_distribution_equal_freq(
#             preacts: jnp.ndarray,          # (p , p , N)
#             threshold: float,
#             p: int
#         ) -> tuple[Counter, Counter, Counter, Counter, Counter]:
#             """
#             Two equal-frequency fits (first pass + residual pass).
#             Returns
#             -------
#             ctr_first     : Counter  of phases from the first pass
#             ctr_second    : Counter  of phases from the residual pass
#             freq_pairs_ctr: Counter  mapping "f1,f2" -> count   (one entry per neuron)
#             """
#             p_float = float(p)
#             fft_lim = p // 2 + 1

#             strong_mask = jnp.max(preacts, axis=(0, 1)) > threshold
#             if not bool(jnp.any(strong_mask)):
#                 return Counter(), Counter(), Counter(), Counter(), Counter()

#             pre_str = jnp.compress(strong_mask, preacts, axis=2)        # (p,p,N’)
#             N_str   = pre_str.shape[-1]

#             # ── helper: ONE equal-freq fit ───────────────────────────────────────
#             def _single_equal_freq_fit(tensor, avoid_f=None):
#                 row_m = jnp.mean(tensor, axis=1)
#                 col_m = jnp.mean(tensor, axis=0)

#                 fft_r = jnp.fft.fft(row_m, axis=0)
#                 fft_c = jnp.fft.fft(col_m, axis=0)
#                 pow_r = jnp.abs(fft_r) ** 2
#                 pow_c = jnp.abs(fft_c) ** 2

#                 row_p = pow_r[1:fft_lim, :]
#                 col_p = pow_c[1:fft_lim, :]

#                 if avoid_f is not None and avoid_f.size:
#                     # avoid_f has shape (k, N') where k=1,2,3 for 3 passes
#                     banned = (avoid_f - 1)[None, ...]                     # → (1,k,N')
#                     rows   = jnp.arange(row_p.shape[0])[:, None, None]   # → (n_rows,1,1)
#                     mask   = jnp.any(rows == banned, axis=1)             # → (n_rows, N')
#                     row_p  = jnp.where(mask, -1.0, row_p)
#                     col_p  = jnp.where(mask, -1.0, col_p)

#                 f_sel = jnp.argmax(row_p + col_p, axis=0) + 1           # (N’,)

#                 coeff_r = jnp.take_along_axis(fft_r, f_sel[None, :], axis=0).squeeze(0)
#                 coeff_c = jnp.take_along_axis(fft_c, f_sel[None, :], axis=0).squeeze(0)

#                 phi_a = (-jnp.angle(coeff_r) * p_float) / (2 * jnp.pi * f_sel.astype(jnp.float32))
#                 phi_b = (-jnp.angle(coeff_c) * p_float) / (2 * jnp.pi * f_sel.astype(jnp.float32))

#                 phi_a_i = jnp.mod(jnp.rint(phi_a), p).astype(jnp.int32)
#                 phi_b_i = jnp.mod(jnp.rint(phi_b), p).astype(jnp.int32)

#                 ctr = Counter()
#                 for a, b in np.asarray(jnp.stack([phi_a_i, phi_b_i], axis=1)):
#                     ctr[f"{int(a)},{int(b)}"] += 1

#                 # build reconstruction for residual
#                 a_lin = jnp.arange(p)[:, None, None]
#                 b_lin = jnp.arange(p)[None, :, None]
#                 two_pi_over_p = 2 * jnp.pi / p
#                 recon = (jnp.sin(two_pi_over_p * f_sel * a_lin + two_pi_over_p * phi_a_i)
#                     + jnp.sin(two_pi_over_p * f_sel * b_lin + two_pi_over_p * phi_b_i))

#                 return ctr, f_sel, recon

#             def build_freq_counter(*freq_arrays: jnp.ndarray) -> Counter[str]:
#                 """
#                 Count how often each tuple of frequencies occurs.
#                 E.g. build_freq_counter(f1, f2)  → Counter of "f1,f2"
#                     build_freq_counter(f1, f2, f3) → Counter of "f1,f2,f3"
#                 """
#                 ctr = Counter()
#                 # turn them into plain Python lists of ints
#                 lists = [np.asarray(arr).reshape(-1).tolist() for arr in freq_arrays]
#                 for freqs in zip(*lists):
#                     key = ",".join(str(int(f)) for f in freqs)
#                     ctr[key] += 1
#                 return ctr

#             # first fit
#             ctr_first, f1, recon1 = _single_equal_freq_fit(pre_str)

#             # second fit on residual
#             residual1 = pre_str - recon1
#             ctr_second, f2, recon2 = _single_equal_freq_fit(residual1, avoid_f=f1)

#             # third fit
#             residual2 = residual1 - recon2
#             avoid_both = jnp.stack([f1, f2], axis=0)
#             ctr_third, f3, _ = _single_equal_freq_fit(residual2, avoid_f=avoid_both)


#             # frequency­pair counter
#             freq_pairs_ctr    = build_freq_counter(f1, f2)
#             # frequency­triplet counter
#             freq_triplets_ctr = build_freq_counter(f1, f2, f3)

#             return ctr_first, ctr_second, ctr_third, freq_pairs_ctr, freq_triplets_ctr

#     def compute_center_mass_distribution(
#         *,
#         neuron_data: Dict[int, Dict[int, Dict[str, Any]]],
#         dominant_freq_clusters,
#         p: int,
#     ) -> Dict[str, int]:
#         """
#         Builds the `distribution_of_center_mass` counter across *all* layers.
#         Keys are `"a,b"` strings with integer-rounded CoM coordinates.
#         """
#         counter = Counter()

#         # iterate layer-wise
#         for layer_idx, layer_dict in neuron_data.items():
#             # assemble (p,p,N) tensor and parallel freq list
#             neuron_ids      = sorted(layer_dict)
#             if not neuron_ids:
#                 continue
#             pre_list        = [layer_dict[n]["real_preactivations"] for n in neuron_ids]
#             pre_layer       = np.stack(pre_list, axis=-1)           # (p,p,N)

#             freqs = np.ones(len(neuron_ids), dtype=np.int32)

#             # GPU pass
#             coms = _layer_centres_of_mass(jnp.asarray(pre_layer), freqs, p)  # (N,2)

#             # round to nearest integer grid point
#             com_int = np.rint(coms).astype(int) % p                 # wrap to 0..p-1
#             for a, b in com_int:
#                 counter[f"{a},{b}"] += 1

#         return dict(counter)

#     def compute_and_track_quantities(
#         *,
#         seed: int,
#         p: int,
#         model,                        # trained DonutMLP
#         params: dict,                 # parameters for this seed
#         neuron_data: Dict[int, Dict[int, Dict[str, Any]]],
#         cluster_groupings: Union[Dict[int, list], list],
#         final_layer_weights: np.ndarray,     # shape (num_neurons_last, p)
#         save_dir: str = ".",
#     ) -> None:
#         """
#         Writes *quantities_{seed}.json* containing:

#         distribution_of_max_preactivations
#         networks_equivariantness_stats      (correct-logit stats)
#         network_margin_stats                (margin  stats)
#         network_loss_stats                  (per-sample CE-loss stats)   ← NEW
#         clusters_equivariantness_stats      (per-cluster correct-logit stats)
#         clusters_margin_stats               (per-cluster margin stats)
#         """

#         # where does each neuron reach its maximum? 
#         dist_counter: collections.Counter[str] = collections.Counter()
#         for layer_dict in neuron_data.values():
#             for nd in layer_dict.values():
#                 real = np.asarray(nd.get("real_preactivations", []))
#                 if real.size:
#                     a_idx, b_idx = np.unravel_index(real.argmax(), real.shape)
#                     dist_counter[f"{a_idx},{b_idx}"] += 1
#         distribution_of_max_preactivations = dict(dist_counter)

#         # run the whole network on the complete p² grid
#         a_grid, b_grid = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
#         x_full = np.stack([a_grid.ravel(), b_grid.ravel()], axis=-1).astype(jnp.int32)

#         logits = model.apply({"params": params}, x_full, training=False)[0]        # (p², p)
#         logits_np = np.asarray(logits)
#         correct_idx = ((a_grid + b_grid) % p).ravel()

#         correct_logits = logits_np[np.arange(p * p), correct_idx]                 # (p²,)

#         # margins
#         tmp = logits_np.copy()
#         tmp[np.arange(p * p), correct_idx] = -np.inf
#         second_logits = tmp.max(axis=1)
#         margins = correct_logits - second_logits

#         # per-sample CE loss (log-softmax trick, row-wise)
#         row_max = logits_np.max(axis=1, keepdims=True)
#         logsumexp = row_max + np.log(np.exp(logits_np - row_max).sum(axis=1, keepdims=True))
#         ce_losses = (logsumexp.squeeze() - correct_logits)                         # (p²,)

#         networks_equivariantness_stats = {
#             "min":  float(correct_logits.min()),
#             "max":  float(correct_logits.max()),
#             "mean": float(correct_logits.mean()),
#             "std":  float(correct_logits.std()),
#         }
#         network_margin_stats = {
#             "avg_margin":     float(margins.mean()),
#             "min_margin":     float(margins.min()),
#             "max_margin":     float(margins.max()),
#             "std_dev_margin": float(margins.std()),
#         }
#         network_loss_stats = {
#             "avg_loss":  float(ce_losses.mean()),
#             "min_loss":  float(ce_losses.min()),
#             "max_loss":  float(ce_losses.max()),
#             "std_dev_loss": float(ce_losses.std()),
#         }

#         # stats for frequency-clusters in last hidden layer
#         if isinstance(cluster_groupings, collections.abc.Mapping):
#             last_clusters = cluster_groupings            
#             last_layer_idx = max(neuron_data)
#         else:
#             last_clusters = cluster_groupings[-1]
#             last_layer_idx = len(cluster_groupings)

#         layer_nd = neuron_data[last_layer_idx]
#         correct_idx_grid = (a_grid + b_grid) % p # p×p

#         clusters_equivariantness_stats = {}
#         clusters_margin_stats = {}

#         for freq, neuron_ids in last_clusters.items():
#             if not neuron_ids:
#                 continue
#             # build cluster logits: (p, p, p)
#             cluster_logits = np.zeros((p, p, p), dtype=float)
#             for n in neuron_ids:
#                 nd = layer_nd.get(n)
#                 if nd is None:
#                     continue
#                 post = np.asarray(
#                     nd.get("postactivations",
#                         np.maximum(nd["real_preactivations"], 0.0))
#                 )                                           # p×p
#                 w_row = final_layer_weights[n]              # p,
#                 cluster_logits += post[..., None] * w_row

#             # correct-logit stats
#             corr = cluster_logits[np.arange(p)[:, None],
#                                 np.arange(p)[None, :],
#                                 correct_idx_grid]
#             corr_flat = corr.ravel()
#             clusters_equivariantness_stats[str(freq)] = {
#                 "min":  float(corr_flat.min()),
#                 "max":  float(corr_flat.max()),
#                 "mean": float(corr_flat.mean()),
#                 "std":  float(corr_flat.std()),
#             }

#             # margin stats (for the cluster contribution alone)
#             logits_flat = cluster_logits.reshape(p * p, p)
#             tmp = logits_flat.copy()
#             tmp[np.arange(p * p), correct_idx] = -np.inf
#             second = tmp.max(axis=1)
#             cluster_margins = corr_flat - second
#             clusters_margin_stats[str(freq)] = {
#                 "avg_margin":     float(cluster_margins.mean()),
#                 "min_margin":     float(cluster_margins.min()),
#                 "max_margin":     float(cluster_margins.max()),
#                 "std_dev_margin": float(cluster_margins.std()),
#             }

#         # write to json
#         out = {
#             "distribution_of_max_preactivations": distribution_of_max_preactivations,
#             "networks_equivariantness_stats":     networks_equivariantness_stats,
#             "network_margin_stats":               network_margin_stats,
#             "network_loss_stats":                 network_loss_stats,   # ← NEW
#             "clusters_equivariantness_stats":     clusters_equivariantness_stats,
#             "clusters_margin_stats":              clusters_margin_stats,
#         }

#         grad_stats, dist_stats = compute_useless_metrics(
#             model=model,
#             params=params,
#             p=p,                    # 59
#             rng_seed=42,
#             max_samples=p*p         # use the full 59² = 3 481 triples
#         )
#         out.update(grad_stats)
#         out.update(dist_stats)

#         distribution_of_center_mass = compute_center_mass_distribution(
#             neuron_data=neuron_data,
#             dominant_freq_clusters=cluster_groupings,
#             p=p,
#         )

#         out["distribution_of_center_mass"] = distribution_of_center_mass

#         # phase / freq–equality histograms
#         phases_free             = Counter()
#         phases_equal_first      = Counter()
#         phases_equal_second_fit = Counter()
#         phases_equal_third_fit  = Counter()
#         freq_pairs_total        = Counter()
#         freq_triplets_total     = Counter()

#         for layer_dict in neuron_data.values():
#             if not layer_dict:
#                 continue
#             # build (p,p,N) array of real pre–activations
#             pre_layer = jnp.stack(
#                 [ layer_dict[n]["real_preactivations"] 
#                 for n in sorted(layer_dict) ], 
#                 axis=-1
#             )  # shape (p, p, N)
#             # free‐freq phase distribution
#             phases_free              += _phase_distribution(pre_layer, 0.01, p)
#             # equal‐freq two‐pass distribution
#             ctr1, ctr2, ctr3, ctr_pairs, ctr_triplets = _phase_distribution_equal_freq(pre_layer, 0.01, p)
#             phases_equal_first       += ctr1
#             phases_equal_second_fit  += ctr2
#             phases_equal_third_fit   += ctr3
#             freq_pairs_total         += ctr_pairs
#             freq_triplets_total      += ctr_triplets

#         out["distribution_of_phases"]                    = dict(phases_free)
#         out["distribution_of_phases_f_a=f_b"]            = dict(phases_equal_first)
#         out["distribution_of_phases_f_a=f_b_second_fit"] = dict(phases_equal_second_fit)
#         out["distribution_of_phases_f_a=f_b_third_fit"]  = dict(phases_equal_third_fit)
#         out["frequencies_equal"]                         = dict(freq_pairs_total)
#         out["frequencies_equal_triplets"]               = dict(freq_triplets_total)
        

#         os.makedirs(save_dir, exist_ok=True)
#         path = os.path.join(save_dir, f"quantities_{seed}.json")
#         with open(path, "w") as f:
#             json.dump(out, f, indent=2)

#         print(f"[compute_and_track_quantities] wrote {path}")

#     equivariantness_dir = os.path.join(
#         BASE_DIR,
#         f"{p}_distributions_equivariantness",
#         f"mlp={mlp_class_lower}_p={p}_bs={batch_size}_k={k}_nn={num_neurons}_wd={weight_decay}_lr={learning_rate}"
#     )
#     os.makedirs(equivariantness_dir, exist_ok=True)

#     # compute_and_track_quantities(
#     #     seed=seed,
#     #     p=p,
#     #     model=model,
#     #     params=model_params_seed,
#     #     neuron_data=neuron_data,
#     #     cluster_groupings=dominant_freq_clusters,
#     #     final_layer_weights=final_layer_weights,
#     #     save_dir=equivariantness_dir
#     # )
####################
    preacts, X_emb, input_weights, cluster_contribs_to_logits, cluster_weights_to_logits = get_all_preacts_and_embeddings(
        model=model,
        params=model_params_seed,
        p=group_size,
        clusters_by_layer=layers_freq,
    )

    pdf_root = os.path.join(graph_dir, "pdf_plots", f"seed_{seed}")
    os.makedirs(pdf_root, exist_ok=True)

    json_root = make_some_jsons(
        preacts=preacts,
        p=group_size,
        clusters_by_layer=layers_freq,                  # == dominant_freq_clusters
        cluster_weights_to_logits=cluster_weights_to_logits,  # dict[freq] -> (|cluster|, p)
        cluster_contribs_to_logits=cluster_contribs_to_logits,# optional correctness check
        save_dir=pdf_root,
        sanity_check=True,
    )
#####################
    # clusters_by_layer = layers_freq # list[dict]

    # # Iterate over hidden layers
    # for layer_idx, (H_full, W_full) in enumerate(zip(preacts, input_weights), start=1):
    #     clusters = clusters_by_layer[layer_idx - 1] if layer_idx - 1 < len(clusters_by_layer) else {}
    #     layer_freqs = sorted(clusters.keys())

    #     generate_pdf_plots_for_matrix(
    #         H_full,
    #         p,
    #         save_dir=pdf_root,
    #         seed=seed,
    #         freq_list=layer_freqs,
    #         tag=f"layer{layer_idx}_preacts",
    #         class_string=mlp_class_lower,
    #         num_principal_components=num_principal_components,
    #     )

    #     generate_pdf_plots_for_matrix(
    #         W_full,
    #         p,
    #         save_dir=pdf_root,
    #         seed=seed,
    #         freq_list=layer_freqs,
    #         tag=f"layer{layer_idx}_weights",
    #         class_string=mlp_class_lower,
    #         num_principal_components=num_principal_components,
    #     )

    #     # Per-frequency cluster plots
    #     for freq, neuron_ids in clusters.items():
    #         if len(neuron_ids) < 4:
    #             continue

    #         H_cluster = H_full[:, neuron_ids]              # (p², |cluster|)

    #         generate_pdf_plots_for_matrix(
    #             H_cluster,
    #             p,
    #             save_dir=pdf_root,
    #             seed=seed,
    #             freq_list=[freq],
    #             tag=f"layer{layer_idx}_freq={freq}",
    #             class_string=mlp_class_lower,
    #             num_principal_components=num_principal_components,
    #         )

    #         W_cluster = W_full[:, neuron_ids]              # (in_dim, |cluster|)
    #         if W_cluster.size == 0:
    #             continue

    #         generate_pdf_plots_for_matrix(
    #             W_cluster,
    #             p,
    #             save_dir=pdf_root,
    #             seed=seed,
    #             freq_list=[freq],
    #             tag=f"layer{layer_idx}_freq={freq}_weights",
    #             class_string=mlp_class_lower,
    #             num_principal_components=num_principal_components,
    #         )
############
    # clusters to logits
    for freq, C_freq in cluster_contribs_to_logits.items():
        # C_freq is (p², p): the total contribution of cluster “freq” to each logit
        generate_pdf_plots_for_matrix(
            C_freq, p, save_dir=pdf_root, seed=seed,
            freq_list=[freq],
            tag=f"cluster_contributions_to_logits_freq={freq}",
            tag_q = "full",
            colour_rule = colour_quad_mod_g,
            class_string=mlp_class_lower,
            num_principal_components=num_principal_components,
        )
    # #have some bugs here
    # for freq, W_block in cluster_weights_to_logits.items():
    #     # W_block has shape (|cluster|, p)
    #     generate_pdf_plots_for_matrix(
    #         W_block, p, save_dir=pdf_root, seed=seed,
    #         freq_list=[freq],
    #         tag=f"cluster_weights_to_logits_freq={freq}",
    #         tag_q = "full",
    #         colour_rule = colour_quad_mod_g,
    #         class_string=mlp_class_lower,
    #         num_principal_components=num_principal_components,
    #     )
################
#     # Embeddings
#     generate_pdf_plots_for_matrix(
#         X_emb,
#         p,
#         save_dir=pdf_root,
#         seed=seed,
#         freq_list=sorted(seed_dict_freqs_list.get(seed, [])),
#         tag="embeds",
#         class_string=mlp_class_lower,
#         num_principal_components=num_principal_components,
#     )

#     print(f"✓ PDF plots written for seed {seed} → {pdf_root}")