# controllers/training_prep_Transformer.py

from typing import Dict, Any, Tuple, List
from functools import partial

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
        logits=logits_last_token(y_pred_3d),
        labels=y
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
    # outputs may be (y_pred, aux, l2_loss). We only need y_pred.
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]

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
# def make_train_batches(p: int, batch_size: int, k: int, random_seed_ints: List[int]):
#     train_ds_list = []
#     for seed in random_seed_ints:
#         x, y = dihedral.make_dihedral_dataset(p, batch_size, k, seed)
#         train_ds_list.append((x, y))
#     x_batches = jnp.stack([x for (x, _) in train_ds_list])
#     y_batches = jnp.stack([y for (_, y) in train_ds_list])
#     print("x_batches.shape =", x_batches.shape, "y_batches.shape =", y_batches.shape)
#     print(f"Number of training batches: {x_batches.shape[1]}")
#     dataset_size_mb = (x_batches.size * x_batches.dtype.itemsize) / (1024 ** 2)
#     print(f"Dataset size per model: {dataset_size_mb:.2f} MB")
#     return train_ds_list, x_batches, y_batches
#
# def shuffle_batches_for_epoch(
#     x_batches: jnp.ndarray,  # (M, K, B, 2)
#     y_batches: jnp.ndarray,  # (M, K, B)
#     epoch: int,
#     seeds: List[int],
#     shuffle_within_batch: bool = True,
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     M, K, B = x_batches.shape[0], x_batches.shape[1], x_batches.shape[2]
#
#     # 1) shuffle batch order (axis=1)
#     perms_k = jnp.stack(
#         [jax.random.permutation(jax.random.fold_in(jax.random.PRNGKey(s), epoch), K)
#          for s in seeds],
#         axis=0
#     )  # (M, K)
#
#     # use take_along_axis to reorder on axis=1
#     gather_idx_k_x = jnp.broadcast_to(perms_k[:, :, None, None], (M, K, B, 1))
#     gather_idx_k_y = jnp.broadcast_to(perms_k[:, :, None],       (M, K, B))
#     x_shuf = jnp.take_along_axis(x_batches, gather_idx_k_x, axis=1)
#     y_shuf = jnp.take_along_axis(y_batches, gather_idx_k_y, axis=1)
#
#     if not shuffle_within_batch:
#         return x_shuf, y_shuf
#
#     # 2) shuffle samples within each batch (axis=2)
#     perms_b = jnp.stack(
#         [jax.random.permutation(jax.random.fold_in(jax.random.PRNGKey(s ^ 0xBEEF), epoch), B)
#          for s in seeds],
#         axis=0
#     )  # (M, B)
#
#     gather_idx_b_x = jnp.broadcast_to(perms_b[:, None, :, None], (M, K, B, 1))
#     gather_idx_b_y = jnp.broadcast_to(perms_b[:, None, :],       (M, K, B))
#     x_shuf = jnp.take_along_axis(x_shuf, gather_idx_b_x, axis=2)
#     y_shuf = jnp.take_along_axis(y_shuf, gather_idx_b_y, axis=2)
#
#     return x_shuf, y_shuf

def make_full_eval_grid(p: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    G, _ = DFT.make_irreps_Dn(p)
    idx = {g: i for i, g in enumerate(G)}
    group_size = len(G)
    x_eval = jnp.array([[idx[g], idx[h]] for g in G for h in G], dtype=jnp.int32)
    y_eval = jnp.array([idx[dihedral.mult(g, h, p)] for g in G for h in G], dtype=jnp.int32)
    return x_eval, y_eval

# def pad_to_batches(x_eval, y_eval, batch_size: int, num_models: int):
#     x_exp = jnp.tile(x_eval[None, :, :], (num_models, 1, 1))
#     y_exp = jnp.tile(y_eval[None, :], (num_models, 1))
#     total, bs = x_eval.shape[0], batch_size
#     n_full, remain = total // bs, total % bs
#     if remain > 0:
#         pad = bs - remain
#         x_pad = x_exp[:, :pad, :]
#         y_pad = y_exp[:, :pad]
#         x_eval_padded = jnp.concatenate([x_exp, x_pad], axis=1)
#         y_eval_padded = jnp.concatenate([y_exp, y_pad], axis=1)
#         n_batches = n_full + 1
#     else:
#         x_eval_padded, y_eval_padded, n_batches = x_exp, y_exp, n_full
#     x_batches = x_eval_padded.reshape(num_models, n_batches, bs, 2)
#     y_batches = y_eval_padded.reshape(num_models, n_batches, bs)
#     return x_batches, y_batches


def make_train_and_test_batches(
    p: int,
    batch_size: int,
    k: int,
    random_seed_ints: List[int],
    *,
    test_batch_size: int | None = None,
    shuffle_test: bool = True,
    drop_remainder: bool = True,
) -> Tuple[
    List[Tuple[jnp.ndarray, jnp.ndarray]],
    jnp.ndarray, jnp.ndarray,       # x_train_batches, y_train_batches
    jnp.ndarray, jnp.ndarray        # x_test_batches,  y_test_batches
]:
    """
    Builds per-seed train batches (as before) and test batches (complement of train).
    Shapes:
      x_train_batches: (M, k, batch_size, 2)
      y_train_batches: (M, k, batch_size)
      x_test_batches:  (M, K_test, B_test, 2)
      y_test_batches:  (M, K_test, B_test)
    """
    train_list: List[Tuple[jnp.ndarray, jnp.ndarray]] = []
    x_train_stack = []
    y_train_stack = []
    x_test_stack = []
    y_test_stack = []

    for seed in random_seed_ints:
        x_tr, y_tr, x_te, y_te = dihedral.make_dihedral_dataset_with_test(
            p, batch_size, k, seed,
            test_batch_size=test_batch_size,
            shuffle_test=shuffle_test,
            drop_remainder=drop_remainder,
        )
        train_list.append((x_tr, y_tr))
        x_train_stack.append(x_tr)
        y_train_stack.append(y_tr)
        x_test_stack.append(x_te)
        y_test_stack.append(y_te)

    x_train_batches = jnp.stack(x_train_stack, axis=0)
    y_train_batches = jnp.stack(y_train_stack, axis=0)

    # Sanity: ensure all seeds produced the same K_test and B_test
    K_tests = [arr.shape[0] for arr in x_test_stack]
    B_tests = [arr.shape[1] for arr in x_test_stack]
    if len(set(K_tests)) != 1 or len(set(B_tests)) != 1:
        raise ValueError(f"Test batch shapes differ across seeds: K={K_tests}, B={B_tests}")

    x_test_batches = jnp.stack(x_test_stack, axis=0)
    y_test_batches = jnp.stack(y_test_stack, axis=0)

    print("x_train_batches.shape =", x_train_batches.shape)
    print("y_train_batches.shape =", y_train_batches.shape)
    print("x_test_batches.shape  =", x_test_batches.shape)
    print("y_test_batches.shape  =", y_test_batches.shape)
    print(f"Number of train batches per model: {x_train_batches.shape[1]}")
    print(f"Number of test  batches per model: {x_test_batches.shape[1]}")

    # Size info (train only, since test size depends on complement)
    dataset_size_bytes = (x_train_batches.size * x_train_batches.dtype.itemsize)
    dataset_size_mb = dataset_size_bytes / (1024 ** 2)
    print(f"Train dataset size per model: {dataset_size_mb:.2f} MB")

    return train_list, x_train_batches, y_train_batches, x_test_batches, y_test_batches


@partial(jax.jit, static_argnames=('shuffle_within_batch', 'debug', 'samples_to_check'))
def shuffle_batches_for_epoch(
    x_batches: jnp.ndarray,   # (M, K, B, 2)
    y_batches: jnp.ndarray,   # (M, K, B)
    epoch: int,
    seeds: jnp.ndarray,       # (M,), dtype=uint32/int32
    shuffle_within_batch: bool = True,
    debug: bool = False,
    samples_to_check: int = 5,   # debug 时每个维度抽多少个 index 做对齐校验
):
    M, K, B = x_batches.shape[0], x_batches.shape[1], x_batches.shape[2]

    # --- 1) 打乱 batch 顺序 (axis=1)
    keys_k  = jax.vmap(lambda s: jax.random.fold_in(jax.random.PRNGKey(s), epoch))(seeds)
    perms_k = jax.vmap(lambda k: jax.random.permutation(k, K))(keys_k)   # (M, K)

    gather_k_x = jnp.broadcast_to(perms_k[:, :, None, None], (M, K, B, 1))
    gather_k_y = jnp.broadcast_to(perms_k[:, :, None],       (M, K, B))
    x_shuf = jnp.take_along_axis(x_batches, gather_k_x, axis=1)
    y_shuf = jnp.take_along_axis(y_batches, gather_k_y, axis=1)

    if not shuffle_within_batch:
        # 可选：在 debug 下验证第一步对齐
        if debug:
            _debug_check_alignment(x_batches, y_batches, x_shuf, y_shuf,
                                   perms_k=perms_k, perms_b=None,
                                   samples_to_check=samples_to_check)
        return x_shuf, y_shuf

    # --- 2) 打乱每个 batch 内样本 (axis=2)
    # 注意 seeds ^ 0xBEEF 时保持无符号类型，避免溢出歧义
    seeds_b = jnp.bitwise_xor(seeds.astype(jnp.uint32), jnp.uint32(0xBEEF))
    keys_b  = jax.vmap(lambda s: jax.random.fold_in(jax.random.PRNGKey(s), epoch))(seeds_b)
    perms_b = jax.vmap(lambda k: jax.random.permutation(k, B))(keys_b)   # (M, B)

    gather_b_x = jnp.broadcast_to(perms_b[:, None, :, None], (M, K, B, 1))
    gather_b_y = jnp.broadcast_to(perms_b[:, None, :],       (M, K, B))
    x_shuf = jnp.take_along_axis(x_shuf, gather_b_x, axis=2)
    y_shuf = jnp.take_along_axis(y_shuf, gather_b_y, axis=2)

    if debug:
        _debug_check_alignment(x_batches, y_batches, x_shuf, y_shuf,
                               perms_k=perms_k, perms_b=perms_b,
                               samples_to_check=samples_to_check)

    return x_shuf, y_shuf


def _debug_check_alignment(x_in, y_in, x_out, y_out, *, perms_k, perms_b, samples_to_check: int):
    """
    仅在 jit 内部调用，纯 JAX 实现对齐校验：
    检查若干 (i,j,k) 位置：输出 (i,j,k) 等于输入 (i, perms_k[i,j], perms_b[i,k]/k)
    """
    # 这些 shape 是编译期常量（来自 .shape），可安全用于 Python range
    M, K, B = x_in.shape[0], x_in.shape[1], x_in.shape[2]
    Ii = tuple(range(min(M, samples_to_check)))
    Jj = tuple(range(min(K, samples_to_check)))
    Kk = tuple(range(min(B, samples_to_check)))

    ok = jnp.array(True, dtype=jnp.bool_)
    for i in Ii:
        for j in Jj:
            for k in Kk:
                jj = perms_k[i, j]
                kk = k if (perms_b is None) else perms_b[i, k]

                x_ref = x_in[i, jj, kk, :]
                y_ref = y_in[i, jj, kk]
                x_now = x_out[i, j, k, :]
                y_now = y_out[i, j, k]

                ok_x = jnp.all(x_ref == x_now)
                ok_y = jnp.array(y_ref == y_now)
                ok   = jnp.logical_and(ok, jnp.logical_and(ok_x, ok_y))

    # 打印并在 host 上断言（不会打断 XLA，失败时抛 Python 异常）
    jax.debug.print("[shuffle] alignment ok? {}", ok)

    def _host_assert(flag):
        if not bool(flag):
            raise AssertionError("[shuffle] debug check failed: x/y misaligned.")
    jax.debug.callback(_host_assert, ok)


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
