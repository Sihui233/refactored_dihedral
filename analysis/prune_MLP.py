# === prune_MLP.py ===
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


def _sorted_dense_keys(params: Dict[str, Any]) -> List[str]:
    return sorted(
        [k for k in params.keys() if k.startswith("dense_")],
        key=lambda s: int(s.split("_")[1]),
    )


def _layer_widths(params: Dict[str, Any], dense_keys: List[str]) -> List[int]:
    return [int(params[k]["bias"].shape[0]) for k in dense_keys]


# --- pure functional layer masking (no .at[...] inside vmap) ---
def _apply_layer_mask(
    params: Dict[str, Any] | FrozenDict,
    dense_keys: List[str],
    li: int,
    col_mask_1d: jnp.ndarray,
) -> Dict[str, Any]:
    """
    Return new params with layer li neuron columns/bias masked by col_mask_1d
    and next-layer rows masked by the same col_mask_1d.
    """
    if isinstance(params, FrozenDict):
        base = params.unfreeze()
    else:
        base = params

    # ensure float dtype matches layer tensors
    k = dense_keys[li]
    H = base[k]["bias"].shape[0]
    cm = col_mask_1d.astype(base[k]["bias"].dtype)  # (H,)
    cm_col = cm[None, :]  # (1,H)
    cm_row = cm[:, None]  # (H,1)

    # current layer: zero selected neuron columns & their bias
    new_layer = dict(base[k])
    new_layer["kernel"] = new_layer["kernel"] * cm_col
    new_layer["bias"] = new_layer["bias"] * cm

    new_params = dict(base)
    new_params[k] = new_layer

    # next layer (or output): zero rows for those neurons
    if li + 1 < len(dense_keys):
        nk = dense_keys[li + 1]
        next_layer = dict(new_params[nk])
        next_layer["kernel"] = next_layer["kernel"] * cm_row
        new_params[nk] = next_layer
    elif "output_dense" in new_params:
        out_l = dict(new_params["output_dense"])
        out_l["kernel"] = out_l["kernel"] * cm_row
        new_params["output_dense"] = out_l

    return new_params


def _apply_prunes_masked(
    params: Dict[str, Any] | FrozenDict,
    dense_keys: List[str],
    prunes: Dict[int, List[int]],
) -> Dict[str, Any]:
    """Apply many layer→indices zeros at once via masks (pure)."""
    out = params.unfreeze() if isinstance(params, FrozenDict) else params
    widths = _layer_widths(out, dense_keys)

    for li, idxs in prunes.items():
        if not idxs:
            continue
        H = widths[li]

        # build 1D mask with zeros at idxs (no .at[...] to avoid tracer issues)
        idxs_arr = jnp.array(list(map(int, idxs)), dtype=jnp.int32)
        cm = jnp.ones((H,), dtype=out[dense_keys[li]]["bias"].dtype)
        # 1 - sum(one_hot(idxs)) creates zeros at idxs; clip in case of repeats
        hot = jax.nn.one_hot(idxs_arr, H, dtype=cm.dtype).sum(0)
        cm = jnp.clip(1.0 - hot, 0.0, 1.0)

        out = _apply_layer_mask(out, dense_keys, li, cm)

    return out


# --- accuracy over full dataset (static-sized via pad+mask, as you had) ---
def _dataset_accuracy(model, params, xs, ys, batch_size: int):
    N = int(xs.shape[0])
    nb = (N + batch_size - 1) // batch_size
    pad = nb * batch_size - N

    if pad:
        xs = jnp.concatenate([xs, jnp.repeat(xs[-1:], pad, axis=0)], axis=0)
        ys = jnp.concatenate([ys, jnp.repeat(ys[-1:], pad, axis=0)], axis=0)
        mask = jnp.concatenate(
            [jnp.ones((N,), bool), jnp.zeros((pad,), bool)], axis=0
        )
    else:
        mask = jnp.ones((N,), bool)

    mask = mask.reshape(nb, batch_size)

    def body(i, acc):
        start = i * batch_size
        xb = jax.lax.dynamic_slice_in_dim(xs, start, batch_size, axis=0)
        yb = jax.lax.dynamic_slice_in_dim(ys, start, batch_size, axis=0)
        mb = mask[i]
        logits, *_ = model.apply({"params": params}, xb, training=False)
        pred = jnp.argmax(logits, axis=-1)
        correct = (pred == yb) & mb
        return acc + jnp.sum(correct.astype(jnp.int32))

    total_correct = jax.lax.fori_loop(0, nb, body, jnp.array(0, jnp.int32))
    return total_correct / jnp.array(N, jnp.float32)


# --- vmapped per-neuron accuracies for a single layer using masks (pure) ---
def _per_neuron_accs_layer(model, params, xs, ys, batch_size: int, dense_keys: List[str], li: int) -> jnp.ndarray:
    k = dense_keys[li]
    H = int(params[k]["bias"].shape[0])

    # Build all “drop-one” masks in one tensor: (H, H) eye = I; rows == neuron idx
    eye = jnp.eye(H, dtype=params[k]["bias"].dtype)
    col_masks = 1.0 - eye  # each row: zeros at j, 1 elsewhere

    def acc_for_mask(cm):
        p_masked = _apply_layer_mask(params, dense_keys, li, cm)  # pure
        return _dataset_accuracy(model, p_masked, xs, ys, batch_size)

    return jax.vmap(acc_for_mask)(col_masks)  # shape (H,)


# --- max ReLU per layer over full grid (pure, no side effects) ---
def _max_relu_activations(model, params, xs, batch_size: int) -> List[jnp.ndarray]:
    N = int(xs.shape[0])
    nb = (N + batch_size - 1) // batch_size
    dense_keys = _sorted_dense_keys(params)
    widths = _layer_widths(params, dense_keys)
    init = tuple(jnp.zeros((w,), dtype=jnp.float32) for w in widths)

    def body(i, carry):
        start = i * batch_size
        xb = jax.lax.dynamic_slice_in_dim(xs, start, batch_size, axis=0)
        _, preacts, *_ = model.apply({"params": params}, xb, training=False)
        # update max per layer
        new = []
        for m_old, pre in zip(carry, preacts):
            m_new = jnp.maximum(m_old, jnp.max(jnp.maximum(pre, 0.0), axis=0))
            new.append(m_new)
        return tuple(new)

    out = jax.lax.fori_loop(0, nb, body, init)
    return list(out)


# === helper ===
def _max_abs_activations(model, params, xs, batch_size: int) -> List[jnp.ndarray]:
    N = int(xs.shape[0])
    nb = (N + batch_size - 1) // batch_size
    dense_keys = _sorted_dense_keys(params)
    widths = _layer_widths(params, dense_keys)
    init = tuple(jnp.zeros((w,), dtype=jnp.float32) for w in widths)

    def body(i, carry):
        start = i * batch_size
        xb = jax.lax.dynamic_slice_in_dim(xs, start, batch_size, axis=0)
        _, preacts, *_ = model.apply({"params": params}, xb, training=False)
        new = []
        for m_old, pre in zip(carry, preacts):
            m_new = jnp.maximum(m_old, jnp.max(jnp.abs(pre), axis=0))
            new.append(m_new)
        return tuple(new)

    out = jax.lax.fori_loop(0, nb, body, init)
    return list(out)


# --- main batched pruner (same API you were calling) ---
def prune_two_stage_by_accuracy_batched(
    *,
    model,
    params: Dict[str, Any] | FrozenDict,
    full_x: jnp.ndarray,
    full_y: jnp.ndarray,
    batch_size: int = 4096,
    abs_acc_th: float = 0.005,  # no usage here
    hard_min_acc: float = 1.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(params, FrozenDict):
        params = params.unfreeze()

    dense_keys = _sorted_dense_keys(params)
    widths = _layer_widths(params, dense_keys)
    L = len(dense_keys)

    baseline = _dataset_accuracy(model, params, full_x, full_y, batch_size)

    report: Dict[str, Any] = {
        "baseline_acc": float(baseline),
        "stageA": {i: [] for i in range(L)},
        "stageA_alive": {i: list(range(widths[i])) for i in range(L)},
        "stageB": {i: [] for i in range(L)},
        "stageB_alive": {i: list(range(widths[i])) for i in range(L)},
    }

    # ===== Stage A: collect candidates per layer (vmapped masks), then prune all at once =====
    maxabs_per_layer = _max_abs_activations(model, params, full_x, batch_size)
    stageA_abs_thresh = abs_acc_th

    candidatesA: Dict[int, List[int]] = {}
    for li in range(L):
        if widths[li] == 0:
            candidatesA[li] = []
            continue
        m = maxabs_per_layer[li]  # shape=(H,)
        cand = jnp.where(m < stageA_abs_thresh, jnp.arange(widths[li]), -1)
        cand = cand[cand >= 0].tolist()
        candidatesA[li] = list(map(int, cand))

    trialA = _apply_prunes_masked(params, dense_keys, candidatesA)
    accA = _dataset_accuracy(model, trialA, full_x, full_y, batch_size)

    if accA >= hard_min_acc:
        paramsA = trialA
        for li in range(L):
            report["stageA"][li] = candidatesA[li]
            alive = [i for i in range(widths[li]) if i not in set(candidatesA[li])]
            report["stageA_alive"][li] = alive
        baseline = accA
    else:
        paramsA = params
        for li in range(L):
            if not candidatesA[li]:
                continue
            trial_layer = _apply_prunes_masked(paramsA, dense_keys, {li: candidatesA[li]})
            acc_layer = _dataset_accuracy(model, trial_layer, full_x, full_y, batch_size)
            if acc_layer >= hard_min_acc:
                paramsA = trial_layer
                report["stageA"][li] = candidatesA[li]
                baseline = acc_layer
                alive = [i for i in range(widths[li]) if i not in set(report["stageA"][li])]
                report["stageA_alive"][li] = alive

    # ===== Stage B: activation floor (0.5% of global max), prune all at once =====
    max_per_layer = _max_relu_activations(model, paramsA, full_x, batch_size)
    global_max = max(float(jnp.max(m)) for m in max_per_layer) if max_per_layer else 0.0
    thresh = 0.005 * global_max

    report["global_activation_max"] = float(global_max)
    report["activation_threshold"] = float(thresh)

    candidatesB: Dict[int, List[int]] = {}
    for li in range(L):
        if widths[li] == 0:
            candidatesB[li] = []
            continue
        m = max_per_layer[li]
        cand = jnp.where(m < thresh, jnp.arange(widths[li]), -1).tolist()
        cand = [int(i) for i in cand if i >= 0 and i not in set(report["stageA"][li])]
        candidatesB[li] = cand

    trialB = _apply_prunes_masked(paramsA, dense_keys, candidatesB)
    accB = _dataset_accuracy(model, trialB, full_x, full_y, batch_size)

    if accB >= hard_min_acc:
        paramsB = trialB
        for li in range(L):
            report["stageB"][li] = candidatesB[li]
            alive = [i for i in report["stageA_alive"][li] if i not in set(candidatesB[li])]
            report["stageB_alive"][li] = alive
        baseline = accB
    else:
        paramsB = paramsA
        for li in range(L):
            report["stageB_alive"][li] = report["stageA_alive"][li]

    report["final_acc"] = float(baseline)
    report["alive_final"] = {li: report["stageB_alive"][li] for li in range(L)}
    report["dead_final"] = {
        li: sorted(
            set(range(_layer_widths(paramsB, dense_keys)[li])) - set(report["alive_final"][li])
        )
        for li in range(L)
    }
    report["alive_counts"] = {li: len(report["alive_final"][li]) for li in range(L)}
    report["dead_counts"] = {li: len(report["dead_final"][li]) for li in range(L)}

    return paramsB, report
