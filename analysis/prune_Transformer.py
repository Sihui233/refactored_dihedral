# analysis/prune_Transformer.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import re

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

# -------------------- helpers to navigate params --------------------

def _get_mlp_subtree(params: Dict[str, Any] | FrozenDict) -> Dict[str, Any]:
    """Return mutable dict for params['blocks_0']['mlp'] (assumes num_layers=1 block).
    If you later train >1 blocks, extend to 'blocks_k' as needed."""
    p = params.unfreeze() if isinstance(params, FrozenDict) else params
    try:
        return p["blocks_0"]["mlp"]
    except KeyError:
        raise KeyError("Expected params['blocks_0']['mlp'] subtree not found.")

def _set_mlp_subtree(params: Dict[str, Any] | FrozenDict, new_mlp: Dict[str, Any]) -> Dict[str, Any]:
    base = params.unfreeze() if isinstance(params, FrozenDict) else params
    base["blocks_0"]["mlp"] = new_mlp
    return base

def _sorted_layer_indices(mlp: Dict[str, Any]) -> List[int]:
    """Find all W_i / b_i (i=0..L-1) by scanning keys; ignore W_out/b_out."""
    idxs = set()
    for k in mlp.keys():
        m = re.fullmatch(r"W_(\d+)", k)
        if m: idxs.add(int(m.group(1)))
    return sorted(idxs)

def _layer_widths(mlp: Dict[str, Any], layer_idxs: List[int]) -> List[int]:
    """Width == number of neurons == len(b_i)."""
    return [int(mlp[f"b_{i}"].shape[0]) for i in layer_idxs]

# -------------------- dataset accuracy (same batching style) --------------------

def _dataset_accuracy(model, params, xs, ys, batch_size: int) -> jnp.ndarray:
    N  = int(xs.shape[0])
    nb = (N + batch_size - 1) // batch_size
    pad = nb * batch_size - N
    if pad:
        xs = jnp.concatenate([xs, jnp.repeat(xs[-1:], pad, axis=0)], axis=0)
        ys = jnp.concatenate([ys, jnp.repeat(ys[-1:], pad, axis=0)], axis=0)
        mask = jnp.concatenate([jnp.ones((N,), bool), jnp.zeros((pad,), bool)], axis=0)
    else:
        mask = jnp.ones((N,), bool)
    mask = mask.reshape(nb, batch_size)

    def body(i, acc):
        start = i * batch_size
        xb = jax.lax.dynamic_slice_in_dim(xs, start, batch_size, axis=0)
        yb = jax.lax.dynamic_slice_in_dim(ys, start, batch_size, axis=0)
        mb = mask[i]
        logits = model.apply({"params": params}, xb, training=False)
        # Transformer: logits shape = (B, seq_len, vocab). 只评估最后一个 token。
        if logits.ndim == 3:
            logits_last = logits[:, -1, :]     # (B, vocab)
        else:
            logits_last = logits               # (B, vocab) for MLP
        pred = jnp.argmax(logits_last, axis=-1)  # (B,)
        correct = (pred == yb) & mb
        return acc + jnp.sum(correct.astype(jnp.int32))

    total_correct = jax.lax.fori_loop(0, nb, body, jnp.array(0, jnp.int32))
    return total_correct / jnp.array(N, jnp.float32)

# -------------------- pull MLP preactivations via HookPoints --------------------

def _find_by_suffix(d, suffix):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(k, str) and k.endswith(suffix):
                if isinstance(v, list):   # flax intermediates store lists
                    return v[0]
                if isinstance(v, dict):
                    return next(iter(v.values()))
                return v
            out = _find_by_suffix(v, suffix)
            if out is not None:
                return out
    elif isinstance(d, list):
        for x in d:
            out = _find_by_suffix(x, suffix)
            if out is not None:
                return out
    return None

def _max_abs_preacts(model, params, xs, batch_size: int, num_mlp_layers: int, last_token_index: int = 1) -> List[jnp.ndarray]:
    """Per-layer max |preact| over dataset (last token only), with pre-padding to fixed batches."""
    N  = int(xs.shape[0])
    nb = (N + batch_size - 1) // batch_size
    pad = nb * batch_size - N
    if pad:
        xs = jnp.concatenate([xs, jnp.repeat(xs[-1:], pad, axis=0)], axis=0)

    xb0 = xs[:1]
    _, inter0 = model.apply({"params": params}, xb0, training=False, mutable=["intermediates"])
    ints0 = inter0["intermediates"]
    widths = []
    for l in range(1, num_mlp_layers + 1):
        arr0 = _find_by_suffix(ints0, f"blocks_0/mlp/hook_pre{l}")
        if arr0 is None:
            arr0 = _find_by_suffix(ints0, f"hook_pre{l}")
        if arr0 is None:
            raise KeyError(f"Could not find hook_pre{l} in intermediates.")
        widths.append(int(arr0.shape[-1]))
    max_vecs = [jnp.zeros((w,), dtype=jnp.float32) for w in widths]

    def body(i, carry):
        max_vecs = carry
        start = i * batch_size
        xb = jax.lax.dynamic_slice_in_dim(xs, start, batch_size, axis=0)
        _, inter = model.apply({"params": params}, xb, training=False, mutable=["intermediates"])
        ints = inter["intermediates"]
        curr = []
        for l in range(1, num_mlp_layers + 1):
            arr = _find_by_suffix(ints, f"blocks_0/mlp/hook_pre{l}")
            if arr is None:
                arr = _find_by_suffix(ints, f"hook_pre{l}")
            if arr is None:
                raise KeyError(f"Could not find hook_pre{l} in batch intermediates.")
            pre = jnp.asarray(arr)[:, last_token_index, :]          # (B, width_l)
            curr.append(jnp.max(jnp.abs(pre), axis=0))              # (width_l,)
        new_max = [jnp.maximum(mv, c) for mv, c in zip(max_vecs, curr)]
        return new_max

    max_vecs = jax.lax.fori_loop(0, nb, body, max_vecs)
    return max_vecs


def _max_relu_preacts(model, params, xs, batch_size: int, num_mlp_layers: int, last_token_index: int = 1) -> List[jnp.ndarray]:
    """Per-layer max ReLU(preact) over dataset (last token), with pre-padding."""
    N  = int(xs.shape[0])
    nb = (N + batch_size - 1) // batch_size
    pad = nb * batch_size - N
    if pad:
        xs = jnp.concatenate([xs, jnp.repeat(xs[-1:], pad, axis=0)], axis=0)

    xb0 = xs[:1]
    _, inter0 = model.apply({"params": params}, xb0, training=False, mutable=["intermediates"])
    ints0 = inter0["intermediates"]
    widths = []
    for l in range(1, num_mlp_layers + 1):
        arr0 = _find_by_suffix(ints0, f"blocks_0/mlp/hook_pre{l}")
        if arr0 is None:
            arr0 = _find_by_suffix(ints0, f"hook_pre{l}")
        if arr0 is None:
            raise KeyError(f"Could not find hook_pre{l} in intermediates.")
        widths.append(int(arr0.shape[-1]))
    max_vecs = [jnp.zeros((w,), dtype=jnp.float32) for w in widths]

    def body(i, carry):
        max_vecs = carry
        start = i * batch_size
        xb = jax.lax.dynamic_slice_in_dim(xs, start, batch_size, axis=0)
        _, inter = model.apply({"params": params}, xb, training=False, mutable=["intermediates"])
        ints = inter["intermediates"]
        curr = []
        for l in range(1, num_mlp_layers + 1):
            arr = _find_by_suffix(ints, f"blocks_0/mlp/hook_pre{l}")
            if arr is None:
                arr = _find_by_suffix(ints, f"hook_pre{l}")
            if arr is None:
                raise KeyError(f"Could not find hook_pre{l} in batch intermediates.")
            pre = jnp.asarray(arr)[:, last_token_index, :]
            curr.append(jnp.max(jnp.maximum(pre, 0.0), axis=0))
        new_max = [jnp.maximum(mv, c) for mv, c in zip(max_vecs, curr)]
        return new_max

    max_vecs = jax.lax.fori_loop(0, nb, body, max_vecs)
    return max_vecs


# -------------------- layer masks (Transformer MLP geometry) --------------------

def _apply_layer_mask_transformer(
    params: Dict[str, Any] | FrozenDict,
    li: int,
    colmask_for_next_1d: jnp.ndarray,
) -> Dict[str, Any]:
    """
    Kill neurons at MLP layer li (inside blocks_0/mlp):
      • Row-mask W_li and its bias (these are the *outputs* / neurons at layer li)
      • Col-mask W_{li+1} (or W_out if li is the last hidden layer)

    Shapes from your transformer_class.MLP:
      W_i:   (out_dim, in_dim)   # einsum("md,bpd->bpm")
      b_i:   (out_dim,)
      W_out: (d_model, d_mlp)    # einsum("dm,bpm->bpd")
    """
    base = params.unfreeze() if isinstance(params, FrozenDict) else params
    mlp = _get_mlp_subtree(base)

    # ensure dtype
    H = int(mlp[f"b_{li}"].shape[0])
    cm = colmask_for_next_1d.astype(mlp[f"b_{li}"].dtype)    # (H,)
    cm_row = cm[:, None]                                     # (H,1)  for *rows* of W_li
    cm_col = cm[None, :]                                     # (1,H)  for *cols* of next W

    # mask current layer outputs (rows) and bias
    Wi = jnp.asarray(mlp[f"W_{li}"])
    bi = jnp.asarray(mlp[f"b_{li}"])
    mlp[f"W_{li}"] = (Wi * cm_row).astype(Wi.dtype)
    mlp[f"b_{li}"] = (bi * cm).astype(bi.dtype)

    # mask next mapping's *columns*
    next_key = f"W_{li+1}"
    if next_key in mlp:
        Wn = jnp.asarray(mlp[next_key])
        mlp[next_key] = (Wn * cm_col).astype(Wn.dtype)
    else:
        # last hidden layer → mask W_out columns
        Wout = jnp.asarray(mlp["W_out"])
        mlp["W_out"] = (Wout * cm_col).astype(Wout.dtype)

    return _set_mlp_subtree(base, mlp)

def _apply_prunes_masked_transformer(
    params: Dict[str, Any] | FrozenDict,
    prunes: Dict[int, List[int]]
) -> Dict[str, Any]:
    """Apply many layer→indices zeros at once via pure masks."""
    out = params.unfreeze() if isinstance(params, FrozenDict) else params
    mlp = _get_mlp_subtree(out)
    layer_idxs = _sorted_layer_indices(mlp)
    widths = _layer_widths(mlp, layer_idxs)
    for li in layer_idxs:
        idxs = list(map(int, prunes.get(li, [])))
        if not idxs:
            continue
        H = widths[layer_idxs.index(li)]
        # build 1D 0/1 column mask with zeros at idxs
        idxs_arr = jnp.array(idxs, dtype=jnp.int32)
        cm = jnp.ones((H,), dtype=mlp[f"b_{li}"].dtype)
        hot = jax.nn.one_hot(idxs_arr, H, dtype=cm.dtype).sum(0)
        cm = jnp.clip(1.0 - hot, 0.0, 1.0)
        out = _apply_layer_mask_transformer(out, li, cm)
        mlp = _get_mlp_subtree(out)  # refresh pointer
    return out

# -------------------- per-neuron gate (optional; not used in main flow) --------------------

def _per_neuron_accs_layer(
    model, params, xs, ys, batch_size: int, li: int
) -> jnp.ndarray:
    """Vmapped 'drop-one' accuracy for MLP layer li."""
    base = params.unfreeze() if isinstance(params, FrozenDict) else params
    mlp = _get_mlp_subtree(base)
    H = int(mlp[f"b_{li}"].shape[0])

    eye = jnp.eye(H, dtype=mlp[f"b_{li}"].dtype)   # (H,H)
    col_masks = 1.0 - eye                           # (H,H), each row kills idx j

    def acc_for_mask(cm):
        p_masked = _apply_layer_mask_transformer(params, li, cm)
        return _dataset_accuracy(model, p_masked, xs, ys, batch_size)

    return jax.vmap(acc_for_mask)(col_masks)  # (H,)

# -------------------- main two-stage pruner --------------------

def prune_two_stage_by_accuracy_batched_transformer(
    *,
    model,
    params: Dict[str, Any] | FrozenDict,
    full_x: jnp.ndarray,                 # e.g., full grid (p^2, 2)
    full_y: jnp.ndarray,
    num_mlp_layers: int,
    batch_size: int = 4096,
    abs_acc_th: float = 0.005,           # Stage A (abs preact floor)
    hard_min_acc: float = 1.0,           # must maintain exact acc (your usual gate)
    last_token_index: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Stage A: prune units whose max |preact| < abs_acc_th (over dataset).
    Stage B: prune units whose max ReLU(preact) < 0.5% of global max ReLU.
    All tests are evaluated *jointly* then fall back to per-layer if needed,
    mirroring your MLP flow.
    """
    if isinstance(params, FrozenDict):
        params = params.unfreeze()

    # layer bookkeeping
    mlp = _get_mlp_subtree(params)
    layer_idxs = _sorted_layer_indices(mlp)
    widths = _layer_widths(mlp, layer_idxs)
    L = len(layer_idxs)

    baseline = _dataset_accuracy(model, params, full_x, full_y, batch_size)

    report: Dict[str, Any] = {
        "baseline_acc": float(baseline),
        "stageA": {i: [] for i in range(L)},
        "stageA_alive": {i: list(range(widths[i])) for i in range(L)},
        "stageB": {i: [] for i in range(L)},
        "stageB_alive": {i: list(range(widths[i])) for i in range(L)},
    }

    # ===== Stage A: abs preactivation threshold (per-layer) =====
    maxabs_per_layer = _max_abs_preacts(model, params, full_x, batch_size, num_mlp_layers, last_token_index)
    candidatesA: Dict[int, List[int]] = {}
    for li in range(L):
        H = widths[li]
        m = maxabs_per_layer[li]  # (H,)
        cand = jnp.where(m < abs_acc_th, jnp.arange(H), -1)
        cand = [int(x) for x in cand.tolist() if x >= 0]
        candidatesA[li] = cand

    trialA = _apply_prunes_masked_transformer(params, candidatesA)
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
            trial_layer = _apply_prunes_masked_transformer(paramsA, {li: candidatesA[li]})
            acc_layer = _dataset_accuracy(model, trial_layer, full_x, full_y, batch_size)
            if acc_layer >= hard_min_acc:
                paramsA = trial_layer
                report["stageA"][li] = candidatesA[li]
                baseline = acc_layer
            alive = [i for i in range(widths[li]) if i not in set(report["stageA"][li])]
            report["stageA_alive"][li] = alive

    # ===== Stage B: activation floor = 0.5% of global max ReLU(preact) =====
    maxrelu_per_layer = _max_relu_preacts(model, paramsA, full_x, batch_size, num_mlp_layers, last_token_index)
    global_max = max(float(jnp.max(m)) for m in maxrelu_per_layer) if maxrelu_per_layer else 0.0
    thresh = 0.005 * global_max
    report["global_activation_max"] = float(global_max)
    report["activation_threshold"]  = float(thresh)

    candidatesB: Dict[int, List[int]] = {}
    for li in range(L):
        H = widths[li]
        m = maxrelu_per_layer[li]  # (H,)
        cand = jnp.where(m < thresh, jnp.arange(H), -1).tolist()
        cand = [int(i) for i in cand if i >= 0 and i not in set(report["stageA"][li])]
        candidatesB[li] = cand

    trialB = _apply_prunes_masked_transformer(paramsA, candidatesB)
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

    # final tallies
    report["final_acc"]   = float(baseline)
    report["alive_final"] = {li: report["stageB_alive"][li] for li in range(L)}
    report["dead_final"]  = {
        li: sorted(set(range(widths[li])) - set(report["alive_final"][li]))
        for li in range(L)
    }
    report["alive_counts"] = {li: len(report["alive_final"][li]) for li in range(L)}
    report["dead_counts"]  = {li: len(report["dead_final"][li])  for li in range(L)}
    return paramsB, report
