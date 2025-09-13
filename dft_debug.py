# ── dft_debug.py ──
# #!/usr/bin/env python3
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

os.environ["JAX_ENABLE_X64"] = "1"
jax.config.update("jax_enable_x64", True)


from DFT import make_irreps_Dn, build_rho_cache, jit_wrap_group_dft, inverse_group_dft_fast

def jax_forward(f_flat: jnp.ndarray, 
                rho_cache, irreps, 
                group_size: int # group_size=|G|
                ):
    """
    f_flat shape: (|G|*|G|, N)
    return Fhat(dict), key is (r_name, s_name, n_idx).
    """
    dft_fn = jit_wrap_group_dft(rho_cache, irreps, group_size)  # 已优化过的包装器
    return dft_fn(f_flat)


tol = 2e-6
# test function
def test_forward_inverse(n, num_neurons=4, tol=1e-5):
    """
    generate radom signals on D_n, do forward → inverse GFT and return the max error.
    """
    G, irreps = make_irreps_Dn(n)
    p = len(G)
    # 构造 rho_cache
    rho_cache_jax = build_rho_cache(G, irreps)

    # 随机信号：形状 (p*p, num_neurons)
    rng = np.random.default_rng(0)
    f_np = rng.standard_normal((p*p, num_neurons))
    f_jax = jnp.array(f_np)

    # forward
    Fhat_jax = jax_forward(f_jax, rho_cache_jax, irreps, p)
    # inverse
    f_rec_jax = inverse_group_dft_fast(Fhat_jax, rho_cache_jax, irreps, p, num_neurons)

    # reshape 回 (p,p,N)
    f_orig_grid = f_np.reshape(p, p, num_neurons)
    f_rec_np    = np.array(f_rec_jax)

    err = np.max(np.abs(f_rec_np - f_orig_grid))
    print(f"[n={n:2d}] max reconstruction error = {err:.2e}", 
          "✔" if err < tol else "✘")
    return err

def test_impulse(n):
    G, irreps = make_irreps_Dn(n)
    p = len(G)
    rho_cache = build_rho_cache(G, irreps)
    # single neuron
    N = 1
    # impulse at grid (i0, j0)
    i0, j0 = 2, 3
    f = np.zeros((p, p, N))
    f[i0, j0, 0] = 1.0
    f_flat = jnp.array(f.reshape(-1, N))
    # forward + inverse
    Fhat = jax_forward(f_flat, rho_cache, irreps, p)
    f_rec = inverse_group_dft_fast(Fhat, rho_cache, irreps, p, N)
    err = np.max(np.abs(f_rec - f))
    print(f"Impulse test n={n}: error={err:.2e}", "PASS" if err < tol else "FAIL")

def test_zero(n):
    G, irreps = make_irreps_Dn(n)
    p = len(G)
    rho_cache = build_rho_cache(G, irreps)
    N = 3
    f = np.zeros((p, p, N))
    f_flat = jnp.array(f.reshape(-1, N))
    Fhat = jax_forward(f_flat, rho_cache, irreps, p)
    f_rec = inverse_group_dft_fast(Fhat, rho_cache, irreps, p, N)
    err = np.max(np.abs(f_rec))
    print(f"Zero test n={n}: max|recon|={err:.2e}", "PASS" if err < tol else "FAIL")

def test_single_freq(n):
    G, irreps = make_irreps_Dn(n)
    p = len(G)
    rho_cache = build_rho_cache(G, irreps)
    # choose first 2D irrep if present, else trivial
    names = [name for name, _, _, _ in irreps]
    r_name, s_name = names[0], names[0]
    # build f(g1,g2) = trace[rho_r(g1)] * trace[rho_s(g2)]
    N = 1
    f = np.zeros((p, p, N), dtype=np.complex64)
    for i, g1 in enumerate(G):
        for j, g2 in enumerate(G):
            val = np.trace(np.array(rho_cache[r_name][i])) * \
                  np.trace(np.array(rho_cache[s_name][j]))
            f[i, j, 0] = val
    f_flat = jnp.array(f.reshape(-1, N))
    Fhat = jax_forward(f_flat, rho_cache, irreps, p)
    # check only (r_name,s_name,0) is nonzero
    nonzero = []
    for key, M in Fhat.items():
        norm = jnp.linalg.norm(M).item()
        if norm > tol:
            nonzero.append((key, norm))
    print("Single freq test n=",n, "nonzero Fhat keys:", nonzero)

def test_complex_input(n):
    G, irreps = make_irreps_Dn(n)
    p = len(G)
    rho_cache = build_rho_cache(G, irreps)
    N = 2
    rng = np.random.default_rng(42)
    real = rng.standard_normal((p*p, N))
    imag = rng.standard_normal((p*p, N))
    f = real + 1j * imag
    f_flat = jnp.array(f)
    Fhat = jax_forward(f_flat, rho_cache, irreps, p)
    f_rec = inverse_group_dft_fast(Fhat, rho_cache, irreps, p, N)
    err = np.max(np.abs(f_rec - f.reshape(p, p, N)))
    print(f"Complex input test n={n}: error={err:.2e}", "PASS" if err < tol else "FAIL")

# def test_plancherel(n, num_neurons=5, tol=5e-6):
#     """
#     Check energy conservation:
#       sum_{g,h,n} |f(g,h,n)|^2  ≈  (1/|G|^2) * sum_{r,s,n} d_r d_s ||Fhat_{r,s}^{(n)}||_F^2
#     (matches your 1/|G|^2 normalization inside _einsum_block)
#     """
#     G, irreps = make_irreps_Dn(n)
#     p = len(G)
#     rho_cache = build_rho_cache(G, irreps)
#     dft_fn = jit_wrap_group_dft(rho_cache, irreps, p)

#     rng = np.random.default_rng(0)
#     # complex input to stress both real/imag paths
#     real = rng.standard_normal((p*p, num_neurons))
#     imag = rng.standard_normal((p*p, num_neurons))
#     f = real + 1j*imag
#     f_flat = jnp.array(f)

#     Fhat = dft_fn(f_flat)

#     energy_in = np.sum(np.abs(f)**2)  # scalar

#     # accumulate output-side energy with weights d_r d_s / |G|^2
#     energy_out = 0.0
#     for (r_name, d_r, _, _ ) in irreps:
#         for (s_name, d_s, _, _ ) in irreps:
#             for n_idx in range(num_neurons):
#                 key = (r_name, s_name, n_idx)
#                 if key in Fhat:
#                     M = np.asarray(Fhat[key])  # (d_r,d_r,d_s,d_s)
#                     # Frobenius norm over ijkl
#                     e = np.sum(np.abs(M)**2)
#                     energy_out += (d_r * d_s) * e
#     energy_out /= (p*p)  # divide by |G|^2

#     err = abs(energy_in - energy_out) / max(1.0, abs(energy_in))
#     print(f"Plancherel test n={n}: rel_err={err:.2e}", "PASS" if err < tol else "FAIL")
#     return err

# 替换你现在的 test_plancherel：
def test_plancherel_via_reconstruction(n, num_neurons=5, tol=5e-6):
    G, irreps = make_irreps_Dn(n); p = len(G)
    rho_cache = build_rho_cache(G, irreps)
    dft_fn = jit_wrap_group_dft(rho_cache, irreps, p)

    rng = np.random.default_rng(0)
    f = rng.standard_normal((p*p, num_neurons)) + 1j*rng.standard_normal((p*p, num_neurons))
    f_flat = jnp.array(f)

    Fhat = dft_fn(f_flat)
    f_rec = inverse_group_dft_fast(Fhat, rho_cache, irreps, p, num_neurons)  # (p,p,N)

    e_in  = np.vdot(f, f).real
    e_rec = np.vdot(f_rec.reshape(p*p, num_neurons), f_rec.reshape(p*p, num_neurons)).real
    rel = abs(e_in - e_rec) / max(1.0, abs(e_in))
    print(f"Plancherel-by-recon n={n}: rel_err={rel:.2e}", "PASS" if rel < tol else "FAIL")
    return rel

def _forward_naive_per_neuron(f_flat, rho_cache, irreps, p):
    """
    Reference: run DFT one neuron at a time (B=1) and stitch dicts.
    """
    N = f_flat.shape[1]
    out = {}
    for n_idx in range(N):
        Fhat_n = jax_forward(f_flat[:, n_idx:n_idx+1], rho_cache, irreps, p)
        for (k, v) in Fhat_n.items():
            # k is (r,s,0) in this 1-neuron call; remap to (r,s,n_idx)
            out[(k[0], k[1], n_idx)] = np.asarray(v)  # drop last dim
    return out

def test_chunk_equivalence(n, num_neurons=7, tol=1e-6):
    """
    Compare your default batched path vs an explicit per-neuron loop.
    """
    G, irreps = make_irreps_Dn(n)
    p = len(G)
    rho_cache = build_rho_cache(G, irreps)
    dft_fn = jit_wrap_group_dft(rho_cache, irreps, p)

    rng = np.random.default_rng(1)
    f = rng.standard_normal((p*p, num_neurons)).astype(np.float32)
    f_flat = jnp.array(f)

    Fhat_batched = dft_fn(f_flat)                      # your default
    Fhat_ref     = _forward_naive_per_neuron(f_flat, rho_cache, irreps, p)

    # compare keys and tensors
    keys_b = set(Fhat_batched.keys())
    keys_r = set(Fhat_ref.keys())
    if keys_b != keys_r:
        missing = keys_r - keys_b
        extra   = keys_b - keys_r
        raise AssertionError(f"Key mismatch. missing={len(missing)} extra={len(extra)}")

    worst = 0.0
    for k in keys_b:
        a = np.asarray(Fhat_batched[k])
        b = np.asarray(Fhat_ref[k])
        # shapes: batched: (d_r,d_r,d_s,d_s) ; ref: same
        diff = np.max(np.abs(a - b))
        worst = max(worst, diff)
        if diff > tol:
            raise AssertionError(f"Chunk equivalence failed at key={k}, max|Δ|={diff}")
    print(f"Chunk equivalence n={n}: worst_abs_diff={worst:.2e} PASS")
    return worst

def test_grad_consistency(n, num_neurons=3, tol=1e-3):
    """
    Check grad wrt f_flat via autodiff vs finite diff on a scalar loss.
    Loss: L = ||inverse(DFT(f)) - f_grid||^2
    """
    G, irreps = make_irreps_Dn(n)
    p = len(G)
    rho_cache = build_rho_cache(G, irreps)
    dft_fn = jit_wrap_group_dft(rho_cache, irreps, p)

    rng = np.random.default_rng(0)
    f0 = rng.standard_normal((p*p, num_neurons)).astype(np.float32)
    f0_j = jnp.array(f0)

    def loss_fn(f_flat):
        Fhat = dft_fn(f_flat)
        f_rec = inverse_group_dft_fast(Fhat, rho_cache, irreps, p, num_neurons)  # (p,p,N)
        f_grid = f_flat.reshape(p, p, num_neurons)
        diff = f_rec - f_grid
        return jnp.vdot(diff, diff).real  # scalar

    # autodiff
    g_auto = jax.grad(loss_fn)(f0_j).astype(jnp.float32)
    g_auto = np.asarray(g_auto)

    # finite diff
    eps = 1e-3
    i, j = 0, 0  # probe a couple entries (更严格可随机多点)
    def loss_with_perturb(delta):
        fpert = f0.copy()
        fpert = f0.copy()
        fpert[i, j] += delta
        return float(loss_fn(jnp.array(fpert)))
    # central difference
    g_fd = (loss_with_perturb(+eps) - loss_with_perturb(-eps)) / (2*eps)

    rel = abs(g_auto[i, j] - g_fd) / max(1.0, abs(g_fd))
    print(f"Grad consistency n={n}: rel_err={rel:.2e}", "PASS" if rel < tol else "FAIL")
    return rel


if __name__ == "__main__":
    # n=3,4,5,6,8,10
    failures = []
    for n in [3,4,5,6,8,10]:
        err = test_forward_inverse(n)
        test_impulse(n)
        test_zero(n)
        test_single_freq(n)
        test_complex_input(n)
        test_plancherel_via_reconstruction(n)
        test_chunk_equivalence(n)
        test_grad_consistency(n)
        
        if err >= 2e-6:
            failures.append((n, err))
    if failures:
        print("\n Some tests failed:", failures)
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)
