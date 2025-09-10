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

if __name__ == "__main__":
    # n=3,4,5,6,8,10
    failures = []
    for n in [3,4,5,6,8,10]:
        err = test_forward_inverse(n)
        test_impulse(n)
        test_zero(n)
        test_single_freq(n)
        test_complex_input(n)
        if err >= 2e-6:
            failures.append((n, err))
    if failures:
        print("\n Some tests failed:", failures)
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)
