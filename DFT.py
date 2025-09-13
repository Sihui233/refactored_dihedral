import jax
import jax.numpy as jnp
from typing import Dict, Tuple
from functools import partial
import numpy as np
import os
os.environ["JAX_ENABLE_X64"] = "1"

def make_irreps_Dn(n):
    """
    Return list of irreps for D_n: (name, dim, R_func), using JAX arrays
    """
    
    irreps = []
    # trivial representation
    irreps.append(('triv', 1, lambda g: jnp.array([[1.0]], dtype=jnp.complex64), 0))
    # sign representation: rotations->+1, reflections->-1
    irreps.append(('sign', 1, lambda g: jnp.array([[1 if g[0]=='rot' else -1]], dtype=jnp.complex64), -1))
    # extra one-dimensional irreps for even n
    if n % 2 == 0:
        irreps.append(('rp', 1,
            lambda g: jnp.array([[(-1.0)**g[1]]], dtype=jnp.complex64), n//2))      # r^m ↦ (-1)^m
        irreps.append(('srp', 1,
            lambda g: jnp.array([[(-1.0)**g[1] *
                                (-1 if g[0] == 'ref' else 1)]], dtype=jnp.complex64), - n//2)) # s r^m ↦ (-1) (-1)^m
    # two-dimensional irreps (standard representations)
    maxk = (n - 1) // 2
    for k in range(1, maxk + 1):
        def Rk(g, k=k):
            m = g[1]
            theta = 2.0 * jnp.pi * k * m / n        # float64
            c, s   = np.cos(theta).astype(np.float64), np.sin(theta).astype(np.float64)
            R      = np.array([[c, -s],
                            [s,  c]], dtype=np.float64)

            if g[0] == 'rot':                       # ρₖ(r^m)
                return jnp.array(R, dtype=jnp.complex64)

            S = np.array([[1.0, 0.0],
                        [0.0,-1.0]], dtype=np.float64)
            return jnp.array(S @ R, dtype=jnp.complex64) # s r^m
            # return jnp.array(R @ S, dtype=jnp.complex64) # r^m s
        
        irreps.append((f'2D_{k}', 2, Rk, k))
    # Build group element list G
    G = [('rot', i) for i in range(n)] + [('ref', i) for i in range(n)]
    return G, irreps

def build_rho_cache(G, irreps):
    return {
        name: jnp.stack([R(g) for g in G], axis=0)
        for name, _, R, _ in irreps
    }


# def _group_dft_preacts_inner(preacts, rho_cache, irreps, group_size):
    
#     f_grid = preacts.reshape(group_size, group_size, -1)  # (|G|,|G|,N)
#     Fhat = {}
    
#     for r_name, d_r, _, _ in irreps:
#         rho_r_dag = rho_cache[r_name].conj().transpose(0,2,1)
#         for s_name, d_s, _, _ in irreps:
#             rho_s_dag = rho_cache[s_name].conj().transpose(0,2,1)
            
#             # einsum: g1ij, g1g2n, g2kl -> ijklN
#             # Aligns with: sum_{g1,g2} f(g1,g2) · rho1(g1)* ⊗ rho2(g2)*
#             M_all = jnp.einsum('aij,abn,bkl->ijkln',
#                    rho_r_dag, f_grid, rho_s_dag) / group_size**2  # normalize by |G|^2
            
#             for n_idx in range(M_all.shape[-1]):
#                 Fhat[(r_name, s_name, n_idx)] = M_all[..., n_idx]  # shape: (d_r, d_r, d_s, d_s)
    
#     return Fhat


# def jit_wrap_group_dft(rho_cache, irreps, group_size):
#     """
#     return a JIT compiled dft_fn = lambda preacts -> Fhat
#     user pass in rho_cache / irreps / group_size
#     """
#     return jax.jit(
#         partial(_group_dft_preacts_inner,
#                 rho_cache=rho_cache,
#                 irreps=irreps,
#                 group_size=group_size),
#         static_argnames=('rho_cache', 'irreps', 'group_size')  # 全静态
#     )

# only JIT this small kernal
# @jax.jit
# def _einsum_block(f_grid, rho_r_dag, rho_s_dag, group_size):
    
#     M = jnp.einsum('aij,abn,bkl->ijkln', rho_r_dag, f_grid, rho_s_dag)
#     # normalized by |G|^2
#     gsq = group_size**2 # f_grid.shape[0] * f_grid.shape[0]
#     return M / gsq
@partial(jax.jit, static_argnums=(3,))
def _einsum_block(f_grid, rho_r_dag, rho_s_dag, group_size):
    """
    f_grid:    (|G|, |G|, N)
    rho_r_dag: (|G|, d_r, d_r)
    rho_s_dag: (|G|, d_s, d_s)
    return:    (d_r, d_r, d_s, d_s, N)
    """
    # step1: sum over 'a' -> (d_r, d_r, |G|, N)
    #   'aij,abn->ijbn'
    tmp = jnp.einsum('aij,abn->ijbn', rho_r_dag, f_grid)

    # step2: sum over 'b' -> (d_r, d_r, d_s, d_s, N)
    #   'bkl,ijbn->ijkln'
    M = jnp.einsum('bkl,ijbn->ijkln', rho_s_dag, tmp)

    inv_gsq = jnp.asarray(1.0, dtype=M.dtype) / (group_size * group_size)
    return M * inv_gsq

# def _group_dft_preacts_inner_nojit(preacts, rho_cache, irreps, group_size):
#     f_grid = jnp.asarray(preacts).reshape(group_size, group_size, -1)  # (|G|,|G|,N)
#     Fhat = {}
#     rho_dag_cache = {
#         name: jnp.asarray(arr).conj().swapaxes(-1, -2)  # 放在 device 上
#         for name, arr in rho_cache.items()
#     }
#     for r_name, d_r, _, _ in irreps:
#             rho_r_dag = rho_dag_cache[r_name]  # (|G|, d_r, d_r)
#             for s_name, d_s, _, _ in irreps:
#                 rho_s_dag = rho_dag_cache[s_name]  # (|G|, d_s, d_s)
                
#                 M_all = _einsum_block(f_grid, rho_r_dag, rho_s_dag, group_size)
                
#                 for n_idx in range(M_all.shape[-1]):
#                     Fhat[(r_name, s_name, n_idx)] = M_all[..., n_idx]
#     return Fhat
def _group_dft_preacts_inner_nojit(preacts, rho_cache, irreps, group_size):
    """
    分块沿着 n 维计算，避免一次 materialize (d_r,d_r,d_s,d_s,N)。
    计算后立即 device_get 到 CPU，避免 GPU 累积内存。
    """
    f_grid = jnp.asarray(preacts).reshape(group_size, group_size, -1)  # (G,G,N)
    N = int(f_grid.shape[-1])

    # 预处理 ρ^† 并确保是 f32（双精会把内存翻倍）
    rho_dag_cache = {}
    for name, arr in rho_cache.items():
        arr = jnp.asarray(arr)
        if jnp.issubdtype(arr.dtype, jnp.complexfloating):
            arr = arr.astype(jnp.complex64)   # 复数 → complex64
        else:
            arr = arr.astype(jnp.float32)     # 实数 → float32
        rho_dag_cache[name] = arr.conj().swapaxes(-1, -2)

    Fhat = {}

    # 简单的自适应 chunk 估计：控制中间 tmp/M 的体积都不超过 ~256MB
    TARGET_BYTES = 256 * 1024 * 1024

    for r_name, d_r, _, _ in irreps:
        rho_r_dag = rho_dag_cache[r_name]  # (G,d_r,d_r)
        for s_name, d_s, _, _ in irreps:
            rho_s_dag = rho_dag_cache[s_name]  # (G,d_s,d_s)

            bytes_per_sample_out = d_r * d_r * d_s * d_s * 4  # float32 4B
            bytes_per_sample_tmp = group_size * d_s * d_s * 4
            bytes_per_sample = max(bytes_per_sample_out, bytes_per_sample_tmp)

            if bytes_per_sample == 0:
                B = N
            else:
                B = max(1, min(N, int(TARGET_BYTES // bytes_per_sample)))

            # 为了避免多次编译，固定 chunk 大小 B（只有最后一块可能触发第二个编译）
            start = 0
            while start < N:
                stop = min(start + B, N)
                # (G,G,B)
                f_chunk = jax.lax.dynamic_slice_in_dim(f_grid, start, stop - start, axis=2)
                # (d_r,d_r,d_s,d_s,B)
                M_chunk = _einsum_block(f_chunk, rho_r_dag, rho_s_dag, group_size)

                # 立刻把每个 n 的结果搬到 CPU，释放 GPU 显存压力
                # 注意：这段不在 jit 里，device_get 会同步该块计算并回拷
                M_host = jax.device_get(M_chunk)  # numpy array on host

                B_eff = M_host.shape[-1]
                for bi in range(B_eff):
                    n_idx = start + bi
                    # 存 CPU 端的 ndarray：后续 Python 分析依然可用
                    Fhat[(r_name, s_name, int(n_idx))] = M_host[..., bi]

                start = stop

    return Fhat


# def _group_dft_preacts_inner_nojit(preacts, rho_cache, irreps, group_size):
#     f_grid = preacts.reshape(group_size, group_size, -1)  # (|G|,|G|,N)
#     Fhat = {}
#     for r_name, d_r, _, _ in irreps:
#         rho_r_dag = rho_cache[r_name].conj().transpose(0, 2, 1)  # (|G|, d_r, d_r)
#         for s_name, d_s, _, _ in irreps:
#             rho_s_dag = rho_cache[s_name].conj().transpose(0, 2, 1)  # (|G|, d_s, d_s)
            
#             M_all = _einsum_block(f_grid, rho_r_dag, rho_s_dag,group_size)
#             for n_idx in range(M_all.shape[-1]):
#                 Fhat[(r_name, s_name, n_idx)] = M_all[..., n_idx]
#     return Fhat

def jit_wrap_group_dft(rho_cache, irreps, group_size):
    return lambda preacts: _group_dft_preacts_inner_nojit(
        preacts, rho_cache, irreps, group_size
    )


# def inverse_group_dft(Fhat, rho_cache, irreps, group_size, num_neurons):
#     recon = jnp.zeros((group_size, group_size, num_neurons), dtype=jnp.complex64)
#     dim_map = {name: dim for name, dim, _, _ in irreps}

#     for (r_name, s_name, n_idx), M in Fhat.items():
#         d_r = dim_map[r_name]
#         d_s = dim_map[s_name]
        
#         rho_r = rho_cache[r_name]  # shape: (|G|, d_r, d_r)
#         rho_s = rho_cache[s_name]
        
#         # Reconstruct using: Tr[ M ⋅ (ρ_r(g1) ⊗ ρ_s(g2)) ]
#         # We reshape for batched kronecker and batched trace
#         # (d_r, d_r, d_s, d_s) → (d_r*d_s, d_r*d_s)
#         M_flat = M.reshape(d_r * d_s, d_r * d_s)

#         # kronecker product: shape (|G|, |G|, d_r * d_s, d_r * d_s)
#         # comp = jnp.einsum('ijlk,gji,hkl->gh', M, rho_r, rho_s) # same as below
#         comp = jnp.einsum('ijkl,gji,hlk->gh', M, rho_r, rho_s)


#         scale = d_r * d_s
#         recon = recon.at[:, :, n_idx].add(scale * comp.astype(recon.dtype))

#     return recon
from collections import defaultdict

@jax.jit
def _inv_block_many(Ms, rho_r, rho_s):
    # Ms: (d_r, d_r, d_s, d_s, Nk)  → comp: (|G|, |G|, Nk)
    return jnp.einsum('ijklN,gji,hlk->ghN', Ms, rho_r, rho_s)

def inverse_group_dft_fast(Fhat, rho_cache, irreps, group_size, num_neurons):
    recon = jnp.zeros((group_size, group_size, num_neurons), dtype=jnp.complex64)
    dim_map = {name: dim for name, dim, _, _ in irreps}

    # group the same (r,s) together
    Ms_by_pair   = defaultdict(list)
    idx_by_pair  = defaultdict(list)
    for (r_name, s_name, n_idx), M in Fhat.items():
        Ms_by_pair[(r_name, s_name)].append(jnp.asarray(M, dtype=jnp.complex64))
        idx_by_pair[(r_name, s_name)].append(n_idx)

    for (r_name, s_name), Ms_list in Ms_by_pair.items():
        d_r = dim_map[r_name]; d_s = dim_map[s_name]
        rho_r = jnp.asarray(rho_cache[r_name], dtype=jnp.complex64)  # (|G|, d_r, d_r)
        rho_s = jnp.asarray(rho_cache[s_name], dtype=jnp.complex64)  # (|G|, d_s, d_s)

        Ms = jnp.stack(Ms_list, axis=-1)           # (d_r, d_r, d_s, d_s, Nk)
        compN = _inv_block_many(Ms, rho_r, rho_s)  # (|G|, |G|, Nk)

        scale = d_r * d_s
        cols = jnp.asarray(idx_by_pair[(r_name, s_name)], dtype=jnp.int32)  # (Nk,)
        recon = recon.at[:, :, cols].add(scale * compN)

    return recon

