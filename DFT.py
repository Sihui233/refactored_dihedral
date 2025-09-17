import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax
from jax import config
config.update("jax_default_matmul_precision", "high")  # 或 "highest"

import jax.numpy as jnp
from typing import Dict, Tuple
from functools import partial
import numpy as np

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

# @partial(jax.jit, static_argnums=(3,))
# def _einsum_block(f_grid, rho_r_dag, rho_s_dag, group_size):
#     """
#     f_grid:    (|G|, |G|, N)
#     rho_r_dag: (|G|, d_r, d_r)
#     rho_s_dag: (|G|, d_s, d_s)
#     return:    (d_r, d_r, d_s, d_s, N)
#     """
#     # step1: sum over 'a' -> (d_r, d_r, |G|, N)
#     #   'aij,abn->ijbn'
#     tmp = jnp.einsum('aij,abn->ijbn', rho_r_dag, f_grid)

#     # step2: sum over 'b' -> (d_r, d_r, d_s, d_s, N)
#     #   'bkl,ijbn->ijkln'
#     M = jnp.einsum('bkl,ijbn->ijkln', rho_s_dag, tmp)

#     inv_gsq = jnp.asarray(1.0, dtype=M.dtype) / (group_size * group_size)
#     return M * inv_gsq

# ## normal GFT
# @partial(jax.jit, static_argnums=(3,))
# def _einsum_block(f_grid, rho_r_dag, rho_s_dag, group_size):
#     """
#     f_grid:    (|G|, |G|, N)
#     rho_r_dag: (|G|, d_r, d_r)
#     rho_s_dag: (|G|, d_s, d_s)
#     return:    (d_r, d_r, d_s, d_s, N)
#     """
#     # --- 动态选择计算 dtype：复数→complex64，实数→float32 ---
#     need_complex = (
#         jnp.issubdtype(f_grid.dtype, jnp.complexfloating) or
#         jnp.issubdtype(rho_r_dag.dtype, jnp.complexfloating) or
#         jnp.issubdtype(rho_s_dag.dtype, jnp.complexfloating)
#     )
#     ctype = jnp.complex64 if need_complex else jnp.float32

#     G   = group_size
#     d_r = int(rho_r_dag.shape[1])
#     d_s = int(rho_s_dag.shape[1])
#     B   = int(f_grid.shape[-1])

#     f   = f_grid.astype(ctype)
#     Ar  = rho_r_dag.astype(ctype)
#     Bs  = rho_s_dag.astype(ctype)

#     # 展平到 GEMM 友好格式（两次 matmul 更快）
#     A_T    = Ar.reshape(G, d_r * d_r).T      # (d_r^2, G)
#     B_flat = Bs.reshape(G, d_s * d_s)        # (G, d_s^2)
#     F      = jnp.moveaxis(f, 2, 0)           # (B, G, G)

#     # 自适应更省 FLOPs 的乘法顺序
#     def path_AT_first(F_):
#         Z = jnp.matmul(A_T[None, ...], F_)   # (B, d_r^2, G)
#         return jnp.matmul(Z, B_flat)         # (B, d_r^2, d_s^2)
#     def path_B_first(F_):
#         Z = jnp.matmul(F_, B_flat)           # (B, G, d_s^2)
#         return jnp.matmul(A_T[None, ...], Z) # (B, d_r^2, d_s^2)

#     Mflat = jax.lax.cond(d_r <= d_s, path_AT_first, path_B_first, F)

#     # 归一化 + 还原形状
#     inv_gsq = jnp.asarray(1.0, dtype=ctype) / (G * G)
#     Mflat = jnp.moveaxis(Mflat * inv_gsq, 0, -1)    # (d_r^2, d_s^2, B)
#     return Mflat.reshape(d_r, d_r, d_s, d_s, B)

# # def _group_dft_preacts_inner_nojit(preacts, rho_cache, irreps, group_size):
# #     f_grid = jnp.asarray(preacts).reshape(group_size, group_size, -1)  # (|G|,|G|,N)
# #     Fhat = {}
# #     rho_dag_cache = {
# #         name: jnp.asarray(arr).conj().swapaxes(-1, -2)  # 放在 device 上
# #         for name, arr in rho_cache.items()
# #     }
# #     for r_name, d_r, _, _ in irreps:
# #             rho_r_dag = rho_dag_cache[r_name]  # (|G|, d_r, d_r)
# #             for s_name, d_s, _, _ in irreps:
# #                 rho_s_dag = rho_dag_cache[s_name]  # (|G|, d_s, d_s)
                
# #                 M_all = _einsum_block(f_grid, rho_r_dag, rho_s_dag, group_size)
                
# #                 for n_idx in range(M_all.shape[-1]):
# #                     Fhat[(r_name, s_name, n_idx)] = M_all[..., n_idx]
# #     return Fhat
# def _group_dft_preacts_inner_nojit(preacts, rho_cache, irreps, group_size):
#     """
#     Batched along nto prevent materializing (d_r,d_r,d_s,d_s,N)。
#     """
#     f_grid = jnp.asarray(preacts).reshape(group_size, group_size, -1)  # (G,G,N)
#     N = int(f_grid.shape[-1])

#     # 预处理 ρ^† 并确保是 f32（双精会把内存翻倍）
#     rho_dag_cache = {}
#     for name, arr in rho_cache.items():
#         arr = jnp.asarray(arr)
#         if jnp.issubdtype(arr.dtype, jnp.complexfloating):
#             arr = arr.astype(jnp.complex64)   # 复数 → complex64
#         else:
#             arr = arr.astype(jnp.float32)     # 实数 → float32
#         rho_dag_cache[name] = arr.conj().swapaxes(-1, -2)

#     Fhat = {}

#     # 简单的自适应 chunk 估计：控制中间 tmp/M 的体积都不超过 ~500MB
#     TARGET_BYTES = 500 * 1024 * 1024

#     for r_name, d_r, _, _ in irreps:
#         rho_r_dag = rho_dag_cache[r_name]  # (G,d_r,d_r)
#         for s_name, d_s, _, _ in irreps:
#             rho_s_dag = rho_dag_cache[s_name]  # (G,d_s,d_s)

#             # —— 估算每样本内存（按实际 dtype 区分复/实）——
#             elem_bytes = 8 if (
#                 jnp.issubdtype(f_grid.dtype, jnp.complexfloating) or
#                 jnp.issubdtype(rho_r_dag.dtype, jnp.complexfloating) or
#                 jnp.issubdtype(rho_s_dag.dtype, jnp.complexfloating)
#             ) else 4

#             # 输出/中间的保守估算（足够决定一个稳妥的 B）
#             bytes_per_sample_out = d_r * d_r * d_s * d_s * elem_bytes
#             bytes_per_sample_tmp = group_size * d_s * d_s * elem_bytes
#             bytes_per_sample = max(bytes_per_sample_out, bytes_per_sample_tmp)

#             if bytes_per_sample == 0:
#                 B = N
#             else:
#                 B = max(1, min(N, int(TARGET_BYTES // bytes_per_sample)))

#             # 为了避免多次编译，固定 chunk 大小 B（只有最后一块可能触发第二个编译）
#             start = 0
#             while start < N:
#                 stop = min(start + B, N)
#                 # (G,G,B)
#                 f_chunk = jax.lax.dynamic_slice_in_dim(f_grid, start, stop - start, axis=2)
#                 # (d_r,d_r,d_s,d_s,B)
#                 M_chunk = _einsum_block(f_chunk, rho_r_dag, rho_s_dag, group_size)

#                 # 立刻把每个 n 的结果搬到 CPU，释放 GPU 显存压力
#                 # 注意：这段不在 jit 里，device_get 会同步该块计算并回拷
#                 M_host = jax.device_get(M_chunk)  # numpy array on host

#                 B_eff = M_host.shape[-1]
#                 for bi in range(B_eff):
#                     n_idx = start + bi
#                     # 存 CPU 端的 ndarray：后续 Python 分析依然可用
#                     Fhat[(r_name, s_name, int(n_idx))] = M_host[..., bi]

#                 start = stop

#     return Fhat


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

#### fast GFT
def _A_from(z):
    """Map z -> R(-θ) 对应的 2x2 实矩阵"""
    return jnp.stack(
        [jnp.stack([ z.real, -z.imag], axis=-1),
         jnp.stack([ z.imag,  z.real], axis=-1)],
        axis=-2
    )  # (..., 2, 2)

def _B_from(z):
    """Map z -> R(-θ)S 对应的 2x2 实矩阵"""
    return jnp.stack(
        [jnp.stack([ z.real,  z.imag], axis=-1),
         jnp.stack([ z.imag, -z.real], axis=-1)],
        axis=-2
    )  # (..., 2, 2)
def _A_from_pair(a_k, a_nk):
    # sum_m f_m R(-mθ) with θ=2πk/n
    # c = ∑ f_m cos(mθ), s = ∑ f_m sin(mθ)
    # Using FFT bins: c = (a_k + a_nk)/2,   s = (a_k - a_nk)/(2i) = -0.5j*(a_k - a_nk)
    c = 0.5 * (a_k + a_nk)
    s = -0.5j * (a_k - a_nk)
    return jnp.stack(
        [jnp.stack([c, -s], axis=-1),
         jnp.stack([s,  c], axis=-1)],
        axis=-2
    )  # (..., 2, 2)

def _B_from_pair(b_k, b_nk):
    # sum_m f_ref_m R(-mθ) S  ⇒ [[c, s], [s, -c]] with same (c,s) from reflection FFT bins
    c = 0.5 * (b_k + b_nk)
    s = -0.5j * (b_k - b_nk)
    return jnp.stack(
        [jnp.stack([c,  s], axis=-1),
         jnp.stack([s, -c], axis=-1)],
        axis=-2
    )  # (..., 2, 2)

# ---------- 1) 构建静态元信息（一次） ----------
def _fft_meta_from_irreps(irreps, group_size: int):
    n = group_size // 2
    even = (n % 2 == 0)
    ks   = tuple(range(1, (n - 1) // 2 + 1))  # 2D 的 k 列表
    oned_names = ['triv', 'sign'] + (['rp', 'srp'] if even else [])
    twod_names = [f'2D_{k}' for k in ks]

    name2idx_1d = {nm: i for i, nm in enumerate(oned_names)}
    name2idx_2d = {nm: i for i, nm in enumerate(twod_names)}
    return {
        'n': n, 'even': even, 'ks': ks,
        'oned_names': oned_names, 'twod_names': twod_names,
        'name2idx_1d': name2idx_1d, 'name2idx_2d': name2idx_2d,
    }

# ---------- 2) stage-1：沿 axis=0 的 dihedral FFT（对所有 r 一次性算好） ----------
from functools import partial

@partial(jax.jit, static_argnums=(1,2,3))
def _fft_stage1_all_r(f_blk_cplx: jnp.ndarray, n: int, even: bool, ks: tuple):
    """
    输入: f_blk_cplx: (G, G, Bc) complex64，G=2n
    输出:
      Y1d_all: (n1d, G, Bc)     对应 r∈{triv,sign,(rp,srp)}
      Y2d_all: (n2d, 2,2, G,Bc) 对应 r∈{2D_k}
    """
    G, _, Bc = f_blk_cplx.shape
    # 切 rot/ref, 对长度 n 的轴做 FFT
    f0_rot = f_blk_cplx[:n, :, :]         # (n, G, Bc)
    f0_ref = f_blk_cplx[n:, :, :]         # (n, G, Bc)
    F0_rot = jnp.fft.fft(f0_rot, axis=0)  # (n, G, Bc)
    F0_ref = jnp.fft.fft(f0_ref, axis=0)  # (n, G, Bc)

    k0 = 0
    # 1D r（固定次序：triv, sign, (rp, srp)）
    acc_1d = []
    triv = F0_rot[k0, :, :] + F0_ref[k0, :, :]
    sign = F0_rot[k0, :, :] - F0_ref[k0, :, :]
    acc_1d += [triv, sign]
    if even:
        kH = n // 2
        rp  = F0_rot[kH, :, :] + F0_ref[kH, :, :]
        srp = F0_rot[kH, :, :] - F0_ref[kH, :, :]
        acc_1d += [rp, srp]
    Y1d_all = jnp.stack(acc_1d, axis=0) if acc_1d else jnp.zeros((0, G, Bc), dtype=f_blk_cplx.dtype)

    # 2D r：用 (k, n-k) 成对合成实 2×2 块
    acc_2d = []
    for k in ks:
        kc = (n - k) % n
        alpha_k, alpha_ck = F0_rot[k, :, :],  F0_rot[kc, :, :]
        beta_k,  beta_ck  = F0_ref[k, :, :],  F0_ref[kc, :, :]
        block = _A_from_pair(alpha_k, alpha_ck) + _B_from_pair(beta_k, beta_ck)   # (G,Bc,2,2)
        block = jnp.transpose(block, (2, 3, 0, 1))  # -> (2,2,G,Bc)
        acc_2d.append(block)
    Y2d_all = jnp.stack(acc_2d, axis=0) if acc_2d else jnp.zeros((0, 2, 2, G, Bc), dtype=f_blk_cplx.dtype)

    return Y1d_all, Y2d_all

# ---------- 3) stage-2：固定 r，沿 axis=1 的 dihedral FFT，得到所有 s 的块 ----------
@partial(jax.jit, static_argnums=(1,2,3))
def _fft_stage2_for_r(Yr: jnp.ndarray, n: int, even: bool, ks: tuple):
    """
    输入: Yr: (d_r, d_r, G, Bc) complex64
    输出:
      S1d: (n1d, d_r, d_r, Bc)             对应 s∈{triv,sign,(rp,srp)}
      S2d: (n2d, d_r, d_r, 2, 2, Bc)       对应 s∈{2D_k}
    """
    # split rot/ref, FFT 长度 n
    Yr_rot = Yr[:, :, :n, :]                  # (d_r,d_r,n,Bc)
    Yr_ref = Yr[:, :,  n:, :]                 # (d_r,d_r,n,Bc)
    F1_rot = jnp.fft.fft(Yr_rot, axis=2)      # (d_r,d_r,n,Bc)
    F1_ref = jnp.fft.fft(Yr_ref, axis=2)      # (d_r,d_r,n,Bc)

    k0 = 0
    acc_1d = []
    # 1D s
    triv = F1_rot[:, :, k0, :] + F1_ref[:, :, k0, :]
    sign = F1_rot[:, :, k0, :] - F1_ref[:, :, k0, :]
    acc_1d += [triv, sign]
    if even:
        kH = n // 2
        rp  = F1_rot[:, :, kH, :] + F1_ref[:, :, kH, :]
        srp = F1_rot[:, :, kH, :] - F1_ref[:, :, kH, :]
        acc_1d += [rp, srp]
    S1d = jnp.stack(acc_1d, axis=0) if acc_1d else jnp.zeros((0, Yr.shape[0], Yr.shape[1], Yr.shape[-1]), dtype=Yr.dtype)

    # 2D s
    acc_2d = []
    for k in ks:
        kc = (n - k) % n
        a_k,  a_ck = F1_rot[:, :, k, :],   F1_rot[:, :, kc, :]
        b_k,  b_ck = F1_ref[:, :, k, :],   F1_ref[:, :, kc, :]
        block = _A_from_pair(a_k, a_ck) + _B_from_pair(b_k, b_ck)   # (d_r,d_r,Bc,2,2)
        block = jnp.moveaxis(block, -3, -1)                         # -> (d_r,d_r,2,2,Bc)
        acc_2d.append(block)
    S2d = jnp.stack(acc_2d, axis=0) if acc_2d else jnp.zeros((0, Yr.shape[0], Yr.shape[1], 2, 2, Yr.shape[-1]), dtype=Yr.dtype)
    return S1d, S2d

def _group_dft_preacts_inner_nojit(preacts, rho_cache, irreps, group_size):
    """
    Fast Dihedral Product FFT (axis-0 then axis-1), chunked over N.
    Keeps API & output dict exactly as before:
      returns Fhat[(r_name, s_name, n_idx)] = (d_r, d_r, d_s, d_s) ndarray.
    """
    # Shapes and dtypes
    G  = int(group_size)                 # |D_n| = 2n
    n  = G // 2                          # length of cyclic part
    assert 2 * n == G, "group_size must be even for D_n"

    f_grid = jnp.asarray(preacts).reshape(G, G, -1)  # (G, G, N)
    N      = int(f_grid.shape[-1])

    # Decide chunk size B (conservative, works for real/complex)
    TARGET_BYTES = 500 * 1024 * 1024
    elem_bytes = 8 if jnp.issubdtype(f_grid.dtype, jnp.complexfloating) else 4
    # Peak per-sample (very rough upper bound): two FFT working sets + small blocks
    bytes_per_sample = (G * G + 2 * n * G + 64) * elem_bytes
    B = max(1, min(N, int(TARGET_BYTES // max(1, bytes_per_sample))))

    # Build quick maps for (name -> dim, meta_k)
    # meta_k = k for 2D_k irreps; for 1D, we’ll use the names directly
    name_to_dim  = {}
    name_to_k    = {}
    for name, dim, _R, meta in irreps:
        name_to_dim[name] = dim
        name_to_k[name]   = meta  # for '2D_k' this is k; for 1D irreps it's not used

    # Enumerate all irrep names by type
    oned_names = [name for name, dim, *_ in irreps if dim == 1]
    twod_names = [name for name, dim, *_ in irreps if dim == 2]

    # Helpers for 1D along a given FFT output
    # Xrot/Xref have shape (..., n, G_or_B) and we index freq k → complex slice (..., G_or_B)
    def _oned_pick(name, Xrot_k, Xref_k):
        if name == 'triv':   return Xrot_k + Xref_k
        if name == 'sign':   return Xrot_k - Xref_k
        if name == 'rp':     return Xrot_k + Xref_k
        if name == 'srp':    return Xref_k * (-1.0) + Xrot_k
        # Fallback (should not happen)
        return Xrot_k + Xref_k

    Fhat = {}

    meta = _fft_meta_from_irreps(irreps, group_size)
    oned_names = meta['oned_names']; twod_names = meta['twod_names']
    name2idx_1d = meta['name2idx_1d']; name2idx_2d = meta['name2idx_2d']
    n = meta['n']; even = meta['even']; ks = meta['ks']
    inv_gsq = jnp.asarray(1.0, dtype=jnp.complex64) / (group_size * group_size)

    start = 0
    while start < N:
        curB  = min(B, N - start)
        f_blk = jax.lax.dynamic_slice_in_dim(f_grid, start, curB, axis=2).astype(jnp.complex64)  # (G,G,curB)

        # === JIT 核 1：axis-0 FFT，得到所有 r 的 Y ===
        Y1d_all, Y2d_all = _fft_stage1_all_r(f_blk, n, even, ks)   # (n1d,G,Bc), (n2d,2,2,G,Bc)

        # 逐个 r（顺序跟 irreps 保持一致，以便 key 保持不变）
        for r_name, d_r, _Rr, _ in irreps:
            # 取 Yr: (d_r,d_r,G,Bc)
            if d_r == 1:
                ridx = name2idx_1d[r_name]
                Yr = Y1d_all[ridx][None, None, :, :]          # (1,1,G,Bc)
            else:
                ridx = name2idx_2d[r_name]
                Yr = Y2d_all[ridx]                             # (2,2,G,Bc)

            # === JIT 核 2：固定 r，axis-1 FFT，得到所有 s 的块 ===
            S1d, S2d = _fft_stage2_for_r(Yr, n, even, ks)     # (n1d,d_r,d_r,Bc), (n2d,d_r,d_r,2,2,Bc)

            # 写回 dict：1D s
            for s_name in oned_names:
                sidx = name2idx_1d[s_name]
                Z = (S1d[sidx] * inv_gsq).astype(jnp.complex64)     # (d_r,d_r,Bc)
                Z = Z[..., None, None, :]                           # -> (d_r,d_r,1,1,Bc)
                for bi in range(curB):
                    Fhat[(r_name, s_name, int(start + bi))] = Z[..., bi]

            # 写回 dict：2D s
            for s_name in twod_names:
                sidx = name2idx_2d[s_name]
                right_block = (S2d[sidx] * inv_gsq).astype(jnp.complex64)  # (d_r,d_r,2,2,Bc)
                for bi in range(curB):
                    Fhat[(r_name, s_name, int(start + bi))] = right_block[..., bi]

        start += curB


    return Fhat

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

