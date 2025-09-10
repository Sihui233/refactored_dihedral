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
@jax.jit
def _einsum_block(f_grid, rho_r_dag, rho_s_dag, group_size):
    
    M = jnp.einsum('aij,abn,bkl->ijkln', rho_r_dag, f_grid, rho_s_dag)
    # normalized by |G|^2
    gsq = group_size**2 # f_grid.shape[0] * f_grid.shape[0]
    return M / gsq

def _group_dft_preacts_inner_nojit(preacts, rho_cache, irreps, group_size):
    f_grid = preacts.reshape(group_size, group_size, -1)  # (|G|,|G|,N)
    Fhat = {}
    for r_name, d_r, _, _ in irreps:
        rho_r_dag = rho_cache[r_name].conj().transpose(0, 2, 1)  # (|G|, d_r, d_r)
        for s_name, d_s, _, _ in irreps:
            rho_s_dag = rho_cache[s_name].conj().transpose(0, 2, 1)  # (|G|, d_s, d_s)
            
            M_all = _einsum_block(f_grid, rho_r_dag, rho_s_dag,group_size)
            for n_idx in range(M_all.shape[-1]):
                Fhat[(r_name, s_name, n_idx)] = M_all[..., n_idx]
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

