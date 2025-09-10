import numpy as np
from ripser import ripser          # pip install ripser
from scipy.spatial import cKDTree  # pip install scipy
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from persim import plot_diagrams
import matplotlib.pyplot as plt
import os
from typing import Dict
import json

_BACKEND = "ripser.py"

def _build_sparse_knn_matrix(points: np.ndarray, n_nbrs: int) -> "csr_matrix":
    """
    Build a symmetric sparse k-NN distance matrix (CSR) for `points`.
    """
    N = points.shape[0]
    n_nbrs = min(n_nbrs, (N)/4)          # ← cap the request
    tree = cKDTree(points)
    dists, idxs = tree.query(points, k=n_nbrs + 1)

    D = lil_matrix((N, N), dtype=np.float32)
    for i in range(N):
        for j, d in zip(idxs[i, 1:], dists[i, 1:]):   # skip self
            # cKDTree pads with N when k > N; double-check anyway
            if j >= N or not np.isfinite(d):
                continue
            D[i, j] = d
            D[j, i] = d
    return D.tocsr()


def _ph(points: np.ndarray,
        *, maxdim: int,
        sparse: bool = True,
        n_nbrs: int = 50) -> list[np.ndarray]:
    """
    Compute persistence diagrams with Ripser-py.

    If `sparse` is True, build a k-NN sparse distance matrix first.
    Returns a list of diagrams per dimension.
    """
    if not sparse:
        return ripser(points,
                      maxdim=maxdim,
                      metric="euclidean",
                      distance_matrix=False)['dgms']

    D = _build_sparse_knn_matrix(points, n_nbrs)
    return ripser(D,
                  maxdim=maxdim,
                  distance_matrix=True)['dgms']


def _long_bar_mask(dgm: np.ndarray, frac: float) -> np.ndarray:
    """
    Bar-length mask for H>=2 using a relative fraction heuristic.
    """
    if dgm.size == 0:
        return np.zeros(0, dtype=bool)
    length = np.where(np.isfinite(dgm[:, 1]), dgm[:, 1], dgm[:, 0]) - dgm[:, 0]
    return length >= frac * length.max()


def run_ph_for_point_cloud(points: np.ndarray,
                           *,
                           maxdim: int = 2,
                           ph_sparse: bool = True,
                           n_nbrs: int = 150,
                           long_bar_frac: float = 0.96,
                           use_dims: int | None = None,
                           save_dir: str | None = None,
                           filename_stem: str = "ph",
                           title: str | None = None) -> Dict[str, int]:
    """
    Compute persistent homology with Ripser-py and (optionally) save:
      • a PDF persistence diagram   <save_dir>/<stem>.pdf
      • a JSON Betti-numbers file   <save_dir>/<stem>.json

    Features:
      - `use_dims`: if set, project `points` onto the first `use_dims` coords before PH.
      - MST-gap heuristic for b0 on sparse graphs.
      - Infinite-bar + gap heuristic for b1.
      - Relative bar-length heuristic for b>=2.
    """
    # Optionally project to lower-dimensional subspace
    if use_dims is not None:
        if use_dims > 0 and use_dims <= points.shape[1]:
            points = points[:, :use_dims]
        else:
            raise ValueError(f"use_dims={use_dims} must be between 1 and {points.shape[1]}")

    # 1) Compute diagrams
    dgms = _ph(points, maxdim=maxdim, sparse=ph_sparse, n_nbrs=n_nbrs)

    # 2) Compute b0 via MST gap on the sparse graph
    if ph_sparse:
        D = _build_sparse_knn_matrix(points, n_nbrs)
        mst = minimum_spanning_tree(D).tocsr()
        weights = mst.data
        if weights.size > 1:
            sorted_w = np.sort(weights)
            gaps = np.diff(sorted_w)
            idx = np.argmax(gaps)
            eps_star = 0.5 * (sorted_w[idx] + sorted_w[idx + 1])
            G = mst.copy()
            G.data = np.where(G.data <= eps_star, G.data, 0)
            b0, _ = connected_components(G, directed=False)
        else:
            b0 = 1
    else:
        b0 = 1

    betti = {'b0': b0}
    for dim, dgm in enumerate(dgms[1:], start=1):
        # count only the bars that never die
        betti[f'b{dim}'] = int(np.isinf(dgm[:, 1]).sum())

    # 5) Save artifacts if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots()
        plot_diagrams(dgms, ax=ax, show=False)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        pdf_path = os.path.join(save_dir, f"{filename_stem}.pdf")
        fig.savefig(pdf_path, format="pdf")
        plt.close(fig)
        json_path = os.path.join(save_dir, f"{filename_stem}.json")
        with open(json_path, "w") as fp:
            json.dump(betti, fp, indent=2)
        print(f"[PH] diagram  → {pdf_path}")
        print(f"[PH] Betti    → {json_path}")

    return betti
