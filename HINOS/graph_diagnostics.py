import math
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from clustering_utils import compute_clustering_metrics, spectral_cluster_affinity, symmetrize_affinity


def label_array(labels_dict: Optional[Dict[int, int]], node_dim: int) -> Optional[np.ndarray]:
    if labels_dict is None:
        return None
    labels = np.empty((node_dim,), dtype=np.int64)
    for node_id in range(node_dim):
        if node_id not in labels_dict:
            return None
        labels[node_id] = int(labels_dict[node_id])
    return labels


def hard_partition_ncut(W: csr_matrix, partition: np.ndarray) -> float:
    W = W.tocsr()
    degree = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)
    eps = 1e-12
    ncut = 0.0
    for cluster_id in np.unique(partition):
        in_cluster = partition == cluster_id
        volume = float(degree[in_cluster].sum())
        if volume <= eps:
            continue
        internal_assoc = 0.0
        for node_id in np.flatnonzero(in_cluster):
            start, end = W.indptr[node_id], W.indptr[node_id + 1]
            neigh = W.indices[start:end]
            weights = W.data[start:end]
            internal_assoc += float(weights[in_cluster[neigh]].sum())
        cut = max(0.0, volume - internal_assoc)
        ncut += cut / (volume + eps)
    return float(ncut)


def purity_at_k(W: csr_matrix, labels: np.ndarray, k: int = 5) -> float:
    W = W.tocsr()
    total = 0.0
    for node_idx in range(W.shape[0]):
        start, end = W.indptr[node_idx], W.indptr[node_idx + 1]
        neigh = W.indices[start:end]
        weights = W.data[start:end]
        valid = neigh != node_idx
        neigh = neigh[valid]
        weights = weights[valid]
        if neigh.size == 0:
            continue
        if neigh.size > k:
            top_pos = np.argpartition(weights, -k)[-k:]
            order = np.argsort(weights[top_pos])[::-1]
            neigh = neigh[top_pos[order]]
        total += float(np.mean(labels[neigh] == labels[node_idx]))
    return total / float(max(1, W.shape[0]))


def weighted_same_partition_ratio(W: csr_matrix, partition: np.ndarray) -> Dict[str, float]:
    W = W.tocsr()
    same_mass = 0.0
    cross_mass = 0.0
    total_mass = 0.0
    for node_id in range(partition.shape[0]):
        start, end = W.indptr[node_id], W.indptr[node_id + 1]
        neigh = W.indices[start:end]
        weights = W.data[start:end].astype(np.float64, copy=False)
        keep = neigh != node_id
        neigh = neigh[keep]
        weights = weights[keep]
        if neigh.size == 0:
            continue
        same = partition[neigh] == partition[node_id]
        same_mass += float(weights[same].sum())
        cross_mass += float(weights[~same].sum())
        total_mass += float(weights.sum())
    ratio = same_mass / total_mass if total_mass > 0 else float("nan")
    return {
        "same_mass": same_mass,
        "cross_mass": cross_mass,
        "total_mass": total_mass,
        "same_partition_mass_ratio": ratio,
        "cross_partition_mass_ratio": 1.0 - ratio if not math.isnan(ratio) else float("nan"),
    }


# Skip eigsh on graphs larger than this; eigsh remains tractable even with which="LA"
# but on million-scale graphs every eval can still take minutes which we want to avoid.
SPEC_GAP_MAX_NODES = 500_000


def normalized_laplacian_spectrum_diag(
    W: csr_matrix,
    num_clusters: int,
    max_nodes: int = SPEC_GAP_MAX_NODES,
) -> Dict[str, float]:
    nan_result = {"spec_gap_c": float("nan"), "spec_gap_c1": float("nan"), "spec_gap_ratio": float("nan")}
    W = symmetrize_affinity(W.tocsr().astype(np.float64))
    n = W.shape[0]
    if n <= 1 or num_clusters <= 0:
        return nan_result
    if n > max_nodes:
        # Skip very large graphs: even eigsh with which="LA" can take many minutes per call.
        return nan_result
    degree = np.asarray(W.sum(axis=1)).ravel()
    degree = np.maximum(degree, 1e-12)
    inv_sqrt_degree = 1.0 / np.sqrt(degree)
    # Reformulate as eigenproblem on M = D^{-1/2} W D^{-1/2}, where L = I - M.
    # The k smallest eigenvalues of L correspond to the k largest eigenvalues of M.
    # ARPACK's which="LA" converges fast on sparse symmetric matrices, while which="SM"
    # without shift-invert is the worst case and can take hours on 28k+ node graphs.
    M = (sp.diags(inv_sqrt_degree) @ W @ sp.diags(inv_sqrt_degree)).tocsr()
    k = min(num_clusters + 1, n if n <= 512 else n - 1)
    try:
        if n <= 512:
            # Small graphs: dense path is cheap and reliable.
            M_dense = M.toarray()
            mu = np.linalg.eigvalsh(M_dense)
            mu_top = np.sort(mu)[-k:]
        else:
            mu_top = eigsh(M, k=k, which="LA", return_eigenvectors=False)
        evals = np.sort(np.maximum(1.0 - mu_top, 0.0))
    except Exception:
        return nan_result
    if evals.size < k:
        return nan_result
    lambda_c = float(evals[num_clusters - 1])
    lambda_c1 = float(evals[num_clusters]) if evals.size > num_clusters else float("nan")
    return {
        "spec_gap_c": lambda_c,
        "spec_gap_c1": lambda_c1,
        "spec_gap_ratio": lambda_c / (lambda_c1 + 1e-12) if not math.isnan(lambda_c1) else float("nan"),
    }


def diagnose_affinity(
    W: csr_matrix,
    labels: Optional[np.ndarray],
    num_clusters: int,
    pred: Optional[np.ndarray] = None,
    random_state: int = 42,
    topk: Optional[int] = None,
    prefix: str = "",
    compute_spectral: bool = True,
    compute_gap: bool = True,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    key = f"_{prefix}" if prefix else ""
    W = W.tocsr()
    spectral_pred = None
    if compute_spectral:
        spectral_pred = spectral_cluster_affinity(W, num_clusters, random_state=random_state, topk=topk)
    if labels is not None:
        out[f"purity_at_5{key}"] = purity_at_k(W, labels, 5)
        out[f"ncut_gt{key}"] = hard_partition_ncut(W, labels)
        if spectral_pred is not None:
            metrics = compute_clustering_metrics(labels, spectral_pred)
            out[f"acc_spectral{key}"] = metrics["ACC"]
            out[f"nmi_spectral{key}"] = metrics["NMI"]
            out[f"ari_spectral{key}"] = metrics["ARI"]
            out[f"f1_spectral{key}"] = metrics["F1"]
            out[f"ncut_pred{key}"] = hard_partition_ncut(W, spectral_pred)
        if pred is not None:
            out[f"ncut_q{key}"] = hard_partition_ncut(W, pred)
    if f"ncut_pred{key}" in out and f"ncut_gt{key}" in out:
        out[f"ncut_pred_over_gt{key}"] = out[f"ncut_pred{key}"] / (out[f"ncut_gt{key}"] + 1e-12)
    if compute_gap:
        out.update(normalized_laplacian_spectrum_diag(W, num_clusters))
    return out
