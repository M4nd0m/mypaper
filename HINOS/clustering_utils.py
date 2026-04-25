import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score


def read_node_labels(path: str) -> Dict[int, int]:
    labels = {}
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            labels[int(parts[0])] = int(parts[1])
    if not labels:
        raise ValueError(f"No labels were read from {path}")
    return labels


def align_labels(gt: Dict[int, int], pred: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    common_nodes = sorted(set(gt.keys()) & set(pred.keys()))
    if not common_nodes:
        raise ValueError("Ground-truth labels and predictions have no overlapping node ids.")
    y_true = np.asarray([gt[node_id] for node_id in common_nodes], dtype=np.int64)
    y_pred = np.asarray([pred[node_id] for node_id in common_nodes], dtype=np.int64)
    return y_true, y_pred


def match_pred_to_true(y_true: np.ndarray, y_pred: np.ndarray):
    true_classes, true_inverse = np.unique(y_true, return_inverse=True)
    pred_classes, pred_inverse = np.unique(y_pred, return_inverse=True)

    cost = np.zeros((pred_classes.size, true_classes.size), dtype=np.int64)
    for pred_idx, true_idx in zip(pred_inverse, true_inverse):
        cost[pred_idx, true_idx] += 1

    row_ind, col_ind = linear_sum_assignment(cost.max() - cost)
    mapping = {int(pred_classes[r]): int(true_classes[c]) for r, c in zip(row_ind, col_ind)}
    return mapping, cost, row_ind, col_ind


def remap_predictions(y_pred: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    next_unused = int(np.max(y_pred)) + int(np.max(list(mapping.values()) + [0])) + 1
    remapped = np.empty_like(y_pred)
    extra_mapping = {}
    for idx, pred_label in enumerate(y_pred.tolist()):
        if pred_label in mapping:
            remapped[idx] = mapping[pred_label]
            continue
        if pred_label not in extra_mapping:
            extra_mapping[pred_label] = next_unused
            next_unused += 1
        remapped[idx] = extra_mapping[pred_label]
    return remapped


def compute_clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mapping, cost, row_ind, col_ind = match_pred_to_true(y_true, y_pred)
    y_pred_aligned = remap_predictions(y_pred, mapping)
    acc = float(cost[row_ind, col_ind].sum()) / float(y_true.size)
    return {
        "ACC": acc,
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred_aligned, average="macro")),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
    }


def save_prediction_file(path: str, pred: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as writer:
        writer.write("node_id cluster_id\n")
        for node_id, cluster_id in enumerate(np.asarray(pred, dtype=np.int64).tolist()):
            writer.write(f"{node_id} {cluster_id}\n")


def kmeans_predict(features: np.ndarray, num_clusters: int, random_state: int) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    model = KMeans(n_clusters=num_clusters, n_init=10, random_state=random_state)
    return model.fit_predict(x).astype(np.int64)


def kmeans_predict_sparse_rows(features: csr_matrix, num_clusters: int, random_state: int) -> np.ndarray:
    model = MiniBatchKMeans(n_clusters=num_clusters, random_state=random_state, batch_size=2048, n_init=5)
    return model.fit_predict(features).astype(np.int64)


def symmetrize_affinity(mat: csr_matrix) -> csr_matrix:
    sym = 0.5 * (mat.tocsr().astype(np.float64) + mat.T.tocsr().astype(np.float64))
    sym.sum_duplicates()
    sym.setdiag(0.0)
    sym.eliminate_zeros()
    return sym.tocsr()


def topk_sparsify_rows(mat: csr_matrix, topk: int) -> csr_matrix:
    if topk <= 0:
        return mat.tocsr()

    mat = mat.tocsr()
    rows = []
    cols = []
    vals = []
    for row in range(mat.shape[0]):
        start, end = mat.indptr[row], mat.indptr[row + 1]
        if end <= start:
            continue
        row_indices = mat.indices[start:end]
        row_values = mat.data[start:end]
        if row_values.size > topk:
            keep = np.argpartition(row_values, -topk)[-topk:]
            keep = keep[np.argsort(-row_values[keep])]
            row_indices = row_indices[keep]
            row_values = row_values[keep]
        rows.extend([row] * row_values.size)
        cols.extend(row_indices.tolist())
        vals.extend(row_values.tolist())
    out = sp.coo_matrix((vals, (rows, cols)), shape=mat.shape, dtype=np.float64).tocsr()
    out.sum_duplicates()
    out.eliminate_zeros()
    return out


def _normalized_spectral_embedding(affinity: csr_matrix, num_clusters: int) -> np.ndarray:
    n = affinity.shape[0]
    if n == 0:
        raise ValueError("Empty affinity matrix.")
    if n == 1 or num_clusters <= 1:
        return np.ones((n, 1), dtype=np.float64)

    degree = np.asarray(affinity.sum(axis=1)).ravel()
    degree = np.maximum(degree, 1e-12)
    inv_sqrt_degree = 1.0 / np.sqrt(degree)
    norm_affinity = sp.diags(inv_sqrt_degree) @ affinity @ sp.diags(inv_sqrt_degree)

    k = min(max(1, num_clusters), n - 1)
    if n <= 512:
        warnings.warn(
            f"Spectral embedding densifies the affinity matrix for small graphs (n={n}). "
            "This is restricted to small datasets to avoid memory risk on large graphs.",
            RuntimeWarning,
            stacklevel=2,
        )
        dense = norm_affinity.toarray()
        _, eigvecs = np.linalg.eigh(dense)
        embed = eigvecs[:, -k:]
    else:
        _, eigvecs = eigsh(norm_affinity, k=k, which="LA")
        embed = eigvecs

    norms = np.linalg.norm(embed, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embed / norms


def spectral_cluster_affinity(
    affinity: csr_matrix,
    num_clusters: int,
    random_state: int,
    topk: Optional[int] = None,
) -> np.ndarray:
    sym = symmetrize_affinity(affinity)
    if topk is not None and topk > 0:
        sym = topk_sparsify_rows(sym, topk)
        sym = symmetrize_affinity(sym)
    embed = _normalized_spectral_embedding(sym, num_clusters)
    return kmeans_predict(embed.astype(np.float32), num_clusters, random_state=random_state)


def metrics_with_prefix(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{name.lower()}_{prefix}": float(value) for name, value in metrics.items()}
