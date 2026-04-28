import argparse
import math
import os
from typing import Dict, Optional

import numpy as np

from clustering_utils import compute_clustering_metrics, read_node_labels, spectral_cluster_affinity
from data_load import compress_time_indices, read_temporal_edges
from sparsification import build_ncut_graph, compute_tppr_cached
from utils import resolve_path, set_random_seed


def get_args():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Diagnose TPPR/TAPS graph quality without training.")
    parser.add_argument("--dataset", type=str, default="dblp", help="dataset name under data_root")
    parser.add_argument("--data_root", type=str, default=os.path.join(cur_dir, "dataset"), help="dataset root")
    parser.add_argument("--cache_dir", type=str, default=os.path.join(cur_dir, "cache"), help="TPPR/TAPS cache root")
    parser.add_argument("--tppr_alpha", type=float, default=0.2, help="TPPR alpha")
    parser.add_argument("--tppr_K", type=int, default=3, help="TPPR truncation order")
    parser.add_argument("--taps_alpha", type=float, default=0.2, help="TAPS alpha")
    parser.add_argument("--taps_tau_eps", type=float, default=1.0, help="TAPS temporal smoothing tau_eps")
    parser.add_argument("--taps_rng_seed", type=int, default=42, help="TAPS RNG seed")
    parser.add_argument("--taps_T_cap", type=int, default=10, help="maximum TAPS time step cap; 0 means no cap")
    parser.add_argument(
        "--taps_budget_mode",
        type=str,
        default="nlogn",
        choices=["sqrt_edges", "nlogn"],
        help="TAPS sampling budget mode",
    )
    parser.add_argument("--taps_budget_beta", type=float, default=0.05, help="TAPS nlogn budget scale")
    parser.add_argument("--spectral_topk", type=int, default=20, help="top-k sparsification for spectral diagnostics")
    parser.add_argument("--seed", type=int, default=42, help="random seed for spectral k-means")
    return parser.parse_args()


def infer_trainer_node_dim(edges_uvt) -> int:
    node_set = set()
    for u, v, _ in edges_uvt:
        node_set.update([int(u), int(v)])
    return len(node_set)


def label_array_or_none(labels: Dict[int, int], node_dim: int) -> Optional[np.ndarray]:
    y = np.empty((node_dim,), dtype=np.int64)
    for node_idx in range(node_dim):
        if node_idx not in labels:
            return None
        y[node_idx] = int(labels[node_idx])
    return y


def purity_at_k(W, labels: np.ndarray, k: int) -> float:
    W = W.tocsr()
    total = 0.0
    node_dim = labels.shape[0]
    for node_idx in range(node_dim):
        row_start, row_end = W.indptr[node_idx], W.indptr[node_idx + 1]
        neigh = W.indices[row_start:row_end]
        weights = W.data[row_start:row_end]
        keep = neigh != node_idx
        neigh = neigh[keep]
        weights = weights[keep]
        if neigh.size == 0:
            continue
        if neigh.size > k:
            top_pos = np.argpartition(weights, -k)[-k:]
            neigh = neigh[top_pos]
        total += float(np.mean(labels[neigh] == labels[node_idx]))
    return total / float(max(1, node_dim))


def ncut_gt_pi(W, labels: np.ndarray) -> float:
    W = W.tocsr()
    degree = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)
    ncut = 0.0
    eps = 1e-12
    for label in np.unique(labels):
        in_cluster = labels == label
        volume = float(degree[in_cluster].sum())
        if volume <= eps:
            continue
        internal_assoc = 0.0
        for node_idx in np.flatnonzero(in_cluster):
            row_start, row_end = W.indptr[node_idx], W.indptr[node_idx + 1]
            neigh = W.indices[row_start:row_end]
            weights = W.data[row_start:row_end]
            internal_assoc += float(weights[in_cluster[neigh]].sum())
        cut = max(0.0, volume - internal_assoc)
        ncut += cut / (volume + eps)
    return float(ncut)


def spectral_nmi(affinity, labels: np.ndarray, num_clusters: int, seed: int, topk: Optional[int]):
    pred = spectral_cluster_affinity(affinity, num_clusters, random_state=seed, topk=topk)
    metrics = compute_clustering_metrics(labels, pred)
    return metrics["NMI"]


def format_float(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{float(value):.6f}"


def main():
    args = get_args()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = resolve_path(cur_dir, args.data_root)
    args.cache_dir = resolve_path(cur_dir, args.cache_dir)
    set_random_seed(args.seed)

    file_path = os.path.join(args.data_root, args.dataset, f"{args.dataset}.txt")
    label_path = os.path.join(args.data_root, args.dataset, "node2label.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Temporal edge file not found: {file_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    edges_uvt = read_temporal_edges(file_path)
    labels_dict = read_node_labels(label_path)
    node_dim = infer_trainer_node_dim(edges_uvt)
    labels = label_array_or_none(labels_dict, node_dim)

    tadj_list, T_total = compress_time_indices(edges_uvt)
    tppr = compute_tppr_cached(file_path, node_dim, args, args.dataset, tadj_list=tadj_list, T_total=T_total).tocsr()
    W = build_ncut_graph(tppr).tocsr()

    density = float(W.nnz) / float(max(1, W.shape[0] * W.shape[1]))
    if labels is None:
        purity = {k: float("nan") for k in (5, 10, 20)}
        leakage = {k: float("nan") for k in (5, 10, 20)}
        ncut = float("nan")
        nmi_spectral = float("nan")
        nmi_spectral_topk = float("nan")
        labeled_nodes = 0
    else:
        num_clusters = int(np.unique(labels).size)
        purity = {k: purity_at_k(W, labels, k) for k in (5, 10, 20)}
        leakage = {k: 1.0 - purity[k] if not math.isnan(purity[k]) else float("nan") for k in (5, 10, 20)}
        ncut = ncut_gt_pi(W, labels)
        nmi_spectral = spectral_nmi(tppr, labels, num_clusters, args.seed, topk=None)
        nmi_spectral_topk = spectral_nmi(tppr, labels, num_clusters, args.seed, topk=args.spectral_topk)
        labeled_nodes = int(labels.shape[0])

    print(
        f"dataset={args.dataset} "
        f"tppr_K={args.tppr_K} "
        f"tppr_alpha={args.tppr_alpha} "
        f"taps_budget_mode={args.taps_budget_mode} "
        f"taps_beta={args.taps_budget_beta} "
        f"nodes={node_dim} "
        f"labeled_nodes={labeled_nodes} "
        f"density={format_float(density)}"
    )
    print(
        f"purity@5={format_float(purity[5])} "
        f"purity@10={format_float(purity[10])} "
        f"purity@20={format_float(purity[20])}"
    )
    print(
        f"leakage@5={format_float(leakage[5])} "
        f"leakage@10={format_float(leakage[10])} "
        f"leakage@20={format_float(leakage[20])}"
    )
    print(
        f"ncut_gt_pi={format_float(ncut)} "
        f"nmi_spectral_pi={format_float(nmi_spectral)} "
        f"nmi_spectral_topk_pi={format_float(nmi_spectral_topk)} "
        f"spectral_topk={args.spectral_topk}"
    )


if __name__ == "__main__":
    main()
