import argparse
import csv
import math
import os
from typing import Dict, Optional

import numpy as np

from clustering_utils import compute_clustering_metrics, read_node_labels
from data_load import compress_time_indices, read_temporal_edges
from sparsification import build_ncut_graph, compute_tppr_cached
from utils import resolve_path, set_random_seed


def get_args():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Compare ground-truth and predicted partitions on the same TPPR-Cut graph."
    )
    parser.add_argument("--dataset", type=str, default="dblp", help="dataset name under data_root")
    parser.add_argument("--pred_path", type=str, required=True, help="prediction file produced by trainer")
    parser.add_argument("--data_root", type=str, default=os.path.join(cur_dir, "dataset"), help="dataset root")
    parser.add_argument("--cache_dir", type=str, default=os.path.join(cur_dir, "cache"), help="TPPR/TAPS cache root")
    parser.add_argument("--output_path", type=str, default="", help="optional CSV path for one-row output")
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
    parser.add_argument("--taps_budget_beta", type=float, default=0.2, help="TAPS nlogn budget scale")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


def infer_node_dim(edges_uvt) -> int:
    nodes = set()
    for u, v, _ in edges_uvt:
        nodes.add(int(u))
        nodes.add(int(v))
    return len(nodes)


def label_array(labels_dict: Dict[int, int], node_dim: int) -> np.ndarray:
    labels = np.empty((node_dim,), dtype=np.int64)
    missing = []
    for node_id in range(node_dim):
        if node_id not in labels_dict:
            missing.append(node_id)
        else:
            labels[node_id] = int(labels_dict[node_id])
    if missing:
        preview = ", ".join(str(x) for x in missing[:10])
        raise ValueError(f"Labels are missing for {len(missing)} nodes, e.g. {preview}")
    return labels


def read_prediction_file(path: str, node_dim: int) -> np.ndarray:
    pred = np.full((node_dim,), -1, dtype=np.int64)
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                node_id = int(parts[0])
                cluster_id = int(parts[1])
            except ValueError:
                continue
            if 0 <= node_id < node_dim:
                pred[node_id] = cluster_id
    missing = np.flatnonzero(pred < 0)
    if missing.size:
        preview = ", ".join(str(int(x)) for x in missing[:10])
        raise ValueError(f"Predictions are missing for {missing.size} nodes, e.g. {preview}")
    return pred


def hard_partition_ncut(W, partition: np.ndarray) -> float:
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


def weighted_same_partition_ratio(W, partition: np.ndarray) -> Dict[str, float]:
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


def write_one_row(path: str, row: Dict[str, object]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def fmt(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def main():
    args = get_args()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = resolve_path(cur_dir, args.data_root)
    args.cache_dir = resolve_path(cur_dir, args.cache_dir)
    args.pred_path = resolve_path(cur_dir, args.pred_path)
    if args.output_path:
        args.output_path = resolve_path(cur_dir, args.output_path)
    set_random_seed(args.seed)

    file_path = os.path.join(args.data_root, args.dataset, f"{args.dataset}.txt")
    label_path = os.path.join(args.data_root, args.dataset, "node2label.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Temporal edge file not found: {file_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    if not os.path.exists(args.pred_path):
        raise FileNotFoundError(f"Prediction file not found: {args.pred_path}")

    edges_uvt = read_temporal_edges(file_path)
    labels_dict = read_node_labels(label_path)
    node_dim = infer_node_dim(edges_uvt)
    labels = label_array(labels_dict, node_dim)
    pred = read_prediction_file(args.pred_path, node_dim)

    tadj_list, T_total = compress_time_indices(edges_uvt)
    tppr = compute_tppr_cached(file_path, node_dim, args, args.dataset, tadj_list=tadj_list, T_total=T_total).tocsr()
    W = build_ncut_graph(tppr).tocsr()

    ncut_gt = hard_partition_ncut(W, labels)
    ncut_pred = hard_partition_ncut(W, pred)
    gt_mass = weighted_same_partition_ratio(W, labels)
    pred_mass = weighted_same_partition_ratio(W, pred)
    metrics = compute_clustering_metrics(labels, pred)

    row = {
        "dataset": args.dataset,
        "pred_path": args.pred_path,
        "tppr_K": args.tppr_K,
        "taps_budget_mode": args.taps_budget_mode,
        "taps_budget_beta": args.taps_budget_beta,
        "ncut_gt_pi": ncut_gt,
        "ncut_pred_pi": ncut_pred,
        "ncut_pred_minus_gt": ncut_pred - ncut_gt,
        "tppr_weighted_purity_gt": gt_mass["same_partition_mass_ratio"],
        "tppr_weighted_leakage_gt": gt_mass["cross_partition_mass_ratio"],
        "tppr_same_cluster_mass_ratio_pred": pred_mass["same_partition_mass_ratio"],
        "tppr_cross_cluster_mass_ratio_pred": pred_mass["cross_partition_mass_ratio"],
        "acc_pred": metrics["ACC"],
        "nmi_pred": metrics["NMI"],
        "ari_pred": metrics["ARI"],
        "f1_pred": metrics["F1"],
    }

    print(
        f"dataset={args.dataset} tppr_K={args.tppr_K} taps_beta={args.taps_budget_beta} "
        f"pred={args.pred_path}"
    )
    print(
        f"ncut_gt_pi={fmt(ncut_gt)} "
        f"ncut_pred_pi={fmt(ncut_pred)} "
        f"ncut_pred_minus_gt={fmt(ncut_pred - ncut_gt)}"
    )
    print(
        f"tppr_weighted_purity_gt={fmt(row['tppr_weighted_purity_gt'])} "
        f"tppr_same_cluster_mass_ratio_pred={fmt(row['tppr_same_cluster_mass_ratio_pred'])}"
    )
    print(
        f"ACC={fmt(metrics['ACC'])} "
        f"NMI={fmt(metrics['NMI'])} "
        f"ARI={fmt(metrics['ARI'])} "
        f"F1={fmt(metrics['F1'])}"
    )
    if args.output_path:
        write_one_row(args.output_path, row)
        print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
