import argparse
import csv
import math
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

from clustering_utils import read_node_labels
from data_load import compress_time_indices, read_temporal_edges
from diagnose_tppr import label_array_or_none, ncut_gt_pi, purity_at_k, tppr_weighted_label_purity
from sparsification import build_ncut_graph, compute_tppr_cached
from utils import ensure_dir, resolve_path, set_random_seed


DEFAULT_GRIDS = {
    "brain": {
        "K": [2, 3],
        "alpha": [0.1, 0.2, 0.4],
        "beta": [0.3, 0.5, 0.8],
        "purity5_target": 0.3,
    },
    "patent": {
        "K": [2, 3],
        "alpha": [0.1, 0.2, 0.4],
        "beta": [0.5, 1.0, 1.5],
        "purity5_target": 0.5,
    },
}


CSV_COLUMNS = [
    "dataset",
    "tppr_K",
    "alpha",
    "taps_budget_beta",
    "nodes",
    "labeled_nodes",
    "pi_asymmetry_norm",
    "pi_cut_density",
    "tppr_nnz",
    "pi_cut_nnz",
    "purity_at_5_pi",
    "purity_at_10_pi",
    "purity_at_20_pi",
    "tppr_weighted_purity_pi",
    "ncut_gt_pi",
    "target_purity_at_5",
    "meets_target",
]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def infer_node_dim(edges_uvt) -> int:
    nodes = set()
    for u, v, _ in edges_uvt:
        nodes.add(int(u))
        nodes.add(int(v))
    return max(nodes) + 1 if nodes else 0


def pi_asymmetry_norm(tppr_mat) -> float:
    pi = tppr_mat.tocsr().astype(np.float64)
    asym = (pi - pi.T).tocsr()
    pi_fro = float(np.sqrt(np.sum(pi.data ** 2))) if pi.nnz > 0 else 0.0
    asym_fro = float(np.sqrt(np.sum(asym.data ** 2))) if asym.nnz > 0 else 0.0
    return asym_fro / (pi_fro + 1e-12)


def dataset_grid(dataset: str, args) -> Dict[str, object]:
    if dataset in DEFAULT_GRIDS and not args.K and not args.alpha and not args.beta:
        return DEFAULT_GRIDS[dataset]
    if not args.K or not args.alpha or not args.beta:
        raise ValueError(
            f"No default grid for dataset={dataset}. Provide --K, --alpha, and --beta explicitly."
        )
    return {
        "K": parse_int_list(args.K),
        "alpha": parse_float_list(args.alpha),
        "beta": parse_float_list(args.beta),
        "purity5_target": float(args.purity5_target),
    }


def iter_sweep_rows(dataset: str, args) -> Iterable[Dict[str, object]]:
    grid = dataset_grid(dataset, args)
    file_path = os.path.join(args.data_root, dataset, f"{dataset}.txt")
    label_path = os.path.join(args.data_root, dataset, "node2label.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Temporal edge file not found: {file_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    edges_uvt = read_temporal_edges(file_path)
    node_dim = infer_node_dim(edges_uvt)
    tadj_list, T_total = compress_time_indices(edges_uvt)
    labels = label_array_or_none(read_node_labels(label_path), node_dim)
    target = float(grid["purity5_target"])

    for tppr_K in grid["K"]:
        for alpha in grid["alpha"]:
            for beta in grid["beta"]:
                args.tppr_K = int(tppr_K)
                args.tppr_alpha = float(alpha)
                args.taps_alpha = float(alpha) if args.sync_taps_alpha else float(args.fixed_taps_alpha)
                args.taps_budget_beta = float(beta)

                tppr = compute_tppr_cached(
                    file_path,
                    node_dim,
                    args,
                    dataset,
                    tadj_list=tadj_list,
                    T_total=T_total,
                ).tocsr()
                W = build_ncut_graph(tppr).tocsr()
                density = float(W.nnz) / float(max(1, W.shape[0] * W.shape[1]))

                if labels is None:
                    purity5 = purity10 = purity20 = weighted_purity = ncut = float("nan")
                    labeled_nodes = 0
                else:
                    purity5 = purity_at_k(W, labels, 5)
                    purity10 = purity_at_k(W, labels, 10)
                    purity20 = purity_at_k(W, labels, 20)
                    weighted_purity = tppr_weighted_label_purity(W, labels)["tppr_weighted_purity"]
                    ncut = ncut_gt_pi(W, labels)
                    labeled_nodes = int(labels.shape[0])

                row = {
                    "dataset": dataset,
                    "tppr_K": int(tppr_K),
                    "alpha": float(alpha),
                    "taps_budget_beta": float(beta),
                    "nodes": int(node_dim),
                    "labeled_nodes": labeled_nodes,
                    "pi_asymmetry_norm": pi_asymmetry_norm(tppr),
                    "pi_cut_density": density,
                    "tppr_nnz": int(tppr.nnz),
                    "pi_cut_nnz": int(W.nnz),
                    "purity_at_5_pi": purity5,
                    "purity_at_10_pi": purity10,
                    "purity_at_20_pi": purity20,
                    "tppr_weighted_purity_pi": weighted_purity,
                    "ncut_gt_pi": ncut,
                    "target_purity_at_5": target,
                    "meets_target": bool(not math.isnan(purity5) and purity5 >= target),
                }
                yield row


def write_rows(path: str, rows: List[Dict[str, object]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8-sig") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: List[Dict[str, object]], topn: int) -> None:
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        ds_rows = [row for row in rows if row["dataset"] == dataset]
        ranked = sorted(
            ds_rows,
            key=lambda row: (
                float(row["purity_at_5_pi"]),
                -float(row["ncut_gt_pi"]),
                -float(row["pi_asymmetry_norm"]),
            ),
            reverse=True,
        )
        print(f"\n[{dataset}] top {min(topn, len(ranked))} by purity_at_5_pi")
        for row in ranked[:topn]:
            print(
                "K={tppr_K} alpha={alpha:g} beta={taps_budget_beta:g} "
                "purity@5={purity_at_5_pi:.6f} ncut_gt={ncut_gt_pi:.6f} "
                "asym={pi_asymmetry_norm:.6f} density={pi_cut_density:.6f} "
                "weighted_purity={tppr_weighted_purity_pi:.6f} target={meets_target}".format(**row)
            )


def get_args():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Sweep TPPR/TAPS grids and report Pi quality diagnostics.")
    parser.add_argument("--datasets", nargs="+", default=["brain", "patent"])
    parser.add_argument("--data_root", type=str, default=os.path.join(cur_dir, "dataset"))
    parser.add_argument("--cache_dir", type=str, default=os.path.join(cur_dir, "cache"))
    parser.add_argument("--output_csv", type=str, default=os.path.join(cur_dir, "diagnostics", "tppr_grid_sweep.csv"))
    parser.add_argument("--K", type=str, default="", help="Comma-separated TPPR truncation orders.")
    parser.add_argument("--alpha", type=str, default="", help="Comma-separated TPPR alpha values.")
    parser.add_argument("--beta", type=str, default="", help="Comma-separated TAPS budget beta values.")
    parser.add_argument("--purity5_target", type=float, default=0.0)
    parser.add_argument("--sync_taps_alpha", type=int, default=1, help="1: set taps_alpha equal to alpha.")
    parser.add_argument("--fixed_taps_alpha", type=float, default=0.2)
    parser.add_argument("--taps_tau_eps", type=float, default=1.0)
    parser.add_argument("--taps_rng_seed", type=int, default=42)
    parser.add_argument("--taps_T_cap", type=int, default=10)
    parser.add_argument("--taps_budget_mode", type=str, default="nlogn", choices=["sqrt_edges", "nlogn"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topn", type=int, default=5)
    return parser.parse_args()


def main():
    args = get_args()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = resolve_path(cur_dir, args.data_root)
    args.cache_dir = resolve_path(cur_dir, args.cache_dir)
    args.output_csv = resolve_path(cur_dir, args.output_csv)
    set_random_seed(args.seed)

    rows: List[Dict[str, object]] = []
    for dataset in args.datasets:
        print(f"[Sweep] dataset={dataset}")
        for row in iter_sweep_rows(dataset, args):
            rows.append(row)
            print(
                "  K={tppr_K} alpha={alpha:g} beta={taps_budget_beta:g} "
                "purity@5={purity_at_5_pi:.6f} ncut_gt={ncut_gt_pi:.6f} "
                "asym={pi_asymmetry_norm:.6f} density={pi_cut_density:.6f}".format(**row)
            )

    write_rows(args.output_csv, rows)
    print(f"\nWrote {len(rows)} row(s) to {args.output_csv}")
    print_summary(rows, args.topn)


if __name__ == "__main__":
    main()
