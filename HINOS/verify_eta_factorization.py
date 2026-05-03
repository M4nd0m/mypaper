import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags

from clustering_utils import read_node_labels
from data_load import compress_time_indices, read_temporal_edges
from diagnose_tppr import label_array_or_none, ncut_gt_pi, purity_at_k, tppr_weighted_label_purity
from sparsification import (
    build_B,
    build_L_alpha_from_edges,
    build_P_from_edges,
    build_XQ_tilde,
    build_ncut_graph,
    compute_taps_cached,
    tppr_truncated_with_B_XQ,
)
from utils import ensure_dir, resolve_path, set_random_seed


CSV_COLUMNS = [
    "dataset",
    "mode",
    "tppr_K",
    "tppr_alpha",
    "taps_alpha",
    "taps_budget_beta",
    "nodes",
    "labeled_nodes",
    "input_nnz",
    "input_density",
    "input_mass",
    "eta_nonzero_pairs",
    "eta_min",
    "eta_median",
    "eta_max",
    "tppr_nnz",
    "pi_cut_nnz",
    "pi_asymmetry_norm",
    "pi_cut_density",
    "purity_at_5_pi",
    "purity_at_10_pi",
    "purity_at_20_pi",
    "tppr_weighted_purity_pi",
    "ncut_gt_pi",
    "target_purity_at_5",
    "meets_target",
]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_modes(value: str) -> List[str]:
    valid = {"raw_static", "raw_eta", "taps", "taps_eta"}
    modes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(modes) - valid)
    if unknown:
        raise ValueError(f"Unknown mode(s): {unknown}. Valid modes: {sorted(valid)}")
    return modes


def infer_node_dim(edges_uvt) -> int:
    nodes = set()
    for u, v, _ in edges_uvt:
        nodes.add(int(u))
        nodes.add(int(v))
    return max(nodes) + 1 if nodes else 0


def build_raw_static_and_eta(edges_uvt, node_dim: int) -> Tuple[csr_matrix, csr_matrix, Dict[str, float]]:
    pair_counts: Dict[Tuple[int, int], float] = defaultdict(float)
    node_events = np.zeros((node_dim,), dtype=np.float64)

    for u, v, _ in edges_uvt:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        pair_counts[(a, b)] += 1.0
        if 0 <= u < node_dim:
            node_events[u] += 1.0
        if 0 <= v < node_dim:
            node_events[v] += 1.0

    raw_rows, raw_cols, raw_data = [], [], []
    eta_rows, eta_cols, eta_data = [], [], []
    eta_values = []

    for (a, b), count in pair_counts.items():
        denom = math.sqrt(float(node_events[a] * node_events[b])) if node_events[a] > 0 and node_events[b] > 0 else 0.0
        eta = float(count / denom) if denom > 0.0 else 0.0
        weight_raw = float(count)
        weight_eta = float(count * eta)

        raw_rows.extend([a, b])
        raw_cols.extend([b, a])
        raw_data.extend([weight_raw, weight_raw])

        if weight_eta > 0.0:
            eta_rows.extend([a, b])
            eta_cols.extend([b, a])
            eta_data.extend([weight_eta, weight_eta])
            eta_values.append(eta)

    raw = sp.coo_matrix((raw_data, (raw_rows, raw_cols)), shape=(node_dim, node_dim), dtype=np.float64).tocsr()
    raw.sum_duplicates()
    raw.eliminate_zeros()

    raw_eta = sp.coo_matrix((eta_data, (eta_rows, eta_cols)), shape=(node_dim, node_dim), dtype=np.float64).tocsr()
    raw_eta.sum_duplicates()
    raw_eta.eliminate_zeros()

    eta_arr = np.asarray(eta_values, dtype=np.float64)
    stats = {
        "eta_nonzero_pairs": int(eta_arr.size),
        "eta_min": float(np.min(eta_arr)) if eta_arr.size else float("nan"),
        "eta_median": float(np.median(eta_arr)) if eta_arr.size else float("nan"),
        "eta_max": float(np.max(eta_arr)) if eta_arr.size else float("nan"),
    }
    return raw, raw_eta, stats


def eta_weight_taps(taps: csr_matrix, raw: csr_matrix, raw_eta: csr_matrix) -> csr_matrix:
    raw = raw.tocsr().astype(np.float64)
    raw_eta = raw_eta.tocsr().astype(np.float64)
    eta = raw_eta.multiply(raw.power(-1))
    eta.data = np.nan_to_num(eta.data, nan=0.0, posinf=0.0, neginf=0.0)
    eta.eliminate_zeros()
    out = taps.tocsr().astype(np.float64).multiply(eta)
    out.sum_duplicates()
    out.eliminate_zeros()
    return out.tocsr()


def tppr_from_adjacency(input_graph: csr_matrix, node_dim: int, alpha: float, K: int) -> csr_matrix:
    A = input_graph.tocsr().astype(np.float64)
    A.eliminate_zeros()
    rows, cols = A.nonzero()
    edges = list(zip(rows.tolist(), cols.tolist()))
    weights = A.data.tolist()

    B = build_B(edges, node_dim)
    XQ = build_XQ_tilde(edges, node_dim)
    P = build_P_from_edges(edges, node_dim, weights=weights)

    deg = np.asarray(P.sum(axis=1)).ravel()
    inv_deg = np.zeros_like(deg)
    inv_deg[deg > 0] = 1.0 / deg[deg > 0]
    D_inv = diags(inv_deg, format="csr")

    L_alpha = build_L_alpha_from_edges(edges, node_dim, alpha, weights=weights)
    D_inv_L_alpha_XQ = D_inv @ (L_alpha @ XQ)
    return tppr_truncated_with_B_XQ(P, B, XQ, D_inv_L_alpha_XQ, alpha, K).tocsr()


def pi_asymmetry_norm(tppr_mat: csr_matrix) -> float:
    pi = tppr_mat.tocsr().astype(np.float64)
    asym = (pi - pi.T).tocsr()
    pi_fro = float(np.sqrt(np.sum(pi.data ** 2))) if pi.nnz > 0 else 0.0
    asym_fro = float(np.sqrt(np.sum(asym.data ** 2))) if asym.nnz > 0 else 0.0
    return asym_fro / (pi_fro + 1e-12)


def evaluate_mode(
    dataset: str,
    mode: str,
    input_graph: csr_matrix,
    eta_stats: Dict[str, float],
    labels: np.ndarray,
    args,
    target: float,
) -> Dict[str, object]:
    node_dim = int(input_graph.shape[0])
    tppr = tppr_from_adjacency(input_graph, node_dim, float(args.tppr_alpha), int(args.tppr_K))
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

    return {
        "dataset": dataset,
        "mode": mode,
        "tppr_K": int(args.tppr_K),
        "tppr_alpha": float(args.tppr_alpha),
        "taps_alpha": float(args.taps_alpha),
        "taps_budget_beta": float(args.taps_budget_beta),
        "nodes": node_dim,
        "labeled_nodes": labeled_nodes,
        "input_nnz": int(input_graph.nnz),
        "input_density": float(input_graph.nnz) / float(max(1, node_dim * node_dim)),
        "input_mass": float(input_graph.sum()),
        "eta_nonzero_pairs": eta_stats["eta_nonzero_pairs"],
        "eta_min": eta_stats["eta_min"],
        "eta_median": eta_stats["eta_median"],
        "eta_max": eta_stats["eta_max"],
        "tppr_nnz": int(tppr.nnz),
        "pi_cut_nnz": int(W.nnz),
        "pi_asymmetry_norm": pi_asymmetry_norm(tppr),
        "pi_cut_density": density,
        "purity_at_5_pi": purity5,
        "purity_at_10_pi": purity10,
        "purity_at_20_pi": purity20,
        "tppr_weighted_purity_pi": weighted_purity,
        "ncut_gt_pi": ncut,
        "target_purity_at_5": target,
        "meets_target": bool(not math.isnan(purity5) and purity5 >= target),
    }


def iter_dataset_rows(dataset: str, args, modes: List[str]) -> Iterable[Dict[str, object]]:
    file_path = os.path.join(args.data_root, dataset, f"{dataset}.txt")
    label_path = os.path.join(args.data_root, dataset, "node2label.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Temporal edge file not found: {file_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    edges_uvt = read_temporal_edges(file_path)
    node_dim = infer_node_dim(edges_uvt)
    labels = label_array_or_none(read_node_labels(label_path), node_dim)
    tadj_list, T_total = compress_time_indices(edges_uvt)

    raw_static, raw_eta, eta_stats = build_raw_static_and_eta(edges_uvt, node_dim)
    input_graphs = {
        "raw_static": raw_static,
        "raw_eta": raw_eta,
    }

    if any(mode.startswith("taps") for mode in modes):
        taps = compute_taps_cached(file_path, args, dataset, tadj_list=tadj_list, T_total=T_total).tocsr()
        taps.sum_duplicates()
        taps.eliminate_zeros()
        input_graphs["taps"] = taps
        input_graphs["taps_eta"] = eta_weight_taps(taps, raw_static, raw_eta)

    target = float(args.target_purity_at_5)
    for mode in modes:
        yield evaluate_mode(dataset, mode, input_graphs[mode], eta_stats, labels, args, target)


def write_rows(path: str, rows: List[Dict[str, object]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8-sig") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: List[Dict[str, object]]) -> None:
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        ranked = sorted(
            [row for row in rows if row["dataset"] == dataset],
            key=lambda row: float(row["purity_at_5_pi"]),
            reverse=True,
        )
        print(f"\n[{dataset}]")
        for row in ranked:
            print(
                "mode={mode} K={tppr_K} alpha={tppr_alpha:g} beta={taps_budget_beta:g} "
                "purity@5={purity_at_5_pi:.6f} ncut_gt={ncut_gt_pi:.6f} "
                "asym={pi_asymmetry_norm:.6f} density={pi_cut_density:.6f} "
                "weighted_purity={tppr_weighted_purity_pi:.6f} target={meets_target}".format(**row)
            )


def get_args():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Verify primitive evidence factorization before implementing learnable rhythm filtering."
    )
    parser.add_argument("--datasets", nargs="+", default=["brain"])
    parser.add_argument("--data_root", type=str, default=os.path.join(cur_dir, "dataset"))
    parser.add_argument("--cache_dir", type=str, default=os.path.join(cur_dir, "cache"))
    parser.add_argument("--output_csv", type=str, default=os.path.join(cur_dir, "diagnostics", "eta_factorization_verify.csv"))
    parser.add_argument("--modes", type=str, default="raw_static,raw_eta,taps,taps_eta")
    parser.add_argument("--K", type=str, default="1,2")
    parser.add_argument("--alpha", type=str, default="0.8")
    parser.add_argument("--taps_alpha", type=float, default=0.8)
    parser.add_argument("--taps_budget_mode", type=str, default="nlogn", choices=["sqrt_edges", "nlogn"])
    parser.add_argument("--taps_budget_beta", type=float, default=0.8)
    parser.add_argument("--taps_tau_eps", type=float, default=1.0)
    parser.add_argument("--taps_rng_seed", type=int, default=42)
    parser.add_argument("--taps_T_cap", type=int, default=10)
    parser.add_argument("--target_purity_at_5", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = resolve_path(cur_dir, args.data_root)
    args.cache_dir = resolve_path(cur_dir, args.cache_dir)
    args.output_csv = resolve_path(cur_dir, args.output_csv)
    set_random_seed(args.seed)

    modes = parse_modes(args.modes)
    rows: List[Dict[str, object]] = []
    for dataset in args.datasets:
        for tppr_K in parse_int_list(args.K):
            for alpha in parse_float_list(args.alpha):
                args.tppr_K = int(tppr_K)
                args.tppr_alpha = float(alpha)
                print(f"[Verify] dataset={dataset} K={args.tppr_K} alpha={args.tppr_alpha:g}")
                for row in iter_dataset_rows(dataset, args, modes):
                    rows.append(row)
                    print(
                        "  mode={mode} purity@5={purity_at_5_pi:.6f} "
                        "ncut_gt={ncut_gt_pi:.6f} asym={pi_asymmetry_norm:.6f} "
                        "density={pi_cut_density:.6f}".format(**row)
                    )

    write_rows(args.output_csv, rows)
    print(f"\nWrote {len(rows)} row(s) to {args.output_csv}")
    print_summary(rows)


if __name__ == "__main__":
    main()
