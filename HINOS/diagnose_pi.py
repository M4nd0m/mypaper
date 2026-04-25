import argparse
import csv
import json
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz

from clustering_utils import (
    compute_clustering_metrics,
    kmeans_predict_sparse_rows,
    read_node_labels,
    spectral_cluster_affinity,
    symmetrize_affinity,
    topk_sparsify_rows,
)
from utils import ensure_dir, resolve_path


def get_args():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Diagnose TPPR affinity matrix Pi.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pi_path", type=str, required=True)
    parser.add_argument("--num_clusters", type=int, required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_path", type=str, default="")
    parser.add_argument("--data_root", type=str, default=os.path.join(cur_dir, "dataset"))
    parser.add_argument("--json_path", type=str, default="")
    parser.add_argument("--csv_path", type=str, default="")
    parser.add_argument("--max_row_kmeans_nodes", type=int, default=20000)
    return parser.parse_args()


def safe_metric_row(prefix, metrics):
    return {
        f"acc_{prefix}": metrics["ACC"],
        f"nmi_{prefix}": metrics["NMI"],
        f"ari_{prefix}": metrics["ARI"],
        f"f1_{prefix}": metrics["F1"],
    }


def main():
    args = get_args()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = resolve_path(cur_dir, args.data_root)
    args.pi_path = resolve_path(cur_dir, args.pi_path)
    args.label_path = resolve_path(cur_dir, args.label_path) if args.label_path else os.path.join(
        args.data_root, args.dataset, "node2label.txt"
    )

    pi = load_npz(args.pi_path).tocsr().astype(np.float64)
    pi_sym = symmetrize_affinity(pi)
    nnz = int(pi.nnz)
    total_entries = float(pi.shape[0] * pi.shape[1])
    data = pi.data if pi.nnz > 0 else np.asarray([0.0], dtype=np.float64)
    row_sums = np.asarray(pi.sum(axis=1)).ravel()
    asym = (pi - pi.T).tocsr()
    has_zero_rows = bool(np.any(row_sums == 0.0))

    result = {
        "dataset": args.dataset,
        "pi_path": args.pi_path,
        "spectral_affinity_used": "pi_sym",
        "shape": [int(pi.shape[0]), int(pi.shape[1])],
        "nnz": nnz,
        "density": float(nnz / total_entries) if total_entries > 0 else 0.0,
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "row_sum_min": float(np.min(row_sums)) if row_sums.size else 0.0,
        "row_sum_max": float(np.max(row_sums)) if row_sums.size else 0.0,
        "row_sum_mean": float(np.mean(row_sums)) if row_sums.size else 0.0,
        "has_zero_rows": has_zero_rows,
        "zero_row_count": int(np.sum(row_sums == 0.0)),
        "is_asymmetric": bool(asym.nnz > 0),
        "asymmetry_nnz": int(asym.nnz),
        "pi_sym_nnz": int(pi_sym.nnz),
        "pi_sym_density": float(pi_sym.nnz / total_entries) if total_entries > 0 else 0.0,
    }

    labels = read_node_labels(args.label_path) if os.path.exists(args.label_path) else None
    y_true = None
    if labels is not None and len(labels) == pi.shape[0]:
        y_true = np.asarray([labels[i] for i in range(pi.shape[0])], dtype=np.int64)
        coo = pi.tocoo()
        same_mask = y_true[coo.row] == y_true[coo.col]
        within_mass = float(np.sum(coo.data[same_mask]))
        total_mass = float(np.sum(coo.data)) + 1e-12
        between_mass = float(total_mass - within_mass)
        result.update(
            {
                "within_class_mass": within_mass / total_mass,
                "between_class_mass": between_mass / total_mass,
                "homophily_ratio": within_mass / (between_mass + 1e-12),
            }
        )

        coo_sym = pi_sym.tocoo()
        same_mask_sym = y_true[coo_sym.row] == y_true[coo_sym.col]
        within_mass_sym = float(np.sum(coo_sym.data[same_mask_sym]))
        total_mass_sym = float(np.sum(coo_sym.data)) + 1e-12
        between_mass_sym = float(total_mass_sym - within_mass_sym)
        result.update(
            {
                "within_class_mass_pi_sym": within_mass_sym / total_mass_sym,
                "between_class_mass_pi_sym": between_mass_sym / total_mass_sym,
                "homophily_ratio_pi_sym": within_mass_sym / (between_mass_sym + 1e-12),
            }
        )
    else:
        result.update(
            {
                "within_class_mass": float("nan"),
                "between_class_mass": float("nan"),
                "homophily_ratio": float("nan"),
                "within_class_mass_pi_sym": float("nan"),
                "between_class_mass_pi_sym": float("nan"),
                "homophily_ratio_pi_sym": float("nan"),
            }
        )

    diag_metrics = {}
    try:
        pred = spectral_cluster_affinity(pi, args.num_clusters, random_state=args.seed)
        if y_true is not None:
            diag_metrics.update(safe_metric_row("spectral_pi", compute_clustering_metrics(y_true, pred)))
    except Exception as exc:
        result["spectral_pi_error"] = str(exc)

    try:
        pred = spectral_cluster_affinity(pi, args.num_clusters, random_state=args.seed, topk=args.topk)
        if y_true is not None:
            diag_metrics.update(safe_metric_row("spectral_topk_pi", compute_clustering_metrics(y_true, pred)))
    except Exception as exc:
        result["spectral_topk_pi_error"] = str(exc)

    try:
        if pi.shape[0] <= args.max_row_kmeans_nodes:
            pred = kmeans_predict_sparse_rows(topk_sparsify_rows(pi, args.topk), args.num_clusters, random_state=args.seed)
            if y_true is not None:
                diag_metrics.update(safe_metric_row("kmeans_rows_pi", compute_clustering_metrics(y_true, pred)))
        else:
            result["kmeans_rows_pi_skipped"] = True
    except Exception as exc:
        result["kmeans_rows_pi_error"] = str(exc)

    result.update(diag_metrics)

    json_path = args.json_path or os.path.splitext(args.pi_path)[0] + "_diagnosis.json"
    csv_path = args.csv_path or os.path.splitext(args.pi_path)[0] + "_diagnosis.csv"
    ensure_dir(os.path.dirname(json_path))
    ensure_dir(os.path.dirname(csv_path))

    with open(json_path, "w", encoding="utf-8") as writer:
        json.dump(result, writer, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=list(result.keys()))
        csv_writer.writeheader()
        csv_writer.writerow(result)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote diagnosis json to {json_path}")
    print(f"Wrote diagnosis csv to {csv_path}")


if __name__ == "__main__":
    main()
