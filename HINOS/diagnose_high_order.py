import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from clustering_utils import read_node_labels
from data_load import compress_time_indices, read_temporal_edges
from sparsification import build_ncut_graph, compute_tppr_cached
from utils import resolve_path, set_random_seed


Edge = Tuple[int, int]


def get_args():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run DBLP high-order relation diagnostics without training.")
    parser.add_argument("--dataset", type=str, default="dblp", help="dataset name under data_root")
    parser.add_argument("--data_root", type=str, default=os.path.join(cur_dir, "dataset"), help="dataset root")
    parser.add_argument(
        "--pretrain_emb_dir",
        type=str,
        default=os.path.join(cur_dir, "pretrain"),
        help="directory containing <dataset>_feature.emb",
    )
    parser.add_argument("--cache_dir", type=str, default=os.path.join(cur_dir, "cache"), help="TPPR/TAPS cache root")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(cur_dir, "diagnostics", "dblp_high_order"),
        help="directory for all diagnostic CSV files and summary.md",
    )
    parser.add_argument("--tppr_alpha", type=float, default=0.2, help="TPPR alpha")
    parser.add_argument("--tppr_K", type=int, default=3, help="TPPR truncation order")
    parser.add_argument("--taps_alpha", type=float, default=0.2, help="TAPS alpha")
    parser.add_argument("--taps_tau_eps", type=float, default=1.0, help="TAPS temporal smoothing tau_eps")
    parser.add_argument("--taps_rng_seed", type=int, default=42, help="TAPS RNG seed")
    parser.add_argument("--taps_T_cap", type=int, default=10, help="maximum TAPS time cap; 0 means no cap")
    parser.add_argument(
        "--taps_budget_mode",
        type=str,
        default="nlogn",
        choices=["sqrt_edges", "nlogn"],
        help="TAPS sampling budget mode",
    )
    parser.add_argument("--taps_budget_beta", type=float, default=0.05, help="TAPS nlogn budget scale")
    parser.add_argument("--seed", type=int, default=42, help="fixed random seed")
    parser.add_argument("--topk_values", type=str, default="5,10,20", help="comma-separated K values")
    parser.add_argument(
        "--max_pairs_per_middle",
        type=int,
        default=5000,
        help="maximum sampled second-order neighbor pairs per middle node",
    )
    parser.add_argument("--cross_topk", type=int, default=10, help="top TPPR neighbors used in cross-label diagnosis")
    parser.add_argument(
        "--knn_metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="metric for node2vec nearest neighbors",
    )
    return parser.parse_args()


def parse_topk_values(value: str) -> List[int]:
    topks = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            topks.append(int(item))
    if not topks:
        raise ValueError("--topk_values must contain at least one positive integer.")
    if any(k <= 0 for k in topks):
        raise ValueError("--topk_values must be positive.")
    return sorted(set(topks))


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
            continue
        labels[node_id] = int(labels_dict[node_id])
    if missing:
        preview = ", ".join(str(x) for x in missing[:10])
        raise ValueError(f"Labels are missing for {len(missing)} nodes, e.g. {preview}")
    return labels


def build_static_graph(edges_uvt, node_dim: int) -> Tuple[Counter, List[set]]:
    multiplicity = Counter()
    adjacency = [set() for _ in range(node_dim)]
    for u_raw, v_raw, _ in edges_uvt:
        u, v = int(u_raw), int(v_raw)
        if u == v:
            continue
        if u >= node_dim or v >= node_dim:
            raise ValueError(
                f"Node id {max(u, v)} exceeds inferred node_dim={node_dim}. "
                "This script expects contiguous ids, matching the trainer diagnostics."
            )
        a, b = (u, v) if u < v else (v, u)
        multiplicity[(a, b)] += 1
        adjacency[a].add(b)
        adjacency[b].add(a)
    return multiplicity, adjacency


def multiplicity_bucket(count: int) -> str:
    if count == 1:
        return "count=1"
    if count <= 3:
        return "count=2-3"
    if count <= 10:
        return "count=4-10"
    return "count>10"


def degree_bucket(degree: int) -> str:
    if degree <= 5:
        return "degree<=5"
    if degree <= 10:
        return "degree=6-10"
    if degree <= 20:
        return "degree=11-20"
    if degree <= 50:
        return "degree=21-50"
    return "degree>50"


def safe_div(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if denom else float("nan")


def format_cell(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def write_csv(path: str, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow(row)


def topk_sparse_row(mat: csr_matrix, row: int, k: int, exclude_self: bool = True) -> List[Tuple[int, float]]:
    start, end = mat.indptr[row], mat.indptr[row + 1]
    cols = mat.indices[start:end]
    vals = mat.data[start:end]
    if exclude_self:
        mask = cols != row
        cols = cols[mask]
        vals = vals[mask]
    if cols.size == 0:
        return []
    keep_count = min(int(k), int(cols.size))
    if cols.size > keep_count:
        keep = np.argpartition(vals, -keep_count)[-keep_count:]
        order = keep[np.argsort(-vals[keep])]
    else:
        order = np.argsort(-vals)
    return [(int(cols[idx]), float(vals[idx])) for idx in order]


def purity_at_k_sparse(mat: csr_matrix, labels: np.ndarray, topk_values: Sequence[int]) -> List[Dict]:
    rows = []
    for k in topk_values:
        total = 0.0
        valid_nodes = 0
        for node_id in range(labels.shape[0]):
            neigh = topk_sparse_row(mat, node_id, k, exclude_self=True)
            if not neigh:
                continue
            neigh_ids = np.asarray([node for node, _ in neigh], dtype=np.int64)
            total += float(np.mean(labels[neigh_ids] == labels[node_id]))
            valid_nodes += 1
        purity = safe_div(total, valid_nodes)
        rows.append(
            {
                "K": int(k),
                "valid_node_count": int(valid_nodes),
                "purity_at_k": purity,
                "leakage_at_k": 1.0 - purity if not math.isnan(purity) else float("nan"),
            }
        )
    return rows


def label_random_baseline(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    probs = counts.astype(np.float64) / float(labels.shape[0])
    return float(np.sum(probs * probs))


def tppr_weighted_label_purity(sym_tppr: csr_matrix, labels: np.ndarray) -> List[Dict]:
    sym_tppr = sym_tppr.tocsr()
    same_label_mass = 0.0
    cross_label_mass = 0.0
    total_mass = 0.0
    nonself_nnz = 0
    for node_id in range(labels.shape[0]):
        start, end = sym_tppr.indptr[node_id], sym_tppr.indptr[node_id + 1]
        neigh = sym_tppr.indices[start:end]
        weights = sym_tppr.data[start:end].astype(np.float64, copy=False)
        keep = neigh != node_id
        neigh = neigh[keep]
        weights = weights[keep]
        if neigh.size == 0:
            continue
        same_mask = labels[neigh] == labels[node_id]
        same_label_mass += float(weights[same_mask].sum())
        cross_label_mass += float(weights[~same_mask].sum())
        total_mass += float(weights.sum())
        nonself_nnz += int(neigh.size)
    purity = safe_div(same_label_mass, total_mass)
    leakage = 1.0 - purity if not math.isnan(purity) else float("nan")
    return [
        {
            "affinity_type": "symmetrized_pi",
            "nonself_nnz": int(nonself_nnz),
            "same_label_mass": same_label_mass,
            "cross_label_mass": cross_label_mass,
            "total_mass": total_mass,
            "tppr_weighted_purity": purity,
            "tppr_weighted_leakage": leakage,
            "random_same_label_baseline": label_random_baseline(labels),
        }
    ]


def experiment_edge_label_purity(multiplicity: Counter, labels: np.ndarray) -> List[Dict]:
    bucket_order = ["count=1", "count=2-3", "count=4-10", "count>10"]
    stats = {bucket: {"edge_count": 0, "same_label_edge_count": 0} for bucket in bucket_order}
    stats["overall"] = {"edge_count": 0, "same_label_edge_count": 0}
    for (u, v), count in multiplicity.items():
        bucket = multiplicity_bucket(int(count))
        same = int(labels[u] == labels[v])
        stats[bucket]["edge_count"] += 1
        stats[bucket]["same_label_edge_count"] += same
        stats["overall"]["edge_count"] += 1
        stats["overall"]["same_label_edge_count"] += same
    rows = []
    for bucket in bucket_order + ["overall"]:
        edge_count = stats[bucket]["edge_count"]
        same_count = stats[bucket]["same_label_edge_count"]
        rows.append(
            {
                "bucket": bucket,
                "edge_count": int(edge_count),
                "same_label_edge_count": int(same_count),
                "purity": safe_div(same_count, edge_count),
            }
        )
    return rows


def sample_neighbor_pairs(neighbors: List[int], max_pairs: int, rng: np.random.RandomState) -> Iterable[Tuple[int, int]]:
    degree = len(neighbors)
    total_pairs = degree * (degree - 1) // 2
    if total_pairs <= max_pairs:
        return combinations(neighbors, 2)

    sampled = set()
    while len(sampled) < max_pairs:
        a, b = rng.choice(degree, size=2, replace=False)
        if a > b:
            a, b = b, a
        sampled.add((int(a), int(b)))
    return ((neighbors[a], neighbors[b]) for a, b in sampled)


def experiment_second_order_paths(
    adjacency: List[set],
    labels: np.ndarray,
    max_pairs_per_middle: int,
    rng: np.random.RandomState,
) -> Tuple[List[Dict], int, int]:
    bucket_order = ["degree<=5", "degree=6-10", "degree=11-20", "degree=21-50", "degree>50"]
    stats = {bucket: {"path_count": 0, "same_label_endpoint_count": 0} for bucket in bucket_order}
    stats["overall"] = {"path_count": 0, "same_label_endpoint_count": 0}
    sampled_middle_nodes = 0
    exact_middle_nodes = 0
    for middle, neigh_set in enumerate(adjacency):
        neighbors = sorted(neigh_set)
        degree = len(neighbors)
        if degree < 2:
            continue
        total_pairs = degree * (degree - 1) // 2
        if total_pairs > max_pairs_per_middle:
            sampled_middle_nodes += 1
        else:
            exact_middle_nodes += 1
        bucket = degree_bucket(degree)
        for u, v in sample_neighbor_pairs(neighbors, max_pairs_per_middle, rng):
            same = int(labels[u] == labels[v])
            stats[bucket]["path_count"] += 1
            stats[bucket]["same_label_endpoint_count"] += same
            stats["overall"]["path_count"] += 1
            stats["overall"]["same_label_endpoint_count"] += same

    rows = []
    for bucket in bucket_order + ["overall"]:
        path_count = stats[bucket]["path_count"]
        same_count = stats[bucket]["same_label_endpoint_count"]
        rows.append(
            {
                "middle_degree_bucket": bucket,
                "path_count": int(path_count),
                "same_label_endpoint_count": int(same_count),
                "purity": safe_div(same_count, path_count),
            }
        )
    return rows, exact_middle_nodes, sampled_middle_nodes


def experiment_directed_vs_symmetric(tppr: csr_matrix, sym_tppr: csr_matrix, labels: np.ndarray, topk_values: Sequence[int]):
    rows = []
    for affinity_type, mat in [("directed_pi", tppr), ("symmetrized_pi", sym_tppr)]:
        for row in purity_at_k_sparse(mat.tocsr(), labels, topk_values):
            rows.append({"affinity_type": affinity_type, **row})
    return rows


def experiment_cross_label_sources(sym_tppr: csr_matrix, adjacency: List[set], labels: np.ndarray, cross_topk: int):
    pair_rows = []
    freq = Counter()
    cn_labels = {}
    cn_degree = {}
    degree_stats = defaultdict(lambda: {"pair_count_covered": 0, "total_common_neighbor_occurrences": 0})

    for source in range(labels.shape[0]):
        for target, weight in topk_sparse_row(sym_tppr, source, cross_topk, exclude_self=True):
            if labels[source] == labels[target]:
                continue
            common = adjacency[source].intersection(adjacency[target])
            pair_rows.append(
                {
                    "source": int(source),
                    "target": int(target),
                    "source_label": int(labels[source]),
                    "target_label": int(labels[target]),
                    "tppr_weight": float(weight),
                    "num_common_neighbors": int(len(common)),
                }
            )
            if not common:
                degree_stats["no_common_neighbor"]["pair_count_covered"] += 1
                continue

            covered_buckets = set()
            for cn in common:
                freq[int(cn)] += 1
                cn_labels[int(cn)] = int(labels[cn])
                cn_degree[int(cn)] = int(len(adjacency[cn]))
                bucket = degree_bucket(len(adjacency[cn]))
                covered_buckets.add(bucket)
                degree_stats[bucket]["total_common_neighbor_occurrences"] += 1
            for bucket in covered_buckets:
                degree_stats[bucket]["pair_count_covered"] += 1

    freq_rows = [
        {
            "common_neighbor": int(node_id),
            "label": int(cn_labels[node_id]),
            "degree": int(cn_degree[node_id]),
            "frequency_in_cross_label_pairs": int(count),
        }
        for node_id, count in freq.most_common()
    ]

    bucket_order = ["degree<=5", "degree=6-10", "degree=11-20", "degree=21-50", "degree>50", "no_common_neighbor"]
    degree_rows = []
    for bucket in bucket_order:
        stat = degree_stats[bucket]
        degree_rows.append(
            {
                "degree_bucket": bucket,
                "pair_count_covered": int(stat["pair_count_covered"]),
                "total_common_neighbor_occurrences": int(stat["total_common_neighbor_occurrences"]),
            }
        )
    return pair_rows, freq_rows, degree_rows


def static_onehop_purity(adjacency: List[set], labels: np.ndarray, topk_values: Sequence[int]) -> List[Dict]:
    rows = []
    sorted_adjacency = [sorted(neigh) for neigh in adjacency]
    for k in topk_values:
        total = 0.0
        valid_nodes = 0
        for node_id, neighbors in enumerate(sorted_adjacency):
            if not neighbors:
                continue
            neigh = np.asarray(neighbors[:k], dtype=np.int64)
            total += float(np.mean(labels[neigh] == labels[node_id]))
            valid_nodes += 1
        purity = safe_div(total, valid_nodes)
        rows.append(
            {
                "neighborhood_type": "static_onehop",
                "K": int(k),
                "valid_node_count": int(valid_nodes),
                "purity_at_k": purity,
                "leakage_at_k": 1.0 - purity if not math.isnan(purity) else float("nan"),
            }
        )
    return rows


def load_node2vec_embedding(path: str, node_dim: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained embedding file not found: {path}")

    vectors = {}
    dim = None
    with open(path, "r", encoding="utf-8") as reader:
        first = reader.readline().strip().split()
        if len(first) == 2 and all(part.lstrip("-").isdigit() for part in first):
            dim = int(first[1])
        elif first:
            node_id = int(first[0])
            vec = np.asarray([float(x) for x in first[1:]], dtype=np.float32)
            dim = int(vec.size)
            vectors[node_id] = vec
        for line in reader:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            node_id = int(parts[0])
            vec = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
            if dim is None:
                dim = int(vec.size)
            if vec.size != dim:
                raise ValueError(f"Inconsistent embedding dimension at node {node_id}: {vec.size} != {dim}")
            vectors[node_id] = vec

    if dim is None:
        raise ValueError(f"Empty embedding file: {path}")
    missing = [node_id for node_id in range(node_dim) if node_id not in vectors]
    if missing:
        preview = ", ".join(str(x) for x in missing[:10])
        raise ValueError(f"Node2vec embeddings are missing for {len(missing)} nodes, e.g. {preview}")
    return np.vstack([vectors[node_id] for node_id in range(node_dim)]).astype(np.float32)


def node2vec_knn_purity(
    embedding: np.ndarray,
    labels: np.ndarray,
    topk_values: Sequence[int],
    metric: str,
) -> List[Dict]:
    max_k = max(topk_values)
    n_neighbors = min(max_k + 1, embedding.shape[0])
    model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="brute")
    model.fit(embedding)
    _, indices = model.kneighbors(embedding, return_distance=True)

    rows = []
    for k in topk_values:
        total = 0.0
        valid_nodes = 0
        for node_id in range(labels.shape[0]):
            neigh = [int(x) for x in indices[node_id].tolist() if int(x) != node_id]
            neigh = neigh[:k]
            if not neigh:
                continue
            total += float(np.mean(labels[np.asarray(neigh, dtype=np.int64)] == labels[node_id]))
            valid_nodes += 1
        purity = safe_div(total, valid_nodes)
        rows.append(
            {
                "neighborhood_type": "node2vec_knn",
                "K": int(k),
                "valid_node_count": int(valid_nodes),
                "purity_at_k": purity,
                "leakage_at_k": 1.0 - purity if not math.isnan(purity) else float("nan"),
            }
        )
    return rows


def temporal_tppr_neighborhood_rows(sym_tppr: csr_matrix, labels: np.ndarray, topk_values: Sequence[int]) -> List[Dict]:
    rows = []
    for row in purity_at_k_sparse(sym_tppr, labels, topk_values):
        rows.append({"neighborhood_type": "temporal_tppr", **row})
    return rows


def markdown_table(rows: Sequence[Dict], columns: Sequence[str], max_rows: Optional[int] = None) -> str:
    visible = list(rows if max_rows is None else rows[:max_rows])
    if not visible:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in visible:
        lines.append("| " + " | ".join(format_cell(row.get(col, "")) for col in columns) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(rows)} rows._")
    return "\n".join(lines)


def write_summary(
    path: str,
    args,
    files: Dict[str, str],
    tables: Dict[str, Tuple[Sequence[Dict], Sequence[str], Optional[int]]],
    exact_middle_nodes: int,
    sampled_middle_nodes: int,
):
    with open(path, "w", encoding="utf-8") as writer:
        writer.write("# DBLP High-Order Diagnostics\n\n")
        writer.write("## Run Parameters\n\n")
        for key, value in sorted(vars(args).items()):
            writer.write(f"- `{key}`: `{value}`\n")
        writer.write("\n")

        writer.write("## Output Files\n\n")
        for name, file_path in files.items():
            writer.write(f"- `{name}`: `{file_path}`\n")
        writer.write("\n")

        writer.write("## Sampling Strategy\n\n")
        writer.write(
            "- Second-order paths enumerate all neighbor pairs when a middle node has at most "
            f"`{args.max_pairs_per_middle}` pairs; otherwise exactly `{args.max_pairs_per_middle}` "
            "neighbor pairs are sampled with the fixed random seed.\n"
        )
        writer.write(f"- Middle nodes enumerated exactly: `{exact_middle_nodes}`\n")
        writer.write(f"- Middle nodes sampled: `{sampled_middle_nodes}`\n")
        writer.write("- Static one-hop neighbors use deterministic sorted neighbor order.\n")
        writer.write(
            "- Static PPR is skipped because this repository does not contain a reusable static PPR "
            "utility and this diagnostic avoids adding a new approximate PPR implementation.\n\n"
        )

        writer.write("## Core Result Tables\n\n")
        for title, (rows, columns, max_rows) in tables.items():
            writer.write(f"### {title}\n\n")
            writer.write(markdown_table(rows, columns, max_rows=max_rows))
            writer.write("\n\n")


def main():
    args = get_args()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = resolve_path(cur_dir, args.data_root)
    args.pretrain_emb_dir = resolve_path(cur_dir, args.pretrain_emb_dir)
    args.cache_dir = resolve_path(cur_dir, args.cache_dir)
    args.output_dir = resolve_path(cur_dir, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    set_random_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    topk_values = parse_topk_values(args.topk_values)

    file_path = os.path.join(args.data_root, args.dataset, f"{args.dataset}.txt")
    label_path = os.path.join(args.data_root, args.dataset, "node2label.txt")
    embedding_path = os.path.join(args.pretrain_emb_dir, f"{args.dataset}_feature.emb")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Temporal edge file not found: {file_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    edges_uvt = read_temporal_edges(file_path)
    labels_dict = read_node_labels(label_path)
    node_dim = infer_node_dim(edges_uvt)
    labels = label_array(labels_dict, node_dim)
    multiplicity, adjacency = build_static_graph(edges_uvt, node_dim)

    tadj_list, T_total = compress_time_indices(edges_uvt)
    tppr = compute_tppr_cached(file_path, node_dim, args, args.dataset, tadj_list=tadj_list, T_total=T_total).tocsr()
    sym_tppr = build_ncut_graph(tppr).tocsr()

    edge_rows = experiment_edge_label_purity(multiplicity, labels)
    second_rows, exact_middle_nodes, sampled_middle_nodes = experiment_second_order_paths(
        adjacency=adjacency,
        labels=labels,
        max_pairs_per_middle=args.max_pairs_per_middle,
        rng=rng,
    )
    directed_rows = experiment_directed_vs_symmetric(tppr, sym_tppr, labels, topk_values)
    weighted_tppr_rows = tppr_weighted_label_purity(sym_tppr, labels)
    pair_rows, cn_freq_rows, cn_degree_rows = experiment_cross_label_sources(
        sym_tppr=sym_tppr,
        adjacency=adjacency,
        labels=labels,
        cross_topk=args.cross_topk,
    )
    neighborhood_rows = []
    neighborhood_rows.extend(static_onehop_purity(adjacency, labels, topk_values))
    neighborhood_rows.extend(temporal_tppr_neighborhood_rows(sym_tppr, labels, topk_values))
    embedding = load_node2vec_embedding(embedding_path, node_dim)
    neighborhood_rows.extend(node2vec_knn_purity(embedding, labels, topk_values, args.knn_metric))

    files = {
        "edge_label_purity_by_multiplicity.csv": os.path.join(args.output_dir, "edge_label_purity_by_multiplicity.csv"),
        "second_order_path_purity_by_middle_degree.csv": os.path.join(
            args.output_dir, "second_order_path_purity_by_middle_degree.csv"
        ),
        "tppr_directed_vs_symmetric_purity.csv": os.path.join(
            args.output_dir, "tppr_directed_vs_symmetric_purity.csv"
        ),
        "tppr_weighted_label_purity.csv": os.path.join(args.output_dir, "tppr_weighted_label_purity.csv"),
        "cross_label_top_tppr_pairs.csv": os.path.join(args.output_dir, "cross_label_top_tppr_pairs.csv"),
        "cross_label_common_neighbor_frequency.csv": os.path.join(
            args.output_dir, "cross_label_common_neighbor_frequency.csv"
        ),
        "cross_label_common_neighbor_degree_summary.csv": os.path.join(
            args.output_dir, "cross_label_common_neighbor_degree_summary.csv"
        ),
        "neighborhood_purity_comparison.csv": os.path.join(args.output_dir, "neighborhood_purity_comparison.csv"),
    }

    write_csv(
        files["edge_label_purity_by_multiplicity.csv"],
        ["bucket", "edge_count", "same_label_edge_count", "purity"],
        edge_rows,
    )
    write_csv(
        files["second_order_path_purity_by_middle_degree.csv"],
        ["middle_degree_bucket", "path_count", "same_label_endpoint_count", "purity"],
        second_rows,
    )
    write_csv(
        files["tppr_directed_vs_symmetric_purity.csv"],
        ["affinity_type", "K", "valid_node_count", "purity_at_k", "leakage_at_k"],
        directed_rows,
    )
    write_csv(
        files["tppr_weighted_label_purity.csv"],
        [
            "affinity_type",
            "nonself_nnz",
            "same_label_mass",
            "cross_label_mass",
            "total_mass",
            "tppr_weighted_purity",
            "tppr_weighted_leakage",
            "random_same_label_baseline",
        ],
        weighted_tppr_rows,
    )
    write_csv(
        files["cross_label_top_tppr_pairs.csv"],
        ["source", "target", "source_label", "target_label", "tppr_weight", "num_common_neighbors"],
        pair_rows,
    )
    write_csv(
        files["cross_label_common_neighbor_frequency.csv"],
        ["common_neighbor", "label", "degree", "frequency_in_cross_label_pairs"],
        cn_freq_rows,
    )
    write_csv(
        files["cross_label_common_neighbor_degree_summary.csv"],
        ["degree_bucket", "pair_count_covered", "total_common_neighbor_occurrences"],
        cn_degree_rows,
    )
    write_csv(
        files["neighborhood_purity_comparison.csv"],
        ["neighborhood_type", "K", "valid_node_count", "purity_at_k", "leakage_at_k"],
        neighborhood_rows,
    )

    summary_path = os.path.join(args.output_dir, "summary.md")
    files["summary.md"] = summary_path
    tables = {
        "Edge Label Purity By Multiplicity": (
            edge_rows,
            ["bucket", "edge_count", "same_label_edge_count", "purity"],
            None,
        ),
        "Second-Order Path Purity By Middle Degree": (
            second_rows,
            ["middle_degree_bucket", "path_count", "same_label_endpoint_count", "purity"],
            None,
        ),
        "Directed TPPR Vs Symmetric TPPR Purity": (
            directed_rows,
            ["affinity_type", "K", "valid_node_count", "purity_at_k", "leakage_at_k"],
            None,
        ),
        "TPPR Weighted Label Purity": (
            weighted_tppr_rows,
            [
                "affinity_type",
                "nonself_nnz",
                "same_label_mass",
                "cross_label_mass",
                "total_mass",
                "tppr_weighted_purity",
                "tppr_weighted_leakage",
                "random_same_label_baseline",
            ],
            None,
        ),
        "Cross-Label Common Neighbor Degree Summary": (
            cn_degree_rows,
            ["degree_bucket", "pair_count_covered", "total_common_neighbor_occurrences"],
            None,
        ),
        "Top Cross-Label Common Neighbor Frequency": (
            cn_freq_rows,
            ["common_neighbor", "label", "degree", "frequency_in_cross_label_pairs"],
            20,
        ),
        "Cross-Label Top TPPR Pairs": (
            pair_rows,
            ["source", "target", "source_label", "target_label", "tppr_weight", "num_common_neighbors"],
            20,
        ),
        "Neighborhood Purity Comparison": (
            neighborhood_rows,
            ["neighborhood_type", "K", "valid_node_count", "purity_at_k", "leakage_at_k"],
            None,
        ),
    }
    write_summary(
        path=summary_path,
        args=args,
        files=files,
        tables=tables,
        exact_middle_nodes=exact_middle_nodes,
        sampled_middle_nodes=sampled_middle_nodes,
    )

    print(f"Wrote diagnostics to {args.output_dir}")
    for name, file_path in files.items():
        print(f"{name}: {file_path}")


if __name__ == "__main__":
    main()
