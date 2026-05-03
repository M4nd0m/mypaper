import argparse
import csv
import os

import numpy as np
import scipy.sparse as sp

from clustering_utils import read_node_labels
from data_load import read_temporal_edges
from graph_diagnostics import diagnose_affinity, label_array
from utils import ensure_dir, resolve_path, set_random_seed


def get_args():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Diagnose spectral/cut alignment for a saved sparse affinity.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--matrix_path", type=str, required=True, help="scipy sparse .npz affinity")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--data_root", type=str, default=os.path.join(cur_dir, "dataset"))
    parser.add_argument("--num_clusters", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def infer_node_dim(edges_uvt) -> int:
    nodes = set()
    for u, v, _ in edges_uvt:
        nodes.add(int(u))
        nodes.add(int(v))
    return max(nodes) + 1 if nodes else 0


def write_one_row(path: str, row):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main():
    args = get_args()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = resolve_path(cur_dir, args.data_root)
    args.matrix_path = resolve_path(cur_dir, args.matrix_path)
    if args.output_path:
        args.output_path = resolve_path(cur_dir, args.output_path)
    set_random_seed(args.seed)

    file_path = os.path.join(args.data_root, args.dataset, f"{args.dataset}.txt")
    label_path = os.path.join(args.data_root, args.dataset, "node2label.txt")
    node_dim = infer_node_dim(read_temporal_edges(file_path))
    labels = label_array(read_node_labels(label_path), node_dim)
    num_clusters = args.num_clusters if args.num_clusters > 0 else int(len(np.unique(labels)))
    W = sp.load_npz(args.matrix_path).tocsr()
    row = {
        "dataset": args.dataset,
        "matrix_path": args.matrix_path,
    }
    row.update(diagnose_affinity(W, labels, num_clusters, random_state=args.seed, prefix="W_U"))
    if args.output_path:
        write_one_row(args.output_path, row)
    print(row)


if __name__ == "__main__":
    main()
