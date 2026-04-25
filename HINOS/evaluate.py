import argparse
import os
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

from utils import resolve_path


def get_args():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Evaluate clustering outputs with ACC, NMI, F1, and ARI.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset name")
    parser.add_argument("--epoch", type=int, default=0, help="training epoch used in the prediction filename")
    parser.add_argument("--pred_path", type=str, default="", help="path to prediction file; overrides --epoch lookup")
    parser.add_argument(
        "--label_path",
        type=str,
        default="",
        help="path to ground-truth node2label file; defaults to dataset/<dataset>/node2label.txt",
    )
    parser.add_argument("--data_root", type=str, default=os.path.join(cur_dir, "dataset"), help="dataset root")
    parser.add_argument("--emb_root", type=str, default=os.path.join(cur_dir, "emb"), help="prediction output root")
    return parser.parse_args()


def read_node_labels(path: str) -> Dict[int, int]:
    labels = {}
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            node_id = int(parts[0])
            label = int(parts[1])
            labels[node_id] = label
    if not labels:
        raise ValueError(f"No labels were read from {path}")
    return labels


def read_predictions(path: str) -> Dict[int, int]:
    pred = {}
    with open(path, "r", encoding="utf-8") as reader:
        for line_idx, line in enumerate(reader):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            if line_idx == 0 and parts[0].lower() == "node_id":
                continue
            node_id = int(parts[0])
            cluster_id = int(parts[1])
            pred[node_id] = cluster_id
    if not pred:
        raise ValueError(f"No predictions were read from {path}")
    return pred


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


def evaluate_clustering(dataset: str, label_path: str, pred_path: str) -> Dict[str, object]:
    gt_labels = read_node_labels(label_path)
    pred_labels = read_predictions(pred_path)
    y_true, y_pred = align_labels(gt_labels, pred_labels)
    mapping, cost, row_ind, col_ind = match_pred_to_true(y_true, y_pred)
    y_pred_aligned = remap_predictions(y_pred, mapping)
    acc = float(cost[row_ind, col_ind].sum()) / float(y_true.size)

    return {
        "dataset": dataset,
        "label_path": label_path,
        "prediction_path": pred_path,
        "evaluated_nodes": int(y_true.size),
        "ACC": acc,
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred_aligned, average="macro"),
        "ARI": adjusted_rand_score(y_true, y_pred),
    }


def main():
    args = get_args()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = resolve_path(cur_dir, args.data_root)
    args.emb_root = resolve_path(cur_dir, args.emb_root)

    label_path = (
        resolve_path(cur_dir, args.label_path)
        if args.label_path
        else os.path.join(args.data_root, args.dataset, "node2label.txt")
    )
    if args.pred_path:
        pred_path = resolve_path(cur_dir, args.pred_path)
    else:
        if args.epoch <= 0:
            raise ValueError("Please provide --pred_path or a positive --epoch.")
        pred_path = os.path.join(args.emb_root, args.dataset, f"{args.dataset}_TGC_{args.epoch}_pred.txt")

    result = evaluate_clustering(args.dataset, label_path, pred_path)

    print("\n========== Evaluation ==========")
    print(f"Dataset={args.dataset}")
    print(f"LabelPath={label_path}")
    print(f"PredictionPath={pred_path}")
    print(f"EvaluatedNodes={result['evaluated_nodes']}")
    for name in ("ACC", "NMI", "F1", "ARI"):
        print(f"{name}={result[name]:.6f}")


if __name__ == "__main__":
    main()
