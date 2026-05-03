import argparse
import csv
import glob
import os
from typing import Dict, List

import numpy as np
import torch

from clustering_utils import compute_clustering_metrics, read_node_labels
from data_load import compress_time_indices, read_temporal_edges
from graph_diagnostics import diagnose_affinity, hard_partition_ncut, label_array
from sparsification import build_ncut_graph, compute_tppr_cached
from sccd_unified import UnifiedSimilarity
from utils import ensure_dir, resolve_path, set_random_seed


def get_args():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Offline SCCD-style U diagnostic without touching trainer.")
    parser.add_argument("--dataset", type=str, default="dblp")
    parser.add_argument("--soft_assign_path", type=str, default="", help="baseline *_soft_assign.npy; latest is used if empty")
    parser.add_argument("--data_root", type=str, default=os.path.join(cur_dir, "dataset"))
    parser.add_argument("--emb_root", type=str, default=os.path.join(cur_dir, "emb"))
    parser.add_argument("--cache_dir", type=str, default=os.path.join(cur_dir, "cache"))
    parser.add_argument("--diagnostics_dir", type=str, default=os.path.join(cur_dir, "diagnostics"))
    parser.add_argument("--tppr_alpha", type=float, default=0.2)
    parser.add_argument("--tppr_K", type=int, default=None)
    parser.add_argument("--taps_alpha", type=float, default=0.2)
    parser.add_argument("--taps_tau_eps", type=float, default=1.0)
    parser.add_argument("--taps_rng_seed", type=int, default=42)
    parser.add_argument("--taps_T_cap", type=int, default=10)
    parser.add_argument("--taps_budget_mode", type=str, default="nlogn", choices=["sqrt_edges", "nlogn"])
    parser.add_argument("--taps_budget_beta", type=float, default=None)
    parser.add_argument("--rho_cut", type=float, default=1.0)
    parser.add_argument("--rho_anchor_values", type=str, default="0.1,1.0,10.0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--tag", type=str, default="default")
    return parser.parse_args()


def apply_dataset_defaults(args):
    configs = {
        "school": {"tppr_K": 5, "taps_budget_beta": 0.5},
        "dblp": {"tppr_K": 3, "taps_budget_beta": 0.3},
        "brain": {"tppr_K": 5, "taps_budget_beta": 0.1},
        "patent": {"tppr_K": 5, "taps_budget_beta": 0.2},
        "arXivAI": {"tppr_K": 5, "taps_budget_beta": 0.05},
        "arXivCS": {"tppr_K": 5, "taps_budget_beta": 0.02},
    }
    cfg = configs.get(args.dataset, {})
    if args.tppr_K is None:
        args.tppr_K = int(cfg.get("tppr_K", 5))
    if args.taps_budget_beta is None:
        args.taps_budget_beta = float(cfg.get("taps_budget_beta", 0.2))
    return args


def infer_node_dim(edges_uvt) -> int:
    nodes = set()
    for u, v, _ in edges_uvt:
        nodes.add(int(u))
        nodes.add(int(v))
    return max(nodes) + 1 if nodes else 0


def latest_soft_assign(emb_root: str, dataset: str) -> str:
    pattern = os.path.join(emb_root, dataset, f"{dataset}_TGC_*_soft_assign.npy")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No soft assignment found under {pattern}")
    return max(paths, key=os.path.getmtime)


def cut_loss(Q: torch.Tensor, edge_i: torch.Tensor, edge_j: torch.Tensor, edge_w: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    if edge_i.numel() == 0:
        return torch.tensor(0.0, device=Q.device)
    delta = Q.index_select(0, edge_i) - Q.index_select(0, edge_j)
    l_mat = delta.t().mm(edge_w.unsqueeze(1) * delta)
    g_mat = Q.t().mm(degree.unsqueeze(1) * Q)
    eye = torch.eye(Q.shape[1], device=Q.device, dtype=Q.dtype)
    return torch.trace(torch.linalg.solve(g_mat + eps * eye, l_mat))


def fit_offline_U(W_pi_csr, Q_init: np.ndarray, rho_cut: float, rho_anchor: float, steps: int, lr: float, device: str):
    module = UnifiedSimilarity(W_pi_csr, init_mode="log_pi", device=torch.device(device)).to(device)
    Q = torch.from_numpy(Q_init.astype(np.float32)).to(device)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    last = {}
    for _ in range(int(steps)):
        opt.zero_grad()
        graph = module()
        loss_cut = cut_loss(Q, graph["edge_i"], graph["edge_j"], graph["edge_w"], graph["degree"])
        loss_anchor = module.anchor_loss(graph["u_data"])
        loss = rho_cut * loss_cut + rho_anchor * loss_anchor
        loss.backward()
        opt.step()
        last = {
            "offline_loss": float(loss.detach().cpu().item()),
            "offline_loss_cut": float(loss_cut.detach().cpu().item()),
            "offline_loss_anchor": float(loss_anchor.detach().cpu().item()),
            "delta_U_pi_fro": module.delta_fro(graph["u_data"]),
        }
    return module, last


def write_rows(path: str, rows: List[Dict[str, object]]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def append_markdown_summary(path: str, rows: List[Dict[str, object]]) -> None:
    ensure_dir(os.path.dirname(path))
    header = [
        "dataset",
        "rho_anchor",
        "nmi_spectral_pi",
        "nmi_spectral_W_U",
        "ncut_gt_pi",
        "ncut_gt_W_U",
        "ncut_pred_over_gt_W_U",
        "purity_at_5_W_U",
        "delta_U_pi_fro",
        "dblp_gate",
    ]
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as writer:
        if not exists:
            writer.write("| " + " | ".join(header) + " |\n")
            writer.write("| " + " | ".join(["---"] * len(header)) + " |\n")
        for row in rows:
            writer.write("| " + " | ".join(str(row.get(k, "")) for k in header) + " |\n")


def fmt(value):
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return value


def main():
    args = get_args()
    args = apply_dataset_defaults(args)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = resolve_path(cur_dir, args.data_root)
    args.emb_root = resolve_path(cur_dir, args.emb_root)
    args.cache_dir = resolve_path(cur_dir, args.cache_dir)
    args.diagnostics_dir = resolve_path(cur_dir, args.diagnostics_dir)
    if args.soft_assign_path:
        args.soft_assign_path = resolve_path(cur_dir, args.soft_assign_path)
    else:
        args.soft_assign_path = latest_soft_assign(args.emb_root, args.dataset)
    set_random_seed(args.seed)

    file_path = os.path.join(args.data_root, args.dataset, f"{args.dataset}.txt")
    label_path = os.path.join(args.data_root, args.dataset, "node2label.txt")
    edges_uvt = read_temporal_edges(file_path)
    node_dim = infer_node_dim(edges_uvt)
    labels = label_array(read_node_labels(label_path), node_dim)
    Q = np.load(args.soft_assign_path).astype(np.float32)
    if Q.shape[0] != node_dim:
        raise ValueError(f"Q rows ({Q.shape[0]}) do not match node_dim ({node_dim})")
    q_pred = np.argmax(Q, axis=1).astype(np.int64)
    q_metrics = compute_clustering_metrics(labels, q_pred) if labels is not None else {}

    tadj_list, T_total = compress_time_indices(edges_uvt)
    tppr = compute_tppr_cached(file_path, node_dim, args, args.dataset, tadj_list=tadj_list, T_total=T_total).tocsr()
    W_pi = build_ncut_graph(tppr).tocsr()
    pi_diag = diagnose_affinity(W_pi, labels, Q.shape[1], random_state=args.seed, prefix="pi")
    ncut_q_pi = hard_partition_ncut(W_pi, q_pred) if labels is not None else float("nan")

    rows = []
    for rho_anchor in [float(x.strip()) for x in args.rho_anchor_values.split(",") if x.strip()]:
        module, train_stats = fit_offline_U(
            W_pi,
            Q,
            rho_cut=float(args.rho_cut),
            rho_anchor=float(rho_anchor),
            steps=int(args.steps),
            lr=float(args.lr),
            device=args.device,
        )
        W_u = module.to_scipy()
        u_diag = diagnose_affinity(W_u, labels, Q.shape[1], random_state=args.seed, prefix="W_U")
        ncut_q_u = hard_partition_ncut(W_u, q_pred) if labels is not None else float("nan")
        dblp_gate = ""
        if args.dataset == "dblp":
            dblp_gate = bool(
                u_diag.get("nmi_spectral_W_U", float("nan")) >= 0.04
                and u_diag.get("nmi_spectral_W_U", float("nan")) > pi_diag.get("nmi_spectral_pi", float("nan"))
                and u_diag.get("ncut_gt_W_U", float("inf")) <= pi_diag.get("ncut_gt_pi", float("inf"))
                and u_diag.get("ncut_pred_over_gt_W_U", 0.0) >= 0.70
                and train_stats["delta_U_pi_fro"] <= 0.30
            )
        row = {
            "dataset": args.dataset,
            "tag": args.tag,
            "soft_assign_path": args.soft_assign_path,
            "rho_cut": float(args.rho_cut),
            "rho_anchor": float(rho_anchor),
            "tppr_K": int(args.tppr_K),
            "taps_budget_beta": float(args.taps_budget_beta),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "nmi_argmax_Q": q_metrics.get("NMI", float("nan")),
            "acc_argmax_Q": q_metrics.get("ACC", float("nan")),
            "ncut_q_pi": ncut_q_pi,
            "ncut_q_W_U": ncut_q_u,
            "dblp_gate": dblp_gate,
        }
        row.update(pi_diag)
        row.update(u_diag)
        row.update(train_stats)
        rows.append({k: fmt(v) for k, v in row.items()})

    csv_path = os.path.join(args.diagnostics_dir, f"offline_U_{args.dataset}_{args.tag}.csv")
    write_rows(csv_path, rows)
    append_markdown_summary(os.path.join(args.diagnostics_dir, "offline_U_summary.md"), rows)
    print(f"wrote {csv_path}")
    print(f"updated {os.path.join(args.diagnostics_dir, 'offline_U_summary.md')}")


if __name__ == "__main__":
    main()
