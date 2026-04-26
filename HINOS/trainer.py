import csv
import json
import math
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import save_npz
from sklearn.cluster import KMeans
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from clustering_utils import (
    compute_clustering_metrics,
    kmeans_predict,
    read_node_labels,
    save_prediction_file,
    spectral_cluster_affinity,
)
from data_load import TGCDataSet, compress_time_indices, read_temporal_edges
from sparsification import build_ncut_graph, compute_tppr_cached
from utils import choose_device, ensure_dir


METRIC_COLUMNS = [
    "epoch",
    "phase",
    "objective_mode",
    "pi_is_symmetric",
    "pi_asymmetry_norm",
    "pi_cut_nnz",
    "pi_cut_density",
    "loss_total",
    "loss_tppr_cut",
    "loss_assign_penalty",
    "loss_com",
    "loss_cut_base",
    "loss_temp",
    "loss_batch",
    "loss_bal",
    "weighted_com",
    "weighted_assign_penalty",
    "weighted_cut_base",
    "weighted_temp",
    "weighted_batch",
    "weighted_bal",
    "rho_assign",
    "lambda_temp",
    "lambda_com",
    "effective_lambda_com",
    "assign_mode",
    "prototype_alpha",
    "cluster_mass",
    "cluster_volume",
    "cluster_volume_min",
    "cluster_volume_max",
    "cluster_volume_entropy",
    "assignment_entropy",
    "grad_norm_cut",
    "grad_norm_temp",
    "grad_norm_batch",
    "grad_norm_bal",
    "acc_argmax_s",
    "nmi_argmax_s",
    "ari_argmax_s",
    "f1_argmax_s",
    "acc_kmeans_z",
    "nmi_kmeans_z",
    "ari_kmeans_z",
    "f1_kmeans_z",
    "acc_kmeans_s",
    "nmi_kmeans_s",
    "ari_kmeans_s",
    "f1_kmeans_s",
    "acc_spectral_pi",
    "nmi_spectral_pi",
    "ari_spectral_pi",
    "f1_spectral_pi",
    "acc_spectral_topk_pi",
    "nmi_spectral_topk_pi",
    "ari_spectral_topk_pi",
    "f1_spectral_topk_pi",
]


class TGCTrainer:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.device = choose_device(args.device)
        self.objective_mode = args.objective_mode
        self.warmup_epochs = int(args.warmup_epochs)
        self.eval_interval = int(args.eval_interval)
        self.grad_eval_interval = int(args.grad_eval_interval) if int(args.grad_eval_interval) > 0 else self.eval_interval
        self.spectral_topk = int(args.spectral_topk)
        self.seed = int(args.seed)
        self.assign_mode = args.assign_mode
        self.prototype_alpha = float(args.prototype_alpha)
        self.main_pred_mode = args.main_pred_mode

        self.file_path = os.path.join(args.data_root, self.dataset_name, f"{self.dataset_name}.txt")
        self.label_path = os.path.join(args.data_root, self.dataset_name, "node2label.txt")
        self.feature_path = os.path.join(args.pretrain_emb_dir, f"{self.dataset_name}_feature.emb")

        tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
        self.run_stem = f"{self.dataset_name}_TGC_{args.epoch}{tag_suffix}"
        self.emb_out_dir = os.path.join(args.emb_root, self.dataset_name)
        self.snapshot_dir = os.path.join(self.emb_out_dir, f"{self.run_stem}_snapshots")
        self.emb_path = os.path.join(self.emb_out_dir, f"{self.run_stem}.emb")
        self.pred_path = os.path.join(self.emb_out_dir, f"{self.run_stem}_pred.txt")
        self.soft_assignment_path = os.path.join(self.emb_out_dir, f"{self.run_stem}_soft_assign.npy")
        self.metrics_csv_path = os.path.join(self.emb_out_dir, f"{self.run_stem}_metrics.csv")
        self.pi_raw_path = os.path.join(self.emb_out_dir, f"{self.run_stem}_pi_raw.npz")
        self.pi_cut_path = os.path.join(self.emb_out_dir, f"{self.run_stem}_pi_cut.npz")
        self.pi_path = self.pi_raw_path

        self.data = TGCDataSet(
            self.file_path,
            args.neg_size,
            args.hist_len,
            self.feature_path,
            bool(args.directed),
            neg_table_size=args.neg_table_size,
        )
        self.node_dim = self.data.get_node_dim()
        self.feature = self.data.get_feature().astype(np.float32)
        self.labels = read_node_labels(self.label_path) if os.path.exists(self.label_path) else None
        self.num_clusters = self._resolve_num_clusters(int(args.num_clusters))

        self.node_emb = nn.Parameter(torch.from_numpy(self.feature).float().to(self.device))
        self.delta = nn.Parameter((torch.zeros(self.node_dim) + 1.0).float().to(self.device))

        edges_uvt = read_temporal_edges(self.file_path)
        self.tadj_list, self.T_total = compress_time_indices(edges_uvt)

        emb_dim = self.feature.shape[1]
        self.cluster_mlp = nn.Sequential(
            nn.Linear(emb_dim, args.cluster_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.cluster_hidden_dim, self.num_clusters),
        ).to(self.device)
        self.cluster_prototypes = None
        if self.assign_mode == "prototype":
            centers = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=self.seed).fit(self.feature).cluster_centers_
            self.cluster_prototypes = nn.Parameter(torch.from_numpy(centers.astype(np.float32)).float().to(self.device))

        self.lambda_com = float(args.lambda_com)
        self.rho_assign = float(args.rho_assign)
        self.lambda_cut = self.lambda_com
        self.lambda_temp = float(args.lambda_temp)
        self.lambda_batch = float(args.lambda_batch)
        self.lambda_bal = float(args.lambda_bal)
        self.com_ramp_epochs = int(args.com_ramp_epochs)

        self.tppr_mat = compute_tppr_cached(
            self.file_path,
            self.node_dim,
            args,
            self.dataset_name,
            tadj_list=self.tadj_list,
            T_total=self.T_total,
        ).tocsr()
        self.tppr_mat.sort_indices()

        self.ncut_graph_csr = build_ncut_graph(self.tppr_mat).tocsr()
        self.ncut_graph_csr.sort_indices()
        self.pi_cut_csr = self.ncut_graph_csr
        self._prepare_full_ncut_tensors()

        self.K = int(self.num_clusters)
        self.neg_size = int(args.neg_size)
        self.hist_len = int(args.hist_len)
        self.batch = int(args.batch_size)
        self.epochs = int(args.epoch)
        self.num_workers = int(args.num_workers)

        params = [self.node_emb, self.delta]
        if self.assign_mode == "prototype":
            params.append(self.cluster_prototypes)
        else:
            params.extend(self.cluster_mlp.parameters())
        self.opt = Adam(lr=args.learning_rate, params=params)

        self.static_predictions: Dict[str, Optional[np.ndarray]] = {}
        self.final_prediction_paths = self._build_final_prediction_paths()
        self._init_metrics_csv()

    def _build_final_prediction_paths(self) -> Dict[str, str]:
        return {
            "argmax_s": os.path.join(self.emb_out_dir, f"{self.run_stem}_argmax_s_pred.txt"),
            "kmeans_z": os.path.join(self.emb_out_dir, f"{self.run_stem}_kmeans_z_pred.txt"),
            "kmeans_s": os.path.join(self.emb_out_dir, f"{self.run_stem}_kmeans_s_pred.txt"),
            "spectral_pi": os.path.join(self.emb_out_dir, f"{self.run_stem}_spectral_pi_pred.txt"),
            "spectral_topk_pi": os.path.join(self.emb_out_dir, f"{self.run_stem}_spectral_topk_pi_pred.txt"),
        }

    def _init_metrics_csv(self) -> None:
        ensure_dir(os.path.dirname(self.metrics_csv_path))
        with open(self.metrics_csv_path, "w", newline="", encoding="utf-8") as writer_file:
            writer = csv.DictWriter(writer_file, fieldnames=METRIC_COLUMNS)
            writer.writeheader()

    def _resolve_num_clusters(self, requested_num_clusters: int) -> int:
        if requested_num_clusters > 0:
            return requested_num_clusters
        if self.labels is None:
            raise ValueError("Unable to infer number of clusters without node2label.txt.")
        num_clusters = int(len(np.unique(np.asarray(list(self.labels.values()), dtype=np.int64))))
        if num_clusters <= 1:
            raise ValueError("Unable to infer a valid number of clusters from node2label.txt.")
        return num_clusters

    def _prepare_full_ncut_tensors(self) -> None:
        W = self.pi_cut_csr.tocoo()
        mask = W.row < W.col
        self.full_ncut_edge_i = torch.from_numpy(W.row[mask].astype(np.int64)).to(self.device)
        self.full_ncut_edge_j = torch.from_numpy(W.col[mask].astype(np.int64)).to(self.device)
        self.full_ncut_edge_w = torch.from_numpy(W.data[mask].astype(np.float32)).to(self.device)
        degree = np.asarray(self.pi_cut_csr.sum(axis=1)).ravel().astype(np.float32)
        self.full_ncut_degree = torch.from_numpy(degree).to(self.device)
        self.pi_cut_degree = self.full_ncut_degree

        pi = self.tppr_mat.tocsr().astype(np.float64)
        asym = (pi - pi.T).tocsr()
        pi_fro = float(np.sqrt(np.sum(pi.data ** 2))) if pi.nnz > 0 else 0.0
        asym_fro = float(np.sqrt(np.sum(asym.data ** 2))) if asym.nnz > 0 else 0.0
        self.pi_is_symmetric = bool(asym.nnz == 0)
        self.pi_asymmetry_norm = asym_fro / (pi_fro + 1e-12)
        self.pi_cut_nnz = int(self.pi_cut_csr.nnz)
        total_entries = float(self.pi_cut_csr.shape[0] * self.pi_cut_csr.shape[1])
        self.pi_cut_density = float(self.pi_cut_nnz / total_entries) if total_entries > 0 else 0.0

    def _assignment(self) -> torch.Tensor:
        if self.assign_mode == "prototype":
            return self._prototype_assignment()
        return F.softmax(self.cluster_mlp(self.node_emb), dim=1)

    def _prototype_assignment(self) -> torch.Tensor:
        eps = 1e-6
        alpha = max(self.prototype_alpha, eps)
        dist = torch.cdist(self.node_emb, self.cluster_prototypes, p=2).pow(2)
        q = (1.0 + dist / alpha).pow(-(alpha + 1.0) / 2.0)
        return q / (q.sum(dim=1, keepdim=True) + eps)

    def _compute_community_terms(
        self, assign: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
        eps = 1e-6
        cluster_mass = assign.sum(dim=0)
        cluster_volume = assign.t().mm(self.pi_cut_degree.unsqueeze(1)).squeeze(1)
        volume_prob = cluster_volume / (cluster_volume.sum() + eps)
        assignment_entropy = -(assign * torch.log(assign + eps)).sum(dim=1).mean()

        degree = self.pi_cut_degree
        degree_sum = degree.sum()
        if float(degree_sum.detach().item()) > eps:
            degree_prob = degree / (degree_sum + eps)
        else:
            degree_prob = torch.full_like(degree, 1.0 / max(1, degree.numel()))

        f = assign.t().mv(degree_prob)
        target = assign.pow(2) / (f.unsqueeze(0) + eps)
        target = target / (target.sum(dim=1, keepdim=True) + eps)
        target = target.detach()
        assign_safe = assign.clamp_min(eps)
        l_assign = degree_prob.unsqueeze(1) * target * torch.log((target + eps) / assign_safe)
        l_assign = l_assign.sum()

        if self.full_ncut_edge_i.numel() == 0:
            zero = torch.tensor(0.0, device=self.device)
            l_com = zero + self.rho_assign * l_assign
            return zero, l_assign.squeeze(), l_com.squeeze(), {
                "cluster_mass": cluster_mass.detach().cpu().numpy(),
                "cluster_volume": cluster_volume.detach().cpu().numpy(),
                "cluster_volume_min": float(cluster_volume.min().item()),
                "cluster_volume_max": float(cluster_volume.max().item()),
                "cluster_volume_entropy": float((-(volume_prob * torch.log(volume_prob + eps)).sum()).item()),
                "assignment_entropy": float(assignment_entropy.item()),
            }

        delta = assign.index_select(0, self.full_ncut_edge_i) - assign.index_select(0, self.full_ncut_edge_j)
        l_mat = delta.t().mm(self.full_ncut_edge_w.unsqueeze(1) * delta)
        g_mat = assign.t().mm(self.full_ncut_degree.unsqueeze(1) * assign)
        g_eps = g_mat + eps * torch.eye(self.K, device=self.device)
        l_tppr_cut = torch.trace(torch.linalg.solve(g_eps, l_mat))
        l_com = l_tppr_cut + self.rho_assign * l_assign

        com_stats = {
            "cluster_mass": cluster_mass.detach().cpu().numpy(),
            "cluster_volume": cluster_volume.detach().cpu().numpy(),
            "cluster_volume_min": float(cluster_volume.min().item()),
            "cluster_volume_max": float(cluster_volume.max().item()),
            "cluster_volume_entropy": float((-(volume_prob * torch.log(volume_prob + eps)).sum()).item()),
            "assignment_entropy": float(assignment_entropy.item()),
        }
        return l_tppr_cut.squeeze(), l_assign.squeeze(), l_com.squeeze(), com_stats

    def _loss_phase(self, epoch_idx: int) -> str:
        if self.objective_mode == "cut_main" and epoch_idx < self.warmup_epochs:
            return "warmup"
        return self.objective_mode

    def _effective_lambda_com(self, epoch_idx: int) -> float:
        if self.objective_mode != "cut_main":
            return self.lambda_com
        if epoch_idx < self.warmup_epochs:
            return 0.0
        progress = float(epoch_idx + 1 - self.warmup_epochs) / float(max(1, self.com_ramp_epochs))
        return self.lambda_com * min(1.0, max(0.0, progress))

    def compute_loss_terms(
        self,
        s_nodes: torch.Tensor,
        t_nodes: torch.Tensor,
        t_times: torch.Tensor,
        n_nodes: torch.Tensor,
        h_nodes: torch.Tensor,
        h_times: torch.Tensor,
        h_time_mask: torch.Tensor,
        epoch_idx: int,
    ) -> Dict[str, object]:
        batch = s_nodes.size(0)

        s_node_emb = self.node_emb.index_select(0, s_nodes.view(-1)).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, t_nodes.view(-1)).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, h_nodes.view(-1)).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb.index_select(0, n_nodes.view(-1)).view(batch, self.neg_size, -1)

        new_st_adj = torch.cosine_similarity(s_node_emb, t_node_emb)
        res_st_loss = torch.norm(1 - new_st_adj, p=2, dim=0)

        new_sh_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), h_node_emb, dim=2)
        new_sh_adj = new_sh_adj * h_time_mask
        new_sn_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), n_node_emb, dim=2)

        res_sh_loss = torch.norm(1 - new_sh_adj, p=2, dim=0).sum(dim=0)
        res_sn_loss = torch.norm(new_sn_adj, p=2, dim=0).sum(dim=0)
        l_batch = (res_st_loss + res_sh_loss + res_sn_loss) / max(1, batch)

        att = F.softmax(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2).neg(), dim=1)
        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()
        delta = self.delta.index_select(0, s_nodes.view(-1)).unsqueeze(1)
        delta_pos = F.softplus(delta)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)
        p_lambda = p_mu + (att * p_alpha * torch.exp(-delta_pos * d_time) * h_time_mask).sum(dim=1)

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()
        time_decay = torch.exp(-delta_pos * d_time).unsqueeze(2)
        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * time_decay * h_time_mask.unsqueeze(2)).sum(dim=1)

        loss_pair = -torch.log(p_lambda.sigmoid() + 1e-6) - torch.log(n_lambda.neg().sigmoid() + 1e-6).sum(dim=1)
        l_temp = loss_pair.mean()

        assign = self._assignment()
        l_tppr_cut, l_assign, l_com, com_stats = self._compute_community_terms(assign)

        phase = self._loss_phase(epoch_idx)
        effective_lambda_com = self._effective_lambda_com(epoch_idx)
        if phase == "warmup":
            weighted_com = torch.zeros_like(l_com)
            weighted_cut = torch.zeros_like(l_tppr_cut)
            weighted_temp = self.lambda_temp * l_temp
            weighted_batch = self.lambda_batch * l_batch
            weighted_assign = torch.zeros_like(l_assign)
            weighted_bal = torch.zeros_like(l_assign)
        else:
            weighted_com = effective_lambda_com * l_com
            weighted_cut = effective_lambda_com * l_tppr_cut
            weighted_temp = self.lambda_temp * l_temp
            weighted_batch = self.lambda_batch * l_batch
            weighted_assign = effective_lambda_com * self.rho_assign * l_assign
            weighted_bal = weighted_assign

        loss_total = weighted_temp + weighted_com + weighted_batch
        return {
            "phase": phase,
            "effective_lambda_com": effective_lambda_com,
            "loss_total": loss_total,
            "loss_tppr_cut": l_tppr_cut,
            "loss_assign_penalty": l_assign,
            "loss_com": l_com,
            "loss_cut_base": l_tppr_cut,
            "loss_temp": l_temp,
            "loss_batch": l_batch,
            "loss_bal": l_assign,
            "weighted_com": weighted_com,
            "weighted_assign_penalty": weighted_assign,
            "weighted_cut_base": weighted_cut,
            "weighted_temp": weighted_temp,
            "weighted_batch": weighted_batch,
            "weighted_bal": weighted_bal,
            "cluster_mass": com_stats["cluster_mass"],
            "cluster_volume": com_stats["cluster_volume"],
            "cluster_volume_min": com_stats["cluster_volume_min"],
            "cluster_volume_max": com_stats["cluster_volume_max"],
            "cluster_volume_entropy": com_stats["cluster_volume_entropy"],
            "assignment_entropy": com_stats["assignment_entropy"],
        }

    def _sample_to_device(self, sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        return (
            sample["source_node"].long().to(self.device),
            sample["target_node"].long().to(self.device),
            sample["target_time"].float().to(self.device),
            sample["neg_nodes"].long().to(self.device),
            sample["history_nodes"].long().to(self.device),
            sample["history_times"].float().to(self.device),
            sample["history_masks"].float().to(self.device),
        )

    def update(self, batch_tensors: Tuple[torch.Tensor, ...], epoch_idx: int) -> Dict[str, float]:
        self.opt.zero_grad()
        loss_terms = self.compute_loss_terms(*batch_tensors, epoch_idx)
        loss_terms["loss_total"].backward()
        self.opt.step()
        return {
            "loss_total": float(loss_terms["loss_total"].item()),
            "loss_tppr_cut": float(loss_terms["loss_tppr_cut"].item()),
            "loss_assign_penalty": float(loss_terms["loss_assign_penalty"].item()),
            "loss_com": float(loss_terms["loss_com"].item()),
            "loss_cut_base": float(loss_terms["loss_cut_base"].item()),
            "loss_temp": float(loss_terms["loss_temp"].item()),
            "loss_batch": float(loss_terms["loss_batch"].item()),
            "loss_bal": float(loss_terms["loss_bal"].item()),
            "effective_lambda_com": float(loss_terms["effective_lambda_com"]),
            "weighted_com": float(loss_terms["weighted_com"].item()),
            "weighted_assign_penalty": float(loss_terms["weighted_assign_penalty"].item()),
            "weighted_cut_base": float(loss_terms["weighted_cut_base"].item()),
            "weighted_temp": float(loss_terms["weighted_temp"].item()),
            "weighted_batch": float(loss_terms["weighted_batch"].item()),
            "weighted_bal": float(loss_terms["weighted_bal"].item()),
            "cluster_mass": np.asarray(loss_terms["cluster_mass"], dtype=np.float64),
            "cluster_volume": np.asarray(loss_terms["cluster_volume"], dtype=np.float64),
            "cluster_volume_min": float(loss_terms["cluster_volume_min"]),
            "cluster_volume_max": float(loss_terms["cluster_volume_max"]),
            "cluster_volume_entropy": float(loss_terms["cluster_volume_entropy"]),
            "assignment_entropy": float(loss_terms["assignment_entropy"]),
        }

    def _grad_norm(self) -> float:
        total = 0.0
        params = [self.node_emb, self.delta]
        if self.assign_mode == "prototype":
            params.append(self.cluster_prototypes)
        else:
            params.extend(self.cluster_mlp.parameters())
        for param in params:
            if param.grad is None:
                continue
            total += float(torch.sum(param.grad.detach() ** 2).item())
        return math.sqrt(total)

    def compute_gradient_diagnostics(self, probe_batch: Tuple[torch.Tensor, ...], epoch_idx: int) -> Dict[str, float]:
        grad_norms = {"grad_norm_cut": 0.0, "grad_norm_temp": 0.0, "grad_norm_batch": 0.0, "grad_norm_bal": 0.0}
        component_map = {
            "grad_norm_cut": "weighted_cut_base",
            "grad_norm_temp": "weighted_temp",
            "grad_norm_batch": "weighted_batch",
            "grad_norm_bal": "weighted_bal",
        }
        for out_name, term_name in component_map.items():
            self.opt.zero_grad()
            loss_terms = self.compute_loss_terms(*probe_batch, epoch_idx)
            term = loss_terms[term_name]
            if float(term.detach().item()) == 0.0:
                continue
            term.backward()
            grad_norms[out_name] = self._grad_norm()
        self.opt.zero_grad()
        return grad_norms

    def save_node_embeddings(self, path: str) -> None:
        ensure_dir(os.path.dirname(path))
        emb = self.node_emb.detach().cpu().numpy()
        with open(path, "w", encoding="utf-8") as writer:
            writer.write(f"{self.node_dim} {emb.shape[1]}\n")
            for i in range(self.node_dim):
                writer.write(str(i) + " " + " ".join(map(str, emb[i])) + "\n")

    def _static_predictions(self) -> Dict[str, Optional[np.ndarray]]:
        if self.static_predictions:
            return self.static_predictions
        self.static_predictions["spectral_pi"] = spectral_cluster_affinity(
            self.tppr_mat, self.K, random_state=self.seed, topk=None
        )
        self.static_predictions["spectral_topk_pi"] = spectral_cluster_affinity(
            self.tppr_mat, self.K, random_state=self.seed, topk=self.spectral_topk
        )
        return self.static_predictions

    def _prediction_metrics(self, pred: np.ndarray) -> Dict[str, float]:
        if self.labels is None:
            return {"ACC": float("nan"), "NMI": float("nan"), "ARI": float("nan"), "F1": float("nan")}
        y_true = np.asarray([self.labels[i] for i in range(self.node_dim)], dtype=np.int64)
        y_pred = np.asarray(pred, dtype=np.int64)
        return compute_clustering_metrics(y_true, y_pred)

    def _snapshot_prediction_paths(self, epoch_num: int) -> Dict[str, str]:
        ensure_dir(self.snapshot_dir)
        prefix = os.path.join(self.snapshot_dir, f"{self.run_stem}_epoch_{epoch_num:04d}")
        return {
            "argmax_s": f"{prefix}_argmax_s_pred.txt",
            "kmeans_z": f"{prefix}_kmeans_z_pred.txt",
            "kmeans_s": f"{prefix}_kmeans_s_pred.txt",
            "spectral_pi": f"{prefix}_spectral_pi_pred.txt",
            "spectral_topk_pi": f"{prefix}_spectral_topk_pi_pred.txt",
        }

    def _save_predictions(self, path_map: Dict[str, str], predictions: Dict[str, np.ndarray]) -> None:
        for method, path in path_map.items():
            ensure_dir(os.path.dirname(path))
            save_prediction_file(path, predictions[method])

    def evaluate_and_log(
        self,
        epoch_num: int,
        epoch_stats: Dict[str, float],
        probe_batch: Optional[Tuple[torch.Tensor, ...]],
        save_final: bool = False,
    ) -> Dict[str, object]:
        emb = self.node_emb.detach().cpu().numpy().astype(np.float32)
        assign = self._assignment().detach().cpu().numpy().astype(np.float32)
        predictions = {
            "argmax_s": np.argmax(assign, axis=1).astype(np.int64),
            "kmeans_z": kmeans_predict(emb, self.K, random_state=self.seed),
            "kmeans_s": kmeans_predict(assign, self.K, random_state=self.seed),
        }
        predictions.update(self._static_predictions())

        snapshot_paths = self._snapshot_prediction_paths(epoch_num)
        self._save_predictions(snapshot_paths, predictions)

        if save_final:
            self._save_predictions(self.final_prediction_paths, predictions)
            main_pred = predictions[self.main_pred_mode]
            save_prediction_file(self.pred_path, main_pred)
            np.save(self.soft_assignment_path, assign)
            self.save_node_embeddings(self.emb_path)
            save_npz(self.pi_raw_path, self.tppr_mat)
            save_npz(self.pi_cut_path, self.pi_cut_csr)

        metric_row = {
            "epoch": int(epoch_num),
            "phase": self._loss_phase(epoch_num - 1),
            "objective_mode": self.objective_mode,
            "pi_is_symmetric": int(self.pi_is_symmetric),
            "pi_asymmetry_norm": self.pi_asymmetry_norm,
            "pi_cut_nnz": self.pi_cut_nnz,
            "pi_cut_density": self.pi_cut_density,
            "loss_total": epoch_stats["loss_total"],
            "loss_tppr_cut": epoch_stats["loss_tppr_cut"],
            "loss_assign_penalty": epoch_stats["loss_assign_penalty"],
            "loss_com": epoch_stats["loss_com"],
            "loss_cut_base": epoch_stats["loss_cut_base"],
            "loss_temp": epoch_stats["loss_temp"],
            "loss_batch": epoch_stats["loss_batch"],
            "loss_bal": epoch_stats["loss_bal"],
            "weighted_com": epoch_stats["weighted_com"],
            "weighted_assign_penalty": epoch_stats["weighted_assign_penalty"],
            "weighted_cut_base": epoch_stats["weighted_cut_base"],
            "weighted_temp": epoch_stats["weighted_temp"],
            "weighted_batch": epoch_stats["weighted_batch"],
            "weighted_bal": epoch_stats["weighted_bal"],
            "rho_assign": self.rho_assign,
            "lambda_temp": self.lambda_temp,
            "lambda_com": self.lambda_com,
            "effective_lambda_com": epoch_stats["effective_lambda_com"],
            "assign_mode": self.assign_mode,
            "prototype_alpha": self.prototype_alpha,
            "cluster_mass": epoch_stats["cluster_mass"],
            "cluster_volume": epoch_stats["cluster_volume"],
            "cluster_volume_min": epoch_stats["cluster_volume_min"],
            "cluster_volume_max": epoch_stats["cluster_volume_max"],
            "cluster_volume_entropy": epoch_stats["cluster_volume_entropy"],
            "assignment_entropy": epoch_stats["assignment_entropy"],
            "grad_norm_cut": 0.0,
            "grad_norm_temp": 0.0,
            "grad_norm_batch": 0.0,
            "grad_norm_bal": 0.0,
        }

        if probe_batch is not None and self.grad_eval_interval > 0 and epoch_num % self.grad_eval_interval == 0:
            metric_row.update(self.compute_gradient_diagnostics(probe_batch, epoch_num - 1))

        for method, pred in predictions.items():
            metrics = self._prediction_metrics(pred)
            prefix = method
            metric_row[f"acc_{prefix}"] = metrics["ACC"]
            metric_row[f"nmi_{prefix}"] = metrics["NMI"]
            metric_row[f"ari_{prefix}"] = metrics["ARI"]
            metric_row[f"f1_{prefix}"] = metrics["F1"]

        with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as writer_file:
            writer = csv.DictWriter(writer_file, fieldnames=METRIC_COLUMNS)
            writer.writerow(metric_row)

        main_method = self.main_pred_mode
        print(
            f"[Eval] epoch={epoch_num} phase={metric_row['phase']} "
            f"loss_total={metric_row['loss_total']:.4f} loss_tppr_cut={metric_row['loss_tppr_cut']:.4f} "
            f"loss_com={metric_row['loss_com']:.4f} "
            f"loss_temp={metric_row['loss_temp']:.4f} loss_batch={metric_row['loss_batch']:.4f} "
            f"loss_assign_penalty={metric_row['loss_assign_penalty']:.4f} "
            f"effective_lambda_com={metric_row['effective_lambda_com']:.4f} "
            f"weighted_com={metric_row['weighted_com']:.4f} "
            f"cluster_volume_min={metric_row['cluster_volume_min']:.4f} "
            f"cluster_volume_max={metric_row['cluster_volume_max']:.4f} "
            f"assignment_entropy={metric_row['assignment_entropy']:.4f} "
            f"{main_method}: ACC={metric_row[f'acc_{main_method}']:.4f} "
            f"NMI={metric_row[f'nmi_{main_method}']:.4f} "
            f"ARI={metric_row[f'ari_{main_method}']:.4f} "
            f"F1={metric_row[f'f1_{main_method}']:.4f}"
        )
        return {
            "predictions": predictions,
            "snapshot_paths": snapshot_paths,
            "metric_row": metric_row,
        }

    def train(self) -> Dict[str, object]:
        total_start = time.time()
        last_eval = None

        for epoch_idx in range(self.epochs):
            epoch_start = time.time()
            loader = DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=self.num_workers)
            num_batches = max(1, len(loader))
            batch_bar = tqdm(loader, desc=f"Epoch {epoch_idx + 1}/{self.epochs}", leave=False)
            probe_batch = None
            epoch_sums = {
                "loss_total": 0.0,
                "loss_tppr_cut": 0.0,
                "loss_assign_penalty": 0.0,
                "loss_com": 0.0,
                "loss_cut_base": 0.0,
                "loss_temp": 0.0,
                "loss_batch": 0.0,
                "loss_bal": 0.0,
                "effective_lambda_com": 0.0,
                "weighted_com": 0.0,
                "weighted_assign_penalty": 0.0,
                "weighted_cut_base": 0.0,
                "weighted_temp": 0.0,
                "weighted_batch": 0.0,
                "weighted_bal": 0.0,
                "cluster_volume_min": 0.0,
                "cluster_volume_max": 0.0,
                "cluster_volume_entropy": 0.0,
                "assignment_entropy": 0.0,
            }
            final_cluster_mass = None
            final_cluster_volume = None

            for i_batch, sample in enumerate(batch_bar, 1):
                batch_tensors = self._sample_to_device(sample)
                if probe_batch is None:
                    probe_batch = tuple(t.detach().clone() for t in batch_tensors)
                step_stats = self.update(batch_tensors, epoch_idx)
                for key in epoch_sums:
                    epoch_sums[key] += step_stats[key]
                final_cluster_mass = step_stats["cluster_mass"]
                final_cluster_volume = step_stats["cluster_volume"]
                batch_bar.set_postfix(loss=float(step_stats["loss_total"]))

            batch_bar.close()
            epoch_time = time.time() - epoch_start

            epoch_avgs = {key: value / num_batches for key, value in epoch_sums.items()}
            final_cluster_mass = np.asarray(
                final_cluster_mass if final_cluster_mass is not None else np.zeros(self.K),
                dtype=np.float64,
            )
            final_cluster_volume = np.asarray(
                final_cluster_volume if final_cluster_volume is not None else np.zeros(self.K),
                dtype=np.float64,
            )
            epoch_avgs["cluster_mass"] = json.dumps([round(float(x), 8) for x in final_cluster_mass.tolist()])
            epoch_avgs["cluster_volume"] = json.dumps([round(float(x), 8) for x in final_cluster_volume.tolist()])
            epoch_avgs["cluster_volume_min"] = float(final_cluster_volume.min()) if final_cluster_volume.size else 0.0
            epoch_avgs["cluster_volume_max"] = float(final_cluster_volume.max()) if final_cluster_volume.size else 0.0

            print(
                f"Epoch {epoch_idx + 1}/{self.epochs} finished in {epoch_time:.2f}s | "
                f"phase={self._loss_phase(epoch_idx)} "
                f"effective_lambda_com={epoch_avgs['effective_lambda_com']:.4f} "
                f"pi_asymmetry_norm={self.pi_asymmetry_norm:.4e} "
                f"pi_cut_density={self.pi_cut_density:.4e} "
                f"loss_total={epoch_avgs['loss_total']:.4f} "
                f"weighted_com={epoch_avgs['weighted_com']:.4f} "
                f"weighted_temp={epoch_avgs['weighted_temp']:.4f} "
                f"weighted_batch={epoch_avgs['weighted_batch']:.4f} "
                f"weighted_assign_penalty={epoch_avgs['weighted_assign_penalty']:.4f} "
                f"cluster_volume_min={epoch_avgs['cluster_volume_min']:.4f} "
                f"cluster_volume_max={epoch_avgs['cluster_volume_max']:.4f} "
                f"assignment_entropy={epoch_avgs['assignment_entropy']:.4f}"
            )

            should_eval = (epoch_idx + 1 == self.epochs) or (
                self.eval_interval > 0 and (epoch_idx + 1) % self.eval_interval == 0
            )
            if should_eval:
                last_eval = self.evaluate_and_log(epoch_idx + 1, epoch_avgs, probe_batch, save_final=(epoch_idx + 1 == self.epochs))

        total_time = time.time() - total_start
        print(f"Training finished in {total_time:.2f}s ({total_time / 60.0:.2f} min)")
        return {
            "epochs_trained": self.epochs,
            "embedding_path": self.emb_path,
            "prediction_path": self.pred_path,
            "soft_assignment_path": self.soft_assignment_path,
            "pi_path": self.pi_path,
            "pi_raw_path": self.pi_raw_path,
            "pi_cut_path": self.pi_cut_path,
            "metrics_csv_path": self.metrics_csv_path,
            "prediction_paths": self.final_prediction_paths,
            "last_eval": last_eval["metric_row"] if last_eval is not None else None,
        }
