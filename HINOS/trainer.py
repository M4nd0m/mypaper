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
from graph_diagnostics import hard_partition_ncut
from sparsification import build_ncut_graph, compute_tppr_cached
from utils import choose_device, ensure_dir


METRIC_COLUMNS = [
    "epoch",
    "phase",
    "objective_mode",
    "pi_asymmetry_norm",
    "pi_cut_density",
    "purity_at_5_pi",
    "purity_at_10_pi",
    "purity_at_20_pi",
    "leakage_at_5_pi",
    "leakage_at_10_pi",
    "leakage_at_20_pi",
    "ncut_gt_pi",
    "ncut_pred_pi",
    "ncut_pred_over_gt_pi",
    "loss_total",
    "loss_tppr_cut",
    "loss_tppr_cut_raw",
    "loss_com",
    "loss_cut_base",
    "loss_temp",
    "loss_batch",
    "loss_ncut_orth",
    "weighted_com",
    "weighted_cut",
    "weighted_temp",
    "weighted_batch",
    "weighted_ncut_orth",
    "lambda_com",
    "lambda_ncut_orth",
    "effective_lambda_com",
    "prototype_alpha",
    "prototype_lr_scale",
    "cluster_volume",
    "cluster_volume_min",
    "cluster_volume_max",
    "cluster_volume_entropy",
    "assignment_entropy",
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
    "spec_gap_c",
    "spec_gap_c1",
    "spec_gap_ratio",
]


class TGCTrainer:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.device = choose_device(args.device)
        self.objective_mode = args.objective_mode
        self.eval_interval = int(args.eval_interval)
        self.spectral_topk = int(args.spectral_topk)
        self.seed = int(args.seed)
        self.prototype_alpha = float(args.prototype_alpha)
        self.main_pred_mode = args.main_pred_mode
        self.freeze_prototypes = bool(args.freeze_prototypes)
        self.prototype_lr_scale = float(args.prototype_lr_scale)

        self.file_path = os.path.join(args.data_root, self.dataset_name, f"{self.dataset_name}.txt")
        self.label_path = os.path.join(args.data_root, self.dataset_name, "node2label.txt")
        self.feature_path = os.path.join(args.pretrain_emb_dir, f"{self.dataset_name}_feature.emb")

        tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
        self.run_stem = f"{self.dataset_name}_TGC_{args.epoch}{tag_suffix}"
        self.emb_out_dir = os.path.join(args.emb_root, self.dataset_name)
        self.emb_path = os.path.join(self.emb_out_dir, f"{self.run_stem}.emb")
        self.pred_path = os.path.join(self.emb_out_dir, f"{self.run_stem}_pred.txt")
        self.soft_assignment_path = os.path.join(self.emb_out_dir, f"{self.run_stem}_soft_assign.npy")
        self.metrics_csv_path = os.path.join(self.emb_out_dir, f"{self.run_stem}_metrics.csv")

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
        self.args.resolved_num_clusters = int(self.num_clusters)

        self.node_emb = nn.Parameter(torch.from_numpy(self.feature).float().to(self.device))
        self.pre_emb = torch.from_numpy(self.feature).float().to(self.device)
        self.delta = nn.Parameter((torch.zeros(self.node_dim) + 1.0).float().to(self.device))

        edges_uvt = read_temporal_edges(self.file_path)
        self.tadj_list, self.T_total = compress_time_indices(edges_uvt)

        self.initial_cluster_centers = KMeans(
            n_clusters=self.num_clusters,
            n_init=10,
            random_state=self.seed,
        ).fit(self.feature).cluster_centers_.astype(np.float32)
        self.initial_cluster_centers_tensor = torch.from_numpy(self.initial_cluster_centers).float().to(self.device)
        self.cluster_prototypes = nn.Parameter(torch.from_numpy(self.initial_cluster_centers).float().to(self.device))

        self.lambda_com = float(args.lambda_com)
        self.lambda_ncut_orth = float(args.lambda_ncut_orth) if args.lambda_ncut_orth is not None else 0.0

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
        self.K = int(self.num_clusters)
        self.eyeK = torch.eye(self.K, device=self.device)
        self.old_ncut_orth_target = self.eyeK / math.sqrt(float(self.K))
        self._prepare_full_ncut_tensors()
        self.tppr_label_diagnostics = self._compute_tppr_label_diagnostics()

        self.neg_size = int(args.neg_size)
        self.hist_len = int(args.hist_len)
        self.batch = int(args.batch_size)
        self.epochs = int(args.epoch)
        self.num_workers = int(args.num_workers)

        params = [{"params": [self.node_emb, self.delta], "lr": args.learning_rate}]
        if not self.freeze_prototypes:
            params.append(
                {
                    "params": [self.cluster_prototypes],
                    "lr": args.learning_rate * self.prototype_lr_scale,
                }
            )
        self.opt = Adam(params=params)

        self.static_predictions: Dict[str, Optional[np.ndarray]] = {}
        self._init_metrics_csv()
        print(f"[Device] resolved        = {self.device}")
        print(f"[Assign] mode            = prototype")
        print(f"[Assign] freeze_proto    = {self.freeze_prototypes}")
        print(
            "[TPPR Diag] "
            f"purity@10={self.tppr_label_diagnostics['purity_at_10_pi']:.4f} "
            f"leakage@10={self.tppr_label_diagnostics['leakage_at_10_pi']:.4f} "
            f"ncut_gt={self.tppr_label_diagnostics['ncut_gt_pi']:.4f}"
        )

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
        self.sqrt_pi_cut_degree = torch.sqrt(self.full_ncut_degree.clamp_min(0.0))
        self.two_m_pi = torch.clamp(self.full_ncut_degree.sum(), min=1e-6)

        pi = self.tppr_mat.tocsr().astype(np.float64)
        asym = (pi - pi.T).tocsr()
        pi_fro = float(np.sqrt(np.sum(pi.data ** 2))) if pi.nnz > 0 else 0.0
        asym_fro = float(np.sqrt(np.sum(asym.data ** 2))) if asym.nnz > 0 else 0.0
        self.pi_is_symmetric = bool(asym.nnz == 0)
        self.pi_asymmetry_norm = asym_fro / (pi_fro + 1e-12)
        self.pi_cut_nnz = int(self.pi_cut_csr.nnz)
        total_entries = float(self.pi_cut_csr.shape[0] * self.pi_cut_csr.shape[1])
        self.pi_cut_density = float(self.pi_cut_nnz / total_entries) if total_entries > 0 else 0.0

    def _label_array_or_none(self) -> Optional[np.ndarray]:
        if self.labels is None:
            return None
        labels = np.empty((self.node_dim,), dtype=np.int64)
        for node_idx in range(self.node_dim):
            if node_idx not in self.labels:
                return None
            labels[node_idx] = int(self.labels[node_idx])
        return labels

    def _compute_tppr_purity_at_k(self, labels: np.ndarray, k: int) -> float:
        W = self.pi_cut_csr.tocsr()
        total = 0.0
        for node_idx in range(self.node_dim):
            row_start, row_end = W.indptr[node_idx], W.indptr[node_idx + 1]
            neigh = W.indices[row_start:row_end]
            weights = W.data[row_start:row_end]
            valid = neigh != node_idx
            neigh = neigh[valid]
            weights = weights[valid]
            if neigh.size == 0:
                continue
            if neigh.size > k:
                top_pos = np.argpartition(weights, -k)[-k:]
                neigh = neigh[top_pos]
                weights = weights[top_pos]
                order = np.argsort(weights)[::-1]
                neigh = neigh[order]
            total += float(np.mean(labels[neigh] == labels[node_idx]))
        return total / float(max(1, self.node_dim))

    def _compute_gt_ncut_pi(self, labels: np.ndarray) -> float:
        W = self.pi_cut_csr.tocsr()
        degree = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)
        unique_labels = np.unique(labels)
        ncut = 0.0
        eps = 1e-12
        for label in unique_labels:
            in_cluster = labels == label
            volume = float(degree[in_cluster].sum())
            if volume <= eps:
                continue
            internal_assoc = 0.0
            cluster_nodes = np.flatnonzero(in_cluster)
            for node_idx in cluster_nodes:
                row_start, row_end = W.indptr[node_idx], W.indptr[node_idx + 1]
                neigh = W.indices[row_start:row_end]
                weights = W.data[row_start:row_end]
                internal_assoc += float(weights[in_cluster[neigh]].sum())
            cut = max(0.0, volume - internal_assoc)
            ncut += cut / (volume + eps)
        return float(ncut)

    def _compute_tppr_label_diagnostics(self) -> Dict[str, float]:
        diag = {
            "purity_at_5_pi": float("nan"),
            "purity_at_10_pi": float("nan"),
            "purity_at_20_pi": float("nan"),
            "leakage_at_5_pi": float("nan"),
            "leakage_at_10_pi": float("nan"),
            "leakage_at_20_pi": float("nan"),
            "ncut_gt_pi": float("nan"),
        }
        labels = self._label_array_or_none()
        if labels is None:
            return diag
        for k in (5, 10, 20):
            purity = self._compute_tppr_purity_at_k(labels, k)
            diag[f"purity_at_{k}_pi"] = purity
            diag[f"leakage_at_{k}_pi"] = 1.0 - purity
        diag["ncut_gt_pi"] = self._compute_gt_ncut_pi(labels)
        return diag

    def _assignment(self) -> torch.Tensor:
        return self._student_t_assignment(self.node_emb, self.cluster_prototypes)

    def _student_t_assignment(self, emb: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        alpha = max(self.prototype_alpha, eps)
        dist = torch.cdist(emb, centers, p=2).pow(2)
        q = (1.0 + dist / alpha).pow(-(alpha + 1.0) / 2.0)
        q = q.clamp_min(eps)
        return q / q.sum(dim=1, keepdim=True).clamp_min(eps)

    def _fixed_graph_bundle(self) -> Dict[str, torch.Tensor]:
        return {
            "edge_i": self.full_ncut_edge_i,
            "edge_j": self.full_ncut_edge_j,
            "edge_w": self.full_ncut_edge_w,
            "degree": self.full_ncut_degree,
        }

    def _compute_search_ncut(self, assign: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph = self._fixed_graph_bundle()
        edge_i = graph["edge_i"]
        edge_j = graph["edge_j"]
        edge_w = graph["edge_w"]
        degree = graph["degree"]
        zero = torch.tensor(0.0, device=self.device)
        if edge_i.numel() == 0:
            return zero, zero, zero

        eps = 1e-6
        delta = assign.index_select(0, edge_i) - assign.index_select(0, edge_j)
        l_mat = delta.t().mm(edge_w.unsqueeze(1) * delta)
        g_mat = assign.t().mm(degree.unsqueeze(1) * assign)
        cut = torch.trace(torch.linalg.solve(g_mat + eps * self.eyeK, l_mat)) / (float(self.K) + 1e-12)
        fro = torch.linalg.norm(g_mat, ord="fro") + eps
        orth = torch.linalg.norm(g_mat / fro - self.old_ncut_orth_target, ord="fro")
        ncut = cut + self.lambda_ncut_orth * orth
        return ncut.squeeze(), cut.squeeze(), orth.squeeze()

    def _loss_phase(self, epoch_idx: int) -> str:
        return self.objective_mode

    def _compute_search_proto_loss_terms(
        self,
        s_nodes: torch.Tensor,
        t_nodes: torch.Tensor,
        t_times: torch.Tensor,
        n_nodes: torch.Tensor,
        h_nodes: torch.Tensor,
        h_times: torch.Tensor,
        h_time_mask: torch.Tensor,
        num_batches: int = 1,
    ) -> Dict[str, object]:
        batch = s_nodes.size(0)
        s_node_emb = self.node_emb.index_select(0, s_nodes.view(-1)).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, t_nodes.view(-1)).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, h_nodes.view(-1)).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb.index_select(0, n_nodes.view(-1)).view(batch, self.neg_size, -1)

        assign = self._assignment()
        eps = 1e-6

        new_st_adj = torch.cosine_similarity(s_node_emb, t_node_emb)
        res_st_loss = torch.norm(1.0 - new_st_adj, p=2, dim=0)
        new_sh_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), h_node_emb, dim=2)
        new_sh_adj = new_sh_adj * h_time_mask
        new_sn_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), n_node_emb, dim=2)
        res_sh_loss = torch.norm(1.0 - new_sh_adj, p=2, dim=0).sum(dim=0)
        res_sn_loss = torch.norm(new_sn_adj, p=2, dim=0).sum(dim=0)
        l_rec = (res_st_loss + res_sh_loss + res_sn_loss) / max(1, batch)

        att = F.softmax(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2).neg(), dim=1)
        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()
        delta = self.delta.index_select(0, s_nodes.view(-1)).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)
        p_lambda = p_mu + (att * p_alpha * torch.exp(delta * d_time) * h_time_mask).sum(dim=1)

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()
        time_decay = torch.exp(delta * d_time).unsqueeze(2)
        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * time_decay * h_time_mask.unsqueeze(2)).sum(dim=1)
        loss_pair = -torch.log(p_lambda.sigmoid() + eps) - torch.log(n_lambda.neg().sigmoid() + eps).sum(dim=1)
        l_temp = loss_pair.mean()

        l_ncut, l_cut_base, l_ncut_orth = self._compute_search_ncut(assign)
        weighted_com = self.lambda_com * l_ncut
        loss_total = l_rec + l_temp + weighted_com

        degree = self.full_ncut_degree
        cluster_mass = assign.sum(dim=0)
        cluster_volume = assign.t().mm(degree.unsqueeze(1)).squeeze(1)
        volume_prob = cluster_volume / (cluster_volume.sum() + eps)
        assignment_entropy = -(assign * torch.log(assign + eps)).sum(dim=1).mean()
        weighted_ncut_orth = self.lambda_com * self.lambda_ncut_orth * l_ncut_orth

        return {
            "phase": self.objective_mode,
            "effective_lambda_com": self.lambda_com,
            "loss_total": loss_total,
            "loss_tppr_cut": l_cut_base,
            "loss_tppr_cut_raw": float(l_cut_base.detach().item()),
            "loss_com": l_ncut,
            "loss_cut_base": l_cut_base,
            "loss_temp": l_temp,
            "loss_batch": l_rec,
            "loss_ncut_orth": l_ncut_orth,
            "weighted_com": weighted_com,
            "weighted_cut_base": self.lambda_com * l_cut_base,
            "weighted_cut": self.lambda_com * l_cut_base,
            "weighted_temp": l_temp,
            "weighted_batch": l_rec,
            "weighted_ncut_orth": weighted_ncut_orth,
            "cluster_mass": cluster_mass.detach().cpu().numpy(),
            "cluster_volume": cluster_volume.detach().cpu().numpy(),
            "cluster_volume_min": float(cluster_volume.min().item()),
            "cluster_volume_max": float(cluster_volume.max().item()),
            "cluster_volume_entropy": float((-(volume_prob * torch.log(volume_prob + eps)).sum()).item()),
            "assignment_entropy": float(assignment_entropy.item()),
        }

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
        num_batches: int = 1,
    ) -> Dict[str, object]:
        return self._compute_search_proto_loss_terms(
            s_nodes,
            t_nodes,
            t_times,
            n_nodes,
            h_nodes,
            h_times,
            h_time_mask,
            num_batches=num_batches,
        )

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

    def update(self, batch_tensors: Tuple[torch.Tensor, ...], epoch_idx: int, num_batches: int = 1) -> Dict[str, float]:
        self.opt.zero_grad()
        loss_terms = self.compute_loss_terms(*batch_tensors, epoch_idx, num_batches=num_batches)
        loss_terms["loss_total"].backward()
        self.opt.step()
        return {
            "loss_total": float(loss_terms["loss_total"].item()),
            "loss_tppr_cut": float(loss_terms["loss_tppr_cut"].item()),
            "loss_tppr_cut_raw": float(loss_terms["loss_tppr_cut_raw"]),
            "loss_com": float(loss_terms["loss_com"].item()),
            "loss_cut_base": float(loss_terms["loss_cut_base"].item()),
            "loss_temp": float(loss_terms["loss_temp"].item()),
            "loss_batch": float(loss_terms["loss_batch"].item()),
            "loss_ncut_orth": float(loss_terms["loss_ncut_orth"].item()),
            "effective_lambda_com": float(loss_terms["effective_lambda_com"]),
            "weighted_com": float(loss_terms["weighted_com"].item()),
            "weighted_cut_base": float(loss_terms["weighted_cut_base"].item()),
            "weighted_cut": float(loss_terms["weighted_cut"].item()),
            "weighted_temp": float(loss_terms["weighted_temp"].item()),
            "weighted_batch": float(loss_terms["weighted_batch"].item()),
            "weighted_ncut_orth": float(loss_terms["weighted_ncut_orth"].item()),
            "cluster_mass": np.asarray(loss_terms["cluster_mass"], dtype=np.float64),
            "cluster_volume": np.asarray(loss_terms["cluster_volume"], dtype=np.float64),
            "cluster_volume_min": float(loss_terms["cluster_volume_min"]),
            "cluster_volume_max": float(loss_terms["cluster_volume_max"]),
            "cluster_volume_entropy": float(loss_terms["cluster_volume_entropy"]),
            "assignment_entropy": float(loss_terms["assignment_entropy"]),
        }

    def _grad_norm(self) -> float:
        total = 0.0
        params = [self.node_emb, self.delta, self.cluster_prototypes]
        for param in params:
            if param.grad is None:
                continue
            total += float(torch.sum(param.grad.detach() ** 2).item())
        return math.sqrt(total)

    def compute_gradient_diagnostics(
        self, probe_batch: Tuple[torch.Tensor, ...], epoch_idx: int, num_batches: int = 1
    ) -> Dict[str, float]:
        grad_norms = {"grad_norm_cut": 0.0, "grad_norm_temp": 0.0, "grad_norm_batch": 0.0}
        component_map = {
            "grad_norm_cut": "weighted_cut",
            "grad_norm_temp": "weighted_temp",
            "grad_norm_batch": "weighted_batch",
        }
        for out_name, term_name in component_map.items():
            self.opt.zero_grad()
            loss_terms = self.compute_loss_terms(*probe_batch, epoch_idx, num_batches=num_batches)
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

    def evaluate_and_log(
        self,
        epoch_num: int,
        epoch_stats: Dict[str, float],
        save_final: bool = False,
        num_batches: int = 1,
    ) -> Dict[str, object]:
        emb = self.node_emb.detach().cpu().numpy().astype(np.float32)
        assign = self._assignment().detach().cpu().numpy().astype(np.float32)
        predictions = {
            "argmax_s": np.argmax(assign, axis=1).astype(np.int64),
            "kmeans_z": kmeans_predict(emb, self.K, random_state=self.seed),
            "kmeans_s": kmeans_predict(assign, self.K, random_state=self.seed),
        }
        predictions.update(self._static_predictions())

        if save_final:
            main_pred = predictions[self.main_pred_mode]
            ensure_dir(os.path.dirname(self.pred_path))
            save_prediction_file(self.pred_path, main_pred)
            np.save(self.soft_assignment_path, assign)
            self.save_node_embeddings(self.emb_path)

        metric_row = {
            "epoch": int(epoch_num),
            "phase": self._loss_phase(epoch_num - 1),
            "objective_mode": self.objective_mode,
            "pi_asymmetry_norm": self.pi_asymmetry_norm,
            "pi_cut_density": self.pi_cut_density,
            "purity_at_5_pi": self.tppr_label_diagnostics["purity_at_5_pi"],
            "purity_at_10_pi": self.tppr_label_diagnostics["purity_at_10_pi"],
            "purity_at_20_pi": self.tppr_label_diagnostics["purity_at_20_pi"],
            "leakage_at_5_pi": self.tppr_label_diagnostics["leakage_at_5_pi"],
            "leakage_at_10_pi": self.tppr_label_diagnostics["leakage_at_10_pi"],
            "leakage_at_20_pi": self.tppr_label_diagnostics["leakage_at_20_pi"],
            "ncut_gt_pi": self.tppr_label_diagnostics["ncut_gt_pi"],
            "ncut_pred_pi": float("nan"),
            "ncut_pred_over_gt_pi": float("nan"),
            "loss_total": epoch_stats["loss_total"],
            "loss_tppr_cut": epoch_stats["loss_tppr_cut"],
            "loss_tppr_cut_raw": epoch_stats["loss_tppr_cut_raw"],
            "loss_com": epoch_stats["loss_com"],
            "loss_cut_base": epoch_stats["loss_cut_base"],
            "loss_temp": epoch_stats["loss_temp"],
            "loss_batch": epoch_stats["loss_batch"],
            "loss_ncut_orth": epoch_stats.get("loss_ncut_orth", 0.0),
            "weighted_com": epoch_stats["weighted_com"],
            "weighted_cut": epoch_stats["weighted_cut"],
            "weighted_temp": epoch_stats["weighted_temp"],
            "weighted_batch": epoch_stats["weighted_batch"],
            "weighted_ncut_orth": epoch_stats.get("weighted_ncut_orth", 0.0),
            "lambda_com": self.lambda_com,
            "lambda_ncut_orth": self.lambda_ncut_orth,
            "effective_lambda_com": epoch_stats["effective_lambda_com"],
            "prototype_alpha": self.prototype_alpha,
            "prototype_lr_scale": self.prototype_lr_scale,
            "cluster_volume": epoch_stats["cluster_volume"],
            "cluster_volume_min": epoch_stats["cluster_volume_min"],
            "cluster_volume_max": epoch_stats["cluster_volume_max"],
            "cluster_volume_entropy": epoch_stats["cluster_volume_entropy"],
            "assignment_entropy": epoch_stats["assignment_entropy"],
            "spec_gap_c": float("nan"),
            "spec_gap_c1": float("nan"),
            "spec_gap_ratio": float("nan"),
        }

        for method, pred in predictions.items():
            metrics = self._prediction_metrics(pred)
            prefix = method
            metric_row[f"acc_{prefix}"] = metrics["ACC"]
            metric_row[f"nmi_{prefix}"] = metrics["NMI"]
            metric_row[f"ari_{prefix}"] = metrics["ARI"]
            metric_row[f"f1_{prefix}"] = metrics["F1"]

        if self.labels is not None and predictions.get("spectral_pi") is not None:
            ncut_pred_pi = hard_partition_ncut(self.pi_cut_csr, predictions["spectral_pi"])
            metric_row["ncut_pred_pi"] = ncut_pred_pi
            metric_row["ncut_pred_over_gt_pi"] = ncut_pred_pi / (metric_row["ncut_gt_pi"] + 1e-12)

        with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as writer_file:
            writer = csv.DictWriter(writer_file, fieldnames=METRIC_COLUMNS)
            writer.writerow(metric_row)

        main_method = self.main_pred_mode
        print(
            f"[Eval] epoch={epoch_num} phase={metric_row['phase']} "
            f"loss_total={metric_row['loss_total']:.4f} loss_tppr_cut={metric_row['loss_tppr_cut']:.4f} "
            f"loss_com={metric_row['loss_com']:.4f} "
            f"loss_ncut_orth={metric_row['loss_ncut_orth']:.4f} "
            f"loss_temp={metric_row['loss_temp']:.4f} loss_batch={metric_row['loss_batch']:.4f} "
            f"effective_lambda_com={metric_row['effective_lambda_com']:.4f} "
            f"vol_min={metric_row['cluster_volume_min']:.4f} "
            f"vol_max={metric_row['cluster_volume_max']:.4f} "
            f"entropy={metric_row['assignment_entropy']:.4f} "
            f"{main_method}: ACC={metric_row[f'acc_{main_method}']:.4f} "
            f"NMI={metric_row[f'nmi_{main_method}']:.4f} "
            f"ARI={metric_row[f'ari_{main_method}']:.4f} "
            f"F1={metric_row[f'f1_{main_method}']:.4f}"
        )
        return {
            "predictions": predictions,
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
            epoch_sums = {
                "loss_total": 0.0,
                "loss_tppr_cut": 0.0,
                "loss_tppr_cut_raw": 0.0,
                "loss_com": 0.0,
                "loss_cut_base": 0.0,
                "loss_temp": 0.0,
                "loss_batch": 0.0,
                "loss_ncut_orth": 0.0,
                "effective_lambda_com": 0.0,
                "weighted_com": 0.0,
                "weighted_cut_base": 0.0,
                "weighted_cut": 0.0,
                "weighted_temp": 0.0,
                "weighted_batch": 0.0,
                "weighted_ncut_orth": 0.0,
                "cluster_volume_min": 0.0,
                "cluster_volume_max": 0.0,
                "cluster_volume_entropy": 0.0,
                "assignment_entropy": 0.0,
            }
            final_cluster_mass = None
            final_cluster_volume = None

            for i_batch, sample in enumerate(batch_bar, 1):
                batch_tensors = self._sample_to_device(sample)
                step_stats = self.update(batch_tensors, epoch_idx, num_batches=num_batches)
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
                f"loss_total={epoch_avgs['loss_total']:.4f} "
                f"loss_com={epoch_avgs['loss_com']:.4f} "
                f"loss_ncut_orth={epoch_avgs['loss_ncut_orth']:.4f} "
                f"loss_temp={epoch_avgs['loss_temp']:.4f} "
                f"loss_batch={epoch_avgs['loss_batch']:.4f} "
                f"vol_min={epoch_avgs['cluster_volume_min']:.4f} "
                f"vol_max={epoch_avgs['cluster_volume_max']:.4f} "
                f"entropy={epoch_avgs['assignment_entropy']:.4f}"
            )

            should_eval = (epoch_idx + 1 == self.epochs) or (
                self.eval_interval > 0 and (epoch_idx + 1) % self.eval_interval == 0
            )
            if should_eval:
                last_eval = self.evaluate_and_log(
                    epoch_idx + 1,
                    epoch_avgs,
                    save_final=(epoch_idx + 1 == self.epochs),
                    num_batches=num_batches,
                )

        total_time = time.time() - total_start
        print(f"Training finished in {total_time:.2f}s ({total_time / 60.0:.2f} min)")
        return {
            "epochs_trained": self.epochs,
            "embedding_path": self.emb_path,
            "prediction_path": self.pred_path,
            "soft_assignment_path": self.soft_assignment_path,
            "metrics_csv_path": self.metrics_csv_path,
            "last_eval": last_eval["metric_row"] if last_eval is not None else None,
        }
