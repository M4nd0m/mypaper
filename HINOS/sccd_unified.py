from typing import Dict

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix


class UnifiedSimilarity(nn.Module):
    """Sparse row-softmax U anchored to row-normalized W_pi."""

    def __init__(self, pi_cut_csr: csr_matrix, init_mode: str = "log_pi", device=None):
        super().__init__()
        if init_mode not in {"log_pi", "uniform"}:
            raise ValueError(f"Unsupported U_init_mode: {init_mode}")

        W = pi_cut_csr.tocsr().astype(np.float64)
        W.setdiag(0.0)
        W.eliminate_zeros()
        W.sort_indices()
        self.num_nodes = int(W.shape[0])
        self.nnz = int(W.nnz)
        self.device_ref = device

        row_counts = np.diff(W.indptr).astype(np.int64)
        rows = np.repeat(np.arange(self.num_nodes, dtype=np.int64), row_counts)
        cols = W.indices.astype(np.int64, copy=True)
        data = W.data.astype(np.float64, copy=True)

        row_sums = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)
        denom = np.maximum(row_sums[rows], 1e-12)
        pi_bar = data / denom
        if init_mode == "log_pi":
            theta0 = np.log(np.maximum(pi_bar, 1e-12)).astype(np.float32)
        else:
            theta0 = np.zeros_like(pi_bar, dtype=np.float32)

        edge_a = np.minimum(rows, cols)
        edge_b = np.maximum(rows, cols)
        valid = edge_a != edge_b
        rows = rows[valid]
        cols = cols[valid]
        pi_bar = pi_bar[valid]
        theta0 = theta0[valid]
        edge_a = edge_a[valid]
        edge_b = edge_b[valid]

        keys = edge_a.astype(np.int64) * np.int64(self.num_nodes) + edge_b.astype(np.int64)
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        undirected_i = (unique_keys // np.int64(self.num_nodes)).astype(np.int64)
        undirected_j = (unique_keys % np.int64(self.num_nodes)).astype(np.int64)

        target_device = device if device is not None else torch.device("cpu")
        self.register_buffer("row_index", torch.from_numpy(rows.astype(np.int64)).to(target_device))
        self.register_buffer("col_index", torch.from_numpy(cols.astype(np.int64)).to(target_device))
        self.register_buffer("pi_bar_data", torch.from_numpy(pi_bar.astype(np.float32)).to(target_device))
        self.register_buffer("undirected_map", torch.from_numpy(inverse.astype(np.int64)).to(target_device))
        self.register_buffer("edge_i", torch.from_numpy(undirected_i).to(target_device))
        self.register_buffer("edge_j", torch.from_numpy(undirected_j).to(target_device))
        self.theta = nn.Parameter(torch.from_numpy(theta0).to(target_device))

    def _row_softmax_values(self) -> torch.Tensor:
        max_per_row = torch.full(
            (self.num_nodes,),
            -torch.inf,
            dtype=self.theta.dtype,
            device=self.theta.device,
        )
        max_per_row.scatter_reduce_(0, self.row_index, self.theta, reduce="amax", include_self=True)
        shifted = self.theta - max_per_row.index_select(0, self.row_index)
        exp_values = torch.exp(shifted)
        denom = torch.zeros((self.num_nodes,), dtype=self.theta.dtype, device=self.theta.device)
        denom.scatter_add_(0, self.row_index, exp_values)
        return exp_values / denom.index_select(0, self.row_index).clamp_min(1e-12)

    def forward(self) -> Dict[str, torch.Tensor]:
        u_data = self._row_softmax_values()
        edge_w = torch.zeros(
            (self.edge_i.numel(),),
            dtype=u_data.dtype,
            device=u_data.device,
        )
        edge_w.index_add_(0, self.undirected_map, 0.5 * u_data)
        degree = torch.zeros((self.num_nodes,), dtype=u_data.dtype, device=u_data.device)
        degree.index_add_(0, self.edge_i, edge_w)
        degree.index_add_(0, self.edge_j, edge_w)
        return {
            "u_data": u_data,
            "edge_i": self.edge_i,
            "edge_j": self.edge_j,
            "edge_w": edge_w,
            "degree": degree,
        }

    def anchor_loss(self, u_data: torch.Tensor) -> torch.Tensor:
        return (u_data - self.pi_bar_data).pow(2).sum()

    def delta_fro(self, u_data: torch.Tensor) -> float:
        with torch.no_grad():
            num = torch.sqrt((u_data - self.pi_bar_data).pow(2).sum()).item()
            den = torch.sqrt(self.pi_bar_data.pow(2).sum()).item()
        return float(num / (den + 1e-12))

    def to_scipy(self) -> csr_matrix:
        with torch.no_grad():
            graph = self.forward()
            edge_i = graph["edge_i"].detach().cpu().numpy()
            edge_j = graph["edge_j"].detach().cpu().numpy()
            edge_w = graph["edge_w"].detach().cpu().numpy().astype(np.float64)
        rows = np.concatenate([edge_i, edge_j])
        cols = np.concatenate([edge_j, edge_i])
        vals = np.concatenate([edge_w, edge_w])
        W = sp.coo_matrix((vals, (rows, cols)), shape=(self.num_nodes, self.num_nodes), dtype=np.float64).tocsr()
        W.sum_duplicates()
        W.setdiag(0.0)
        W.eliminate_zeros()
        return W
