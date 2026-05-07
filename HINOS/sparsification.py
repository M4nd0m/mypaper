import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix, diags, load_npz, save_npz

from data_load import compress_time_indices, read_temporal_edges
from utils import cache_paths


EdgePair = Tuple[int, int]


def count_static_edges(tadj_list: Dict[int, Dict[int, set]]) -> int:
    seen = set()
    for u, nbrs in tadj_list.items():
        for v in nbrs.keys():
            a, b = (u, v) if u < v else (v, u)
            if a != b:
                seen.add((a, b))
    return len(seen)


def sqrt_static_edge_budget(static_edges: int) -> int:
    if static_edges <= 0:
        return 0
    return max(1, int(math.ceil(math.sqrt(float(static_edges)))))


def count_temporal_nodes(tadj_list: Dict[int, Dict[int, set]]) -> int:
    all_nodes = set(tadj_list.keys())
    for u in tadj_list:
        all_nodes.update(tadj_list[u].keys())
    return (max(all_nodes) + 1) if all_nodes else 0


def taps_budget(args, tadj_list: Dict[int, Dict[int, set]]) -> Tuple[int, int, int]:
    static_edges = count_static_edges(tadj_list)
    num_nodes = count_temporal_nodes(tadj_list)
    mode = getattr(args, "taps_budget_mode", "sqrt_edges")
    beta = float(getattr(args, "taps_budget_beta", 1.0))
    if beta < 0:
        raise ValueError("--taps_budget_beta must be non-negative.")
    if mode == "sqrt_edges":
        taps_N = sqrt_static_edge_budget(static_edges)
    elif mode == "nlogn":
        # For full-graph TPPR-Cut, the affinity enters normalized cut. An
        # n log n path budget better tracks spectral sparsification scale than
        # the efficiency-first sqrt(static_edges) community-search budget.
        taps_N = int(math.ceil(beta * float(num_nodes) * math.log(float(num_nodes) + 1.0))) if num_nodes > 0 else 0
    else:
        raise ValueError(f"Unsupported taps_budget_mode: {mode}")
    return int(taps_N), int(static_edges), int(num_nodes)


def build_B(edges: List[EdgePair], num_nodes: int) -> csr_matrix:
    m = len(edges)
    if m == 0:
        return csr_matrix((0, num_nodes), dtype=np.float32)
    rows = np.arange(m, dtype=np.int32)
    cols = np.fromiter((int(v) for _, v in edges), count=m, dtype=np.int32)
    data = np.ones(m, dtype=np.float32)
    return coo_matrix((data, (rows, cols)), shape=(m, num_nodes), dtype=np.float32).tocsr()


def build_XQ_tilde(edges: List[EdgePair], num_nodes: int) -> csr_matrix:
    m = len(edges)
    if m == 0:
        return csr_matrix((num_nodes, 0), dtype=np.float32)
    src = np.fromiter((int(u) for u, _ in edges), count=m, dtype=np.int32)
    outdeg = np.bincount(src, minlength=num_nodes).astype(np.int64)
    rows = src
    cols = np.arange(m, dtype=np.int32)
    data = np.zeros(m, dtype=np.float32)
    nz = outdeg[rows] > 0
    data[nz] = 1.0 / outdeg[rows[nz]]
    return coo_matrix((data, (rows, cols)), shape=(num_nodes, m), dtype=np.float32).tocsr()


def build_P_from_edges(edges: List[EdgePair], num_nodes: int, weights=None) -> csr_matrix:
    m = len(edges)
    if m == 0 or num_nodes <= 0:
        return sp.identity(num_nodes, format="csr", dtype=np.float64)
    data = np.ones(m, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    rows = np.fromiter((int(u) for u, _ in edges), count=m, dtype=np.int32)
    cols = np.fromiter((int(v) for _, v in edges), count=m, dtype=np.int32)
    A = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float64)
    A.sum_duplicates()
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_deg = np.zeros_like(deg)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]
    P = diags(inv_deg) @ A
    if np.any(~mask):
        P = P + diags((~mask).astype(np.float64))
    return P


def build_L_alpha_from_edges(edges: List[EdgePair], num_nodes: int, alpha: float, weights=None) -> csr_matrix:
    P = build_P_from_edges(edges, num_nodes, weights=weights)
    deg = np.asarray(P.sum(axis=1)).ravel()
    D = diags(deg, 0)
    return D - alpha * P


def tppr_truncated_with_B_XQ(
    P: csr_matrix,
    B: csr_matrix,
    XQ: csr_matrix,
    D_inv_L_alpha_XQ: csr_matrix,
    alpha: float,
    K: int,
) -> csr_matrix:
    S = (XQ @ B).tocsr()
    Y = (alpha * S).tocsr()
    correction = (XQ - D_inv_L_alpha_XQ) @ B
    Y = (Y + alpha * correction).tocsr()

    T = S
    coef = 1.0 - alpha
    for r in range(1, K + 1):
        T = (T @ P).tocsr()
        if T.nnz == 0:
            break
        Y = (Y + alpha * (coef ** r) * T).tocsr()
    Y.sum_duplicates()
    Y.eliminate_zeros()
    return Y


class TimeAwarePathSparsifier:
    def __init__(
        self,
        tadj_list: Dict[int, Dict[int, set]],
        T: int,
        alpha: float,
        tau_eps: float,
        rng_seed: int,
        T_cap: Optional[int],
        N: int,
    ):
        self.tadj = tadj_list
        self.T = int(T)
        if T_cap is not None and int(T_cap) > 0:
            self.T = min(self.T, int(T_cap))
        self.alpha = float(alpha)
        self.tau_eps = float(tau_eps)
        self.rng = np.random.RandomState(rng_seed)
        self.N = int(N)

        self.edges_static = []
        seen = set()
        for u, nbrs in self.tadj.items():
            for v in nbrs.keys():
                a, b = (u, v) if u < v else (v, u)
                if a != b and (a, b) not in seen:
                    seen.add((a, b))
                    self.edges_static.append((a, b))
        self.m = len(self.edges_static)

        self.num_nodes = count_temporal_nodes(self.tadj)

        if self.T > 0:
            rs = np.arange(1, self.T + 1, dtype=float)
            pr = self.alpha * np.power((1.0 - self.alpha), rs)
            pr = pr / pr.sum()
            self.r_values = rs.astype(int).tolist()
            self.r_cum = np.cumsum(pr).tolist()
        else:
            self.r_values = [1]
            self.r_cum = [1.0]

    def _sample_start_edge(self) -> EdgePair:
        return self.edges_static[self.rng.randint(self.m)]

    def _time_aware_next(self, x: int, t_ref: int):
        nbrs = self.tadj.get(x, {})
        if not nbrs:
            return x, t_ref
        weights, candidates, chosen_tau = [], [], []
        for y, times in nbrs.items():
            tau_star = min(times, key=lambda tt: abs(tt - t_ref))
            weight = 1.0 / (abs(tau_star - t_ref) + self.tau_eps)
            weights.append(weight)
            candidates.append(y)
            chosen_tau.append(tau_star)
        total_weight = sum(weights)
        if total_weight <= 0:
            idx = self.rng.randint(len(candidates))
            return candidates[idx], chosen_tau[idx]
        r = self.rng.random_sample() * total_weight
        acc = 0.0
        for y, w, tau_star in zip(candidates, weights, chosen_tau):
            acc += w
            if r <= acc:
                return y, tau_star
        return candidates[-1], chosen_tau[-1]

    def _sample_r(self) -> int:
        if self.T <= 1:
            return 1
        x = self.rng.random_sample()
        lo, hi = 0, len(self.r_cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if x <= self.r_cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        return int(self.r_values[lo])

    def _path_sampling(self, e: EdgePair, r: int):
        u, v = e
        times_uv = list(self.tadj[u][v])
        t0 = times_uv[self.rng.randint(len(times_uv))]
        j = self.rng.randint(1, r + 1)

        cur_u, t_u = u, t0
        Zp = 0.0
        for _ in range(max(0, j - 1)):
            nxt, t_u = self._time_aware_next(cur_u, t_u)
            A = len(self.tadj.get(cur_u, {}).get(nxt, []))
            if A > 0:
                Zp += 2.0 / float(A)
            cur_u = nxt
        n0 = cur_u

        A_uv = len(self.tadj[u][v])
        if A_uv > 0:
            Zp += 2.0 / float(A_uv)

        cur_v, t_v = v, t0
        for _ in range(max(0, r - j)):
            nxt, t_v = self._time_aware_next(cur_v, t_v)
            A = len(self.tadj.get(cur_v, {}).get(nxt, []))
            if A > 0:
                Zp += 2.0 / float(A)
            cur_v = nxt
        nr = cur_v
        return n0, nr, max(Zp, 1.0)

    def construct_sparsifier(self) -> csr_matrix:
        if self.m == 0 or self.num_nodes == 0:
            return sp.identity(self.num_nodes, format="csr", dtype=np.float64)

        accum = {}
        for _ in range(self.N):
            u, v = self._sample_start_edge()
            r = self._sample_r()
            n0, nr, Zp_path = self._path_sampling((u, v), r)
            w = (2.0 * r * self.m) / (self.N * Zp_path)
            a, b = (n0, nr) if n0 <= nr else (nr, n0)
            accum[(a, b)] = accum.get((a, b), 0.0) + w

        rows, cols, data = [], [], []
        for (a, b), w in accum.items():
            rows.append(a)
            cols.append(b)
            data.append(w)
            if a != b:
                rows.append(b)
                cols.append(a)
                data.append(w)

        A = coo_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_nodes), dtype=np.float64).tocsr()
        A.sum_duplicates()
        A.eliminate_zeros()
        return A


def taps_cache_config(args, tadj_list):
    taps_N, static_edges, num_nodes = taps_budget(args, tadj_list)
    budget_mode = getattr(args, "taps_budget_mode", "sqrt_edges")
    budget_beta = float(getattr(args, "taps_budget_beta", 1.0))
    cfg = {
        "alpha": float(args.taps_alpha),
        "N_mode": str(budget_mode),
        "taps_budget_mode": str(budget_mode),
        "taps_budget_beta": float(budget_beta),
        "N": int(taps_N),
        "computed_N_TAPS": int(taps_N),
        "static_edges": int(static_edges),
        "num_nodes": int(num_nodes),
        "tau_eps": float(args.taps_tau_eps),
        "rng": int(args.taps_rng_seed),
        "Tcap": int(args.taps_T_cap) if int(args.taps_T_cap) > 0 else -1,
    }
    return cfg


def compute_taps_cached(
    file_path: str,
    args,
    dataset: str,
    tadj_list=None,
    T_total=None,
) -> csr_matrix:
    if tadj_list is None or T_total is None:
        edges_uvt = read_temporal_edges(file_path)
        tadj_list, T_total = compress_time_indices(edges_uvt)

    tppr_cfg = {"tppr_alpha": float(args.tppr_alpha), "tppr_K": int(args.tppr_K)}
    taps_cfg = taps_cache_config(args, tadj_list)
    taps_path, _ = cache_paths(args.cache_dir, dataset, tppr_cfg, taps_cfg)
    if os.path.exists(taps_path):
        return load_npz(taps_path).tocsr()

    A_time = TimeAwarePathSparsifier(
        tadj_list=tadj_list,
        T=T_total,
        alpha=args.taps_alpha,
        tau_eps=args.taps_tau_eps,
        rng_seed=args.taps_rng_seed,
        T_cap=args.taps_T_cap,
        N=taps_cfg["computed_N_TAPS"],
    ).construct_sparsifier()
    save_npz(taps_path, A_time)
    return A_time.tocsr()


def compute_tppr_cached(
    file_path: str,
    num_nodes: int,
    args,
    dataset: str,
    tadj_list=None,
    T_total=None,
) -> csr_matrix:
    if tadj_list is None or T_total is None:
        edges_uvt = read_temporal_edges(file_path)
        tadj_list, T_total = compress_time_indices(edges_uvt)

    tppr_cfg = {"tppr_alpha": float(args.tppr_alpha), "tppr_K": int(args.tppr_K)}
    taps_path, tppr_path = cache_paths(args.cache_dir, dataset, tppr_cfg, taps_cache_config(args, tadj_list))
    if os.path.exists(tppr_path):
        return load_npz(tppr_path).tocsr()

    if os.path.exists(taps_path):
        A_time = load_npz(taps_path).tocsr()
    else:
        A_time = compute_taps_cached(
            file_path,
            args,
            dataset,
            tadj_list=tadj_list,
            T_total=T_total,
        )

    A_time.eliminate_zeros()
    rows, cols = A_time.nonzero()
    edges = list(zip(rows.tolist(), cols.tolist()))
    weights = A_time.data.tolist()

    B = build_B(edges, num_nodes)
    XQ = build_XQ_tilde(edges, num_nodes)
    P = build_P_from_edges(edges, num_nodes, weights=weights)

    deg = np.asarray(P.sum(axis=1)).ravel()
    inv_deg = np.zeros_like(deg)
    inv_deg[deg > 0] = 1.0 / deg[deg > 0]
    D_inv = diags(inv_deg, format="csr")

    L_alpha = build_L_alpha_from_edges(edges, num_nodes, args.tppr_alpha, weights=weights)
    D_inv_L_alpha_XQ = D_inv @ (L_alpha @ XQ)
    tppr_mat = tppr_truncated_with_B_XQ(P, B, XQ, D_inv_L_alpha_XQ, args.tppr_alpha, args.tppr_K)
    save_npz(tppr_path, tppr_mat.tocsr())
    return tppr_mat.tocsr()


def build_ncut_graph(tppr_mat: csr_matrix) -> csr_matrix:
    W = tppr_mat.tocsr().astype(np.float64)
    W = 0.5 * (W + W.T)
    W = W.tocsr()
    W.sum_duplicates()
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W
