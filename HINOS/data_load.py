from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset

from utils import DEFAULT_NEG_TABLE_SIZE


TemporalEdge = Tuple[int, int, float]


def read_temporal_edges(txt_path: str) -> List[TemporalEdge]:
    edges_uvt: List[TemporalEdge] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                u, v, t = int(float(parts[0])), int(float(parts[1])), float(parts[2])
            except ValueError:
                continue
            if u != v:
                edges_uvt.append((u, v, t))
    return edges_uvt


def compress_time_indices(edges_uvt: List[TemporalEdge]):
    if not edges_uvt:
        return {}, 0
    ts = sorted(set(int(t) for _, _, t in edges_uvt))
    tmap = {t: i + 1 for i, t in enumerate(ts)}
    tadj_list = defaultdict(lambda: defaultdict(set))
    for u, v, t in edges_uvt:
        ti = tmap[int(t)]
        tadj_list[u][v].add(ti)
        tadj_list[v][u].add(ti)
    return dict(tadj_list), len(ts)


def read_labels(label_path: str, node_dim: int) -> np.ndarray:
    labels = np.zeros(node_dim, dtype=np.int64)
    with open(label_path, "r", encoding="utf-8") as reader:
        for line in reader:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            nid = int(parts[0])
            lid = int(parts[1])
            if 0 <= nid < node_dim:
                labels[nid] = lid
    return labels


def compute_birth_time(edges_uvt: List[TemporalEdge], node_dim: int) -> np.ndarray:
    birth_time = np.full((node_dim,), np.inf, dtype=np.float32)
    for u, v, t in edges_uvt:
        if 0 <= u < node_dim:
            birth_time[u] = min(birth_time[u], float(t))
        if 0 <= v < node_dim:
            birth_time[v] = min(birth_time[v], float(t))
    finite = np.isfinite(birth_time)
    if np.any(finite):
        birth_time[~finite] = float(np.max(birth_time[finite]) + 1.0)
    return birth_time


def build_static_adj(tadj_list: Dict[int, Dict[int, set]]):
    static_adj = defaultdict(list)
    for u, nbrs in tadj_list.items():
        for v in nbrs.keys():
            static_adj[u].append(v)
    return static_adj


def compute_temporal_total_volume(tadj_list: Dict[int, Dict[int, set]]) -> int:
    total = 0
    for u in tadj_list:
        for times in tadj_list[u].values():
            total += len(times)
    return total


class TGCDataSet(Dataset):
    def __init__(
        self,
        file_path: str,
        neg_size: int,
        hist_len: int,
        feature_path: str,
        directed: bool,
        neg_table_size: int = DEFAULT_NEG_TABLE_SIZE,
    ):
        self.neg_size = int(neg_size)
        self.hist_len = int(hist_len)
        self.directed = bool(directed)
        self.feature_path = feature_path
        self.neg_table_size = int(neg_table_size)

        self.max_d_time = float("-inf")
        self.neg_sampling_power = 0.75
        self.node2hist = {}
        self.node_set = set()
        self.degrees = {}
        self.edges_uvt: List[TemporalEdge] = []
        self.edge_num = 0

        with open(file_path, "r", encoding="utf-8") as infile:
            for line in infile:
                parts = line.split()
                if len(parts) < 3:
                    continue
                self.edge_num += 1
                s_node = int(parts[0])
                t_node = int(parts[1])
                d_time = float(parts[2])
                if s_node == t_node:
                    continue
                self.edges_uvt.append((s_node, t_node, d_time))
                self.node_set.update([s_node, t_node])

                self.node2hist.setdefault(s_node, []).append((t_node, d_time))
                if not self.directed:
                    self.node2hist.setdefault(t_node, []).append((s_node, d_time))

                self.max_d_time = max(self.max_d_time, d_time)
                self.degrees[s_node] = self.degrees.get(s_node, 0) + 1
                self.degrees[t_node] = self.degrees.get(t_node, 0) + 1

        self.node_dim = len(self.node_set)
        self.birth_time = compute_birth_time(self.edges_uvt, self.node_dim)
        self.risk_nodes_by_time = {}
        self.data_size = 0
        for s in self.node2hist:
            hist = sorted(self.node2hist[s], key=lambda x: x[1])
            self.node2hist[s] = hist
            self.data_size += len(hist)

        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0
        for s_node in self.node2hist:
            for t_idx in range(len(self.node2hist[s_node])):
                self.idx2source_id[idx] = s_node
                self.idx2target_id[idx] = t_idx
                idx += 1

        self.neg_table = np.zeros((self.neg_table_size,), dtype=np.int32)
        self.init_neg_table()

    def get_node_dim(self) -> int:
        return self.node_dim

    def get_edge_num(self) -> int:
        return self.edge_num

    def get_birth_time(self) -> np.ndarray:
        return self.birth_time.copy()

    def get_feature(self) -> np.ndarray:
        node_emb = {}
        with open(self.feature_path, "r", encoding="utf-8") as reader:
            first_line = reader.readline().strip().split()
            has_header = len(first_line) == 2 and all(tok.replace("-", "").isdigit() for tok in first_line)
            if not has_header and first_line:
                embeds = np.fromstring(" ".join(first_line), dtype=float, sep=" ")
                node_emb[int(embeds[0])] = embeds[1:]
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                embeds = np.fromstring(line, dtype=float, sep=" ")
                node_emb[int(embeds[0])] = embeds[1:]

        feature = [node_emb[i] for i in range(len(node_emb))]
        return np.asarray(feature, dtype=np.float32)

    def init_neg_table(self) -> None:
        tot_sum = 0.0
        for k in range(self.node_dim):
            tot_sum += np.power(self.degrees.get(k, 0), self.neg_sampling_power)

        n_id = 0
        cur_sum = 0.0
        por = 0.0
        for k in range(self.neg_table_size):
            if (k + 1.0) / self.neg_table_size > por:
                cur_sum += np.power(self.degrees.get(n_id, 0), self.neg_sampling_power)
                por = cur_sum / tot_sum
                n_id += 1
            self.neg_table[k] = n_id - 1

    def _risk_nodes(self, t_time: float, exclude_nodes=None) -> np.ndarray:
        key = float(t_time)
        if key not in self.risk_nodes_by_time:
            self.risk_nodes_by_time[key] = np.flatnonzero(self.birth_time <= key).astype(np.int64)
        risk_nodes = self.risk_nodes_by_time[key]
        if exclude_nodes is None or risk_nodes.size == 0:
            return risk_nodes
        exclude = set(int(x) for x in exclude_nodes)
        kept = [int(node) for node in risk_nodes.tolist() if int(node) not in exclude]
        if kept:
            return np.asarray(kept, dtype=np.int64)
        return risk_nodes

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx: int):
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        t_time = self.node2hist[s_node][t_idx][1]

        if t_idx - self.hist_len < 0:
            hist = self.node2hist[s_node][0:t_idx]
        else:
            hist = self.node2hist[s_node][t_idx - self.hist_len:t_idx]

        hist_nodes = [h[0] for h in hist]
        hist_times = [h[1] for h in hist]

        np_h_nodes = np.zeros((self.hist_len,), dtype=np.int64)
        np_h_nodes[:len(hist_nodes)] = hist_nodes
        np_h_times = np.zeros((self.hist_len,), dtype=np.float32)
        np_h_times[:len(hist_times)] = hist_times
        np_h_masks = np.zeros((self.hist_len,), dtype=np.float32)
        np_h_masks[:len(hist_nodes)] = 1.0

        return {
            "source_node": s_node,
            "target_node": t_node,
            "target_time": t_time,
            "history_nodes": np_h_nodes,
            "history_times": np_h_times,
            "history_masks": np_h_masks,
            "neg_nodes": self.negative_sampling(t_time, s_node, t_node),
            "risk_negative_resample_count": self.last_risk_negative_resample_count,
        }

    def negative_sampling(self, t_time: float, source_node: int, target_node: int) -> np.ndarray:
        negatives = []
        resample_count = 0
        exclude = {int(source_node), int(target_node)}
        max_attempts = max(100, self.neg_size * 50)

        while len(negatives) < self.neg_size and resample_count < max_attempts:
            candidate = int(self.neg_table[np.random.randint(0, self.neg_table_size)])
            resample_count += 1
            if candidate in exclude:
                continue
            if self.birth_time[candidate] <= float(t_time):
                negatives.append(candidate)

        if len(negatives) < self.neg_size:
            risk_nodes = self._risk_nodes(t_time, exclude_nodes=exclude)
            if risk_nodes.size == 0:
                risk_nodes = self._risk_nodes(t_time)
            if risk_nodes.size == 0:
                raise ValueError(f"No risk-set negative candidates are available at time {t_time}.")
            fill = np.random.choice(risk_nodes, size=self.neg_size - len(negatives), replace=True)
            negatives.extend(int(x) for x in fill.tolist())

        self.last_risk_negative_resample_count = max(0, resample_count - self.neg_size)
        return np.asarray(negatives, dtype=np.int64)
