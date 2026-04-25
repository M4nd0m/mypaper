import math
from typing import Dict, Set, Tuple


def metric_td_tc(S: Set[int], tadj_list: Dict[int, Dict[int, set]], total_edges: int) -> Tuple[float, float]:
    if len(S) == 0:
        return 0.0, 1.0
    S = set(S)

    temporal_edge_S = 0
    time_S = set()
    for u in S:
        if u not in tadj_list:
            continue
        for v, times in tadj_list[u].items():
            if v in S and u < v:
                for t in times:
                    time_S.add(t)
                    temporal_edge_S += 1

    if len(S) <= 1 or len(time_S) == 0:
        TD = 0.0
    else:
        denom = (len(S) * (len(S) - 1) / 2.0) * len(time_S)
        TD = float(temporal_edge_S) / float(denom) if denom > 0 else 0.0

    temporal_cut_S = 0
    temporal_vol_S = 0
    for u in S:
        if u not in tadj_list:
            continue
        for v, times in tadj_list[u].items():
            cnt = len(times)
            if v not in S:
                temporal_cut_S += cnt
            temporal_vol_S += cnt

    temporal_vol_S = min(temporal_vol_S, total_edges - temporal_vol_S)
    TC = 1.0 if temporal_vol_S <= 0 else float(temporal_cut_S) / float(temporal_vol_S)
    return float(TD), float(TC)


def binary_clustering_metrics(pred_set: Set[int], gt_set: Set[int], total_nodes: int):
    if len(pred_set) == 0:
        return {"F1": 0.0, "JAC": 0.0, "NMI": 0.0}

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    jac = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    n11 = tp
    n10 = fn
    n01 = fp
    n00 = total_nodes - tp - fp - fn
    n = float(total_nodes)

    n1_ = n11 + n10
    n0_ = n01 + n00
    n_1 = n11 + n01
    n_0 = n10 + n00

    MI = 0.0
    for nij, nc, nk in [(n11, n1_, n_1), (n10, n1_, n_0), (n01, n0_, n_1), (n00, n0_, n_0)]:
        if nij > 0 and nc > 0 and nk > 0:
            MI += (nij / n) * math.log((n * nij) / (nc * nk + 1e-12) + 1e-12)

    Hc = 0.0
    for nc in [n1_, n0_]:
        if nc > 0:
            p = nc / n
            Hc -= p * math.log(p + 1e-12)

    Hk = 0.0
    for nk in [n_1, n_0]:
        if nk > 0:
            p = nk / n
            Hk -= p * math.log(p + 1e-12)

    nmi = MI / math.sqrt(Hc * Hk) if Hc > 0 and Hk > 0 else 0.0
    return {"F1": float(f1), "JAC": float(jac), "NMI": float(nmi)}
