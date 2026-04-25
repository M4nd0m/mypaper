import hashlib
import os

import numpy as np
import torch


DEFAULT_NEG_TABLE_SIZE = int(1e6)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def hash_cfg(cfg: dict) -> str:
    items = "|".join(f"{key}={cfg[key]}" for key in sorted(cfg.keys()))
    return hashlib.md5(items.encode("utf-8")).hexdigest()[:10]


def resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def cache_paths(base_cache_dir: str, dataset: str, tppr_cfg: dict, taps_cfg: dict):
    ensure_dir(base_cache_dir)
    ds_dir = os.path.join(base_cache_dir, dataset)
    ensure_dir(ds_dir)

    taps_hash = hash_cfg(taps_cfg)
    tppr_cfg2 = dict(tppr_cfg)
    tppr_cfg2.update({"useTAPS": 1, "taps_hash": taps_hash})
    tppr_hash = hash_cfg(tppr_cfg2)

    taps_path = os.path.join(ds_dir, f"taps_{taps_hash}.npz")
    tppr_path = os.path.join(ds_dir, f"tppr_{tppr_hash}.npz")
    return taps_path, tppr_path
