import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.dl_aug import augment_sample

from src.dl_cache import CACHE_DIR, RAW_FS, SEQ_FS


def load_cache(records) -> dict[str, dict[str, np.ndarray]]:
    cache = {}
    for rec in tqdm(sorted(set(records)), desc="load_cache", unit="npz", dynamic_ncols=True):
        d = np.load(CACHE_DIR / f"{rec}.npz")
        cache[rec] = {
            "rr_edr": d["rr_edr"],            # (2, n_seq)
            "raw_ecg": d["raw_ecg"][None, :], # (1, n_raw)
        }
    return cache


class ApneaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cache: dict,
        mode: str = "rr_edr",   # "rr_edr", "rr", or "raw_ecg"
        context_mins: int = 3,  # 3 -> [t-1, t, t+1], 1 -> [t]
        augment: bool = False,
    ):
        assert mode in ("rr_edr", "rr", "raw_ecg")
        assert context_mins in (1, 3)
        self.records = df["record"].to_numpy()
        self.minutes = df["minute"].to_numpy(dtype=np.int64)
        self.labels = df["label"].to_numpy(dtype=np.float32)
        self.cache = cache
        self.mode = mode
        self.context_mins = context_mins
        self.augment = augment

        self.fs = RAW_FS if mode == "raw_ecg" else SEQ_FS
        self.win_len = context_mins * 60 * self.fs
        self.pad_mins = (context_mins - 1) // 2

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        m = int(self.minutes[idx])

        if self.mode == "raw_ecg":
            sig = self.cache[rec]["raw_ecg"]
        elif self.mode == "rr":
            sig = self.cache[rec]["rr_edr"][:1]
        else:
            sig = self.cache[rec]["rr_edr"]

        start = (m - self.pad_mins) * 60 * self.fs
        end = start + self.win_len
        n_total = sig.shape[1]

        s0, e0 = max(0, start), min(n_total, end)
        x = np.zeros((sig.shape[0], self.win_len), dtype=np.float32)
        if e0 > s0:
            x[:, (s0 - start) : (s0 - start) + (e0 - s0)] = sig[:, s0:e0]

        if self.augment:
            x = augment_sample(x, mode=self.mode)

        return torch.from_numpy(x), torch.tensor(self.labels[idx], dtype=torch.float32)