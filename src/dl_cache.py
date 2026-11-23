from concurrent.futures import ProcessPoolExecutor
import numpy as np
import neurokit2 as nk
from scipy.interpolate import PchipInterpolator
from scipy.signal import resample_poly
from tqdm.auto import tqdm

from src.data import ALL_RECORDS, FS, PROCESSED, load_ecg
from src.features import rr_intervals

SEQ_FS = 3 # 3 Hz -> 180 steps/min, 540 steps/3-min context
RAW_FS = 50 # 50 Hz -> 3000 stWeps/min
CACHE_DIR = PROCESSED / "dl_cache"


def _robust_zscore(x: np.ndarray, clip: float = 6.0) -> np.ndarray:
    med = np.median(x)
    q75, q25 = np.percentile(x, [75, 25])
    scale = (q75 - q25) if (q75 - q25) > 1e-6 else (np.std(x) + 1e-6)
    return np.clip((x - med) / scale, -clip, clip).astype(np.float32)


def _process_record(rec: str) -> str:
    ecg = load_ecg(rec)
    duration_sec = len(ecg) / FS

    # Same cleaning, R-peak detector, and RR artifact filter as src/features.py
    cleaned = nk.ecg_clean(ecg, sampling_rate=FS)
    _, info = nk.ecg_peaks(cleaned, sampling_rate=FS)
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)

    rr_df = rr_intervals(peaks, fs=FS)
    # R-peak amplitude on baseline-removed ECG (EDR proxy), aligned with rr_df
    edr_all = cleaned[peaks[1:]]

    valid = rr_df["valid"].to_numpy()
    t_valid = rr_df["t"].to_numpy()[valid]
    rr_valid = rr_df["rr"].to_numpy()[valid]
    edr_valid = edr_all[valid]

    # Uniform 3 Hz continuous grid across the entire night
    n_seq = int(np.floor(duration_sec * SEQ_FS))
    t_grid = np.arange(n_seq, dtype=np.float64) / SEQ_FS
    t_clamped = np.clip(t_grid, t_valid[0], t_valid[-1])

    rr_3hz = PchipInterpolator(t_valid, rr_valid)(t_clamped)
    edr_3hz = PchipInterpolator(t_valid, edr_valid)(t_clamped)

    rr_norm = _robust_zscore(rr_3hz, clip=6.0)
    edr_norm = _robust_zscore(edr_3hz, clip=6.0)
    rr_edr = np.stack([rr_norm, edr_norm], axis=0)  # shape: (2, n_seq)

    # Anti-aliased 50 Hz cleaned raw ECG, per-record z-scored
    raw_50hz = resample_poly(cleaned, up=1, down=FS // RAW_FS)
    raw_norm = np.clip(
        (raw_50hz - np.mean(raw_50hz)) / (np.std(raw_50hz) + 1e-6), -10.0, 10.0
    ).astype(np.float32)

    np.savez_compressed(
        CACHE_DIR / f"{rec}.npz",
        rr_edr=rr_edr,
        raw_ecg=raw_norm,
        peaks=peaks.astype(np.int32),
    )
    return f"{rec}: rr_edr={rr_edr.shape}, raw_ecg={raw_norm.shape}"


def build_cache(max_workers: int = 4) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n = len(ALL_RECORDS)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        bar = tqdm(ex.map(_process_record, ALL_RECORDS), total=n, desc="build_cache", unit="rec", dynamic_ncols=True)
        for i, msg in enumerate(bar, 1):
            if i % 10 == 0 or i == n:
                bar.write(f"[{i}/{n}] {msg}")


if __name__ == "__main__":
    build_cache()