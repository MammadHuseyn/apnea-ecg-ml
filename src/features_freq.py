"""Set E features: frequency-domain HRV + ECG-derived respiration (EDR)
from: data/processed/dl_cache/<rec>.npz
"""
import numpy as np
import pandas as pd
from scipy.signal import welch

from src.data import PROCESSED

CACHE_DIR = PROCESSED / "dl_cache"
SEQ_FS = 3                 # samples per second in the cache
PER_MIN = 60 * SEQ_FS      # 180 samples per minute
WIN_MIN = 5                # window for frequency features (minutes)

VLF = (0.01, 0.04)         # apnea cycles: one every ~25-100 s
LF = (0.04, 0.15)
HF = (0.15, 0.40)          # breathing-linked heart-rate changes
RESP = (0.10, 0.50)        # breathing: 6-30 breaths per minute

FREQ_COLS = ["rr_vlf", "rr_lf", "rr_hf", "rr_lf_hf", "rr_vlf_frac",
             "edr_std", "edr_range", "edr_resp_rate", "edr_resp_frac", "edr_vlf_frac"]


def _psd(x):
    """Power spectrum of an evenly sampled window (Welch's method)."""
    return welch(x, fs=SEQ_FS, nperseg=min(len(x), 512), detrend="linear")


def _band(f, p, band):
    m = (f >= band[0]) & (f < band[1])
    return p[m].sum()


def minute_freq_features(rr, edr, m):
    """Features for minute m of one record (rr, edr = whole-night 3 Hz arrays)."""
    # 5-minute window centred on the minute, cut at the start/end of the night
    lo = max(0, (m - WIN_MIN // 2) * PER_MIN)
    hi = min(len(rr), (m + WIN_MIN // 2 + 1) * PER_MIN)
    out = dict.fromkeys(FREQ_COLS, np.nan)
    if hi - lo < 3 * PER_MIN:          # need at least 3 minutes for slow cycles
        return out

    f, p_rr = _psd(rr[lo:hi])
    vlf, lf, hf = (_band(f, p_rr, b) for b in (VLF, LF, HF))
    out["rr_vlf"] = np.log(vlf + 1e-8)
    out["rr_lf"] = np.log(lf + 1e-8)
    out["rr_hf"] = np.log(hf + 1e-8)
    out["rr_lf_hf"] = np.log((lf + 1e-8) / (hf + 1e-8))
    out["rr_vlf_frac"] = vlf / (vlf + lf + hf + 1e-8)

    e_min = edr[m * PER_MIN:(m + 1) * PER_MIN]       # the minute itself
    out["edr_std"] = e_min.std()
    out["edr_range"] = np.percentile(e_min, 95) - np.percentile(e_min, 5)

    f, p_e = _psd(edr[lo:hi])
    total = _band(f, p_e, (0.01, 1.0)) + 1e-8
    r = (f >= RESP[0]) & (f < RESP[1])
    out["edr_resp_rate"] = f[r][np.argmax(p_e[r])] * 60     # breaths per minute
    out["edr_resp_frac"] = _band(f, p_e, RESP) / total
    out["edr_vlf_frac"] = _band(f, p_e, VLF) / total
    return out


def build_freq_table(df):
    """One row of FREQ_COLS per (record, minute) in df."""
    rows = []
    for rec, g in df.groupby("record"):
        rr, edr = np.load(CACHE_DIR / f"{rec}.npz")["rr_edr"]
        for m in g["minute"]:
            rows.append({"record": rec, "minute": m, **minute_freq_features(rr, edr, int(m))})
    return pd.DataFrame(rows)