import numpy as np
import neurokit2 as nk
import pandas as pd
from src.data import load_ecg, load_labels


def detect_rpeaks(ecg, fs=100):
    cleaned = nk.ecg_clean(ecg, sampling_rate=fs)
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
    return np.asarray(info["ECG_R_Peaks"])

def rr_intervals(peaks, fs=100):
    df = pd.DataFrame({"t": peaks[1:] / fs, "rr": np.diff(peaks) / fs})
    in_range = (df.rr >= 0.3) & (df.rr <= 2.0)
    local_median = df.rr.rolling(11, center=True, min_periods=1).median()
    no_jump = (df.rr - local_median).abs() <= 0.3 * local_median
    df["valid"] = in_range & no_jump
    return df

def minute_feats(rr_df, t0):
    in_min = (rr_df.t >= t0) & (rr_df.t <= t0+60)
    seg = rr_df[in_min]
    rr_m = seg[seg.valid].rr.to_numpy()

    f = {'n_beats': len(seg),
         'frac_valid': seg.valid.mean() if len(seg) else 0.0}

    if len(rr_m) < 5:
        for k in ["rr_mean", "rr_sdnn", "rr_rmssd", "rr_pnn50",
                  "rr_min", "rr_max", "rr_range", "hr_mean"]:
            f[k] = np.nan
        return f

    d = np.diff(rr_m)
    f["rr_mean"] = rr_m.mean()
    f["rr_sdnn"] = rr_m.std(ddof=1) * 1000
    f["rr_rmssd"] = np.sqrt(np.mean(d ** 2)) * 1000
    f["rr_pnn50"] = np.mean(np.abs(d) > 0.05) * 100
    f["rr_min"] = rr_m.min()
    f["rr_max"] = rr_m.max()
    f["rr_range"] = rr_m.max() - rr_m.min()
    f["hr_mean"] = (60 / rr_m).mean()

    return f

def record_feats(rec, fs=100):
    ecg = load_ecg(rec)
    labels = load_labels(rec)
    peaks = detect_rpeaks(ecg, fs)
    rr_df = rr_intervals(peaks, fs)

    rows = []
    for _, label in labels.iterrows():
        f = minute_feats(rr_df, label.start/fs)
        f["record"] = rec
        f["minute"] = label.minute
        f["label"] = label.label
        rows.append(f)
    return pd.DataFrame(rows)
