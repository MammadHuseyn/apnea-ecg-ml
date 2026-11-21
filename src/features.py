import numpy as np
import neurokit2 as nk
import pandas as pd

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