from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

RAW = Path(__file__).resolve().parents[1] / 'data' / 'raw'
PROCESSED = Path(__file__).resolve().parents[1] / 'data' / 'processed'

LEARNING = [f"a{i:02d}" for i in range(1, 21)] + \
           [f"b{i:02d}" for i in range(1, 6)] + \
           [f"c{i:02d}" for i in range(1, 11)]

TEST = [f"x{i:02d}" for i in range(1, 36)]

ALL_RECORDS = LEARNING+TEST

FS = 100 #ECG sampling frequency
MINUTE = 60 * FS

def record_group(rec):
    return rec[0]

def load_ecg(rec):
    r = wfdb.rdrecord(str(RAW / rec))
    return r.p_signal[:, 0].astype(np.float64)

def load_labels(rec):
    ann = wfdb.rdann( str(RAW / rec), 'apn')
    samples = np.asarray(ann.sample)
    symbols = np.asarray(ann.symbol)
    #one symbol per minute: A -> apnea, N -> normal
    return pd.DataFrame({
        'record':rec,
        'minute': samples // MINUTE,
        'start': samples,
        'label': (symbols == 'A').astype(int)
    })

def load_qrs(rec):
    ann = wfdb.rdann(str(RAW / rec), "qrs")
    return np.asarray(ann.sample)
