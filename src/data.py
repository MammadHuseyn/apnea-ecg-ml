from pathlib import Path
import numpy as np
import wfdb

RAW = Path(__file__).resolve().parents[1] / 'data' / 'raw'

LEARNING = [f"a{i:02d}" for i in range(1, 21)] + \
           [f"b{i:02d}" for i in range(1, 6)] + \
           [f"c{i:02d}" for i in range(1, 11)]

TEST = [f"x{i:02d}" for i in range(1, 36)]
FS = 100 #ECG sampling frequency

def load_ecg(rec):
    r = wfdb.rdrecord(str(RAW / rec))
    return r.p_signal[:, 0]

def load_labels(rec):
    ann = wfdb.rdann(str(RAW / rec), 'apn')
    #one symbol per minute: A -> apnea, N -> normal
    return np.array([1 if s == 'A' else 0 for s in ann.symbol])