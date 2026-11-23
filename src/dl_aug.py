import numpy as np


def augment_sample(x: np.ndarray, mode: str = "rr_edr") -> np.ndarray:
    """Light time-domain 1D augmentations. x: (C, T) float32."""
    C, T = x.shape
    x = x.copy()

    # 1. SignalJitter (time shift + gain scale)
    if np.random.rand() < 0.5:
        shift = np.random.randint(-25, 26) if mode == "raw_ecg" else np.random.randint(-6, 7)
        x = np.roll(x, shift, axis=-1)
        x *= np.random.uniform(0.92, 1.08, size=(C, 1)).astype(np.float32)

    # 2. GaussianNoise
    if np.random.rand() < 0.5:
        x += np.random.normal(0.0, 0.03, size=x.shape).astype(np.float32)

    # 3. RandomErasing (short time cutout -> 0 = per-record median)
    if np.random.rand() < 0.3:
        length = max(1, int(T * np.random.uniform(0.05, 0.10)))
        start = np.random.randint(0, T - length + 1)
        x[:, start : start + length] = 0.0

    return x