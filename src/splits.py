# src/splits.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.data import LEARNING, PROCESSED, record_group

N_SPLITS = 5
SEED = 42


def make_record_folds(train_df: pd.DataFrame, n_splits: int = N_SPLITS) -> pd.DataFrame:
    stats = (
        train_df.groupby("record")
        .agg(group=("group", "first"), n_min=("label", "size"), n_pos=("label", "sum"))
        .reset_index()
    )
    assert len(stats) == len(LEARNING), "Missing learning records in train_clean.csv"

    fold_pos, fold_cnt = np.zeros(n_splits), np.zeros(n_splits)
    rec_fold = {}

    for g in ["a", "b", "c"]:
        key = "n_min" if g == "c" else "n_pos"
        gdf = stats[stats["group"] == g].sort_values(key, ascending=False)
        for start in range(0, len(gdf), n_splits):
            chunk = gdf.iloc[start : start + n_splits]
            order = np.argsort(fold_cnt if g == "c" else fold_pos)
            for f, row in zip(order, chunk.itertuples()):
                rec_fold[row.record] = int(f)
                fold_pos[f] += row.n_pos
                fold_cnt[f] += row.n_min

    stats["patient_fold"] = stats["record"].map(rec_fold)
    return stats[["record", "group", "patient_fold"]]


def attach_folds(df: pd.DataFrame, rec_folds: pd.DataFrame | None = None,
                 n_splits: int = N_SPLITS, seed: int = SEED) -> pd.DataFrame:
    """Attach locked patient_fold and minute-level random_fold (leakage audit) to train df."""
    if rec_folds is None:
        rec_folds = pd.read_csv(PROCESSED / "record_folds.csv")

    out = df[["record", "minute", "label", "group"]].copy()
    fold_map = dict(zip(rec_folds["record"], rec_folds["patient_fold"]))
    out["patient_fold"] = out["record"].map(fold_map)
    assert out["patient_fold"].isna().sum() == 0, "Unmapped record found in train df"

    # Naive minute-level random split for the Leakage Gap experiment
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rand_folds = np.empty(len(out), dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(out, out["label"])):
        rand_folds[val_idx] = fold
    out["random_fold"] = rand_folds

    # Sanity check: zero patient overlap across patient folds
    for f in range(n_splits):
        tr_recs = set(out.loc[out.patient_fold != f, "record"])
        va_recs = set(out.loc[out.patient_fold == f, "record"])
        assert tr_recs.isdisjoint(va_recs), f"Patient leakage detected in fold {f}!"

    return out


def get_cv_splits(df: pd.DataFrame, split_col: str = "patient_fold"):
    """Yield (train_idx, val_idx) tuples compatible with sklearn GridSearchCV & PyTorch."""
    for f in range(N_SPLITS):
        tr_idx = np.where(df[split_col].to_numpy() != f)[0]
        va_idx = np.where(df[split_col].to_numpy() == f)[0]
        yield tr_idx, va_idx


if __name__ == "__main__":
    train_path = PROCESSED / "train_clean.csv"
    train_df = pd.read_csv(train_path)
    if train_path.exists():
        rec_folds = make_record_folds(train_df)
        rec_folds.to_csv(PROCESSED / "record_folds.csv", index=False)
        print("Record-level fold distribution:")
        print(pd.crosstab(rec_folds["patient_fold"], rec_folds["group"]))

        folds_df = attach_folds(train_df, rec_folds)
        folds_df.to_csv(PROCESSED / "folds.csv", index=False)
        print("\nMinute-level apnea rate per patient_fold:")
        print(folds_df.groupby("patient_fold")["label"].agg(["count", "mean"]).round(3))