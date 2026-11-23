# src/eval.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import wandb

ROOT = Path(__file__).resolve().parents[1]
PREDS_DIR = ROOT / "reports" / "preds"

WANDB_ENTITY = "emammadli"
WANDB_PROJECT = "apnea-ecg-ml"

def fetch_oof_threshold(
    run_name: str,
    split_type: str = "patient_fold",
    project: str = "apnea-ecg-csci4734",
    entity: str | None = None,
) -> float:
    """Fetch the locked OOF threshold from the newest finished W&B CV run."""
    
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    target = f"{run_name}_{split_type}"

    runs = [
        r for r in api.runs(path)
        if r.state == "finished" and r.name == target and "threshold" in r.summary
    ]
    if not runs:
        raise ValueError(f"No finished W&B run with 'threshold' found for '{target}'.")

    latest = max(runs, key=lambda r: r.created_at)
    return float(latest.summary["threshold"])


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Grid-search probability threshold maximizing F1 on OOF validation predictions."""
    thresholds = np.linspace(0.10, 0.90, 161)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        score = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    records: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Per-minute binary metrics + per-patient overnight severity metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    # Per-patient apnea fraction (minutes/hour proxy; clinical screening >= 5 min/hr)
    df_rec = pd.DataFrame({"record": records, "y_true": y_true, "y_pred": y_pred})
    rec_agg = df_rec.groupby("record")[["y_true", "y_pred"]].mean()
    patient_mae = float(np.mean(np.abs(rec_agg["y_true"] - rec_agg["y_pred"])))
    patient_acc = float(
        accuracy_score(
            (rec_agg["y_true"] >= 5 / 60).astype(int),
            (rec_agg["y_pred"] >= 5 / 60).astype(int),
        )
    )

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "patient_frac_mae": patient_mae,
        "patient_screen_acc": patient_acc,
    }


def evaluate_and_log(
    df_meta: pd.DataFrame,
    y_prob: np.ndarray,
    model_name: str,
    split_type: str,
    threshold: float | None = None,
    config: dict | None = None,
    use_wandb: bool = True,
    wandb_entity: str = WANDB_ENTITY,
    wandb_project: str = WANDB_PROJECT,
    wandb_run=None,
) -> dict:
    """Evaluate predictions, save CSV to reports/preds/, and log to W&B.

    wandb_run: an already-open run to log into (used by run_cv_experiment, so the
    per-fold history and these final metrics live in ONE run). The caller owns that
    run's lifecycle; this function only finishes runs it opened itself.
    """
    y_true = df_meta["label"].to_numpy()
    records = df_meta["record"].to_numpy()

    if threshold is None:
        threshold = find_best_threshold(y_true, y_prob)

    metrics = compute_metrics(y_true, y_prob, records, threshold=threshold)

    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    out_df = df_meta[["record", "minute", "label"]].copy()
    out_df["y_prob"] = y_prob
    out_df["y_pred"] = (y_prob >= threshold).astype(int)
    out_df.to_csv(PREDS_DIR / f"{split_type}_{model_name}.csv", index=False)

    if use_wandb:
        import wandb

        if wandb_run is not None:
            wandb_run.log(metrics)
        else:
            run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=f"{model_name}_{split_type}",
                config={"model": model_name, "split_type": split_type, **(config or {})},
                reinit="finish_previous",
            )
            wandb.log(metrics)
            run.finish()
            print(f"  wandb -> {run.url}")

    print(
        f"[{split_type} | {model_name}] "
        f"Acc={metrics['accuracy']:.4f} Sens={metrics['sensitivity']:.4f} "
        f"Spec={metrics['specificity']:.4f} F1={metrics['f1']:.4f} "
        f"AUC={metrics['roc_auc']:.4f} (thr={metrics['threshold']:.2f}, "
        f"PatAcc={metrics['patient_screen_acc']:.3f})"
    )
    return metrics