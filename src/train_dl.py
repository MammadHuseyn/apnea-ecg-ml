from pathlib import Path
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import PROCESSED
from src.dl_dataset import ApneaDataset, load_cache
from src.dl_models import ApneaSeqNet
from src.eval import WANDB_ENTITY, WANDB_PROJECT, evaluate_and_log, fetch_oof_threshold

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "models"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_amp_dtype(device: torch.device) -> torch.dtype | None:
    """bf16 autocast on GPUs that support it, otherwise fp32. Never fp16 + GradScaler:
    bf16 has the same dynamic range as fp32, so no loss scaling is needed."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device, desc: str | None = None) -> np.ndarray:
    model.eval()
    probs = []
    stream = tqdm(loader, desc=desc, unit="bt", leave=False, dynamic_ncols=True) if desc else loader
    for x, _ in stream:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def train_fold(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    cache: dict,
    mode: str,
    context_mins: int,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 5e-3,
    patience: int = 6,
    seed: int = 42,
    fold: int | None = None,
    wandb_run=None,
):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = device.type == "cuda"
    amp_dtype = resolve_amp_dtype(device)

    tr_ds = ApneaDataset(tr_df, cache, mode=mode, context_mins=context_mins, augment=True)
    va_ds = ApneaDataset(va_df, cache, mode=mode, context_mins=context_mins, augment=False)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, pin_memory=on_cuda)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False, pin_memory=on_cuda)

    model = ApneaSeqNet(mode=mode).to(device)
    n_pos = max(float(tr_df["label"].sum()), 1.0)
    n_neg = float(len(tr_df) - n_pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=device))

    # ScheduleFree replaces both the optimizer and the LR schedule: warmup is built in
    # (counted in optimizer steps), here over the first epoch.
    optimizer = AdamWScheduleFree(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        warmup_steps=len(tr_loader),
        r=0.9,
    )

    best_auc, wait = -1.0, 0
    best_probs, best_state = None, None
    va_y = va_df["label"].to_numpy()
    tag = "" if fold is None else f"f{fold} "

    for epoch in range(1, epochs + 1):
        optimizer.train()
        model.train()
        epoch_loss, n_batches = 0.0, 0
        with tqdm(
            tr_loader,
            desc=f"{tag}ep {epoch:02d}/{epochs}",
            unit="bt",
            leave=False,
            dynamic_ncols=True,
        ) as bar:
            for x, y in bar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                    loss = criterion(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_val = loss.item()
                epoch_loss += loss_val
                n_batches += 1
                bar.set_postfix_str(f"loss={loss_val:.4f}")

        # Eval-mode weights are the ones used for inference, so validation and
        # checkpointing must both happen after optimizer.eval().
        optimizer.eval()
        va_probs = predict_loader(model, va_loader, device)
        auc = roc_auc_score(va_y, va_probs)
        if auc > best_auc:
            best_auc, wait = auc, 0
            best_probs = va_probs
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        tqdm.write(f"{tag}ep {epoch:02d} val_auc={auc:.4f} best={best_auc:.4f} wait={wait}/{patience}")
        if wandb_run is not None:
            wandb_run.log({
                "fold": fold,
                "epoch": epoch,
                "epoch/loss": epoch_loss / max(n_batches, 1),
                "epoch/val_auc": auc,
                "epoch/best_auc": best_auc,
                "epoch/patience_wait": wait,
            })
        if wait >= patience:
            break

    return best_probs, best_state, best_auc


def run_cv_experiment(
    mode: str = "rr_edr",
    context_mins: int = 3,
    split_col: str = "patient_fold",
    run_name: str = "dl_rredr_3m",
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 1e-3,
    use_wandb: bool = True,
    cache: dict | None = None,
):
    folds_df = pd.read_csv(PROCESSED / "folds.csv")
    if cache is None:
        cache = load_cache(folds_df["record"].unique())

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    oof_probs = np.zeros(len(folds_df), dtype=np.float32)
    folds = sorted(set(folds_df[split_col].to_numpy()))

    cfg = {
        "mode": mode,
        "context_mins": context_mins,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "split_col": split_col,
        "n_folds": len(folds),
        "optimizer": "AdamWScheduleFree(r=0.9, wd=1e-2, warmup=1 epoch)",
        "amp_dtype": str(resolve_amp_dtype(torch.device("cuda" if torch.cuda.is_available() else "cpu"))),
    }

    wb_run = None
    if use_wandb:
        import wandb

        wb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"{run_name}_{split_col}",
            config={"model": run_name, **cfg},
            reinit="finish_previous",
        )

    for fold in folds:
        tr_mask = folds_df[split_col].to_numpy() != fold
        va_mask = ~tr_mask
        va_probs, state, auc = train_fold(
            folds_df[tr_mask],
            folds_df[va_mask],
            cache=cache,
            mode=mode,
            context_mins=context_mins,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=42 + int(fold),
            fold=int(fold),
            wandb_run=wb_run,
        )
        oof_probs[va_mask] = va_probs
        torch.save(state, CKPT_DIR / f"{run_name}_{split_col}_f{fold}.pt")
        tqdm.write(f"  fold {fold} best val AUC = {auc:.4f}")
        if wb_run is not None:
            wb_run.log({"fold": int(fold), "fold/val_auc_best": auc})

    metrics = evaluate_and_log(
        folds_df,
        oof_probs,
        model_name=run_name,
        split_type=split_col,
        config=cfg,
        use_wandb=use_wandb,
        wandb_run=wb_run,
    )
    if wb_run is not None:
        print(f"  wandb -> {wb_run.url}")
        wb_run.finish()
    return oof_probs, metrics


def eval_heldout_test(
    mode: str,
    context_mins: int,
    run_name: str,
    threshold: float | None = None,
    use_wandb: bool = True,
    wandb_project: str = WANDB_PROJECT,
    cache: dict | None = None,
):
    """Run ONCE after locking hyperparameters on patient_fold CV."""
    test_df = pd.read_csv(PROCESSED / "test_clean.csv")
    if cache is None:
        cache = load_cache(test_df["record"].unique())

    if threshold is None:
        threshold = fetch_oof_threshold(run_name, project=wandb_project)
        print(f"Fetched OOF threshold for {run_name} from W&B: {threshold:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ApneaDataset(test_df, cache, mode=mode, context_mins=context_mins, augment=False)
    loader = DataLoader(ds, batch_size=256, shuffle=False, pin_memory=(device.type == "cuda"))

    preds = []
    for fold in range(5):
        model = ApneaSeqNet(mode=mode).to(device)
        ckpt = CKPT_DIR / f"{run_name}_patient_fold_f{fold}.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        preds.append(predict_loader(model, loader, device, desc=f"test f{fold}"))

    test_probs = np.mean(preds, axis=0)
    return evaluate_and_log(
        test_df,
        test_probs,
        model_name=run_name,
        split_type="test",
        threshold=threshold,
        config={"mode": mode, "context_mins": context_mins},
        use_wandb=use_wandb,
    )