"""
training/train_stgcn_full.py — Comprehensive ST-GCN evaluation across all conditions.

Experiments
-----------
A1  EC3D LOSO (4 rounds)            — mean ± std across rounds
A2  REHAB24 LOSO (9 rounds)         — mean ± std across rounds
A4  UI-PRMD LOSO (10 rounds)       — mean ± std across rounds
A3  Custom 5-Fold CV (5 folds)      — mean ± std across folds

B1  Train EC3D 80%     → test REHAB24 OOD
B2  Train EC3D 80%     → test Custom  OOD
B3  Train REHAB24 80%  → test EC3D    OOD
B4  Train REHAB24 80%  → test Custom  OOD
B5  Train Custom 80%   → test EC3D    OOD
B6  Train Custom 80%   → test REHAB24 OOD
B7  Train UI-PRMD 80%  → test EC3D    OOD
B8  Train UI-PRMD 80%  → test REHAB24 OOD
B9  Train UI-PRMD 80%  → test Custom  OOD
B10 Train EC3D 80%     → test UI-PRMD OOD
B11 Train REHAB24 80%  → test UI-PRMD OOD
B12 Train Custom 80%   → test UI-PRMD OOD

C1  Train EC3D+REHAB24   → test Custom  OOD
C2  Train EC3D+Custom    → test REHAB24 OOD
C3  Train REHAB24+Custom → test EC3D    OOD
C4  Train EC3D+UI-PRMD         → test Custom  OOD
C5  Train REHAB24+UI-PRMD      → test Custom  OOD
C6  Train EC3D+REHAB24+Custom  → test UI-PRMD OOD
C8  Train EC3D+REHAB24+UI-PRMD → test Custom  OOD
C9  Train EC3D+Custom+UI-PRMD  → test REHAB24 OOD
C10 Train REHAB24+Custom+UI-PRMD → test EC3D    OOD
C7  Train Custom+UI-PRMD       → test EC3D    OOD

D1  Mixed — all 3 datasets 80% train / 10% val / 10% test

Training: early stopping on val loss (patience=20), GPU if available.
Each experiment saves its own checkpoint.

Usage
-----
    cd <project_root>
    python training/train_stgcn_full.py
    python training/train_stgcn_full.py --ood-only
    python training/train_stgcn_full.py --mixed-only

Prerequisites (run in order)
-----------------------------
  1. ec3d_to_stgcn.py           → output/ec3d_train_data.npy
  2. rehab24_to_stgcn.py        → output/rehab24_train_data.npy
  3. custom_squat_to_stgcn.py   → output/custom_squat_train_data.npy
  4. split_loso.py              → output/ec3d_loso_round*_*.npy  +  rehab24_loso_round*_*.npy
     (also uiprmd_to_stgcn.py → output/uiprmd_loso_round*_*.npy for A4)
  5. split_kfold.py             → output/custom_kfold_fold*_*.npy
  6. split_iid.py               → output/ec3d_iid_*.npy  +  rehab24_iid_*.npy
  7. split_ood.py               → output/*_ood_test_*.npy
"""

import sys
import os
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_utils import (
    ConfmatLogger,
    compute_metrics,
    confmat_log_path,
    iid_results_path,
    mixed_results_path,
    print_and_save_summary,
)
from models.builders import build_stgcn_model, build_ctrgcn_model
from configs.constants import (
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_WEIGHT_DECAY,
    OUTPUT_DIR, TRAIN_SEED as RANDOM_SEED,
)

POS_CLASS      = 1    # 1 = correct squat
EARLY_STOP_PAT = 20   # patience for early stopping on val loss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

_confmat_logger = None
SCRIPT_NAME = "train_stgcn_full.py"
_MODEL_LOG_NAME = "ST-GCN"
STGCN_SLUG = "stgcn"
CTRGCN_SLUG = "ctrgcn"

SECTION_LABELS = {
    "A": "IID (within-distribution)",
    "B": "OOD — single source train",
    "C": "OOD — dual source train",
    "D": "Mixed (all datasets)",
}


def _init_confmat(model_slug: str) -> None:
    global _confmat_logger
    _confmat_logger = ConfmatLogger(confmat_log_path(OUTPUT_DIR, model_slug))
    _confmat_logger.ensure_header()


# ── Path helper ───────────────────────────────────────────────────────────────

def p(name: str) -> str:
    return os.path.join(OUTPUT_DIR, name)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_ds(data_path: str, label_path: str) -> TensorDataset:
    X = torch.from_numpy(np.load(data_path)).float()
    y = torch.from_numpy(np.load(label_path)).long()
    return TensorDataset(X, y)


def concat_ds(*pairs: Tuple[str, str]) -> TensorDataset:
    """Concatenate multiple (data_path, label_path) pairs."""
    X_parts, y_parts = [], []
    for dp, lp in pairs:
        X_parts.append(np.load(dp))
        y_parts.append(np.load(lp))
    X = torch.from_numpy(np.concatenate(X_parts, axis=0)).float()
    y = torch.from_numpy(np.concatenate(y_parts, axis=0)).long()
    return TensorDataset(X, y)


def split_frac(
    ds: TensorDataset,
    train_frac: float,
    seed: int = RANDOM_SEED,
) -> Tuple[TensorDataset, TensorDataset]:
    """Split ds into (train, val) by train_frac / (1 - train_frac)."""
    N   = len(ds)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_tr   = int(train_frac * N)
    tr_idx = torch.tensor(idx[:n_tr])
    va_idx = torch.tensor(idx[n_tr:])
    return (
        TensorDataset(*[t[tr_idx] for t in ds.tensors]),
        TensorDataset(*[t[va_idx] for t in ds.tensors]),
    )


def split_80_10_10(
    ds: TensorDataset,
    seed: int = RANDOM_SEED,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Split ds 80 / 10 / 10."""
    N   = len(ds)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n80    = int(0.80 * N)
    n10    = int(0.10 * N)
    tr_idx = torch.tensor(idx[:n80])
    va_idx = torch.tensor(idx[n80 : n80 + n10])
    te_idx = torch.tensor(idx[n80 + n10 :])
    return (
        TensorDataset(*[t[tr_idx] for t in ds.tensors]),
        TensorDataset(*[t[va_idx] for t in ds.tensors]),
        TensorDataset(*[t[te_idx] for t in ds.tensors]),
    )


def check_files(*paths: str) -> bool:
    missing = [f for f in paths if not os.path.exists(f)]
    for m in missing:
        print(f"  [MISSING] {m}")
    return len(missing) == 0


# ── Metrics ───────────────────────────────────────────────────────────────────


def evaluate(model: nn.Module, loader: DataLoader, tag: str = "") -> dict:
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds_all.append(model(X_b.to(DEVICE)).argmax(dim=1).cpu().numpy())
            labels_all.append(y_b.numpy())
    m = compute_metrics(np.concatenate(preds_all), np.concatenate(labels_all))
    if _confmat_logger is not None and tag:
        _confmat_logger.append(SCRIPT_NAME, _MODEL_LOG_NAME, tag, m)
    return m


def evaluate_with_confusion(tag: str, ckpt: str, ood_d: str, ood_l: str) -> None:
    """
    Reload a saved checkpoint and print a confusion matrix on the OOD test set.
    Does not retrain — only loads the .pth file written by run_ood().
    """
    ckpt = os.path.abspath(ckpt)
    ood_d = os.path.abspath(ood_d)
    ood_l = os.path.abspath(ood_l)
    print(f"  [DEBUG] Looking for checkpoint: {ckpt}")
    print(f"  [DEBUG] File exists: {os.path.exists(ckpt)}")
    if not os.path.exists(ckpt):
        print(f"  [SKIP confusion matrix] Checkpoint not found: {ckpt}")
        return

    model = build_stgcn_model(checkpoint_path=ckpt, device=DEVICE)
    model.eval()

    preds_all, labels_all = [], []
    with torch.no_grad():
        for X_b, y_b in DataLoader(load_ds(ood_d, ood_l), batch_size=TRAIN_BATCH_SIZE):
            preds_all.append(model(X_b.to(DEVICE)).argmax(dim=1).cpu().numpy())
            labels_all.append(y_b.numpy())

    preds  = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)

    tp = int(((preds == POS_CLASS) & (labels == POS_CLASS)).sum())
    fp = int(((preds == POS_CLASS) & (labels != POS_CLASS)).sum())
    fn = int(((preds != POS_CLASS) & (labels == POS_CLASS)).sum())
    tn = int(((preds != POS_CLASS) & (labels != POS_CLASS)).sum())
    pred_c = int((preds == POS_CLASS).sum())
    pred_i = int((preds != POS_CLASS).sum())

    print(f"\n  Confusion Matrix for {tag}:")
    print(f"                       Predicted Correct    Predicted Incorrect")
    print(f"  Actual Correct       {tp:>17d}    {fn:>19d}")
    print(f"  Actual Incorrect     {fp:>17d}    {tn:>19d}")
    print(
        f"  True Positives={tp}  True Negatives={tn}  "
        f"False Positives={fp}  False Negatives={fn}"
    )
    print(
        f"  Majority class prediction check: "
        f"model predicted correct={pred_c} times, incorrect={pred_i} times"
    )

    model.cpu()
    del model
    torch.cuda.empty_cache()


# ── Training loop with early stopping on validation loss ──────────────────────

def train_model(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    checkpoint_path: str,
    label: str,
) -> nn.Module:
    """
    Train ST-GCN.  Stops early when val loss does not improve for
    EARLY_STOP_PAT consecutive epochs.  Saves best checkpoint by val loss.
    """
    labels_np = train_ds.tensors[1].numpy()
    n0 = int((labels_np == 0).sum())
    n1 = int((labels_np == 1).sum())
    N  = len(labels_np)
    w = torch.tensor(
        [N / (2 * max(n0, 1)), N / (2 * max(n1, 1))]
    ).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)
    print(f"  Class weights: incorrect={w[0]:.3f}  correct={w[1]:.3f}")

    model     = build_stgcn_model(checkpoint_path=None, device=DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=TRAIN_BATCH_SIZE)

    best_val_loss = float('inf')
    no_improve    = 0
    best_epoch    = 0

    print(f"\n  [{label}]  train={len(train_ds)}  val={len(val_ds)}")

    for epoch in range(1, TRAIN_EPOCHS + 1):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        t_losses: List[float] = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            t_losses.append(loss.item())

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        v_losses: List[float] = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                v_losses.append(criterion(model(X_b), y_b).item())

        t_loss = float(np.mean(t_losses))
        v_loss = float(np.mean(v_losses))
        scheduler.step(v_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}  t_loss={t_loss:.4f}  v_loss={v_loss:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch    = epoch
            no_improve    = 0
            try:
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                torch.save(model.state_dict(), checkpoint_path)
            except RuntimeError as e:
                print(f"  [SAVE ERROR] Failed to write checkpoint: {checkpoint_path}")
                print(f"  {e}")
                raise
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PAT:
                print(
                    f"    Early stop @ epoch {epoch}  "
                    f"(best epoch={best_epoch}, v_loss={best_val_loss:.4f})"
                )
                break

    print(
        f"  Best v_loss={best_val_loss:.4f} @ epoch {best_epoch}  "
        f"→ {checkpoint_path}"
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def train_model_ctrgcn(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    checkpoint_path: str,
    label: str,
) -> nn.Module:
    """
    Train CTR-GCN.  Stops early when val loss does not improve for
    EARLY_STOP_PAT consecutive epochs.  Saves best checkpoint by val loss.
    """
    labels_np = train_ds.tensors[1].numpy()
    n0 = int((labels_np == 0).sum())
    n1 = int((labels_np == 1).sum())
    N  = len(labels_np)
    w = torch.tensor(
        [N / (2 * max(n0, 1)), N / (2 * max(n1, 1))]
    ).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)
    print(f"  Class weights: incorrect={w[0]:.3f}  correct={w[1]:.3f}")

    model     = build_ctrgcn_model(checkpoint_path=None, device=DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=TRAIN_BATCH_SIZE)

    best_val_loss = float('inf')
    no_improve    = 0
    best_epoch    = 0

    print(f"\n  [{label}]  train={len(train_ds)}  val={len(val_ds)}")

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        t_losses: List[float] = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            t_losses.append(loss.item())

        model.eval()
        v_losses: List[float] = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                v_losses.append(criterion(model(X_b), y_b).item())

        t_loss = float(np.mean(t_losses))
        v_loss = float(np.mean(v_losses))
        scheduler.step(v_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}  t_loss={t_loss:.4f}  v_loss={v_loss:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch    = epoch
            no_improve    = 0
            try:
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                torch.save(model.state_dict(), checkpoint_path)
            except RuntimeError as e:
                print(f"  [SAVE ERROR] Failed to write checkpoint: {checkpoint_path}")
                print(f"  {e}")
                raise
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PAT:
                print(
                    f"    Early stop @ epoch {epoch}  "
                    f"(best epoch={best_epoch}, v_loss={best_val_loss:.4f})"
                )
                break

    print(
        f"  Best v_loss={best_val_loss:.4f} @ epoch {best_epoch}  "
        f"→ {checkpoint_path}"
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


# ── Aggregation helpers ───────────────────────────────────────────────────────

def mean_std(metrics_list: List[dict], key: str) -> Tuple[float, float]:
    vals = [m[key] for m in metrics_list]
    return float(np.mean(vals)), float(np.std(vals))


# ── Summary-table helpers ─────────────────────────────────────────────────────

COL = 38

HDR = (
    f"  {'Condition':<{COL}}"
    f"  {'Acc':>10}"
    f"  {'Prec':>10}"
    f"  {'Rec':>10}"
    f"  {'F1':>10}"
    f"  {'MacroF1':>10}"
    f"  N"
)
SEP = "  " + "-" * (len(HDR) - 2)


def _row(tag: str, m: dict) -> str:
    return (
        f"  {tag:<{COL}}"
        f"  {m['acc']*100:6.1f}%   "
        f"  {m['prec']*100:6.1f}%   "
        f"  {m['rec']*100:6.1f}%   "
        f"  {m['f1']*100:6.1f}%   "
        f"  {m['macro_f1']*100:6.1f}%   "
        f"  {m['n']}"
    )


def _row_ms(tag: str, ml: List[dict]) -> str:
    def ms(k: str) -> str:
        mu, sd = mean_std(ml, k)
        return f"{mu*100:4.1f}±{sd*100:3.1f}%"
    return (
        f"  {tag:<{COL}}"
        f"  {ms('acc'):>10}"
        f"  {ms('prec'):>10}"
        f"  {ms('rec'):>10}"
        f"  {ms('f1'):>10}"
        f"  {ms('macro_f1'):>10}"
        f"  n={sum(m['n'] for m in ml)}"
    )


# ── Experiment runners ────────────────────────────────────────────────────────

def run_loso(prefix: str, n_rounds: int, ckpt_prefix: str) -> List[dict]:
    results: List[dict] = []
    for r in range(1, n_rounds + 1):
        tr_d = p(f"{prefix}_loso_round{r}_train_data.npy")
        tr_l = p(f"{prefix}_loso_round{r}_train_labels.npy")
        va_d = p(f"{prefix}_loso_round{r}_val_data.npy")
        va_l = p(f"{prefix}_loso_round{r}_val_labels.npy")
        te_d = p(f"{prefix}_loso_round{r}_test_data.npy")
        te_l = p(f"{prefix}_loso_round{r}_test_labels.npy")
        ckpt = p(f"{ckpt_prefix}_round{r}.pth")

        if not check_files(tr_d, tr_l, va_d, va_l, te_d, te_l):
            print(f"  Skipping {prefix.upper()} LOSO round {r} — missing files.")
            continue

        model = train_model(
            load_ds(tr_d, tr_l), load_ds(va_d, va_l),
            ckpt, f"{prefix.upper()} LOSO round {r}/{n_rounds}",
        )
        m = evaluate(model, DataLoader(load_ds(te_d, te_l), batch_size=TRAIN_BATCH_SIZE), tag=f"{prefix.upper()} LOSO round {r}")
        print(
            f"  Round {r:2d} test: "
            f"acc={m['acc']:.3f}  prec={m['prec']:.3f}  "
            f"rec={m['rec']:.3f}  F1={m['f1']:.3f}  n={m['n']}"
        )
        results.append(m)
        model.cpu()
        del model
        torch.cuda.empty_cache()
    return results


def run_kfold(n_folds: int, ckpt_prefix: str) -> List[dict]:
    results: List[dict] = []
    for f in range(1, n_folds + 1):
        tr_d = p(f"custom_kfold_fold{f}_train_data.npy")
        tr_l = p(f"custom_kfold_fold{f}_train_labels.npy")
        va_d = p(f"custom_kfold_fold{f}_val_data.npy")
        va_l = p(f"custom_kfold_fold{f}_val_labels.npy")
        te_d = p(f"custom_kfold_fold{f}_test_data.npy")
        te_l = p(f"custom_kfold_fold{f}_test_labels.npy")
        ckpt = p(f"{ckpt_prefix}_fold{f}.pth")

        if not check_files(tr_d, tr_l, va_d, va_l, te_d, te_l):
            print(f"  Skipping custom 5-fold fold {f} — missing files.")
            continue

        model = train_model(
            load_ds(tr_d, tr_l), load_ds(va_d, va_l),
            ckpt, f"Custom 5-Fold fold {f}/{n_folds}",
        )
        m = evaluate(model, DataLoader(load_ds(te_d, te_l), batch_size=TRAIN_BATCH_SIZE), tag=f"custom 5-fold fold {f}")
        print(
            f"  Fold {f} test: "
            f"acc={m['acc']:.3f}  prec={m['prec']:.3f}  "
            f"rec={m['rec']:.3f}  F1={m['f1']:.3f}  n={m['n']}"
        )
        results.append(m)
        model.cpu()
        del model
        torch.cuda.empty_cache()
    return results


def run_ood(
    label: str,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    ood_d: str,
    ood_l: str,
    ckpt: str,
) -> dict:
    if not check_files(ood_d, ood_l):
        print(f"  Skipping {label} — missing OOD test files.")
        return {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0, "n": 0}
    model = train_model(train_ds, val_ds, ckpt, label)
    m = evaluate(model, DataLoader(load_ds(ood_d, ood_l), batch_size=TRAIN_BATCH_SIZE), tag=label)
    print(
        f"  OOD test: "
        f"acc={m['acc']:.3f}  prec={m['prec']:.3f}  "
        f"rec={m['rec']:.3f}  F1={m['f1']:.3f}  n={m['n']}"
    )
    model.cpu()
    del model
    torch.cuda.empty_cache()
    return m


def run_loso_ctrgcn(prefix: str, n_rounds: int, ckpt_prefix: str) -> List[dict]:
    results: List[dict] = []
    for r in range(1, n_rounds + 1):
        tr_d = p(f"{prefix}_loso_round{r}_train_data.npy")
        tr_l = p(f"{prefix}_loso_round{r}_train_labels.npy")
        va_d = p(f"{prefix}_loso_round{r}_val_data.npy")
        va_l = p(f"{prefix}_loso_round{r}_val_labels.npy")
        te_d = p(f"{prefix}_loso_round{r}_test_data.npy")
        te_l = p(f"{prefix}_loso_round{r}_test_labels.npy")
        ckpt = p(f"{ckpt_prefix}_round{r}.pth")

        if not check_files(tr_d, tr_l, va_d, va_l, te_d, te_l):
            print(f"  Skipping {prefix.upper()} LOSO round {r} — missing files.")
            continue

        model = train_model_ctrgcn(
            load_ds(tr_d, tr_l), load_ds(va_d, va_l),
            ckpt, f"{prefix.upper()} LOSO round {r}/{n_rounds}",
        )
        m = evaluate(model, DataLoader(load_ds(te_d, te_l), batch_size=TRAIN_BATCH_SIZE), tag=f"{prefix.upper()} LOSO round {r}")
        print(
            f"  Round {r:2d} test: "
            f"acc={m['acc']:.3f}  prec={m['prec']:.3f}  "
            f"rec={m['rec']:.3f}  F1={m['f1']:.3f}  n={m['n']}"
        )
        results.append(m)
        model.cpu()
        del model
        torch.cuda.empty_cache()
    return results


def run_kfold_ctrgcn(n_folds: int, ckpt_prefix: str) -> List[dict]:
    results: List[dict] = []
    for f in range(1, n_folds + 1):
        tr_d = p(f"custom_kfold_fold{f}_train_data.npy")
        tr_l = p(f"custom_kfold_fold{f}_train_labels.npy")
        va_d = p(f"custom_kfold_fold{f}_val_data.npy")
        va_l = p(f"custom_kfold_fold{f}_val_labels.npy")
        te_d = p(f"custom_kfold_fold{f}_test_data.npy")
        te_l = p(f"custom_kfold_fold{f}_test_labels.npy")
        ckpt = p(f"{ckpt_prefix}_fold{f}.pth")

        if not check_files(tr_d, tr_l, va_d, va_l, te_d, te_l):
            print(f"  Skipping custom 5-fold fold {f} — missing files.")
            continue

        model = train_model_ctrgcn(
            load_ds(tr_d, tr_l), load_ds(va_d, va_l),
            ckpt, f"Custom 5-Fold fold {f}/{n_folds}",
        )
        m = evaluate(model, DataLoader(load_ds(te_d, te_l), batch_size=TRAIN_BATCH_SIZE), tag=f"custom 5-fold fold {f}")
        print(
            f"  Fold {f} test: "
            f"acc={m['acc']:.3f}  prec={m['prec']:.3f}  "
            f"rec={m['rec']:.3f}  F1={m['f1']:.3f}  n={m['n']}"
        )
        results.append(m)
        model.cpu()
        del model
        torch.cuda.empty_cache()
    return results


def run_ood_ctrgcn(
    label: str,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    ood_d: str,
    ood_l: str,
    ckpt: str,
) -> dict:
    if not check_files(ood_d, ood_l):
        print(f"  Skipping {label} — missing OOD test files.")
        return {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0, "n": 0}
    model = train_model_ctrgcn(train_ds, val_ds, ckpt, label)
    m = evaluate(model, DataLoader(load_ds(ood_d, ood_l), batch_size=TRAIN_BATCH_SIZE), tag=label)
    print(
        f"  OOD test: "
        f"acc={m['acc']:.3f}  prec={m['prec']:.3f}  "
        f"rec={m['rec']:.3f}  F1={m['f1']:.3f}  n={m['n']}"
    )
    model.cpu()
    del model
    torch.cuda.empty_cache()
    return m


# ── Main ──────────────────────────────────────────────────────────────────────

def main(ood_only: bool = False, mixed_only: bool = False, iid_only: bool = False):
    global _confmat_logger, _MODEL_LOG_NAME
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _MODEL_LOG_NAME = "ST-GCN"
    if mixed_only:
        _confmat_logger = None
    else:
        _init_confmat(STGCN_SLUG)
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random seed fixed at 42 — results are reproducible")
    print(f"Device        : {DEVICE}")
    print(f"Early-stop    : patience={EARLY_STOP_PAT} (val loss)")
    print(f"Max epochs    : {TRAIN_EPOCHS}")
    if ood_only:
        print("Mode          : --ood-only (skipping IID LOSO / k-fold experiments)")
    if mixed_only:
        print("Mode          : --mixed-only (D1 ST-GCN + D1b CTR-GCN only)")
    if iid_only:
        print("Mode          : --iid-only (IID A1-A4 only; skipping OOD and mixed)")
    if sum((ood_only, mixed_only, iid_only)) > 1:
        raise SystemExit("Use at most one of --ood-only, --mixed-only, --iid-only.")

    summary: List[Tuple[str, str, str]] = []

    if mixed_only:
        section_labels = {
            "A": "IID (within-distribution)",
            "B": "OOD — single source train",
            "C": "OOD — dual source train",
            "D": "Mixed (all datasets)",
        }

        ec3d_full_d = p("ec3d_train_data.npy")
        ec3d_full_l = p("ec3d_train_labels.npy")
        re_full_d   = p("rehab24_train_data.npy")
        re_full_l   = p("rehab24_train_labels.npy")
        custom_d    = p("custom_squat_train_data.npy")
        custom_l    = p("custom_squat_train_labels.npy")
        uiprmd_d    = p("uiprmd_data.npy")
        uiprmd_l    = p("uiprmd_labels.npy")

        print("\n\n" + "=" * 70)
        print("D1 — Mixed: all 4 datasets  80% train / 10% val / 10% test")
        print("=" * 70)
        if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, custom_d, custom_l,
                       uiprmd_d, uiprmd_l):
            all_ds = concat_ds(
                (ec3d_full_d, ec3d_full_l),
                (re_full_d,   re_full_l),
                (custom_d,    custom_l),
                (uiprmd_d,    uiprmd_l),
            )
            d_tr, d_va, d_te = split_80_10_10(all_ds, seed=RANDOM_SEED)
            print(
                f"  Combined: {len(all_ds)} reps  →  "
                f"train={len(d_tr)}  val={len(d_va)}  test={len(d_te)}"
            )
            model_d = train_model(d_tr, d_va, p("mixed_all_datasets.pth"), "D1 Mixed")
            d1 = evaluate(model_d, DataLoader(d_te, batch_size=TRAIN_BATCH_SIZE), tag="")
            print(
                f"  Mixed test: "
                f"acc={d1['acc']:.3f}  prec={d1['prec']:.3f}  "
                f"rec={d1['rec']:.3f}  F1={d1['f1']:.3f}  "
                f"MacroF1={d1['macro_f1']:.3f}  n={d1['n']}"
            )
            summary.append(("D", "D1 Mixed all 80/10/10", _row("D1 Mixed all 80/10/10", d1)))

        print_and_save_summary(
            summary,
            "ST-GCN FINAL SUMMARY",
            HDR,
            SEP,
            section_labels,
            save_path=mixed_results_path(OUTPUT_DIR, STGCN_SLUG),
        )
        print(f"\nCheckpoints saved in {OUTPUT_DIR}/:")
        print("  mixed_all_datasets.pth")

        summary_ctrgcn: List[Tuple[str, str, str]] = []

        _MODEL_LOG_NAME = "CTR-GCN"

        print("\n\n" + "=" * 70)
        print("CTR-GCN EXPERIMENTS")
        print("=" * 70)

        print("\n\n" + "=" * 70)
        print("D1b — Mixed: all 4 datasets  80% train / 10% val / 10% test")
        print("=" * 70)
        if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, custom_d, custom_l,
                       uiprmd_d, uiprmd_l):
            all_ds_b = concat_ds(
                (ec3d_full_d, ec3d_full_l),
                (re_full_d,   re_full_l),
                (custom_d,    custom_l),
                (uiprmd_d,    uiprmd_l),
            )
            d_tr_b, d_va_b, d_te_b = split_80_10_10(all_ds_b, seed=RANDOM_SEED)
            print(
                f"  Combined: {len(all_ds_b)} reps  →  "
                f"train={len(d_tr_b)}  val={len(d_va_b)}  test={len(d_te_b)}"
            )
            model_d_b = train_model_ctrgcn(
                d_tr_b, d_va_b, p("ctrgcn_mixed_all_datasets.pth"), "D1b Mixed",
            )
            d1b = evaluate(model_d_b, DataLoader(d_te_b, batch_size=TRAIN_BATCH_SIZE), tag="")
            print(
                f"  Mixed test: "
                f"acc={d1b['acc']:.3f}  prec={d1b['prec']:.3f}  "
                f"rec={d1b['rec']:.3f}  F1={d1b['f1']:.3f}  "
                f"MacroF1={d1b['macro_f1']:.3f}  n={d1b['n']}"
            )
            summary_ctrgcn.append(("D", "D1b Mixed all 80/10/10", _row("D1b Mixed all 80/10/10", d1b)))

        print_and_save_summary(
            summary_ctrgcn,
            "CTR-GCN FINAL SUMMARY",
            HDR,
            SEP,
            section_labels,
            save_path=mixed_results_path(OUTPUT_DIR, CTRGCN_SLUG),
        )
        print("\nMode: --mixed-only — skipping IID and OOD experiments.")
        print(f"\nCTR-GCN checkpoints saved in {OUTPUT_DIR}/:")
        print("  ctrgcn_mixed_all_datasets.pth")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # A — IID experiments
    # ─────────────────────────────────────────────────────────────────────────

    if not ood_only:
        print("\n\n" + "=" * 70)
        print("A1 — EC3D LOSO  (4 rounds)")
        print("=" * 70)
        a1 = run_loso("ec3d", 4, "loso_ec3d")
        if a1:
            summary.append(("A", "A1 EC3D LOSO (4 rounds)", _row_ms("A1 EC3D LOSO (4 rounds)", a1)))

        print("\n\n" + "=" * 70)
        print("A2 — REHAB24 LOSO  (9 rounds)")
        print("=" * 70)
        a2 = run_loso("rehab24", 9, "loso_rehab24")
        if a2:
            summary.append(("A", "A2 REHAB24 LOSO (9 rounds)", _row_ms("A2 REHAB24 LOSO (9 rounds)", a2)))

        print("\n\n" + "=" * 70)
        print("A4 — UI-PRMD LOSO  (10 rounds)")
        print("=" * 70)
        a4 = run_loso("uiprmd", 10, "loso_uiprmd")
        if a4:
            summary.append(("A", "A4 UI-PRMD LOSO (10 rounds)", _row_ms("A4 UI-PRMD LOSO (10 rounds)", a4)))

        print("\n\n" + "=" * 70)
        print("A3 — Custom 5-Fold CV")
        print("=" * 70)
        a3 = run_kfold(5, "kfold_custom")
        if a3:
            summary.append(("A", "A3 Custom 5-Fold CV", _row_ms("A3 Custom 5-Fold CV", a3)))

    if iid_only:
        print_and_save_summary(
            summary,
            "ST-GCN FINAL SUMMARY",
            HDR,
            SEP,
            SECTION_LABELS,
            save_path=iid_results_path(OUTPUT_DIR, STGCN_SLUG),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # B — OOD, single source
    # ─────────────────────────────────────────────────────────────────────────

    if not iid_only:
        ec3d_full_d = p("ec3d_train_data.npy");    ec3d_full_l = p("ec3d_train_labels.npy")
        re_full_d   = p("rehab24_train_data.npy"); re_full_l   = p("rehab24_train_labels.npy")
        custom_d      = p("custom_squat_train_data.npy"); custom_l      = p("custom_squat_train_labels.npy")

        # B1 + B2 share the same EC3D 80% train split (inline from full dataset)
        print("\n\n" + "=" * 70)
        print("B1 — Train EC3D 80%  →  test REHAB24 OOD")
        print("=" * 70)
        if check_files(ec3d_full_d, ec3d_full_l):
            ec3d_tr, ec3d_va = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l)), 0.80, seed=RANDOM_SEED
            )
            b1 = run_ood(
            "B1 EC3D→REHAB24 OOD", ec3d_tr, ec3d_va,
            p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
            p("ood_b1_ec3d_to_rehab24.pth"),
            )
            summary.append(("B", "B1 EC3D→REHAB24 OOD", _row("B1 EC3D→REHAB24 OOD", b1)))
            evaluate_with_confusion("B1 EC3D→REHAB24 OOD",
                                p("ood_b1_ec3d_to_rehab24.pth"),
                                p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"))

            print("\n\n" + "=" * 70)
            print("B2 — Train EC3D 80%  →  test Custom OOD")
            print("=" * 70)
            b2 = run_ood(
            "B2 EC3D→Custom OOD", ec3d_tr, ec3d_va,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p("ood_b2_ec3d_to_custom.pth"),
            )
            summary.append(("B", "B2 EC3D→Custom OOD", _row("B2 EC3D→Custom OOD", b2)))
            evaluate_with_confusion("B2 EC3D→Custom OOD",
                                p("ood_b2_ec3d_to_custom.pth"),
                                p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"))

        # B3 + B4 share the same REHAB24 80% train split (inline from full dataset)
        print("\n\n" + "=" * 70)
        print("B3 — Train REHAB24 80%  →  test EC3D OOD")
        print("=" * 70)
        if check_files(re_full_d, re_full_l):
            re_tr, re_va = split_frac(
            concat_ds((re_full_d, re_full_l)), 0.80, seed=RANDOM_SEED
            )
            b3 = run_ood(
            "B3 REHAB24→EC3D OOD", re_tr, re_va,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p("ood_b3_rehab24_to_ec3d.pth"),
            )
            summary.append(("B", "B3 REHAB24→EC3D OOD", _row("B3 REHAB24→EC3D OOD", b3)))
            evaluate_with_confusion("B3 REHAB24→EC3D OOD",
                                p("ood_b3_rehab24_to_ec3d.pth"),
                                p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"))

            print("\n\n" + "=" * 70)
            print("B4 — Train REHAB24 80%  →  test Custom OOD")
            print("=" * 70)
            b4 = run_ood(
            "B4 REHAB24→Custom OOD", re_tr, re_va,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p("ood_b4_rehab24_to_custom.pth"),
            )
            summary.append(("B", "B4 REHAB24→Custom OOD", _row("B4 REHAB24→Custom OOD", b4)))
            evaluate_with_confusion("B4 REHAB24→Custom OOD",
                                p("ood_b4_rehab24_to_custom.pth"),
                                p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"))

        # B5 + B6 share the same Custom 80% train split (inline)
        print("\n\n" + "=" * 70)
        print("B5 — Train Custom 80%  →  test EC3D OOD")
        print("=" * 70)
        if check_files(custom_d, custom_l):
            c_tr, c_va = split_frac(concat_ds((custom_d, custom_l)), 0.80, seed=RANDOM_SEED)
            b5 = run_ood(
            "B5 Custom→EC3D OOD", c_tr, c_va,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p("ood_b5_custom_to_ec3d.pth"),
            )
            summary.append(("B", "B5 Custom→EC3D OOD", _row("B5 Custom→EC3D OOD", b5)))
            evaluate_with_confusion("B5 Custom→EC3D OOD",
                                p("ood_b5_custom_to_ec3d.pth"),
                                p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"))

            print("\n\n" + "=" * 70)
            print("B6 — Train Custom 80%  →  test REHAB24 OOD")
            print("=" * 70)
            b6 = run_ood(
            "B6 Custom→REHAB24 OOD", c_tr, c_va,
            p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
            p("ood_b6_custom_to_rehab24.pth"),
            )
            summary.append(("B", "B6 Custom→REHAB24 OOD", _row("B6 Custom→REHAB24 OOD", b6)))
            evaluate_with_confusion("B6 Custom→REHAB24 OOD",
                                p("ood_b6_custom_to_rehab24.pth"),
                                p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"))

            # ─────────────────────────────────────────────────────────────────────────
        # C — OOD, dual source (90% train / 10% inline val)
        # ─────────────────────────────────────────────────────────────────────────

        print("\n\n" + "=" * 70)
        print("C1 — Train EC3D+REHAB24  →  test Custom OOD")
        print("=" * 70)
        if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l):
            c1_tr, c1_va = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (re_full_d, re_full_l)),
            0.90, seed=RANDOM_SEED,
            )
            c1 = run_ood(
            "C1 EC3D+REHAB24→Custom OOD", c1_tr, c1_va,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p("ood_c1_ec3d_rehab24_to_custom.pth"),
            )
            summary.append(("C", "C1 EC3D+REHAB24→Custom OOD", _row("C1 EC3D+REHAB24→Custom OOD", c1)))
            evaluate_with_confusion("C1 EC3D+REHAB24→Custom OOD",
                                p("ood_c1_ec3d_rehab24_to_custom.pth"),
                                p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"))

            print("\n\n" + "=" * 70)
            print("C2 — Train EC3D+Custom  →  test REHAB24 OOD")
            print("=" * 70)
        if check_files(ec3d_full_d, ec3d_full_l, custom_d, custom_l):
            c2_tr, c2_va = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (custom_d, custom_l)),
            0.90, seed=RANDOM_SEED,
            )
            c2 = run_ood(
            "C2 EC3D+Custom→REHAB24 OOD", c2_tr, c2_va,
            p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
            p("ood_c2_ec3d_custom_to_rehab24.pth"),
            )
            summary.append(("C", "C2 EC3D+Custom→REHAB24 OOD", _row("C2 EC3D+Custom→REHAB24 OOD", c2)))
            evaluate_with_confusion("C2 EC3D+Custom→REHAB24 OOD",
                                p("ood_c2_ec3d_custom_to_rehab24.pth"),
                                p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"))

            print("\n\n" + "=" * 70)
            print("C3 — Train REHAB24+Custom  →  test EC3D OOD")
            print("=" * 70)
        if check_files(re_full_d, re_full_l, custom_d, custom_l):
            c3_tr, c3_va = split_frac(
            concat_ds((re_full_d, re_full_l), (custom_d, custom_l)),
            0.90, seed=RANDOM_SEED,
            )
            c3 = run_ood(
            "C3 REHAB24+Custom→EC3D OOD", c3_tr, c3_va,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p("ood_c3_rehab24_custom_to_ec3d.pth"),
            )
            summary.append(("C", "C3 REHAB24+Custom→EC3D OOD", _row("C3 REHAB24+Custom→EC3D OOD", c3)))
            evaluate_with_confusion("C3 REHAB24+Custom→EC3D OOD",
                                p("ood_c3_rehab24_custom_to_ec3d.pth"),
                                p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"))

            uiprmd_d = p("uiprmd_data.npy")
            uiprmd_l = p("uiprmd_labels.npy")

            print("\n\n" + "=" * 70)
            print("B7 — Train UI-PRMD 80%  →  test EC3D OOD")
            print("=" * 70)
        if check_files(uiprmd_d, uiprmd_l):
            ui_tr, ui_va = split_frac(concat_ds((uiprmd_d, uiprmd_l)), 0.80, seed=RANDOM_SEED)
            b7 = run_ood(
            "B7 UI-PRMD→EC3D OOD", ui_tr, ui_va,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p("ood_b7_uiprmd_to_ec3d.pth"),
            )
            summary.append(("B", "B7 UI-PRMD→EC3D OOD", _row("B7 UI-PRMD→EC3D OOD", b7)))

            print("\n\n" + "=" * 70)
            print("B8 — Train UI-PRMD 80%  →  test REHAB24 OOD")
            print("=" * 70)
            b8 = run_ood(
            "B8 UI-PRMD→REHAB24 OOD", ui_tr, ui_va,
            p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
            p("ood_b8_uiprmd_to_rehab24.pth"),
            )
            summary.append(("B", "B8 UI-PRMD→REHAB24 OOD", _row("B8 UI-PRMD→REHAB24 OOD", b8)))

            print("\n\n" + "=" * 70)
            print("B9 — Train UI-PRMD 80%  →  test Custom OOD")
            print("=" * 70)
            b9 = run_ood(
            "B9 UI-PRMD→Custom OOD", ui_tr, ui_va,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p("ood_b9_uiprmd_to_custom.pth"),
            )
            summary.append(("B", "B9 UI-PRMD→Custom OOD", _row("B9 UI-PRMD→Custom OOD", b9)))

            print("\n\n" + "=" * 70)
            print("B10 — Train EC3D 80%  →  test UI-PRMD OOD")
            print("=" * 70)
        if check_files(ec3d_full_d, ec3d_full_l):
            ec3d_tr, ec3d_va = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l)), 0.80, seed=RANDOM_SEED
            )
            b10 = run_ood(
            "B10 EC3D→UI-PRMD OOD", ec3d_tr, ec3d_va,
            p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
            p("ood_b10_ec3d_to_uiprmd.pth"),
            )
            summary.append(("B", "B10 EC3D→UI-PRMD OOD", _row("B10 EC3D→UI-PRMD OOD", b10)))

            print("\n\n" + "=" * 70)
            print("B11 — Train REHAB24 80%  →  test UI-PRMD OOD")
            print("=" * 70)
        if check_files(re_full_d, re_full_l):
            re_tr, re_va = split_frac(
            concat_ds((re_full_d, re_full_l)), 0.80, seed=RANDOM_SEED
            )
            b11 = run_ood(
            "B11 REHAB24→UI-PRMD OOD", re_tr, re_va,
            p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
            p("ood_b11_rehab24_to_uiprmd.pth"),
            )
            summary.append(("B", "B11 REHAB24→UI-PRMD OOD", _row("B11 REHAB24→UI-PRMD OOD", b11)))

            print("\n\n" + "=" * 70)
            print("B12 — Train Custom 80%  →  test UI-PRMD OOD")
            print("=" * 70)
        if check_files(custom_d, custom_l):
            c_tr_b12, c_va_b12 = split_frac(concat_ds((custom_d, custom_l)), 0.80, seed=RANDOM_SEED)
            b12 = run_ood(
            "B12 Custom→UI-PRMD OOD", c_tr_b12, c_va_b12,
            p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
            p("ood_b12_custom_to_uiprmd.pth"),
            )
            summary.append(("B", "B12 Custom→UI-PRMD OOD", _row("B12 Custom→UI-PRMD OOD", b12)))

            print("\n\n" + "=" * 70)
            print("C4 — Train EC3D+UI-PRMD  →  test Custom OOD")
            print("=" * 70)
        if check_files(ec3d_full_d, ec3d_full_l, uiprmd_d, uiprmd_l):
            c4_tr, c4_va = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (uiprmd_d, uiprmd_l)),
            0.90, seed=RANDOM_SEED,
            )
            c4 = run_ood(
            "C4 EC3D+UI-PRMD→Custom OOD", c4_tr, c4_va,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p("ood_c4_ec3d_uiprmd_to_custom.pth"),
            )
            summary.append(("C", "C4 EC3D+UI-PRMD→Custom OOD", _row("C4 EC3D+UI-PRMD→Custom OOD", c4)))

            print("\n\n" + "=" * 70)
            print("C5 — Train REHAB24+UI-PRMD  →  test Custom OOD")
            print("=" * 70)
        if check_files(re_full_d, re_full_l, uiprmd_d, uiprmd_l):
            c5_tr, c5_va = split_frac(
            concat_ds((re_full_d, re_full_l), (uiprmd_d, uiprmd_l)),
            0.90, seed=RANDOM_SEED,
            )
            c5 = run_ood(
            "C5 REHAB24+UI-PRMD→Custom OOD", c5_tr, c5_va,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p("ood_c5_rehab24_uiprmd_to_custom.pth"),
            )
            summary.append(("C", "C5 REHAB24+UI-PRMD→Custom OOD", _row("C5 REHAB24+UI-PRMD→Custom OOD", c5)))

            print("\n\n" + "=" * 70)
            print("C6 — Train EC3D+REHAB24+Custom  →  test UI-PRMD OOD")
            print("=" * 70)
        if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, custom_d, custom_l):
            c6_tr, c6_va = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (re_full_d, re_full_l), (custom_d, custom_l)),
            0.90, seed=RANDOM_SEED,
            )
            c6 = run_ood(
            "C6 EC3D+REHAB24+Custom→UI-PRMD OOD", c6_tr, c6_va,
            p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
            p("ood_c6_ec3d_rehab24_custom_to_uiprmd.pth"),
            )
            summary.append(("C", "C6 EC3D+REHAB24+Custom→UI-PRMD OOD",
                        _row("C6 EC3D+REHAB24+Custom→UI-PRMD OOD", c6)))

            print("\n\n" + "=" * 70)
            print("C8 — Train EC3D+REHAB24+UI-PRMD  →  test Custom OOD")
            print("=" * 70)
        if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, uiprmd_d, uiprmd_l):
            c8_tr, c8_va = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (re_full_d, re_full_l), (uiprmd_d, uiprmd_l)),
            0.90, seed=RANDOM_SEED,
            )
            c8 = run_ood(
            "C8 EC3D+REHAB24+UI-PRMD→Custom OOD", c8_tr, c8_va,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p("ood_c8_ec3d_rehab24_uiprmd_to_custom.pth"),
            )
            summary.append(("C", "C8 EC3D+REHAB24+UI-PRMD→Custom OOD",
                        _row("C8 EC3D+REHAB24+UI-PRMD→Custom OOD", c8)))

            print("\n\n" + "=" * 70)
            print("C9 — Train EC3D+Custom+UI-PRMD  →  test REHAB24 OOD")
            print("=" * 70)
        if check_files(ec3d_full_d, ec3d_full_l, custom_d, custom_l, uiprmd_d, uiprmd_l):
            c9_tr, c9_va = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (custom_d, custom_l), (uiprmd_d, uiprmd_l)),
            0.90, seed=RANDOM_SEED,
            )
            c9 = run_ood(
            "C9 EC3D+Custom+UI-PRMD→REHAB24 OOD", c9_tr, c9_va,
            p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
            p("ood_c9_ec3d_custom_uiprmd_to_rehab24.pth"),
            )
            summary.append(("C", "C9 EC3D+Custom+UI-PRMD→REHAB24 OOD",
                        _row("C9 EC3D+Custom+UI-PRMD→REHAB24 OOD", c9)))

            print("\n\n" + "=" * 70)
            print("C10 — Train REHAB24+Custom+UI-PRMD  →  test EC3D OOD")
            print("=" * 70)
        if check_files(re_full_d, re_full_l, custom_d, custom_l, uiprmd_d, uiprmd_l):
            c10_tr, c10_va = split_frac(
            concat_ds((re_full_d, re_full_l), (custom_d, custom_l), (uiprmd_d, uiprmd_l)),
            0.90, seed=RANDOM_SEED,
            )
            c10 = run_ood(
            "C10 REHAB24+Custom+UI-PRMD→EC3D OOD", c10_tr, c10_va,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p("ood_c10_rehab24_custom_uiprmd_to_ec3d.pth"),
            )
            summary.append(("C", "C10 REHAB24+Custom+UI-PRMD→EC3D OOD",
                        _row("C10 REHAB24+Custom+UI-PRMD→EC3D OOD", c10)))

            print("\n\n" + "=" * 70)
            print("C7 — Train Custom+UI-PRMD  →  test EC3D OOD")
            print("=" * 70)
        if check_files(custom_d, custom_l, uiprmd_d, uiprmd_l):
            c7_tr, c7_va = split_frac(
            concat_ds((custom_d, custom_l), (uiprmd_d, uiprmd_l)),
            0.90, seed=RANDOM_SEED,
            )
            c7 = run_ood(
            "C7 Custom+UI-PRMD→EC3D OOD", c7_tr, c7_va,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p("ood_c7_custom_uiprmd_to_ec3d.pth"),
            )
            summary.append(("C", "C7 Custom+UI-PRMD→EC3D OOD", _row("C7 Custom+UI-PRMD→EC3D OOD", c7)))

            # ─────────────────────────────────────────────────────────────────────────
        # D — Mixed (all 3 datasets, 80/10/10)
        # ─────────────────────────────────────────────────────────────────────────

        print("\n\n" + "=" * 70)
        print("D1 — Mixed: all 3 datasets  80% train / 10% val / 10% test")
        print("=" * 70)
        if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, custom_d, custom_l,
                   uiprmd_d, uiprmd_l):
            all_ds = concat_ds(
            (ec3d_full_d, ec3d_full_l),
            (re_full_d,   re_full_l),
            (custom_d,    custom_l),
            (uiprmd_d,    uiprmd_l),
            )
            d_tr, d_va, d_te = split_80_10_10(all_ds, seed=RANDOM_SEED)
            print(
            f"  Combined: {len(all_ds)} reps  →  "
            f"train={len(d_tr)}  val={len(d_va)}  test={len(d_te)}"
            )
            model_d = train_model(d_tr, d_va, p("mixed_all_datasets.pth"), "D1 Mixed")
            d1 = evaluate(model_d, DataLoader(d_te, batch_size=TRAIN_BATCH_SIZE), tag="D1 Mixed all 80/10/10")
            print(
            f"  Mixed test: "
            f"acc={d1['acc']:.3f}  prec={d1['prec']:.3f}  "
            f"rec={d1['rec']:.3f}  F1={d1['f1']:.3f}  n={d1['n']}"
            )
            summary.append(("D", "D1 Mixed all 80/10/10", _row("D1 Mixed all 80/10/10", d1)))

            # ─────────────────────────────────────────────────────────────────────────
        # FINAL SUMMARY TABLE
        # ─────────────────────────────────────────────────────────────────────────

        section_labels = {
            "A": "IID (within-distribution)",
            "B": "OOD — single source train",
            "C": "OOD — dual source train",
            "D": "Mixed (all datasets)",
        }

        print("\n\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(HDR)

        last_sec = None
        for sec, _tag, row_str in summary:
            if sec != last_sec:
                print(SEP)
                print(f"  [{section_labels.get(sec, sec)}]")
                last_sec = sec
            print(row_str)
        print(SEP)

        print(f"\nCheckpoints saved in {OUTPUT_DIR}/:")
        for name in [
            "loso_ec3d_round{{1..4}}.pth",
            "loso_rehab24_round{{1..9}}.pth",
            "loso_uiprmd_round{{1..10}}.pth",
            "kfold_custom_fold{{1..5}}.pth",
            "ood_b{{1..12}}_*.pth",
            "ood_c{{1..10}}_*.pth",
            "mixed_all_datasets.pth",
        ]:
            print(f"  {name}")

    # ─────────────────────────────────────────────────────────────────────────
    # CTR-GCN — mirror of all ST-GCN experiments
    # ─────────────────────────────────────────────────────────────────────────

    summary_ctrgcn: List[Tuple[str, str, str]] = []

    _MODEL_LOG_NAME = "CTR-GCN"
    _init_confmat(CTRGCN_SLUG)

    print("\n\n" + "=" * 70)
    print("CTR-GCN EXPERIMENTS")
    print("=" * 70)

    if not ood_only:
        print("\n\n" + "=" * 70)
        print("A1b — EC3D LOSO  (4 rounds)")
        print("=" * 70)
        a1b = run_loso_ctrgcn("ec3d", 4, "ctrgcn_loso_ec3d")
        if a1b:
            summary_ctrgcn.append(("A", "A1b EC3D LOSO (4 rounds)",
                                   _row_ms("A1b EC3D LOSO (4 rounds)", a1b)))

        print("\n\n" + "=" * 70)
        print("A2b — REHAB24 LOSO  (9 rounds)")
        print("=" * 70)
        a2b = run_loso_ctrgcn("rehab24", 9, "ctrgcn_loso_rehab24")
        if a2b:
            summary_ctrgcn.append(("A", "A2b REHAB24 LOSO (9 rounds)",
                                   _row_ms("A2b REHAB24 LOSO (9 rounds)", a2b)))

        print("\n\n" + "=" * 70)
        print("A4b — UI-PRMD LOSO  (10 rounds)")
        print("=" * 70)
        a4b = run_loso_ctrgcn("uiprmd", 10, "ctrgcn_loso_uiprmd")
        if a4b:
            summary_ctrgcn.append(("A", "A4b UI-PRMD LOSO (10 rounds)",
                                   _row_ms("A4b UI-PRMD LOSO (10 rounds)", a4b)))

        print("\n\n" + "=" * 70)
        print("A3b — Custom 5-Fold CV")
        print("=" * 70)
        a3b = run_kfold_ctrgcn(5, "ctrgcn_kfold_custom")
        if a3b:
            summary_ctrgcn.append(("A", "A3b Custom 5-Fold CV",
                                   _row_ms("A3b Custom 5-Fold CV", a3b)))

    if iid_only:
        print_and_save_summary(
            summary_ctrgcn,
            "CTR-GCN FINAL SUMMARY",
            HDR,
            SEP,
            SECTION_LABELS,
            save_path=iid_results_path(OUTPUT_DIR, CTRGCN_SLUG),
        )
        print("\nMode: --iid-only — skipping OOD and mixed experiments.")
        return

    print("\n\n" + "=" * 70)
    print("B1b — Train EC3D 80%  →  test REHAB24 OOD")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l):
        ec3d_tr_b, ec3d_va_b = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l)), 0.80, seed=RANDOM_SEED
        )
        b1b = run_ood_ctrgcn(
            "B1b EC3D→REHAB24 OOD", ec3d_tr_b, ec3d_va_b,
            p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
            p("ctrgcn_ood_b1_ec3d_to_rehab24.pth"),
        )
        summary_ctrgcn.append(("B", "B1b EC3D→REHAB24 OOD", _row("B1b EC3D→REHAB24 OOD", b1b)))

        print("\n\n" + "=" * 70)
        print("B2b — Train EC3D 80%  →  test Custom OOD")
        print("=" * 70)
        b2b = run_ood_ctrgcn(
            "B2b EC3D→Custom OOD", ec3d_tr_b, ec3d_va_b,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p("ctrgcn_ood_b2_ec3d_to_custom.pth"),
        )
        summary_ctrgcn.append(("B", "B2b EC3D→Custom OOD", _row("B2b EC3D→Custom OOD", b2b)))

    print("\n\n" + "=" * 70)
    print("B3b — Train REHAB24 80%  →  test EC3D OOD")
    print("=" * 70)
    if check_files(re_full_d, re_full_l):
        re_tr_b, re_va_b = split_frac(
            concat_ds((re_full_d, re_full_l)), 0.80, seed=RANDOM_SEED
        )
        b3b = run_ood_ctrgcn(
            "B3b REHAB24→EC3D OOD", re_tr_b, re_va_b,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p("ctrgcn_ood_b3_rehab24_to_ec3d.pth"),
        )
        summary_ctrgcn.append(("B", "B3b REHAB24→EC3D OOD", _row("B3b REHAB24→EC3D OOD", b3b)))

        print("\n\n" + "=" * 70)
        print("B4b — Train REHAB24 80%  →  test Custom OOD")
        print("=" * 70)
        b4b = run_ood_ctrgcn(
            "B4b REHAB24→Custom OOD", re_tr_b, re_va_b,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p("ctrgcn_ood_b4_rehab24_to_custom.pth"),
        )
        summary_ctrgcn.append(("B", "B4b REHAB24→Custom OOD", _row("B4b REHAB24→Custom OOD", b4b)))

    print("\n\n" + "=" * 70)
    print("B5b — Train Custom 80%  →  test EC3D OOD")
    print("=" * 70)
    if check_files(custom_d, custom_l):
        c_tr_b, c_va_b = split_frac(concat_ds((custom_d, custom_l)), 0.80, seed=RANDOM_SEED)
        b5b = run_ood_ctrgcn(
            "B5b Custom→EC3D OOD", c_tr_b, c_va_b,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p("ctrgcn_ood_b5_custom_to_ec3d.pth"),
        )
        summary_ctrgcn.append(("B", "B5b Custom→EC3D OOD", _row("B5b Custom→EC3D OOD", b5b)))

        print("\n\n" + "=" * 70)
        print("B6b — Train Custom 80%  →  test REHAB24 OOD")
        print("=" * 70)
        b6b = run_ood_ctrgcn(
            "B6b Custom→REHAB24 OOD", c_tr_b, c_va_b,
            p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
            p("ctrgcn_ood_b6_custom_to_rehab24.pth"),
        )
        summary_ctrgcn.append(("B", "B6b Custom→REHAB24 OOD", _row("B6b Custom→REHAB24 OOD", b6b)))

    print("\n\n" + "=" * 70)
    print("C1b — Train EC3D+REHAB24  →  test Custom OOD")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l):
        c1_tr_b, c1_va_b = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (re_full_d, re_full_l)),
            0.90, seed=RANDOM_SEED,
        )
        c1b = run_ood_ctrgcn(
            "C1b EC3D+REHAB24→Custom OOD", c1_tr_b, c1_va_b,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p("ctrgcn_ood_c1_ec3d_rehab24_to_custom.pth"),
        )
        summary_ctrgcn.append(("C", "C1b EC3D+REHAB24→Custom OOD",
                               _row("C1b EC3D+REHAB24→Custom OOD", c1b)))

    print("\n\n" + "=" * 70)
    print("C2b — Train EC3D+Custom  →  test REHAB24 OOD")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l, custom_d, custom_l):
        c2_tr_b, c2_va_b = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (custom_d, custom_l)),
            0.90, seed=RANDOM_SEED,
        )
        c2b = run_ood_ctrgcn(
            "C2b EC3D+Custom→REHAB24 OOD", c2_tr_b, c2_va_b,
            p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
            p("ctrgcn_ood_c2_ec3d_custom_to_rehab24.pth"),
        )
        summary_ctrgcn.append(("C", "C2b EC3D+Custom→REHAB24 OOD",
                               _row("C2b EC3D+Custom→REHAB24 OOD", c2b)))

    print("\n\n" + "=" * 70)
    print("C3b — Train REHAB24+Custom  →  test EC3D OOD")
    print("=" * 70)
    if check_files(re_full_d, re_full_l, custom_d, custom_l):
        c3_tr_b, c3_va_b = split_frac(
            concat_ds((re_full_d, re_full_l), (custom_d, custom_l)),
            0.90, seed=RANDOM_SEED,
        )
        c3b = run_ood_ctrgcn(
            "C3b REHAB24+Custom→EC3D OOD", c3_tr_b, c3_va_b,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p("ctrgcn_ood_c3_rehab24_custom_to_ec3d.pth"),
        )
        summary_ctrgcn.append(("C", "C3b REHAB24+Custom→EC3D OOD",
                               _row("C3b REHAB24+Custom→EC3D OOD", c3b)))

    uiprmd_d_b = p("uiprmd_data.npy")
    uiprmd_l_b = p("uiprmd_labels.npy")

    print("\n\n" + "=" * 70)
    print("B7b — Train UI-PRMD 80%  →  test EC3D OOD")
    print("=" * 70)
    if check_files(uiprmd_d_b, uiprmd_l_b):
        ui_tr_b, ui_va_b = split_frac(concat_ds((uiprmd_d_b, uiprmd_l_b)), 0.80, seed=RANDOM_SEED)
        b7b = run_ood_ctrgcn(
            "B7b UI-PRMD→EC3D OOD", ui_tr_b, ui_va_b,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p("ctrgcn_ood_b7_uiprmd_to_ec3d.pth"),
        )
        summary_ctrgcn.append(("B", "B7b UI-PRMD→EC3D OOD", _row("B7b UI-PRMD→EC3D OOD", b7b)))

        print("\n\n" + "=" * 70)
        print("B8b — Train UI-PRMD 80%  →  test REHAB24 OOD")
        print("=" * 70)
        b8b = run_ood_ctrgcn(
            "B8b UI-PRMD→REHAB24 OOD", ui_tr_b, ui_va_b,
            p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
            p("ctrgcn_ood_b8_uiprmd_to_rehab24.pth"),
        )
        summary_ctrgcn.append(("B", "B8b UI-PRMD→REHAB24 OOD", _row("B8b UI-PRMD→REHAB24 OOD", b8b)))

        print("\n\n" + "=" * 70)
        print("B9b — Train UI-PRMD 80%  →  test Custom OOD")
        print("=" * 70)
        b9b = run_ood_ctrgcn(
            "B9b UI-PRMD→Custom OOD", ui_tr_b, ui_va_b,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p("ctrgcn_ood_b9_uiprmd_to_custom.pth"),
        )
        summary_ctrgcn.append(("B", "B9b UI-PRMD→Custom OOD", _row("B9b UI-PRMD→Custom OOD", b9b)))

    print("\n\n" + "=" * 70)
    print("B10b — Train EC3D 80%  →  test UI-PRMD OOD")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l):
        ec3d_tr_b, ec3d_va_b = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l)), 0.80, seed=RANDOM_SEED
        )
        b10b = run_ood_ctrgcn(
            "B10b EC3D→UI-PRMD OOD", ec3d_tr_b, ec3d_va_b,
            p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
            p("ctrgcn_ood_b10_ec3d_to_uiprmd.pth"),
        )
        summary_ctrgcn.append(("B", "B10b EC3D→UI-PRMD OOD", _row("B10b EC3D→UI-PRMD OOD", b10b)))

    print("\n\n" + "=" * 70)
    print("B11b — Train REHAB24 80%  →  test UI-PRMD OOD")
    print("=" * 70)
    if check_files(re_full_d, re_full_l):
        re_tr_b, re_va_b = split_frac(
            concat_ds((re_full_d, re_full_l)), 0.80, seed=RANDOM_SEED
        )
        b11b = run_ood_ctrgcn(
            "B11b REHAB24→UI-PRMD OOD", re_tr_b, re_va_b,
            p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
            p("ctrgcn_ood_b11_rehab24_to_uiprmd.pth"),
        )
        summary_ctrgcn.append(("B", "B11b REHAB24→UI-PRMD OOD", _row("B11b REHAB24→UI-PRMD OOD", b11b)))

    print("\n\n" + "=" * 70)
    print("B12b — Train Custom 80%  →  test UI-PRMD OOD")
    print("=" * 70)
    if check_files(custom_d, custom_l):
        c_tr_b12b, c_va_b12b = split_frac(concat_ds((custom_d, custom_l)), 0.80, seed=RANDOM_SEED)
        b12b = run_ood_ctrgcn(
            "B12b Custom→UI-PRMD OOD", c_tr_b12b, c_va_b12b,
            p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
            p("ctrgcn_ood_b12_custom_to_uiprmd.pth"),
        )
        summary_ctrgcn.append(("B", "B12b Custom→UI-PRMD OOD", _row("B12b Custom→UI-PRMD OOD", b12b)))

    print("\n\n" + "=" * 70)
    print("C4b — Train EC3D+UI-PRMD  →  test Custom OOD")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l, uiprmd_d_b, uiprmd_l_b):
        c4_tr_b, c4_va_b = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (uiprmd_d_b, uiprmd_l_b)),
            0.90, seed=RANDOM_SEED,
        )
        c4b = run_ood_ctrgcn(
            "C4b EC3D+UI-PRMD→Custom OOD", c4_tr_b, c4_va_b,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p("ctrgcn_ood_c4_ec3d_uiprmd_to_custom.pth"),
        )
        summary_ctrgcn.append(("C", "C4b EC3D+UI-PRMD→Custom OOD",
                               _row("C4b EC3D+UI-PRMD→Custom OOD", c4b)))

    print("\n\n" + "=" * 70)
    print("C5b — Train REHAB24+UI-PRMD  →  test Custom OOD")
    print("=" * 70)
    if check_files(re_full_d, re_full_l, uiprmd_d_b, uiprmd_l_b):
        c5_tr_b, c5_va_b = split_frac(
            concat_ds((re_full_d, re_full_l), (uiprmd_d_b, uiprmd_l_b)),
            0.90, seed=RANDOM_SEED,
        )
        c5b = run_ood_ctrgcn(
            "C5b REHAB24+UI-PRMD→Custom OOD", c5_tr_b, c5_va_b,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p("ctrgcn_ood_c5_rehab24_uiprmd_to_custom.pth"),
        )
        summary_ctrgcn.append(("C", "C5b REHAB24+UI-PRMD→Custom OOD",
                               _row("C5b REHAB24+UI-PRMD→Custom OOD", c5b)))

    print("\n\n" + "=" * 70)
    print("C6b — Train EC3D+REHAB24+Custom  →  test UI-PRMD OOD")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, custom_d, custom_l):
        c6_tr_b, c6_va_b = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (re_full_d, re_full_l), (custom_d, custom_l)),
            0.90, seed=RANDOM_SEED,
        )
        c6b = run_ood_ctrgcn(
            "C6b EC3D+REHAB24+Custom→UI-PRMD OOD", c6_tr_b, c6_va_b,
            p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
            p("ctrgcn_ood_c6_ec3d_rehab24_custom_to_uiprmd.pth"),
        )
        summary_ctrgcn.append(("C", "C6b EC3D+REHAB24+Custom→UI-PRMD OOD",
                               _row("C6b EC3D+REHAB24+Custom→UI-PRMD OOD", c6b)))

    print("\n\n" + "=" * 70)
    print("C8b — Train EC3D+REHAB24+UI-PRMD  →  test Custom OOD")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, uiprmd_d_b, uiprmd_l_b):
        c8_tr_b, c8_va_b = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (re_full_d, re_full_l), (uiprmd_d_b, uiprmd_l_b)),
            0.90, seed=RANDOM_SEED,
        )
        c8b = run_ood_ctrgcn(
            "C8b EC3D+REHAB24+UI-PRMD→Custom OOD", c8_tr_b, c8_va_b,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p("ctrgcn_ood_c8_ec3d_rehab24_uiprmd_to_custom.pth"),
        )
        summary_ctrgcn.append(("C", "C8b EC3D+REHAB24+UI-PRMD→Custom OOD",
                               _row("C8b EC3D+REHAB24+UI-PRMD→Custom OOD", c8b)))

    print("\n\n" + "=" * 70)
    print("C9b — Train EC3D+Custom+UI-PRMD  →  test REHAB24 OOD")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l, custom_d, custom_l, uiprmd_d_b, uiprmd_l_b):
        c9_tr_b, c9_va_b = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (custom_d, custom_l), (uiprmd_d_b, uiprmd_l_b)),
            0.90, seed=RANDOM_SEED,
        )
        c9b = run_ood_ctrgcn(
            "C9b EC3D+Custom+UI-PRMD→REHAB24 OOD", c9_tr_b, c9_va_b,
            p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
            p("ctrgcn_ood_c9_ec3d_custom_uiprmd_to_rehab24.pth"),
        )
        summary_ctrgcn.append(("C", "C9b EC3D+Custom+UI-PRMD→REHAB24 OOD",
                               _row("C9b EC3D+Custom+UI-PRMD→REHAB24 OOD", c9b)))

    print("\n\n" + "=" * 70)
    print("C10b — Train REHAB24+Custom+UI-PRMD  →  test EC3D OOD")
    print("=" * 70)
    if check_files(re_full_d, re_full_l, custom_d, custom_l, uiprmd_d_b, uiprmd_l_b):
        c10_tr_b, c10_va_b = split_frac(
            concat_ds((re_full_d, re_full_l), (custom_d, custom_l), (uiprmd_d_b, uiprmd_l_b)),
            0.90, seed=RANDOM_SEED,
        )
        c10b = run_ood_ctrgcn(
            "C10b REHAB24+Custom+UI-PRMD→EC3D OOD", c10_tr_b, c10_va_b,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p("ctrgcn_ood_c10_rehab24_custom_uiprmd_to_ec3d.pth"),
        )
        summary_ctrgcn.append(("C", "C10b REHAB24+Custom+UI-PRMD→EC3D OOD",
                               _row("C10b REHAB24+Custom+UI-PRMD→EC3D OOD", c10b)))

    print("\n\n" + "=" * 70)
    print("C7b — Train Custom+UI-PRMD  →  test EC3D OOD")
    print("=" * 70)
    if check_files(custom_d, custom_l, uiprmd_d_b, uiprmd_l_b):
        c7_tr_b, c7_va_b = split_frac(
            concat_ds((custom_d, custom_l), (uiprmd_d_b, uiprmd_l_b)),
            0.90, seed=RANDOM_SEED,
        )
        c7b = run_ood_ctrgcn(
            "C7b Custom+UI-PRMD→EC3D OOD", c7_tr_b, c7_va_b,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p("ctrgcn_ood_c7_custom_uiprmd_to_ec3d.pth"),
        )
        summary_ctrgcn.append(("C", "C7b Custom+UI-PRMD→EC3D OOD",
                               _row("C7b Custom+UI-PRMD→EC3D OOD", c7b)))

    print("\n\n" + "=" * 70)
    print("D1b — Mixed: all 3 datasets  80% train / 10% val / 10% test")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, custom_d, custom_l,
                   uiprmd_d_b, uiprmd_l_b):
        all_ds_b = concat_ds(
            (ec3d_full_d, ec3d_full_l),
            (re_full_d,   re_full_l),
            (custom_d,    custom_l),
            (uiprmd_d_b,  uiprmd_l_b),
        )
        d_tr_b, d_va_b, d_te_b = split_80_10_10(all_ds_b, seed=RANDOM_SEED)
        print(
            f"  Combined: {len(all_ds_b)} reps  →  "
            f"train={len(d_tr_b)}  val={len(d_va_b)}  test={len(d_te_b)}"
        )
        model_d_b = train_model_ctrgcn(
            d_tr_b, d_va_b, p("ctrgcn_mixed_all_datasets.pth"), "D1b Mixed",
        )
        d1b = evaluate(model_d_b, DataLoader(d_te_b, batch_size=TRAIN_BATCH_SIZE))
        print(
            f"  Mixed test: "
            f"acc={d1b['acc']:.3f}  prec={d1b['prec']:.3f}  "
            f"rec={d1b['rec']:.3f}  F1={d1b['f1']:.3f}  n={d1b['n']}"
        )
        summary_ctrgcn.append(("D", "D1b Mixed all 80/10/10", _row("D1b Mixed all 80/10/10", d1b)))

    print("\n\n" + "=" * 70)
    print("CTR-GCN FINAL SUMMARY")
    print("=" * 70)
    print(HDR)

    last_sec = None
    for sec, _tag, row_str in summary_ctrgcn:
        if sec != last_sec:
            print(SEP)
            print(f"  [{section_labels.get(sec, sec)}]")
            last_sec = sec
        print(row_str)
    print(SEP)

    print(f"\nCTR-GCN checkpoints saved in {OUTPUT_DIR}/:")
    for name in [
        "ctrgcn_loso_ec3d_round{{1..4}}.pth",
        "ctrgcn_loso_rehab24_round{{1..9}}.pth",
        "ctrgcn_loso_uiprmd_round{{1..10}}.pth",
        "ctrgcn_kfold_custom_fold{{1..5}}.pth",
        "ctrgcn_ood_b{{1..12}}_*.pth",
        "ctrgcn_ood_c{{1..10}}_*.pth",
        "ctrgcn_mixed_all_datasets.pth",
    ]:
        print(f"  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ood-only",
        action="store_true",
        help="Skip IID LOSO/k-fold; run B1-B12, C1-C10, and D1 only",
    )
    parser.add_argument(
        "--mixed-only",
        action="store_true",
        help="Run D1 (ST-GCN) and D1b (CTR-GCN) mixed only; write mixed_results_*.txt",
    )
    parser.add_argument(
        "--iid-only",
        action="store_true",
        help="Run IID LOSO/k-fold (A1-A4) for ST-GCN and CTR-GCN only; write iid_results_*.txt",
    )
    args = parser.parse_args()
    main(ood_only=args.ood_only, mixed_only=args.mixed_only, iid_only=args.iid_only)
