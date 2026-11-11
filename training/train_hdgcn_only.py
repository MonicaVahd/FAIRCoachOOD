"""
training/train_hdgcn_only.py — Comprehensive HD-GCN evaluation across all conditions.

Experiments (HD-GCN only)
-------------------------
A1e EC3D LOSO (4 rounds)            — mean ± std across rounds
A2e REHAB24 LOSO (9 rounds)         — mean ± std across rounds
A4e UI-PRMD LOSO (10 rounds)        — mean ± std across rounds
A3e Custom 5-Fold CV (5 folds)      — mean ± std across folds

B1e–B12e OOD single-source train
C1e–C10e OOD dual-source train
D1e      Mixed — all 4 datasets 80% train / 10% val / 10% test

HD-GCN forward() takes a single input x; loss is weighted CrossEntropyLoss only.
Checkpoints saved with hdgcn_ prefix in output/.

Usage
-----
    cd <project_root>
    python training/train_hdgcn_only.py

Prerequisites: same as train_stgcn_full.py (data prep + split scripts).
"""

import argparse
import os
import random
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_utils import ConfmatLogger, compute_metrics, confmat_log_path, iid_results_path, mixed_results_path
from models.builders import build_hdgcn_model
from configs.constants import (
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_WEIGHT_DECAY,
    OUTPUT_DIR, TRAIN_SEED as RANDOM_SEED,
)

POS_CLASS = 1
EARLY_STOP_PAT = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SLUG = "hdgcn"
CONFMAT_CSV = confmat_log_path(OUTPUT_DIR, MODEL_SLUG)
_confmat_logger = None
SCRIPT_NAME = "train_hdgcn_only.py"
_MODEL_LOG_NAME = "HD-GCN"
CKPT_PREFIX = "hdgcn_"


def p(name: str) -> str:
    return os.path.join(OUTPUT_DIR, name)


def load_ds(data_path: str, label_path: str) -> TensorDataset:
    X = torch.from_numpy(np.load(data_path)).float()
    y = torch.from_numpy(np.load(label_path)).long()
    return TensorDataset(X, y)


def concat_ds(*pairs: Tuple[str, str]) -> TensorDataset:
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
    N = len(ds)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_tr = int(train_frac * N)
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
    N = len(ds)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n80 = int(0.80 * N)
    n10 = int(0.10 * N)
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


def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """HD-GCN forward() returns logits directly."""
    return model(x)



def evaluate(model: nn.Module, loader: DataLoader, tag: str = "") -> dict:
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(DEVICE)
            preds_all.append(
                forward_logits(model, X_b).argmax(dim=1).cpu().numpy()
            )
            labels_all.append(y_b.numpy())
    m = compute_metrics(np.concatenate(preds_all), np.concatenate(labels_all))
    if _confmat_logger is not None and tag:
        _confmat_logger.append(SCRIPT_NAME, _MODEL_LOG_NAME, tag, m)
    return m


def train_model(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    checkpoint_path: str,
    label: str,
) -> nn.Module:
    """Train HD-GCN with weighted CE and early stopping."""
    labels_np = train_ds.tensors[1].numpy()
    n0 = int((labels_np == 0).sum())
    n1 = int((labels_np == 1).sum())
    N = len(labels_np)
    w = torch.tensor(
        [N / (2 * max(n0, 1)), N / (2 * max(n1, 1))]
    ).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    print(f"  Class weights: incorrect={w[0]:.3f}  correct={w[1]:.3f}")

    model = build_hdgcn_model(checkpoint_path=None, device=DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    train_loader = DataLoader(
        train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=TRAIN_BATCH_SIZE)

    best_val_loss = float("inf")
    no_improve = 0
    best_epoch = 0

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
            best_epoch = epoch
            no_improve = 0
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
        f"-> {checkpoint_path}"
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def mean_std(metrics_list: List[dict], key: str) -> Tuple[float, float]:
    vals = [m[key] for m in metrics_list]
    return float(np.mean(vals)), float(np.std(vals))


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

SECTION_LABELS = {
    "A": "IID (within-distribution)",
    "B": "OOD - single source train",
    "C": "OOD - dual source train",
    "D": "Mixed (all datasets)",
}


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


def format_summary(summary: List[Tuple[str, str, str]]) -> List[str]:
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("HD-GCN FINAL SUMMARY")
    lines.append("=" * 70)
    lines.append(HDR)
    last_sec = None
    for sec, _tag, row_str in summary:
        if sec != last_sec:
            lines.append(SEP)
            lines.append(f"  [{SECTION_LABELS.get(sec, sec)}]")
            last_sec = sec
        lines.append(row_str)
    lines.append(SEP)
    return lines


def save_summary(summary: List[Tuple[str, str, str]], path: str) -> None:
    lines = format_summary(summary)
    text = "\n".join(lines) + "\n"
    print("\n".join(lines))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved -> {path}")


def main(mixed_only: bool = False, iid_only: bool = False):
    global _confmat_logger
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if mixed_only:
        _confmat_logger = None
    else:
        _confmat_logger = ConfmatLogger(CONFMAT_CSV)
        _confmat_logger.ensure_header()
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
    if iid_only:
        print("Mode          : --iid-only (IID A1-A4 only; skipping OOD and mixed)")
    if mixed_only:
        print("Mode          : --mixed-only (D1e Mixed only; skipping IID and OOD)")
    if mixed_only and iid_only:
        raise SystemExit("Cannot combine --iid-only with --mixed-only.")

    summary: List[Tuple[str, str, str]] = []

    if mixed_only:
        ec3d_full_d = p("ec3d_train_data.npy")
        ec3d_full_l = p("ec3d_train_labels.npy")
        re_full_d   = p("rehab24_train_data.npy")
        re_full_l   = p("rehab24_train_labels.npy")
        custom_d    = p("custom_squat_train_data.npy")
        custom_l    = p("custom_squat_train_labels.npy")
        uiprmd_d    = p("uiprmd_data.npy")
        uiprmd_l    = p("uiprmd_labels.npy")

        print("\n\n" + "=" * 70)
        print("D1e — Mixed: all 4 datasets  80% train / 10% val / 10% test")
        print("=" * 70)
        if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, custom_d, custom_l,
                       uiprmd_d, uiprmd_l):
            all_ds = concat_ds(
                (ec3d_full_d, ec3d_full_l),
                (re_full_d, re_full_l),
                (custom_d, custom_l),
                (uiprmd_d, uiprmd_l),
            )
            d_tr, d_va, d_te = split_80_10_10(all_ds, seed=RANDOM_SEED)
            print(
                f"  Combined: {len(all_ds)} reps  ->  "
                f"train={len(d_tr)}  val={len(d_va)}  test={len(d_te)}"
            )
            model_d = train_model(
                d_tr, d_va, p(f"{CKPT_PREFIX}mixed_all_datasets.pth"), "D1e Mixed",
            )
            d1 = evaluate(model_d, DataLoader(d_te, batch_size=TRAIN_BATCH_SIZE), tag="")
            print(
                f"  Mixed test: "
                f"acc={d1['acc']:.3f}  prec={d1['prec']:.3f}  "
                f"rec={d1['rec']:.3f}  F1={d1['f1']:.3f}  "
                f"MacroF1={d1['macro_f1']:.3f}  n={d1['n']}"
            )
            summary.append(("D", "D1e Mixed all 80/10/10", _row("D1e Mixed all 80/10/10", d1)))

        save_summary(summary, mixed_results_path(OUTPUT_DIR, MODEL_SLUG))
        print("\nMode: --mixed-only — skipping IID and OOD experiments.")
        print(f"\nCheckpoints saved in {OUTPUT_DIR}/:")
        print(f"  {CKPT_PREFIX}mixed_all_datasets.pth")
        return

    # ── A — IID ───────────────────────────────────────────────────────────────

    print("\n\n" + "=" * 70)
    print("A1e — EC3D LOSO  (4 rounds)")
    print("=" * 70)
    a1 = run_loso("ec3d", 4, f"{CKPT_PREFIX}loso_ec3d")
    if a1:
        summary.append(("A", "A1e EC3D LOSO (4 rounds)",
                        _row_ms("A1e EC3D LOSO (4 rounds)", a1)))

    print("\n\n" + "=" * 70)
    print("A2e — REHAB24 LOSO  (9 rounds)")
    print("=" * 70)
    a2 = run_loso("rehab24", 9, f"{CKPT_PREFIX}loso_rehab24")
    if a2:
        summary.append(("A", "A2e REHAB24 LOSO (9 rounds)",
                        _row_ms("A2e REHAB24 LOSO (9 rounds)", a2)))

    print("\n\n" + "=" * 70)
    print("A4e — UI-PRMD LOSO  (10 rounds)")
    print("=" * 70)
    a4 = run_loso("uiprmd", 10, f"{CKPT_PREFIX}loso_uiprmd")
    if a4:
        summary.append(("A", "A4e UI-PRMD LOSO (10 rounds)",
                        _row_ms("A4e UI-PRMD LOSO (10 rounds)", a4)))

    print("\n\n" + "=" * 70)
    print("A3e — Custom 5-Fold CV")
    print("=" * 70)
    a3 = run_kfold(5, f"{CKPT_PREFIX}kfold_custom")
    if a3:
        summary.append(("A", "A3e Custom 5-Fold CV",
                        _row_ms("A3e Custom 5-Fold CV", a3)))

    if iid_only:
        save_summary(summary, iid_results_path(OUTPUT_DIR, MODEL_SLUG))
        print("\nMode: --iid-only — skipping OOD and mixed experiments.")
        return

    # ── B — OOD single source ─────────────────────────────────────────────────

    ec3d_iid_tr_d = p("ec3d_iid_train_data.npy")
    ec3d_iid_tr_l = p("ec3d_iid_train_labels.npy")
    ec3d_iid_te_d = p("ec3d_iid_test_data.npy")
    ec3d_iid_te_l = p("ec3d_iid_test_labels.npy")
    re_iid_tr_d = p("rehab24_iid_train_data.npy")
    re_iid_tr_l = p("rehab24_iid_train_labels.npy")
    re_iid_te_d = p("rehab24_iid_test_data.npy")
    re_iid_te_l = p("rehab24_iid_test_labels.npy")
    custom_d = p("custom_squat_train_data.npy")
    custom_l = p("custom_squat_train_labels.npy")

    print("\n\n" + "=" * 70)
    print("B1e — Train EC3D 80%  ->  test REHAB24 OOD")
    print("=" * 70)
    if check_files(ec3d_iid_tr_d, ec3d_iid_tr_l, ec3d_iid_te_d, ec3d_iid_te_l):
        ec3d_tr = load_ds(ec3d_iid_tr_d, ec3d_iid_tr_l)
        ec3d_va = load_ds(ec3d_iid_te_d, ec3d_iid_te_l)
        b1 = run_ood(
            "B1e EC3D->REHAB24 OOD", ec3d_tr, ec3d_va,
            p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_b1_ec3d_to_rehab24.pth"),
        )
        summary.append(("B", "B1e EC3D->REHAB24 OOD", _row("B1e EC3D->REHAB24 OOD", b1)))

        print("\n\n" + "=" * 70)
        print("B2e — Train EC3D 80%  ->  test Custom OOD")
        print("=" * 70)
        b2 = run_ood(
            "B2e EC3D->Custom OOD", ec3d_tr, ec3d_va,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_b2_ec3d_to_custom.pth"),
        )
        summary.append(("B", "B2e EC3D->Custom OOD", _row("B2e EC3D->Custom OOD", b2)))

    print("\n\n" + "=" * 70)
    print("B3e — Train REHAB24 80%  ->  test EC3D OOD")
    print("=" * 70)
    if check_files(re_iid_tr_d, re_iid_tr_l, re_iid_te_d, re_iid_te_l):
        re_tr = load_ds(re_iid_tr_d, re_iid_tr_l)
        re_va = load_ds(re_iid_te_d, re_iid_te_l)
        b3 = run_ood(
            "B3e REHAB24->EC3D OOD", re_tr, re_va,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_b3_rehab24_to_ec3d.pth"),
        )
        summary.append(("B", "B3e REHAB24->EC3D OOD", _row("B3e REHAB24->EC3D OOD", b3)))

        print("\n\n" + "=" * 70)
        print("B4e — Train REHAB24 80%  ->  test Custom OOD")
        print("=" * 70)
        b4 = run_ood(
            "B4e REHAB24->Custom OOD", re_tr, re_va,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_b4_rehab24_to_custom.pth"),
        )
        summary.append(("B", "B4e REHAB24->Custom OOD", _row("B4e REHAB24->Custom OOD", b4)))

    print("\n\n" + "=" * 70)
    print("B5e — Train Custom 80%  ->  test EC3D OOD")
    print("=" * 70)
    if check_files(custom_d, custom_l):
        c_tr, c_va = split_frac(concat_ds((custom_d, custom_l)), 0.80, seed=RANDOM_SEED)
        b5 = run_ood(
            "B5e Custom->EC3D OOD", c_tr, c_va,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_b5_custom_to_ec3d.pth"),
        )
        summary.append(("B", "B5e Custom->EC3D OOD", _row("B5e Custom->EC3D OOD", b5)))

        print("\n\n" + "=" * 70)
        print("B6e — Train Custom 80%  ->  test REHAB24 OOD")
        print("=" * 70)
        b6 = run_ood(
            "B6e Custom->REHAB24 OOD", c_tr, c_va,
            p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_b6_custom_to_rehab24.pth"),
        )
        summary.append(("B", "B6e Custom->REHAB24 OOD", _row("B6e Custom->REHAB24 OOD", b6)))

    # ── C — OOD dual source ───────────────────────────────────────────────────

    ec3d_full_d = p("ec3d_train_data.npy")
    ec3d_full_l = p("ec3d_train_labels.npy")
    re_full_d = p("rehab24_train_data.npy")
    re_full_l = p("rehab24_train_labels.npy")

    print("\n\n" + "=" * 70)
    print("C1e — Train EC3D+REHAB24  ->  test Custom OOD")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l):
        c1_tr, c1_va = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (re_full_d, re_full_l)),
            0.90, seed=RANDOM_SEED,
        )
        c1 = run_ood(
            "C1e EC3D+REHAB24->Custom OOD", c1_tr, c1_va,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_c1_ec3d_rehab24_to_custom.pth"),
        )
        summary.append(("C", "C1e EC3D+REHAB24->Custom OOD",
                        _row("C1e EC3D+REHAB24->Custom OOD", c1)))

    print("\n\n" + "=" * 70)
    print("C2e — Train EC3D+Custom  ->  test REHAB24 OOD")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l, custom_d, custom_l):
        c2_tr, c2_va = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (custom_d, custom_l)),
            0.90, seed=RANDOM_SEED,
        )
        c2 = run_ood(
            "C2e EC3D+Custom->REHAB24 OOD", c2_tr, c2_va,
            p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_c2_ec3d_custom_to_rehab24.pth"),
        )
        summary.append(("C", "C2e EC3D+Custom->REHAB24 OOD",
                        _row("C2e EC3D+Custom->REHAB24 OOD", c2)))

    print("\n\n" + "=" * 70)
    print("C3e — Train REHAB24+Custom  ->  test EC3D OOD")
    print("=" * 70)
    if check_files(re_full_d, re_full_l, custom_d, custom_l):
        c3_tr, c3_va = split_frac(
            concat_ds((re_full_d, re_full_l), (custom_d, custom_l)),
            0.90, seed=RANDOM_SEED,
        )
        c3 = run_ood(
            "C3e REHAB24+Custom->EC3D OOD", c3_tr, c3_va,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_c3_rehab24_custom_to_ec3d.pth"),
        )
        summary.append(("C", "C3e REHAB24+Custom->EC3D OOD",
                        _row("C3e REHAB24+Custom->EC3D OOD", c3)))

    uiprmd_d = p("uiprmd_data.npy")
    uiprmd_l = p("uiprmd_labels.npy")

    print("\n\n" + "=" * 70)
    print("B7e — Train UI-PRMD 80%  ->  test EC3D OOD")
    print("=" * 70)
    if check_files(uiprmd_d, uiprmd_l):
        ui_tr, ui_va = split_frac(concat_ds((uiprmd_d, uiprmd_l)), 0.80, seed=RANDOM_SEED)
        b7 = run_ood(
            "B7e UI-PRMD->EC3D OOD", ui_tr, ui_va,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_b7_uiprmd_to_ec3d.pth"),
        )
        summary.append(("B", "B7e UI-PRMD->EC3D OOD", _row("B7e UI-PRMD->EC3D OOD", b7)))

        print("\n\n" + "=" * 70)
        print("B8e — Train UI-PRMD 80%  ->  test REHAB24 OOD")
        print("=" * 70)
        b8 = run_ood(
            "B8e UI-PRMD->REHAB24 OOD", ui_tr, ui_va,
            p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_b8_uiprmd_to_rehab24.pth"),
        )
        summary.append(("B", "B8e UI-PRMD->REHAB24 OOD", _row("B8e UI-PRMD->REHAB24 OOD", b8)))

        print("\n\n" + "=" * 70)
        print("B9e — Train UI-PRMD 80%  ->  test Custom OOD")
        print("=" * 70)
        b9 = run_ood(
            "B9e UI-PRMD->Custom OOD", ui_tr, ui_va,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_b9_uiprmd_to_custom.pth"),
        )
        summary.append(("B", "B9e UI-PRMD->Custom OOD", _row("B9e UI-PRMD->Custom OOD", b9)))

    print("\n\n" + "=" * 70)
    print("B10e — Train EC3D 80%  ->  test UI-PRMD OOD")
    print("=" * 70)
    if check_files(ec3d_iid_tr_d, ec3d_iid_tr_l, ec3d_iid_te_d, ec3d_iid_te_l):
        b10 = run_ood(
            "B10e EC3D->UI-PRMD OOD",
            load_ds(ec3d_iid_tr_d, ec3d_iid_tr_l), load_ds(ec3d_iid_te_d, ec3d_iid_te_l),
            p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_b10_ec3d_to_uiprmd.pth"),
        )
        summary.append(("B", "B10e EC3D->UI-PRMD OOD", _row("B10e EC3D->UI-PRMD OOD", b10)))

    print("\n\n" + "=" * 70)
    print("B11e — Train REHAB24 80%  ->  test UI-PRMD OOD")
    print("=" * 70)
    if check_files(re_iid_tr_d, re_iid_tr_l, re_iid_te_d, re_iid_te_l):
        b11 = run_ood(
            "B11e REHAB24->UI-PRMD OOD",
            load_ds(re_iid_tr_d, re_iid_tr_l), load_ds(re_iid_te_d, re_iid_te_l),
            p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_b11_rehab24_to_uiprmd.pth"),
        )
        summary.append(("B", "B11e REHAB24->UI-PRMD OOD", _row("B11e REHAB24->UI-PRMD OOD", b11)))

    print("\n\n" + "=" * 70)
    print("B12e — Train Custom 80%  ->  test UI-PRMD OOD")
    print("=" * 70)
    if check_files(custom_d, custom_l):
        c_tr_b12, c_va_b12 = split_frac(concat_ds((custom_d, custom_l)), 0.80, seed=RANDOM_SEED)
        b12 = run_ood(
            "B12e Custom->UI-PRMD OOD", c_tr_b12, c_va_b12,
            p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_b12_custom_to_uiprmd.pth"),
        )
        summary.append(("B", "B12e Custom->UI-PRMD OOD", _row("B12e Custom->UI-PRMD OOD", b12)))

    print("\n\n" + "=" * 70)
    print("C4e — Train EC3D+UI-PRMD  ->  test Custom OOD")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l, uiprmd_d, uiprmd_l):
        c4_tr, c4_va = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (uiprmd_d, uiprmd_l)),
            0.90, seed=RANDOM_SEED,
        )
        c4 = run_ood(
            "C4e EC3D+UI-PRMD->Custom OOD", c4_tr, c4_va,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_c4_ec3d_uiprmd_to_custom.pth"),
        )
        summary.append(("C", "C4e EC3D+UI-PRMD->Custom OOD",
                        _row("C4e EC3D+UI-PRMD->Custom OOD", c4)))

    print("\n\n" + "=" * 70)
    print("C5e — Train REHAB24+UI-PRMD  ->  test Custom OOD")
    print("=" * 70)
    if check_files(re_full_d, re_full_l, uiprmd_d, uiprmd_l):
        c5_tr, c5_va = split_frac(
            concat_ds((re_full_d, re_full_l), (uiprmd_d, uiprmd_l)),
            0.90, seed=RANDOM_SEED,
        )
        c5 = run_ood(
            "C5e REHAB24+UI-PRMD->Custom OOD", c5_tr, c5_va,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_c5_rehab24_uiprmd_to_custom.pth"),
        )
        summary.append(("C", "C5e REHAB24+UI-PRMD->Custom OOD",
                        _row("C5e REHAB24+UI-PRMD->Custom OOD", c5)))

    print("\n\n" + "=" * 70)
    print("C6e — Train EC3D+REHAB24+Custom  ->  test UI-PRMD OOD")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, custom_d, custom_l):
        c6_tr, c6_va = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (re_full_d, re_full_l), (custom_d, custom_l)),
            0.90, seed=RANDOM_SEED,
        )
        c6 = run_ood(
            "C6e EC3D+REHAB24+Custom->UI-PRMD OOD", c6_tr, c6_va,
            p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_c6_ec3d_rehab24_custom_to_uiprmd.pth"),
        )
        summary.append(("C", "C6e EC3D+REHAB24+Custom->UI-PRMD OOD",
                        _row("C6e EC3D+REHAB24+Custom->UI-PRMD OOD", c6)))

    print("\n\n" + "=" * 70)
    print("C8e — Train EC3D+REHAB24+UI-PRMD  ->  test Custom OOD")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, uiprmd_d, uiprmd_l):
        c8_tr, c8_va = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (re_full_d, re_full_l), (uiprmd_d, uiprmd_l)),
            0.90, seed=RANDOM_SEED,
        )
        c8 = run_ood(
            "C8e EC3D+REHAB24+UI-PRMD->Custom OOD", c8_tr, c8_va,
            p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_c8_ec3d_rehab24_uiprmd_to_custom.pth"),
        )
        summary.append(("C", "C8e EC3D+REHAB24+UI-PRMD->Custom OOD",
                        _row("C8e EC3D+REHAB24+UI-PRMD->Custom OOD", c8)))

    print("\n\n" + "=" * 70)
    print("C9e — Train EC3D+Custom+UI-PRMD  ->  test REHAB24 OOD")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l, custom_d, custom_l, uiprmd_d, uiprmd_l):
        c9_tr, c9_va = split_frac(
            concat_ds((ec3d_full_d, ec3d_full_l), (custom_d, custom_l), (uiprmd_d, uiprmd_l)),
            0.90, seed=RANDOM_SEED,
        )
        c9 = run_ood(
            "C9e EC3D+Custom+UI-PRMD->REHAB24 OOD", c9_tr, c9_va,
            p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_c9_ec3d_custom_uiprmd_to_rehab24.pth"),
        )
        summary.append(("C", "C9e EC3D+Custom+UI-PRMD->REHAB24 OOD",
                        _row("C9e EC3D+Custom+UI-PRMD->REHAB24 OOD", c9)))

    print("\n\n" + "=" * 70)
    print("C10e — Train REHAB24+Custom+UI-PRMD  ->  test EC3D OOD")
    print("=" * 70)
    if check_files(re_full_d, re_full_l, custom_d, custom_l, uiprmd_d, uiprmd_l):
        c10_tr, c10_va = split_frac(
            concat_ds((re_full_d, re_full_l), (custom_d, custom_l), (uiprmd_d, uiprmd_l)),
            0.90, seed=RANDOM_SEED,
        )
        c10 = run_ood(
            "C10e REHAB24+Custom+UI-PRMD->EC3D OOD", c10_tr, c10_va,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_c10_rehab24_custom_uiprmd_to_ec3d.pth"),
        )
        summary.append(("C", "C10e REHAB24+Custom+UI-PRMD->EC3D OOD",
                        _row("C10e REHAB24+Custom+UI-PRMD->EC3D OOD", c10)))

    print("\n\n" + "=" * 70)
    print("C7e — Train Custom+UI-PRMD  ->  test EC3D OOD")
    print("=" * 70)
    if check_files(custom_d, custom_l, uiprmd_d, uiprmd_l):
        c7_tr, c7_va = split_frac(
            concat_ds((custom_d, custom_l), (uiprmd_d, uiprmd_l)),
            0.90, seed=RANDOM_SEED,
        )
        c7 = run_ood(
            "C7e Custom+UI-PRMD->EC3D OOD", c7_tr, c7_va,
            p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
            p(f"{CKPT_PREFIX}ood_c7_custom_uiprmd_to_ec3d.pth"),
        )
        summary.append(("C", "C7e Custom+UI-PRMD->EC3D OOD",
                        _row("C7e Custom+UI-PRMD->EC3D OOD", c7)))

    # ── D — Mixed (all 4 datasets) ───────────────────────────────────────────

    print("\n\n" + "=" * 70)
    print("D1e — Mixed: all 4 datasets  80% train / 10% val / 10% test")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, custom_d, custom_l,
                   uiprmd_d, uiprmd_l):
        all_ds = concat_ds(
            (ec3d_full_d, ec3d_full_l),
            (re_full_d, re_full_l),
            (custom_d, custom_l),
            (uiprmd_d, uiprmd_l),
        )
        d_tr, d_va, d_te = split_80_10_10(all_ds, seed=RANDOM_SEED)
        print(
            f"  Combined: {len(all_ds)} reps  ->  "
            f"train={len(d_tr)}  val={len(d_va)}  test={len(d_te)}"
        )
        model_d = train_model(
            d_tr, d_va, p(f"{CKPT_PREFIX}mixed_all_datasets.pth"), "D1e Mixed",
        )
        d1 = evaluate(model_d, DataLoader(d_te, batch_size=TRAIN_BATCH_SIZE), tag="D1 Mixed all 80/10/10")
        print(
            f"  Mixed test: "
            f"acc={d1['acc']:.3f}  prec={d1['prec']:.3f}  "
            f"rec={d1['rec']:.3f}  F1={d1['f1']:.3f}  n={d1['n']}"
        )
        summary.append(("D", "D1e Mixed all 80/10/10", _row("D1e Mixed all 80/10/10", d1)))

    save_summary(summary, p("hdgcn_results.txt"))

    print(f"\nCheckpoints saved in {OUTPUT_DIR}/:")
    for name in [
        f"{CKPT_PREFIX}loso_ec3d_round{{1..4}}.pth",
        f"{CKPT_PREFIX}loso_rehab24_round{{1..9}}.pth",
        f"{CKPT_PREFIX}loso_uiprmd_round{{1..10}}.pth",
        f"{CKPT_PREFIX}kfold_custom_fold{{1..5}}.pth",
        f"{CKPT_PREFIX}ood_b{{1..12}}_*.pth",
        f"{CKPT_PREFIX}ood_c{{1..10}}_*.pth",
        f"{CKPT_PREFIX}mixed_all_datasets.pth",
    ]:
        print(f"  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mixed-only",
        action="store_true",
        help="Run D1 Mixed (80/10/10) only; write to mixed_results_<model>.txt",
    )
    parser.add_argument(
        "--iid-only",
        action="store_true",
        help="Run IID LOSO/k-fold (A1-A4) only; write to iid_results_<model>.txt",
    )
    args = parser.parse_args()
    main(mixed_only=args.mixed_only, iid_only=args.iid_only)
