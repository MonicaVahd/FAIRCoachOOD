"""
training/train_freqmixformer_full.py — Comprehensive FreqMixFormer evaluation.

Matches train_hyperformer_full.py experiment flow (A/B/C/D), using
FreqMixFormer as the model backend.

Experiments
-----------
A1  EC3D LOSO (4 rounds)
A2  REHAB24 LOSO (9 rounds)
A4  UI-PRMD LOSO (10 rounds)
A3  Custom 5-Fold CV (5 folds)

B1–B12  OOD (single-source train)
C1–C10  OOD (dual/multi-source train)
D1      Mixed all datasets (80/10/10)
"""

import argparse
import os
import random
import subprocess
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_utils import ConfmatLogger, compute_metrics, confmat_log_path, iid_results_path, mixed_results_path, print_and_save_summary
from models.builders import build_freqmixformer_model
from configs.constants import (
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_WEIGHT_DECAY,
    OUTPUT_DIR, TRAIN_SEED as RANDOM_SEED,
)

POS_CLASS = 1
EARLY_STOP_PAT = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SLUG = "freqmixformer"
CONFMAT_CSV = confmat_log_path(OUTPUT_DIR, MODEL_SLUG)
_confmat_logger = None
SCRIPT_NAME = "train_freqmixformer_full.py"
_MODEL_LOG_NAME = "FreqMixFormer"
CKPT_PREFIX = "freqmixformer_"

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
    va_idx = torch.tensor(idx[n80: n80 + n10])
    te_idx = torch.tensor(idx[n80 + n10:])
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



def evaluate(model: nn.Module, loader: DataLoader, tag: str = "") -> dict:
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(DEVICE)
            preds_all.append(model(X_b).argmax(dim=1).cpu().numpy())
            labels_all.append(y_b.numpy())
    m = compute_metrics(np.concatenate(preds_all), np.concatenate(labels_all))
    if _confmat_logger is not None and tag:
        _confmat_logger.append(SCRIPT_NAME, _MODEL_LOG_NAME, tag, m)
    return m


def run_check_confusion() -> None:
    """Run training/check_confusion.py after each experiment block."""
    try:
        subprocess.run(
            [sys.executable, os.path.join("training", "check_confusion.py")],
            check=False,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
    except Exception as e:
        print(f"  [WARN] check_confusion.py call failed: {e}")


def train_model(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    checkpoint_path: str,
    label: str,
) -> nn.Module:
    labels_np = train_ds.tensors[1].numpy()
    n0 = int((labels_np == 0).sum())
    n1 = int((labels_np == 1).sum())
    N = len(labels_np)
    w = torch.tensor([N / (2 * max(n0, 1)), N / (2 * max(n1, 1))]).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)
    print(f"  Class weights: incorrect={w[0]:.3f}  correct={w[1]:.3f}")

    model = build_freqmixformer_model(checkpoint_path=None, device=DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
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
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PAT:
                print(
                    f"    Early stop @ epoch {epoch}  "
                    f"(best epoch={best_epoch}, v_loss={best_val_loss:.4f})"
                )
                break

    print(f"  Best v_loss={best_val_loss:.4f} @ epoch {best_epoch} -> {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def mean_std(metrics_list: List[dict], key: str) -> Tuple[float, float]:
    vals = [m[key] for m in metrics_list]
    return float(np.mean(vals)), float(np.std(vals))


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
            f"acc={m['acc']:.3f}  prec={m['prec']:.3f}  rec={m['rec']:.3f}  F1={m['f1']:.3f}  n={m['n']}"
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
            f"acc={m['acc']:.3f}  prec={m['prec']:.3f}  rec={m['rec']:.3f}  F1={m['f1']:.3f}  n={m['n']}"
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
        f"acc={m['acc']:.3f}  prec={m['prec']:.3f}  rec={m['rec']:.3f}  F1={m['f1']:.3f}  n={m['n']}"
    )
    model.cpu()
    del model
    torch.cuda.empty_cache()
    return m


def print_summary(summary: List[Tuple[str, str, str]], title: str = "FreqMixFormer FINAL SUMMARY") -> None:
    print("\n\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(HDR)
    last_sec = None
    for sec, _tag, row_str in summary:
        if sec != last_sec:
            print(SEP)
            print(f"  [{SECTION_LABELS.get(sec, sec)}]")
            last_sec = sec
        print(row_str)
    print(SEP)


def main(ood_only: bool = False, mixed_only: bool = False, iid_only: bool = False):
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
    if ood_only:
        print("Mode          : --ood-only (skipping IID LOSO / k-fold experiments)")
    if mixed_only:
        print("Mode          : --mixed-only (D1 FreqMixFormer only)")
    if iid_only:
        print("Mode          : --iid-only (IID A1-A4 only; skipping OOD and mixed)")
    if sum((ood_only, mixed_only, iid_only)) > 1:
        raise SystemExit("Use at most one of --ood-only, --mixed-only, --iid-only.")

    summary: List[Tuple[str, str, str]] = []

    ec3d_full_d = p("ec3d_train_data.npy");    ec3d_full_l = p("ec3d_train_labels.npy")
    re_full_d   = p("rehab24_train_data.npy"); re_full_l   = p("rehab24_train_labels.npy")
    custom_d    = p("custom_squat_train_data.npy"); custom_l = p("custom_squat_train_labels.npy")
    uiprmd_d    = p("uiprmd_data.npy"); uiprmd_l = p("uiprmd_labels.npy")

    if mixed_only:
        print("\n\n" + "=" * 70)
        print("D1i — Mixed: all 4 datasets  80% train / 10% val / 10% test")
        print("=" * 70)
        if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, custom_d, custom_l, uiprmd_d, uiprmd_l):
            all_ds = concat_ds(
                (ec3d_full_d, ec3d_full_l),
                (re_full_d,   re_full_l),
                (custom_d,    custom_l),
                (uiprmd_d,    uiprmd_l),
            )
            d_tr, d_va, d_te = split_80_10_10(all_ds, seed=RANDOM_SEED)
            model_d = train_model(d_tr, d_va, p(f"{CKPT_PREFIX}mixed_all_datasets.pth"), "D1i Mixed")
            d1 = evaluate(model_d, DataLoader(d_te, batch_size=TRAIN_BATCH_SIZE), tag="")
            summary.append(("D", "D1i Mixed all 80/10/10", _row("D1i Mixed all 80/10/10", d1)))
        print_and_save_summary(
            summary,
            "FreqMixFormer FINAL SUMMARY",
            HDR,
            SEP,
            SECTION_LABELS,
            save_path=mixed_results_path(OUTPUT_DIR, MODEL_SLUG),
        )
        print("\nMode: --mixed-only — skipping IID and OOD experiments.")
        return

    if not ood_only:
        print("\n\n" + "=" * 70)
        print("A1i — EC3D LOSO  (4 rounds)")
        print("=" * 70)
        a1 = run_loso("ec3d", 4, f"{CKPT_PREFIX}loso_ec3d")
        if a1:
            summary.append(("A", "A1i EC3D LOSO (4 rounds)", _row_ms("A1i EC3D LOSO (4 rounds)", a1)))
            run_check_confusion()

        print("\n\n" + "=" * 70)
        print("A2i — REHAB24 LOSO  (9 rounds)")
        print("=" * 70)
        a2 = run_loso("rehab24", 9, f"{CKPT_PREFIX}loso_rehab24")
        if a2:
            summary.append(("A", "A2i REHAB24 LOSO (9 rounds)", _row_ms("A2i REHAB24 LOSO (9 rounds)", a2)))
            run_check_confusion()

        print("\n\n" + "=" * 70)
        print("A4i — UI-PRMD LOSO  (10 rounds)")
        print("=" * 70)
        a4 = run_loso("uiprmd", 10, f"{CKPT_PREFIX}loso_uiprmd")
        if a4:
            summary.append(("A", "A4i UI-PRMD LOSO (10 rounds)", _row_ms("A4i UI-PRMD LOSO (10 rounds)", a4)))
            run_check_confusion()

        print("\n\n" + "=" * 70)
        print("A3i — Custom 5-Fold CV")
        print("=" * 70)
        a3 = run_kfold(5, f"{CKPT_PREFIX}kfold_custom")
        if a3:
            summary.append(("A", "A3i Custom 5-Fold CV", _row_ms("A3i Custom 5-Fold CV", a3)))
            run_check_confusion()

    if iid_only:
        print_and_save_summary(
            summary,
            "FreqMixFormer FINAL SUMMARY",
            HDR,
            SEP,
            SECTION_LABELS,
            save_path=iid_results_path(OUTPUT_DIR, MODEL_SLUG),
        )
        print("\nMode: --iid-only — skipping OOD and mixed experiments.")
        return

    if check_files(ec3d_full_d, ec3d_full_l):
        ec3d_tr, ec3d_va = split_frac(concat_ds((ec3d_full_d, ec3d_full_l)), 0.80, seed=RANDOM_SEED)
        b1 = run_ood("B1i EC3D->REHAB24 OOD", ec3d_tr, ec3d_va,
                     p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_b1_ec3d_to_rehab24.pth"))
        summary.append(("B", "B1i EC3D->REHAB24 OOD", _row("B1i EC3D->REHAB24 OOD", b1))); run_check_confusion()
        b2 = run_ood("B2i EC3D->Custom OOD", ec3d_tr, ec3d_va,
                     p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_b2_ec3d_to_custom.pth"))
        summary.append(("B", "B2i EC3D->Custom OOD", _row("B2i EC3D->Custom OOD", b2))); run_check_confusion()
        b10 = run_ood("B10i EC3D->UI-PRMD OOD", ec3d_tr, ec3d_va,
                      p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
                      p(f"{CKPT_PREFIX}ood_b10_ec3d_to_uiprmd.pth"))
        summary.append(("B", "B10i EC3D->UI-PRMD OOD", _row("B10i EC3D->UI-PRMD OOD", b10))); run_check_confusion()

    if check_files(re_full_d, re_full_l):
        re_tr, re_va = split_frac(concat_ds((re_full_d, re_full_l)), 0.80, seed=RANDOM_SEED)
        b3 = run_ood("B3i REHAB24->EC3D OOD", re_tr, re_va,
                     p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_b3_rehab24_to_ec3d.pth"))
        summary.append(("B", "B3i REHAB24->EC3D OOD", _row("B3i REHAB24->EC3D OOD", b3))); run_check_confusion()
        b4 = run_ood("B4i REHAB24->Custom OOD", re_tr, re_va,
                     p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_b4_rehab24_to_custom.pth"))
        summary.append(("B", "B4i REHAB24->Custom OOD", _row("B4i REHAB24->Custom OOD", b4))); run_check_confusion()
        b11 = run_ood("B11i REHAB24->UI-PRMD OOD", re_tr, re_va,
                      p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
                      p(f"{CKPT_PREFIX}ood_b11_rehab24_to_uiprmd.pth"))
        summary.append(("B", "B11i REHAB24->UI-PRMD OOD", _row("B11i REHAB24->UI-PRMD OOD", b11))); run_check_confusion()

    if check_files(custom_d, custom_l):
        c_tr, c_va = split_frac(concat_ds((custom_d, custom_l)), 0.80, seed=RANDOM_SEED)
        b5 = run_ood("B5i Custom->EC3D OOD", c_tr, c_va,
                     p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_b5_custom_to_ec3d.pth"))
        summary.append(("B", "B5i Custom->EC3D OOD", _row("B5i Custom->EC3D OOD", b5))); run_check_confusion()
        b6 = run_ood("B6i Custom->REHAB24 OOD", c_tr, c_va,
                     p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_b6_custom_to_rehab24.pth"))
        summary.append(("B", "B6i Custom->REHAB24 OOD", _row("B6i Custom->REHAB24 OOD", b6))); run_check_confusion()
        b12 = run_ood("B12i Custom->UI-PRMD OOD", c_tr, c_va,
                      p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
                      p(f"{CKPT_PREFIX}ood_b12_custom_to_uiprmd.pth"))
        summary.append(("B", "B12i Custom->UI-PRMD OOD", _row("B12i Custom->UI-PRMD OOD", b12))); run_check_confusion()

    if check_files(uiprmd_d, uiprmd_l):
        ui_tr, ui_va = split_frac(concat_ds((uiprmd_d, uiprmd_l)), 0.80, seed=RANDOM_SEED)
        b7 = run_ood("B7i UI-PRMD->EC3D OOD", ui_tr, ui_va,
                     p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_b7_uiprmd_to_ec3d.pth"))
        summary.append(("B", "B7i UI-PRMD->EC3D OOD", _row("B7i UI-PRMD->EC3D OOD", b7))); run_check_confusion()
        b8 = run_ood("B8i UI-PRMD->REHAB24 OOD", ui_tr, ui_va,
                     p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_b8_uiprmd_to_rehab24.pth"))
        summary.append(("B", "B8i UI-PRMD->REHAB24 OOD", _row("B8i UI-PRMD->REHAB24 OOD", b8))); run_check_confusion()
        b9 = run_ood("B9i UI-PRMD->Custom OOD", ui_tr, ui_va,
                     p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_b9_uiprmd_to_custom.pth"))
        summary.append(("B", "B9i UI-PRMD->Custom OOD", _row("B9i UI-PRMD->Custom OOD", b9))); run_check_confusion()

    if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l):
        c1_tr, c1_va = split_frac(concat_ds((ec3d_full_d, ec3d_full_l), (re_full_d, re_full_l)), 0.90, seed=RANDOM_SEED)
        c1 = run_ood("C1i EC3D+REHAB24->Custom OOD", c1_tr, c1_va,
                     p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_c1_ec3d_rehab24_to_custom.pth"))
        summary.append(("C", "C1i EC3D+REHAB24->Custom OOD", _row("C1i EC3D+REHAB24->Custom OOD", c1))); run_check_confusion()

    if check_files(ec3d_full_d, ec3d_full_l, custom_d, custom_l):
        c2_tr, c2_va = split_frac(concat_ds((ec3d_full_d, ec3d_full_l), (custom_d, custom_l)), 0.90, seed=RANDOM_SEED)
        c2 = run_ood("C2i EC3D+Custom->REHAB24 OOD", c2_tr, c2_va,
                     p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_c2_ec3d_custom_to_rehab24.pth"))
        summary.append(("C", "C2i EC3D+Custom->REHAB24 OOD", _row("C2i EC3D+Custom->REHAB24 OOD", c2))); run_check_confusion()

    if check_files(re_full_d, re_full_l, custom_d, custom_l):
        c3_tr, c3_va = split_frac(concat_ds((re_full_d, re_full_l), (custom_d, custom_l)), 0.90, seed=RANDOM_SEED)
        c3 = run_ood("C3i REHAB24+Custom->EC3D OOD", c3_tr, c3_va,
                     p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_c3_rehab24_custom_to_ec3d.pth"))
        summary.append(("C", "C3i REHAB24+Custom->EC3D OOD", _row("C3i REHAB24+Custom->EC3D OOD", c3))); run_check_confusion()

    if check_files(ec3d_full_d, ec3d_full_l, uiprmd_d, uiprmd_l):
        c4_tr, c4_va = split_frac(concat_ds((ec3d_full_d, ec3d_full_l), (uiprmd_d, uiprmd_l)), 0.90, seed=RANDOM_SEED)
        c4 = run_ood("C4i EC3D+UI-PRMD->Custom OOD", c4_tr, c4_va,
                     p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_c4_ec3d_uiprmd_to_custom.pth"))
        summary.append(("C", "C4i EC3D+UI-PRMD->Custom OOD", _row("C4i EC3D+UI-PRMD->Custom OOD", c4))); run_check_confusion()

    if check_files(re_full_d, re_full_l, uiprmd_d, uiprmd_l):
        c5_tr, c5_va = split_frac(concat_ds((re_full_d, re_full_l), (uiprmd_d, uiprmd_l)), 0.90, seed=RANDOM_SEED)
        c5 = run_ood("C5i REHAB24+UI-PRMD->Custom OOD", c5_tr, c5_va,
                     p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_c5_rehab24_uiprmd_to_custom.pth"))
        summary.append(("C", "C5i REHAB24+UI-PRMD->Custom OOD", _row("C5i REHAB24+UI-PRMD->Custom OOD", c5))); run_check_confusion()

    if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, custom_d, custom_l):
        c6_tr, c6_va = split_frac(concat_ds((ec3d_full_d, ec3d_full_l), (re_full_d, re_full_l), (custom_d, custom_l)), 0.90, seed=RANDOM_SEED)
        c6 = run_ood("C6i EC3D+REHAB24+Custom->UI-PRMD OOD", c6_tr, c6_va,
                     p("uiprmd_ood_test_data.npy"), p("uiprmd_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_c6_ec3d_rehab24_custom_to_uiprmd.pth"))
        summary.append(("C", "C6i EC3D+REHAB24+Custom->UI-PRMD OOD", _row("C6i EC3D+REHAB24+Custom->UI-PRMD OOD", c6))); run_check_confusion()

    if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, uiprmd_d, uiprmd_l):
        c8_tr, c8_va = split_frac(concat_ds((ec3d_full_d, ec3d_full_l), (re_full_d, re_full_l), (uiprmd_d, uiprmd_l)), 0.90, seed=RANDOM_SEED)
        c8 = run_ood("C8i EC3D+REHAB24+UI-PRMD->Custom OOD", c8_tr, c8_va,
                     p("custom_ood_test_data.npy"), p("custom_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_c8_ec3d_rehab24_uiprmd_to_custom.pth"))
        summary.append(("C", "C8i EC3D+REHAB24+UI-PRMD->Custom OOD", _row("C8i EC3D+REHAB24+UI-PRMD->Custom OOD", c8))); run_check_confusion()

    if check_files(ec3d_full_d, ec3d_full_l, custom_d, custom_l, uiprmd_d, uiprmd_l):
        c9_tr, c9_va = split_frac(concat_ds((ec3d_full_d, ec3d_full_l), (custom_d, custom_l), (uiprmd_d, uiprmd_l)), 0.90, seed=RANDOM_SEED)
        c9 = run_ood("C9i EC3D+Custom+UI-PRMD->REHAB24 OOD", c9_tr, c9_va,
                     p("rehab24_ood_test_data.npy"), p("rehab24_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_c9_ec3d_custom_uiprmd_to_rehab24.pth"))
        summary.append(("C", "C9i EC3D+Custom+UI-PRMD->REHAB24 OOD", _row("C9i EC3D+Custom+UI-PRMD->REHAB24 OOD", c9))); run_check_confusion()

    if check_files(re_full_d, re_full_l, custom_d, custom_l, uiprmd_d, uiprmd_l):
        c10_tr, c10_va = split_frac(concat_ds((re_full_d, re_full_l), (custom_d, custom_l), (uiprmd_d, uiprmd_l)), 0.90, seed=RANDOM_SEED)
        c10 = run_ood("C10i REHAB24+Custom+UI-PRMD->EC3D OOD", c10_tr, c10_va,
                      p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
                      p(f"{CKPT_PREFIX}ood_c10_rehab24_custom_uiprmd_to_ec3d.pth"))
        summary.append(("C", "C10i REHAB24+Custom+UI-PRMD->EC3D OOD", _row("C10i REHAB24+Custom+UI-PRMD->EC3D OOD", c10))); run_check_confusion()

    if check_files(custom_d, custom_l, uiprmd_d, uiprmd_l):
        c7_tr, c7_va = split_frac(concat_ds((custom_d, custom_l), (uiprmd_d, uiprmd_l)), 0.90, seed=RANDOM_SEED)
        c7 = run_ood("C7i Custom+UI-PRMD->EC3D OOD", c7_tr, c7_va,
                     p("ec3d_ood_test_data.npy"), p("ec3d_ood_test_labels.npy"),
                     p(f"{CKPT_PREFIX}ood_c7_custom_uiprmd_to_ec3d.pth"))
        summary.append(("C", "C7i Custom+UI-PRMD->EC3D OOD", _row("C7i Custom+UI-PRMD->EC3D OOD", c7))); run_check_confusion()

    print("\n\n" + "=" * 70)
    print("D1i — Mixed: all 4 datasets  80% train / 10% val / 10% test")
    print("=" * 70)
    if check_files(ec3d_full_d, ec3d_full_l, re_full_d, re_full_l, custom_d, custom_l, uiprmd_d, uiprmd_l):
        all_ds = concat_ds(
            (ec3d_full_d, ec3d_full_l),
            (re_full_d, re_full_l),
            (custom_d, custom_l),
            (uiprmd_d, uiprmd_l),
        )
        d_tr, d_va, d_te = split_80_10_10(all_ds, seed=RANDOM_SEED)
        model_d = train_model(d_tr, d_va, p(f"{CKPT_PREFIX}mixed_all_datasets.pth"), "D1i Mixed")
        d1 = evaluate(model_d, DataLoader(d_te, batch_size=TRAIN_BATCH_SIZE), tag="D1 Mixed all 80/10/10")
        summary.append(("D", "D1i Mixed all 80/10/10", _row("D1i Mixed all 80/10/10", d1)))
        run_check_confusion()

    print_summary(summary)
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
        "--ood-only",
        action="store_true",
        help="Skip IID LOSO/k-fold; run B1-B12, C1-C10, and D1 only",
    )
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
    main(ood_only=args.ood_only, mixed_only=args.mixed_only, iid_only=args.iid_only)
