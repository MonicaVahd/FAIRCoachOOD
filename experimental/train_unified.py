# =============================================================================
# EXPERIMENTAL -- NOT USED FOR THE PAPER -- PARITY WITH THE PAPER'S SCRIPTS IS UNVERIFIED
# This is a refactored single-file trainer written during release preparation.
# The paper's results were produced by the eight original trainer scripts in
# training/ (see docs/PAPER_TO_CODE.md).  Do not use this file to reproduce or
# compare against the paper.  One known difference found by reading the code:
# LR_PATIENCE = 5 here, whereas the paper (Section 3.3) states a
# ReduceLROnPlateau patience of 10.  Other differences were not audited.
# =============================================================================
"""
train/train.py — Unified benchmark trainer for all nine models.

Implements the training protocol from the paper (§3.3):
  - Optimiser: Adam, lr=1e-3, weight_decay=1e-4
  - Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
  - Early stopping: patience=20 on validation loss
  - Class weights: w_c = N / (2 * n_c)
  - Max epochs: 200
  - Seed: 42 (fixed once at start of main(); D7 note: cells run in
    sequence inside one process may not be fully independent — see
    docs/KNOWN_DIFFERENCES.md)

Supported modes:
  --mode iid    A1 (EC3D LOSO), A2 (REHAB24 LOSO), A4 (UI-PRMD LOSO)
  --mode ood    B-conditions (--conditions B1,B3,B7,B8,B10,B11)
  --mode mixed  D1 (all public datasets; Custom needed for paper result)

Custom YouTube dataset:
  Conditions A3, B2, B4-B6, B9, B12, C1-C10 require the Custom dataset
  (not distributed).  Pass --custom-data /path/to/custom_squat_train_data.npy
  and --custom-labels /path/to/custom_squat_train_labels.npy to enable them.

Usage
-----
  cd <repo root>
  python train/train.py --model stgcn --mode iid --out-dir output/
  python train/train.py --model hdgcn  --mode ood --conditions B1,B3 --out-dir output/
  python train/train.py --model protogcn --mode mixed --out-dir output/

Prerequisites
-------------
  Preprocessing tensors in --data-dir (default: output/):
    ec3d_train_data.npy, ec3d_train_labels.npy
    rehab24_train_data.npy, rehab24_train_labels.npy
    uiprmd_data.npy, uiprmd_labels.npy, uiprmd_subjects.npy
    ec3d_ood_test_data.npy, ec3d_ood_test_labels.npy
    rehab24_ood_test_data.npy, rehab24_ood_test_labels.npy
    uiprmd_ood_test_data.npy, uiprmd_ood_test_labels.npy
    ec3d_loso_round{1-4}_{train,val,test}_{data,labels}.npy
    rehab24_loso_round{1-9}_{train,val,test}_{data,labels}.npy
    uiprmd_loso_round{1-10}_{train,val,test}_{data,labels}.npy

CAUTION: This refactored trainer was not used to produce the paper's results.
The paper used eight separate trainer scripts.  Run with --model stgcn first
and compare results/ood_results.csv to verify your environment is correct.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Path setup ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, "models"))

from builders import BUILDERS
from metrics_utils import compute_metrics

# ── Training hyper-parameters (paper §3.3) ───────────────────────────────────
TRAIN_SEED         = 42
TRAIN_EPOCHS       = 200
TRAIN_BATCH_SIZE   = 16
TRAIN_LR           = 1e-3
TRAIN_WEIGHT_DECAY = 1e-4
EARLY_STOP_PAT     = 20
LR_PATIENCE        = 5
LR_FACTOR          = 0.5
POS_CLASS          = 1    # label 1 = correct squat

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Utilities ─────────────────────────────────────────────────────────────────

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ld(data_path: str, label_path: str) -> TensorDataset:
    """Load a data/label .npy pair into a TensorDataset."""
    return TensorDataset(
        torch.from_numpy(np.load(data_path)).float(),
        torch.from_numpy(np.load(label_path)).long(),
    )


def cat_ds(*datasets: TensorDataset) -> TensorDataset:
    return TensorDataset(*[torch.cat([d.tensors[i] for d in datasets]) for i in range(2)])


def split_frac(ds: TensorDataset, train_frac: float) -> Tuple[TensorDataset, TensorDataset]:
    N = len(ds)
    perm = np.random.default_rng(TRAIN_SEED).permutation(N)
    n_tr = int(train_frac * N)
    idx_tr = torch.tensor(perm[:n_tr])
    idx_va = torch.tensor(perm[n_tr:])
    return (TensorDataset(*[t[idx_tr] for t in ds.tensors]),
            TensorDataset(*[t[idx_va] for t in ds.tensors]))


def split_80_10_10(ds: TensorDataset) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    N = len(ds)
    perm = np.random.default_rng(TRAIN_SEED).permutation(N)
    n80, n10 = int(0.80 * N), int(0.10 * N)
    t, v, e = perm[:n80], perm[n80:n80+n10], perm[n80+n10:]
    return (TensorDataset(*[d.tensors[i][torch.tensor(t)] for i in range(2)]),
            TensorDataset(*[d.tensors[i][torch.tensor(v)] for i in range(2)]),
            TensorDataset(*[d.tensors[i][torch.tensor(e)] for i in range(2)]))


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    """w_c = N / (2 * n_c) for binary classification."""
    N = len(labels)
    w = torch.zeros(2)
    for c in range(2):
        nc = (labels == c).sum().item()
        w[c] = N / (2 * nc) if nc > 0 else 1.0
    return w.to(DEVICE)


def make_loader(ds: TensorDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def check(*paths: str) -> bool:
    missing = [p for p in paths if not os.path.exists(p)]
    for m in missing:
        print(f"  [MISSING] {m}")
    return not missing


# ── Training loop ─────────────────────────────────────────────────────────────

def train_and_eval(
    model_name: str,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    test_ds: TensorDataset,
    ckpt_path: str,
    tag: str,
) -> dict:
    """
    Train one experiment and evaluate on the test set.

    Returns a metrics dict with tp/tn/fp/fn/macro_f1 etc.
    """
    # Build model
    builder = BUILDERS.get(model_name)
    if builder is None:
        raise ValueError(f"Unknown model: {model_name}")
    model = builder(device=DEVICE)

    # Class-weighted cross-entropy
    all_train_labels = train_ds.tensors[1]
    crit = nn.CrossEntropyLoss(weight=class_weights(all_train_labels))

    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=LR_PATIENCE, factor=LR_FACTOR)

    tr_loader = make_loader(train_ds, TRAIN_BATCH_SIZE, shuffle=True)
    va_loader = make_loader(val_ds,   TRAIN_BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0

    model.train()
    for epoch in range(1, TRAIN_EPOCHS + 1):
        # ── training ──
        model.train()
        for X, y in tr_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            # BlockGCN needs (x, y_onehot, joint); other models need (x,)
            try:
                logits = model(X)
            except TypeError:
                y_oh = torch.zeros(len(y), 2, device=DEVICE).scatter_(1, y.unsqueeze(1), 1)
                logits = model(X, y_oh, X)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

        # ── validation ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in va_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                try:
                    logits = model(X)
                except TypeError:
                    y_oh = torch.zeros(len(y), 2, device=DEVICE).scatter_(1, y.unsqueeze(1), 1)
                    logits = model(X, y_oh, X)
                val_loss += crit(logits, y).item() * len(y)
        val_loss /= len(val_ds)
        sched.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PAT:
                print(f"  [{tag}] early stop at epoch {epoch}")
                break

    # Save best checkpoint
    if best_state is not None:
        os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
        torch.save(best_state, ckpt_path)
        model.load_state_dict(best_state)

    # ── test evaluation ──
    model.eval()
    all_preds, all_labels = [], []
    te_loader = make_loader(test_ds, TRAIN_BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for X, y in te_loader:
            X = X.to(DEVICE)
            try:
                logits = model(X)
            except TypeError:
                y_oh = torch.zeros(len(y), 2, device=DEVICE).scatter_(1, y.unsqueeze(1), 1)
                logits = model(X, y_oh, X)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    m = compute_metrics(preds, labels, pos_class=POS_CLASS)
    print(f"  [{tag}] macro_F1={m['macro_f1']*100:.1f}%  "
          f"tp={m['tp']} tn={m['tn']} fp={m['fp']} fn={m['fn']}")
    return m


# ── Experiment runners ────────────────────────────────────────────────────────

def run_loso(
    model_name: str, dataset: str, n_rounds: int,
    data_dir: str, out_dir: str,
) -> List[dict]:
    """Run LOSO experiments (A1, A2, A4)."""
    results = []
    for r in range(1, n_rounds + 1):
        def p(suffix): return os.path.join(data_dir, f"{dataset}_loso_round{r}_{suffix}.npy")
        if not check(p("train_data"), p("train_labels"), p("val_data"), p("val_labels"),
                     p("test_data"), p("test_labels")):
            print(f"  Skipping {dataset} round {r} (missing files)")
            continue
        tag = f"{dataset.upper()} LOSO round {r}"
        print(f"\n── {tag} ──")
        train_ds = ld(p("train_data"),  p("train_labels"))
        val_ds   = ld(p("val_data"),    p("val_labels"))
        test_ds  = ld(p("test_data"),   p("test_labels"))
        ckpt = os.path.join(out_dir, f"{model_name.lower()}_loso_{dataset}_round{r}.pth")
        m = train_and_eval(model_name, train_ds, val_ds, test_ds, ckpt, tag)
        results.append({"round": r, **m})
    return results


def run_b_condition(
    model_name: str, cond: str,
    source: str, target: str,
    data_dir: str, out_dir: str,
) -> Optional[dict]:
    """Run one B-condition OOD experiment."""
    src_data  = os.path.join(data_dir, f"{source}_train_data.npy")
    src_label = os.path.join(data_dir, f"{source}_train_labels.npy")
    tgt_data  = os.path.join(data_dir, f"{target}_ood_test_data.npy")
    tgt_label = os.path.join(data_dir, f"{target}_ood_test_labels.npy")
    if not check(src_data, src_label, tgt_data, tgt_label):
        return None
    # For UI-PRMD source: use uiprmd_data.npy / uiprmd_labels.npy
    if not os.path.exists(src_data):
        src_data  = os.path.join(data_dir, f"{source}_data.npy")
        src_label = os.path.join(data_dir, f"{source}_labels.npy")

    tag = f"{cond} {source.upper()}→{target.upper()} OOD"
    print(f"\n── {tag} ──")
    full_src = ld(src_data, src_label)
    train_ds, val_ds = split_frac(full_src, 0.80)
    test_ds = ld(tgt_data, tgt_label)
    ckpt = os.path.join(out_dir, f"{model_name.lower()}_ood_{cond.lower()}_{source}_to_{target}.pth")
    return train_and_eval(model_name, train_ds, val_ds, test_ds, ckpt, tag)


def run_mixed_d1(
    model_name: str,
    data_dir: str, out_dir: str,
    custom_data: Optional[str] = None,
    custom_labels: Optional[str] = None,
) -> Optional[dict]:
    """Run D1 mixed experiment (all datasets 80/10/10)."""
    ec3d_d  = os.path.join(data_dir, "ec3d_train_data.npy")
    ec3d_l  = os.path.join(data_dir, "ec3d_train_labels.npy")
    re_d    = os.path.join(data_dir, "rehab24_train_data.npy")
    re_l    = os.path.join(data_dir, "rehab24_train_labels.npy")
    ui_d    = os.path.join(data_dir, "uiprmd_data.npy")
    ui_l    = os.path.join(data_dir, "uiprmd_labels.npy")
    required = [ec3d_d, ec3d_l, re_d, re_l, ui_d, ui_l]
    if custom_data:
        required += [custom_data, custom_labels]
    if not check(*required):
        return None

    parts = [ld(ec3d_d, ec3d_l), ld(re_d, re_l), ld(ui_d, ui_l)]
    if custom_data:
        parts.append(ld(custom_data, custom_labels))
    all_ds = cat_ds(*parts)

    train_ds, val_ds, test_ds = split_80_10_10(all_ds)
    tag = "D1 Mixed all datasets"
    print(f"\n── {tag} ──")
    ckpt = os.path.join(out_dir, f"{model_name.lower()}_mixed_all_datasets.pth")
    return train_and_eval(model_name, train_ds, val_ds, test_ds, ckpt, tag)


# ── B-condition catalogue (public-data only) ──────────────────────────────────

PUBLIC_B = {
    "B1":  ("ec3d",   "rehab24"),
    "B3":  ("rehab24","ec3d"),
    "B7":  ("uiprmd", "ec3d"),
    "B8":  ("uiprmd", "rehab24"),
    "B10": ("ec3d",   "uiprmd"),
    "B11": ("rehab24","uiprmd"),
}

# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified benchmark trainer.")
    p.add_argument("--model",       required=True,
                   choices=list(BUILDERS.keys()),
                   help="Model name (see models/builders.py)")
    p.add_argument("--mode",        required=True,
                   choices=["iid", "ood", "mixed"],
                   help="Experiment mode")
    p.add_argument("--conditions",  default="B1,B3,B7,B8,B10,B11",
                   help="Comma-separated OOD conditions (mode=ood only)")
    p.add_argument("--out-dir",     default="output",
                   help="Output directory for checkpoints and logs")
    p.add_argument("--data-dir",    default="output",
                   help="Directory containing preprocessed .npy tensors")
    p.add_argument("--custom-data", default=None,
                   help="Path to custom_squat_train_data.npy (optional)")
    p.add_argument("--custom-labels", default=None,
                   help="Path to custom_squat_train_labels.npy (optional)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    seed_all(TRAIN_SEED)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Model : {args.model}")
    print(f"Mode  : {args.mode}")
    print(f"Device: {DEVICE}")

    if args.mode == "iid":
        print("\n=== A1: EC3D LOSO ===")
        run_loso(args.model, "ec3d", 4, args.data_dir, args.out_dir)
        print("\n=== A2: REHAB24 LOSO ===")
        run_loso(args.model, "rehab24", 9, args.data_dir, args.out_dir)
        print("\n=== A4: UI-PRMD LOSO ===")
        run_loso(args.model, "uiprmd", 10, args.data_dir, args.out_dir)

    elif args.mode == "ood":
        conditions = [c.strip().upper() for c in args.conditions.split(",")]
        for cond in conditions:
            if cond not in PUBLIC_B:
                print(f"WARNING: {cond} not in public-data catalogue; skipping.")
                continue
            source, target = PUBLIC_B[cond]
            run_b_condition(args.model, cond, source, target, args.data_dir, args.out_dir)

    elif args.mode == "mixed":
        run_mixed_d1(args.model, args.data_dir, args.out_dir,
                     args.custom_data, args.custom_labels)


if __name__ == "__main__":
    main()
