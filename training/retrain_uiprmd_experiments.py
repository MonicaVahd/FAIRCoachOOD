"""
training/retrain_uiprmd_experiments.py

Handles two categories of UI-PRMD-affected experiments for all 9 models,
following the unified preprocessing fix (normalize_xy now applied to UI-PRMD).

Category A â€” Full retrain + evaluate (UI-PRMD in training source):
  B7  UI-PRMD â†’ EC3D           B8  UI-PRMD â†’ REHAB24
  B9  UI-PRMD â†’ Custom         C4  EC3D+UI-PRMD â†’ Custom
  C5  REHAB24+UI-PRMD â†’ Custom C7  Custom+UI-PRMD â†’ EC3D
  C8  EC3D+REHAB24+UI-PRMD â†’ Custom
  C9  EC3D+Custom+UI-PRMD â†’ REHAB24
  C10 REHAB24+Custom+UI-PRMD â†’ EC3D
  D1  All 4 datasets mixed (80/10/10)

Category B â€” Re-evaluate only on new uiprmd_ood_test_data.npy:
  B10 EC3D â†’ UI-PRMD     B11 REHAB24 â†’ UI-PRMD
  B12 Custom â†’ UI-PRMD   C6  EC3D+REHAB24+Custom â†’ UI-PRMD

Outputs:
  output/uiprmd_rerun_results.txt      â€” same format as *_results.txt
  output/uiprmd_rerun_confusion.txt    â€” same format as confusion_all_models.txt
  (comparison table printed to stdout and appended to results file)

Prerequisites:
  uiprmd_to_stgcn.py must have been re-run to regenerate uiprmd_data.npy.
  This script will automatically refresh uiprmd_ood_test_data.npy.
"""

import os
import re
import shutil
import random
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_utils import ConfmatLogger, compute_metrics
from configs.constants import (
    OUTPUT_DIR, TRAIN_SEED as RANDOM_SEED, TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS, TRAIN_LR, TRAIN_WEIGHT_DECAY,
)
from models.builders import (
    build_blockgcn_model, build_ctrgcn_model, build_freqmixformer_model,
    build_hdgcn_model, build_hyperformer_model, build_infogcn_model,
    build_protogcn_model, build_skateformer_model, build_stgcn_model,
)

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

EARLY_STOP_PAT   = 20
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
NEAR_DEG_THRESH  = 5
POS_CLASS        = 1
RESULTS_OUT      = os.path.join(OUTPUT_DIR, "uiprmd_rerun_results.txt")
CONFUSION_OUT    = os.path.join(OUTPUT_DIR, "uiprmd_rerun_confusion.txt")
CONFMAT_CSV      = os.path.join(OUTPUT_DIR, "confusion_matrices_log.csv")
_confmat_logger: Optional[ConfmatLogger] = None
COL              = 42


# â”€â”€ Model table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# (name, tag_suffix, ckpt_prefix, build_fn, results_file)

MODELS: List[Tuple[str, str, str, Callable, str]] = [
    ("ST-GCN",        "",   "",               build_stgcn_model,        "stgcn_results.txt"),
    ("CTR-GCN",       "b",  "ctrgcn_",        build_ctrgcn_model,       "ctrgcn_results.txt"),
    ("InfoGCN",       "c",  "infogcn_",       build_infogcn_model,      "infogcn_results.txt"),
    ("BlockGCN",      "d",  "blockgcn_",      build_blockgcn_model,     "blockgcn_results.txt"),
    ("HD-GCN",        "e",  "hdgcn_",         build_hdgcn_model,        "hdgcn_results.txt"),
    ("ProtoGCN",      "f",  "protogcn_",      build_protogcn_model,     "protogcn_results.txt"),
    ("SkateFormer",   "g",  "skateformer_",   build_skateformer_model,  "skateformer_results.txt"),
    ("HyperFormer",   "h",  "hyperformer_",   build_hyperformer_model,  "hyperformer_results.txt"),
    ("FreqMixFormer", "i",  "freqmixformer_", build_freqmixformer_model,"freqmixformer_results.txt"),
]


# â”€â”€ Experiment specs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Each dict contains fields used to build train/test splits and checkpoints.
# "is_d1": True means split_80_10_10 (inline test); otherwise split_frac + OOD test.

CAT_A_EXPS: List[dict] = [
    {
        "id": "B7",   "section": "B",
        "label": "UI-PRMDâ†’EC3D OOD",
        "train_npy": [("uiprmd_data.npy", "uiprmd_labels.npy")],
        "train_frac": 0.80,   "is_d1": False,
        "test_npy":  ("ec3d_ood_test_data.npy", "ec3d_ood_test_labels.npy"),
        "old_ckpt_stem": "ood_b7_uiprmd_to_ec3d",
        "new_ckpt_stem": "ood_b7_uiprmd_to_ec3d_rerun",
    },
    {
        "id": "B8",   "section": "B",
        "label": "UI-PRMDâ†’REHAB24 OOD",
        "train_npy": [("uiprmd_data.npy", "uiprmd_labels.npy")],
        "train_frac": 0.80,   "is_d1": False,
        "test_npy":  ("rehab24_ood_test_data.npy", "rehab24_ood_test_labels.npy"),
        "old_ckpt_stem": "ood_b8_uiprmd_to_rehab24",
        "new_ckpt_stem": "ood_b8_uiprmd_to_rehab24_rerun",
    },
    {
        "id": "B9",   "section": "B",
        "label": "UI-PRMDâ†’Custom OOD",
        "train_npy": [("uiprmd_data.npy", "uiprmd_labels.npy")],
        "train_frac": 0.80,   "is_d1": False,
        "test_npy":  ("custom_ood_test_data.npy", "custom_ood_test_labels.npy"),
        "old_ckpt_stem": "ood_b9_uiprmd_to_custom",
        "new_ckpt_stem": "ood_b9_uiprmd_to_custom_rerun",
    },
    {
        "id": "C4",   "section": "C",
        "label": "EC3D+UI-PRMDâ†’Custom OOD",
        "train_npy": [("ec3d_train_data.npy", "ec3d_train_labels.npy"),
                      ("uiprmd_data.npy", "uiprmd_labels.npy")],
        "train_frac": 0.90,   "is_d1": False,
        "test_npy":  ("custom_ood_test_data.npy", "custom_ood_test_labels.npy"),
        "old_ckpt_stem": "ood_c4_ec3d_uiprmd_to_custom",
        "new_ckpt_stem": "ood_c4_ec3d_uiprmd_to_custom_rerun",
    },
    {
        "id": "C5",   "section": "C",
        "label": "REHAB24+UI-PRMDâ†’Custom OOD",
        "train_npy": [("rehab24_train_data.npy", "rehab24_train_labels.npy"),
                      ("uiprmd_data.npy", "uiprmd_labels.npy")],
        "train_frac": 0.90,   "is_d1": False,
        "test_npy":  ("custom_ood_test_data.npy", "custom_ood_test_labels.npy"),
        "old_ckpt_stem": "ood_c5_rehab24_uiprmd_to_custom",
        "new_ckpt_stem": "ood_c5_rehab24_uiprmd_to_custom_rerun",
    },
    {
        "id": "C7",   "section": "C",
        "label": "Custom+UI-PRMDâ†’EC3D OOD",
        "train_npy": [("custom_squat_train_data.npy", "custom_squat_train_labels.npy"),
                      ("uiprmd_data.npy", "uiprmd_labels.npy")],
        "train_frac": 0.90,   "is_d1": False,
        "test_npy":  ("ec3d_ood_test_data.npy", "ec3d_ood_test_labels.npy"),
        "old_ckpt_stem": "ood_c7_custom_uiprmd_to_ec3d",
        "new_ckpt_stem": "ood_c7_custom_uiprmd_to_ec3d_rerun",
    },
    {
        "id": "C8",   "section": "C",
        "label": "EC3D+REHAB24+UI-PRMDâ†’Custom OOD",
        "train_npy": [("ec3d_train_data.npy", "ec3d_train_labels.npy"),
                      ("rehab24_train_data.npy", "rehab24_train_labels.npy"),
                      ("uiprmd_data.npy", "uiprmd_labels.npy")],
        "train_frac": 0.90,   "is_d1": False,
        "test_npy":  ("custom_ood_test_data.npy", "custom_ood_test_labels.npy"),
        "old_ckpt_stem": "ood_c8_ec3d_rehab24_uiprmd_to_custom",
        "new_ckpt_stem": "ood_c8_ec3d_rehab24_uiprmd_to_custom_rerun",
    },
    {
        "id": "C9",   "section": "C",
        "label": "EC3D+Custom+UI-PRMDâ†’REHAB24 OOD",
        "train_npy": [("ec3d_train_data.npy", "ec3d_train_labels.npy"),
                      ("custom_squat_train_data.npy", "custom_squat_train_labels.npy"),
                      ("uiprmd_data.npy", "uiprmd_labels.npy")],
        "train_frac": 0.90,   "is_d1": False,
        "test_npy":  ("rehab24_ood_test_data.npy", "rehab24_ood_test_labels.npy"),
        "old_ckpt_stem": "ood_c9_ec3d_custom_uiprmd_to_rehab24",
        "new_ckpt_stem": "ood_c9_ec3d_custom_uiprmd_to_rehab24_rerun",
    },
    {
        "id": "C10",  "section": "C",
        "label": "REHAB24+Custom+UI-PRMDâ†’EC3D OOD",
        "train_npy": [("rehab24_train_data.npy", "rehab24_train_labels.npy"),
                      ("custom_squat_train_data.npy", "custom_squat_train_labels.npy"),
                      ("uiprmd_data.npy", "uiprmd_labels.npy")],
        "train_frac": 0.90,   "is_d1": False,
        "test_npy":  ("ec3d_ood_test_data.npy", "ec3d_ood_test_labels.npy"),
        "old_ckpt_stem": "ood_c10_rehab24_custom_uiprmd_to_ec3d",
        "new_ckpt_stem": "ood_c10_rehab24_custom_uiprmd_to_ec3d_rerun",
    },
    {
        "id": "D1",   "section": "D",
        "label": "Mixed all 80/10/10",
        "train_npy": [("ec3d_train_data.npy", "ec3d_train_labels.npy"),
                      ("rehab24_train_data.npy", "rehab24_train_labels.npy"),
                      ("custom_squat_train_data.npy", "custom_squat_train_labels.npy"),
                      ("uiprmd_data.npy", "uiprmd_labels.npy")],
        "train_frac": None,   "is_d1": True,
        "test_npy":  None,
        "old_ckpt_stem": "mixed_all_datasets",
        "new_ckpt_stem": "mixed_all_datasets_rerun",
    },
]

CAT_B_EXPS: List[dict] = [
    {
        "id": "B10",  "section": "B",
        "label": "EC3Dâ†’UI-PRMD OOD",
        "old_ckpt_stem": "ood_b10_ec3d_to_uiprmd",
        "test_npy": ("uiprmd_ood_test_data.npy", "uiprmd_ood_test_labels.npy"),
    },
    {
        "id": "B11",  "section": "B",
        "label": "REHAB24â†’UI-PRMD OOD",
        "old_ckpt_stem": "ood_b11_rehab24_to_uiprmd",
        "test_npy": ("uiprmd_ood_test_data.npy", "uiprmd_ood_test_labels.npy"),
    },
    {
        "id": "B12",  "section": "B",
        "label": "Customâ†’UI-PRMD OOD",
        "old_ckpt_stem": "ood_b12_custom_to_uiprmd",
        "test_npy": ("uiprmd_ood_test_data.npy", "uiprmd_ood_test_labels.npy"),
    },
    {
        "id": "C6",   "section": "C",
        "label": "EC3D+REHAB24+Customâ†’UI-PRMD OOD",
        "old_ckpt_stem": "ood_c6_ec3d_rehab24_custom_to_uiprmd",
        "test_npy": ("uiprmd_ood_test_data.npy", "uiprmd_ood_test_labels.npy"),
    },
]


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def p(name: str) -> str:
    return os.path.join(OUTPUT_DIR, name)


def check_files(*paths: str) -> bool:
    missing = [f for f in paths if not os.path.exists(f)]
    for m in missing:
        print(f"  [MISSING] {m}")
    return len(missing) == 0


def load_ds(data_path: str, label_path: str) -> TensorDataset:
    X = torch.from_numpy(np.load(data_path)).float()
    y = torch.from_numpy(np.load(label_path)).long()
    return TensorDataset(X, y)


def concat_ds(npy_pairs: List[Tuple[str, str]]) -> TensorDataset:
    X_parts, y_parts = [], []
    for dp, lp in npy_pairs:
        X_parts.append(np.load(p(dp)))
        y_parts.append(np.load(p(lp)))
    X = torch.from_numpy(np.concatenate(X_parts, axis=0)).float()
    y = torch.from_numpy(np.concatenate(y_parts, axis=0)).long()
    return TensorDataset(X, y)


def split_frac(ds: TensorDataset, train_frac: float) -> Tuple[TensorDataset, TensorDataset]:
    N   = len(ds)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(N)
    n_tr   = int(train_frac * N)
    tr_idx = torch.tensor(idx[:n_tr])
    va_idx = torch.tensor(idx[n_tr:])
    return (
        TensorDataset(*[t[tr_idx] for t in ds.tensors]),
        TensorDataset(*[t[va_idx] for t in ds.tensors]),
    )


def split_80_10_10(ds: TensorDataset) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    N   = len(ds)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(N)
    n80    = int(0.80 * N)
    n10    = int(0.10 * N)
    tr_idx = torch.tensor(idx[:n80])
    va_idx = torch.tensor(idx[n80:n80 + n10])
    te_idx = torch.tensor(idx[n80 + n10:])
    return (
        TensorDataset(*[t[tr_idx] for t in ds.tensors]),
        TensorDataset(*[t[va_idx] for t in ds.tensors]),
        TensorDataset(*[t[te_idx] for t in ds.tensors]),
    )


# â”€â”€ Model-specific forward â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def forward_logits_and_loss(
    model_name: str,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if model_name == "InfoGCN":
        logits, _ = model(x)
        loss = criterion(logits, y)
    elif model_name == "BlockGCN":
        y_oh = F.one_hot(y, num_classes=2).float()
        logits, topo_loss = model(x, y_oh, x)
        cls_loss = criterion(logits, y)
        if isinstance(topo_loss, torch.Tensor) and topo_loss.ndim == 0:
            loss = cls_loss + topo_loss
        else:
            loss = cls_loss
    elif model_name == "SkateFormer":
        bsz, _, t, _, _ = x.shape
        index_t = torch.zeros(bsz, t, dtype=torch.long, device=x.device)
        logits = model(x, index_t)
        loss = criterion(logits, y)
    elif model_name == "HyperFormer":
        logits, _ = model(x, y)
        loss = criterion(logits, y)
    else:
        logits = model(x)
        loss = criterion(logits, y)
    return logits, loss


# â”€â”€ Train / evaluate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train_model(
    model_name: str,
    build_fn: Callable,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    ckpt_path: str,
    label: str,
) -> nn.Module:
    labels_np = train_ds.tensors[1].numpy()
    n0 = int((labels_np == 0).sum())
    n1 = int((labels_np == 1).sum())
    N  = len(labels_np)
    w  = torch.tensor([N / (2 * max(n0, 1)), N / (2 * max(n1, 1))]).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)
    print(f"  [{label}] train={len(train_ds)} val={len(val_ds)}")
    print(f"  Class weights: incorrect={w[0]:.3f}  correct={w[1]:.3f}")

    model     = build_fn(checkpoint_path=None, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=False)

    best_val = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        t_losses: List[float] = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            _, loss = forward_logits_and_loss(model_name, model, X_b, y_b, criterion)
            loss.backward()
            optimizer.step()
            t_losses.append(float(loss.item()))

        model.eval()
        v_losses: List[float] = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                _, loss = forward_logits_and_loss(model_name, model, X_b, y_b, criterion)
                v_losses.append(float(loss.item()))

        t_loss = float(np.mean(t_losses)) if t_losses else 0.0
        v_loss = float(np.mean(v_losses)) if v_losses else 0.0
        scheduler.step(v_loss)

        if epoch == 1 or epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}  t_loss={t_loss:.4f}  v_loss={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            best_epoch = epoch
            no_improve = 0
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PAT:
                print(f"    Early stop @ epoch {epoch} (best={best_epoch}, val={best_val:.4f})")
                break

    print(f"  Best val={best_val:.4f} @ epoch {best_epoch}  â†’  {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model


def evaluate_with_confmat(
    model_name: str,
    model: nn.Module,
    test_ds: TensorDataset,
    ckpt_path: str,
    test_data_path: str,
    test_labels_path: str,
    tag: str,
) -> Tuple[dict, dict]:
    """
    Evaluate model on test_ds.
    Returns (metrics_dict, confmat_dict).
    confmat_dict keys: tp, fp, fn, tn, pred_correct, pred_incorrect, tag,
                       ckpt_path, test_data_path, test_labels_path.
    """
    dummy_crit = nn.CrossEntropyLoss().to(DEVICE)
    loader = DataLoader(test_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=False)
    preds_all, labels_all = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(DEVICE)
            y_b_dev = y_b.to(DEVICE)
            logits, _ = forward_logits_and_loss(model_name, model, X_b, y_b_dev, dummy_crit)
            preds_all.append(logits.argmax(dim=1).cpu().numpy())
            labels_all.append(y_b.numpy())

    preds  = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)

    metrics = compute_metrics(preds, labels)
    pc = metrics["pred_correct"]
    pi = metrics["pred_incorrect"]
    confmat = {
        "tag": tag, "ckpt_path": ckpt_path,
        "test_data_path": test_data_path, "test_labels_path": test_labels_path,
        "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"], "tn": metrics["tn"],
        "pred_correct": pc, "pred_incorrect": pi,
    }
    if _confmat_logger is not None:
        _confmat_logger.append("retrain_uiprmd_experiments.py", model_name, tag, metrics)
    return metrics, confmat


def classify_degeneracy(pred_correct: int, pred_incorrect: int) -> str:
    if pred_correct == 0 or pred_incorrect == 0:
        return "DEGâ€ "
    if pred_correct <= NEAR_DEG_THRESH or pred_incorrect <= NEAR_DEG_THRESH:
        return "~DEG"
    return "OK"


# â”€â”€ Old F1 collection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def parse_f1_from_results_file(results_file: str, exp_id: str, tag_suffix: str) -> Optional[float]:
    """
    Parse F1 from the model's existing results file for a given experiment.
    Looks for a line containing '{exp_id}{tag_suffix} ' (e.g. 'B7c ', 'D1 ').
    """
    full_path = p(results_file)
    if not os.path.exists(full_path):
        return None
    search = f"{exp_id}{tag_suffix} "
    with open(full_path, encoding="utf-8") as f:
        for line in f:
            if search in line:
                pcts = re.findall(r"(\d+\.\d+)%", line)
                if len(pcts) >= 4:
                    return float(pcts[3]) / 100.0
    return None


def collect_old_f1() -> Dict[Tuple[str, str], Optional[float]]:
    """
    Parse old F1 for every affected experiment from the existing *_results.txt
    files (written before the UI-PRMD fix).  This avoids any dependency on
    the original test-data files, which may already have been refreshed.
    Returns {(model_name, exp_id): f1_or_None}.
    """
    old_f1: Dict[Tuple[str, str], Optional[float]] = {}
    all_exps = CAT_A_EXPS + CAT_B_EXPS
    for mname, msuffix, _, _, mresults in MODELS:
        for exp in all_exps:
            exp_id = exp["id"]
            f1 = parse_f1_from_results_file(mresults, exp_id, msuffix)
            if f1 is None:
                print(f"  [WARN] Could not find old F1 for {mname} {exp_id} in {mresults}")
            old_f1[(mname, exp_id)] = f1
    return old_f1


# â”€â”€ Refresh uiprmd_ood_test_data.npy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def refresh_uiprmd_ood_test() -> None:
    src_data   = p("uiprmd_data.npy")
    src_labels = p("uiprmd_labels.npy")
    dst_data   = p("uiprmd_ood_test_data.npy")
    dst_labels = p("uiprmd_ood_test_labels.npy")
    if not check_files(src_data, src_labels):
        raise FileNotFoundError("uiprmd_data.npy or labels missing - run uiprmd_to_stgcn.py first.")
    # Skip if already refreshed (file size is a quick proxy)
    if (os.path.exists(dst_data)
            and os.path.getsize(dst_data) == os.path.getsize(src_data)
            and os.path.exists(dst_labels)
            and os.path.getsize(dst_labels) == os.path.getsize(src_labels)):
        print("  uiprmd_ood_test_data.npy already matches source -- skipping copy.")
        return
    shutil.copy2(src_data,   dst_data)
    shutil.copy2(src_labels, dst_labels)
    y = np.load(dst_labels)
    print(f"  Refreshed uiprmd_ood_test_data.npy  ({len(y)} seqs, "
          f"correct={int((y==1).sum())}, incorrect={int((y==0).sum())})")


# â”€â”€ Main experiment runners â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_cat_b_old(
) -> Tuple[Dict[Tuple[str, str], dict], List[dict]]:
    """
    Phase 0: evaluate existing Cat B checkpoints on current (still-old)
    uiprmd_ood_test_data.npy to capture pre-fix F1.
    Returns (old_metrics_dict, confmat_list_for_old).
    """
    print("\n" + "=" * 70)
    print("Phase 0: Collecting old Cat B F1 (pre-fix uiprmd_ood_test_data.npy)")
    print("=" * 70)
    old_metrics: Dict[Tuple[str, str], dict] = {}
    confmats: List[dict] = []

    for mname, msuffix, mprefix, mbuild, _ in MODELS:
        for exp in CAT_B_EXPS:
            ckpt = p(mprefix + exp["old_ckpt_stem"] + ".pth")
            td, tl = exp["test_npy"]
            tag = f"OLD {exp['id']}{msuffix} {exp['label']}"
            print(f"\n  {mname} {exp['id']} (old) â€” {exp['label']}")

            if not os.path.exists(ckpt):
                print(f"    [SKIP] checkpoint not found: {ckpt}")
                continue
            if not check_files(p(td), p(tl)):
                continue

            model = mbuild(checkpoint_path=ckpt, device=DEVICE)
            model.eval()
            test_ds = load_ds(p(td), p(tl))
            m, cm = evaluate_with_confmat(mname, model, test_ds, ckpt, p(td), p(tl), tag)
            old_metrics[(mname, exp["id"])] = m
            confmats.append({**cm, "model": mname})
            print(f"    F1={m['f1']*100:.1f}%  Acc={m['acc']*100:.1f}%  "
                  f"prec={m['prec']*100:.1f}%  rec={m['rec']*100:.1f}%  n={m['n']}")
            model.cpu(); del model; torch.cuda.empty_cache()

    return old_metrics, confmats


def run_cat_a(
) -> Tuple[Dict[Tuple[str, str], dict], List[dict]]:
    """
    Phase 2: Retrain Cat A experiments with _rerun checkpoints.
    Returns (new_metrics_dict, confmat_list).
    """
    print("\n" + "=" * 70)
    print("Phase 2: Category A â€” Full retrain (UI-PRMD in training source)")
    print("=" * 70)
    new_metrics: Dict[Tuple[str, str], dict] = {}
    confmats: List[dict] = []

    for mname, msuffix, mprefix, mbuild, _ in MODELS:
        print(f"\n{'='*60}\n  MODEL: {mname}\n{'='*60}")

        for exp in CAT_A_EXPS:
            exp_id = exp["id"]
            full_label = f"{exp_id}{msuffix} {exp['label']}"
            print(f"\n  --- {full_label} ---")

            # Check all required source files
            src_files = [p(d) for d, _ in exp["train_npy"]] + [p(l) for _, l in exp["train_npy"]]
            if not check_files(*src_files):
                print(f"    [SKIP] missing source data")
                continue

            ckpt = p(mprefix + exp["new_ckpt_stem"] + ".pth")

            # Build train/val/(test) splits
            full_ds = concat_ds(exp["train_npy"])

            if exp["is_d1"]:
                train_ds, val_ds, test_ds = split_80_10_10(full_ds)
                test_data_path   = "inline"
                test_labels_path = "inline"
            else:
                train_ds, val_ds = split_frac(full_ds, exp["train_frac"])
                td, tl = exp["test_npy"]
                if not check_files(p(td), p(tl)):
                    continue
                test_ds          = load_ds(p(td), p(tl))
                test_data_path   = p(td)
                test_labels_path = p(tl)

            # Load cached checkpoint if already exists
            if os.path.exists(ckpt):
                print(f"    [cache] {os.path.basename(ckpt)}")
                model = mbuild(checkpoint_path=ckpt, device=DEVICE)
                model.eval()
            else:
                model = train_model(mname, mbuild, train_ds, val_ds, ckpt, full_label)

            # Evaluate
            m, cm = evaluate_with_confmat(
                mname, model, test_ds, ckpt,
                test_data_path, test_labels_path, full_label,
            )
            new_metrics[(mname, exp_id)] = m
            confmats.append({**cm, "model": mname, "exp_id": exp_id, "exp_label": exp["label"]})
            status = classify_degeneracy(cm["pred_correct"], cm["pred_incorrect"])
            print(f"    â†’ F1={m['f1']*100:.1f}%  Acc={m['acc']*100:.1f}%  "
                  f"prec={m['prec']*100:.1f}%  rec={m['rec']*100:.1f}%  "
                  f"n={m['n']}  [{status}]")
            model.cpu(); del model; torch.cuda.empty_cache()

    return new_metrics, confmats


def run_cat_b_new(
) -> Tuple[Dict[Tuple[str, str], dict], List[dict]]:
    """
    Phase 3: Re-evaluate Cat B with original checkpoint on new uiprmd_ood_test_data.npy.
    Returns (new_metrics_dict, confmat_list).
    """
    print("\n" + "=" * 70)
    print("Phase 3: Category B â€” Re-evaluate on refreshed uiprmd_ood_test_data.npy")
    print("=" * 70)
    new_metrics: Dict[Tuple[str, str], dict] = {}
    confmats: List[dict] = []

    for mname, msuffix, mprefix, mbuild, _ in MODELS:
        for exp in CAT_B_EXPS:
            exp_id = exp["id"]
            full_label = f"{exp_id}{msuffix} {exp['label']}"
            print(f"\n  {mname} {exp_id} (new) â€” {exp['label']}")
            ckpt = p(mprefix + exp["old_ckpt_stem"] + ".pth")
            td, tl = exp["test_npy"]

            if not os.path.exists(ckpt):
                print(f"    [SKIP] checkpoint not found: {ckpt}")
                continue
            if not check_files(p(td), p(tl)):
                continue

            model = mbuild(checkpoint_path=ckpt, device=DEVICE)
            model.eval()
            test_ds = load_ds(p(td), p(tl))
            m, cm = evaluate_with_confmat(mname, model, test_ds, ckpt, p(td), p(tl), full_label)
            new_metrics[(mname, exp_id)] = m
            confmats.append({**cm, "model": mname, "exp_id": exp_id, "exp_label": exp["label"]})
            status = classify_degeneracy(cm["pred_correct"], cm["pred_incorrect"])
            print(f"    â†’ F1={m['f1']*100:.1f}%  Acc={m['acc']*100:.1f}%  "
                  f"prec={m['prec']*100:.1f}%  rec={m['rec']*100:.1f}%  "
                  f"n={m['n']}  [{status}]")
            model.cpu(); del model; torch.cuda.empty_cache()

    return new_metrics, confmats


# â”€â”€ Output writers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

HDR = (
    f"  {'Condition':<{COL}}"
    f"  {'Acc':>10}"
    f"  {'Prec':>10}"
    f"  {'Rec':>10}"
    f"  {'F1':>10}"
    f"  {'MacroF1':>10}"
    f"  {'N':>6}"
    f"  {'Status':>6}"
)
SEP = "  " + "-" * (len(HDR) - 2)
SECTION_LABELS = {
    "B": "OOD â€” single source train",
    "C": "OOD â€” dual/triple source train",
    "D": "Mixed (all datasets)",
}


def fmt_row(tag: str, m: dict, status: str = "") -> str:
    sc = f"{status:>6}" if status else f"{'':>6}"
    return (
        f"  {tag:<{COL}}"
        f"  {m['acc']*100:6.1f}%   "
        f"  {m['prec']*100:6.1f}%   "
        f"  {m['rec']*100:6.1f}%   "
        f"  {m['f1']*100:6.1f}%   "
        f"  {m['macro_f1']*100:6.1f}%   "
        f"  {m['n']:>6}"
        f"  {sc}"
    )


def write_results(
    cat_a_metrics: Dict[Tuple[str, str], dict],
    cat_b_metrics: Dict[Tuple[str, str], dict],
    old_f1:        Dict[Tuple[str, str], Optional[float]],
    comparison_lines: List[str],
) -> None:
    all_exps_ordered = CAT_A_EXPS + CAT_B_EXPS
    lines: List[str] = []

    lines.append("=" * 110)
    lines.append("UI-PRMD Rerun Results â€” All 9 Models")
    lines.append("(Category A = full retrain; Category B = re-eval with existing ckpt on new uiprmd_ood_test_data.npy)")
    lines.append("=" * 110)
    lines.append(f"Device: {DEVICE}")
    lines.append(f"Epochs (max): {TRAIN_EPOCHS}  Early-stop patience: {EARLY_STOP_PAT}")
    lines.append("")

    ood_totals = {"total": 0, "DEGâ€ ": 0, "~DEG": 0, "OK": 0}

    for mname, msuffix, _, _, _ in MODELS:
        lines.append("=" * 110)
        lines.append(f"  {mname} â€” UI-PRMD Rerun Summary")
        lines.append("=" * 110)
        lines.append(HDR)

        last_sec = None
        for exp in all_exps_ordered:
            sec = exp["section"]
            if sec != last_sec:
                lines.append(SEP)
                lines.append(f"  [{SECTION_LABELS.get(sec, sec)}]")
                last_sec = sec

            exp_id  = exp["id"]
            tag     = f"{exp_id}{msuffix} {exp['label']}"
            is_cat_b = exp in CAT_B_EXPS
            m = (cat_b_metrics if is_cat_b else cat_a_metrics).get((mname, exp_id))

            if m is None:
                lines.append(f"  {tag:<{COL}}  (skipped)")
                continue

            status = classify_degeneracy(m.get("pred_correct", 1), m.get("pred_incorrect", 1))
            if sec in ("B", "C"):
                ood_totals["total"] += 1
                ood_totals[status] = ood_totals.get(status, 0) + 1

            lines.append(fmt_row(tag, m, status))

        lines.append(SEP)
        lines.append("")

    # OOD degeneracy summary
    total = ood_totals["total"]
    lines.append("=" * 110)
    lines.append("  OOD degeneracy summary (across all models, Cat A + Cat B):")
    if total > 0:
        deg = ood_totals["DEGâ€ "] + ood_totals.get("~DEG", 0)
        lines.append(f"    Total OOD experiments : {total}")
        lines.append(f"    DEGâ€                   : {ood_totals['DEGâ€ ']} ({100*ood_totals['DEGâ€ ']/total:.1f}%)")
        lines.append(f"    ~DEG                  : {ood_totals.get('~DEG',0)} ({100*ood_totals.get('~DEG',0)/total:.1f}%)")
        lines.append(f"    OK                    : {ood_totals.get('OK',0)} ({100*ood_totals.get('OK',0)/total:.1f}%)")
    lines.append("  â€ Degenerate: model predicted single class for all test samples")
    lines.append("")

    # Append comparison table
    lines.append("")
    lines.extend(comparison_lines)

    text = "\n".join(lines) + "\n"
    print(text)
    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved -> {RESULTS_OUT}")


def write_confusion(
    cat_a_confmats: List[dict],
    cat_b_confmats: List[dict],
) -> None:
    lines: List[str] = []
    lines.append(f"Device: {DEVICE}")
    lines.append(f"Output dir: {OUTPUT_DIR}")
    lines.append("")

    all_confmats = cat_a_confmats + cat_b_confmats
    current_model = None

    for cm in all_confmats:
        mname = cm["model"]
        if mname != current_model:
            lines.append("")
            lines.append("#" * 70)
            lines.append(f"# {mname}")
            lines.append("#" * 70)
            current_model = mname

        tag = cm["tag"]
        lines.append("")
        lines.append("=" * 70)
        lines.append(tag)
        lines.append("=" * 70)
        lines.append(f"  Checkpoint : {cm['ckpt_path']}")
        lines.append(f"  Test data  : {cm['test_data_path']}")
        lines.append(f"  Test labels: {cm['test_labels_path']}")
        lines.append("")
        lines.append(f"  Confusion Matrix for {tag}:")
        lines.append(f"                       Predicted Correct    Predicted Incorrect")
        lines.append(f"  Actual Correct       {cm['tp']:>17d}    {cm['fn']:>19d}")
        lines.append(f"  Actual Incorrect     {cm['fp']:>17d}    {cm['tn']:>19d}")
        lines.append(
            f"  True Positives={cm['tp']}  True Negatives={cm['tn']}  "
            f"False Positives={cm['fp']}  False Negatives={cm['fn']}"
        )
        lines.append(
            f"  [{tag}] Majority class prediction check: "
            f"model predicted correct={cm['pred_correct']} times, "
            f"incorrect={cm['pred_incorrect']} times"
        )

    text = "\n".join(lines) + "\n"
    with open(CONFUSION_OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved -> {CONFUSION_OUT}")


def build_comparison_table(
    cat_a_metrics: Dict[Tuple[str, str], dict],
    cat_b_metrics: Dict[Tuple[str, str], Optional[float]],
    old_f1: Dict[Tuple[str, str], Optional[float]],
) -> List[str]:
    all_exp_ids = [e["id"] for e in CAT_A_EXPS] + [e["id"] for e in CAT_B_EXPS]
    model_names = [m for m, *_ in MODELS]

    lines: List[str] = []
    lines.append("=" * 90)
    lines.append("COMPARISON TABLE â€” Old F1 vs New F1 vs Delta")
    lines.append("(Cat A: new = retrained rerun; Cat B: new = old ckpt on refreshed test data)")
    lines.append("(Old = pre-fix value; D1 old parsed from *_results.txt; others evaluated from old ckpt)")
    lines.append("=" * 90)
    lines.append(
        f"  {'Experiment':<36}  {'Model':<14}  "
        f"{'Old F1':>8}  {'New F1':>8}  {'Î” F1':>8}"
    )
    lines.append("  " + "-" * 82)

    for exp_id in all_exp_ids:
        is_cat_b = exp_id in [e["id"] for e in CAT_B_EXPS]
        new_src  = cat_b_metrics if is_cat_b else cat_a_metrics
        exp_label = next(
            (e["label"] for e in (CAT_A_EXPS + CAT_B_EXPS) if e["id"] == exp_id), exp_id
        )
        tag_col = f"{exp_id} {exp_label}"[:36]

        for i, (mname, *_) in enumerate(MODELS):
            new_m = new_src.get((mname, exp_id))
            old_v = old_f1.get((mname, exp_id))

            new_f1_str = f"{new_m['f1']*100:6.1f}%" if new_m else "   N/A"
            old_f1_str = f"{old_v*100:6.1f}%"       if old_v is not None else "   N/A"
            if new_m and old_v is not None:
                delta = (new_m["f1"] - old_v) * 100
                delta_str = f"{delta:+6.1f}%"
            else:
                delta_str = "     N/A"

            exp_col = tag_col if i == 0 else ""
            lines.append(
                f"  {exp_col:<36}  {mname:<14}  "
                f"{old_f1_str:>8}  {new_f1_str:>8}  {delta_str:>8}"
            )
        lines.append("")

    return lines


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main() -> None:
    global _confmat_logger
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _confmat_logger = ConfmatLogger(CONFMAT_CSV)
    _confmat_logger.reset()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    print("Random seed fixed at 42 â€” results are reproducible")
    print(f"Device       : {DEVICE}")
    print(f"Early-stop   : patience={EARLY_STOP_PAT}")
    print(f"Max epochs   : {TRAIN_EPOCHS}")
    print(f"Batch size   : {TRAIN_BATCH_SIZE}")

    # Phase 0: parse old F1 for all experiments from existing results files
    print("\nPhase 0: Parsing old F1 scores from existing results files ...")
    old_f1 = collect_old_f1()

    # Phase 1: refresh uiprmd_ood_test_data.npy
    print("\n" + "=" * 70)
    print("Phase 1: Refreshing uiprmd_ood_test_data.npy from updated uiprmd_data.npy")
    print("=" * 70)
    refresh_uiprmd_ood_test()

    # Phase 2: Cat A full retrain
    cat_a_metrics, cat_a_confmats = run_cat_a()

    # Phase 3: Cat B re-eval on new test data
    cat_b_metrics, cat_b_confmats = run_cat_b_new()

    # Build comparison table
    comparison_lines = build_comparison_table(cat_a_metrics, cat_b_metrics, old_f1)

    # Write outputs
    print("\n" + "=" * 70)
    print("Writing output files ...")
    print("=" * 70)
    write_results(cat_a_metrics, cat_b_metrics, old_f1, comparison_lines)
    write_confusion(cat_a_confmats, cat_b_confmats)

    # Print comparison table standalone
    print("\n" + "\n".join(comparison_lines))


if __name__ == "__main__":
    main()

