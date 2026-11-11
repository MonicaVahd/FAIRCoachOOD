"""
training/check_confusion.py — Standalone confusion-matrix report for OOD checkpoints.

Loads saved OOD checkpoints for ST-GCN, CTR-GCN, InfoGCN, BlockGCN, HD-GCN,
ProtoGCN, SkateFormer, HyperFormer, and FreqMixFormer; prints a confusion matrix on each matching OOD test
set.  Does not retrain.  Missing checkpoints are skipped with a [SKIP] message.

Usage
-----
    cd <project_root>
    python training/check_confusion.py

Prerequisites
-------------
  split_ood.py  → output/*_ood_test_*.npy
  train scripts → output/{prefix}ood_b*.pth, output/{prefix}ood_c*.pth
"""

import os
import sys
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.builders import (
    build_stgcn_model,
    build_ctrgcn_model,
    build_infogcn_model,
    build_blockgcn_model,
    build_hdgcn_model,
    build_protogcn_model,
    build_skateformer_model,
    build_hyperformer_model,
    build_freqmixformer_model,
)
from configs.constants import TRAIN_BATCH_SIZE, OUTPUT_DIR

POS_CLASS = 1  # 1 = correct squat
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (tag, checkpoint filename, test_data filename, test_labels filename)
OOD_EXPERIMENTS = [
    ("B1 EC3D->REHAB24 OOD",              "ood_b1_ec3d_to_rehab24.pth",
     "rehab24_ood_test_data.npy",          "rehab24_ood_test_labels.npy"),
    ("B2 EC3D->Custom OOD",               "ood_b2_ec3d_to_custom.pth",
     "custom_ood_test_data.npy",           "custom_ood_test_labels.npy"),
    ("B3 REHAB24->EC3D OOD",              "ood_b3_rehab24_to_ec3d.pth",
     "ec3d_ood_test_data.npy",             "ec3d_ood_test_labels.npy"),
    ("B4 REHAB24->Custom OOD",            "ood_b4_rehab24_to_custom.pth",
     "custom_ood_test_data.npy",           "custom_ood_test_labels.npy"),
    ("B5 Custom->EC3D OOD",               "ood_b5_custom_to_ec3d.pth",
     "ec3d_ood_test_data.npy",             "ec3d_ood_test_labels.npy"),
    ("B6 Custom->REHAB24 OOD",            "ood_b6_custom_to_rehab24.pth",
     "rehab24_ood_test_data.npy",          "rehab24_ood_test_labels.npy"),
    ("B7 UI-PRMD->EC3D OOD",              "ood_b7_uiprmd_to_ec3d.pth",
     "ec3d_ood_test_data.npy",             "ec3d_ood_test_labels.npy"),
    ("B8 UI-PRMD->REHAB24 OOD",           "ood_b8_uiprmd_to_rehab24.pth",
     "rehab24_ood_test_data.npy",          "rehab24_ood_test_labels.npy"),
    ("B9 UI-PRMD->Custom OOD",            "ood_b9_uiprmd_to_custom.pth",
     "custom_ood_test_data.npy",           "custom_ood_test_labels.npy"),
    ("B10 EC3D->UI-PRMD OOD",             "ood_b10_ec3d_to_uiprmd.pth",
     "uiprmd_ood_test_data.npy",           "uiprmd_ood_test_labels.npy"),
    ("B11 REHAB24->UI-PRMD OOD",          "ood_b11_rehab24_to_uiprmd.pth",
     "uiprmd_ood_test_data.npy",           "uiprmd_ood_test_labels.npy"),
    ("B12 Custom->UI-PRMD OOD",           "ood_b12_custom_to_uiprmd.pth",
     "uiprmd_ood_test_data.npy",           "uiprmd_ood_test_labels.npy"),
    ("C1 EC3D+REHAB24->Custom OOD",       "ood_c1_ec3d_rehab24_to_custom.pth",
     "custom_ood_test_data.npy",           "custom_ood_test_labels.npy"),
    ("C2 EC3D+Custom->REHAB24 OOD",       "ood_c2_ec3d_custom_to_rehab24.pth",
     "rehab24_ood_test_data.npy",          "rehab24_ood_test_labels.npy"),
    ("C3 REHAB24+Custom->EC3D OOD",       "ood_c3_rehab24_custom_to_ec3d.pth",
     "ec3d_ood_test_data.npy",             "ec3d_ood_test_labels.npy"),
    ("C4 EC3D+UI-PRMD->Custom OOD",       "ood_c4_ec3d_uiprmd_to_custom.pth",
     "custom_ood_test_data.npy",           "custom_ood_test_labels.npy"),
    ("C5 REHAB24+UI-PRMD->Custom OOD",    "ood_c5_rehab24_uiprmd_to_custom.pth",
     "custom_ood_test_data.npy",           "custom_ood_test_labels.npy"),
    ("C6 EC3D+REHAB24+Custom->UI-PRMD OOD", "ood_c6_ec3d_rehab24_custom_to_uiprmd.pth",
     "uiprmd_ood_test_data.npy",           "uiprmd_ood_test_labels.npy"),
    ("C8 EC3D+REHAB24+UI-PRMD->Custom OOD", "ood_c8_ec3d_rehab24_uiprmd_to_custom.pth",
     "custom_ood_test_data.npy",           "custom_ood_test_labels.npy"),
    ("C9 EC3D+Custom+UI-PRMD->REHAB24 OOD", "ood_c9_ec3d_custom_uiprmd_to_rehab24.pth",
     "rehab24_ood_test_data.npy",          "rehab24_ood_test_labels.npy"),
    ("C10 REHAB24+Custom+UI-PRMD->EC3D OOD", "ood_c10_rehab24_custom_uiprmd_to_ec3d.pth",
     "ec3d_ood_test_data.npy",             "ec3d_ood_test_labels.npy"),
    ("C7 Custom+UI-PRMD->EC3D OOD",       "ood_c7_custom_uiprmd_to_ec3d.pth",
     "ec3d_ood_test_data.npy",             "ec3d_ood_test_labels.npy"),
]

def p(name: str) -> str:
    return os.path.join(OUTPUT_DIR, name)


def load_ds(data_path: str, label_path: str) -> TensorDataset:
    X = torch.from_numpy(np.load(data_path)).float()
    y = torch.from_numpy(np.load(label_path)).long()
    return TensorDataset(X, y)


def add_tag_suffix(tag: str, suffix: str) -> str:
    if not suffix:
        return tag
    parts = tag.split(" ", 1)
    return f"{parts[0]}{suffix}" + (f" {parts[1]}" if len(parts) > 1 else "")


def infogcn_forward(
    model: nn.Module, x: torch.Tensor, y_b: torch.Tensor
) -> torch.Tensor:
    y_hat, _ = model(x)
    return y_hat


def blockgcn_forward(
    model: nn.Module, x: torch.Tensor, y_b: torch.Tensor
) -> torch.Tensor:
    y_onehot = F.one_hot(y_b, num_classes=2).float()
    y_hat, _ = model(x, y_onehot, x)
    return y_hat


def skateformer_forward(
    model: nn.Module, x: torch.Tensor, y_b: torch.Tensor
) -> torch.Tensor:
    B, C, T, V, M = x.shape
    index_t = torch.zeros(B, T, dtype=torch.long, device=x.device)
    return model(x, index_t)


def hyperformer_forward(
    model: nn.Module, x: torch.Tensor, y_b: torch.Tensor
) -> torch.Tensor:
    """HyperFormer forward(x, y) -> (logits, y); use logits only."""
    logits, _ = model(x, y_b)
    return logits


# (display name, build_fn, checkpoint prefix, tag suffix e.g. "b" for B1b, forward_fn or None)
MODEL_CONFIGS: List[Tuple[str, Callable, str, str, Optional[Callable]]] = [
    ("ST-GCN",        build_stgcn_model,        "",               "",  None),
    ("CTR-GCN",       build_ctrgcn_model,       "ctrgcn_",        "b", None),
    ("InfoGCN",       build_infogcn_model,      "infogcn_",       "c", infogcn_forward),
    ("BlockGCN",      build_blockgcn_model,     "blockgcn_",      "d", blockgcn_forward),
    ("HD-GCN",        build_hdgcn_model,        "hdgcn_",         "e", None),
    ("ProtoGCN",      build_protogcn_model,     "protogcn_",      "f", None),
    ("SkateFormer",   build_skateformer_model,  "skateformer_",   "g", skateformer_forward),
    ("HyperFormer",    build_hyperformer_model,   "hyperformer_",    "h", hyperformer_forward),
    ("FreqMixFormer",  build_freqmixformer_model, "freqmixformer_",  "i", None),
]


def print_confusion(
    tag: str,
    ckpt: str,
    ood_d: str,
    ood_l: str,
    build_fn: Callable,
    forward_fn: Optional[Callable] = None,
) -> bool:
    """Print confusion matrix. Returns True if evaluated, False if skipped."""
    ckpt = os.path.abspath(ckpt)
    ood_d = os.path.abspath(ood_d)
    ood_l = os.path.abspath(ood_l)

    print(f"\n{'=' * 70}")
    print(tag)
    print(f"{'=' * 70}")
    print(f"  Checkpoint : {ckpt}")
    print(f"  Test data  : {ood_d}")
    print(f"  Test labels: {ood_l}")

    if not os.path.exists(ckpt):
        print(f"  [SKIP] Checkpoint not found: {ckpt}")
        return False

    missing_test = [path for path in (ood_d, ood_l) if not os.path.exists(path)]
    if missing_test:
        for path in missing_test:
            print(f"  [SKIP] File not found: {path}")
        return False

    model = build_fn(checkpoint_path=ckpt, device=DEVICE)
    model.eval()

    preds_all, labels_all = [], []
    with torch.no_grad():
        for X_b, y_b in DataLoader(load_ds(ood_d, ood_l), batch_size=TRAIN_BATCH_SIZE):
            X_dev = X_b.to(DEVICE)
            if forward_fn is None:
                logits = model(X_dev)
            else:
                logits = forward_fn(model, X_dev, y_b.to(DEVICE))
            preds_all.append(logits.argmax(dim=1).cpu().numpy())
            labels_all.append(y_b.numpy())

    preds = np.concatenate(preds_all)
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
        f"  [{tag}] Majority class prediction check: "
        f"model predicted correct={pred_c} times, incorrect={pred_i} times"
    )

    model.cpu()
    del model
    torch.cuda.empty_cache()
    return True


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Output dir: {os.path.abspath(OUTPUT_DIR)}")

    evaluated = 0
    skipped = 0

    for model_name, build_fn, ckpt_prefix, tag_suffix, forward_fn in MODEL_CONFIGS:
        print(f"\n\n{'#' * 70}")
        print(f"# {model_name}")
        print(f"{'#' * 70}")

        for tag, ckpt_name, te_data, te_labels in OOD_EXPERIMENTS:
            full_tag = add_tag_suffix(tag, tag_suffix)
            ckpt = p(f"{ckpt_prefix}{ckpt_name}")
            if print_confusion(
                full_tag, ckpt, p(te_data), p(te_labels), build_fn, forward_fn
            ):
                evaluated += 1
            else:
                skipped += 1

    print(f"\n{'=' * 70}")
    print(f"Done — {evaluated} confusion matrices printed, {skipped} skipped.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
