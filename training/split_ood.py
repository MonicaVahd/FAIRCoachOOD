"""
training/split_ood.py — OOD (out-of-distribution) test set creation.

Saves each complete dataset as the OOD test set for every cross-dataset
evaluation direction.

No shuffling or splitting is performed; every rep is retained so the full
cross-dataset distribution is captured.

Outputs
-------
    output/ec3d_ood_test_data.npy       — all EC3D sequences  (N, 3, T, 18, 1)
    output/ec3d_ood_test_labels.npy     — all EC3D labels      (N,)
    output/rehab24_ood_test_data.npy    — all REHAB24 sequences
    output/rehab24_ood_test_labels.npy  — all REHAB24 labels
    output/custom_ood_test_data.npy     — all custom squat sequences
    output/custom_ood_test_labels.npy   — all custom squat labels
    output/uiprmd_ood_test_data.npy     — all UI-PRMD sequences
    output/uiprmd_ood_test_labels.npy   — all UI-PRMD labels

Interpretation
--------------
    Train on EC3D           → OOD test on rehab24_ood_test + custom_ood_test
    Train on REHAB24        → OOD test on ec3d_ood_test    + custom_ood_test
    Train on Custom         → OOD test on ec3d_ood_test    + rehab24_ood_test
    Train on EC3D+REHAB24   → OOD test on custom_ood_test
    Train on EC3D+Custom    → OOD test on rehab24_ood_test
    Train on REHAB24+Custom → OOD test on ec3d_ood_test
    Train on UI-PRMD        → OOD test on ec3d_ood_test + rehab24_ood_test + custom_ood_test
    Train on EC3D+UI-PRMD   → OOD test on rehab24_ood_test + custom_ood_test
    Train on REHAB24+UI-PRMD → OOD test on ec3d_ood_test + custom_ood_test
    Train on Custom+UI-PRMD  → OOD test on ec3d_ood_test + rehab24_ood_test
    Train on EC3D+REHAB24+Custom → OOD test on uiprmd_ood_test

Usage
-----
    cd <project_root>
    python training/split_ood.py

Prerequisites
-------------
    ec3d_to_stgcn.py, rehab24_to_stgcn.py, and custom_squat_to_stgcn.py
    must have been run first.
    uiprmd_to_stgcn.py must have been run for UI-PRMD OOD test sets.
"""

import sys
import os
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.constants import OUTPUT_DIR


def save_ood(name: str, src_data: str, src_labels: str,
             dst_data: str, dst_labels: str) -> None:
    """
    Copy an existing full-dataset .npy pair to OOD output paths.

    Simply copies (or re-saves) the files so the OOD filenames are
    self-documenting and decoupled from the IID split filenames.
    """
    if not os.path.exists(src_data):
        print(f"[ERROR] {name}: source data not found: {src_data}")
        print(f"        Run the corresponding *_to_stgcn.py script first.")
        return

    X = np.load(src_data)
    y = np.load(src_labels)

    np.save(dst_data,   X)
    np.save(dst_labels, y)

    print(f"\n{name} OOD test set")
    print(f"  {len(y)} sequences  "
          f"(correct={int(y.sum())}, incorrect={int((y == 0).sum())})")
    print(f"  Saved data   -> {dst_data}")
    print(f"  Saved labels -> {dst_labels}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== OOD Test Set Creation ===")

    # EC3D → OOD test for models trained on REHAB24, Custom, or REHAB24+Custom
    save_ood(
        name="EC3D (OOD test for REHAB24 / Custom / REHAB24+Custom-trained models)",
        src_data=os.path.join(OUTPUT_DIR, "ec3d_train_data.npy"),
        src_labels=os.path.join(OUTPUT_DIR, "ec3d_train_labels.npy"),
        dst_data=os.path.join(OUTPUT_DIR, "ec3d_ood_test_data.npy"),
        dst_labels=os.path.join(OUTPUT_DIR, "ec3d_ood_test_labels.npy"),
    )

    # REHAB24 → OOD test for a model trained on EC3D, Custom, or REHAB24+Custom
    save_ood(
        name="REHAB24 (OOD test for EC3D / Custom / REHAB24+Custom-trained models)",
        src_data=os.path.join(OUTPUT_DIR, "rehab24_train_data.npy"),
        src_labels=os.path.join(OUTPUT_DIR, "rehab24_train_labels.npy"),
        dst_data=os.path.join(OUTPUT_DIR, "rehab24_ood_test_data.npy"),
        dst_labels=os.path.join(OUTPUT_DIR, "rehab24_ood_test_labels.npy"),
    )

    # Custom → OOD test for models trained on EC3D, REHAB24, or EC3D+REHAB24
    save_ood(
        name="Custom (OOD test for EC3D / REHAB24 / EC3D+REHAB24-trained models)",
        src_data=os.path.join(OUTPUT_DIR, "custom_squat_train_data.npy"),
        src_labels=os.path.join(OUTPUT_DIR, "custom_squat_train_labels.npy"),
        dst_data=os.path.join(OUTPUT_DIR, "custom_ood_test_data.npy"),
        dst_labels=os.path.join(OUTPUT_DIR, "custom_ood_test_labels.npy"),
    )

    # UI-PRMD → OOD test for models trained on EC3D, REHAB24, Custom, or combos
    save_ood(
        name="UI-PRMD (OOD test for EC3D / REHAB24 / Custom-trained models)",
        src_data=os.path.join(OUTPUT_DIR, "uiprmd_data.npy"),
        src_labels=os.path.join(OUTPUT_DIR, "uiprmd_labels.npy"),
        dst_data=os.path.join(OUTPUT_DIR, "uiprmd_ood_test_data.npy"),
        dst_labels=os.path.join(OUTPUT_DIR, "uiprmd_ood_test_labels.npy"),
    )

    print("\nOOD evaluation map (all directions now covered):")
    print("  Train EC3D           → test on rehab24_ood_test + custom_ood_test")
    print("  Train REHAB24        → test on ec3d_ood_test    + custom_ood_test")
    print("  Train Custom         → test on ec3d_ood_test    + rehab24_ood_test")
    print("  Train EC3D+REHAB24   → test on custom_ood_test")
    print("  Train EC3D+Custom    → test on rehab24_ood_test")
    print("  Train REHAB24+Custom → test on ec3d_ood_test")
    print("  Train UI-PRMD        → test on ec3d_ood_test + rehab24_ood_test + custom_ood_test")
    print("  Train EC3D+UI-PRMD   → test on rehab24_ood_test + custom_ood_test")
    print("  Train REHAB24+UI-PRMD → test on ec3d_ood_test + custom_ood_test")
    print("  Train Custom+UI-PRMD  → test on ec3d_ood_test + rehab24_ood_test")
    print("  Train EC3D+REHAB24+Custom → test on uiprmd_ood_test")
    print("\nDone.  All OOD test sets saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
