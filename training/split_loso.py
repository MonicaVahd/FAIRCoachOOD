"""
training/split_loso.py — Leave-One-Subject-Out (LOSO) cross-validation splits.

EC3D  : 4 subjects (names read from configs/ec3d_subjects.yaml)  → 4 rounds
REHAB24: 9 subjects (derived from person_id in CSV) → 9 rounds
UI-PRMD: 10 subjects (from uiprmd_subjects.npy)    → 10 rounds

For each round:
  - test  = all reps of the held-out subject
  - train+val pool = all reps of the remaining subjects
  - 90 / 10 split of pool → train / val   (seed 42)

Output files per round (in output/):
  ec3d_loso_round{n}_train_data.npy   (N_train, 3, 100, 18, 1)
  ec3d_loso_round{n}_train_labels.npy (N_train,)
  ec3d_loso_round{n}_val_data.npy     (N_val,   3, 100, 18, 1)
  ec3d_loso_round{n}_val_labels.npy   (N_val,)
  ec3d_loso_round{n}_test_data.npy    (N_test,  3, 100, 18, 1)
  ec3d_loso_round{n}_test_labels.npy  (N_test,)
  (and the same pattern for rehab24_loso_round{n}_* and uiprmd_loso_round{n}_*)

Usage
-----
    cd <project_root>
    python training/split_loso.py

Prerequisites
-------------
    This script reads the raw pickle / CSV directly to preserve per-subject
    identity, which is stripped from the combined .npy files produced by
    ec3d_to_stgcn.py and rehab24_to_stgcn.py.
    For UI-PRMD, run uiprmd_to_stgcn.py first to produce uiprmd_*.npy.
"""

import sys
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.common import normalize_xy, pad_or_trim, ntu25_to_openpose18, rehab26_to_openpose18
from configs.constants import (
    EC3D_PICKLE, REHAB24_CSV, REHAB24_DATA_DIR,
    STGCN_WINDOW, OUTPUT_DIR, TRAIN_SEED as RANDOM_SEED,
)

from configs.subjects import load_ec3d_subjects
EC3D_SUBJECTS = load_ec3d_subjects()
VAL_FRACTION  = 0.10   # fraction of train+val pool used for validation


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_ec3d_by_subject() -> dict:
    """Return {subject_name: (X, y)} where X: (N, 3, T, 18, 1), y: (N,)."""
    with open(EC3D_PICKLE, "rb") as f:
        data = pickle.load(f)

    labels = np.array(data["labels"])   # (N_frames, 5): action, subject, ex_type, rep, frame
    poses  = np.array(data["poses"])    # (N_frames, 3, 25)

    squat_mask = labels[:, 0] == "SQUAT"
    sq_labels  = labels[squat_mask]
    sq_poses   = poses[squat_mask]

    seq_dict: dict = defaultdict(list)
    for i in range(len(sq_labels)):
        subject  = sq_labels[i, 1]
        ex_type  = sq_labels[i, 2]
        rep      = sq_labels[i, 3]
        frame_id = int(sq_labels[i, 4])
        key = (subject, ex_type, rep)
        seq_dict[key].append((frame_id, sq_poses[i]))

    by_subject: dict = defaultdict(lambda: ([], []))
    for (subject, ex_type, _rep), frames in seq_dict.items():
        frames.sort(key=lambda x: x[0])
        seq = np.stack([p for _, p in frames], axis=0)  # (T, 3, 25)

        seq18   = ntu25_to_openpose18(seq)              # (T, 3, 18)

        # center skeleton on OpenPose RHip (joint 8)
        seq18 = seq18 - seq18[:, :, 8:9]

        # scale by mean torso height (neck joint 1 -> hip joint 8)
        torso_height = np.linalg.norm(
            seq18[:, :, 1] - seq18[:, :, 8], axis=1
        ).mean()
        if torso_height > 1e-6:
            seq18 = seq18 / torso_height

        seq_fix = pad_or_trim(seq18, STGCN_WINDOW)      # (T, 3, 18)
        arr     = np.transpose(seq_fix, (1, 0, 2))       # (3, T, 18)
        arr     = normalize_xy(arr)
        arr     = arr[:, :, :, np.newaxis]               # (3, T, 18, 1)

        label = 1 if ex_type == "1" else 0
        by_subject[subject][0].append(arr)
        by_subject[subject][1].append(label)

    return {
        subj: (
            np.stack(X_list, axis=0).astype(np.float32),
            np.array(y_list, dtype=np.int64),
        )
        for subj, (X_list, y_list) in by_subject.items()
    }


def load_rehab24_by_subject() -> dict:
    """Return {person_id: (X, y)} where X: (N, 3, T, 18, 1), y: (N,)."""
    df = pd.read_csv(REHAB24_CSV, sep=';')
    df.columns = df.columns.str.strip()
    squat_df = df[(df['exercise_id'] == 6) & (df['mocap_erroneous'] == 0)].copy()

    npy_cache: dict = {}
    by_subject: dict = defaultdict(lambda: ([], []))
    skipped = 0

    for _, row in squat_df.iterrows():
        vid    = row['video_id']
        person = row['person_id']
        start  = int(row['first_frame'])
        end    = int(row['last_frame'])
        label  = int(row['correctness'])

        npy_file = os.path.join(REHAB24_DATA_DIR, f"{vid}-c17-30fps.npy")
        if not os.path.exists(npy_file):
            skipped += 1
            continue

        if npy_file not in npy_cache:
            npy_cache[npy_file] = np.load(npy_file).astype(np.float32)

        full_seq = npy_cache[npy_file]
        end = min(end, len(full_seq))
        if end <= start:
            skipped += 1
            continue

        rep     = full_seq[start:end]                       # (t, 26, 2)
        rep18   = rehab26_to_openpose18(rep)                # (t, 3, 18)

        # center skeleton on OpenPose RHip (joint 8)
        rep18 = rep18 - rep18[:, :, 8:9]

        # scale by mean torso height (neck joint 1 -> hip joint 8)
        torso_height = np.linalg.norm(
            rep18[:, :, 1] - rep18[:, :, 8], axis=1
        ).mean()
        if torso_height > 1e-6:
            rep18 = rep18 / torso_height

        rep_fix = pad_or_trim(rep18, STGCN_WINDOW)          # (T, 3, 18)
        arr     = np.transpose(rep_fix, (1, 0, 2))           # (3, T, 18)
        arr     = normalize_xy(arr)
        arr     = arr[:, :, :, np.newaxis]                   # (3, T, 18, 1)

        by_subject[person][0].append(arr)
        by_subject[person][1].append(label)

    if skipped:
        print(f"  REHAB24: skipped {skipped} rows (missing file or bad frames)")

    return {
        person: (
            np.stack(X_list, axis=0).astype(np.float32),
            np.array(y_list, dtype=np.int64),
        )
        for person, (X_list, y_list) in by_subject.items()
    }


def load_uiprmd_by_subject() -> dict:
    """Return {subject_id: (X, y)} from preprocessed UI-PRMD .npy files."""
    data_path     = os.path.join(OUTPUT_DIR, "uiprmd_data.npy")
    labels_path   = os.path.join(OUTPUT_DIR, "uiprmd_labels.npy")
    subjects_path = os.path.join(OUTPUT_DIR, "uiprmd_subjects.npy")

    for path in (data_path, labels_path, subjects_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path} — run uiprmd_to_stgcn.py first"
            )

    X        = np.load(data_path)
    y        = np.load(labels_path)
    subjects = np.load(subjects_path)

    by_subject: dict = {}
    for sid in np.unique(subjects):
        mask = subjects == sid
        by_subject[int(sid)] = (X[mask], y[mask])
    return by_subject


# ── LOSO split helper ─────────────────────────────────────────────────────────

def pool_train_val(subjects, all_data, test_subject, val_fraction, seed):
    """
    Concatenate data for all subjects except test_subject, then split 90/10.

    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    pool_X, pool_y = [], []
    for s in subjects:
        if s == test_subject:
            continue
        X, y = all_data[s]
        pool_X.append(X)
        pool_y.append(y)

    X_pool = np.concatenate(pool_X, axis=0)
    y_pool = np.concatenate(pool_y, axis=0)

    N   = len(y_pool)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    n_val     = max(1, int(val_fraction * N))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    X_test, y_test = all_data[test_subject]

    return (
        X_pool[train_idx], y_pool[train_idx],
        X_pool[val_idx],   y_pool[val_idx],
        X_test,            y_test,
    )


def save_round(prefix, round_n, X_tr, y_tr, X_val, y_val, X_te, y_te):
    """Save all six arrays for one LOSO round."""
    base = os.path.join(OUTPUT_DIR, f"{prefix}_loso_round{round_n}")
    np.save(f"{base}_train_data.npy",   X_tr)
    np.save(f"{base}_train_labels.npy", y_tr)
    np.save(f"{base}_val_data.npy",     X_val)
    np.save(f"{base}_val_labels.npy",   y_val)
    np.save(f"{base}_test_data.npy",    X_te)
    np.save(f"{base}_test_labels.npy",  y_te)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── EC3D LOSO (4 rounds) ──────────────────────────────────────────────────
    print("=" * 60)
    print("EC3D LOSO — 4 subjects, 4 rounds")
    print("=" * 60)

    ec3d_data = load_ec3d_by_subject()

    found = set(ec3d_data.keys())
    for s in EC3D_SUBJECTS:
        if s not in found:
            print(f"  [WARNING] EC3D subject '{s}' not found in pickle")

    subjects_ec3d = [s for s in EC3D_SUBJECTS if s in found]

    for round_n, test_subject in enumerate(subjects_ec3d, start=1):
        X_tr, y_tr, X_val, y_val, X_te, y_te = pool_train_val(
            subjects_ec3d, ec3d_data, test_subject, VAL_FRACTION, RANDOM_SEED,
        )
        save_round("ec3d", round_n, X_tr, y_tr, X_val, y_val, X_te, y_te)
        print(
            f"  Round {round_n} | test={test_subject:<10} "
            f"| train={len(y_tr):>4}  val={len(y_val):>4}  test={len(y_te):>4}"
        )

    # ── REHAB24 LOSO (10 rounds) ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("REHAB24 LOSO — 9 subjects, 9 rounds")
    print("=" * 60)

    rehab_data    = load_rehab24_by_subject()
    subjects_rehab = sorted(rehab_data.keys())

    if len(subjects_rehab) != 9:
        print(
            f"  [WARNING] Expected 9 subjects, found {len(subjects_rehab)}: "
            f"{subjects_rehab}"
        )

    for round_n, test_subject in enumerate(subjects_rehab, start=1):
        X_tr, y_tr, X_val, y_val, X_te, y_te = pool_train_val(
            subjects_rehab, rehab_data, test_subject, VAL_FRACTION, RANDOM_SEED,
        )
        save_round("rehab24", round_n, X_tr, y_tr, X_val, y_val, X_te, y_te)
        print(
            f"  Round {round_n} | test={str(test_subject):<12} "
            f"| train={len(y_tr):>4}  val={len(y_val):>4}  test={len(y_te):>4}"
        )

    # ── UI-PRMD LOSO (10 rounds) ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("UI-PRMD LOSO — 10 subjects, 10 rounds")
    print("=" * 60)

    try:
        uiprmd_data = load_uiprmd_by_subject()
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
    else:
        subjects_uiprmd = sorted(uiprmd_data.keys())

        if len(subjects_uiprmd) != 10:
            print(
                f"  [WARNING] Expected 10 subjects, found {len(subjects_uiprmd)}: "
                f"{subjects_uiprmd}"
            )

        for round_n, test_subject in enumerate(subjects_uiprmd, start=1):
            X_tr, y_tr, X_val, y_val, X_te, y_te = pool_train_val(
                subjects_uiprmd, uiprmd_data, test_subject,
                VAL_FRACTION, RANDOM_SEED,
            )
            save_round("uiprmd", round_n, X_tr, y_tr, X_val, y_val, X_te, y_te)
            print(
                f"  Round {round_n} | test=s{test_subject:02d}       "
                f"| train={len(y_tr):>4}  val={len(y_val):>4}  "
                f"test={len(y_te):>4}"
            )

    print("\nDone. All LOSO splits saved to output/")


if __name__ == "__main__":
    main()
