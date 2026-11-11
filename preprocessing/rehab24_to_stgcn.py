"""
REHAB24 Ex6 (squat) repetitions -> ST-GCN training data.

CSV columns (semicolon-separated):
  video_id, repetition_number, exercise_id, person_id,
  first_frame, last_frame, ..., mocap_erroneous, ..., correctness

We use c17-30fps .npy files (shape: T, 26, 2  -> x,y pixels).

REHAB24 26-joint -> OpenPose 18-joint mapping (14 real, 4 face=zeros):
  Head(4)->nose, Neck(3)->neck,
  RightShoulder(11)->Rshoulder, RightArm(12)->Relbow, RightHand(14)->Rwrist,
  LeftShoulder(6)->Lshoulder,  LeftArm(7)->Lelbow,   LeftHand(9)->Lwrist,
  RightUpLeg(21)->Rhip, RightLeg(22)->Rknee, RightFoot(23)->Rankle,
  LeftUpLeg(16)->Lhip,  LeftLeg(17)->Lknee,  LeftFoot(18)->Lankle,
  joints 14-17 (eyes/ears) -> zeros
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.common import normalize_xy, pad_or_trim, rehab26_to_openpose18
from configs.constants import REHAB24_CSV, REHAB24_DATA_DIR, STGCN_WINDOW, OUTPUT_DIR


def main():
    OUT_DATA   = os.path.join(OUTPUT_DIR, "rehab24_train_data.npy")
    OUT_LABELS = os.path.join(OUTPUT_DIR, "rehab24_train_labels.npy")

    # 1) Load and parse CSV (semicolon-separated)
    df = pd.read_csv(REHAB24_CSV, sep=';')
    df.columns = df.columns.str.strip()

    # 2) Filter: Ex6 squats only, skip erroneous mocap rows
    squat_df = df[(df['exercise_id'] == 6) & (df['mocap_erroneous'] == 0)].copy()
    print(f"Squat repetitions after filtering: {len(squat_df)}")

    # 3) Cache loaded .npy files to avoid reloading same file repeatedly
    npy_cache = {}

    X_list, y_list = [], []
    skipped = 0

    for _, row in squat_df.iterrows():
        vid   = row['video_id']          # e.g. 'PM_008'
        start = int(row['first_frame'])
        end   = int(row['last_frame'])
        label = int(row['correctness'])  # 1=correct, 0=incorrect

        npy_file = os.path.join(REHAB24_DATA_DIR, f"{vid}-c17-30fps.npy")
        if not os.path.exists(npy_file):
            skipped += 1
            continue

        if npy_file not in npy_cache:
            npy_cache[npy_file] = np.load(npy_file).astype(np.float32)  # (T, 26, 2)

        full_seq = npy_cache[npy_file]

        # guard against out-of-bounds frame indices
        end = min(end, len(full_seq))
        if end <= start:
            skipped += 1
            continue

        rep = full_seq[start:end]                      # (t, 26, 2)
        rep18 = rehab26_to_openpose18(rep)             # (t, 3, 18)

        # center skeleton on OpenPose RHip (joint 8)
        rep18 = rep18 - rep18[:, :, 8:9]

        # scale by mean torso height (neck joint 1 -> hip joint 8)
        torso_height = np.linalg.norm(
            rep18[:, :, 1] - rep18[:, :, 8], axis=1
        ).mean()
        if torso_height > 1e-6:
            rep18 = rep18 / torso_height

        rep_fixed = pad_or_trim(rep18, STGCN_WINDOW)   # (T, 3, 18)

        arr = np.transpose(rep_fixed, (1, 0, 2))   # (3, T, 18)
        arr = normalize_xy(arr)                     # x,y -> [0, 1]
        arr = arr[:, :, :, np.newaxis]              # (3, T, 18, 1)

        X_list.append(arr)
        y_list.append(label)

    print(f"Skipped (file missing or bad frames): {skipped}")

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, 3, T, 18, 1)
    y = np.array(y_list, dtype=np.int64)

    print(f"Sequences: {len(X)} total")
    print(f"  Correct  (label=1): {int(y.sum())}")
    print(f"  Incorrect(label=0): {int((y == 0).sum())}")
    print(f"X shape (N, C, T, V, M): {X.shape}")

    np.save(OUT_DATA, X)
    np.save(OUT_LABELS, y)
    print(f"\nSaved data   -> {OUT_DATA}")
    print(f"Saved labels -> {OUT_LABELS}")


if __name__ == "__main__":
    main()
