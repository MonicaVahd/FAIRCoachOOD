"""
EC3D squat sequences -> ST-GCN training data.

EC3D label columns: [action, subject, exercise_type, repetition, frame_id]
  exercise_type == '1' => correct squat
  exercise_type != '1' => incorrect squat (various error categories)

Pose shape per frame: (3, 25)  (x,y,z x 25 NTU joints)

We select 18 joints from the 25 NTU joints to match the OpenPose layout
used by RTMPose (so inference on your own videos works directly).

NTU (0-indexed) -> OpenPose position mapping:
  3=head -> 0(nose), 2=neck -> 1(neck),
  8=R.shoulder->2, 9=R.elbow->3, 10=R.wrist->4,
  4=L.shoulder->5, 5=L.elbow->6,  6=L.wrist->7,
  16=R.hip->8,    17=R.knee->9,   18=R.ankle->10,
  12=L.hip->11,   13=L.knee->12,  14=L.ankle->13,
  joints 14-17 (eyes/ears) padded with zeros (no face joints in NTU).
"""

import sys
import os
import pickle
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.common import normalize_xy, pad_or_trim, ntu25_to_openpose18
from configs.constants import EC3D_PICKLE, STGCN_WINDOW, OUTPUT_DIR


def main():
    OUT_DATA   = os.path.join(OUTPUT_DIR, "ec3d_train_data.npy")
    OUT_LABELS = os.path.join(OUTPUT_DIR, "ec3d_train_labels.npy")

    # 1) Load pickle
    with open(EC3D_PICKLE, "rb") as f:
        data = pickle.load(f)

    labels = np.array(data["labels"])   # (N_frames, 5)
    poses  = np.array(data["poses"])    # (N_frames, 3, 25)

    # 2) Keep squat rows only
    squat_mask = labels[:, 0] == "SQUAT"
    sq_labels = labels[squat_mask]
    sq_poses  = poses[squat_mask]

    # 3) Group frames into sequences by (action, subject, exercise_type, repetition)
    seq_dict = defaultdict(list)
    for i in range(len(sq_labels)):
        key = tuple(sq_labels[i, :4])        # e.g. ('SQUAT', '<subject>', '1', '1')
        frame_id = int(sq_labels[i, 4])
        seq_dict[key].append((frame_id, sq_poses[i]))

    # sort each sequence by frame_id and stack
    sequences = {}
    for k, frames in seq_dict.items():
        frames.sort(key=lambda x: x[0])
        sequences[k] = np.stack([p for _, p in frames], axis=0)  # (T, 3, 25)

    # 4) Assign labels:  exercise_type == '1' -> 1 (correct), else -> 0 (incorrect)
    X_list, y_list = [], []
    for key, seq in sequences.items():
        exercise_type = key[2]
        label = 1 if exercise_type == "1" else 0

        # map 25 NTU joints -> 18 openpose joints
        seq18 = ntu25_to_openpose18(seq)               # (T, 3, 18)

        # center skeleton on OpenPose RHip (joint 8)
        seq18 = seq18 - seq18[:, :, 8:9]

        # scale by mean torso height (neck joint 1 -> hip joint 8)
        torso_height = np.linalg.norm(
            seq18[:, :, 1] - seq18[:, :, 8], axis=1
        ).mean()
        if torso_height > 1e-6:
            seq18 = seq18 / torso_height

        # pad or trim to fixed window size
        seq_fixed = pad_or_trim(seq18, STGCN_WINDOW)   # (T, 3, 18)

        # reshape from (T, 3, 18) -> (C=3, T, V=18, M=1)
        arr = np.transpose(seq_fixed, (1, 0, 2))       # (3, T, 18)
        arr = normalize_xy(arr)                         # x,y -> [0, 1]
        arr = arr[:, :, :, np.newaxis]                  # (3, T, 18, 1)

        X_list.append(arr)
        y_list.append(label)

    # 5) Stack into dataset tensors
    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, 3, T, 18, 1)
    y = np.array(y_list, dtype=np.int64)              # (N,)

    print(f"Sequences: {len(sequences)} total")
    print(f"  Correct  (label=1): {int(y.sum())}")
    print(f"  Incorrect(label=0): {int((y == 0).sum())}")
    print(f"X shape (N, C, T, V, M): {X.shape}")
    print(f"y shape (N,): {y.shape}")

    # 6) Save
    np.save(OUT_DATA, X)
    np.save(OUT_LABELS, y)
    print(f"\nSaved data  -> {OUT_DATA}")
    print(f"Saved labels-> {OUT_LABELS}")


if __name__ == "__main__":
    main()
