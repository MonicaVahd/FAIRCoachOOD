"""
UI-PRMD deep squat (m01) sequences -> ST-GCN training data.

Data layout
-----------
  datasets/ui_prmd/correct/   — m01_s01_e01_positions.txt etc.       (label=1)
  datasets/ui_prmd/incorrect/ — m01_s01_e01_positions_inc.txt etc.  (label=0)

Each .txt file is one sequence.  Each line is one frame of 66 comma-separated
values (22 joints x 3 coordinates x,y,z).

UI-PRMD Kinect joint order (22 joints):
  0 Waist, 1 Spine, 2 Chest, 3 Neck, 4 Head, ...

Processing (mirrors ec3d_to_stgcn.py pipeline, adapted for 22 joints):
  1. Load frames -> (T, 22, 3)
  2. Map 22 Kinect joints -> 18 OpenPose joints
  3. Hip-center on OpenPose RHip (joint 8)
  4. Scale by mean torso height (neck joint 1 -> hip joint 8)
  5. Pad or interpolate to STGCN_WINDOW=100 frames
  6. Reshape to ST-GCN format (3, 100, 18, 1)

Outputs
-------
  output/uiprmd_data.npy      — (N, 3, 100, 18, 1)
  output/uiprmd_labels.npy    — (N,)
  output/uiprmd_subjects.npy  — (N,) subject IDs parsed from filename (s01->1)

Usage
-----
    cd <project_root>
    python training/uiprmd_to_stgcn.py
"""

import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.constants import STGCN_WINDOW, OUTPUT_DIR
from preprocessing.common import normalize_xy

CORRECT_DIR   = os.path.join("datasets", "ui_prmd", "correct")
INCORRECT_DIR = os.path.join("datasets", "ui_prmd", "incorrect")

NUM_JOINTS = 22
NUM_COORDS = 3
OPENPOSE_JOINTS = 18

# Kinect joint index for each OpenPose position (0..17)
KINECT_TO_OPENPOSE = [3, 2, 8, 9, 10, 4, 5, 6, 16, 17, 18, 12, 13, 14, 1, 0, 0, 0]


def load_sequence(filepath: str) -> np.ndarray:
    """Load one .txt sequence -> (T, 22, 3)."""
    frames = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = np.fromstring(line, sep=",", dtype=np.float32)
            if vals.size != NUM_JOINTS * NUM_COORDS:
                raise ValueError(
                    f"Expected {NUM_JOINTS * NUM_COORDS} values, "
                    f"got {vals.size} in {filepath}"
                )
            frames.append(vals.reshape(NUM_JOINTS, NUM_COORDS))
    if not frames:
        raise ValueError(f"No frames found in {filepath}")
    return np.stack(frames, axis=0)


def kinect22_to_openpose18(seq: np.ndarray) -> np.ndarray:
    """
    Map 22 Kinect joints to 18 OpenPose joints (same style as ntu25_to_openpose18).
    seq: (T, 22, 3) -> (T, 18, 3)
    """
    T = seq.shape[0]
    out = np.zeros((T, OPENPOSE_JOINTS, NUM_COORDS), dtype=np.float32)
    for op_idx, kinect_idx in enumerate(KINECT_TO_OPENPOSE):
        out[:, op_idx, :] = seq[:, kinect_idx, :]
    return out


def pad_or_interpolate(seq: np.ndarray, target_T: int) -> np.ndarray:
    """
    Resample sequence to exactly target_T frames.
    Short sequences are interpolated (stretched); long sequences are downsampled.
    seq: (t, V, 3) -> (target_T, V, 3)
    """
    t, n_joints, n_coords = seq.shape
    if t == target_T:
        return seq.astype(np.float32)
    if t == 1:
        return np.repeat(seq, target_T, axis=0).astype(np.float32)

    x_old = np.linspace(0.0, 1.0, t)
    x_new = np.linspace(0.0, 1.0, target_T)
    out = np.zeros((target_T, n_joints, n_coords), dtype=np.float32)
    for j in range(n_joints):
        for c in range(n_coords):
            out[:, j, c] = np.interp(x_new, x_old, seq[:, j, c])
    return out


def parse_subject_id(filename: str) -> int:
    """Extract subject ID from filename, e.g. m01_s03_e02 -> 3."""
    match = re.search(r"_s(\d+)_", filename)
    if not match:
        raise ValueError(f"Cannot parse subject ID from filename: {filename}")
    return int(match.group(1))


def collect_m01_files() -> list:
    """Return list of (filepath, label) for all m01 squat files."""
    entries = []
    for directory, label in ((CORRECT_DIR, 1), (INCORRECT_DIR, 0)):
        if not os.path.isdir(directory):
            print(f"  [MISSING] {directory}")
            continue
        for fname in sorted(os.listdir(directory)):
            if fname.startswith("m01") and fname.endswith(".txt"):
                entries.append((os.path.join(directory, fname), label))
    return entries


def sequence_to_stgcn(seq: np.ndarray) -> np.ndarray:
    """Full processing pipeline for one sequence -> (3, T, 18, 1)."""
    seq = kinect22_to_openpose18(seq)

    # center skeleton on OpenPose RHip (joint 8)
    seq = seq - seq[:, 8:9, :]

    # scale by mean torso height (neck joint 1 -> hip joint 8)
    torso_height = np.linalg.norm(
        seq[:, 1, :] - seq[:, 8, :], axis=1
    ).mean()
    if torso_height > 1e-6:
        seq = seq / torso_height

    seq = pad_or_interpolate(seq, STGCN_WINDOW)
    arr = np.transpose(seq, (2, 0, 1))          # (3, T, 18)
    arr = normalize_xy(arr)                     # x,y -> [0, 1]
    return arr[:, :, :, np.newaxis].astype(np.float32)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_data     = os.path.join(OUTPUT_DIR, "uiprmd_data.npy")
    out_labels   = os.path.join(OUTPUT_DIR, "uiprmd_labels.npy")
    out_subjects = os.path.join(OUTPUT_DIR, "uiprmd_subjects.npy")

    files = collect_m01_files()
    if not files:
        print("No m01 squat files found. Check datasets/ui_prmd/ paths.")
        return

    X_list, y_list, subj_list = [], [], []
    skipped = 0

    for filepath, label in files:
        try:
            seq = load_sequence(filepath)
            arr = sequence_to_stgcn(seq)
            subject_id = parse_subject_id(os.path.basename(filepath))

            X_list.append(arr)
            y_list.append(label)
            subj_list.append(subject_id)
        except (ValueError, OSError) as e:
            print(f"  [SKIP] {filepath}: {e}")
            skipped += 1

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(subj_list, dtype=np.int64)

    subject_counts = Counter(subj_list)

    print(f"Files found   : {len(files)}")
    print(f"Skipped       : {skipped}")
    print(f"Sequences     : {len(X)} total")
    print(f"  Correct  (label=1): {int((y == 1).sum())}")
    print(f"  Incorrect(label=0): {int((y == 0).sum())}")
    print(f"X shape (N, C, T, V, M): {X.shape}")
    print(f"y shape (N,): {y.shape}")
    print(f"subjects shape (N,): {subjects.shape}")
    print("Per-subject counts:")
    for sid in sorted(subject_counts):
        n_correct   = int(((subjects == sid) & (y == 1)).sum())
        n_incorrect = int(((subjects == sid) & (y == 0)).sum())
        print(f"  s{sid:02d}: {subject_counts[sid]} total  "
              f"(correct={n_correct}, incorrect={n_incorrect})")

    np.save(out_data, X)
    np.save(out_labels, y)
    np.save(out_subjects, subjects)
    print(f"\nSaved data     -> {out_data}")
    print(f"Saved labels   -> {out_labels}")
    print(f"Saved subjects -> {out_subjects}")


if __name__ == "__main__":
    main()
