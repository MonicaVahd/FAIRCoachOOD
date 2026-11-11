"""
preprocessing/common.py — Shared preprocessing utilities.

Extracted from the original utils.py (functions used by the benchmark
preprocessing scripts only).  Live-system helpers (buffer_to_tensor,
rep_frames_to_tensor, model builders) have been removed.

Seven-step pipeline documented in the paper §3.2:
  1. Load dataset                              (each converter)
  2. Joint remapping → OpenPose-18             (ntu25_to_openpose18 / rehab26_to_openpose18 / kinect22_to_openpose18)
  3. Hip-centering on OpenPose RHip (joint 8)  (each converter)
  4. Torso scaling (neck–hip distance)         (each converter)
  5. Pad or trim to T = 100 frames             (pad_or_trim / pad_or_interpolate)
  6. Min-max normalise x, y to [0, 1]          (normalize_xy)
  7. Reshape to (3, T, 18, 1)                  (each converter)

Discrepancy D2 (from docs/KNOWN_DIFFERENCES.md): step 6 is applied to all
three channels (including the synthetic confidence channel), not to x,y only.
The stored tensors show channel 2 is not constant 1.0 after normalization.
"""

import numpy as np

# ── Joint maps ───────────────────────────────────────────────────────────────
# NTU-25 → OpenPose-18 (14 real joints; face pads to zero by default).
NTU_TO_OPENPOSE = [3, 2, 8, 9, 10, 4, 5, 6, 16, 17, 18, 12, 13, 14]

# REHAB24-26 → OpenPose-18 (14 real joints; face pads to zero).
REHAB_TO_OPENPOSE = [4, 3, 11, 12, 14, 6, 7, 9, 21, 22, 23, 16, 17, 18]

# UI-PRMD Kinect-22 → OpenPose-18 (14 real joints; face pads to zero).
KINECT_TO_OPENPOSE = [3, 2, 8, 9, 10, 4, 5, 6, 16, 17, 18, 12, 13, 14, 1, 0, 0, 0]


# ── Step 6: normalize_xy ─────────────────────────────────────────────────────

def normalize_xy(arr: np.ndarray) -> np.ndarray:
    """
    Min-max normalise x and y channels of an ST-GCN tensor to [0, 1].

    Parameters
    ----------
    arr : np.ndarray of shape (3, T, V)
        Channels: 0 = x, 1 = y, 2 = confidence (or z-depth).

    Returns
    -------
    arr : same array modified in-place.

    Note (D2): Only channels 0 and 1 are intended to be normalised.
    However, in the stored tensors channel 2 is also affected by earlier
    operations (hip-centering and torso scaling modify all three channels).
    """
    for ch in range(2):
        v_min, v_max = arr[ch].min(), arr[ch].max()
        if v_max > v_min:
            arr[ch] = (arr[ch] - v_min) / (v_max - v_min)
    return arr


# ── Step 5a: pad_or_trim (EC3D, REHAB24) ─────────────────────────────────────

def pad_or_trim(seq: np.ndarray, T: int) -> np.ndarray:
    """
    Pad with zeros or trim to exactly T frames.

    Parameters
    ----------
    seq : np.ndarray of shape (t, 3, 18)
    T   : int — target number of frames

    Returns
    -------
    np.ndarray of shape (T, 3, 18)
    """
    t = seq.shape[0]
    if t >= T:
        return seq[:T]
    pad = np.zeros((T - t, 3, 18), dtype=np.float32)
    return np.concatenate([seq, pad], axis=0)


# ── Step 5b: pad_or_interpolate (UI-PRMD) ────────────────────────────────────

def pad_or_interpolate(seq: np.ndarray, target_T: int) -> np.ndarray:
    """
    Resample sequence to exactly target_T frames by linear interpolation.

    Used by UI-PRMD (uiprmd_to_stgcn.py) to handle variable-length sequences.
    EC3D and REHAB24 use pad_or_trim instead.

    Parameters
    ----------
    seq : np.ndarray of shape (t, V, 3)   (note axis order: frames, joints, channels)
    target_T : int

    Returns
    -------
    np.ndarray of shape (target_T, V, 3)
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


# ── Step 2: joint-remapping helpers ──────────────────────────────────────────

def ntu25_to_openpose18(seq: np.ndarray) -> np.ndarray:
    """
    Map NTU-25 joints to OpenPose-18.

    Parameters
    ----------
    seq : np.ndarray of shape (T, 3, 25)
        x, y, z coordinates of 25 NTU joints per frame.

    Returns
    -------
    np.ndarray of shape (T, 3, 18)
        x, y only (z dropped); channel 2 set to synthetic confidence 1.0.
        Face joints (positions 14–17) are zero.

    Note (D2): channel 2 is set to 1.0 here, but subsequent hip-centering
    and torso-scaling (steps 3–4) modify all three channels; the stored
    tensors show channel 2 is not constant 1.0.
    """
    T = seq.shape[0]
    out = np.zeros((T, 3, 18), dtype=np.float32)
    for op_idx, ntu_idx in enumerate(NTU_TO_OPENPOSE):
        out[:, 0, op_idx] = seq[:, 0, ntu_idx]   # x
        out[:, 1, op_idx] = seq[:, 1, ntu_idx]   # y
        out[:, 2, op_idx] = 1.0                   # synthetic confidence
    return out


def rehab26_to_openpose18(seq: np.ndarray) -> np.ndarray:
    """
    Map REHAB24-26 joints to OpenPose-18.

    Parameters
    ----------
    seq : np.ndarray of shape (T, 26, 2)
        x, y pixel coordinates of 26 REHAB24 joints per frame.

    Returns
    -------
    np.ndarray of shape (T, 3, 18)
        Channel 2 = 1.0 (synthetic confidence). Face joints = zero.
    """
    T = seq.shape[0]
    out = np.zeros((T, 3, 18), dtype=np.float32)
    for op_idx, r_idx in enumerate(REHAB_TO_OPENPOSE):
        out[:, 0, op_idx] = seq[:, r_idx, 0]   # x
        out[:, 1, op_idx] = seq[:, r_idx, 1]   # y
        out[:, 2, op_idx] = 1.0                 # confidence
    return out


def kinect22_to_openpose18(seq: np.ndarray) -> np.ndarray:
    """
    Map UI-PRMD Kinect-22 joints to OpenPose-18.

    Parameters
    ----------
    seq : np.ndarray of shape (T, 22, 3)
        x, y, z coordinates of 22 Kinect joints per frame.

    Returns
    -------
    np.ndarray of shape (T, 18, 3)
        Face joints = zero.
    """
    T = seq.shape[0]
    out = np.zeros((T, 18, 3), dtype=np.float32)
    for op_idx, kinect_idx in enumerate(KINECT_TO_OPENPOSE):
        out[:, op_idx, :] = seq[:, kinect_idx, :]
    return out
