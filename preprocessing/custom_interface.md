# Custom dataset interface

The Custom YouTube dataset is not distributed (see README §Datasets).
If you want to train on your own squat dataset as a plug-in source,
prepare your data in the following format and place it in `output/`.

## Required output arrays

| File | Shape | dtype | Description |
|------|-------|-------|-------------|
| `custom_squat_train_data.npy` | `(N, 3, 100, 18, 1)` | float32 | ST-GCN tensors |
| `custom_squat_train_labels.npy` | `(N,)` | int64 | 1 = correct, 0 = incorrect |

## Preprocessing pipeline

Apply the **same seven-step pipeline** as EC3D, REHAB24-6, and UI-PRMD
(see paper §3.2 and `preprocessing/common.py`):

1. Load your keypoints as (T, joints, 2 or 3) arrays.
2. Remap to **OpenPose-18** skeleton layout (see `common.py` for joint maps).
3. **Hip-center**: subtract the RHip (OpenPose joint 8) position from all joints and frames.
4. **Torso-scale**: divide by the mean Euclidean distance between the Neck (joint 1) and RHip (joint 8) across the sequence.  Skip if `torso_height < 1e-6`.
5. **Pad/trim**: use `pad_or_trim(seq, T=100)` for fixed-length sequences, or `pad_or_interpolate(seq, 100)` for variable-length.
6. **Min-max normalise** x and y channels to [0, 1] using `normalize_xy(arr)`.
7. **Reshape** to `(3, 100, 18, 1)` and stack into `(N, 3, 100, 18, 1)`.

## Class labels

| Label | Meaning |
|-------|---------|
| 1 | Correct squat repetition |
| 0 | Incorrect squat repetition (any error category) |

## Notes

- Subject identity is not required for OOD conditions.
- No clip IDs, video URLs, or personal information should be stored in the arrays.
- The expected class balance for the Custom dataset in the paper is 162 correct / 120 incorrect (282 total).  Your dataset may differ.
