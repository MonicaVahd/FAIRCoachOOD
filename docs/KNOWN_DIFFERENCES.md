# Known differences between the paper and the stored code/data

This file compares the published PDF with the code and the stored results. It records evidence only: it does not say which side is right,
and it gives no cause for a difference unless the cause was verified. Entry numbers are kept stable, so some numbers are absent.

---

## D2 — Synthetic confidence channel not constant 1.0

| | |
|--|--|
| **Status** | Open |
| PDF statement | §3.1: "set to a synthetic confidence value of 1.0"; §3.2 step 6: "the third channel remains unchanged" |
| Evidence 1 (code, read not run) | `ntu25_to_openpose18` assigns 1.0 to channel 2; the converters then subtract the RHip joint and divide by the torso length over all three channels (e.g. `preprocessing/ec3d_to_stgcn.py`). |
| Evidence 2 (stored tensors, original `output/*_train_data.npy`, `uiprmd_data.npy`) | Minimum and maximum of channel 2 over all samples, frames and joints: EC3D −14.82 / 0.0 (141 samples, 142 distinct values), REHAB24-6 −0.0068 / 0.0 (191 samples, 192 distinct values), Custom −0.0048 / 0.0024, UI-PRMD −5.10 / 0.28. None of these is a constant 1.0. |
| Cause | Not verified. |

---

## D3 — UI-PRMD preprocessing note in the rerun script vs run dates

| | |
|--|--|
| **Status** | Open (evidence incomplete) |
| PDF statement | §3.2: "All sequences undergo the same seven-step process" (includes min-max step 6) |
| Code | `training/retrain_uiprmd_experiments.py` docstring: "following the unified preprocessing fix (normalize_xy now applied to UI-PRMD)". |
| File dates (original tree) | UI-PRMD checkpoints `ood_b7_uiprmd_to_ec3d.pth` etc.: 2026-06-16; `confusion_all_models.txt` (source of the Table 6 values): 2026-06-20; `uiprmd_data.npy`: 2026-06-21 18:53; `uiprmd_rerun_confusion.txt`: 2026-06-22 |
| Not available | No run manifest, data hash or log states which UI-PRMD preprocessing the 2026-06-16 checkpoints were trained with. |

---

## D4 — UI-PRMD rerun: status-change counts

| | |
|--|--|
| **Status** | Open (counts reproduced; the definition is not stated in the paper) |
| PDF statement | §6.5: "117 UI-PRMD-related cells were rerun … 25 status changes: 11 recoveries, 8 regressions, 6 FC↔PC". The PDF does not define "recovery" or "regression". |

**Scope.** 117 cells = 9 models × 13 conditions.
Category A, 81 cells (9 conditions × 9 models): B7, B8, B9, C4, C5, C7, C8, C9, C10, retrained.
Category B, 36 cells (4 conditions × 9 models): B10, B11, B12, C6; the original checkpoints were re-evaluated on refreshed test data.

**Facts.**
* The rerun followed the `normalize_xy` preprocessing fix for UI-PRMD (docstring of `training/retrain_uiprmd_experiments.py`).
* The Table 6 values are the pre-rerun values: all 117 equal the corresponding entries of `results/ood_results.csv`.
* No further cause is stated.

**Definition.** Status of a cell from its confusion counts (positive class = correct squat): n1 = TP + FP, n0 = TN + FN.
FC: n0 = 0 or n1 = 0. PC: not FC, and n0 ≤ 5 or n1 ≤ 5. OK: otherwise.
Recovery: FC or PC → OK. Regression: OK → FC or PC. FC↔PC: FC → PC or PC → FC.
"Old" counts are those of `results/ood_results.csv`; "new" counts are those of `results/uiprmd_rerun/uiprmd_rerun_confusion.txt`.

**Counts** (computed by `eval/rerun_status_changes.py`; 46 of the 117 cells have different confusion counts):

| Count | Recoveries | Regressions | FC↔PC | Total |
|---|---|---|---|---|
| Paper, §6.5 | 11 | 8 | 6 | 25 |
| Three-way rule above, from the two stored files | 11 | 8 | 6 | 25 |

The paper's counts are reproduced exactly by the three-way rule. The cell-by-cell table is in
[`uiprmd_rerun_comparison.md`](uiprmd_rerun_comparison.md).

---

## D5 — SkateFormer D1 early-stopping patience = 50

| | |
|--|--|
| **Status** | Open |
| PDF statement | §3.3: "early stopping … patience of 20" for all models |
| Code | `training/train_skateformer_only.py:50` `MIXED_EARLY_STOP_PAT = 50`, used at `:467` for D1 |
| Evidence | The stored script uses patience 50 for the D1 run of SkateFormer; Table 5 lists 85.0% for SkateFormer D1 (`results/mixed_results/mixed_results_skateformer.txt`). The log file does not record the patience value. |

---

## D7 — Seeds

| | |
|--|--|
| **Status** | Open |
| PDF statement | §3.3: "The random seed is fixed at 42"; §6.5: "same nominal seed" |
| Code | `random`, `numpy` and `torch` are seeded once at the start of `main()` in each trainer script (e.g. `training/train_stgcn_full.py`, search `manual_seed`); the split helper `split_frac` is seeded again on every call. |
| Not tested | Whether a single cell run in isolation reproduces its stored value was not tested. |

---

## D8 — Tables 5 and 6 from different run dates

| | |
|--|--|
| **Status** | Open |
| Table 6 | Built from `confusion_all_models.txt` dated 2026-06-20 |
| Table 5 | Built from `iid_results_*.txt` dated 2026-07-14 and `mixed_results_*.txt` dated 2026-07-15 |
| Not available | No manifest links either table to a code version. |

---

## N1 — Three or four universally collapsed conditions

| | |
|--|--|
| **Status** | Open |
| PDF §4.2 (p. 6) | "Four conditions … B3, B7, B12, and C10" universally collapsed |
| PDF §5.2 (p. 7) | "Three conditions – B3, B7, and B12 – are universally weak" |
| Stored data | All 9 models collapse in exactly B3, B7, B12, C10 (four conditions) |

---

## N2 — SkateFormer "no temporal positional embedding"

| | |
|--|--|
| **Status** | Open |
| PDF claim | Table 2: "no temporal positional embedding" |
| Code (original tree) | `utils.py:509` passes `index_t=False`; `lib/net/skateformer.py:364-367` creates a learned `joint_person_temporal_embedding`, added at `:443` |
| Not resolved | Which of the two the PDF means was not determined. |

---

## N4 — IID values: "rounds and datasets" vs "rounds or folds"

| | |
|--|--|
| **Status** | Open |
| PDF §3.4 (p. 5) | "averages across rounds and datasets" |
| PDF Table 5 caption | "across LOSO rounds or CV folds" |
| Stored data | `iid_results_*.txt` has per-dataset columns, which agrees with the caption. |
