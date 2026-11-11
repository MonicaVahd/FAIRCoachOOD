# Reproduction guide

What can be rebuilt from the three public datasets, and how. Read `README.md` and `docs/KNOWN_DIFFERENCES.md` first.
Nothing below was run on the real datasets during release preparation; see "What was tested" at the end.

## Conditions

| Reproducible from public data | Not reproducible (need the Custom YouTube collection, which is not part of this release) |
|---|---|
| A1 (EC3D LOSO), A2 (REHAB24-6 LOSO), A4 (UI-PRMD LOSO) | A3 |
| B1, B3, B7, B8, B10, B11 | B2, B4, B5, B6, B9, B12, C1–C10, D1 |

Models CTR-GCN, InfoGCN and FreqMixFormer are not included (`scripts/fetch_upstream.py` only downloads upstream files; see its header).
The six included models are ST-GCN, HD-GCN, ProtoGCN, SkateFormer, BlockGCN and HyperFormer.

## Steps

All commands are run from the repository root with the Python 3.8 environment of `requirements.txt`.
On Windows set `PYTHONUTF8=1` (some scripts print non-ASCII characters) and, if OpenMP complains, `KMP_DUPLICATE_LIB_OK=TRUE`.

1. **Datasets.** Obtain EC3D, REHAB24-6 and UI-PRMD from their authors (`docs/data_access.md`). The scripts read them from the locations in
   `configs/constants.py` (`datasets/EC3D/data_3D.pickle`, `datasets/REHAB24/2d_joints/...`) and, for UI-PRMD, from `datasets/ui_prmd/correct`
   and `datasets/ui_prmd/incorrect` (fixed inside `preprocessing/uiprmd_to_stgcn.py`). Edit those paths if your copies are elsewhere.
2. **EC3D subject list.** `cp configs/ec3d_subjects.example.yaml configs/ec3d_subjects.yaml`, then replace `S1..S4` with the subject names found in
   your EC3D pickle (column 1 of its `labels` array). The file is git-ignored. The order defines the LOSO rounds.
3. **Preprocess.**
   ```bash
   python preprocessing/ec3d_to_stgcn.py
   python preprocessing/rehab24_to_stgcn.py
   python preprocessing/uiprmd_to_stgcn.py
   ```
   Output: `output/{ec3d,rehab24}_train_{data,labels}.npy` and `output/uiprmd_{data,labels,subjects}.npy`.
4. **Split arrays** used by the original trainers: `python training/split_loso.py` and `python training/split_ood.py`.
   (`split_ood.py` also looks for the Custom arrays and skips what is missing.)
   Optionally write index lists: `python splits/make_split_indices.py --ec3d-pickle ... --rehab24-csv ... --rehab24-data ... --uiprmd-dir ...`
   (`--verify` re-derives the hashes of an existing set of files).
5. **Train** with the original scripts: `bash scripts/run_trainer.sh <model> <iid|ood|mixed|all>`
   (model: `stgcn`, `hdgcn`, `protogcn`, `skateformer`, `blockgcn`, `hyperformer`). The trainers read `./output`.
   Sections whose input arrays are missing are skipped only where the original script checks for them; run the conditions you need
   and ignore failures in the Custom-dependent sections. A full run of one script trains many models and takes many hours on a GPU.
6. **Confusion counts and tables.**
   ```bash
   python training/check_confusion.py > output/confusion_all_models.txt   # re-evaluates the OOD checkpoints
   python eval/recompute_macro_f1.py      # writes output/ood_results_corrected.csv
   python eval/collapse.py                # Table 4
   python eval/fc_direction.py
   python eval/wilcoxon.py                # Table 7 / Wilcoxon
   ```
   To re-derive the paper's tables from the shipped counts without training, copy the stored files first:
   `mkdir -p output && cp results/confusion_all_models.txt output/` and run the four `eval/` scripts.
   `recompute_macro_f1.py` leaves the `old_f1`, `diff` and `status_old` columns empty because the legacy per-model result files are not shipped.
   The redirect in the first command is how the stored file appears to have been produced; the original command is not recorded.

## What was tested during release preparation

* `tests/test_preprocessing.py`: 19 unit tests on synthetic arrays, pass.
* The six included model builders construct on CPU; ST-GCN, HD-GCN, ProtoGCN and SkateFormer were also run on a random input
  (BlockGCN and HyperFormer were only constructed: their forward passes need extra inputs).
* All eight trainers, the three converters, `check_confusion.py`, `retrain_uiprmd_experiments.py` and `split_ood.py` import without error. **No trainer or converter was run.**
* `eval/recompute_macro_f1.py`, `collapse.py`, `fc_direction.py`, `wilcoxon.py` ran on synthetic confusion counts (198 rows and 36 pairs produced).
* `splits/make_split_indices.py` ran on synthetic datasets: 14 of 14 checks passed, including equality with `training/split_loso.py::pool_train_val`
  on index-valued arrays and the `--verify` behaviour. Two bugs found by this test were fixed (see `CHANGES.md`).
* Not tested: any run on the real datasets; training; reproduction of any number in the paper's tables from raw data.
