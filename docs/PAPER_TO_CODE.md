# Which script produced which paper table

Evidence level is given for every row: **code** = read from the scripts in this repository, **header** = printed header of a stored
result file, **date** = file modification time in the author's development tree, **not recorded** = no record exists.

## Trainers (the scripts the paper's models were trained with)

| Model(s) | Original script (`training/`) | Evidence |
|---|---|---|
| ST-GCN and CTR-GCN | `train_stgcn_full.py` (trains both; sections A, B, C, D) | code: `build_stgcn_model`, `build_ctrgcn_model`, `CTRGCN_SLUG` |
| InfoGCN | `train_infogcn_only.py` | code: `MODEL_SLUG = "infogcn"` |
| BlockGCN | `train_blockgcn_only.py` | code |
| HD-GCN | `train_hdgcn_only.py` | code |
| ProtoGCN | `train_protogcn_only.py` | code |
| SkateFormer | `train_skateformer_only.py` | code (D1 patience 50, see `KNOWN_DIFFERENCES.md` D5) |
| HyperFormer | `train_hyperformer_full.py` | code |
| FreqMixFormer | `train_freqmixformer_full.py` | code |

These eight files are copies of the author's scripts. The only differences are the import lines (`from utils import ...` became
`from models.builders import ...`; `from config import ...` became `from configs.constants import ...`, with `TRAIN_SEED as RANDOM_SEED`).
`CHANGES.md` lists every edit. The single-file trainer in `experimental/` was not used for the paper.

## Tables and numbers

| Paper element | Produced by | Stored file in `results/` | Evidence |
|---|---|---|---|
| Table 1 counts (141 / 191 / 282 / 200) | `preprocessing/{ec3d,rehab24,uiprmd}_to_stgcn.py` (+ Custom pipeline, not included) | — | code; counts checked against the stored arrays |
| Table 5, columns A1, A2, A3, A4 (mean ± std) | trainers with `--iid-only` | `iid_results/iid_results_<model>.txt` | code (flag help text); date 2026-07-14 |
| Table 5, column D1 | trainers with `--mixed-only` | `mixed_results/mixed_results_<model>.txt` | code (flag help text); date 2026-07-15 |
| Table 6, every cell (OOD macro F1 and FC/PC mark) | OOD sections of the trainers (checkpoints `ood_*.pth`, dated 2026-06-16), then `training/check_confusion.py`, then `eval/recompute_macro_f1.py` | `confusion_all_models.txt`, `ood_results.csv` | code for the trainers and `check_confusion.py`; header: `confusion_all_models.txt` starts with the lines `Device:` / `Output dir:` that `check_confusion.py` prints; the command that wrote the file is **not recorded**; date 2026-06-20 |
| Table 4 (FC / PC per model) | `eval/collapse.py` | `collapse_stats.txt` | code; date 2026-07-16 |
| Section 4.2 (104 all-incorrect / 37 all-correct) | `eval/fc_direction.py` | (printed output; no stored file) | code |
| Table 7 (mean OOD macro F1) and Wilcoxon row | `eval/wilcoxon.py` on `ood_results.csv` | `wilcoxon.csv` | code; date 2026-07-14 |
| Section 6.5 rerun (117 UI-PRMD-related cells) | `training/retrain_uiprmd_experiments.py` | `uiprmd_rerun/uiprmd_rerun_results.txt`, `uiprmd_rerun_confusion.txt` | code; date 2026-06-22 |
| Status-change counts for that rerun | `eval/rerun_status_changes.py` | (printed output) | code; see `KNOWN_DIFFERENCES.md` D4 |

## Split and test-set arrays used by the trainers

| Arrays | Script | Note |
|---|---|---|
| `*_loso_round*_{train,val,test}_{data,labels}.npy` | `training/split_loso.py` | needs `configs/ec3d_subjects.yaml` for EC3D |
| `*_ood_test_{data,labels}.npy` | `training/split_ood.py` | copies the full dataset arrays |
| 80/20 and 90/10 source splits of the B and C conditions | inside the trainers (`split_frac`, seed 42) | |
| Custom 5-fold arrays | `split_kfold.py` | not included |

`splits/make_split_indices.py` is new in this release: it writes integer index lists that replay the same rules. It was tested on
synthetic data only (14 checks, see `docs/reproduction.md`); it was not compared with the author's stored split arrays.
