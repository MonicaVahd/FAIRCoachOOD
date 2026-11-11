# CHANGES.md — differences between this repository and the author's development tree

Every difference between the files shipped here and the author's original files, as of 2026-11-10.
The original scripts that produced the paper's numbers are copied with the smallest possible edits; the edits are listed exactly.

## 1. Edits to original scripts (no logic changes)

| Edit | Meaning | Files |
|---|---|---|
| E1 | `from utils import <builders>` → `from models.builders import <builders>`. The builders of the three excluded models raise a clear `ImportError` when *called* (lazy), so importing a trainer never fails | `training/train_{stgcn_full,infogcn_only,blockgcn_only,hdgcn_only,protogcn_only,skateformer_only,hyperformer_full,freqmixformer_full}.py`, `training/check_confusion.py`, `training/retrain_uiprmd_experiments.py` |
| E2 | `from config import ...` → `from configs.constants import ...`; the name `RANDOM_SEED` is imported as `TRAIN_SEED as RANDOM_SEED` (both are 42) | the same files, plus `training/split_loso.py`, `training/split_ood.py`, `preprocessing/{ec3d,rehab24,uiprmd}_to_stgcn.py` |
| E3 | helper imports `from utils import normalize_xy, ...` → `from preprocessing.common import normalize_xy, ...` | `training/split_loso.py`, `preprocessing/*_to_stgcn.py` |
| E4 | hard-coded EC3D subject list replaced by `load_ec3d_subjects()` (reads `configs/ec3d_subjects.yaml`, which is git-ignored); the docstring no longer names the subjects; a comment naming a subject was replaced by `<subject>` | `training/split_loso.py`, `preprocessing/ec3d_to_stgcn.py` |
| E5 | `from compute_collapse_stats import` → `from collapse import` (the file was renamed) | `eval/fc_direction.py` |
| E6 | the legacy per-model `*_results.txt` files are optional: a missing file prints a note and leaves the `old_f1`, `diff`, `status_old` columns empty instead of raising `FileNotFoundError` | `eval/recompute_macro_f1.py` |

Verified with `git diff --no-index` against the originals: each trainer differs in exactly 3 lines (E1, E2), `split_ood.py` in 1 line,
`check_confusion.py` in 2 lines, `rehab24_to_stgcn.py` and `uiprmd_to_stgcn.py` in 2 lines, `ec3d_to_stgcn.py` in 3 lines (E2, E3 and one comment that contained a subject name, now `<subject>`), `retrain_uiprmd_experiments.py` in 3 lines, `split_loso.py` in 5 lines (E2–E4),
`training/metrics_utils.py` in none.

## 2. Model files

* Copied from `lib/net/`; a header (upstream URL and file, upstream copyright line, licence, modification notice) was prepended and a byte-order mark removed.
  The code after the header is unchanged (checked line by line).
* `models/builders.py` was written for this release from the builder functions of the original `utils.py`. The model arguments were compared with the original
  (`num_class=2`, `num_point=18`, graphs, `adaptive=True`, `CoM=8`, `base_channels=96`, SkateFormer `index_t=False`, ...). It registers a package named `net`
  whose search path is the six model folders, because the model files import each other as `net.<module>`.
  The six builders construct on CPU and ST-GCN, HD-GCN, ProtoGCN and SkateFormer were run on a random input.
* Excluded models: `ctrgcn`, `infogcn*`, `freqmixformer*` (see `NOTICE`).
* `THIRD_PARTY_LICENSES/` holds each upstream project's own LICENSE file, fetched from its GitHub repository on 2026-11-10 (no generic licence texts).

## 3. Helpers rewritten or re-created

* `preprocessing/common.py`: `normalize_xy`, `pad_or_trim`, `ntu25_to_openpose18`, `rehab26_to_openpose18` have the same function bodies as in the original
  `utils.py` (compared as syntax trees, ignoring docstrings and type annotations); added `pad_or_interpolate` and `kinect22_to_openpose18`, which the original keeps inside `uiprmd_to_stgcn.py`.
* `configs/constants.py` (trimmed from `config.py`: only constants used by the scripts here), `configs/subjects.py` (new), `configs/ec3d_subjects.example.yaml` (new, S1–S4).

## 4. New files

`README.md`, `NOTICE`, `CITATION.cff`, `requirements.txt`, `CHANGES.md`, `.gitignore`, `docs/{KNOWN_DIFFERENCES,uiprmd_rerun_comparison,PAPER_TO_CODE,reproduction,data_access}.md`,
`preprocessing/custom_interface.md`, `scripts/run_trainer.sh`, `scripts/fetch_upstream.py`, `splits/make_split_indices.py`, `eval/rerun_status_changes.py`,
`tests/test_preprocessing.py`, `experimental/train_unified.py`.

## 5. Results (`results/`)

* Aggregate counts and metrics only. Checked: no subject names, REHAB24/UI-PRMD/EC3D sample identifiers, per-sample predictions, e-mail addresses or local paths.
* UTF-16 → UTF-8: `collapse_stats.txt`. All 1,600 numeric tokens equal the original; only two header lines differ (local paths replaced by `results/...`).
* `confusion_all_models.txt` (added): the original text with local paths removed from the checkpoint/data path lines and from the saved warning messages;
  numbers unchanged. It lets `eval/` regenerate Tables 4 and 6 without training.
* A UTF-8 byte-order mark that an earlier copy step had added to `ood_results.csv`, `wilcoxon.csv` and `collapse_stats.txt` was removed (the originals have none; read with plain UTF-8 the first column name becomes `﻿model`).
* All other files are byte-identical in content to the originals (numeric tokens compared: iid ×9, mixed ×9, rerun ×2, `ood_results.csv`, `wilcoxon.csv`).

## 6. Corrections made during this review

* **Unified trainer removed from the release.** An earlier version replaced the eight trainers by one refactored `train/train.py`. It is now `experimental/train_unified.py`
  with a header stating that it was not used for the paper. (One difference found by reading it: scheduler patience 5; the paper states 10.)
* **Converters restored.** The three converters had also been rewritten with command-line flags; the originals are restored (E2/E3 only) and the rewritten versions deleted.
* **`splits/make_split_indices.py`**: the script had never been run. Two bugs were found by running it on synthetic data and fixed: a `TypeError` in `_meta()` (keyword/parameter name mismatch)
  and condition B1, which listed EC3D as its test dataset instead of REHAB24-6. A guard was added that stops when EC3D subjects are missing from the YAML list.
* **`docs/KNOWN_DIFFERENCES.md`**: cause statements removed; the entries that only compared an earlier draft with the PDF were removed; the 117-row rerun table moved to `docs/uiprmd_rerun_comparison.md`.
  with the parsing fixed, the same definition gives 25 (11 / 8 / 6), the figure in the paper. See D4.
* `README.md`, `NOTICE`, `CITATION.cff`, `docs/*.md` were rewritten: paper title and author names now follow the PDF; an unsupported statement about "eleven of 36 model pairs" and an unverified
  claim about dataset licences of checkpoints were removed; links that pointed to the paper's DOI instead of the datasets were removed; a reference to a script that does not exist (`eval/make_tables.py`) was removed.
* `scripts/fetch_upstream.py` now pins the default-branch head commit of each upstream repository (read 2026-11-10); the commit used for the paper is unknown. It no longer claims that the builders pick up the downloaded files.

## 7. Excluded

Data and derived arrays (`*.npy`), checkpoints (`*.pth`), the Custom YouTube pipeline and clip lists, local copies of CTR-GCN, InfoGCN and FreqMixFormer, thesis-era code,
private documents, the paper PDF and LaTeX source (the DOI is linked instead), legacy result files, secrets.

## 8. Not done / not verified

* No trainer, converter or split script was run on real data. The index-list generator was tested on synthetic data only.
* The parity of `models/builders.py` with the original `utils.py` was checked by reading; no model output was compared with a checkpoint.
* The EC3D and UI-PRMD licences are unverified. The commit of each upstream repository that the local model files came from is unknown.
