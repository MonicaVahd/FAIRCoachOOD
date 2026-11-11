# Skeleton-based squat quality under domain shift — code, preprocessing and splits

Code, preprocessing tools and benchmark split generation for the paper

> Monica Vahdati, Kamran Gholizadeh HamlAbadi, Ali Etemad, Abdulmotaleb El Saddik.
> **Skeleton-Based Exercise Quality Assessment Under Domain Shift: A Benchmark Study of Cross-Domain Failure Modes.**
> The 3rd International Workshop on Multimedia Computing for Health and Medicine (MCHM '26), Rio de Janeiro, 2026. ACM.
> DOI: [10.1145/3840471.3842592](https://doi.org/10.1145/3840471.3842592) · Project page: <https://skeletoneqa.github.io/>

The paper evaluates nine skeleton models on four squat datasets (EC3D, REHAB24-6, UI-PRMD and a Custom YouTube collection)
under within-distribution (IID), out-of-distribution (OOD, 22 conditions per model, 198 experiments) and pooled (D1) protocols,
and counts full and partial classifier collapse from the confusion matrices.

## Read this first

1. **Single-seed, exploratory results.** Every experiment was trained once with seed 42. The paper states that the
   resulting rankings and statistical tests "should be regarded as exploratory". In the paper's Wilcoxon analysis one of 36 model
   pairs is significant before correction (HD-GCN vs HyperFormer, p = 0.046) and none after Bonferroni correction.
2. **Only part of the benchmark can be rebuilt from public data.** The Custom YouTube collection is not part of this release
   (no data, no downloader, no clip lists; `preprocessing/custom_interface.md` documents only the array format a loader would have to produce).
   16 of the 22 OOD conditions, A3 and D1 need it and are **not reproducible** from this repository:
   B2, B4, B5, B6, B9, B12, C1–C10, A3, D1. Reproducible from the three public datasets: **A1, A2, A4** (LOSO) and
   **B1, B3, B7, B8, B10, B11** (single-source OOD).
3. **No data, derived tensors or checkpoints are included.** `results/` holds aggregate counts and metrics only.
4. **Differences between the paper and the stored code/results** are listed with evidence in [`docs/KNOWN_DIFFERENCES.md`](docs/KNOWN_DIFFERENCES.md).
   The map from each paper table to the script and stored file that produced it is in [`docs/PAPER_TO_CODE.md`](docs/PAPER_TO_CODE.md).
5. **Three of the nine models are not included** (CTR-GCN, InfoGCN, FreqMixFormer; licence reasons, see below).
6. **Restructured and only partly tested.** The scripts were restructured for this release and tested only on synthetic data. No trainer or converter has been run
   end to end on the public datasets in this layout, so the commands in `docs/reproduction.md` are unverified. `requirements.txt` has not been installed from scratch in a clean Python 3.8 environment.

## Datasets (not distributed here)

Obtain each dataset from its authors; see the references in the paper. Licences are listed as far as they were checked.

| Dataset (paper reference) | Public-only conditions that use it | Licence |
|---|---|---|
| EC3D — Zhao et al., ACCV 2022 | A1, B1, B3, B7, B10 | not verified |
| REHAB24-6 — Černek et al., SISAP 2024 | A2, B1, B3, B8, B11 | CC BY-NC 4.0 (Zenodo record); verify at the source |
| UI-PRMD — Vakanski et al., Data 2018 | A4, B7, B8, B10, B11 | not verified |
| Custom YouTube | — | not part of this release |

Details: [`docs/data_access.md`](docs/data_access.md).

## Install

Python 3.8 (the environment the results were produced in; newer versions are untested).

```bash
conda create -n ood-bench python=3.8
conda activate ood-bench
pip install -r requirements.txt
```

## Repository layout

| Path | Content |
|---|---|
| `preprocessing/` | the original converters (`ec3d_to_stgcn.py`, `rehab24_to_stgcn.py`, `uiprmd_to_stgcn.py`) and shared helpers (`common.py`) |
| `configs/` | constants (`constants.py`), EC3D subject-list loader (`subjects.py`) and `ec3d_subjects.example.yaml` |
| `models/` | six model implementations with their upstream licences, and `builders.py` |
| `training/` | the original trainer scripts used for the paper, `split_loso.py`, `split_ood.py`, `check_confusion.py`, `retrain_uiprmd_experiments.py` |
| `eval/` | Table 4–7 scripts (`recompute_macro_f1.py`, `collapse.py`, `fc_direction.py`, `wilcoxon.py`) and `rerun_status_changes.py` |
| `splits/` | `make_split_indices.py`, which writes integer index lists |
| `results/` | aggregate result files used for the paper's tables |
| `scripts/` | `run_trainer.sh` (launcher), `fetch_upstream.py` |
| `experimental/` | a single-file trainer that was **not** used for the paper (see its header) |
| `docs/` | `PAPER_TO_CODE.md`, `KNOWN_DIFFERENCES.md`, `reproduction.md`, `data_access.md` |

## Reproducing the public-data conditions

Step-by-step commands are in [`docs/reproduction.md`](docs/reproduction.md). In short:

1. Preprocess: `python preprocessing/ec3d_to_stgcn.py`, `rehab24_to_stgcn.py`, `uiprmd_to_stgcn.py` (dataset paths are set in `configs/constants.py`).
2. EC3D subjects: copy `configs/ec3d_subjects.example.yaml` to `configs/ec3d_subjects.yaml` (git-ignored) and enter the subject names of your EC3D copy.
3. Build the split arrays with `training/split_loso.py` and `training/split_ood.py`, or write index lists with `splits/make_split_indices.py`.
4. Train with the original scripts: `bash scripts/run_trainer.sh <model> <iid|ood|mixed|all>`.
5. Evaluate with the scripts in `eval/`.

The paper's numbers come from single runs; re-running is not expected to reproduce individual cells exactly
(see `docs/KNOWN_DIFFERENCES.md`, D7).

## Models

| Model | Upstream | Licence | In this repository |
|---|---|---|---|
| ST-GCN | https://github.com/yysijie/st-gcn | BSD 2-Clause | yes |
| HD-GCN | https://github.com/Jho-Yonsei/HD-GCN | MIT | yes |
| ProtoGCN | https://github.com/firework8/ProtoGCN | MIT | yes |
| SkateFormer | https://github.com/KAIST-VICLab/SkateFormer | MIT | yes |
| BlockGCN | https://github.com/ZhouYuxuanYX/BlockGCN | Apache-2.0 | yes, modified |
| HyperFormer | https://github.com/ZhouYuxuanYX/Hyperformer | Apache-2.0 | yes, modified |
| CTR-GCN | https://github.com/Uason-Chen/CTR-GCN | CC BY-NC 4.0 | no |
| InfoGCN | https://github.com/stnoah1/infogcn | no licence file | no |
| FreqMixFormer | https://github.com/wenhanwu95/FreqMixFormer | no licence file | no |

`scripts/fetch_upstream.py` can download the upstream files of the three excluded models at a pinned commit. It only downloads:
the paper's configuration changes are not applied and `models/builders.py` still raises an error for these three models.
The commit used for the paper is unknown; see the header of the script. Licence texts: `THIRD_PARTY_LICENSES/`; notices: `NOTICE`.

## Licence and citation

Own code: Apache License 2.0 (`LICENSE`), subject to the third-party licences above.
Cite the paper above; machine-readable metadata is in `CITATION.cff`.

This code was developed with AI assistance; the author reviewed the output.
