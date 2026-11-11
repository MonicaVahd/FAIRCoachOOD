# Dataset access and licences

None of the datasets is distributed with this repository. Obtain them from their authors (references are in the paper).
Licences are stated only where they were checked; everything else is marked "not verified".

## EC3D

- Reference: Zhao et al., "3D pose based feedback for physical exercises", ACCV 2022 (paper reference [23]).
- Paper statistics (Table 1): squat subset, 141 repetitions (41 correct / 100 incorrect), 4 subjects.
- Licence: not verified. A direct dataset page with licence terms could not be found.
- Format read by `preprocessing/ec3d_to_stgcn.py`: a pickle (`data_3D.pickle`) with a `poses` array (frames × 3 × 25, NTU-25 joints) and a `labels`
  array whose columns are `[action, subject, exercise_type, repetition, frame_id]`. Subject names are read from your own copy of the
  data; they are not stored in this repository (`configs/ec3d_subjects.example.yaml` uses placeholders).

## REHAB24-6

- Reference: Černek, Sedmidubsky, Budikova, "REHAB24-6: Physical therapy dataset for analyzing pose estimation methods", SISAP 2024 (paper reference [3]).
- Paper statistics: squat subset (`exercise_id = 6`, `mocap_erroneous = 0`), 191 repetitions (130 correct / 61 incorrect), 9 subjects.
- Licence: CC BY-NC 4.0 (Zenodo record); verify at the source.
- Format read by `preprocessing/rehab24_to_stgcn.py`: `Segmentation.csv` (semicolon-separated) and one `<video_id>-c17-30fps.npy` per video (frames × 26 joints × 2).

## UI-PRMD

- Reference: Vakanski, Jun, Paul, Baker, "A data set of human body movements for physical rehabilitation exercises", Data 3(1):2, 2018 (paper reference [18]).
- Paper statistics: deep-squat subset (file prefix `m01`), 200 repetitions (100 correct / 100 incorrect), 10 subjects.
- Licence: not verified. The project page that was tried did not load.
- Format read by `preprocessing/uiprmd_to_stgcn.py`: one `.txt` file per sequence in `correct/` and `incorrect/` folders, named like
  `m01_sXX_eYY_positions[_inc].txt` (subject number `XX`).

## Custom YouTube collection

- Not part of this release: no data, no downloader, no clip lists.
- The paper reports 282 repetitions (162 correct / 120 incorrect) for it (Table 1).
- To run Custom-dependent conditions with your own data, `preprocessing/custom_interface.md` describes the array shape and label format the trainers expect.
