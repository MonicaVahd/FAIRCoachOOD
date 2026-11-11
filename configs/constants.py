"""
configs/constants.py — Benchmark constants for the FAIR-Coach OOD repository.

Extracted from the original config.py; thesis-era paths and live-system
constants have been removed.  Edit DATASET_* paths or pass them via
command-line arguments (--data-dir) in the preprocessing scripts.

Paper reference: §3.2 (preprocessing) and §3.3 (training protocol).
"""

# ── Joint remapping tables ──────────────────────────────────────────────────
# NTU-25 joint indices → OpenPose-18 positions (14 real body joints).
# Used in preprocessing/ec3d_to_stgcn.py
# Source: utils.py, also documented in paper §3.1 (Table 1 footnote).
NTU_TO_OPENPOSE = [3, 2, 8, 9, 10, 4, 5, 6, 16, 17, 18, 12, 13, 14]

# REHAB24-26 joint indices → OpenPose-18 positions (14 real body joints).
# Used in preprocessing/rehab24_to_stgcn.py
REHAB_TO_OPENPOSE = [4, 3, 11, 12, 14, 6, 7, 9, 21, 22, 23, 16, 17, 18]

# COCO-17 → OpenPose-18 index lookup.  None = neck (synthesised as average
# of left shoulder (5) and right shoulder (6)).
# Used by buffer_to_tensor in preprocessing/common.py
COCO_TO_OPENPOSE = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

# OpenPose-18 → COCO-17 index lookup (inverse).
OPENPOSE_TO_COCO = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

# ── Preprocessing parameters ────────────────────────────────────────────────
STGCN_WINDOW   = 100    # T: target temporal length after pad/trim/interpolate
STGCN_NUM_CLASSES = 2   # 0 = incorrect, 1 = correct
STGCN_IN_CHANNELS = 3   # x, y, (synthetic confidence)
STGCN_DROPOUT  = 0.5

# ── Training protocol (paper §3.3) ──────────────────────────────────────────
TRAIN_EPOCHS       = 200
TRAIN_BATCH_SIZE   = 16
TRAIN_LR           = 0.001       # Adam
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_SEED         = 42

# Early-stopping patience (all models except SkateFormer D1 which uses 50).
EARLY_STOP_PAT     = 20

# ── Default dataset paths ────────────────────────────────────────────────────
# Override via --data-dir command-line argument or by editing here.
EC3D_PICKLE     = "datasets/EC3D/data_3D.pickle"
REHAB24_CSV     = "datasets/REHAB24/2d_joints/Segmentation.csv"
REHAB24_DATA_DIR = "datasets/REHAB24/2d_joints/Ex6"
UIPRMD_DIR      = "datasets/ui_prmd"   # expects ui_prmd/correct/ and incorrect/

OUTPUT_DIR = "output"
