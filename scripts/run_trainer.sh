#!/usr/bin/env bash
# scripts/run_trainer.sh -- run the ORIGINAL paper trainer of one model.
#
# Usage (from the repository root):
#   bash scripts/run_trainer.sh <model> <mode>
#     model: stgcn | ctrgcn | infogcn | blockgcn | hdgcn | protogcn | skateformer | hyperformer | freqmixformer
#     mode : iid | mixed | ood | all
#
# Mapping (same as the paper's launchers run_iid.ps1 / run_mixed.ps1):
#   stgcn, ctrgcn  -> training/train_stgcn_full.py        (one script trains BOTH models)
#   infogcn        -> training/train_infogcn_only.py
#   blockgcn       -> training/train_blockgcn_only.py
#   hdgcn          -> training/train_hdgcn_only.py
#   protogcn       -> training/train_protogcn_only.py
#   skateformer    -> training/train_skateformer_only.py
#   hyperformer    -> training/train_hyperformer_full.py
#   freqmixformer  -> training/train_freqmixformer_full.py
#
# Flags used: --iid-only, --mixed-only, --ood-only.  Only train_stgcn_full.py,
# train_hyperformer_full.py and train_freqmixformer_full.py define --ood-only;
# for the other scripts mode "ood" and "all" run every section (A, B, C, D)
# without a flag.  Models ctrgcn, infogcn and freqmixformer need
# `python scripts/fetch_upstream.py` first (see README).
#
# The trainers read arrays from ./output (OUTPUT_DIR in configs/constants.py).
# Conditions whose input arrays are missing are skipped only where the original
# script guards them with check_files(); sections that need the Custom YouTube
# data are not reproducible from public data (see docs/reproduction.md).

set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUTF8=1

MODEL="${1:-}"
MODE="${2:-}"
if [[ -z "$MODEL" || -z "$MODE" ]]; then
    sed -n '2,12p' "$0"
    exit 1
fi

case "$MODEL" in
    stgcn|ctrgcn)  SCRIPT="training/train_stgcn_full.py"; HAS_OOD_FLAG=1;;
    infogcn)       SCRIPT="training/train_infogcn_only.py"; HAS_OOD_FLAG=0;;
    blockgcn)      SCRIPT="training/train_blockgcn_only.py"; HAS_OOD_FLAG=0;;
    hdgcn)         SCRIPT="training/train_hdgcn_only.py"; HAS_OOD_FLAG=0;;
    protogcn)      SCRIPT="training/train_protogcn_only.py"; HAS_OOD_FLAG=0;;
    skateformer)   SCRIPT="training/train_skateformer_only.py"; HAS_OOD_FLAG=0;;
    hyperformer)   SCRIPT="training/train_hyperformer_full.py"; HAS_OOD_FLAG=1;;
    freqmixformer) SCRIPT="training/train_freqmixformer_full.py"; HAS_OOD_FLAG=1;;
    *) echo "Unknown model: $MODEL" >&2; exit 1;;
esac

FLAG=""
case "$MODE" in
    iid)   FLAG="--iid-only";;
    mixed) FLAG="--mixed-only";;
    ood)   if [[ "$HAS_OOD_FLAG" == "1" ]]; then FLAG="--ood-only"; else echo "NOTE: $SCRIPT has no --ood-only flag; running all sections." >&2; fi;;
    all)   FLAG="";;
    *) echo "Unknown mode: $MODE" >&2; exit 1;;
esac

mkdir -p output
echo "Running: python $SCRIPT $FLAG"
python -u "$SCRIPT" $FLAG
