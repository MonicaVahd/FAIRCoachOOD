"""
splits/make_split_indices.py — Generate split index files for the benchmark.

Writes JSON index lists (integers only — no raw data) for the conditions
reproducible from the three public datasets:
  - A1  EC3D LOSO (4 rounds)
  - A2  REHAB24-6 LOSO (9 rounds)
  - A4  UI-PRMD LOSO (10 rounds)
  - B1  EC3D → REHAB24 OOD (80/20 source split of EC3D)
  - B3  REHAB24 → EC3D OOD (80/20 source split of REHAB24)
  - B7  UI-PRMD → EC3D OOD
  - B8  UI-PRMD → REHAB24 OOD
  - B10 EC3D → UI-PRMD OOD
  - B11 REHAB24 → UI-PRMD OOD

Design:
  - Indices refer to the canonical sample order produced by the converters.
  - Seeds, numpy version, per-class counts, and SHA-256 hashes are recorded
    in every file under the "meta" key.
  - A --verify mode recomputes hashes from your dataset and stops if they
    differ (protects against a different dataset version or converter order).

NOTE: the index files shipped with a release were generated from the author's copies of the datasets. This script was
tested on synthetic data only (see docs/reproduction.md); it has not been compared with the stored split arrays of the
paper's runs.

Usage
-----
  python splits/make_split_indices.py \\
      --ec3d-pickle  /path/to/EC3D/data_3D.pickle \\
      --rehab24-csv  /path/to/REHAB24/2d_joints/Segmentation.csv \\
      --rehab24-data /path/to/REHAB24/2d_joints/Ex6 \\
      --uiprmd-dir   /path/to/ui_prmd \\
      --subjects-yaml configs/ec3d_subjects.yaml \\
      --out-dir splits/

  # Verify existing files:
  python splits/make_split_indices.py ... --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

SEED = 42
LOSO_VAL_FRACTION = 0.10
B_TRAIN_FRACTION  = 0.80
NUM_CLASSES = 2


# ── Canonical sample ordering ─────────────────────────────────────────────────

def _ec3d_canonical(pickle_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Return (labels_array, key_list) in canonical order.

    Canonical order: first-appearance order of (action, subject, exercise_type,
    repetition) keys from the pickle (grouped by (action, subject, ex, rep),
    sorted within each group by frame_id).  Matches ec3d_to_stgcn.py.

    Returns
    -------
    labels : np.ndarray of shape (N,) int64 — 1 = correct, 0 = incorrect
    keys   : list of str — '<subject>/<exercise_type>/<rep>' (no raw data)
    """
    with open(pickle_path, "rb") as f:
        raw = pickle.load(f)
    all_labels = np.array(raw["labels"])   # (N_frames, 5)
    all_poses  = np.array(raw["poses"])    # (N_frames, 3, 25) — not used here

    squat_mask = all_labels[:, 0] == "SQUAT"
    sq_labels  = all_labels[squat_mask]

    seq_dict: Dict[tuple, int] = {}  # key -> first_frame_index (for ordering)
    seq_label: Dict[tuple, int] = {}
    for i in range(len(sq_labels)):
        key = tuple(sq_labels[i, :4])
        if key not in seq_dict:
            seq_dict[key] = i   # insertion order = canonical order
            seq_label[key] = 1 if sq_labels[i, 2] == "1" else 0

    ordered_keys = sorted(seq_dict, key=lambda k: seq_dict[k])
    labels  = np.array([seq_label[k] for k in ordered_keys], dtype=np.int64)
    key_strs = [f"{k[1]}/{k[2]}/{k[3]}" for k in ordered_keys]  # no raw names in JSON
    return labels, key_strs


def _rehab24_canonical(csv_path: str, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (labels, person_ids) in canonical order (CSV row order, filtered).

    Filters: exercise_id == 6 and mocap_erroneous == 0 and file exists and end > start.
    Matches rehab24_to_stgcn.py.

    Returns
    -------
    labels     : np.ndarray (N,) int64
    person_ids : np.ndarray (N,) int64
    """
    df = pd.read_csv(csv_path, sep=";")
    df.columns = df.columns.str.strip()
    squat_df = df[(df["exercise_id"] == 6) & (df["mocap_erroneous"] == 0)].copy()

    labels_list: List[int] = []
    pids_list:   List[int] = []

    for _, row in squat_df.iterrows():
        vid   = row["video_id"]
        start = int(row["first_frame"])
        end   = int(row["last_frame"])
        label = int(row["correctness"])
        pid   = int(row["person_id"])
        npy_file = os.path.join(data_dir, f"{vid}-c17-30fps.npy")
        if not os.path.exists(npy_file):
            continue
        arr_len = np.load(npy_file, mmap_mode="r").shape[0]
        end = min(end, arr_len)
        if end <= start:
            continue
        labels_list.append(label)
        pids_list.append(pid)

    return np.array(labels_list, dtype=np.int64), np.array(pids_list, dtype=np.int64)


def _uiprmd_canonical(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (labels, subject_ids) in canonical order.

    Canonical order: sorted(os.listdir(correct_dir)) then sorted(os.listdir(incorrect_dir)),
    files starting with 'm01' and ending with '.txt'.  Matches uiprmd_to_stgcn.py.

    Returns
    -------
    labels      : np.ndarray (N,) int64
    subject_ids : np.ndarray (N,) int64
    """
    correct_dir   = os.path.join(data_dir, "correct")
    incorrect_dir = os.path.join(data_dir, "incorrect")
    labels_list: List[int] = []
    sids_list:   List[int] = []

    for directory, label in ((correct_dir, 1), (incorrect_dir, 0)):
        if not os.path.isdir(directory):
            continue
        for fname in sorted(os.listdir(directory)):
            if fname.startswith("m01") and fname.endswith(".txt"):
                m = re.search(r"_s(\d+)_", fname)
                if m:
                    labels_list.append(label)
                    sids_list.append(int(m.group(1)))

    return np.array(labels_list, dtype=np.int64), np.array(sids_list, dtype=np.int64)


# ── Hash helpers ─────────────────────────────────────────────────────────────

def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _meta(seed: int, n_samples: int, n_correct: int, n_incorrect: int,
          labels_sha256: str, **extra) -> dict:
    return {
        "seed": seed,
        "rng": "numpy.random.default_rng",
        "numpy_version": np.__version__,
        "n_samples": n_samples,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "labels_sha256": labels_sha256,
        **extra,
    }


# ── LOSO split ───────────────────────────────────────────────────────────────

def _loso_round(labels: np.ndarray, subject_ids: np.ndarray, subjects: List[Any],
                round_idx: int) -> dict:
    """
    Compute one LOSO round.

    Protocol (from split_loso.py:201-232):
      - test = all indices where subject_ids == subjects[round_idx]
      - pool = remaining indices in subject order
      - perm = default_rng(42).permutation(len(pool))
      - n_val = max(1, int(VAL_FRACTION * len(pool)))
      - val  = pool[perm[:n_val]]
      - train = pool[perm[n_val:]]
    """
    test_subj = subjects[round_idx]
    all_idx   = np.arange(len(labels))
    test_mask = (subject_ids == test_subj)
    test_idx  = all_idx[test_mask]

    pool_parts: List[np.ndarray] = []
    for s in subjects:
        if s == test_subj:
            continue
        pool_parts.append(all_idx[subject_ids == s])
    pool = np.concatenate(pool_parts) if pool_parts else np.array([], dtype=int)

    perm  = np.random.default_rng(SEED).permutation(len(pool))
    n_val = max(1, int(LOSO_VAL_FRACTION * len(pool)))
    val_idx   = pool[perm[:n_val]].tolist()
    train_idx = pool[perm[n_val:]].tolist()
    test_idx  = test_idx.tolist()

    return {
        "round": round_idx + 1,
        "test_subject_index": int(round_idx),
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
    }


def build_loso_splits(labels: np.ndarray, subject_ids: np.ndarray,
                      subjects: List[Any], dataset_name: str) -> dict:
    rounds = [
        _loso_round(labels, subject_ids, subjects, i)
        for i in range(len(subjects))
    ]
    return {
        "dataset": dataset_name,
        "protocol": "LOSO",
        "meta": _meta(
            seed=SEED,
            n_samples=len(labels),
            n_correct=int((labels == 1).sum()),
            n_incorrect=int((labels == 0).sum()),
            labels_sha256=_sha256(labels),
            val_fraction=LOSO_VAL_FRACTION,
            rule="test=all reps of held-out subject; pool=remaining in subject order; perm=default_rng(42); n_val=max(1,int(0.10*len(pool))); val=pool[perm[:n_val]]; train=pool[perm[n_val:]]",
        ),
        "rounds": rounds,
    }


# ── B-condition 80/20 source split ───────────────────────────────────────────

def _source_split_80_20(labels: np.ndarray) -> dict:
    """
    Protocol (from train_stgcn_full.py:135-150):
      perm = default_rng(42).permutation(N)
      n_tr = int(0.80 * N)
      train = perm[:n_tr], val = perm[n_tr:]
    """
    N    = len(labels)
    perm = np.random.default_rng(SEED).permutation(N)
    n_tr = int(B_TRAIN_FRACTION * N)
    return {
        "train": sorted(perm[:n_tr].tolist()),
        "val":   sorted(perm[n_tr:].tolist()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate and verify benchmark split index files."
    )
    p.add_argument("--ec3d-pickle",  required=True)
    p.add_argument("--rehab24-csv",  required=True)
    p.add_argument("--rehab24-data", required=True)
    p.add_argument("--uiprmd-dir",   required=True)
    p.add_argument("--subjects-yaml", default="configs/ec3d_subjects.yaml",
                   help="EC3D subject list YAML (see ec3d_subjects.example.yaml)")
    p.add_argument("--out-dir",      default="splits/")
    p.add_argument("--verify",       action="store_true",
                   help="Re-derive hashes and compare with existing files")
    return p


def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"  written → {path}")


def _verify_json(path: str, new_obj: dict) -> bool:
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return False
    with open(path, encoding="utf-8") as f:
        existing = json.load(f)
    new_sha = new_obj.get("meta", {}).get("labels_sha256")
    old_sha = existing.get("meta", {}).get("labels_sha256")
    if new_sha != old_sha:
        print(f"  [MISMATCH] {path}  new={new_sha[:12]}... old={old_sha[:12]}...")
        return False
    print(f"  [OK] {path}")
    return True


def main() -> None:
    args = build_parser().parse_args()

    # Load subject list
    with open(args.subjects_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ec3d_subjects = cfg["subjects"]
    print(f"EC3D subjects: {ec3d_subjects}")

    # ── Load canonical orderings ──────────────────────────────────────────────
    print("\nLoading EC3D canonical order...")
    ec3d_labels, ec3d_keys = _ec3d_canonical(args.ec3d_pickle)
    ec3d_subj_ids = np.array([ec3d_subjects.index(k.split("/")[0]) if k.split("/")[0] in ec3d_subjects else -1
                               for k in ec3d_keys], dtype=np.int64)
    # Subject IDs for LOSO: use subject name from key, map to index
    ec3d_subj_names = np.array([k.split("/")[0] for k in ec3d_keys])
    # Build integer subject ID array aligned with ec3d_subjects list
    ec3d_subj_int = np.array([ec3d_subjects.index(s) if s in ec3d_subjects else -1
                               for s in ec3d_subj_names], dtype=np.int64)
    if (ec3d_subj_int < 0).any():
        found = sorted(set(ec3d_subj_names.tolist()))
        sys.exit(f"ERROR: {int((ec3d_subj_int < 0).sum())} EC3D samples belong to subjects that are not in {args.subjects_yaml}. "
                 f"The pickle contains {len(found)} subject(s); list their names in the YAML file (in LOSO round order).")

    print(f"  EC3D: {len(ec3d_labels)} reps, correct={int((ec3d_labels==1).sum())}, incorrect={int((ec3d_labels==0).sum())}")

    print("Loading REHAB24 canonical order...")
    rehab_labels, rehab_pids = _rehab24_canonical(args.rehab24_csv, args.rehab24_data)
    rehab_subjects = sorted(set(rehab_pids.tolist()))
    print(f"  REHAB24: {len(rehab_labels)} reps, subjects={rehab_subjects}")

    print("Loading UI-PRMD canonical order...")
    ui_labels, ui_sids = _uiprmd_canonical(args.uiprmd_dir)
    ui_subjects = sorted(set(ui_sids.tolist()))
    print(f"  UI-PRMD: {len(ui_labels)} reps, subjects={ui_subjects}")

    # ── Build splits ──────────────────────────────────────────────────────────
    out = args.out_dir
    verify = args.verify
    ok = True

    print("\n── A1: EC3D LOSO ──────────────────────────────────────────────────")
    obj = build_loso_splits(ec3d_labels, ec3d_subj_int,
                            list(range(len(ec3d_subjects))), "EC3D")
    # Annotate test_subject_name as S1..S4 (not raw names)
    for r in obj["rounds"]:
        r["test_subject_label"] = f"S{r['test_subject_index']+1}"
    path = os.path.join(out, "ec3d_loso.json")
    if verify:
        ok &= _verify_json(path, obj)
    else:
        _write_json(path, obj)

    print("\n── A2: REHAB24-6 LOSO ──────────────────────────────────────────────")
    obj = build_loso_splits(rehab_labels, rehab_pids, rehab_subjects, "REHAB24-6")
    path = os.path.join(out, "rehab24_loso.json")
    if verify:
        ok &= _verify_json(path, obj)
    else:
        _write_json(path, obj)

    print("\n── A4: UI-PRMD LOSO ───────────────────────────────────────────────")
    obj = build_loso_splits(ui_labels, ui_sids, ui_subjects, "UI-PRMD")
    path = os.path.join(out, "uiprmd_loso.json")
    if verify:
        ok &= _verify_json(path, obj)
    else:
        _write_json(path, obj)

    print("\n── B-condition 80/20 source splits ────────────────────────────────")
    source_splits = {
        "ec3d":   _source_split_80_20(ec3d_labels),
        "rehab24": _source_split_80_20(rehab_labels),
        "uiprmd":  _source_split_80_20(ui_labels),
    }
    source_obj = {
        "meta": {"seed": SEED, "train_fraction": B_TRAIN_FRACTION,
                 "rule": "perm=default_rng(42).permutation(N); n_tr=int(0.80*N); train=perm[:n_tr]; val=perm[n_tr:]",
                 "ec3d_labels_sha256": _sha256(ec3d_labels),
                 "rehab24_labels_sha256": _sha256(rehab_labels),
                 "uiprmd_labels_sha256": _sha256(ui_labels)},
        **source_splits,
    }
    path = os.path.join(out, "source_80_20.json")
    if verify:
        ok &= _verify_json(path, source_obj)
    else:
        _write_json(path, source_obj)

    print("\n── OOD conditions (public only) ────────────────────────────────────")
    ood_obj = {
        "meta": {"note": "Indices into canonical sample order of each dataset"},
        "B1":  {"train_source": "ec3d",   "test_dataset": "rehab24", "test_indices": list(range(len(rehab_labels)))},
        "B3":  {"train_source": "rehab24","test_dataset": "ec3d",    "test_indices": list(range(len(ec3d_labels)))},
        "B7":  {"train_source": "uiprmd", "test_dataset": "ec3d",    "test_indices": list(range(len(ec3d_labels)))},
        "B8":  {"train_source": "uiprmd", "test_dataset": "rehab24", "test_indices": list(range(len(rehab_labels)))},
        "B10": {"train_source": "ec3d",   "test_dataset": "uiprmd",  "test_indices": list(range(len(ui_labels)))},
        "B11": {"train_source": "rehab24","test_dataset": "uiprmd",  "test_indices": list(range(len(ui_labels)))},
    }
    path = os.path.join(out, "ood_conditions.json")
    if verify:
        # Basic check: compare counts
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        for cond, d in ood_obj.items():
            if cond == "meta":
                continue
            if cond not in existing:
                print(f"  [MISSING cond] {cond}")
                ok = False
            elif len(existing[cond]["test_indices"]) != len(d["test_indices"]):
                print(f"  [COUNT MISMATCH] {cond}")
                ok = False
            else:
                print(f"  [OK] {cond}")
    else:
        _write_json(path, ood_obj)

    # ── MANIFEST ──────────────────────────────────────────────────────────────
    if not verify:
        import hashlib as _hl
        manifest = {}
        for fname in ["ec3d_loso.json", "rehab24_loso.json", "uiprmd_loso.json",
                      "source_80_20.json", "ood_conditions.json"]:
            fpath = os.path.join(out, fname)
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    manifest[fname] = _hl.sha256(f.read()).hexdigest()
        _write_json(os.path.join(out, "MANIFEST.json"), manifest)

    if verify:
        print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
        sys.exit(0 if ok else 1)
    else:
        print("\nSplit index files written.  Run with --verify to confirm reproducibility.")


if __name__ == "__main__":
    main()
