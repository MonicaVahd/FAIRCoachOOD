"""
Recompute OOD macro F1 from saved confusion-matrix counts.

Reads output/confusion_all_models.txt (198 OOD cells: 9 models × 22 conditions),
compares against the legacy class-1 F1 in output/*_results.txt, and writes
output/ood_results_corrected.csv without modifying any existing result files.

Usage
-----
    cd <project_root>
    python training/recompute_ood_macro_f1.py
"""

from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
CONFUSION_PATH = os.path.join(OUTPUT_DIR, "confusion_all_models.txt")
CORRECTED_CSV = os.path.join(OUTPUT_DIR, "ood_results_corrected.csv")

NEAR_DEG_THRESHOLD = 5
SUBSTANTIAL_DIFF_PP = 5.0

CANONICAL_ORDER = [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B9", "B10", "B11", "B12",
    "C1", "C2", "C3", "C4", "C5", "C6",
    "C7", "C8", "C9", "C10",
]

RESULT_FILES: Dict[str, str] = {
    "ST-GCN": "stgcn_results.txt",
    "CTR-GCN": "ctrgcn_results.txt",
    "InfoGCN": "infogcn_results.txt",
    "BlockGCN": "blockgcn_results.txt",
    "HD-GCN": "hdgcn_results.txt",
    "ProtoGCN": "protogcn_results.txt",
    "SkateFormer": "skateformer_results.txt",
    "HyperFormer": "hyperformer_results.txt",
    "FreqMixFormer": "freqmixformer_results.txt",
}

MODEL_HEADER_RE = re.compile(r"^#\s+(.+?)\s*$")
CONFUSION_TAG_RE = re.compile(
    r"Confusion Matrix for\s+([BC]\d+)[a-z]?\s",
    re.IGNORECASE,
)
COUNTS_RE = re.compile(
    r"True Positives=(\d+)\s+True Negatives=(\d+)\s+"
    r"False Positives=(\d+)\s+False Negatives=(\d+)"
)
OOD_LINE_RE = re.compile(r"^\s+([BC])(\d+)[a-z]?\s")
BARE_PCT_RE = re.compile(r"(?<![±\d])(\d+\.\d+)%")
STATUS_RE = re.compile(r"\s(\d+)\s+(DEG[^\s]*|~DEG|OK)\s*$")


@dataclass
class ConfusionCell:
    model: str
    condition: str
    tp: int
    tn: int
    fp: int
    fn: int


def classify_degeneracy(pred_correct: int, pred_incorrect: int) -> str:
    if pred_correct == 0 or pred_incorrect == 0:
        return "DEG†"
    if pred_correct <= NEAR_DEG_THRESHOLD or pred_incorrect <= NEAR_DEG_THRESHOLD:
        return "~DEG"
    return "OK"


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict:
    """Same formulas as training/metrics_utils.compute_metrics for binary F1."""
    n = tp + tn + fp + fn
    pred_correct = tp + fp
    pred_incorrect = tn + fn

    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0
    macro_f1 = (f1 + f1_0) / 2.0

    return {
        "f1": f1,
        "f1_0": f1_0,
        "macro_f1": macro_f1,
        "pred_correct": pred_correct,
        "pred_incorrect": pred_incorrect,
        "n": n,
    }


def parse_confusion_file(path: str) -> List[ConfusionCell]:
    cells: List[ConfusionCell] = []
    current_model: Optional[str] = None
    pending_tag: Optional[str] = None

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            header = MODEL_HEADER_RE.match(line.strip())
            if header:
                current_model = header.group(1).strip()
                continue

            tag_m = CONFUSION_TAG_RE.search(line)
            if tag_m:
                pending_tag = tag_m.group(1).upper()
                if pending_tag.startswith("B") or pending_tag.startswith("C"):
                    # Normalise B10 not b10 — already upper from pattern
                    pass
                continue

            counts_m = COUNTS_RE.search(line)
            if counts_m and current_model and pending_tag:
                tp, tn, fp, fn = (int(x) for x in counts_m.groups())
                cells.append(
                    ConfusionCell(
                        model=current_model,
                        condition=pending_tag,
                        tp=tp,
                        tn=tn,
                        fp=fp,
                        fn=fn,
                    )
                )
                pending_tag = None

    return cells


def parse_old_results(path: str) -> Dict[str, Tuple[float, str]]:
    """Return {condition: (old_f1_percent, status)} for OOD rows."""
    out: Dict[str, Tuple[float, str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = OOD_LINE_RE.match(line)
            if not m:
                continue
            key = f"{m.group(1)}{m.group(2)}"
            pcts = BARE_PCT_RE.findall(line)
            if len(pcts) < 4:
                continue
            old_f1 = float(pcts[3])
            status_m = STATUS_RE.search(line.rstrip())
            status = status_m.group(2) if status_m else ""
            out[key] = (old_f1, status)
    return out


def load_all_old_results() -> Dict[str, Dict[str, Tuple[float, str]]]:
    by_model: Dict[str, Dict[str, Tuple[float, str]]] = {}
    for model, fname in RESULT_FILES.items():
        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
            # The legacy per-model *_results.txt files are not shipped in this repository; they only feed the
            # old_f1 / diff / status_old comparison columns.  The macro F1 and status columns do not need them.
            print(f"[NOTE] legacy results file not found, old_f1 columns left empty for {model}: {path}")
            by_model[model] = {}
            continue
        by_model[model] = parse_old_results(path)
    return by_model


def normalize_status(s: str) -> str:
    """Treat DEG variants as equivalent for comparison."""
    if s.startswith("DEG"):
        return "DEG†"
    return s


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not os.path.exists(CONFUSION_PATH):
        print(f"[ERROR] Missing {CONFUSION_PATH}")
        sys.exit(1)

    cells = parse_confusion_file(CONFUSION_PATH)
    old_by_model = load_all_old_results()

    rows: List[dict] = []
    status_mismatches: List[str] = []
    missing_old: List[str] = []

    for cell in cells:
        old_map = old_by_model.get(cell.model, {})
        old_entry = old_map.get(cell.condition)
        if old_entry is None:
            missing_old.append(f"{cell.model} {cell.condition}")
            old_f1 = float("nan")
            status_old = ""
        else:
            old_f1, status_old = old_entry

        m = metrics_from_counts(cell.tp, cell.tn, cell.fp, cell.fn)
        status_new = classify_degeneracy(m["pred_correct"], m["pred_incorrect"])

        if old_entry and normalize_status(status_new) != normalize_status(status_old):
            status_mismatches.append(
                f"{cell.model} {cell.condition}: "
                f"computed={status_new!r} file={status_old!r} "
                f"(pred_correct={m['pred_correct']}, pred_incorrect={m['pred_incorrect']})"
            )

        f1_pct = round(m["f1"] * 100, 4)
        f1_0_pct = round(m["f1_0"] * 100, 4)
        macro_pct = round(m["macro_f1"] * 100, 4)
        diff_pp = round(macro_pct - old_f1, 4) if old_entry else float("nan")

        rows.append(
            {
                "model": cell.model,
                "condition": cell.condition,
                "tp": cell.tp,
                "tn": cell.tn,
                "fp": cell.fp,
                "fn": cell.fn,
                "f1": f1_pct,
                "f1_0": f1_0_pct,
                "macro_f1": macro_pct,
                "old_f1": old_f1,
                "diff": diff_pp,
                "status": status_new,
                "status_old": status_old,
            }
        )

    expected = len(RESULT_FILES) * len(CANONICAL_ORDER)
    if len(rows) != expected:
        print(f"[WARN] Expected {expected} rows, got {len(rows)}")

    legacy_present = any(old_by_model.values())
    if missing_old and legacy_present:
        print(f"[ERROR] Missing old F1 for {len(missing_old)} cells:")
        for item in missing_old[:10]:
            print(f"  {item}")
        sys.exit(1)

    if status_mismatches:
        print(f"[WARN] Status mismatch on {len(status_mismatches)} cells:")
        for item in status_mismatches[:10]:
            print(f"  {item}")
        if len(status_mismatches) > 10:
            print(f"  ... and {len(status_mismatches) - 10} more")

    fieldnames = [
        "model", "condition", "tp", "tn", "fp", "fn",
        "f1", "f1_0", "macro_f1", "old_f1", "diff", "status", "status_old",
    ]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CORRECTED_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {CORRECTED_CSV}")

    # Sanity: ST-GCN B1
    b1 = next(r for r in rows if r["model"] == "ST-GCN" and r["condition"] == "B1")
    print(
        f"Sanity ST-GCN B1: tp={b1['tp']} tn={b1['tn']} fp={b1['fp']} fn={b1['fn']} "
        f"old_f1={b1['old_f1']} macro_f1={b1['macro_f1']} diff={b1['diff']}"
    )

    substantial = [r for r in rows if abs(r["diff"]) > SUBSTANTIAL_DIFF_PP]
    print(
        f"\nCells with |diff| > {SUBSTANTIAL_DIFF_PP:.0f} pp: "
        f"{len(substantial)} / {len(rows)}"
    )

    top10 = sorted(rows, key=lambda r: abs(r["diff"]), reverse=True)[:10]
    print("\nTop 10 |diff| (model, condition, old_f1, macro_f1, diff):")
    for r in top10:
        print(
            f"  {r['model']:<14} {r['condition']:<5} "
            f"old={r['old_f1']:6.1f}  macro={r['macro_f1']:6.1f}  "
            f"diff={r['diff']:+6.1f}  status={r['status']}"
        )


if __name__ == "__main__":
    main()
