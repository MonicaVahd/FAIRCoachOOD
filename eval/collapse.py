"""
Compute FC / PC / OK collapse statistics from output/confusion_all_models.txt
and cross-check against output/ood_results_corrected.csv.

Definitions (from TP/TN/FP/FN):
  pred_correct   = TP + FP   (predicted class 1 / "correct squat")
  pred_incorrect = TN + FN   (predicted class 0 / "incorrect squat")
  FC  — one predicted class count is exactly 0
  PC  — one predicted class count is 1–5 (and not FC)
  OK  — otherwise

Maps to ood_results_corrected.csv status:
  FC <-> DEG†
  PC <-> ~DEG
  OK <-> OK
"""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
CONFUSION_PATH = os.path.join(OUTPUT_DIR, "confusion_all_models.txt")
CORRECTED_CSV = os.path.join(OUTPUT_DIR, "ood_results_corrected.csv")

NEAR_DEG_THRESHOLD = 5

CANONICAL_MODELS = [
    "ST-GCN",
    "CTR-GCN",
    "InfoGCN",
    "BlockGCN",
    "HD-GCN",
    "ProtoGCN",
    "SkateFormer",
    "HyperFormer",
    "FreqMixFormer",
]

CANONICAL_CONDITIONS = [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B9", "B10", "B11", "B12",
    "C1", "C2", "C3", "C4", "C5", "C6",
    "C7", "C8", "C9", "C10",
]

MODEL_HEADER_RE = re.compile(r"^#\s+(.+?)\s*$")
CONFUSION_TAG_RE = re.compile(
    r"Confusion Matrix for\s+([BC]\d+)[a-z]?\s",
    re.IGNORECASE,
)
COUNTS_RE = re.compile(
    r"True Positives=(\d+)\s+True Negatives=(\d+)\s+"
    r"False Positives=(\d+)\s+False Negatives=(\d+)"
)

FC_TO_STATUS = {"FC": "DEG†", "PC": "~DEG", "OK": "OK"}
STATUS_TO_FC = {v: k for k, v in FC_TO_STATUS.items()}


@dataclass(frozen=True)
class Cell:
    model: str
    condition: str
    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def pred_correct(self) -> int:
        return self.tp + self.fp

    @property
    def pred_incorrect(self) -> int:
        return self.tn + self.fn

    @property
    def collapse(self) -> str:
        pc, pi = self.pred_correct, self.pred_incorrect
        if pc == 0 or pi == 0:
            return "FC"
        if pc <= NEAR_DEG_THRESHOLD or pi <= NEAR_DEG_THRESHOLD:
            return "PC"
        return "OK"


def parse_confusion_file(path: str) -> List[Cell]:
    cells: List[Cell] = []
    current_model: Optional[str] = None
    pending_tag: Optional[str] = None

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            header = MODEL_HEADER_RE.match(line.strip())
            if header and line.strip().startswith("#"):
                current_model = header.group(1).strip()
                continue

            tag_m = CONFUSION_TAG_RE.search(line)
            if tag_m:
                pending_tag = tag_m.group(1).upper()
                continue

            counts_m = COUNTS_RE.search(line)
            if counts_m and current_model and pending_tag:
                tp, tn, fp, fn = (int(x) for x in counts_m.groups())
                cells.append(
                    Cell(
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


def load_csv_status(path: str) -> Dict[Tuple[str, str], str]:
    out: Dict[Tuple[str, str], str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[(row["model"], row["condition"])] = row["status"]
    return out


def normalize_status(s: str) -> str:
    if s.startswith("DEG"):
        return "DEG†"
    return s


def main() -> int:
    cells = parse_confusion_file(CONFUSION_PATH)
    csv_status = load_csv_status(CORRECTED_CSV)

    print(f"Source: {CONFUSION_PATH}")
    print(f"Cross-check: {CORRECTED_CSV}")
    print(f"Parsed cells: {len(cells)}")
    print()

    if len(cells) != 198:
        print(f"WARNING: expected 198 cells, got {len(cells)}", file=sys.stderr)

    # Index by (model, condition)
    grid: Dict[Tuple[str, str], Cell] = {}
    for c in cells:
        key = (c.model, c.condition)
        if key in grid:
            print(f"WARNING: duplicate cell {key}", file=sys.stderr)
        grid[key] = c

    # Cross-check vs CSV
    mismatches: List[str] = []
    missing_in_txt: List[Tuple[str, str]] = []
    missing_in_csv: List[Tuple[str, str]] = []

    for key, status in sorted(csv_status.items()):
        if key not in grid:
            missing_in_txt.append(key)
            continue
        computed = grid[key].collapse
        expected_status = normalize_status(status)
        expected_fc = STATUS_TO_FC.get(expected_status, "?")
        if computed != expected_fc:
            c = grid[key]
            mismatches.append(
                f"  {key[0]:<14} {key[1]:<4}  "
                f"computed={computed}  csv_status={status!r}  "
                f"pred_correct={c.pred_correct}  pred_incorrect={c.pred_incorrect}  "
                f"TP={c.tp} TN={c.tn} FP={c.fp} FN={c.fn}"
            )

    for key in sorted(grid.keys()):
        if key not in csv_status:
            missing_in_csv.append(key)

    print("=" * 72)
    print("CROSS-CHECK vs ood_results_corrected.csv status column")
    print("=" * 72)
    if missing_in_txt:
        print(f"Rows in CSV but missing in confusion file: {len(missing_in_txt)}")
        for k in missing_in_txt:
            print(f"  {k}")
    if missing_in_csv:
        print(f"Rows in confusion file but missing in CSV: {len(missing_in_csv)}")
        for k in missing_in_csv:
            print(f"  {k}")
    if mismatches:
        print(f"MISMATCHES: {len(mismatches)}")
        for m in mismatches:
            print(m)
    else:
        matched = len(csv_status) - len(missing_in_txt)
        print(f"All {matched} comparable cells match exactly (FC<->DEG†, PC<->~DEG, OK<->OK).")
    print()

    # Totals
    total_fc = sum(1 for c in cells if c.collapse == "FC")
    total_pc = sum(1 for c in cells if c.collapse == "PC")
    total_ok = sum(1 for c in cells if c.collapse == "OK")
    total = len(cells)
    collapsed = total_fc + total_pc

    print("=" * 72)
    print("AUTHORITATIVE TOTALS (198 OOD experiments)")
    print("=" * 72)
    print(f"  FC  (full collapse, 0 preds for one class):     {total_fc}")
    print(f"  PC  (partial collapse, 1–5 preds for one class): {total_pc}")
    print(f"  Collapsed (FC + PC):                           {collapsed}")
    print(f"  OK  (genuine):                                 {total_ok}")
    print(f"  Total:                                         {total}")
    if total:
        print(f"  Collapse percentage (FC+PC)/Total:             {100.0 * collapsed / total:.6f}%")
        print(f"  FC percentage:                                 {100.0 * total_fc / total:.6f}%")
        print(f"  PC percentage:                                 {100.0 * total_pc / total:.6f}%")
        print(f"  OK percentage:                                 {100.0 * total_ok / total:.6f}%")
    print()

    # Per-model breakdown
    print("=" * 72)
    print("PER-MODEL BREAKDOWN")
    print("=" * 72)
    print(f"  {'Model':<14} {'FC':>4} {'PC':>4} {'OK':>4} {'Total':>5} {'Collapse%':>10}")
    print("  " + "-" * 44)
    for model in CANONICAL_MODELS:
        sub = [c for c in cells if c.model == model]
        fc = sum(1 for c in sub if c.collapse == "FC")
        pc = sum(1 for c in sub if c.collapse == "PC")
        ok = sum(1 for c in sub if c.collapse == "OK")
        n = len(sub)
        pct = f"{100.0 * (fc + pc) / n:.2f}%" if n else "N/A"
        print(f"  {model:<14} {fc:>4} {pc:>4} {ok:>4} {n:>5} {pct:>10}")
    print()

    # Per-condition breakdown
    print("=" * 72)
    print("PER-CONDITION BREAKDOWN")
    print("=" * 72)
    print(f"  {'Cond':<6} {'FC':>4} {'PC':>4} {'OK':>4} {'Total':>5} {'Collapse%':>10}")
    print("  " + "-" * 36)
    for cond in CANONICAL_CONDITIONS:
        sub = [c for c in cells if c.condition == cond]
        fc = sum(1 for c in sub if c.collapse == "FC")
        pc = sum(1 for c in sub if c.collapse == "PC")
        ok = sum(1 for c in sub if c.collapse == "OK")
        n = len(sub)
        pct = f"{100.0 * (fc + pc) / n:.2f}%" if n else "N/A"
        print(f"  {cond:<6} {fc:>4} {pc:>4} {ok:>4} {n:>5} {pct:>10}")
    print()

    # Full grid for Table 6 verification
    print("=" * 72)
    print("FULL PER-CELL GRID (model × condition → FC / PC / OK)")
    print("  pred_correct=TP+FP, pred_incorrect=TN+FN")
    print("=" * 72)
    hdr = f"  {'':14}" + "".join(f"{c:>4}" for c in CANONICAL_CONDITIONS)
    print(hdr)
    print("  " + "-" * (14 + 4 * len(CANONICAL_CONDITIONS)))
    for model in CANONICAL_MODELS:
        row = f"  {model:<14}"
        for cond in CANONICAL_CONDITIONS:
            c = grid.get((model, cond))
            if c is None:
                row += "  --"
            else:
                row += f"{c.collapse:>4}"
        print(row)
    print()

    # Detailed grid with prediction counts
    print("=" * 72)
    print("DETAILED GRID (collapse | pred_correct | pred_incorrect)")
    print("=" * 72)
    for model in CANONICAL_MODELS:
        print(f"\n  [{model}]")
        for cond in CANONICAL_CONDITIONS:
            c = grid.get((model, cond))
            if c is None:
                print(f"    {cond}: MISSING")
            else:
                print(
                    f"    {cond}: {c.collapse:>2}  "
                    f"pred_correct={c.pred_correct:>3}  pred_incorrect={c.pred_incorrect:>3}  "
                    f"(TP={c.tp} TN={c.tn} FP={c.fp} FN={c.fn})"
                )

    return 1 if mismatches or missing_in_txt or missing_in_csv else 0


if __name__ == "__main__":
    raise SystemExit(main())
