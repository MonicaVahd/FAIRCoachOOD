"""
Shared binary classification metrics for training / evaluation scripts.

Class 1 = correct squat (positive class for legacy ``f1`` column).
``macro_f1`` = mean of per-class F1 for classes 0 and 1.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

POS_CLASS = 1


def iid_results_path(output_dir: str, model_slug: str) -> str:
    """Path for IID-only summary text (never overwrites *_results.txt OOD rows)."""
    return os.path.join(output_dir, f"iid_results_{model_slug}.txt")


def mixed_results_path(output_dir: str, model_slug: str) -> str:
    """Path for D1 mixed-only summary text (never overwrites *_results.txt or iid_results_*.txt)."""
    return os.path.join(output_dir, f"mixed_results_{model_slug}.txt")


def confmat_log_path(output_dir: str, model_slug: str) -> str:
    """Per-model confusion-matrix CSV (avoids wiping a shared log file)."""
    return os.path.join(output_dir, f"confusion_matrices_log_{model_slug}.csv")


def save_summary_to_file(lines: List[str], path: str) -> None:
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved -> {path}")


def build_summary_lines(
    summary: List[Tuple[str, str, str]],
    title: str,
    hdr: str,
    sep: str,
    section_labels: Dict[str, str],
) -> List[str]:
    lines = ["", "=" * 70, title, "=" * 70, hdr]
    last_sec = None
    for sec, _tag, row_str in summary:
        if sec != last_sec:
            lines.append(sep)
            lines.append(f"  [{section_labels.get(sec, sec)}]")
            last_sec = sec
        lines.append(row_str)
    lines.append(sep)
    return lines


def print_and_save_summary(
    summary: List[Tuple[str, str, str]],
    title: str,
    hdr: str,
    sep: str,
    section_labels: Dict[str, str],
    save_path: Optional[str] = None,
) -> None:
    lines = build_summary_lines(summary, title, hdr, sep, section_labels)
    print("\n".join(lines))
    if save_path:
        save_summary_to_file(lines, save_path)


CONFMAT_CSV_HEADER = [
    "source",
    "model",
    "tag",
    "round",
    "n",
    "tp",
    "tn",
    "fp",
    "fn",
    "acc",
    "prec",
    "rec",
    "f1",
    "prec0",
    "rec0",
    "f1_0",
    "macro_f1",
]


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    pos_class: int = POS_CLASS,
) -> dict:
    """Return accuracy, per-class metrics, macro F1, and confusion counts."""
    tp = int(((preds == pos_class) & (labels == pos_class)).sum())
    fp = int(((preds == pos_class) & (labels != pos_class)).sum())
    fn = int(((preds != pos_class) & (labels == pos_class)).sum())
    tn = int(((preds != pos_class) & (labels != pos_class)).sum())
    pred_correct = int((preds == pos_class).sum())
    pred_incorrect = int((preds != pos_class).sum())

    acc = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0

    macro_f1 = (f1 + f1_0) / 2.0

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "prec0": prec0,
        "rec0": rec0,
        "f1_0": f1_0,
        "macro_f1": macro_f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": len(labels),
        "pred_correct": pred_correct,
        "pred_incorrect": pred_incorrect,
    }


class ConfmatLogger:
    """Append one CSV row per evaluation for future metric audits."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path

    def reset(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CONFMAT_CSV_HEADER).writeheader()

    def ensure_header(self) -> None:
        """Create file with header only if it does not exist (never truncates)."""
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CONFMAT_CSV_HEADER).writeheader()

    def append(
        self,
        source: str,
        model: str,
        tag: str,
        metrics: dict,
        round_id: str = "",
    ) -> None:
        row: Dict[str, object] = {
            "source": source,
            "model": model,
            "tag": tag,
            "round": round_id,
            "n": metrics["n"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "acc": round(metrics["acc"], 6),
            "prec": round(metrics["prec"], 6),
            "rec": round(metrics["rec"], 6),
            "f1": round(metrics["f1"], 6),
            "prec0": round(metrics["prec0"], 6),
            "rec0": round(metrics["rec0"], 6),
            "f1_0": round(metrics["f1_0"], 6),
            "macro_f1": round(metrics["macro_f1"], 6),
        }
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CONFMAT_CSV_HEADER)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
