"""
training/wilcoxon_model_comparison.py

Loads corrected OOD macro F1 scores (22 experiments each) for all 9 models from
output/ood_results_corrected.csv, then runs paired two-sided Wilcoxon
signed-rank tests for every model pair (36 pairs).

Output
------
  output/wilcoxon_results_corrected.csv  — macro-F1-based pairwise tests
  stdout                                 — summary table + legacy comparison

Usage
-----
    cd <project_root>
    python training/wilcoxon_model_comparison.py
"""

import csv
import itertools
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.stats import wilcoxon

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"

CORRECTED_CSV = OUTPUT_DIR / "ood_results_corrected.csv"
OUT_CSV = OUTPUT_DIR / "wilcoxon_results_corrected.csv"
LEGACY_CSV = OUTPUT_DIR / "wilcoxon_results.csv"

MODEL_ORDER = [
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

CANONICAL_ORDER = [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B9", "B10", "B11", "B12",
    "C1", "C2", "C3", "C4", "C5", "C6",
    "C7", "C8", "C9", "C10",
]

N_PAIRS = 36
BONFERRONI_ALPHA = 0.05 / N_PAIRS  # ≈ 0.001389


def load_macro_f1_from_corrected(path: Path) -> Dict[str, np.ndarray]:
    """
    Return {model_name: np.array shape (22,)} of macro_f1 percentages,
    aligned to CANONICAL_ORDER.
    """
    if not path.exists():
        print(f"[ERROR] Missing corrected CSV: {path}")
        sys.exit(1)

    by_model: Dict[str, Dict[str, float]] = {m: {} for m in MODEL_ORDER}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            model = row["model"]
            cond = row["condition"]
            if model not in by_model:
                print(f"[WARN] Unknown model in CSV: {model}")
                continue
            by_model[model][cond] = float(row["macro_f1"])

    model_macro: Dict[str, np.ndarray] = {}
    for model in MODEL_ORDER:
        missing = [k for k in CANONICAL_ORDER if k not in by_model[model]]
        if missing:
            print(f"[ERROR] {model}: missing conditions {missing}")
            sys.exit(1)
        vec = np.array([by_model[model][k] for k in CANONICAL_ORDER], dtype=float)
        model_macro[model] = vec
        print(
            f"  Loaded {model:14s}  mean_macro_F1={vec.mean():.2f}%  "
            f"n={len(vec)}"
        )
    return model_macro


def run_wilcoxon(model_macro: Dict[str, np.ndarray]) -> List[dict]:
    """Run all 36 pairwise two-sided Wilcoxon signed-rank tests."""
    models = list(model_macro.keys())
    rows: List[dict] = []

    for m_a, m_b in itertools.combinations(models, 2):
        a = model_macro[m_a]
        b = model_macro[m_b]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        except ValueError as exc:
            stat, p = float("nan"), float("nan")
            note = str(exc)
        else:
            note = ""

        direction = (
            f"{m_a} > {m_b}" if a.mean() > b.mean() else
            f"{m_b} > {m_a}" if b.mean() > a.mean() else
            "tie"
        )

        rows.append({
            "model_A":            m_a,
            "model_B":            m_b,
            "mean_macro_F1_A":    round(float(a.mean()), 4),
            "mean_macro_F1_B":    round(float(b.mean()), 4),
            "delta_mean":         round(float(a.mean() - b.mean()), 4),
            "direction":          direction,
            "W_statistic":        stat if stat == stat else "",
            "p_value":            round(p, 6) if p == p else "",
            "sig_p05":            ("Yes" if p == p and p < 0.05 else "No"),
            "sig_bonferroni":     ("Yes" if p == p and p < BONFERRONI_ALPHA else "No"),
            "bonferroni_alpha":   round(BONFERRONI_ALPHA, 6),
            "macro_f1_A":         ";".join(f"{v:.2f}" for v in a),
            "macro_f1_B":         ";".join(f"{v:.2f}" for v in b),
            "note":               note,
        })

    return rows


CSV_COLUMNS = [
    "model_A", "model_B",
    "mean_macro_F1_A", "mean_macro_F1_B", "delta_mean", "direction",
    "W_statistic", "p_value",
    "sig_p05", "sig_bonferroni", "bonferroni_alpha",
    "macro_f1_A", "macro_f1_B", "note",
]


def save_csv(rows: List[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {path}")


def print_ranking(model_macro: Dict[str, np.ndarray]) -> None:
    ranked = sorted(
        ((m, float(v.mean())) for m, v in model_macro.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\n  OOD mean macro-F1 ranking (22 experiments):")
    for rank, (model, mf1) in enumerate(ranked, 1):
        print(f"    {rank}. {model:14s}  {mf1:.2f}%")


def print_pairwise_summary(rows: List[dict]) -> None:
    sig05 = [r for r in rows if r["sig_p05"] == "Yes"]
    sig_bf = [r for r in rows if r["sig_bonferroni"] == "Yes"]

    print("\n" + "=" * 80)
    print("Wilcoxon signed-rank tests on corrected macro F1  (paired, two-sided, n=22)")
    print("=" * 80)
    header = (
        f"  {'Model A':14s}  {'Model B':14s}  "
        f"{'meanA':>7}  {'meanB':>7}  {'d_mean':>7}  {'p-value':>10}  "
        f"sig(0.05)  sig(Bonf)"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in rows:
        pval_str = f"{r['p_value']:.6f}" if r["p_value"] != "" else "     NaN"
        print(
            f"  {r['model_A']:14s}  {r['model_B']:14s}  "
            f"{r['mean_macro_F1_A']:>7.2f}  {r['mean_macro_F1_B']:>7.2f}  "
            f"{r['delta_mean']:>+7.2f}  {pval_str:>10}  "
            f"{'*' if r['sig_p05'] == 'Yes' else ' ':^9}  "
            f"{'*' if r['sig_bonferroni'] == 'Yes' else ' ':^9}"
        )

    print()
    print(f"  Significant pairs (p < 0.05, uncorrected):           {len(sig05)}/{len(rows)}")
    print(f"  Significant pairs (Bonferroni p < {BONFERRONI_ALPHA:.5f}):  {len(sig_bf)}/{len(rows)}")

    if sig05:
        print("\n  Uncorrected significant pairs (p < 0.05):")
        for r in sorted(sig05, key=lambda x: float(x["p_value"])):
            print(
                f"    {r['model_A']} vs {r['model_B']}: "
                f"p={float(r['p_value']):.6f}  "
                f"mean={r['mean_macro_F1_A']:.2f} vs {r['mean_macro_F1_B']:.2f}  "
                f"({r['direction']})"
            )
    else:
        print("\n  No uncorrected significant pairs (p < 0.05).")

    if sig_bf:
        print(f"\n  Bonferroni-significant pairs (p < {BONFERRONI_ALPHA:.5f}):")
        for r in sorted(sig_bf, key=lambda x: float(x["p_value"])):
            print(
                f"    {r['model_A']} vs {r['model_B']}: "
                f"p={float(r['p_value']):.6f}  "
                f"mean={r['mean_macro_F1_A']:.2f} vs {r['mean_macro_F1_B']:.2f}"
            )
    else:
        print(f"\n  No Bonferroni-significant pairs (p < {BONFERRONI_ALPHA:.5f}).")


def _pair_key(model_a: str, model_b: str) -> Tuple[str, str]:
    return tuple(sorted((model_a, model_b)))


def load_legacy_sig(path: Path, col: str) -> Set[Tuple[str, str]]:
    """Load set of significant pairs from legacy or corrected CSV."""
    if not path.exists():
        return set()
    out: Set[Tuple[str, str]] = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get(col) == "Yes":
                out.add(_pair_key(row["model_A"], row["model_B"]))
    return out


def compare_with_legacy(new_rows: List[dict]) -> None:
    if not LEGACY_CSV.exists():
        print("\n[NOTE] Legacy wilcoxon_results.csv not found — skipping comparison.")
        return

    old_p05 = load_legacy_sig(LEGACY_CSV, "sig_p05")
    old_bf = load_legacy_sig(LEGACY_CSV, "sig_bonferroni")
    new_p05 = {_pair_key(r["model_A"], r["model_B"]) for r in new_rows if r["sig_p05"] == "Yes"}
    new_bf = {_pair_key(r["model_A"], r["model_B"]) for r in new_rows if r["sig_bonferroni"] == "Yes"}

    lost_p05 = old_p05 - new_p05
    gained_p05 = new_p05 - old_p05
    lost_bf = old_bf - new_bf
    gained_bf = new_bf - old_bf

    print("\n" + "=" * 80)
    print("Comparison: legacy class-1 F1 vs corrected macro F1")
    print("=" * 80)

    print(f"\n  Uncorrected (p < 0.05):")
    print(f"    Legacy significant:   {len(old_p05)} pairs")
    print(f"    Corrected significant: {len(new_p05)} pairs")

    if lost_p05:
        print(f"\n    Significant under OLD only ({len(lost_p05)}):")
        for a, b in sorted(lost_p05):
            print(f"      {a} vs {b}")
    else:
        print("\n    Significant under OLD only: (none)")

    if gained_p05:
        print(f"\n    Significant under CORRECTED only ({len(gained_p05)}):")
        for a, b in sorted(gained_p05):
            print(f"      {a} vs {b}")
    else:
        print("\n    Significant under CORRECTED only: (none)")

    if not lost_p05 and not gained_p05:
        print("\n    Uncorrected significant set unchanged.")

    print(f"\n  Bonferroni (p < {BONFERRONI_ALPHA:.5f}):")
    print(f"    Legacy significant:   {len(old_bf)} pairs")
    print(f"    Corrected significant: {len(new_bf)} pairs")

    if lost_bf:
        print(f"\n    Bonferroni-significant under OLD only ({len(lost_bf)}):")
        for a, b in sorted(lost_bf):
            print(f"      {a} vs {b}")
    else:
        print("\n    Bonferroni-significant under OLD only: (none)")

    if gained_bf:
        print(f"\n    Bonferroni-significant under CORRECTED only ({len(gained_bf)}):")
        for a, b in sorted(gained_bf):
            print(f"      {a} vs {b}")
    else:
        print("\n    Bonferroni-significant under CORRECTED only: (none)")

    if not lost_bf and not gained_bf:
        print("\n    Bonferroni significant set unchanged (both empty).")


def main() -> None:
    print(f"Loading corrected macro F1 from {CORRECTED_CSV} …")
    model_macro = load_macro_f1_from_corrected(CORRECTED_CSV)

    print_ranking(model_macro)

    print(f"\nRunning {N_PAIRS} pairwise Wilcoxon signed-rank tests …")
    rows = run_wilcoxon(model_macro)

    print_pairwise_summary(rows)
    compare_with_legacy(rows)
    save_csv(rows, OUT_CSV)


if __name__ == "__main__":
    main()
