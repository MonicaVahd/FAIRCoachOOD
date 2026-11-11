"""
eval/rerun_status_changes.py -- count status changes between the original OOD results and the UI-PRMD rerun.

Inputs (both shipped in this repository):
  results/ood_results.csv                          columns model, condition, tp, tn, fp, fn, ...   (original runs)
  results/uiprmd_rerun/uiprmd_rerun_confusion.txt  confusion counts of the rerun (117 UI-PRMD-related cells)

Status of a cell, from its confusion counts (positive class = correct squat):
  n1 = tp + fp   (samples predicted "correct"),  n0 = tn + fn   (samples predicted "incorrect")
  FC  : n0 == 0 or n1 == 0
  PC  : not FC, and n0 <= 5 or n1 <= 5
  OK  : otherwise

Two ways of counting changes are printed (the paper does not state which one it uses):
  3-way  : recovery   = FC/PC -> OK
           regression = OK -> FC/PC
           FC<->PC    = FC -> PC or PC -> FC
  binary : FC -> not-FC ("recovery") and not-FC -> FC ("regression"); PC counts as not-FC

Usage:  python eval/rerun_status_changes.py
"""
import collections
import csv
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_CSV = os.path.join(ROOT, "results", "ood_results.csv")
NEW_TXT = os.path.join(ROOT, "results", "uiprmd_rerun", "uiprmd_rerun_confusion.txt")
CAT_A = {"B7", "B8", "B9", "C4", "C5", "C7", "C8", "C9", "C10"}   # retrained
CAT_B = {"B10", "B11", "B12", "C6"}                                # re-evaluated only


def status(tp, tn, fp, fn):
    n1, n0 = tp + fp, tn + fn
    if n0 == 0 or n1 == 0:
        return "FC"
    if n0 <= 5 or n1 <= 5:
        return "PC"
    return "OK"


def load_old(path=OLD_CSV):
    old = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            old[(r["model"], r["condition"])] = (int(r["tp"]), int(r["tn"]), int(r["fp"]), int(r["fn"]))
    return old


def load_new(path=NEW_TXT):
    """Parse the rerun confusion file.  Every 'Confusion Matrix for <cond>' line resets the current condition,
    including D1, so D1's counts are never attributed to the preceding condition."""
    txt = open(path, "rb").read()
    for enc in ("utf-8", "utf-16"):
        try:
            txt = txt.decode(enc)
            break
        except Exception:
            continue
    else:
        txt = txt.decode("utf-8", "replace")
    model, cond, new = None, None, {}
    for line in txt.splitlines():
        m = re.match(r"^#\s+([A-Za-z\- ]+?)\s*$", line)
        if m and not set(line.strip()) <= {"#"}:
            model = m.group(1).strip()
            continue
        m = re.search(r"Confusion Matrix for\s+([A-Z]\d+)[a-z]?\s", line)
        if m:
            cond = m.group(1)
            continue
        m = re.search(r"True Positives=(\d+)\s+True Negatives=(\d+)\s+False Positives=(\d+)\s+False Negatives=(\d+)", line)
        if m and model and cond and cond != "D1":
            new[(model, cond)] = tuple(int(x) for x in m.groups())
    return new


def count(old, new):
    three, binary = collections.Counter(), collections.Counter()
    by_cat = {"A": collections.Counter(), "B": collections.Counter()}
    differing = 0
    for key, nv in new.items():
        ov = old[key]
        differing += ov != nv
        so, sn = status(*ov), status(*nv)
        cat = "A" if key[1] in CAT_A else "B"
        if so != sn:
            if so in ("FC", "PC") and sn == "OK":
                kind = "recovery"
            elif so == "OK" and sn in ("FC", "PC"):
                kind = "regression"
            else:
                kind = "FC<->PC"
            three[kind] += 1
            by_cat[cat][kind] += 1
        fo, fn_ = so == "FC", sn == "FC"
        if fo and not fn_:
            binary["FC->not-FC"] += 1
        elif fn_ and not fo:
            binary["not-FC->FC"] += 1
    return three, binary, by_cat, differing


def main():
    old, new = load_old(), load_new()
    print(f"cells in rerun file: {len(new)} (expected 117)")
    three, binary, by_cat, differing = count(old, new)
    print(f"cells with different confusion counts: {differing}")
    print(f"3-way : {dict(three)}  total={sum(three.values())}")
    print(f"        category A (retrained): {dict(by_cat['A'])} | category B (re-evaluated only): {dict(by_cat['B'])}")
    print(f"binary: {dict(binary)}  total={sum(binary.values())}")


if __name__ == "__main__":
    main()
