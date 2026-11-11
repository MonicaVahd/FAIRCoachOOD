"""
scripts/fetch_upstream.py -- download the upstream source files of the three models that are NOT
included in this repository (CTR-GCN, InfoGCN, FreqMixFormer) at a pinned commit.

Status of this script
---------------------
* Only the DOWNLOAD step is implemented: it clones each upstream repository, checks out the pinned
  commit and copies the listed upstream files, unmodified, to models/<name>/.
* The configuration changes of the paper (InfoGCN k=1 and dropout 0; FreqMixFormer frequency split d=9
  and Linear(36,18); OpenPose-18 graphs) are NOT applied by this script, and models/builders.py still
  raises ImportError for these three models.  Wiring the downloaded files into the builders is not
  implemented in this release.
* The commit used for the paper is UNKNOWN.  The hashes below are the head of each repository's default
  branch on the date given next to it; they are a reproducible choice, not a record of what was used.

Licences (read before downloading; nothing here grants you a licence)
  CTR-GCN        https://github.com/Uason-Chen/CTR-GCN          CC BY-NC 4.0 (non-commercial)
  InfoGCN        https://github.com/stnoah1/infogcn             no licence file
  FreqMixFormer  https://github.com/wenhanwu95/FreqMixFormer   no licence file

Usage (not run during release preparation)
  python scripts/fetch_upstream.py --model ctrgcn
  python scripts/fetch_upstream.py --model all --dry-run

If a commit hash is left as None the script stops with a non-zero exit code.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Hashes read from the GitHub pages of the repositories on 2026-11-10 (local date; 2026-11-11T02:11Z).
UPSTREAM = {
    "ctrgcn": {
        "repo": "https://github.com/Uason-Chen/CTR-GCN.git",
        "branch": "main",
        "commit": "e13d7582e281d06711eeecb380a472b278ae1663",   # committed 2021-11-20T07:32:55Z
        "source_files": ["model/ctrgcn.py"],
        "dest": os.path.join(ROOT, "models", "ctrgcn"),
        "licence": "CC BY-NC 4.0 (non-commercial)",
        "note": "Local development copy was identical to upstream model/ctrgcn.py (checked 2026-11-03).",
    },
    "infogcn": {
        "repo": "https://github.com/stnoah1/infogcn.git",
        "branch": "master",
        "commit": "873feaa85160317335a83e04013e0ffa3f63525e",   # committed 2022-11-22T20:00:40Z
        "source_files": ["model/infogcn.py", "model/activation.py", "model/modules.py", "model/ms_tcn.py"],
        "dest": os.path.join(ROOT, "models", "infogcn"),
        "licence": "no licence file in the repository",
        "note": "The paper uses neighbourhood order k=1 and dropout 0; not applied by this script.",
    },
    "freqmixformer": {
        "repo": "https://github.com/wenhanwu95/FreqMixFormer.git",
        "branch": "main",
        "commit": "1d3c2add885888f6eaa93135b9fb11dfdbc42be2",   # committed 2026-04-28T00:49:22Z
        "source_files": ["model/skefreqmixformer.py"],
        "dest": os.path.join(ROOT, "models", "freqmixformer"),
        "licence": "no licence file in the repository",
        "note": ("The development copy differed from upstream (line similarity about 0.59 when compared on "
                 "2026-11-03): frequency split d=9 and Linear(36,18) are not applied by this script."),
    },
}


def build_parser():
    p = argparse.ArgumentParser(description="Download upstream sources of the excluded models.")
    p.add_argument("--model", choices=list(UPSTREAM) + ["all"], default="all")
    p.add_argument("--tmp-dir", default=None, help="Clone directory (default: a new temporary directory)")
    p.add_argument("--dry-run", action="store_true", help="Print the steps without executing them")
    return p


def fetch_one(name, info, tmp_root, dry_run):
    print("=" * 60)
    print(f"Model:   {name}\nRepo:    {info['repo']}\nCommit:  {info['commit']}\nLicence: {info['licence']}")
    print(f"Note:    {info['note']}")
    if not info.get("commit"):
        print(f"ERROR: no pinned commit hash for '{name}'.  Fill in UPSTREAM['{name}']['commit'] first.", file=sys.stderr)
        sys.exit(2)
    clone_dir = os.path.join(tmp_root, name)
    steps = [["git", "clone", info["repo"], clone_dir], ["git", "-C", clone_dir, "checkout", info["commit"]]]
    if dry_run:
        for s in steps:
            print("  [dry-run]", " ".join(s))
        for f in info["source_files"]:
            print(f"  [dry-run] copy {f} -> {info['dest']}")
        return
    for s in steps:
        r = subprocess.run(s, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"ERROR: {' '.join(s)}\n{r.stderr}", file=sys.stderr)
            sys.exit(1)
    os.makedirs(info["dest"], exist_ok=True)
    for f in info["source_files"]:
        shutil.copy2(os.path.join(clone_dir, f), os.path.join(info["dest"], os.path.basename(f)))
    print(f"Copied upstream files to {info['dest']}")


def main():
    args = build_parser().parse_args()
    names = list(UPSTREAM) if args.model == "all" else [args.model]
    tmp_root = args.tmp_dir or tempfile.mkdtemp(prefix="oodbench_upstream_")
    for n in names:
        fetch_one(n, UPSTREAM[n], tmp_root, args.dry_run)
    print("Done.  The downloaded files are not wired into models/builders.py (see the header of this script).")


if __name__ == "__main__":
    main()
