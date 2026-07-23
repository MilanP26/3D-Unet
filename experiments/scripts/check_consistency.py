#!/usr/bin/env python
"""Run the skeleton-consistency error oracle (#1) over one tree's phase-B
predictions and write a ranked proofreading worklist.

    py experiments/scripts/check_consistency.py \
        --predictions outputs/phase_b_affinity/tree_1/predictions.npz

Writes experiments/outputs/consistency_tree_<id>.csv (worst findings first) and
prints a summary. No GPU / model / retrain -- pure post-processing.
"""
import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from skeleton_priors.consistency import analyze_tree  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True, help="path to a tree's predictions.npz")
    p.add_argument("--skeleton-csv", type=Path, default=None, help="override skeleton CSV (default: Data/VAST_skeleton_data.csv)")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments" / "outputs")
    p.add_argument("--min-orphan-voxels", type=int, default=200)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = analyze_tree(
        args.predictions,
        skeleton_csv=args.skeleton_csv,
        min_orphan_voxels=args.min_orphan_voxels,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / f"consistency_tree_{report.tree_id}.csv"
    ranked = report.ranked()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "severity", "kind", "patch_index", "seed_local_id", "x", "y", "z", "detail"])
        for rank, fnd in enumerate(ranked, start=1):
            x, y, z = fnd.seed_xyz
            w.writerow([rank, fnd.severity, fnd.kind, fnd.patch_index, fnd.seed_local_id, x, y, z, fnd.detail])

    counts = report.counts()
    fg = report.fg_fractions
    print(f"Tree {report.tree_id}: {report.num_patches} patches analyzed")
    print(f"  foreground fraction: min {min(fg):.1%}  mean {sum(fg)/len(fg):.1%}  max {max(fg):.1%}")
    print(f"  total findings: {len(report.findings)}")
    for kind in ("MERGE_LEAK", "ORPHAN_BLOB", "SPLIT", "SEED_UNCOVERED", "DEGENERATE"):
        if counts.get(kind):
            print(f"    {kind:<15} {counts[kind]}")
    print(f"\nRanked worklist written to {out_csv}")
    if ranked:
        print("\nTop findings (proofread these first):")
        for fnd in ranked[:10]:
            print(f"  [{fnd.severity:>5.0f}] {fnd.kind:<13} patch {fnd.patch_index:<4} node {fnd.seed_local_id:<5} -- {fnd.detail}")
    else:
        print("\nNo inconsistencies found -- the segmentation is self-consistent with the skeletons.")


if __name__ == "__main__":
    main()
