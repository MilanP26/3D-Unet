#!/usr/bin/env python
"""Loads every Training Data/<Stack>/ folder, decodes it, verifies that the
decoded mask actually agrees with webKnossos's own recorded anchor point for
each segment (PLAN.md section 0/1 alignment check), and prints per-stack /
per-instance size stats to help pick patch size and a min-voxel cutoff.

Usage: py scripts/inspect_data.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seeded_unet.stack_io import load_all_stacks  # noqa: E402


def main():
    stacks = load_all_stacks()
    for stack in stacks:
        print(f"\n=== {stack.name} (scene_group={stack.scene_group!r}) ===")
        print(f"raw shape (Z,Y,X): {stack.raw.shape}, scale_nm (x,y,z): {stack.scale_nm}")

        mismatches = 0
        for sid, meta in stack.segments.items():
            if meta.anchor_xyz is None:
                continue
            x, y, z = meta.anchor_xyz
            if not (0 <= z < stack.labels.shape[0] and 0 <= y < stack.labels.shape[1] and 0 <= x < stack.labels.shape[2]):
                continue
            actual = stack.labels[z, y, x]
            if actual != sid:
                mismatches += 1
                print(f"  ALIGNMENT MISMATCH: segment {sid} anchor(x={x},y={y},z={z}) -> label {actual}")
        n_checked = sum(1 for m in stack.segments.values() if m.anchor_xyz is not None)
        print(f"Anchor-point alignment check: {n_checked - mismatches}/{n_checked} segments verified OK")

        ids, counts = np.unique(stack.labels, return_counts=True)
        fg = [(i, c) for i, c in zip(ids, counts) if i != 0]
        fg.sort(key=lambda t: -t[1])
        named = sum(1 for sid, _ in fg if stack.segments.get(int(sid)) and stack.segments[int(sid)].name)
        print(f"{len(fg)} labeled instances ({named} explicitly named/colored in webKnossos)")
        if fg:
            voxel_counts = np.array([c for _, c in fg])
            print(
                f"voxel count: min={voxel_counts.min()} median={int(np.median(voxel_counts))} "
                f"max={voxel_counts.max()}"
            )


if __name__ == "__main__":
    main()
