"""Phase B for the affinity+LSD model: run the trained affinity model over real
VAST skeleton seeds against the full hard-drive EM stack, mirroring
phase_b_infer.py's walk-the-subsampled-trace structure but swapping in
affinity+seeded-mutex-watershed inference per patch instead of the seeded
per-instance mask model.

Unlike the seeded model, this network is never told which neuron it's
looking at, so each patch is seeded with EVERY real tree that happens to
have a node inside it (not just the target tree) -- exactly the
multi-seed-per-patch setup that was confirmed (2026-07-17) to correctly keep
touching real neurons separate, instead of a single point per patch.

Output format is a raw, mergeable intermediate (per-seed packed-bit target
mask + placement, same idea as phase_b_infer.py) rather than a final
VAST-importable file -- what VAST accepts for import is still an open
question (see PLAN.md/CLAUDE.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .affinity_infer import run_affinity_inference, seeded_agglomerate
from .affinity_model import AffinityLSDUNet3D
from .phase_b_stack import load_vsvi, read_region_centered
from .vast_skeleton import DEFAULT_SKELETON_CSV, load_skeletons, subsample_seeds


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tree-id", type=int, required=True, help="skeleton/tree id from the CSV")
    p.add_argument("--skeleton-csv", type=Path, default=DEFAULT_SKELETON_CSV)
    p.add_argument(
        "--vsvi", type=Path, required=True,
        help=r"path to the full stack's volume.vsvi, e.g. F:\ppa_b4v5s13\aligned_stack\volume.vsvi "
        "(requires the hard drive to be attached)",
    )
    p.add_argument("--target-spacing-nm", type=float, default=500.0)
    p.add_argument("--max-seeds", type=int, default=None, help="cap seeds processed, for a quick test run")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/phase_b_affinity"))
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    cfg = load_vsvi(args.vsvi)
    print(
        f"Full stack: {cfg.size_x}x{cfg.size_y}x{cfg.size_z} voxels (x,y,z), "
        f"scale_nm(x,y,z)={cfg.scale_nm_xyz}"
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    predict_lsd = not train_args.get("no_lsd", False)
    offsets = [tuple(o) for o in train_args["offsets"]]
    model = AffinityLSDUNet3D(
        num_offsets=len(offsets), base_channels=train_args["base_channels"], predict_lsd=predict_lsd
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    patch_shape_zyx = tuple(train_args["patch_size"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, patch_shape_zyx={patch_shape_zyx}, offsets={offsets}")

    trees = load_skeletons(args.skeleton_csv)
    if args.tree_id not in trees:
        raise ValueError(f"tree_id {args.tree_id} not found in {args.skeleton_csv} ({len(trees)} trees present)")
    nodes = trees[args.tree_id]
    seeds = subsample_seeds(nodes, cfg.scale_nm_xyz, target_spacing_nm=args.target_spacing_nm)
    print(f"Tree {args.tree_id}: {len(nodes)} traced nodes -> {len(seeds)} subsampled seeds")
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]
        print(f"--max-seeds set: only running the first {len(seeds)}")

    # One flat array of every real node from every tree, so each patch's "who else is in
    # here" lookup is a vectorized bounding-box filter instead of a per-tree Python loop --
    # cheap even across all ~213k real nodes and hundreds of patches.
    all_tids, all_x, all_y, all_z, all_local_ids = [], [], [], [], []
    for tid, tnodes in trees.items():
        for n in tnodes:
            all_tids.append(tid)
            all_x.append(n.x)
            all_y.append(n.y)
            all_z.append(n.z)
            all_local_ids.append(n.local_id)
    all_tids = np.array(all_tids)
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)

    out_dir = args.output_dir / f"tree_{args.tree_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pz, py, px = patch_shape_zyx
    seed_local_ids, seed_xyz, patch_origins_zyx, packed_masks = [], [], [], []
    for node in tqdm(seeds, desc=f"tree {args.tree_id} seeds", unit="seed"):
        center_zyx = (node.z, node.y, node.x)
        origin_zyx = tuple(c - p // 2 for c, p in zip(center_zyx, patch_shape_zyx))
        raw_patch = read_region_centered(cfg, center_zyx, patch_shape_zyx)

        in_patch = (
            (all_z >= origin_zyx[0]) & (all_z < origin_zyx[0] + pz)
            & (all_y >= origin_zyx[1]) & (all_y < origin_zyx[1] + py)
            & (all_x >= origin_zyx[2]) & (all_x < origin_zyx[2] + px)
        )
        seed_points: dict[int, list[tuple[int, int, int]]] = {}
        for tid, x, y, z in zip(all_tids[in_patch], all_x[in_patch], all_y[in_patch], all_z[in_patch]):
            seed_points.setdefault(int(tid), []).append((int(z - origin_zyx[0]), int(y - origin_zyx[1]), int(x - origin_zyx[2])))

        aff_probs, _lsd_pred = run_affinity_inference(model, raw_patch, device)
        pred_labels = seeded_agglomerate(aff_probs, seed_points, offsets)
        mask = pred_labels == args.tree_id

        seed_local_ids.append(node.local_id)
        seed_xyz.append((node.x, node.y, node.z))
        patch_origins_zyx.append(list(origin_zyx))
        packed_masks.append(np.packbits(mask))

    np.savez_compressed(
        out_dir / "predictions.npz",
        tree_id=args.tree_id,
        patch_shape_zyx=np.array(patch_shape_zyx),
        seed_local_ids=np.array(seed_local_ids),
        seed_xyz=np.array(seed_xyz),
        patch_origin_zyx=np.array(patch_origins_zyx),
        packed_masks=np.stack(packed_masks),
    )
    print(f"Saved {len(seeds)} patch predictions to {out_dir / 'predictions.npz'}")


if __name__ == "__main__":
    main()
