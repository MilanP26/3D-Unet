"""Merges one neuron's per-seed patch predictions into a single mask and
writes it as a plain 8-bit grayscale PNG image stack that VAST can import
natively via 'File / Import / Import Segmentation from Images ...'
(VAST Lite 1.5.0 manual section 4.7) -- this sidesteps writing the
proprietary compressed `.vsseg` binary format entirely, which turned out to
have no verifiable ground truth to reverse-engineer against (see
PLAN.md/CLAUDE.md).

Per the manual: for plain grayscale (non-RGB) images, the pixel value *is*
the segment id directly (the RGB bit-splitting scheme only applies to 24-bit
RGB imports). This writes segment id 1 for predicted foreground, 0
elsewhere.

Merge strategy: a neuron's seeds are spaced along its trace (see
`vast_skeleton.subsample_seeds`) so neighboring seeds' 32x256x256 patches
overlap substantially. Overlapping predictions are combined with a logical
OR (union) -- simple, and errs toward inclusion rather than dropping a
neuron's real extent because one seed's patch missed it. A stricter
majority-vote merge is a natural follow-up if union proves too generous.

The exported box is a single dense rectangle covering the union bounding
box of ALL predicted foreground voxels (not just the seeds' patch extents),
per the researcher's request -- every z-slice in that box's range is written
(including all-background ones) rather than skipping empty slices, since
it's not confirmed VAST's image-stack importer tolerates gaps in the slice
sequence; blank slices cost almost nothing once PNG-compressed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_union_bbox(masks: np.ndarray, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mins, maxs) as inclusive (z, y, x) bounds, in absolute full-stack
    voxel coordinates, of every foreground voxel across all patches."""
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])
    for i in range(len(masks)):
        fg = np.argwhere(masks[i])
        if fg.size == 0:
            continue
        abs_coords = fg + origins[i]
        mins = np.minimum(mins, abs_coords.min(axis=0))
        maxs = np.maximum(maxs, abs_coords.max(axis=0))
    if not np.isfinite(mins).all():
        raise ValueError("No foreground voxels found across any patch -- nothing to export")
    return mins.astype(np.int64), maxs.astype(np.int64)


def export_mask_stack(
    predictions_npz: Path,
    out_dir: Path,
    segment_id: int = 1,
) -> dict:
    d = np.load(predictions_npz)
    patch_shape = tuple(d["patch_shape_zyx"].tolist())
    n_voxels = int(np.prod(patch_shape))
    masks = np.unpackbits(d["packed_masks"], axis=1)[:, :n_voxels].reshape(-1, *patch_shape).astype(bool)
    origins = d["patch_origin_zyx"]

    bbox_min, bbox_max = compute_union_bbox(masks, origins)
    z_min, y_min, x_min = bbox_min.tolist()
    z_max, y_max, x_max = bbox_max.tolist()
    y_extent = y_max - y_min + 1
    x_extent = x_max - x_min + 1
    z_extent = z_max - z_min + 1

    out_dir.mkdir(parents=True, exist_ok=True)
    pz, py, px = patch_shape

    for z in tqdm(range(z_min, z_max + 1), desc="writing slices", unit="slice"):
        canvas = np.zeros((y_extent, x_extent), dtype=bool)
        for i in range(len(masks)):
            oz, oy, ox = origins[i].tolist()
            if not (oz <= z < oz + pz):
                continue
            local_z = z - oz
            slab = masks[i, local_z]  # (py, px)
            # Where this patch's (y, x) window lands on the shared canvas -- clipped,
            # since the bbox only guarantees to contain foreground voxels, not every
            # patch's full window (a patch can extend past the tight foreground bbox).
            cy0, cx0 = oy - y_min, ox - x_min
            cy_lo, cy_hi = max(0, cy0), min(y_extent, cy0 + py)
            cx_lo, cx_hi = max(0, cx0), min(x_extent, cx0 + px)
            if cy_lo >= cy_hi or cx_lo >= cx_hi:
                continue
            canvas[cy_lo:cy_hi, cx_lo:cx_hi] |= slab[cy_lo - cy0:cy_hi - cy0, cx_lo - cx0:cx_hi - cx0]

        img = (canvas.astype(np.uint8) * segment_id)
        Image.fromarray(img, mode="L").save(out_dir / f"mask_z{z - z_min:04d}.png")

    manifest = {
        "start_x": int(x_min),
        "start_y": int(y_min),
        "start_z": int(z_min),
        "size_x": int(x_extent),
        "size_y": int(y_extent),
        "size_z": int(z_extent),
        "first_slice_index": 0,
        "last_slice_index": int(z_extent - 1),
        "filename_template": "mask_z%04d.png",
        "segment_id": segment_id,
    }
    with open(out_dir / "vast_import_params.txt", "w") as f:
        f.write(
            "VAST import instructions (File / Import / Import Segmentation from Images...)\n"
            "=============================================================================\n"
            f"File name template : mask_z%04d.png\n"
            f"Parameter order    : Slice (Z) only (single tile per section)\n"
            f"No of first slice (Z): 0\n"
            f"No of last slice (Z) : {z_extent - 1}\n"
            f"Start coordinates (in the full stack, absolute mip0 voxels):\n"
            f"  X = {x_min}\n"
            f"  Y = {y_min}\n"
            f"  Z = {z_min}\n"
            f"Image size: {x_extent} x {y_extent} pixels, {z_extent} slices\n"
            f"Pixel format: 8-bit grayscale, value {segment_id} = this neuron, 0 = background\n"
            "Check 'Import images in the order selected (stacks with one image per "
            "section only)' since this is a single-tile-per-slice stack.\n"
        )

    return manifest


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True, help="path to a tree_<id>/predictions.npz")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--segment-id", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = export_mask_stack(args.predictions, args.output_dir, args.segment_id)
    print(f"Wrote {manifest['size_z']} slices ({manifest['size_x']}x{manifest['size_y']} px) to {args.output_dir}")
    print(f"See {args.output_dir / 'vast_import_params.txt'} for the exact VAST import dialog values")


if __name__ == "__main__":
    main()
