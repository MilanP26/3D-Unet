"""Runs the affinity+LSD model over real neuron clumps found via slice_regions.py across
(a range of) the full stack, and writes VAST-importable segmentation PNG tiles.

Design directly follows what VAST's own manual documents for "Import Segmentation from
Images" (section 4.7, verified against the VAST Lite 1.5.0 manual, 2026-07-17):
  - full resolution is required (no mip-level flexibility for this import path)
  - placement is flexible: VAST natively supports importing a segmentation as a GRID OF
    SMALLER TILES per section (the manual's own example: 25x25 tiles of 1024x1024px each),
    the same convention already used for reading the raw EM stack -- so there's no need to
    write a full 102400x36864-voxel canvas per z-slice, only tiles for the regions that
    actually contain neurons, each tagged with its own placement.
  - segmentations are capped at 16-bit (65535 labels) and pixel value = segment id, 0 =
    background -- mwatershed's auto-generated orphan-fragment ids (arbitrary, can be in the
    millions) are NOT safe to export directly and are dropped (mapped to background) here;
    only real, known tree ids survive into the exported PNGs.

Oversized merged regions (the nerve ring itself: many real neurons packed densely enough
that their padded footprints all touch) are tiled into overlapping sub-windows -- each
tile trusts only its "core" (non-overlap) area once composited, so a neuron near a tile
boundary is never predicted with truncated context. Real tree ids stay globally consistent
across tiles for free (a seed's identity is the real tree id, not a per-tile label), so no
explicit cross-tile identity reconciliation is needed for seeded (real) neurons.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .affinity_infer import run_affinity_inference, seeded_agglomerate

MAX_TILE_XY = 768  # in-plane size safe for one forward pass, with headroom under what has
# already been confirmed to run cleanly ((12, 512, 512) and (40, 462, 371) windows)
TILE_OVERLAP = 128  # halo discarded from each tile's edge before compositing


def _tile_1d(lo: int, hi: int, size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Covers [lo, hi) with tiles of width `size`, each pair overlapping by `overlap`.
    Returns (read_start, read_end, core_start, core_end) -- only [core_start, core_end) of
    each tile's prediction should be trusted/composited; the rest is context-only halo.
    The first/last tile's core always reaches the true [lo, hi) edge (no halo trim there,
    since there's no neighboring tile on that side to need protecting from)."""
    if hi - lo <= size:
        return [(lo, hi, lo, hi)]
    stride = size - overlap
    starts = sorted(set(list(range(lo, hi - size, stride)) + [hi - size]))
    tiles = []
    for i, s in enumerate(starts):
        e = s + size
        core_s = lo if i == 0 else s + overlap // 2
        core_e = hi if i == len(starts) - 1 else e - overlap // 2
        tiles.append((s, e, core_s, core_e))
    return tiles


def tile_bbox_2d(
    y0: int, y1: int, x0: int, x1: int, max_tile_xy: int = MAX_TILE_XY, overlap: int = TILE_OVERLAP
) -> list[tuple[int, int, int, int, int, int, int, int]]:
    """Returns (ry0, ry1, rx0, rx1, cy0, cy1, cx0, cx1) per tile -- read bounds and core
    (trusted, composited) bounds, both in the same absolute voxel frame as the input bbox."""
    y_tiles = _tile_1d(y0, y1, max_tile_xy, overlap)
    x_tiles = _tile_1d(x0, x1, max_tile_xy, overlap)
    return [
        (ry0, ry1, rx0, rx1, cy0, cy1, cx0, cx1)
        for (ry0, ry1, cy0, cy1) in y_tiles
        for (rx0, rx1, cx0, cx1) in x_tiles
    ]


def process_region(
    cfg, model, offsets, device, read_region_fn, all_nodes_flat, z0: int, z1: int,
    y0: int, y1: int, x0: int, x1: int, keep_orphans: bool = False,
) -> np.ndarray:
    """Runs inference over one merged region (z0:z1, y0:y1, x0:x1), tiling internally if it's
    too big for one forward pass. Returns a (z1-z0, y1-y0, x1-x0) int32 label volume,
    composited from each tile's trusted core only.

    By default (keep_orphans=False) this contains ONLY real tree ids (0 = everything else) --
    the safe, exportable case, since mwatershed's auto-generated orphan-fragment ids are
    arbitrary and not consistent across tiles. Set keep_orphans=True to instead keep every
    discovered fragment (real or auto-discovered), remapped to fresh small ids that are at
    least unique across tiles -- useful for visually inspecting the model's own unsupervised
    discovery, at the cost of a real caveat: an auto-discovered object that happens to span a
    tile boundary will show up as two different ids on either side of that seam, since
    (unlike real tree ids) an auto-discovered fragment's identity isn't a global reference
    that different tiles could agree on independently.

    all_nodes_flat: (tid, x, y, z) arrays (see full_stack_export.build_node_arrays) covering
    every real node from every tree, so each tile's seed lookup is a vectorized bounding-box
    filter."""
    all_tids, all_x, all_y, all_z = all_nodes_flat
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.int32)
    next_orphan_id = 10_000  # clear of any real tree id, unique across the whole call

    for ry0, ry1, rx0, rx1, cy0, cy1, cx0, cx1 in tile_bbox_2d(y0, y1, x0, x1):
        raw_tile = read_region_fn(cfg, (z0, z1), (ry0, ry1), (rx0, rx1))

        in_tile = (
            (all_z >= z0) & (all_z < z1)
            & (all_y >= ry0) & (all_y < ry1)
            & (all_x >= rx0) & (all_x < rx1)
        )
        seed_points: dict[int, list[tuple[int, int, int]]] = {}
        for tid, x, y, z in zip(all_tids[in_tile], all_x[in_tile], all_y[in_tile], all_z[in_tile]):
            seed_points.setdefault(int(tid), []).append((int(z - z0), int(y - ry0), int(x - rx0)))

        aff_probs, _lsd = run_affinity_inference(model, raw_tile, device)
        pred_labels = seeded_agglomerate(aff_probs, seed_points, offsets) if seed_points else np.zeros(raw_tile.shape, dtype=np.int64)

        known_ids = set(seed_points.keys())
        if keep_orphans:
            tile_out = np.where(np.isin(pred_labels, list(known_ids)), pred_labels, 0) if known_ids else np.zeros_like(pred_labels)
            orphan_ids = sorted(set(np.unique(pred_labels).tolist()) - known_ids - {0})
            for oid in orphan_ids:
                tile_out[pred_labels == oid] = next_orphan_id
                next_orphan_id += 1
        else:
            # Only real, known tree ids are trustworthy to export (see docstring) --
            # everything else (background, unmerged orphan fragments) becomes 0.
            tile_out = np.where(np.isin(pred_labels, list(known_ids)), pred_labels, 0) if known_ids else np.zeros_like(pred_labels)

        # Composite only this tile's trusted core into the region output, in the region's
        # own local coordinate frame.
        core = tile_out[:, cy0 - ry0:cy1 - ry0, cx0 - rx0:cx1 - rx0]
        out[:, cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = core

    return out


def build_node_arrays(trees: dict[int, list]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tids, xs, ys, zs = [], [], [], []
    for tid, nodes in trees.items():
        for n in nodes:
            tids.append(tid)
            xs.append(n.x)
            ys.append(n.y)
            zs.append(n.z)
    return np.array(tids), np.array(xs), np.array(ys), np.array(zs)


def remap_ids(label_volume: np.ndarray, id_map: dict[int, int]) -> np.ndarray:
    """Remaps real tree ids to a small, dense, gap-free range (VAST prefers this -- see
    module docstring) via a vectorized lookup table. Background (0) always maps to 0."""
    if not id_map:
        return np.zeros_like(label_volume, dtype=np.uint16)
    max_id = int(label_volume.max())
    lut = np.zeros(max_id + 1, dtype=np.uint16)
    for real_id, new_id in id_map.items():
        if real_id <= max_id:
            lut[real_id] = new_id
    return lut[label_volume]


def write_region_pngs(
    label_volume: np.ndarray, z0: int, y0: int, x0: int, out_dir: Path, tag: str,
) -> list[dict]:
    """One 16-bit PNG per z-slice that has any nonzero content, named/placed for VAST's
    tiled segmentation import (section 4.7 of the manual): pixel value = remapped segment
    id, 0 = background. Returns a manifest (list of dicts) with each written file's exact
    z-slice and (x0, y0) start coordinates, for the "Start coordinates" field on import."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(label_volume.shape[0]):
        z = z0 + i
        slice_arr = label_volume[i]
        if not slice_arr.any():
            continue
        fname = f"{tag}_z{z:05d}.png"
        Image.fromarray(slice_arr.astype(np.uint16), mode="I;16").save(out_dir / fname)
        manifest.append({
            "file": fname, "z": z, "start_x": x0, "start_y": y0,
            "width": int(slice_arr.shape[1]), "height": int(slice_arr.shape[0]),
        })
    with open(out_dir / f"{tag}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
