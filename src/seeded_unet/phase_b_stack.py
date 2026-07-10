"""Windowed reader for the full hard-drive EM stack (Phase B).

Unlike `Training Data/<Stack>/`, the full stack is stored in VAST's native
tiled, multi-resolution-pyramid format: a `volume.vsvi` config (JSON-like,
but with unescaped backslashes in its path templates, so it needs a lenient
parse) plus `mip0/<section-folder>/*_tr<row>-tc<col>.png` tiles, each
`SourceTileSizeX` x `SourceTileSizeY` pixels. The full stack is far too big
to load into RAM (see PLAN.md) -- this module only ever reads the tiles that
intersect a requested (z, y, x) window.

Confirmed against `E:\\ppa_b4v5s13\\aligned_stack\\volume.vsvi` (2026-07-09):
102400 x 36864 x 1060 voxels at 2/2/30 nm, in 4096x4096 tiles, 9 rows x 25
cols, mip0 only used here (full resolution, matching Training Data's scale).
`MissingImagePolicy: black` means edge tiles that don't exist on disk (the
tissue doesn't fill the full rectangular tile grid) should read as zeros,
not an error.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

# Adjacent seeds along a skeleton trace usually fall in the same or a
# neighboring tile, so decoded tiles are cached rather than re-read from disk
# every patch. 128 tiles * 16MB (4096x4096 uint8) ~= 2GB, comfortably within
# RAM on the GPU training machine.
_TILE_CACHE_SIZE = 128


@lru_cache(maxsize=_TILE_CACHE_SIZE)
def _load_tile_cached(tile_path: str) -> np.ndarray:
    return np.asarray(Image.open(tile_path).convert("L"), dtype=np.uint8)


@dataclass
class VsviConfig:
    source_dir: Path
    tile_size_x: int
    tile_size_y: int
    min_r: int
    max_r: int
    min_c: int
    max_c: int
    min_s: int
    max_s: int
    size_x: int
    size_y: int
    size_z: int
    offset_xyz: tuple[int, int, int]
    scale_nm_xyz: tuple[float, float, float]


def load_vsvi(vsvi_path: Path) -> VsviConfig:
    """Parses a .vsvi file. It's JSON-*like*, not strict JSON -- Windows path
    templates contain single backslashes, which are invalid JSON escapes, so
    they're doubled before parsing rather than trying to hand-write a real
    grammar for what is otherwise a flat key:value config."""
    raw = vsvi_path.read_text(encoding="utf-8", errors="replace")
    d = json.loads(raw.replace("\\", "\\\\"))

    return VsviConfig(
        source_dir=vsvi_path.parent,
        tile_size_x=int(d["SourceTileSizeX"]),
        tile_size_y=int(d["SourceTileSizeY"]),
        min_r=int(d["SourceMinR"]),
        max_r=int(d["SourceMaxR"]),
        min_c=int(d["SourceMinC"]),
        max_c=int(d["SourceMaxC"]),
        min_s=int(d["SourceMinS"]),
        max_s=int(d["SourceMaxS"]),
        size_x=int(d["TargetDataSizeX"]),
        size_y=int(d["TargetDataSizeY"]),
        size_z=int(d["TargetDataSizeZ"]),
        offset_xyz=(int(d["OffsetX"]), int(d["OffsetY"]), int(d["OffsetZ"])),
        scale_nm_xyz=(
            float(d["TargetVoxelSizeXnm"]),
            float(d["TargetVoxelSizeYnm"]),
            float(d["TargetVoxelSizeZnm"]),
        ),
    )


def _find_section_dir(cfg: VsviConfig, s: int) -> Path | None:
    # cfg (a dataclass) isn't hashable, so cache on its (stable, string) source_dir
    # instead of the whole config -- this glob is re-run for every seed in the same
    # section otherwise, which is pure repeated filesystem overhead.
    return _find_section_dir_cached(str(cfg.source_dir), s)


@lru_cache(maxsize=None)
def _find_section_dir_cached(source_dir: str, s: int) -> Path | None:
    # Folder name template is "%05d_%04d_Section *" -- both numeric prefixes
    # are the same section index s (just different zero-padding widths); the
    # "Section *" suffix is a literal wildcard in VAST's own template (the
    # human-readable section number embedded there is s+1, 3-digit, but
    # that offset isn't recorded anywhere -- glob it instead of computing it).
    pattern = f"{s:05d}_{s:04d}_Section *"
    matches = list((Path(source_dir) / "mip0").glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous section dir for s={s}: {matches}")
    return matches[0]


@lru_cache(maxsize=None)
def _find_tile_path(section_dir: Path, s: int, r: int, c: int) -> Path | None:
    pattern = f"{s:04d}_Section *_tr{r}-tc{c}.png"
    matches = list(section_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous tile for s={s} r={r} c={c}: {matches}")
    return matches[0]


def read_region(
    cfg: VsviConfig,
    z_range: tuple[int, int],
    y_range: tuple[int, int],
    x_range: tuple[int, int],
) -> np.ndarray:
    """Reads a dense (Z, Y, X) uint8 volume from the full tiled stack, only
    touching the tiles that intersect the requested window. Ranges are
    half-open [start, end) in the same voxel space as TargetDataSizeX/Y/Z
    (mip0, i.e. full resolution). Missing tiles (tissue doesn't fill the
    full rectangular tile grid) are filled with zeros, per the vsvi's own
    `MissingImagePolicy: black`."""
    z0, z1 = z_range
    y0, y1 = y_range
    x0, x1 = x_range
    if not (0 <= z0 < z1 <= cfg.size_z):
        raise ValueError(f"z_range {z_range} out of bounds [0, {cfg.size_z}]")
    if not (0 <= y0 < y1 <= cfg.size_y):
        raise ValueError(f"y_range {y_range} out of bounds [0, {cfg.size_y}]")
    if not (0 <= x0 < x1 <= cfg.size_x):
        raise ValueError(f"x_range {x_range} out of bounds [0, {cfg.size_x}]")

    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.uint8)
    tx, ty = cfg.tile_size_x, cfg.tile_size_y

    r_lo = y0 // ty + cfg.min_r
    r_hi = (y1 - 1) // ty + cfg.min_r
    c_lo = x0 // tx + cfg.min_c
    c_hi = (x1 - 1) // tx + cfg.min_c

    for zi, s in enumerate(range(z0 + cfg.offset_xyz[2], z1 + cfg.offset_xyz[2])):
        section_dir = _find_section_dir(cfg, s)
        if section_dir is None:
            continue  # whole slice missing -> stays zero
        for r in range(max(r_lo, cfg.min_r), min(r_hi, cfg.max_r) + 1):
            tile_y0 = (r - cfg.min_r) * ty
            for c in range(max(c_lo, cfg.min_c), min(c_hi, cfg.max_c) + 1):
                tile_x0 = (c - cfg.min_c) * tx
                tile_path = _find_tile_path(section_dir, s, r, c)
                if tile_path is None:
                    continue  # missing edge tile -> stays zero

                # Overlap between this tile's full extent and the requested window.
                oy0, oy1 = max(y0, tile_y0), min(y1, tile_y0 + ty)
                ox0, ox1 = max(x0, tile_x0), min(x1, tile_x0 + tx)
                if oy0 >= oy1 or ox0 >= ox1:
                    continue

                tile = _load_tile_cached(str(tile_path))
                out[
                    zi,
                    oy0 - y0:oy1 - y0,
                    ox0 - x0:ox1 - x0,
                ] = tile[
                    oy0 - tile_y0:oy1 - tile_y0,
                    ox0 - tile_x0:ox1 - tile_x0,
                ]

    return out


def read_region_centered(
    cfg: VsviConfig,
    center_zyx: tuple[int, int, int],
    patch_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Like `read_region`, but takes a center point and patch shape, clipping
    to the volume bounds and zero-padding whatever falls outside -- mirrors
    `dataset._crop_with_padding`'s semantics so Phase A/B patches behave the
    same way at volume edges. Real seeds here are all well within bounds (see
    PLAN.md), but this keeps behavior defined instead of raising."""
    bounds = (cfg.size_z, cfg.size_y, cfg.size_x)
    starts = [c - p // 2 for c, p in zip(center_zyx, patch_shape_zyx)]
    ends = [s + p for s, p in zip(starts, patch_shape_zyx)]

    pad_before = [max(0, -s) for s in starts]
    pad_after = [max(0, e - b) for e, b in zip(ends, bounds)]
    clipped_starts = [max(0, s) for s in starts]
    clipped_ends = [min(b, e) for b, e in zip(bounds, ends)]

    region = read_region(
        cfg,
        (clipped_starts[0], clipped_ends[0]),
        (clipped_starts[1], clipped_ends[1]),
        (clipped_starts[2], clipped_ends[2]),
    )
    if any(pad_before) or any(pad_after):
        region = np.pad(region, list(zip(pad_before, pad_after)), mode="constant", constant_values=0)
    return region
