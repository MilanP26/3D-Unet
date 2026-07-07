"""Discovery, decoding, and caching of Training Data/<Stack>/ folders.

Each stack folder is expected to contain:
  - a sequence of raw EM slice PNGs (grayscale, one file per z slice)
  - exactly one .zip file: a webKnossos volume-annotation export, which itself
    contains a `<name>.nml` (metadata: scale, segment list) and a nested
    `data_Volume.zip` (the WKW-format instance label volume).

See PLAN.md section 0 for how this layout was reverse-engineered and verified.
"""
from __future__ import annotations

import hashlib
import shutil
import warnings
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import wkw
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_DATA_DIR = REPO_ROOT / "Training Data"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache"


@dataclass
class SegmentMeta:
    id: int
    name: str | None
    has_color: bool
    anchor_xyz: tuple[int, int, int] | None


@dataclass
class Stack:
    name: str
    stack_dir: Path
    raw: np.ndarray  # (Z, Y, X) uint8
    labels: np.ndarray  # (Z, Y, X) uint32, 0 = background
    scale_nm: tuple[float, float, float]  # (x, y, z)
    segments: dict[int, SegmentMeta]
    raw_hash: str
    scene_group: str = ""  # filled in by group_stacks_by_raw_hash

    def instance_ids(self, min_voxels: int = 1) -> list[int]:
        ids, counts = np.unique(self.labels, return_counts=True)
        keep = (ids != 0) & (counts >= min_voxels)
        return sorted(ids[keep].tolist())

    def instance_mask(self, instance_id: int) -> np.ndarray:
        return self.labels == instance_id


def find_stack_dirs(training_data_dir: Path = DEFAULT_TRAINING_DATA_DIR) -> list[Path]:
    if not training_data_dir.exists():
        raise FileNotFoundError(f"No training data directory at {training_data_dir}")
    return sorted(
        p for p in training_data_dir.iterdir()
        if p.is_dir() and list(p.glob("*.png")) and list(p.glob("*.zip"))
    )


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_raw_volume(png_paths: list[Path]) -> str:
    """Cheap-but-safe identity hash: full bytes of first, middle, last slice
    plus the count. Two stacks sharing a raw EM crop (as Juliet_Stack2 and
    Juliet_Stack3 do) will collide here; two different crops essentially never
    will."""
    h = hashlib.sha256()
    h.update(str(len(png_paths)).encode())
    for p in (png_paths[0], png_paths[len(png_paths) // 2], png_paths[-1]):
        h.update(_hash_file(p).encode())
    return h.hexdigest()


def _parse_nml(nml_path: Path) -> tuple[tuple[float, float, float], str, dict[int, SegmentMeta]]:
    tree = ET.parse(nml_path)
    root = tree.getroot()
    scale_el = root.find("./parameters/scale")
    scale_nm = (
        float(scale_el.get("x")), float(scale_el.get("y")), float(scale_el.get("z"))
    )
    volume_el = root.find("./volume")
    location = volume_el.get("location")
    segments: dict[int, SegmentMeta] = {}
    for seg_el in volume_el.findall("./segments/segment"):
        sid = int(seg_el.get("id"))
        anchor = None
        if seg_el.get("anchorPositionX") is not None:
            anchor = (
                int(seg_el.get("anchorPositionX")),
                int(seg_el.get("anchorPositionY")),
                int(seg_el.get("anchorPositionZ")),
            )
        segments[sid] = SegmentMeta(
            id=sid,
            name=seg_el.get("name"),
            has_color=seg_el.get("color.r") is not None,
            anchor_xyz=anchor,
        )
    return scale_nm, location, segments


def _decode_wkw_labels(volume_zip_path: Path, extract_dir: Path, shape_xyz: tuple[int, int, int]) -> np.ndarray:
    data_volume_dir = extract_dir / "data_Volume"
    # Always extract fresh: this is only reached when the outer npz-level cache in
    # load_stack() already decided a re-decode is needed (e.g. the annotation zip's
    # content changed), so a stale leftover extraction here must not be reused.
    if data_volume_dir.exists():
        shutil.rmtree(data_volume_dir)
    with zipfile.ZipFile(volume_zip_path) as zf:
        zf.extractall(data_volume_dir)

    mag_dirs = [p for p in data_volume_dir.iterdir() if p.is_dir() and (p / "header.wkw").exists()]
    if not mag_dirs:
        raise RuntimeError(f"No WKW mag directory (with header.wkw) found under {data_volume_dir}")
    mag_dir = sorted(mag_dirs, key=lambda p: int(p.name) if p.name.isdigit() else 999)[0]

    with wkw.Dataset.open(str(mag_dir)) as ds:
        arr = ds.read((0, 0, 0), shape_xyz)  # (channels, X, Y, Z)
    arr = arr[0]  # (X, Y, Z)
    return np.transpose(arr, (2, 1, 0))  # -> (Z, Y, X)


def _load_raw_png_stack(png_paths: list[Path]) -> np.ndarray:
    slices = [np.asarray(Image.open(p).convert("L"), dtype=np.uint8) for p in png_paths]
    shapes = {s.shape for s in slices}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent PNG slice shapes in stack: {shapes}")
    return np.stack(slices, axis=0)  # (Z, Y, X)


def load_stack(stack_dir: Path, cache_dir: Path = DEFAULT_CACHE_DIR, use_cache: bool = True) -> Stack:
    name = stack_dir.name
    cache_path = cache_dir / f"{name}.npz"

    png_paths = sorted(stack_dir.glob("*.png"))
    zip_paths = list(stack_dir.glob("*.zip"))
    if len(zip_paths) != 1:
        raise ValueError(f"Expected exactly one annotation zip in {stack_dir}, found {len(zip_paths)}")
    annotation_zip = zip_paths[0]
    # Hashing the (small) annotation zip's bytes -- not just its mtime -- catches the case
    # where a placeholder/incomplete annotation gets replaced by a real one later with the
    # same PNG count: the cache must not keep serving the old decoded mask in that case.
    annotation_zip_hash = _hash_file(annotation_zip)

    if use_cache and cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as npz:
            cached_zip_hash = str(npz["annotation_zip_hash"]) if "annotation_zip_hash" in npz else None
            if int(npz["png_count"]) == len(png_paths) and cached_zip_hash == annotation_zip_hash:
                segments = {
                    int(sid): SegmentMeta(*meta) for sid, meta in npz["segments"].item().items()
                }
                return Stack(
                    name=name,
                    stack_dir=stack_dir,
                    raw=npz["raw"],
                    labels=npz["labels"],
                    scale_nm=tuple(npz["scale_nm"].tolist()),
                    segments=segments,
                    raw_hash=str(npz["raw_hash"]),
                )

    raw = _load_raw_png_stack(png_paths)
    z, y, x = raw.shape

    extract_dir = cache_dir / "_extracted" / name
    # Always start from a clean directory: if a previous (e.g. placeholder) annotation zip
    # used a differently-named .nml than the one that replaces it, leftover files from the
    # old extraction would otherwise sit alongside the new ones.
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(annotation_zip) as zf:
        zf.extractall(extract_dir)
    nml_paths = list(extract_dir.glob("*.nml"))
    if len(nml_paths) != 1:
        raise ValueError(f"Expected exactly one .nml in {annotation_zip}, found {len(nml_paths)}")
    scale_nm, volume_zip_name, segments = _parse_nml(nml_paths[0])
    volume_zip_path = extract_dir / volume_zip_name

    labels = _decode_wkw_labels(volume_zip_path, extract_dir, shape_xyz=(x, y, z))

    raw_hash = _hash_raw_volume(png_paths)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        raw=raw,
        labels=labels,
        scale_nm=np.array(scale_nm, dtype=np.float64),
        segments=np.array(
            {sid: (m.id, m.name, m.has_color, m.anchor_xyz) for sid, m in segments.items()},
            dtype=object,
        ),
        raw_hash=raw_hash,
        png_count=len(png_paths),
        annotation_zip_hash=annotation_zip_hash,
    )

    return Stack(
        name=name,
        stack_dir=stack_dir,
        raw=raw,
        labels=labels,
        scale_nm=scale_nm,
        segments=segments,
        raw_hash=raw_hash,
    )


def load_all_stacks(
    training_data_dir: Path = DEFAULT_TRAINING_DATA_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> list[Stack]:
    """Loads every stack folder found under training_data_dir. A folder that fails to
    decode (malformed zip, corrupt PNG, wrong internal structure, etc.) is skipped with a
    loud warning rather than aborting the whole run -- new data folders get dropped in
    over time, sometimes as placeholders or mid-upload, and one bad one shouldn't block
    training on everything else."""
    stacks = []
    for stack_dir in find_stack_dirs(training_data_dir):
        try:
            stacks.append(load_stack(stack_dir, cache_dir, use_cache))
        except Exception as e:
            warnings.warn(
                f"Skipping stack {stack_dir.name!r}: failed to load ({type(e).__name__}: {e}). "
                "This stack will be excluded from training until the problem is fixed.",
                stacklevel=2,
            )
    if not stacks:
        raise RuntimeError(f"No stack in {training_data_dir} could be loaded -- see warnings above.")
    group_stacks_by_raw_hash(stacks)
    return stacks


def group_stacks_by_raw_hash(stacks: list[Stack]) -> dict[str, list[str]]:
    """Assigns `scene_group` on each Stack in place. Stacks sharing the same
    raw EM (e.g. two independent annotation passes over the same crop, as with
    Juliet_Stack2/Juliet_Stack3) are forced into the same group name so a
    train/val/test split never separates them onto opposite sides."""
    groups: dict[str, list[str]] = {}
    for s in stacks:
        groups.setdefault(s.raw_hash, []).append(s.name)
    for s in stacks:
        s.scene_group = "+".join(sorted(groups[s.raw_hash]))
    return groups
