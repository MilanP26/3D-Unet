"""Seeded mutex watershed (mwatershed, github.com/pattonw/mwatershed -- from the
same lab as the LSD paper), split out from affinity_infer.py into its own
torch-free module.

Why this matters (found the hard way, 2026-07-20): full_stack_export.py runs
watershed in parallel worker processes to get real wall-clock benefit from
CPU cores mwatershed itself can't use (it's single-threaded). Those workers
never touch the GPU model at all -- but Windows' multiprocessing 'spawn'
start method re-imports whatever module a submitted function lives in, and if
that module (or anything it imports) pulls in torch, each worker ends up
touching CUDA on import. With several workers plus the main process all
doing that on the same GPU at once, the result was a real stall (worker CPU
time froze entirely, observed directly), not just a slowdown. Keeping this
module free of any torch import means worker processes genuinely never
touch the GPU, avoiding that contention entirely.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import mwatershed as mws

from .affinity_targets import DEFAULT_OFFSETS


def seeded_agglomerate(
    aff_probs: np.ndarray,
    seed_points: dict[int, list[tuple[int, int, int]]],
    offsets=DEFAULT_OFFSETS,
) -> np.ndarray:
    """aff_probs: (num_offsets, Z, Y, X) in [0, 1] (as returned by
    run_affinity_inference). seed_points: {real_instance_id: [(z, y, x), ...]}
    -- one or more real seed voxels per neuron, e.g. real VAST skeleton nodes
    or, for the Training Data prototype, ground-truth-sampled interior points.

    mwatershed's convention is signed affinities (positive = same instance,
    negative = different), so probs are rescaled from [0, 1] to [-1, 1]
    first. Any nonzero seed voxel is guaranteed to keep its given id in the
    output, and everything else is grown/merged from there via mutex
    watershed -- so two different real neurons structurally cannot end up
    sharing an id, unlike two independently-thresholded seeded masks."""
    shape = aff_probs.shape[1:]
    signed = (2.0 * aff_probs - 1.0).astype(np.float64)

    seeds = np.zeros(shape, dtype=np.uint64)
    for instance_id, points in seed_points.items():
        for z, y, x in points:
            if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                seeds[z, y, x] = instance_id

    return mws.agglom(signed, offsets, seeds=seeds)


def run_watershed_worker(aff_probs, seed_points, offsets):
    """Thin wrapper so ProcessPoolExecutor workers (full_stack_export.py) can submit this
    directly. Must stay defined in this torch-free module, not in full_stack_export.py --
    Windows' 'spawn' start method re-imports whichever module a submitted function lives in,
    and full_stack_export.py itself imports affinity_infer.py (which imports torch) at the
    top level. Defining it here instead means a worker unpickling this function only ever
    imports this module, never touching torch/CUDA at all."""
    return seeded_agglomerate(aff_probs, seed_points, offsets)


def _watershed_target(aff_probs, seed_points, offsets, result_queue):
    result_queue.put(seeded_agglomerate(aff_probs, seed_points, offsets))


def seeded_agglomerate_with_timeout(
    aff_probs: np.ndarray,
    seed_points: dict[int, list[tuple[int, int, int]]],
    offsets,
    timeout_s: float,
) -> np.ndarray | None:
    """Same as seeded_agglomerate, but bounded: runs the call in its own subprocess and
    kills it if it hasn't finished within timeout_s. Returns None on timeout (caller should
    treat that tile as skipped) instead of blocking indefinitely.

    Added 2026-07-20 after finding that some real tiles' mutex watershed calls take wildly
    longer than others for reasons not yet root-caused (one seeded tile with only 24 nodes
    ran past 29 minutes, next to plenty of tiles finishing in under a minute) -- rather than
    risk an unattended overnight run stalling on one bad tile for hours, every tile gets a
    hard ceiling and the run always keeps moving."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_watershed_target, args=(aff_probs, seed_points, offsets, result_queue))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None
    if not result_queue.empty():
        return result_queue.get()
    return None  # process died without producing a result (rare, treat like a timeout)
