"""Inference for the affinity+LSD model: predict dense affinities over a raw
patch, then run *seeded* mutex watershed (mwatershed, github.com/pattonw/mwatershed
-- from the same lab as the LSD paper) using real per-neuron seed points as
markers. Unlike the seeded per-instance model (infer.py), identity comes from
the seed points handed to watershed, not from anything the network itself was
conditioned on -- the network only ever predicts local same-instance affinity.
"""
from __future__ import annotations

import numpy as np
import torch
import mwatershed as mws

from .affinity_targets import DEFAULT_OFFSETS


def run_affinity_inference(
    model: torch.nn.Module, raw_patch: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray | None]:
    """raw_patch: (Z, Y, X) uint8. Returns (affinity_probs, lsd_pred_or_None),
    both (channels, Z, Y, X) float32 -- affinity_probs in [0, 1] (sigmoid,
    not yet thresholded or signed for watershed)."""
    inp = torch.from_numpy((raw_patch.astype(np.float32) / 255.0)[None, None]).to(device)
    model.eval()
    with torch.no_grad():
        affinity_logits, lsd_pred = model(inp)
        aff_probs = torch.sigmoid(affinity_logits)[0].cpu().numpy()
        lsd = lsd_pred[0].cpu().numpy() if lsd_pred is not None else None
    return aff_probs, lsd


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
