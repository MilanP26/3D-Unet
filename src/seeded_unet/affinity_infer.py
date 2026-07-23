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

# Re-exported for backward compatibility -- every existing caller does
# `from .affinity_infer import ..., seeded_agglomerate`. The implementation lives in
# watershed.py now, kept deliberately free of any torch import (see that module's
# docstring for why: parallel watershed worker processes must never touch CUDA).
from .watershed import seeded_agglomerate  # noqa: F401


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
