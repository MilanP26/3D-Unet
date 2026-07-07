"""Validation-style inference: given a stack and a seed point, run the trained
model and produce a binary mask (PLAN.md section 10). This operates on the
small Training Data/<Stack>/ crops in this repo; running against the full
hard-drive EM stack with real VAST seeds (Phase B) is a separate, later step
that needs the coordinate mapping discussed in PLAN.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .dataset import _crop_with_padding
from .model import SeededUNet3D
from .seeds import gaussian_heatmap, physical_sigma_to_voxels
from .stack_io import DEFAULT_CACHE_DIR, DEFAULT_TRAINING_DATA_DIR, load_stack


def run_inference(
    model: torch.nn.Module,
    raw: np.ndarray,
    scale_nm,
    seed_zyx: tuple[int, int, int],
    patch_shape_zyx: tuple[int, int, int],
    seed_sigma_nm: float,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    raw_patch = _crop_with_padding(raw, seed_zyx, patch_shape_zyx).astype(np.float32) / 255.0
    center_in_patch = tuple(p // 2 for p in patch_shape_zyx)
    sigma_zyx = physical_sigma_to_voxels(seed_sigma_nm, scale_nm)
    heatmap = gaussian_heatmap(patch_shape_zyx, center_in_patch, sigma_zyx)

    inp = torch.from_numpy(np.stack([raw_patch, heatmap], axis=0)[None]).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return probs > threshold


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--stack-name", type=str, required=True, help="folder name under Training Data/")
    p.add_argument("--seed-zyx", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    p.add_argument("--training-data-dir", type=Path, default=DEFAULT_TRAINING_DATA_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output", type=Path, default=Path("outputs/inference_mask.npy"))
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt["args"]
    model = SeededUNet3D(base_channels=train_args["base_channels"]).to(device)
    model.load_state_dict(ckpt["model"])

    stack = load_stack(args.training_data_dir / args.stack_name, args.cache_dir)
    mask_patch = run_inference(
        model,
        stack.raw,
        stack.scale_nm,
        tuple(args.seed_zyx),
        tuple(train_args["patch_size"]),
        train_args["seed_sigma_nm"],
        device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, mask_patch)
    print(f"Predicted patch mask shape={mask_patch.shape} foreground_voxels={mask_patch.sum()}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
