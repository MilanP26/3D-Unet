"""Trains the seeded 3D U-Net by iterating over every Training Data/<Stack>/
folder, treating each labeled neuron instance as a training example with a
synthetically-sampled seed point (PLAN.md sections 3-4).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SeededPatchDataset, build_instances, split_instances
from .losses import DiceBCELoss, dice_iou_metrics
from .model import SeededUNet3D
from .stack_io import DEFAULT_CACHE_DIR, DEFAULT_TRAINING_DATA_DIR, load_all_stacks


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-data-dir", type=Path, default=DEFAULT_TRAINING_DATA_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--patch-size", type=int, nargs=3, default=(32, 256, 256), metavar=("Z", "Y", "X"))
    p.add_argument("--min-instance-voxels", type=int, default=500)
    p.add_argument("--seed-sigma-nm", type=float, default=150.0)
    p.add_argument("--samples-per-instance", type=int, default=8)
    p.add_argument("--val-samples-per-instance", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-channels", type=int, default=24)
    p.add_argument("--bce-weight", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="cuda|cpu, default: auto-detect")
    p.add_argument("--no-cache", action="store_true", help="ignore/rebuild decoded stack cache")
    p.add_argument(
        "--exclude-stacks", type=str, nargs="*", default=[],
        help="Training Data/<Stack> folder name(s) to skip entirely, e.g. a stack whose "
        "annotation is still a known placeholder",
    )
    return p.parse_args(argv)


def build_dataloaders(args) -> tuple[DataLoader, DataLoader, list, list]:
    stacks = load_all_stacks(
        args.training_data_dir, args.cache_dir, use_cache=not args.no_cache,
        exclude_names=tuple(args.exclude_stacks),
    )
    print(f"Loaded {len(stacks)} stack(s):")
    for s in stacks:
        n_ids = len(s.instance_ids(min_voxels=1))
        print(
            f"  {s.name}: raw{tuple(s.raw.shape)} scene_group={s.scene_group!r} "
            f"scale_nm(x,y,z)={s.scale_nm} instances(any size)={n_ids}"
        )

    instances = build_instances(stacks, min_voxels=args.min_instance_voxels)
    print(f"{len(instances)} instances with >= {args.min_instance_voxels} voxels")

    train_inst, val_inst, _test_inst = split_instances(instances, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Split: {len(train_inst)} train instances, {len(val_inst)} val instances")
    if not val_inst:
        raise RuntimeError("Validation split is empty -- add more data or lower --val-fraction")

    train_ds = SeededPatchDataset(
        train_inst,
        patch_shape_zyx=tuple(args.patch_size),
        seed_sigma_nm=args.seed_sigma_nm,
        samples_per_instance=args.samples_per_instance,
        rng_seed=args.seed,
    )
    val_ds = SeededPatchDataset(
        val_inst,
        patch_shape_zyx=tuple(args.patch_size),
        seed_sigma_nm=args.seed_sigma_nm,
        samples_per_instance=args.val_samples_per_instance,
        jitter_voxels_zyx=(0, 0, 0),
        rng_seed=args.seed + 1,
    )

    # Pays the one-time per-instance distance-transform cost up front, with a
    # progress bar, instead of silently during the first epoch (this is what
    # made the first smoke test look "stuck" with no feedback).
    train_ds.precompute_seed_distributions()
    val_ds.precompute_seed_distributions()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader, train_inst, val_inst


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    log_path = args.output_dir / "training_log.csv"

    # Path objects aren't safe globals under torch's weights_only=True default (2.6+),
    # so checkpoints store plain strings instead -- infer.py/phase_b_infer.py only ever
    # read primitive values (patch_size, base_channels, etc.) back out of this anyway.
    checkpoint_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    train_loader, val_loader, train_inst, val_inst = build_dataloaders(args)

    model = SeededUNet3D(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = DiceBCELoss(bce_weight=args.bce_weight)

    best_val_dice = -1.0
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "seconds"])

        training_start = time.time()
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            train_losses = []
            train_bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs} [train]", unit="batch", leave=False)
            for inputs, targets in train_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_bar.set_postfix(loss=f"{np.mean(train_losses):.4f}")

            model.eval()
            val_losses, val_dices, val_ious = [], [], []
            val_bar = tqdm(val_loader, desc=f"epoch {epoch}/{args.epochs} [val]", unit="batch", leave=False)
            with torch.no_grad():
                for inputs, targets in val_bar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    logits = model(inputs)
                    val_losses.append(criterion(logits, targets).item())
                    m = dice_iou_metrics(logits, targets)
                    val_dices.append(m["dice"])
                    val_ious.append(m["iou"])

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            val_dice = float(np.mean(val_dices)) if val_dices else float("nan")
            val_iou = float(np.mean(val_ious)) if val_ious else float("nan")
            dt = time.time() - t0
            avg_epoch_time = (time.time() - training_start) / epoch
            eta_seconds = avg_epoch_time * (args.epochs - epoch)

            print(
                f"epoch {epoch:3d}/{args.epochs} | train_loss {train_loss:.4f} | "
                f"val_loss {val_loss:.4f} | val_dice {val_dice:.4f} | val_iou {val_iou:.4f} | "
                f"{dt:.1f}s (avg {avg_epoch_time:.1f}s/epoch, ETA {eta_seconds / 60:.1f} min)"
            )
            writer.writerow([epoch, train_loss, val_loss, val_dice, val_iou, round(dt, 1)])
            log_file.flush()

            torch.save({"model": model.state_dict(), "args": checkpoint_args, "epoch": epoch}, ckpt_dir / "last.pt")
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save({"model": model.state_dict(), "args": checkpoint_args, "epoch": epoch}, ckpt_dir / "best.pt")

    print(f"Done. Best val dice: {best_val_dice:.4f}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
