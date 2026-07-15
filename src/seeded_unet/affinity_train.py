"""Trains the affinity+LSD model (the non-seeded alternative to train.py's
seeded per-instance model) over every Training Data/<Stack>/ folder. Instead
of one binary mask per patch for a chosen instance, this predicts dense
per-voxel affinities (+ LSDs) describing every instance in the patch at once
-- see affinity_targets.py and PLAN.md's affinity-model section for why."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .affinity_dataset import DenseAffinityPatchDataset, stacks_from_instances
from .affinity_model import AffinityLSDUNet3D
from .affinity_targets import DEFAULT_OFFSETS
from .dataset import build_instances, split_instances
from .losses import DiceBCELoss, dice_iou_metrics, lsd_loss
from .stack_io import DEFAULT_CACHE_DIR, DEFAULT_TRAINING_DATA_DIR, load_all_stacks


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-data-dir", type=Path, default=DEFAULT_TRAINING_DATA_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output-dir", type=Path, default=Path("outputs_affinity"))
    p.add_argument("--patch-size", type=int, nargs=3, default=(32, 256, 256), metavar=("Z", "Y", "X"))
    p.add_argument("--min-instance-voxels", type=int, default=500)
    p.add_argument(
        "--samples-per-stack", type=int, default=32,
        help="random dense patches drawn per stack per epoch (no per-instance seeding here)",
    )
    p.add_argument("--val-samples-per-stack", type=int, default=8)
    p.add_argument(
        "--min-labeled-fraction", type=float, default=0.02,
        help="resample a random patch (up to a few tries) if less than this fraction is "
        "labeled foreground, so training isn't dominated by empty-background crops",
    )
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-channels", type=int, default=24)
    p.add_argument("--bce-weight", type=float, default=0.5)
    p.add_argument("--no-lsd", action="store_true", help="disable the auxiliary LSD head; on by default")
    p.add_argument("--lsd-weight", type=float, default=1.0)
    p.add_argument("--lsd-sigma-nm", type=float, default=60.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="cuda|cpu, default: auto-detect")
    p.add_argument("--no-cache", action="store_true", help="ignore/rebuild decoded stack cache")
    p.add_argument("--exclude-stacks", type=str, nargs="*", default=[])
    p.add_argument(
        "--resume-from", type=Path, default=None,
        help="continue training from a checkpoint instead of starting fresh -- see train.py's "
        "--resume-from, same semantics",
    )
    return p.parse_args(argv)


def build_dataloaders(args) -> tuple[DataLoader, DataLoader]:
    stacks = load_all_stacks(
        args.training_data_dir, args.cache_dir, use_cache=not args.no_cache,
        exclude_names=tuple(args.exclude_stacks),
    )
    print(f"Loaded {len(stacks)} stack(s)")

    # Reuses dataset.py's scene-group-aware split (leakage-safe train/val stack
    # boundaries) even though dense patches aren't per-instance -- just need the
    # unique stacks each split touches.
    instances = build_instances(stacks, min_voxels=args.min_instance_voxels)
    train_inst, val_inst, _test_inst = split_instances(instances, val_fraction=args.val_fraction, seed=args.seed)
    train_stacks = stacks_from_instances(train_inst)
    val_stacks = stacks_from_instances(val_inst)
    print(f"Split: {len(train_stacks)} train stack(s), {len(val_stacks)} val stack(s)")
    if not val_stacks:
        raise RuntimeError("Validation split is empty -- add more data or lower --val-fraction")

    train_ds = DenseAffinityPatchDataset(
        train_stacks, patch_shape_zyx=tuple(args.patch_size), samples_per_stack=args.samples_per_stack,
        min_labeled_fraction=args.min_labeled_fraction, rng_seed=args.seed,
        predict_lsd=not args.no_lsd, lsd_sigma_nm=args.lsd_sigma_nm,
    )
    val_ds = DenseAffinityPatchDataset(
        val_stacks, patch_shape_zyx=tuple(args.patch_size), samples_per_stack=args.val_samples_per_stack,
        min_labeled_fraction=args.min_labeled_fraction, rng_seed=args.seed + 1,
        predict_lsd=not args.no_lsd, lsd_sigma_nm=args.lsd_sigma_nm,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader


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

    resume_ckpt = None
    start_epoch = 1
    if args.resume_from is not None:
        resume_ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        resume_args = resume_ckpt["args"]
        args.base_channels = resume_args["base_channels"]
        args.no_lsd = resume_args.get("no_lsd", False)
        start_epoch = resume_ckpt["epoch"] + 1
        if start_epoch > args.epochs:
            raise SystemExit(
                f"Checkpoint {args.resume_from} is already at epoch {resume_ckpt['epoch']}, "
                f"but --epochs={args.epochs}. Pass a larger --epochs to continue training."
            )
        print(f"Resuming from {args.resume_from} (epoch {resume_ckpt['epoch']}) -> training epochs {start_epoch}-{args.epochs}")

    checkpoint_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    checkpoint_args["offsets"] = DEFAULT_OFFSETS

    train_loader, val_loader = build_dataloaders(args)

    predict_lsd = not args.no_lsd
    model = AffinityLSDUNet3D(
        num_offsets=len(DEFAULT_OFFSETS), base_channels=args.base_channels, predict_lsd=predict_lsd
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = DiceBCELoss(bce_weight=args.bce_weight)

    best_val_dice = -1.0
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if log_path.exists():
            with open(log_path, newline="") as f:
                prior_dices = [float(r["val_dice"]) for r in csv.DictReader(f) if r["val_dice"] != "nan"]
            if prior_dices:
                best_val_dice = max(prior_dices)
                print(f"Best val_dice so far (from existing log): {best_val_dice:.4f}")

    def step(batch):
        if predict_lsd:
            inputs, targets, lsd_targets = batch
            inputs, targets, lsd_targets = inputs.to(device), targets.to(device), lsd_targets.to(device)
        else:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

        affinity_logits, lsd_pred = model(inputs)
        affinity_loss_value = criterion(affinity_logits, targets)
        if predict_lsd:
            lsd_loss_value = lsd_loss(lsd_pred, lsd_targets)
            combined = affinity_loss_value + args.lsd_weight * lsd_loss_value
            return combined, affinity_logits, targets, affinity_loss_value.item(), lsd_loss_value.item()
        return affinity_loss_value, affinity_logits, targets, affinity_loss_value.item(), None

    append_log = resume_ckpt is not None and log_path.exists()
    with open(log_path, "a" if append_log else "w", newline="") as log_file:
        writer = csv.writer(log_file)
        if not append_log:
            writer.writerow([
                "epoch", "train_loss", "train_affinity_loss", "train_lsd_loss",
                "val_loss", "val_affinity_loss", "val_lsd_loss", "val_dice", "val_iou", "seconds",
            ])

        training_start = time.time()
        epochs_run = 0
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            model.train()
            train_losses, train_aff_losses, train_lsd_losses = [], [], []
            train_bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs} [train]", unit="batch", leave=False)
            for batch in train_bar:
                optimizer.zero_grad()
                loss, _, _, aff_loss_val, lsd_loss_val = step(batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_aff_losses.append(aff_loss_val)
                if lsd_loss_val is not None:
                    train_lsd_losses.append(lsd_loss_val)
                train_bar.set_postfix(loss=f"{np.mean(train_losses):.4f}")

            model.eval()
            val_losses, val_aff_losses, val_lsd_losses, val_dices, val_ious = [], [], [], [], []
            val_bar = tqdm(val_loader, desc=f"epoch {epoch}/{args.epochs} [val]", unit="batch", leave=False)
            with torch.no_grad():
                for batch in val_bar:
                    loss, affinity_logits, targets, aff_loss_val, lsd_loss_val = step(batch)
                    val_losses.append(loss.item())
                    val_aff_losses.append(aff_loss_val)
                    if lsd_loss_val is not None:
                        val_lsd_losses.append(lsd_loss_val)
                    m = dice_iou_metrics(affinity_logits, targets)
                    val_dices.append(m["dice"])
                    val_ious.append(m["iou"])

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            train_aff_loss = float(np.mean(train_aff_losses)) if train_aff_losses else float("nan")
            train_lsd_loss = float(np.mean(train_lsd_losses)) if train_lsd_losses else float("nan")
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            val_aff_loss = float(np.mean(val_aff_losses)) if val_aff_losses else float("nan")
            val_lsd_loss = float(np.mean(val_lsd_losses)) if val_lsd_losses else float("nan")
            val_dice = float(np.mean(val_dices)) if val_dices else float("nan")
            val_iou = float(np.mean(val_ious)) if val_ious else float("nan")
            dt = time.time() - t0
            epochs_run += 1
            avg_epoch_time = (time.time() - training_start) / epochs_run
            eta_seconds = avg_epoch_time * (args.epochs - epoch)

            print(
                f"epoch {epoch:3d}/{args.epochs} | train_loss {train_loss:.4f} | "
                f"val_loss {val_loss:.4f} | val_dice(affinity) {val_dice:.4f} | val_iou {val_iou:.4f} | "
                f"{dt:.1f}s (avg {avg_epoch_time:.1f}s/epoch, ETA {eta_seconds / 60:.1f} min)"
            )
            writer.writerow([
                epoch, train_loss, train_aff_loss, train_lsd_loss,
                val_loss, val_aff_loss, val_lsd_loss, val_dice, val_iou, round(dt, 1),
            ])
            log_file.flush()

            ckpt = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "args": checkpoint_args, "epoch": epoch,
            }
            torch.save(ckpt, ckpt_dir / "last.pt")
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(ckpt, ckpt_dir / "best.pt")

    print(f"Done. Best val dice(affinity): {best_val_dice:.4f}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
