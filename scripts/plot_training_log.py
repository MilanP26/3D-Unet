#!/usr/bin/env python
"""Plots train/val loss and val dice/IoU curves from a training_log.csv
produced by scripts/train.py. Runs entirely locally -- no GPU needed, just
the CSV file (copy it over from wherever training actually ran).

Usage: py scripts/plot_training_log.py outputs/training_log.csv
"""
import argparse
import csv
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log_csv", type=Path, help="path to training_log.csv")
    p.add_argument("--output", type=Path, default=None, help="default: <log_csv's folder>/loss_curves.png")
    args = p.parse_args(argv)

    with open(args.log_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{args.log_csv} has no data rows yet")

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    val_dice = [float(r["val_dice"]) for r in rows]
    val_iou = [float(r["val_iou"]) for r in rows]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(epochs, train_loss, label="train_loss", marker="o", markersize=3)
    ax1.plot(epochs, val_loss, label="val_loss", marker="o", markersize=3)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss (Dice+BCE)")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_dice, label="val_dice", marker="o", markersize=3, color="tab:green")
    ax2.plot(epochs, val_iou, label="val_iou", marker="o", markersize=3, color="tab:orange")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("score")
    ax2.set_title("Validation Dice / IoU")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(str(args.log_csv))
    fig.tight_layout()

    out_path = args.output or (args.log_csv.parent / "loss_curves.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"{len(rows)} epochs plotted -> {out_path}")
    print(f"Best val_dice: {max(val_dice):.4f} at epoch {epochs[val_dice.index(max(val_dice))]}")


if __name__ == "__main__":
    main()
