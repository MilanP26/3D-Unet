"""Dice + BCE combo loss (PLAN.md section 7 baseline)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, target.ndim))
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = soft_dice_loss(logits, target)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


def lsd_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain MSE regression loss for the auxiliary LSD head (lsd.py). Both
    tensors are already normalized to a comparable [-1, 1]-ish scale in
    lsd.compute_lsd_target, so a raw MSE (no extra weighting per-channel) is
    reasonable -- unlike the raw voxel-unit offsets/covariances, which would
    have swamped a Dice+BCE-scale mask loss."""
    return F.mse_loss(pred, target)


@torch.no_grad()
def dice_iou_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    preds = (torch.sigmoid(logits) > threshold).float()
    dims = tuple(range(1, target.ndim))
    intersection = (preds * target).sum(dim=dims)
    union = preds.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    return {"dice": dice.mean().item(), "iou": iou.mean().item()}
