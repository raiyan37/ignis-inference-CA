from __future__ import annotations

import torch


def fire_class_iou(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> float:
    """IoU of the fire class only. Background pixels are excluded from the
    denominator: a perfect prediction on a tile that is mostly background still
    scores 1.0. A tile with no fire in either the prediction or the target is
    defined as a perfect match (IoU = 1.0) to avoid undefined behavior.
    """
    pred = torch.sigmoid(logits) > threshold
    truth = target > 0.5
    intersection = (pred & truth).float().sum().item()
    union = (pred | truth).float().sum().item()
    if union == 0.0:
        return 1.0 if intersection == 0.0 else 0.0
    return intersection / union
