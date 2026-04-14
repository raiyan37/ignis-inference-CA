from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score


PIXEL_AREA_KM2: dict[str, float] = {
    # Cross-scale cells (A2, B2) have two heads; runner picks the right area
    # per head at score time. These are the per-pixel areas used by
    # ``growth_rate_mae``.
    "fine": 0.0009,      # 30 m x 30 m
    "coarse": 0.140625,  # 375 m x 375 m
}


def precision_recall_at_threshold(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """Pixel-level fire-class precision and recall at a fixed threshold.

    Matches the convention of ``training.metrics.fire_class_iou``: the 0.5
    threshold on ``sigmoid(logits)`` is applied. When the prediction has zero
    positives AND the target has zero positives, both precision and recall are
    defined as 0.0 (rather than NaN) to keep aggregation pipelines simple.
    """
    pred = torch.sigmoid(logits) > threshold
    truth = target > 0.5
    tp = float((pred & truth).sum().item())
    fp = float((pred & ~truth).sum().item())
    fn = float((~pred & truth).sum().item())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def auc_pr(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Area under the precision-recall curve over flattened pixels.

    Threshold-free. Delegates to ``sklearn.metrics.average_precision_score``.
    Returns 0.0 when the target contains no positives (AUC-PR is undefined in
    that case; sklearn raises a warning we'd rather short-circuit).
    """
    truth_np = (target > 0.5).flatten().cpu().numpy().astype(np.int64)
    if truth_np.sum() == 0:
        return 0.0
    probs_np = torch.sigmoid(logits).flatten().cpu().numpy()
    return float(average_precision_score(truth_np, probs_np))
