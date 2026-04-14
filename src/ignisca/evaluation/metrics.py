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


def expected_calibration_error(
    logits: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Pixel-level Expected Calibration Error with equal-width bins on [0, 1].

    For each bin, compute confidence (mean predicted probability in the bin),
    accuracy (fraction of positives in the bin), and weight (bin count / total).
    ECE is the weighted sum of |acc - conf| over all non-empty bins. Background
    pixels are included — this matches the standard pixel-level convention.
    """
    probs = torch.sigmoid(logits).flatten()
    truth = (target > 0.5).flatten().float()
    total = probs.numel()
    if total == 0:
        return 0.0

    # Edges: 0.0, 0.1, 0.2, ..., 1.0 (n_bins + 1 points).
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Final bin is closed on the right so prob==1.0 is included.
        if i == n_bins - 1:
            in_bin = (probs >= lo) & (probs <= hi)
        else:
            in_bin = (probs >= lo) & (probs < hi)
        n_in = int(in_bin.sum().item())
        if n_in == 0:
            continue
        bin_conf = float(probs[in_bin].mean().item())
        bin_acc = float(truth[in_bin].mean().item())
        weight = n_in / total
        ece += weight * abs(bin_acc - bin_conf)
    return ece


def growth_rate_mae(
    logits: torch.Tensor,
    target: torch.Tensor,
    input_mask: torch.Tensor,
    pixel_area_km2: float,
    dt_hours: float = 1.0,
) -> float:
    """Mean absolute error of the per-sample fire growth rate.

    Growth rate is defined as ``(next_area - current_area) / dt_hours`` in
    km²/h, where ``next_area`` is computed from the predicted next-step fire
    mask and ``current_area`` from the input fire mask. Returns a mean over the
    batch as a Python float.
    """
    if dt_hours <= 0:
        raise ValueError(f"dt_hours must be positive, got {dt_hours}")
    pred_bin = (torch.sigmoid(logits) > 0.5).float()
    pred_area_next = pred_bin.sum(dim=(-1, -2, -3)) * pixel_area_km2
    true_area_next = (target > 0.5).float().sum(dim=(-1, -2, -3)) * pixel_area_km2
    curr_area = (input_mask > 0.5).float().sum(dim=(-1, -2, -3)) * pixel_area_km2
    pred_growth = (pred_area_next - curr_area) / dt_hours
    true_growth = (true_area_next - curr_area) / dt_hours
    return float((pred_growth - true_growth).abs().mean().item())
