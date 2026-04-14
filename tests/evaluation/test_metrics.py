import math

import torch

from ignisca.evaluation.metrics import (
    auc_pr,
    precision_recall_at_threshold,
)


def test_precision_recall_on_hand_computed_case():
    # 4 pixels, one batch element. TP=1, FP=1, FN=1, TN=1.
    logits = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    p, r = precision_recall_at_threshold(logits, target, threshold=0.5)
    assert math.isclose(p, 0.5, rel_tol=1e-6)
    assert math.isclose(r, 0.5, rel_tol=1e-6)


def test_precision_recall_perfect_prediction():
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    logits = torch.where(target > 0.5, torch.full_like(target, 10.0), torch.full_like(target, -10.0))
    p, r = precision_recall_at_threshold(logits, target, threshold=0.5)
    assert math.isclose(p, 1.0, rel_tol=1e-6)
    assert math.isclose(r, 1.0, rel_tol=1e-6)


def test_precision_recall_no_positives_returns_zero():
    logits = torch.full((1, 1, 2, 2), -10.0)
    target = torch.zeros(1, 1, 2, 2)
    p, r = precision_recall_at_threshold(logits, target, threshold=0.5)
    # No predicted positives and no target positives — define both as 0.
    assert p == 0.0
    assert r == 0.0


def test_auc_pr_matches_sklearn_reference():
    from sklearn.metrics import average_precision_score

    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8) * 3
    target = (torch.rand(2, 1, 8, 8) > 0.7).float()
    probs_flat = torch.sigmoid(logits).flatten().numpy()
    target_flat = target.flatten().numpy()
    expected = average_precision_score(target_flat, probs_flat)
    actual = auc_pr(logits, target)
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)


def test_auc_pr_all_negative_target_returns_zero():
    logits = torch.randn(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    assert auc_pr(logits, target) == 0.0


def test_ece_perfect_prediction_is_zero():
    from ignisca.evaluation.metrics import expected_calibration_error

    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    # Pick logits that saturate sigmoid to 0 or 1 exactly matching target.
    logits = torch.where(
        target > 0.5,
        torch.full_like(target, 30.0),
        torch.full_like(target, -30.0),
    )
    ece = expected_calibration_error(logits, target, n_bins=10)
    assert ece < 1e-6


def test_ece_constant_half_prediction_on_balanced_target():
    from ignisca.evaluation.metrics import expected_calibration_error

    # All predictions exactly 0.5, target is 50% positive.
    logits = torch.zeros(1, 1, 4, 4)   # sigmoid(0) = 0.5
    target = torch.tensor([
        [[[1.0, 1.0, 0.0, 0.0],
          [1.0, 1.0, 0.0, 0.0],
          [1.0, 1.0, 0.0, 0.0],
          [1.0, 1.0, 0.0, 0.0]]]
    ])
    ece = expected_calibration_error(logits, target, n_bins=10)
    # Single bin [0.5, 0.6) has conf=0.5, acc=0.5 → weighted |diff|=0.
    assert ece < 1e-6


def test_ece_scripted_miscalibration_matches_hand_compute():
    from ignisca.evaluation.metrics import expected_calibration_error

    # 4 pixels all with sigmoid≈0.9 (logit≈2.197), target all 0.
    # Single bin [0.9, 1.0): conf≈0.9, acc=0, weight=1.0 → ECE≈0.9
    logits = torch.full((1, 1, 2, 2), 2.1972)
    target = torch.zeros(1, 1, 2, 2)
    ece = expected_calibration_error(logits, target, n_bins=10)
    assert abs(ece - 0.9) < 1e-3


def test_growth_rate_mae_zero_when_pred_equals_target():
    from ignisca.evaluation.metrics import growth_rate_mae

    target = (torch.rand(2, 1, 8, 8) > 0.6).float()
    input_mask = (torch.rand(2, 1, 8, 8) > 0.8).float()
    # Build saturated logits from target so sigmoid(logits) > 0.5 == target.
    logits = torch.where(
        target > 0.5,
        torch.full_like(target, 10.0),
        torch.full_like(target, -10.0),
    )
    mae = growth_rate_mae(logits, target, input_mask, pixel_area_km2=1.0, dt_hours=1.0)
    assert mae == 0.0


def test_growth_rate_mae_scripted_case():
    from ignisca.evaluation.metrics import growth_rate_mae

    # B=1, H=W=4. current area = 2 pixels, true next area = 8 pixels,
    # predicted next area = 4 pixels. pixel_area = 0.5 km², dt = 2 h.
    # true growth = (8 - 2) * 0.5 / 2 = 1.5 km²/h
    # pred growth = (4 - 2) * 0.5 / 2 = 0.5 km²/h
    # |pred - true| = 1.0
    input_mask = torch.zeros(1, 1, 4, 4)
    input_mask[0, 0, 0, 0] = 1.0
    input_mask[0, 0, 0, 1] = 1.0

    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0, :] = 1.0
    target[0, 0, 1, :] = 1.0  # 8 positives total

    logits = torch.full((1, 1, 4, 4), -10.0)
    logits[0, 0, 0, :] = 10.0  # 4 predicted positives

    mae = growth_rate_mae(logits, target, input_mask, pixel_area_km2=0.5, dt_hours=2.0)
    assert abs(mae - 1.0) < 1e-6


def test_growth_rate_mae_batch_mean():
    from ignisca.evaluation.metrics import growth_rate_mae

    # Two samples with known abs errors of 2.0 and 0.0 → mean 1.0.
    input_mask = torch.zeros(2, 1, 2, 2)
    target = torch.zeros(2, 1, 2, 2)
    target[0, 0, 0, 0] = 1.0  # sample 0 has 1 true positive
    target[0, 0, 0, 1] = 1.0  # sample 0 has 2 true positives
    logits = torch.full((2, 1, 2, 2), -10.0)
    # sample 0: pred 0 positives, true 2 → error 2
    # sample 1: pred 0 positives, true 0 → error 0
    mae = growth_rate_mae(logits, target, input_mask, pixel_area_km2=1.0, dt_hours=1.0)
    assert abs(mae - 1.0) < 1e-6
