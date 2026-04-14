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
