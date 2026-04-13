import torch

from ignisca.training.metrics import fire_class_iou


def test_iou_perfect_prediction_is_one():
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 2:6, 2:6] = 1.0
    logits = torch.where(
        target > 0.5, torch.full_like(target, 10.0), torch.full_like(target, -10.0)
    )
    assert fire_class_iou(logits, target) == 1.0


def test_iou_no_overlap_is_zero():
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 0:4, 0:4] = 1.0
    logits = torch.full_like(target, -10.0)
    logits[0, 0, 4:8, 4:8] = 10.0
    assert fire_class_iou(logits, target) == 0.0


def test_iou_empty_target_and_empty_prediction_is_one():
    target = torch.zeros(1, 1, 8, 8)
    logits = torch.full_like(target, -10.0)
    assert fire_class_iou(logits, target) == 1.0


def test_iou_excludes_background_from_denominator():
    target = torch.zeros(1, 1, 100, 100)
    target[0, 0, 0, 0] = 1.0
    logits = torch.full_like(target, -10.0)
    logits[0, 0, 0, 0] = 10.0
    assert fire_class_iou(logits, target) == 1.0
