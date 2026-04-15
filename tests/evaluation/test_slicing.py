import math

import torch

from ignisca.evaluation.slicing import (
    EARLY_FIRE_AREA_KM2,
    SANTA_ANA_DIR_RANGE,
    SANTA_ANA_SPEED_MIN,
    classify_santa_ana,
    is_early_fire,
    slice_groups,
)


def _features_with_uniform_wind(u: float, v: float, batch: int = 2) -> torch.Tensor:
    x = torch.zeros(batch, 12, 32, 32)
    x[:, 7] = u
    x[:, 8] = v
    return x


def test_constants_have_expected_values():
    assert SANTA_ANA_SPEED_MIN == 7.0
    assert SANTA_ANA_DIR_RANGE == (0, 90)
    assert EARLY_FIRE_AREA_KM2 == 5.0


def test_classify_santa_ana_ne_origin_fast_wind_is_true(santa_ana_batch):
    flags = classify_santa_ana(santa_ana_batch)
    assert flags.dtype == torch.bool
    assert flags.shape == (2,)
    assert bool(flags[0]) is True
    assert bool(flags[1]) is True


def test_classify_santa_ana_sw_origin_is_false():
    # Wind flowing NE (u=+7.07, v=+7.07) → coming FROM SW → not Santa Ana.
    features = _features_with_uniform_wind(7.07, 7.07)
    flags = classify_santa_ana(features)
    assert bool(flags[0]) is False


def test_classify_santa_ana_below_speed_threshold_is_false():
    # Wind flowing SW at ~3 m/s (u=-2.12, v=-2.12) → from NE but too slow.
    features = _features_with_uniform_wind(-2.12, -2.12)
    flags = classify_santa_ana(features)
    assert bool(flags[0]) is False


def test_classify_santa_ana_ne_origin_exactly_at_speed_threshold_is_true():
    # Exactly 7 m/s from the NE (45°): u=v=-4.95.
    u = -SANTA_ANA_SPEED_MIN / math.sqrt(2)
    features = _features_with_uniform_wind(u, u)
    flags = classify_santa_ana(features)
    assert bool(flags[0]) is True


def test_is_early_fire_boundary_at_five_km2():
    # pixel_area = 1 km², so an input mask with 4 positives is early, 5 is not.
    early = torch.zeros(1, 1, 4, 4)
    early[0, 0, 0, :4] = 1.0  # 4 positives

    mature = torch.zeros(1, 1, 4, 4)
    mature[0, 0, 0, :4] = 1.0
    mature[0, 0, 1, 0] = 1.0  # 5 positives

    assert bool(is_early_fire(early, pixel_area_km2=1.0)[0]) is True
    assert bool(is_early_fire(mature, pixel_area_km2=1.0)[0]) is False


def test_slice_groups_returns_all_four_masks(santa_ana_batch):
    input_mask = torch.zeros(2, 1, 32, 32)
    input_mask[0, 0, :1, :2] = 1.0  # sample 0 is small
    input_mask[1, 0, :10, :10] = 1.0  # sample 1 is large

    groups = slice_groups(santa_ana_batch, input_mask, pixel_area_km2=1.0)
    assert set(groups.keys()) == {"santa_ana", "non_santa_ana", "early", "mature"}
    for mask in groups.values():
        assert mask.shape == (2,)
        assert mask.dtype == torch.bool
    # Both samples have Santa Ana winds; sample 0 is early, sample 1 is mature.
    assert bool(groups["santa_ana"][0]) is True
    assert bool(groups["santa_ana"][1]) is True
    assert bool(groups["non_santa_ana"][0]) is False
    assert bool(groups["early"][0]) is True
    assert bool(groups["early"][1]) is False
    assert bool(groups["mature"][1]) is True
