from __future__ import annotations

import math
from typing import Dict

import torch

from ignisca.data.features import CHANNEL_NAMES

# Assert the channel ordering we rely on, same pattern as training/losses.py.
_EXPECTED = {7: "wind_u", 8: "wind_v"}
for _idx, _name in _EXPECTED.items():
    if CHANNEL_NAMES[_idx] != _name:
        raise AssertionError(
            f"ignisca.evaluation.slicing expects channel {_idx}={_name!r}, "
            f"but CHANNEL_NAMES[{_idx}]={CHANNEL_NAMES[_idx]!r}. "
            "Reorder ignisca.data.features.CHANNEL_NAMES or update slicing.py."
        )


SANTA_ANA_SPEED_MIN: float = 7.0           # m/s (~15 mph)
SANTA_ANA_DIR_RANGE: tuple[int, int] = (0, 90)   # meteorological "from" angle
EARLY_FIRE_AREA_KM2: float = 5.0


def classify_santa_ana(features: torch.Tensor) -> torch.Tensor:
    """Per-sample Santa Ana flag from HRRR wind channels 7 and 8.

    Meteorological convention: "wind direction" is the compass angle the wind
    is COMING FROM. Santa Anas are offshore NE-origin winds for SoCal, so we
    accept directions in [0, 90)°. The standard (u, v) pair describes the vector
    the wind is flowing TOWARD, so we negate it inside ``atan2`` to get the
    origin angle.
    """
    if features.ndim != 4 or features.shape[1] < 9:
        raise ValueError(
            f"classify_santa_ana expects (B, 12+, H, W), got shape {tuple(features.shape)}"
        )
    u = features[:, 7]
    v = features[:, 8]
    u_mean = u.mean(dim=(-2, -1))
    v_mean = v.mean(dim=(-2, -1))
    speed = torch.sqrt(u_mean * u_mean + v_mean * v_mean + 1e-8)
    # ``atan2(x, y)`` returns the angle of (x, y) from the +y axis clockwise.
    # Meteorology measures from North (positive y) clockwise, so we use
    # atan2(east_component, north_component). Here -u is the east component of
    # the origin vector and -v is the north component.
    dir_rad = torch.atan2(-u_mean, -v_mean)
    dir_deg = (dir_rad * 180.0 / math.pi) % 360.0
    in_offshore = (dir_deg >= SANTA_ANA_DIR_RANGE[0]) & (dir_deg < SANTA_ANA_DIR_RANGE[1])
    fast_enough = speed >= SANTA_ANA_SPEED_MIN
    return fast_enough & in_offshore


def is_early_fire(input_mask: torch.Tensor, pixel_area_km2: float) -> torch.Tensor:
    """Per-sample boolean: current fire area < EARLY_FIRE_AREA_KM2."""
    if input_mask.ndim != 4 or input_mask.shape[1] != 1:
        raise ValueError(
            f"is_early_fire expects (B, 1, H, W), got shape {tuple(input_mask.shape)}"
        )
    area = (input_mask > 0.5).float().sum(dim=(-1, -2, -3)) * pixel_area_km2
    return area < EARLY_FIRE_AREA_KM2


def slice_groups(
    features: torch.Tensor,
    input_mask: torch.Tensor,
    pixel_area_km2: float,
) -> Dict[str, torch.Tensor]:
    """Return boolean per-sample masks for each slice the runner scores.

    Four slices: santa_ana, non_santa_ana, early, mature. Each value is a
    1-D bool Tensor of length ``features.shape[0]``. Slices overlap — a sample
    can be both ``santa_ana`` and ``early``.
    """
    sa = classify_santa_ana(features)
    early = is_early_fire(input_mask, pixel_area_km2=pixel_area_km2)
    return {
        "santa_ana": sa,
        "non_santa_ana": ~sa,
        "early": early,
        "mature": ~early,
    }
