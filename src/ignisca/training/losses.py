from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
) / 8.0
_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
) / 8.0


def sobel_gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """|∇x| via fixed 3x3 Sobel kernels with replicate padding. Expects (B, 1, H, W)."""
    kx = _SOBEL_X.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    x_padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(x_padded, kx, padding=0)
    gy = F.conv2d(x_padded, ky, padding=0)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def rothermel_spread_rate(
    fuel_load: torch.Tensor,
    slope: torch.Tensor,
    wind_u: torch.Tensor,
    wind_v: torch.Tensor,
) -> torch.Tensor:
    """Simplified dimensionless Rothermel-style spread-rate field.

    F = fuel_factor * (1 + wind_factor + slope_factor)

    Qualitative proxy, not a physically calibrated rate. Captures the monotonicity
    that Rothermel's full model imposes: spread increases with fuel load, with
    wind magnitude, and with upslope angle. Used as a per-pixel weighting field
    inside the level-set physics residual.
    """
    wind_speed = torch.sqrt(wind_u * wind_u + wind_v * wind_v + 1e-8)
    wind_factor = 0.5 * wind_speed
    slope_factor = 0.3 * torch.clamp(slope, min=0.0)
    fuel_factor = torch.clamp(fuel_load, min=0.0, max=1.0)
    return fuel_factor * (1.0 + wind_factor + slope_factor)
