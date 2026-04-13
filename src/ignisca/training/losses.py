from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ignisca.data.features import CHANNEL_NAMES

_EXPECTED_PHYSICS_CHANNELS = {
    0: "fire_mask",
    1: "fuel_model",
    4: "slope",
    7: "wind_u",
    8: "wind_v",
}
for _idx, _name in _EXPECTED_PHYSICS_CHANNELS.items():
    if CHANNEL_NAMES[_idx] != _name:
        raise AssertionError(
            f"IgnisLoss physics branch expects channel {_idx}={_name!r}, "
            f"but CHANNEL_NAMES[{_idx}]={CHANNEL_NAMES[_idx]!r}. "
            "Reorder ignisca.data.features.CHANNEL_NAMES or update the indices in losses.py."
        )

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


def level_set_residual(
    pred_sigmoid: torch.Tensor,
    input_mask: torch.Tensor,
    spread_rate: torch.Tensor,
) -> torch.Tensor:
    """Soft level-set Hamilton-Jacobi residual: mean((∂φ/∂t − F·|∇φ|)²).

    Treats the sigmoid prediction as a soft indicator φ where φ ≈ 1 inside the
    fire and φ ≈ 0 outside. Under this "indicator" sign convention, points that
    transition from outside to inside have ∂φ/∂t > 0, and the Hamilton-Jacobi
    fire-spread relation simplifies to ∂φ/∂t = F·|∇φ| at boundary points.
    The residual penalizes deviations from that balance.
    """
    dphi_dt = pred_sigmoid - input_mask
    grad_mag = sobel_gradient_magnitude(pred_sigmoid)
    residual = dphi_dt - spread_rate * grad_mag
    return residual.pow(2).mean()


class IgnisLoss(nn.Module):
    """Weighted sum of BCE and the level-set physics residual.

    - λ_phys == 0: behaves exactly like ``BCEWithLogitsLoss``.
    - λ_phys > 0: requires the 12-channel feature stack so the spread-rate field
      can be computed from channel 0 (current fire mask), channel 1 (fuel),
      channel 4 (slope), channel 7 (wind u), and channel 8 (wind v). The channel
      indices are fixed by ``ignisca.data.features.CHANNEL_NAMES``.
    """

    def __init__(self, lambda_data: float = 1.0, lambda_phys: float = 0.0) -> None:
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        data = self.bce(logits, target.float())
        if self.lambda_phys == 0.0:
            return self.lambda_data * data
        if features is None:
            raise ValueError("features must be provided when lambda_phys > 0")
        pred_sigmoid = torch.sigmoid(logits)
        input_mask = features[:, 0:1]
        fuel = features[:, 1:2]
        slope = features[:, 4:5]
        wind_u = features[:, 7:8]
        wind_v = features[:, 8:9]
        F_field = rothermel_spread_rate(fuel, slope, wind_u, wind_v)
        phys = level_set_residual(pred_sigmoid, input_mask, F_field)
        return self.lambda_data * data + self.lambda_phys * phys
