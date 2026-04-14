# IgnisCA Model + Training Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a trainable ResU-Net with BCE + level-set physics loss and an ablation runner that materializes the 18-run 4-cell × 3-seed grid defined in the design spec §4.

**Architecture:** A single 12-channel ResU-Net backbone is instantiated once per training run. Cross-scale operates as two independent training runs of the same backbone (one per cache resolution) plus an inference-time router. The physics-informed variant adds a soft level-set Hamilton-Jacobi residual term to the BCE loss, using a simplified Rothermel-style spread-rate field derived from the fuel/slope/wind channels of the feature stack. Training reads only from the Plan 1 `.npz` shard cache — no raw archive access.

**Tech Stack:** PyTorch 2.x (already in pyproject), plain `torch.utils.data.DataLoader`, plain Python for config (no Lightning, no YAML, no W&B in this plan — deferred to Plan 3). JSONL per-run metrics logs. All deps already present in `pyproject.toml` from Plan 1.

**Spec reference:** `/Users/h.raiyan/ignis-inference-CA/docs/superpowers/specs/2026-04-11-ignisca-design.md` §3 (model), §4 (training & ablation).

**Plan 1 dependencies:**
- `ignisca.data.dataset.IgnisDataset` — reads `cache_root/{split}/*.npz` shards and yields `(inputs, target)` pairs with `inputs` shape `(12, H, W)` and `target` shape `(H, W)`.
- 12-channel ordering from `ignisca.data.features.CHANNEL_NAMES` — this plan hardcodes the same ordering (fire_mask=0, fuel=1, slope=4, wind_u=7, wind_v=8) for the physics loss's spread-rate field.

**Out of scope for this plan (deferred to Plan 3):**
- Full evaluation metric suite (precision/recall, AUC-PR, ECE, growth-rate error) — Plan 2 only ships fire-class IoU for checkpoint selection.
- MC Dropout inference rendering.
- Per-slice analysis and failure-mode visualization.
- Hyperparameter sweeps beyond the four primary cells.
- W&B integration, README ablation table.
- NDWS pretrain shard ingestion — Plan 2 trains only on caches produced by `scripts/preprocess.py`.

---

## File Structure

**New package `src/ignisca/models/`:**
- `__init__.py` — empty
- `resunet.py` — `ResidualBlock`, `ResUNet` (12-channel input, 1-channel logit output, 4-stage encoder/decoder, 2D spatial dropout)
- `router.py` — `select_head(current_fire_area_km2, threshold_km2)` pure function

**New package `src/ignisca/training/`:**
- `__init__.py` — empty
- `losses.py` — `sobel_gradient_magnitude`, `rothermel_spread_rate`, `level_set_residual`, `IgnisLoss`
- `config.py` — `TrainConfig` dataclass
- `metrics.py` — `fire_class_iou` (background excluded from denominator)
- `loop.py` — `set_seed`, `train_one_run`
- `ablation.py` — `ABLATION_CELLS`, `SEEDS`, `run_ablation`

**New scripts:**
- `scripts/train.py` — single-run CLI
- `scripts/run_ablation.py` — 18-run ablation CLI

**New tests:**
- `tests/models/test_resunet.py`
- `tests/models/test_router.py`
- `tests/training/test_losses.py`
- `tests/training/test_metrics.py`
- `tests/training/test_loop.py`
- `tests/training/test_ablation.py`

Test directories `tests/models/` and `tests/training/` must be created. No `__init__.py` is needed under `tests/` — Plan 1's `tests/data/sources/` works without one.

---

## Task 1: ResU-Net backbone

**Files:**
- Create: `src/ignisca/models/__init__.py`
- Create: `src/ignisca/models/resunet.py`
- Create: `tests/models/test_resunet.py`

- [ ] **Step 1: Create the empty package marker**

Write `src/ignisca/models/__init__.py` with zero bytes (truly empty file). Do not add any imports or exports.

- [ ] **Step 2: Write the failing forward-shape test**

Create `tests/models/test_resunet.py`:

```python
import torch

from ignisca.models.resunet import ResUNet


def test_resunet_forward_shape_matches_input():
    model = ResUNet(in_channels=12, base=16, dropout=0.2)
    x = torch.randn(2, 12, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_resunet_backward_produces_gradients():
    model = ResUNet(in_channels=12, base=16, dropout=0.2)
    x = torch.randn(2, 12, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"missing grad for {name}"


def test_resunet_dropout_active_when_train_mode_set():
    """MC Dropout requires dropout to remain stochastic when the module is in train mode."""
    torch.manual_seed(0)
    model = ResUNet(in_channels=12, base=16, dropout=0.5).train()
    x = torch.randn(1, 12, 32, 32)
    out_a = model(x)
    out_b = model(x)
    assert not torch.allclose(out_a, out_b)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/models/test_resunet.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.models.resunet'`.

- [ ] **Step 4: Implement `ResidualBlock` and `ResUNet`**

Write `src/ignisca/models/resunet.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two 3x3 conv + GroupNorm + ReLU with an additive skip and 2D spatial dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout2d(p=dropout)
        self.skip: nn.Module = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = torch.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = torch.relu(out + residual)
        return self.dropout(out)


class ResUNet(nn.Module):
    """ResU-Net for next-timestep fire perimeter prediction.

    Input: (B, in_channels, H, W) — IgnisCA feature stack, 12 channels.
    Output: (B, 1, H, W) raw logits (sigmoid is applied in the loss / at inference).
    """

    def __init__(self, in_channels: int = 12, base: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.enc1 = ResidualBlock(in_channels, c1, dropout=dropout)
        self.enc2 = ResidualBlock(c1, c2, dropout=dropout)
        self.enc3 = ResidualBlock(c2, c3, dropout=dropout)
        self.enc4 = ResidualBlock(c3, c4, dropout=dropout)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ResidualBlock(c4, c4, dropout=dropout),
            ResidualBlock(c4, c4, dropout=dropout),
        )

        self.up4 = nn.ConvTranspose2d(c4, c4, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(c4 + c4, c4, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(c3 + c3, c3, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(c2 + c2, c2, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(c1 + c1, c1, dropout=dropout)

        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))
        d4 = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return self.head(d1)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/models/test_resunet.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ignisca/models/__init__.py src/ignisca/models/resunet.py tests/models/test_resunet.py
git commit -m "feat(models): ResU-Net backbone with residual blocks and spatial dropout"
```

---

## Task 2: Cross-scale router

**Files:**
- Create: `src/ignisca/models/router.py`
- Create: `tests/models/test_router.py`

- [ ] **Step 1: Write the failing test**

Create `tests/models/test_router.py`:

```python
from ignisca.models.router import select_head


def test_router_picks_fine_below_threshold():
    assert select_head(current_fire_area_km2=1.0, threshold_km2=5.0) == "fine"


def test_router_picks_coarse_at_or_above_threshold():
    assert select_head(current_fire_area_km2=5.0, threshold_km2=5.0) == "coarse"
    assert select_head(current_fire_area_km2=25.0, threshold_km2=5.0) == "coarse"


def test_router_default_threshold_is_five_sqkm():
    assert select_head(current_fire_area_km2=4.9) == "fine"
    assert select_head(current_fire_area_km2=5.1) == "coarse"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/models/test_router.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.models.router'`.

- [ ] **Step 3: Implement the router**

Write `src/ignisca/models/router.py`:

```python
from __future__ import annotations


def select_head(current_fire_area_km2: float, threshold_km2: float = 5.0) -> str:
    """Cross-scale inference router.

    Returns ``"fine"`` for small fires (use the 30m checkpoint) or ``"coarse"``
    once the predicted burn area reaches the handoff threshold (use the 375m
    checkpoint). The threshold defaults to 5 km² per design spec §3.2 and is
    swept in the secondary ablation grid.
    """
    if current_fire_area_km2 < threshold_km2:
        return "fine"
    return "coarse"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/models/test_router.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/models/router.py tests/models/test_router.py
git commit -m "feat(models): cross-scale inference router with 5km² default handoff"
```

---

## Task 3: Sobel gradient and Rothermel spread-rate field

**Files:**
- Create: `src/ignisca/training/__init__.py`
- Create: `src/ignisca/training/losses.py`
- Create: `tests/training/test_losses.py`

- [ ] **Step 1: Create the empty package marker**

Write `src/ignisca/training/__init__.py` with zero bytes.

- [ ] **Step 2: Write the failing test for Sobel and Rothermel**

Create `tests/training/test_losses.py`:

```python
import torch

from ignisca.training.losses import (
    rothermel_spread_rate,
    sobel_gradient_magnitude,
)


def test_sobel_of_constant_is_zero():
    x = torch.full((1, 1, 16, 16), 0.7)
    g = sobel_gradient_magnitude(x)
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-5)


def test_sobel_of_step_has_nonzero_edge():
    x = torch.zeros(1, 1, 16, 16)
    x[:, :, :, 8:] = 1.0
    g = sobel_gradient_magnitude(x)
    assert torch.allclose(g[0, 0, 4:12, 0:6], torch.zeros(8, 6), atol=1e-5)
    assert torch.allclose(g[0, 0, 4:12, 10:16], torch.zeros(8, 6), atol=1e-5)
    assert g[0, 0, 8, 7:10].max() > 0.0


def test_rothermel_monotonic_in_wind_speed():
    fuel = torch.full((1, 1, 4, 4), 0.5)
    slope = torch.zeros(1, 1, 4, 4)
    u_weak = torch.full((1, 1, 4, 4), 1.0)
    v = torch.zeros(1, 1, 4, 4)
    u_strong = torch.full((1, 1, 4, 4), 5.0)
    f_weak = rothermel_spread_rate(fuel, slope, u_weak, v)
    f_strong = rothermel_spread_rate(fuel, slope, u_strong, v)
    assert (f_strong > f_weak).all()


def test_rothermel_monotonic_in_slope():
    fuel = torch.full((1, 1, 4, 4), 0.5)
    u = torch.zeros(1, 1, 4, 4)
    v = torch.zeros(1, 1, 4, 4)
    slope_flat = torch.zeros(1, 1, 4, 4)
    slope_steep = torch.full((1, 1, 4, 4), 1.0)
    f_flat = rothermel_spread_rate(fuel, slope_flat, u, v)
    f_steep = rothermel_spread_rate(fuel, slope_steep, u, v)
    assert (f_steep > f_flat).all()


def test_rothermel_zero_fuel_gives_zero_spread():
    fuel = torch.zeros(1, 1, 4, 4)
    slope = torch.full((1, 1, 4, 4), 0.5)
    u = torch.full((1, 1, 4, 4), 3.0)
    v = torch.full((1, 1, 4, 4), 2.0)
    f = rothermel_spread_rate(fuel, slope, u, v)
    assert torch.allclose(f, torch.zeros_like(f))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/training/test_losses.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.training.losses'`.

- [ ] **Step 4: Implement Sobel and Rothermel**

Write `src/ignisca/training/losses.py`:

```python
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
    """|∇x| via fixed 3x3 Sobel kernels. Expects (B, 1, H, W)."""
    kx = _SOBEL_X.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/training/test_losses.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ignisca/training/__init__.py src/ignisca/training/losses.py tests/training/test_losses.py
git commit -m "feat(training): Sobel gradient operator and simplified Rothermel spread-rate field"
```

---

## Task 4: Level-set residual and combined `IgnisLoss`

**Files:**
- Modify: `src/ignisca/training/losses.py` (append)
- Modify: `tests/training/test_losses.py` (append)

- [ ] **Step 1: Write the failing test for the level-set residual and `IgnisLoss`**

Append to `tests/training/test_losses.py`:

```python
from ignisca.training.losses import IgnisLoss, level_set_residual


def test_level_set_residual_is_nonnegative():
    pred = torch.rand(2, 1, 16, 16)
    mask = torch.randint(0, 2, (2, 1, 16, 16)).float()
    spread = torch.rand(2, 1, 16, 16)
    r = level_set_residual(pred, mask, spread)
    assert r.item() >= 0.0


def test_level_set_residual_zero_when_pred_matches_mask_and_F_zero():
    mask = torch.zeros(1, 1, 8, 8)
    mask[0, 0, 2:6, 2:6] = 1.0
    pred = mask.clone()
    spread = torch.zeros_like(mask)
    r = level_set_residual(pred, mask, spread)
    assert torch.allclose(r, torch.zeros(()), atol=1e-6)


def test_ignis_loss_data_only_matches_bce():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 16, 16)).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss_fn = IgnisLoss(lambda_data=1.0, lambda_phys=0.0)
    ignis = loss_fn(logits, target, features=None)
    assert torch.allclose(bce, ignis)


def test_ignis_loss_physics_requires_features():
    logits = torch.randn(1, 1, 16, 16)
    target = torch.randint(0, 2, (1, 1, 16, 16)).float()
    loss_fn = IgnisLoss(lambda_data=1.0, lambda_phys=0.1)
    try:
        loss_fn(logits, target, features=None)
    except ValueError as exc:
        assert "features" in str(exc)
        return
    raise AssertionError("expected ValueError when features is None and lambda_phys > 0")


def test_ignis_loss_physics_branch_is_differentiable():
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 16, 16)).float()
    features = torch.rand(2, 12, 16, 16)
    loss_fn = IgnisLoss(lambda_data=1.0, lambda_phys=0.1)
    loss = loss_fn(logits, target, features=features)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/training/test_losses.py -v`
Expected: 5 new failures with `ImportError: cannot import name 'IgnisLoss'` or `'level_set_residual'`.

- [ ] **Step 3: Append the residual and the combined loss to `losses.py`**

Append to `src/ignisca/training/losses.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/training/test_losses.py -v`
Expected: 10 passed (5 from Task 3 + 5 from Task 4).

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/training/losses.py tests/training/test_losses.py
git commit -m "feat(training): level-set residual and combined IgnisLoss (BCE + physics)"
```

---

## Task 5: Training config and fire-class IoU

**Files:**
- Create: `src/ignisca/training/config.py`
- Create: `src/ignisca/training/metrics.py`
- Create: `tests/training/test_metrics.py`

- [ ] **Step 1: Write the failing metric test**

Create `tests/training/test_metrics.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/training/test_metrics.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.training.metrics'`.

- [ ] **Step 3: Implement the metric and the config dataclass**

Write `src/ignisca/training/metrics.py`:

```python
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
```

Write `src/ignisca/training/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    cache_root: Path
    out_dir: Path = field(default_factory=lambda: Path("runs"))
    run_name: str = "default"
    train_split: str = "train"
    val_split: str = "val"
    seed: int = 0
    epochs: int = 30
    batch_size: int = 4
    lr: float = 1e-5
    lambda_data: float = 1.0
    lambda_phys: float = 0.0
    dropout: float = 0.2
    base_channels: int = 64
    num_workers: int = 0
    device: str = "cuda"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/training/test_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/training/config.py src/ignisca/training/metrics.py tests/training/test_metrics.py
git commit -m "feat(training): TrainConfig dataclass and fire-class IoU metric"
```

---

## Task 6: End-to-end training loop with checkpoint selection

**Files:**
- Create: `src/ignisca/training/loop.py`
- Create: `tests/training/test_loop.py`

- [ ] **Step 1: Write the failing loop test**

Create `tests/training/test_loop.py`:

```python
import json
from pathlib import Path

import numpy as np
import torch

from ignisca.data.cache import CacheShard, save_shard
from ignisca.training.config import TrainConfig
from ignisca.training.loop import set_seed, train_one_run


def _write_tiny_cache(root: Path) -> None:
    """Two training shards and one validation shard at 32x32, 12 channels."""
    rng = np.random.default_rng(0)
    for split, n in [("train", 2), ("val", 1)]:
        (root / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            inputs = rng.standard_normal((12, 32, 32)).astype(np.float32)
            target = np.zeros((32, 32), dtype=np.uint8)
            target[10:20, 10:20] = 1
            save_shard(
                root / split / f"{i:05d}.npz",
                CacheShard(inputs=inputs, target=target, metadata={"idx": i}),
            )


def test_set_seed_makes_torch_deterministic():
    set_seed(123)
    a = torch.randn(4)
    set_seed(123)
    b = torch.randn(4)
    assert torch.allclose(a, b)


def test_train_one_run_writes_checkpoint_and_log(tmp_path: Path):
    _write_tiny_cache(tmp_path)
    cfg = TrainConfig(
        cache_root=tmp_path,
        out_dir=tmp_path / "runs",
        run_name="smoke",
        epochs=2,
        batch_size=1,
        lr=1e-3,
        base_channels=8,
        device="cpu",
    )
    result = train_one_run(cfg)

    ckpt = tmp_path / "runs" / "smoke" / "best.pt"
    log = tmp_path / "runs" / "smoke" / "metrics.jsonl"
    assert ckpt.exists(), "best checkpoint was not written"
    assert log.exists(), "metrics log was not written"

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        entry = json.loads(line)
        assert {"epoch", "train_loss", "val_iou", "run_name"} <= entry.keys()

    assert "best_val_iou" in result
    assert "best_epoch" in result


def test_train_one_run_with_physics_loss_runs(tmp_path: Path):
    _write_tiny_cache(tmp_path)
    cfg = TrainConfig(
        cache_root=tmp_path,
        out_dir=tmp_path / "runs",
        run_name="smoke_phys",
        epochs=1,
        batch_size=1,
        lr=1e-3,
        base_channels=8,
        lambda_phys=0.1,
        device="cpu",
    )
    result = train_one_run(cfg)
    assert (tmp_path / "runs" / "smoke_phys" / "best.pt").exists()
    assert result["best_val_iou"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/training/test_loop.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.training.loop'`.

- [ ] **Step 3: Implement the training loop**

Write `src/ignisca/training/loop.py`:

```python
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ignisca.data.dataset import IgnisDataset
from ignisca.models.resunet import ResUNet
from ignisca.training.config import TrainConfig
from ignisca.training.losses import IgnisLoss
from ignisca.training.metrics import fire_class_iou


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def train_one_run(cfg: TrainConfig) -> Dict[str, float]:
    """Train a single model end-to-end.

    Writes a best-IoU checkpoint to ``{out_dir}/{run_name}/best.pt`` and a
    per-epoch JSONL log to ``{out_dir}/{run_name}/metrics.jsonl``. Returns a
    dict with ``best_val_iou`` and ``best_epoch``.
    """
    set_seed(cfg.seed)
    device = _resolve_device(cfg.device)

    train_ds = IgnisDataset(cfg.cache_root, split=cfg.train_split)
    val_ds = IgnisDataset(cfg.cache_root, split=cfg.val_split)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = ResUNet(in_channels=12, base=cfg.base_channels, dropout=cfg.dropout).to(device)
    loss_fn = IgnisLoss(lambda_data=cfg.lambda_data, lambda_phys=cfg.lambda_phys).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    run_dir = Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "metrics.jsonl"
    ckpt_path = run_dir / "best.pt"
    log_path.write_text("")

    best_iou = -1.0
    best_epoch = -1

    for epoch in range(cfg.epochs):
        model.train(True)
        train_loss_sum = 0.0
        n_train_batches = 0
        for x, y in train_loader:
            x = x.to(device).float()
            y = y.to(device).float().unsqueeze(1)
            logits = model(x)
            loss = loss_fn(logits, y, features=x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            n_train_batches += 1

        model.train(False)
        val_iou_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device).float()
                y = y.to(device).float().unsqueeze(1)
                logits = model(x)
                val_iou_sum += fire_class_iou(logits, y)
                n_val_batches += 1

        train_loss = train_loss_sum / max(n_train_batches, 1)
        val_iou = val_iou_sum / max(n_val_batches, 1)

        with log_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_iou": val_iou,
                        "run_name": cfg.run_name,
                    }
                )
                + "\n"
            )

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in cfg.__dict__.items()
                    },
                },
                ckpt_path,
            )

    return {"best_val_iou": float(best_iou), "best_epoch": float(best_epoch)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/training/test_loop.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full test suite**

Run: `source .venv/bin/activate && pytest -v`
Expected: previous 33 tests + new training/model tests all green.

- [ ] **Step 6: Commit**

```bash
git add src/ignisca/training/loop.py tests/training/test_loop.py
git commit -m "feat(training): training loop with best-IoU checkpoint selection and JSONL log"
```

---

## Task 7: Ablation runner

**Files:**
- Create: `src/ignisca/training/ablation.py`
- Create: `tests/training/test_ablation.py`

- [ ] **Step 1: Write the failing ablation test**

Create `tests/training/test_ablation.py`:

```python
from pathlib import Path

import numpy as np

from ignisca.data.cache import CacheShard, save_shard
from ignisca.training.ablation import ABLATION_CELLS, SEEDS, run_ablation
from ignisca.training.config import TrainConfig


def _write_tiny_cache(root: Path) -> None:
    rng = np.random.default_rng(0)
    for split, n in [("train", 2), ("val", 1)]:
        (root / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            inputs = rng.standard_normal((12, 32, 32)).astype(np.float32)
            target = np.zeros((32, 32), dtype=np.uint8)
            target[10:20, 10:20] = 1
            save_shard(
                root / split / f"{i:05d}.npz",
                CacheShard(inputs=inputs, target=target, metadata={"idx": i}),
            )


def test_ablation_cells_cover_the_full_grid():
    names = {cell["name"] for cell in ABLATION_CELLS}
    assert names == {"A1_single_data", "A2_cross_data", "B1_single_phys", "B2_cross_phys"}


def test_ablation_has_three_seeds():
    assert len(SEEDS) == 3
    assert set(SEEDS) == {0, 1, 2}


def test_run_ablation_produces_one_result_per_cell_and_head_and_seed(tmp_path: Path):
    """Single-scale cells train one head; cross-scale cells train two (fine + coarse).
    Expected total = (2 single × 1 head + 2 cross × 2 heads) × 3 seeds = 18 runs.
    """
    _write_tiny_cache(tmp_path)
    base = TrainConfig(
        cache_root=tmp_path,
        out_dir=tmp_path / "runs",
        epochs=1,
        batch_size=1,
        lr=1e-3,
        base_channels=8,
        device="cpu",
    )
    results = run_ablation(base, cache_fine=tmp_path, cache_coarse=tmp_path)
    assert len(results) == 18

    for r in results:
        assert {"run_name", "cell", "head", "seed", "best_val_iou"} <= r.keys()

    for r in results:
        ckpt = tmp_path / "runs" / r["run_name"] / "best.pt"
        assert ckpt.exists(), f"missing checkpoint for {r['run_name']}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/training/test_ablation.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.training.ablation'`.

- [ ] **Step 3: Implement the ablation runner**

Write `src/ignisca/training/ablation.py`:

```python
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List

from ignisca.training.config import TrainConfig
from ignisca.training.loop import train_one_run


ABLATION_CELLS: List[Dict[str, object]] = [
    {"name": "A1_single_data", "scale": "single", "lambda_phys": 0.0},
    {"name": "A2_cross_data", "scale": "cross", "lambda_phys": 0.0},
    {"name": "B1_single_phys", "scale": "single", "lambda_phys": 0.1},
    {"name": "B2_cross_phys", "scale": "cross", "lambda_phys": 0.1},
]

SEEDS: tuple[int, ...] = (0, 1, 2)


def _heads_for(scale: str) -> List[str]:
    if scale == "single":
        return ["coarse"]
    if scale == "cross":
        return ["fine", "coarse"]
    raise ValueError(f"unknown scale: {scale}")


def run_ablation(
    base_cfg: TrainConfig,
    cache_fine: Path,
    cache_coarse: Path,
) -> List[Dict[str, object]]:
    """Run the full 4-cell × 3-seed ablation.

    Single-scale cells train only the coarse (375m) head. Cross-scale cells
    train both a fine (30m) and a coarse (375m) head as two independent runs,
    to be selected by the inference-time router in
    ``ignisca.models.router.select_head``.

    Per design spec §4.3 this produces 18 total runs: 2 single-scale × 1 head ×
    3 seeds + 2 cross-scale × 2 heads × 3 seeds = 18.
    """
    results: List[Dict[str, object]] = []
    for cell in ABLATION_CELLS:
        for seed in SEEDS:
            for head in _heads_for(str(cell["scale"])):
                run_name = f"{cell['name']}_{head}_seed{seed}"
                cache_root = cache_fine if head == "fine" else cache_coarse
                cfg = replace(
                    base_cfg,
                    cache_root=cache_root,
                    run_name=run_name,
                    seed=seed,
                    lambda_phys=float(cell["lambda_phys"]),
                )
                metrics = train_one_run(cfg)
                results.append(
                    {
                        "run_name": run_name,
                        "cell": cell["name"],
                        "head": head,
                        "seed": seed,
                        "best_val_iou": metrics["best_val_iou"],
                        "best_epoch": metrics["best_epoch"],
                    }
                )
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/training/test_ablation.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/training/ablation.py tests/training/test_ablation.py
git commit -m "feat(training): 18-run ablation grid (4 cells x 3 seeds x head count)"
```

---

## Task 8: CLI entrypoints

**Files:**
- Create: `scripts/train.py`
- Create: `scripts/run_ablation.py`

- [ ] **Step 1: Implement the single-run CLI**

Write `scripts/train.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from ignisca.training.config import TrainConfig
from ignisca.training.loop import train_one_run


def main() -> None:
    parser = argparse.ArgumentParser(description="IgnisCA single training run")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda-data", type=float, default=1.0)
    parser.add_argument("--lambda-phys", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = TrainConfig(
        cache_root=args.cache_root,
        out_dir=args.out_dir,
        run_name=args.run_name,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_data=args.lambda_data,
        lambda_phys=args.lambda_phys,
        dropout=args.dropout,
        base_channels=args.base_channels,
        num_workers=args.num_workers,
        device=args.device,
    )
    result = train_one_run(cfg)
    print(
        f"Run {cfg.run_name} finished: best_val_iou={result['best_val_iou']:.4f} "
        f"at epoch {int(result['best_epoch'])}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the single-run CLI help**

Run: `source .venv/bin/activate && python scripts/train.py --help`
Expected: argparse help text, exit 0.

- [ ] **Step 3: Implement the ablation CLI**

Write `scripts/run_ablation.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ignisca.training.ablation import run_ablation
from ignisca.training.config import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="IgnisCA ablation runner")
    parser.add_argument("--cache-fine", type=Path, required=True)
    parser.add_argument("--cache-coarse", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Optional path to dump the per-run results list as JSON",
    )
    args = parser.parse_args()

    base = TrainConfig(
        cache_root=args.cache_coarse,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        base_channels=args.base_channels,
        num_workers=args.num_workers,
        device=args.device,
    )
    results = run_ablation(base, cache_fine=args.cache_fine, cache_coarse=args.cache_coarse)

    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(results, indent=2))

    print(f"Ablation finished: {len(results)} runs")
    for r in results:
        print(
            f"  {r['run_name']}: iou={r['best_val_iou']:.4f} "
            f"(cell={r['cell']}, head={r['head']}, seed={r['seed']})"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke-run the ablation CLI help**

Run: `source .venv/bin/activate && python scripts/run_ablation.py --help`
Expected: argparse help text, exit 0.

- [ ] **Step 5: Run the full test suite one final time**

Run: `source .venv/bin/activate && pytest -v`
Expected: all tests green (Plan 1 33 tests + Plan 2 new tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/train.py scripts/run_ablation.py
git commit -m "feat(training): CLI entrypoints for single runs and full ablation"
```

---

## Self-Review

**Spec coverage (design spec §3, §4):**

- §3.1 ResU-Net backbone with residual blocks and spatial dropout → Task 1 ✓
- §3.2 Cross-scale routing with 5 km² handoff → Task 2 ✓ (router) and Task 7 ✓ (two-head training)
- §3.3 Level-set physics loss with simplified Rothermel spread rate → Tasks 3 and 4 ✓
- §3.4 MC Dropout inference-time uncertainty → **partial**: Task 1 verifies dropout remains stochastic in train mode (the prerequisite for MC Dropout), but the actual N=20 forward-pass wrapper is deferred to Plan 3 where it belongs alongside the uncertainty rendering.
- §4.1 PyTorch env → Tasks 5/6 ✓ (plain PyTorch, Lightning omitted; documented in plan header)
- §4.2 Per-cell pipeline (preprocess → pretrain → fine-tune → checkpoint select) → Task 6 ✓ implements the loop; NDWS pretrain stage is documented as out-of-scope in the header.
- §4.3 Four-cell ablation grid → Task 7 ✓
- §4.4 Primary sweeps (scale × loss × 3 seeds) → Task 7 ✓. Secondary sweeps (λ_phys, handoff threshold) are explicitly deferred to Plan 3.
- §4.5 Fixed seeds across numpy/torch/cuda → Task 6 ✓ `set_seed`.

**Placeholder scan:** No "TBD" / "implement later" / vague steps. Every task has actual code.

**Type consistency:**
- `TrainConfig` is used identically in Tasks 5, 6, 7, 8. Field names match (`cache_root`, `out_dir`, `run_name`, `base_channels`, `lambda_phys`, …).
- `train_one_run` returns `{"best_val_iou": float, "best_epoch": float}` in Task 6 and is consumed with those keys in Task 7.
- `run_ablation` returns records with `{"run_name", "cell", "head", "seed", "best_val_iou", "best_epoch"}`; the Task 7 test asserts on exactly this set, and Task 8's `scripts/run_ablation.py` reads `r['run_name']`, `r['best_val_iou']`, `r['cell']`, `r['head']`, `r['seed']` consistently.
- `IgnisLoss` signature `(logits, target, features=None)` is used identically in Tasks 4 and 6.
- `ResUNet(in_channels=12, base=..., dropout=...)` — same kwargs in Tasks 1 and 6.
- Channel indices `0=fire_mask, 1=fuel, 4=slope, 7=wind_u, 8=wind_v` in Task 4's `IgnisLoss` must match `ignisca.data.features.CHANNEL_NAMES` from Plan 1 (`fire_mask, fuel_model, canopy_cover, elevation, slope, aspect_sin, aspect_cos, wind_u, wind_v, relative_humidity, temperature_k, days_since_rain`). Indices verified: fire_mask=0 ✓, fuel_model=1 ✓, slope=4 ✓, wind_u=7 ✓, wind_v=8 ✓.

**Known deviations from spec, documented in plan header:**

1. **No PyTorch Lightning.** Spec §4.1 mentions Lightning; plan uses plain PyTorch to keep the training loop auditable and to avoid a new dependency. Structure is simple enough that Lightning could be retrofitted later.
2. **No W&B.** Spec §4.1 and §4.5 mention W&B; this plan writes JSONL logs locally and defers experiment tracking to Plan 3, which owns reporting.
3. **No NDWS pretrain stage.** Spec §4.2 prescribes a 50-epoch NDWS pretrain before SoCal fine-tuning. Plan 2 trains only on caches produced by `scripts/preprocess.py` (Plan 1). Adding NDWS ingestion is a small plumbing task that can land as a Plan 1 follow-up without changing any Plan 2 interface.
4. **No full MC Dropout inference wrapper.** Task 1 verifies the dropout layers stay stochastic in train mode, which is the hard constraint MC Dropout imposes on the backbone. The actual `mc_predict(model, x, n=20)` function lives with the uncertainty rendering in Plan 3.

---

*End of Plan 2.*
