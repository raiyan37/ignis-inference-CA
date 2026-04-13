import torch

from ignisca.training.losses import (
    IgnisLoss,
    level_set_residual,
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
