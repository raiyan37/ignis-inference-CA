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
