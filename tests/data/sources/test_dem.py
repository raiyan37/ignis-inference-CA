import numpy as np

from ignisca.data.grid import TargetGrid
from ignisca.data.sources.dem import load_dem


def test_load_dem_returns_all_terrain_channels(synthetic_geotiff):
    # Synthetic ramp: elevation increases east-west → slope nonzero, aspect westward
    elev = np.tile(np.arange(32, dtype=np.float32) * 10.0, (32, 1))
    bounds = (-118.6, 34.0, -118.5, 34.1)
    dem_path = synthetic_geotiff("dem.tif", elev, bounds=bounds)

    grid = TargetGrid.fine(center_lon=-118.55, center_lat=34.05, size_px=32)
    out = load_dem(dem_path, target=grid)

    assert out.elevation.shape == (32, 32)
    assert out.slope.shape == (32, 32)
    assert out.aspect_sin.shape == (32, 32)
    assert out.aspect_cos.shape == (32, 32)
    # Slope should be strictly positive across the east-west ramp
    assert (out.slope > 0).sum() >= 30 * 30
    # Aspect sin/cos in [-1, 1]
    assert out.aspect_sin.min() >= -1.0 and out.aspect_sin.max() <= 1.0
    assert out.aspect_cos.min() >= -1.0 and out.aspect_cos.max() <= 1.0
