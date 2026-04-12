import numpy as np

from ignisca.data.grid import TargetGrid
from ignisca.data.sources.landfire import load_landfire


def test_load_landfire_aligns_to_target_grid(synthetic_geotiff):
    fuel = (np.random.randint(0, 100, (32, 32)).astype(np.float32))
    canopy = (np.random.rand(32, 32).astype(np.float32) * 100)
    bounds = (-118.6, 34.0, -118.5, 34.1)
    fuel_path = synthetic_geotiff("fuel.tif", fuel, bounds=bounds)
    canopy_path = synthetic_geotiff("canopy.tif", canopy, bounds=bounds)

    grid = TargetGrid.fine(center_lon=-118.55, center_lat=34.05, size_px=32)
    out = load_landfire(fuel_path, canopy_path, target=grid)

    assert out.fuel_model.shape == (grid.height, grid.width)
    assert out.canopy_cover.shape == (grid.height, grid.width)
    assert out.fuel_model.dtype == np.float32
    assert out.canopy_cover.dtype == np.float32
