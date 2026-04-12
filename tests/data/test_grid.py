import numpy as np
import pytest
from ignisca.data.grid import TargetGrid, reproject_array


def test_target_grid_fine_has_30m_resolution():
    grid = TargetGrid.fine(center_lon=-118.5, center_lat=34.0, size_px=512)
    assert grid.resolution_m == 30.0
    assert grid.width == 512
    assert grid.height == 512
    assert grid.crs == "EPSG:3857"


def test_target_grid_coarse_has_375m_resolution():
    grid = TargetGrid.coarse(center_lon=-118.5, center_lat=34.0, size_px=64)
    assert grid.resolution_m == 375.0
    assert grid.width == 64
    assert grid.height == 64


def test_target_grid_bounds_are_square():
    grid = TargetGrid.fine(center_lon=-118.5, center_lat=34.0, size_px=512)
    minx, miny, maxx, maxy = grid.bounds
    assert pytest.approx(maxx - minx, rel=1e-6) == 512 * 30.0
    assert pytest.approx(maxy - miny, rel=1e-6) == 512 * 30.0


def test_reproject_array_identity_same_crs():
    src = np.arange(16, dtype=np.float32).reshape(4, 4)
    src_bounds = (0.0, 0.0, 120.0, 120.0)
    dst_grid = TargetGrid(
        crs="EPSG:3857",
        resolution_m=30.0,
        bounds=(0.0, 0.0, 120.0, 120.0),
    )
    out = reproject_array(src, src_crs="EPSG:3857", src_bounds=src_bounds, target=dst_grid)
    assert out.shape == (4, 4)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, src, atol=1e-5)
