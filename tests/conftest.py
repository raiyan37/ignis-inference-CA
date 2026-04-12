from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_bounds
from shapely.geometry import Polygon


@pytest.fixture
def synthetic_geotiff(tmp_path: Path):
    """Factory that writes a small GeoTIFF and returns its path."""

    def _make(
        name: str,
        data: np.ndarray,
        crs: str = "EPSG:4326",
        bounds: tuple = (-118.6, 34.0, -118.5, 34.1),
        dtype: str = "float32",
    ) -> Path:
        if data.ndim != 2:
            raise ValueError("synthetic_geotiff expects 2-D data")
        h, w = data.shape
        transform = from_bounds(*bounds, w, h)
        path = tmp_path / name
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
        ) as ds:
            ds.write(data.astype(dtype), 1)
        return path

    return _make


@pytest.fixture
def synthetic_netcdf(tmp_path: Path):
    """Factory that writes a small netCDF and returns its path."""

    def _make(
        name: str,
        variables: dict,
        lats: np.ndarray,
        lons: np.ndarray,
        times: list,
    ) -> Path:
        ds = xr.Dataset(
            data_vars={
                var_name: (("time", "lat", "lon"), arr)
                for var_name, arr in variables.items()
            },
            coords={
                "time": times,
                "lat": lats,
                "lon": lons,
            },
        )
        path = tmp_path / name
        ds.to_netcdf(path)
        return path

    return _make


@pytest.fixture
def palisades_stub():
    """Small synthetic HeldOutFire for Palisades."""
    from ignisca.data.holdout import HeldOutFire

    return HeldOutFire(
        name="palisades_2025",
        ignition_utc=datetime(2025, 1, 7, 18, 30),
        perimeter=Polygon(
            [(-118.56, 34.05), (-118.50, 34.05), (-118.50, 34.10), (-118.56, 34.10)]
        ),
    )
