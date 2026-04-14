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


@pytest.fixture
def tiny_resunet():
    """ResU-Net(base=4) on CPU — ~5k params, fast enough for every eval test.

    H=W=32 is the smallest size that survives four 2x downsamples (32 / 16 = 2).
    """
    import torch  # noqa: F401 — imported lazily so fixture collection is cheap

    from ignisca.models.resunet import ResUNet

    return ResUNet(in_channels=12, base=4, dropout=0.3)


@pytest.fixture
def tiny_checkpoint(tmp_path, tiny_resunet):
    """Save the tiny ResU-Net to a .pt file matching Plan 2's checkpoint schema."""
    import torch

    ckpt_path = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": tiny_resunet.state_dict(),
            "config": {"base_channels": 4, "dropout": 0.3},
        },
        ckpt_path,
    )
    return ckpt_path


@pytest.fixture
def synthetic_batch():
    """(x, y) with plausible per-channel distributions matching CHANNEL_NAMES."""
    import torch

    torch.manual_seed(0)
    x = torch.randn(4, 12, 32, 32) * 0.5
    x[:, 0] = (torch.rand(4, 32, 32) > 0.85).float()   # fire_mask
    x[:, 1] = torch.rand(4, 32, 32)                    # fuel_model
    x[:, 4] = torch.relu(torch.randn(4, 32, 32))       # slope >= 0
    x[:, 7] = torch.randn(4, 32, 32) * 5               # wind_u
    x[:, 8] = torch.randn(4, 32, 32) * 5               # wind_v
    y = (torch.rand(4, 32, 32) > 0.80).float()
    return x, y


@pytest.fixture
def santa_ana_batch():
    """Uniform SW-flowing wind (u=-7.07, v=-7.07) ≈ 10 m/s from NE (offshore).

    A ground-truth positive case for the Santa Ana classifier.
    """
    import torch

    x = torch.zeros(2, 12, 32, 32)
    x[:, 7] = -7.07
    x[:, 8] = -7.07
    return x
