from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from rasterio.warp import Resampling

from ignisca.data.grid import TargetGrid, reproject_array


@dataclass
class HrrrSample:
    wind_u: np.ndarray           # (H, W) m/s
    wind_v: np.ndarray           # (H, W) m/s
    relative_humidity: np.ndarray  # (H, W) percent
    temperature_k: np.ndarray     # (H, W) Kelvin
    days_since_rain: np.ndarray  # (H, W) days


def load_hrrr_at(path: Path, ts: datetime, target: TargetGrid) -> HrrrSample:
    ds = xr.open_dataset(path)
    snap = ds.sel(time=ts, method="nearest")

    src_bounds = _bounds_from_coords(ds)
    src_crs = "EPSG:4326"

    def reproj(var: str) -> np.ndarray:
        arr = snap[var].values.astype(np.float32)
        return reproject_array(arr, src_crs, src_bounds, target, resampling=Resampling.bilinear)

    wind_u = reproj("UGRD_10m")
    wind_v = reproj("VGRD_10m")
    rh = reproj("RH_2m")
    tmp = reproj("TMP_2m")
    dsr = _days_since_rain(ds, ts, target)

    return HrrrSample(
        wind_u=wind_u,
        wind_v=wind_v,
        relative_humidity=rh,
        temperature_k=tmp,
        days_since_rain=dsr,
    )


def _bounds_from_coords(ds: xr.Dataset) -> tuple[float, float, float, float]:
    lons = ds["lon"].values
    lats = ds["lat"].values
    return (float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max()))


def _days_since_rain(
    ds: xr.Dataset, ts: datetime, target: TargetGrid, threshold_mm: float = 0.1
) -> np.ndarray:
    """Hours since the last pixel-wise APCP above threshold, converted to days.

    Uses the precipitation record from the start of the dataset up to `ts`.
    """
    window = ds.sel(time=slice(None, ts))
    precip = window["APCP"].values  # (T, H, W)
    if precip.size == 0:
        zeros = np.zeros((1, 1, 1), dtype=np.float32)
    else:
        zeros = (precip > threshold_mm).astype(np.float32)

    T, H, W = zeros.shape
    cumulative = np.full((H, W), T, dtype=np.float32)
    for t in range(T):
        wet = zeros[t] > 0
        cumulative[wet] = T - 1 - t
    dsr_hours = cumulative.astype(np.float32)
    dsr_days = dsr_hours / 24.0

    src_bounds = _bounds_from_coords(ds)
    return reproject_array(
        dsr_days, "EPSG:4326", src_bounds, target, resampling=Resampling.bilinear
    )
