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
    with xr.open_dataset(path) as ds:
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
        H = ds.sizes["lat"] if "lat" in ds.sizes else ds["lat"].shape[0]
        W = ds.sizes["lon"] if "lon" in ds.sizes else ds["lon"].shape[0]
        dsr_days = np.full((H, W), 30.0, dtype=np.float32)
    else:
        wet = (precip > threshold_mm).astype(np.float32)
        T, H, W = wet.shape
        # Slots-since-last-wet. Default T means "no wet slot found in window".
        cumulative = np.full((H, W), T, dtype=np.float32)
        for t in range(T):
            mask = wet[t] > 0
            cumulative[mask] = T - 1 - t
        # Convert slot count to hours using the actual native timestep interval.
        times = window["time"].values
        if len(times) >= 2:
            dt_hours = float((times[1] - times[0]) / np.timedelta64(1, "h"))
        else:
            dt_hours = 1.0
        dsr_days = (cumulative * dt_hours / 24.0).astype(np.float32)

    src_bounds = _bounds_from_coords(ds)
    return reproject_array(
        dsr_days, "EPSG:4326", src_bounds, target, resampling=Resampling.bilinear
    )
