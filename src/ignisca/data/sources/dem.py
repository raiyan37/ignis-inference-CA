from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling

from ignisca.data.grid import TargetGrid, reproject_array


@dataclass
class DemSample:
    elevation: np.ndarray   # (H, W) meters
    slope: np.ndarray       # (H, W) degrees
    aspect_sin: np.ndarray  # (H, W) sin(aspect)
    aspect_cos: np.ndarray  # (H, W) cos(aspect)


def load_dem(dem_path: Path, target: TargetGrid) -> DemSample:
    elevation = _read_and_reproject(dem_path, target)
    slope_rad, aspect_rad = _slope_aspect(elevation, target.resolution_m)
    slope_deg = np.rad2deg(slope_rad)
    return DemSample(
        elevation=elevation.astype(np.float32),
        slope=slope_deg.astype(np.float32),
        aspect_sin=np.sin(aspect_rad).astype(np.float32),
        aspect_cos=np.cos(aspect_rad).astype(np.float32),
    )


def _read_and_reproject(path: Path, target: TargetGrid) -> np.ndarray:
    with rasterio.open(path) as ds:
        data = ds.read(1).astype(np.float32)
        src_crs = ds.crs.to_string()
        src_bounds = tuple(ds.bounds)
    return reproject_array(data, src_crs, src_bounds, target, resampling=Resampling.bilinear)


def _slope_aspect(elev: np.ndarray, cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Horn's (1981) slope and aspect via 3x3 window finite differences."""
    dzdx = np.zeros_like(elev)
    dzdy = np.zeros_like(elev)
    dzdx[:, 1:-1] = (elev[:, 2:] - elev[:, :-2]) / (2.0 * cell_size_m)
    dzdy[1:-1, :] = (elev[2:, :] - elev[:-2, :]) / (2.0 * cell_size_m)
    # Edges: replicate
    dzdx[:, 0] = dzdx[:, 1]
    dzdx[:, -1] = dzdx[:, -2]
    dzdy[0, :] = dzdy[1, :]
    dzdy[-1, :] = dzdy[-2, :]

    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    return slope, aspect
