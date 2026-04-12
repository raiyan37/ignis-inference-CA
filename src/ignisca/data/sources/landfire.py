from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling

from ignisca.data.grid import TargetGrid, reproject_array


@dataclass
class LandfireSample:
    fuel_model: np.ndarray   # (H, W) float32 — categorical fuel code
    canopy_cover: np.ndarray  # (H, W) float32 — percent


def load_landfire(
    fuel_path: Path,
    canopy_path: Path,
    target: TargetGrid,
) -> LandfireSample:
    fuel = _read_and_reproject(fuel_path, target, Resampling.nearest)
    canopy = _read_and_reproject(canopy_path, target, Resampling.bilinear)
    return LandfireSample(fuel_model=fuel, canopy_cover=canopy)


def _read_and_reproject(
    path: Path, target: TargetGrid, resampling: Resampling
) -> np.ndarray:
    with rasterio.open(path) as ds:
        data = ds.read(1).astype(np.float32)
        src_crs = ds.crs.to_string()
        src_bounds = tuple(ds.bounds)
    return reproject_array(data, src_crs, src_bounds, target, resampling=resampling)
