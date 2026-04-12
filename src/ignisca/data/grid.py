from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

FINE_RESOLUTION_M = 30.0
COARSE_RESOLUTION_M = 375.0
TARGET_CRS = "EPSG:3857"


@dataclass(frozen=True)
class TargetGrid:
    crs: str
    resolution_m: float
    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy) in target CRS

    @property
    def width(self) -> int:
        minx, _, maxx, _ = self.bounds
        return int(round((maxx - minx) / self.resolution_m))

    @property
    def height(self) -> int:
        _, miny, _, maxy = self.bounds
        return int(round((maxy - miny) / self.resolution_m))

    @property
    def transform(self):
        return from_bounds(*self.bounds, self.width, self.height)

    @classmethod
    def _from_center(
        cls, center_lon: float, center_lat: float, resolution_m: float, size_px: int
    ) -> "TargetGrid":
        transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
        cx, cy = transformer.transform(center_lon, center_lat)
        half = size_px * resolution_m / 2.0
        bounds = (cx - half, cy - half, cx + half, cy + half)
        return cls(crs=TARGET_CRS, resolution_m=resolution_m, bounds=bounds)

    @classmethod
    def fine(cls, center_lon: float, center_lat: float, size_px: int = 512) -> "TargetGrid":
        return cls._from_center(center_lon, center_lat, FINE_RESOLUTION_M, size_px)

    @classmethod
    def coarse(cls, center_lon: float, center_lat: float, size_px: int = 64) -> "TargetGrid":
        return cls._from_center(center_lon, center_lat, COARSE_RESOLUTION_M, size_px)


def reproject_array(
    src: np.ndarray,
    src_crs: str,
    src_bounds: Tuple[float, float, float, float],
    target: TargetGrid,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    if src.ndim != 2:
        raise ValueError(f"reproject_array expects 2-D input, got shape {src.shape}")
    src_height, src_width = src.shape
    src_transform = from_bounds(*src_bounds, src_width, src_height)
    dst = np.zeros((target.height, target.width), dtype=src.dtype)
    reproject(
        source=src,
        destination=dst,
        src_transform=src_transform,
        src_crs=CRS.from_string(src_crs),
        dst_transform=target.transform,
        dst_crs=CRS.from_string(target.crs),
        resampling=resampling,
    )
    return dst
