from __future__ import annotations

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

from ignisca.data.grid import TargetGrid


def load_nifc_perimeter_at(path: Path, ts: datetime, target: TargetGrid) -> np.ndarray:
    """Return the binary fire mask representing the union of all perimeters at or before `ts`."""
    gdf = gpd.read_file(path)
    if "timestamp" not in gdf.columns:
        raise ValueError(f"expected 'timestamp' column in {path}")

    gdf["timestamp"] = gdf["timestamp"].astype("datetime64[ns]")
    eligible = gdf[gdf["timestamp"] <= np.datetime64(ts)]
    if len(eligible) == 0:
        return np.zeros((target.height, target.width), dtype=np.uint8)

    eligible = eligible.to_crs(target.crs)
    shapes = [(geom, 1) for geom in eligible.geometry]
    mask = rasterize(
        shapes=shapes,
        out_shape=(target.height, target.width),
        transform=target.transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    return mask
