from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from ignisca.data.grid import TargetGrid
from ignisca.data.sources.nifc import load_nifc_perimeter_at


def test_load_nifc_perimeter_rasterizes_correctly(tmp_path: Path):
    perim_t1 = Polygon([(-118.555, 34.055), (-118.545, 34.055), (-118.545, 34.065), (-118.555, 34.065)])
    perim_t2 = Polygon([(-118.56, 34.05), (-118.54, 34.05), (-118.54, 34.07), (-118.56, 34.07)])
    gdf = gpd.GeoDataFrame(
        {
            "timestamp": [datetime(2025, 1, 7, 20), datetime(2025, 1, 7, 23)],
            "geometry": [perim_t1, perim_t2],
        },
        crs="EPSG:4326",
    )
    path = tmp_path / "palisades.geojson"
    gdf.to_file(path, driver="GeoJSON")

    grid = TargetGrid.fine(center_lon=-118.55, center_lat=34.06, size_px=64)
    mask = load_nifc_perimeter_at(path, ts=datetime(2025, 1, 7, 21), target=grid)

    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    # At 21:00 we should see perim_t1 (the only one <= ts)
    assert mask.sum() > 0

    # Before any perimeter, expect empty
    mask0 = load_nifc_perimeter_at(path, ts=datetime(2025, 1, 7, 19), target=grid)
    assert mask0.sum() == 0
