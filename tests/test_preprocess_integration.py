from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from ignisca.data.dataset import IgnisDataset
from ignisca.data.holdout import HeldOutFire


def _write_fake_inputs(
    tmp_path: Path,
    synthetic_geotiff,
    synthetic_netcdf,
):
    """Materialize all archival source fixtures for a single synthetic fire."""
    bounds = (-118.6, 34.0, -118.5, 34.1)

    fuel = np.full((32, 32), 5.0, dtype=np.float32)
    canopy = np.full((32, 32), 40.0, dtype=np.float32)
    elev = np.tile(np.arange(32, dtype=np.float32) * 5.0, (32, 1))
    fuel_path = synthetic_geotiff("fuel.tif", fuel, bounds=bounds)
    canopy_path = synthetic_geotiff("canopy.tif", canopy, bounds=bounds)
    dem_path = synthetic_geotiff("dem.tif", elev, bounds=bounds)

    lats = np.linspace(34.0, 34.1, 8)
    lons = np.linspace(-118.6, -118.5, 8)
    times = [datetime(2018, 11, 9, h) for h in range(12, 18)]
    shape = (6, 8, 8)
    hrrr_path = synthetic_netcdf(
        "hrrr.nc",
        {
            "UGRD_10m": np.full(shape, 5.0, dtype=np.float32),
            "VGRD_10m": np.full(shape, -2.0, dtype=np.float32),
            "RH_2m": np.full(shape, 15.0, dtype=np.float32),
            "TMP_2m": np.full(shape, 305.0, dtype=np.float32),
            "APCP": np.zeros(shape, dtype=np.float32),
        },
        lats,
        lons,
        times,
    )

    perim_t1 = Polygon([(-118.555, 34.055), (-118.545, 34.055), (-118.545, 34.065), (-118.555, 34.065)])
    perim_t2 = Polygon([(-118.56, 34.05), (-118.54, 34.05), (-118.54, 34.07), (-118.56, 34.07)])
    nifc_path = tmp_path / "woolsey.geojson"
    gpd.GeoDataFrame(
        {
            "timestamp": [datetime(2018, 11, 9, 14), datetime(2018, 11, 9, 16)],
            "geometry": [perim_t1, perim_t2],
        },
        crs="EPSG:4326",
    ).to_file(nifc_path, driver="GeoJSON")

    viirs_path = tmp_path / "viirs.csv"
    pd.DataFrame(
        [
            {"latitude": 34.06, "longitude": -118.55, "acq_datetime": "2018-11-09 13:30", "confidence": 90},
        ]
    ).to_csv(viirs_path, index=False)

    return {
        "fuel": fuel_path,
        "canopy": canopy_path,
        "dem": dem_path,
        "hrrr": hrrr_path,
        "nifc": nifc_path,
        "viirs": viirs_path,
    }


def test_preprocess_end_to_end(tmp_path, synthetic_geotiff, synthetic_netcdf):
    paths = _write_fake_inputs(tmp_path, synthetic_geotiff, synthetic_netcdf)
    from scripts.preprocess import preprocess_fire

    cache_root = tmp_path / "cache"
    preprocess_fire(
        fire_name="woolsey_2018_smoke",
        center_lon=-118.55,
        center_lat=34.06,
        size_px=32,
        resolution="fine",
        timesteps=[datetime(2018, 11, 9, 13), datetime(2018, 11, 9, 15)],
        delta_hours=2,
        paths=paths,
        cache_root=cache_root,
        split="train",
        held_out=[],
    )

    shards = sorted((cache_root / "train").glob("*.npz"))
    assert len(shards) >= 1

    ds = IgnisDataset(cache_root, split="train")
    x, y = ds[0]
    assert x.shape == (12, 32, 32)
    assert y.shape == (32, 32)


def test_preprocess_honors_held_out_fires(tmp_path, synthetic_geotiff, synthetic_netcdf):
    """A held-out fire overlapping the tile must cause all timesteps to be skipped."""
    paths = _write_fake_inputs(tmp_path, synthetic_geotiff, synthetic_netcdf)
    from scripts.preprocess import preprocess_fire

    held_out = [
        HeldOutFire(
            name="woolsey_holdout",
            ignition_utc=datetime(2018, 11, 9, 12),
            perimeter=Polygon(
                [(-118.58, 34.02), (-118.52, 34.02), (-118.52, 34.08), (-118.58, 34.08)]
            ),
        )
    ]
    cache_root = tmp_path / "cache_holdout"
    n = preprocess_fire(
        fire_name="woolsey_2018_smoke",
        center_lon=-118.55,
        center_lat=34.06,
        size_px=32,
        resolution="fine",
        timesteps=[datetime(2018, 11, 9, 13), datetime(2018, 11, 9, 15)],
        delta_hours=2,
        paths=paths,
        cache_root=cache_root,
        split="train",
        held_out=held_out,
    )
    assert n == 0
    assert not (cache_root / "train").exists() or not any((cache_root / "train").glob("*.npz"))
