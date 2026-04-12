from datetime import datetime, timedelta

import numpy as np
import rasterio
import xarray as xr


def test_synthetic_geotiff_factory(synthetic_geotiff):
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    path = synthetic_geotiff("t.tif", data)
    with rasterio.open(path) as ds:
        read = ds.read(1)
    np.testing.assert_allclose(read, data)


def test_synthetic_netcdf_factory(synthetic_netcdf):
    lats = np.linspace(34.0, 34.1, 4)
    lons = np.linspace(-118.6, -118.5, 4)
    times = [datetime(2025, 1, 7, 12) + timedelta(hours=i) for i in range(3)]
    data = np.random.rand(3, 4, 4).astype(np.float32)
    path = synthetic_netcdf("t.nc", {"wind_u": data}, lats, lons, times)
    loaded = xr.open_dataset(path)
    np.testing.assert_allclose(loaded["wind_u"].values, data)
