from datetime import datetime

import numpy as np

from ignisca.data.grid import TargetGrid
from ignisca.data.sources.hrrr import load_hrrr_at


def test_load_hrrr_at_extracts_all_channels(synthetic_netcdf):
    lats = np.linspace(34.0, 34.1, 8)
    lons = np.linspace(-118.6, -118.5, 8)
    times = [datetime(2025, 1, 7, h) for h in range(12, 15)]
    shape = (3, 8, 8)

    variables = {
        "UGRD_10m": np.ones(shape, dtype=np.float32) * 5.0,
        "VGRD_10m": np.ones(shape, dtype=np.float32) * -3.0,
        "RH_2m": np.ones(shape, dtype=np.float32) * 20.0,
        "TMP_2m": np.ones(shape, dtype=np.float32) * 308.0,
        "APCP": np.zeros(shape, dtype=np.float32),  # no precip → high DSR
    }
    path = synthetic_netcdf("hrrr.nc", variables, lats, lons, times)

    grid = TargetGrid.fine(center_lon=-118.55, center_lat=34.05, size_px=32)
    out = load_hrrr_at(path, ts=datetime(2025, 1, 7, 13), target=grid)

    assert out.wind_u.shape == (32, 32)
    assert np.allclose(out.wind_u, 5.0, atol=0.1)
    assert np.allclose(out.wind_v, -3.0, atol=0.1)
    assert np.allclose(out.relative_humidity, 20.0, atol=0.1)
    assert np.allclose(out.temperature_k, 308.0, atol=0.1)
    # Days since rain: zero precip across full window → positive
    assert out.days_since_rain.mean() > 0


def test_days_since_rain_respects_timestep_interval(synthetic_netcdf):
    """DSR must convert slot counts to hours using actual dt, not assume hourly."""
    lats = np.linspace(34.0, 34.1, 8)
    lons = np.linspace(-118.6, -118.5, 8)
    # 3-hourly timesteps, all dry — 6 slots × 3h = 18 hours = 0.75 days
    times = [datetime(2025, 1, 7, h) for h in range(0, 18, 3)]
    shape = (6, 8, 8)
    variables = {
        "UGRD_10m": np.zeros(shape, dtype=np.float32),
        "VGRD_10m": np.zeros(shape, dtype=np.float32),
        "RH_2m": np.zeros(shape, dtype=np.float32),
        "TMP_2m": np.zeros(shape, dtype=np.float32),
        "APCP": np.zeros(shape, dtype=np.float32),
    }
    path = synthetic_netcdf("hrrr_3h.nc", variables, lats, lons, times)
    grid = TargetGrid.fine(center_lon=-118.55, center_lat=34.05, size_px=32)

    out = load_hrrr_at(path, ts=datetime(2025, 1, 7, 15), target=grid)

    # With dt=3h and 6 all-dry slots, DSR must be 18h/24 = 0.75 days, not 6/24 = 0.25
    assert np.allclose(out.days_since_rain.mean(), 0.75, atol=0.05)
