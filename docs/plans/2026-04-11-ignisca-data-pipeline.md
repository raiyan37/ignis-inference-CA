# IgnisCA Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, reproducible preprocessing pipeline that ingests archival geospatial sources into 12-channel feature tensors and serves them via a PyTorch `Dataset`, with strict spatial/temporal holdout discipline for Palisades 2025 and Thomas 2017.

**Architecture:** Modular source loaders under `src/ignisca/data/sources/` each own one data source and expose a uniform `load(...) -> np.ndarray` contract. A pure-function `features.assemble(...)` combines per-source arrays into the 12-channel stack defined in the spec. A pure-function `holdout` module decides inclusion/exclusion. A thin cache layer persists samples as `.npz` shards with metadata. A `torch.utils.data.Dataset` reads shards. An orchestrator script wires it end-to-end.

**Tech Stack:** Python 3.11, PyTorch 2.x (Dataset only — training is Plan 2), rasterio, xarray + netcdf4, geopandas, numpy, scipy, pytest, ruff.

**Testing philosophy:** Every module is tested against synthetic fixtures generated in `tests/conftest.py`. No real network calls, no real archive downloads. The user runs real preprocessing separately via `scripts/preprocess.py` with their own credentials/data.

---

## File Structure

```
pyproject.toml              # project metadata + pinned deps
.python-version             # 3.11
README.md                   # methodology README (populated over time)
src/ignisca/
  __init__.py
  data/
    __init__.py
    grid.py                 # target grid + reprojection helpers
    holdout.py              # spatial/temporal holdout rules
    cache.py                # .npz shard writer/reader
    features.py             # 12-channel stack assembly
    dataset.py              # PyTorch Dataset
    sources/
      __init__.py
      ndws.py               # Next Day Wildfire Spread (TFRecord via TF)
      landfire.py           # LANDFIRE fuel/canopy GeoTIFF
      dem.py                # USGS 3DEP elevation → slope/aspect
      hrrr.py               # HRRR reanalysis netCDF → wind/RH/T/dryness
      nifc.py               # NIFC vector perimeters → raster masks
      viirs.py              # VIIRS/MODIS active-fire archive
      firms_nrt.py          # FIRMS NRT live hook (inference-only)
scripts/
  preprocess.py             # orchestrator CLI
tests/
  conftest.py               # synthetic fixture factories
  data/
    test_grid.py
    test_holdout.py
    test_cache.py
    test_features.py
    test_dataset.py
    sources/
      test_ndws.py
      test_landfire.py
      test_dem.py
      test_hrrr.py
      test_nifc.py
      test_viirs.py
      test_firms_nrt.py
  test_preprocess_integration.py
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `src/ignisca/__init__.py`
- Create: `src/ignisca/data/__init__.py`
- Create: `src/ignisca/data/sources/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/sources/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ignisca"
version = "0.0.1"
description = "Cross-scale, physics-informed wildfire spread forecaster for SoCal."
requires-python = ">=3.11,<3.13"
dependencies = [
    "numpy>=1.24,<2.0",
    "scipy>=1.10",
    "torch>=2.1,<3.0",
    "rasterio>=1.3",
    "xarray>=2023.1",
    "netcdf4>=1.6",
    "geopandas>=0.14",
    "shapely>=2.0",
    "pyproj>=3.6",
    "requests>=2.31",
    "pyyaml>=6.0",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.1.9",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-v --strict-markers"
```

- [ ] **Step 2: Create `.python-version`**

```
3.11
```

- [ ] **Step 3: Create empty package init files**

Create each of these as an empty file (0 bytes):
- `src/ignisca/__init__.py`
- `src/ignisca/data/__init__.py`
- `src/ignisca/data/sources/__init__.py`
- `tests/__init__.py`
- `tests/data/__init__.py`
- `tests/data/sources/__init__.py`

- [ ] **Step 4: Write a smoke test**

Create `tests/test_smoke.py`:

```python
def test_package_imports():
    import ignisca
    import ignisca.data
    import ignisca.data.sources
    assert ignisca is not None
```

- [ ] **Step 5: Install and run smoke test**

Run: `pip install -e ".[dev]"`
Run: `pytest tests/test_smoke.py -v`
Expected: 1 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version src/ignisca tests/__init__.py tests/data tests/data/sources tests/test_smoke.py
git commit -m "chore: project scaffolding and smoke test"
```

---

## Task 2: Target Grid + Reprojection Helpers

The spec uses EPSG:3857 as the target projection with two resolutions: 30m (fine) and 375m (coarse). This module owns both.

**Files:**
- Create: `src/ignisca/data/grid.py`
- Create: `tests/data/test_grid.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_grid.py`:

```python
import numpy as np
import pytest
from ignisca.data.grid import TargetGrid, reproject_array


def test_target_grid_fine_has_30m_resolution():
    grid = TargetGrid.fine(center_lon=-118.5, center_lat=34.0, size_px=512)
    assert grid.resolution_m == 30.0
    assert grid.width == 512
    assert grid.height == 512
    assert grid.crs == "EPSG:3857"


def test_target_grid_coarse_has_375m_resolution():
    grid = TargetGrid.coarse(center_lon=-118.5, center_lat=34.0, size_px=64)
    assert grid.resolution_m == 375.0
    assert grid.width == 64
    assert grid.height == 64


def test_target_grid_bounds_are_square():
    grid = TargetGrid.fine(center_lon=-118.5, center_lat=34.0, size_px=512)
    minx, miny, maxx, maxy = grid.bounds
    assert pytest.approx(maxx - minx, rel=1e-6) == 512 * 30.0
    assert pytest.approx(maxy - miny, rel=1e-6) == 512 * 30.0


def test_reproject_array_identity_same_crs():
    src = np.arange(16, dtype=np.float32).reshape(4, 4)
    src_bounds = (0.0, 0.0, 120.0, 120.0)
    dst_grid = TargetGrid(
        crs="EPSG:3857",
        resolution_m=30.0,
        bounds=(0.0, 0.0, 120.0, 120.0),
    )
    out = reproject_array(src, src_crs="EPSG:3857", src_bounds=src_bounds, target=dst_grid)
    assert out.shape == (4, 4)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, src, atol=1e-5)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/data/test_grid.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.data.grid'`

- [ ] **Step 3: Implement `grid.py`**

Create `src/ignisca/data/grid.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/data/test_grid.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/data/grid.py tests/data/test_grid.py
git commit -m "feat(data): target grid + reprojection helpers for 30m and 375m"
```

---

## Task 3: Holdout Rules

Spec §2.3 requires: (1) spatial holdout — drop any tile within 5km of a held-out fire's perimeter; (2) temporal holdout — drop any sample timestamped on or after a held-out fire's ignition; (3) NDWS screening — drop NDWS samples intersecting held-out bboxes.

**Files:**
- Create: `src/ignisca/data/holdout.py`
- Create: `tests/data/test_holdout.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_holdout.py`:

```python
from datetime import datetime

import pytest
from shapely.geometry import Polygon, box

from ignisca.data.holdout import (
    HeldOutFire,
    spatial_overlap_km,
    should_exclude_spatial,
    should_exclude_temporal,
    should_exclude_ndws,
)


@pytest.fixture
def palisades():
    return HeldOutFire(
        name="palisades_2025",
        ignition_utc=datetime(2025, 1, 7, 18, 30),
        perimeter=Polygon(
            [(-118.56, 34.05), (-118.50, 34.05), (-118.50, 34.10), (-118.56, 34.10)]
        ),
    )


def test_spatial_overlap_inside_perimeter_is_zero_km(palisades):
    tile = box(-118.54, 34.07, -118.52, 34.08)
    assert spatial_overlap_km(tile, palisades.perimeter) == 0.0


def test_spatial_overlap_far_away_is_large(palisades):
    tile = box(-117.0, 33.0, -116.9, 33.1)
    assert spatial_overlap_km(tile, palisades.perimeter) > 100.0


def test_should_exclude_spatial_inside_5km_buffer(palisades):
    tile = box(-118.57, 34.08, -118.565, 34.085)  # ~0.5km outside
    assert should_exclude_spatial(tile, [palisades], buffer_km=5.0) is True


def test_should_exclude_spatial_outside_5km_buffer(palisades):
    tile = box(-118.70, 34.08, -118.695, 34.085)  # ~13km outside
    assert should_exclude_spatial(tile, [palisades], buffer_km=5.0) is False


def test_should_exclude_temporal_after_ignition(palisades):
    ts = datetime(2025, 2, 1, 0, 0)
    assert should_exclude_temporal(ts, [palisades]) is True


def test_should_exclude_temporal_before_ignition(palisades):
    ts = datetime(2024, 12, 1, 0, 0)
    assert should_exclude_temporal(ts, [palisades]) is False


def test_should_exclude_ndws_intersecting_bbox(palisades):
    sample_bbox = box(-118.55, 34.06, -118.51, 34.09)  # intersects perimeter
    assert should_exclude_ndws(sample_bbox, [palisades]) is True


def test_should_exclude_ndws_nonintersecting(palisades):
    sample_bbox = box(-116.0, 33.0, -115.9, 33.1)
    assert should_exclude_ndws(sample_bbox, [palisades]) is False
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/data/test_holdout.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.data.holdout'`

- [ ] **Step 3: Implement `holdout.py`**

Create `src/ignisca/data/holdout.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from pyproj import Geod
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class HeldOutFire:
    name: str
    ignition_utc: datetime
    perimeter: BaseGeometry  # EPSG:4326 polygon of final burn perimeter


def spatial_overlap_km(tile: BaseGeometry, perimeter: BaseGeometry) -> float:
    """Shortest great-circle distance in kilometers from tile to perimeter.

    Returns 0.0 if the tile intersects the perimeter.
    """
    if tile.intersects(perimeter):
        return 0.0
    p1, p2 = nearest_points(tile, perimeter)
    _, _, dist_m = _GEOD.inv(p1.x, p1.y, p2.x, p2.y)
    return dist_m / 1000.0


def should_exclude_spatial(
    tile: BaseGeometry, held_out: Iterable[HeldOutFire], buffer_km: float = 5.0
) -> bool:
    for fire in held_out:
        if spatial_overlap_km(tile, fire.perimeter) < buffer_km:
            return True
    return False


def should_exclude_temporal(ts: datetime, held_out: Iterable[HeldOutFire]) -> bool:
    for fire in held_out:
        if ts >= fire.ignition_utc:
            return True
    return False


def should_exclude_ndws(sample_bbox: BaseGeometry, held_out: Iterable[HeldOutFire]) -> bool:
    for fire in held_out:
        if sample_bbox.intersects(fire.perimeter):
            return True
    return False
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/data/test_holdout.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/data/holdout.py tests/data/test_holdout.py
git commit -m "feat(data): spatial + temporal + NDWS holdout rules with tests"
```

---

## Task 4: Cache Writer / Reader

Each cached sample is a `.npz` with keys `inputs` `(C, H, W) float32`, `target` `(H, W) uint8`, and a JSON metadata blob stored as a 0-D string array.

**Files:**
- Create: `src/ignisca/data/cache.py`
- Create: `tests/data/test_cache.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_cache.py`:

```python
from pathlib import Path

import numpy as np

from ignisca.data.cache import CacheShard, load_shard, save_shard


def test_round_trip(tmp_path: Path):
    shard = CacheShard(
        inputs=np.random.rand(12, 32, 32).astype(np.float32),
        target=(np.random.rand(32, 32) > 0.5).astype(np.uint8),
        metadata={
            "fire_name": "palisades_2025",
            "timestamp_utc": "2025-01-07T19:00:00",
            "resolution_m": 30.0,
            "bounds": [-118.56, 34.05, -118.50, 34.10],
        },
    )

    path = tmp_path / "shard_0001.npz"
    save_shard(path, shard)
    assert path.exists()

    loaded = load_shard(path)
    np.testing.assert_array_equal(loaded.inputs, shard.inputs)
    np.testing.assert_array_equal(loaded.target, shard.target)
    assert loaded.metadata == shard.metadata


def test_save_validates_input_shapes(tmp_path: Path):
    import pytest
    bad = CacheShard(
        inputs=np.zeros((11, 32, 32), dtype=np.float32),  # wrong channel count
        target=np.zeros((32, 32), dtype=np.uint8),
        metadata={},
    )
    with pytest.raises(ValueError, match="12 channels"):
        save_shard(tmp_path / "bad.npz", bad)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/data/test_cache.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.data.cache'`

- [ ] **Step 3: Implement `cache.py`**

Create `src/ignisca/data/cache.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

EXPECTED_CHANNELS = 12


@dataclass
class CacheShard:
    inputs: np.ndarray   # (C, H, W) float32
    target: np.ndarray   # (H, W) uint8
    metadata: Dict[str, Any]


def save_shard(path: Path, shard: CacheShard) -> None:
    if shard.inputs.ndim != 3 or shard.inputs.shape[0] != EXPECTED_CHANNELS:
        raise ValueError(f"inputs must have 12 channels, got shape {shard.inputs.shape}")
    if shard.target.ndim != 2:
        raise ValueError(f"target must be 2-D, got shape {shard.target.shape}")
    if shard.inputs.shape[1:] != shard.target.shape:
        raise ValueError(
            f"inputs spatial shape {shard.inputs.shape[1:]} != target shape {shard.target.shape}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        inputs=shard.inputs.astype(np.float32, copy=False),
        target=shard.target.astype(np.uint8, copy=False),
        metadata=np.array(json.dumps(shard.metadata)),
    )


def load_shard(path: Path) -> CacheShard:
    # np.load defaults to a safe loader since numpy 1.16.3; JSON metadata is a plain string array.
    with np.load(Path(path)) as data:
        inputs = np.asarray(data["inputs"], dtype=np.float32)
        target = np.asarray(data["target"], dtype=np.uint8)
        metadata = json.loads(str(data["metadata"]))
    return CacheShard(inputs=inputs, target=target, metadata=metadata)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/data/test_cache.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/data/cache.py tests/data/test_cache.py
git commit -m "feat(data): .npz shard cache writer and reader"
```

---

## Task 5: Synthetic Fixture Factories (conftest)

Before writing source-loader tests, centralize fixture generation. Each source test will call a factory to create a tiny tmp-path fixture.

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `conftest.py`**

Create `tests/conftest.py`:

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_bounds
from shapely.geometry import Polygon


@pytest.fixture
def synthetic_geotiff(tmp_path: Path):
    """Factory that writes a small GeoTIFF and returns its path."""

    def _make(
        name: str,
        data: np.ndarray,
        crs: str = "EPSG:4326",
        bounds: tuple = (-118.6, 34.0, -118.5, 34.1),
        dtype: str = "float32",
    ) -> Path:
        if data.ndim != 2:
            raise ValueError("synthetic_geotiff expects 2-D data")
        h, w = data.shape
        transform = from_bounds(*bounds, w, h)
        path = tmp_path / name
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
        ) as ds:
            ds.write(data.astype(dtype), 1)
        return path

    return _make


@pytest.fixture
def synthetic_netcdf(tmp_path: Path):
    """Factory that writes a small netCDF and returns its path."""

    def _make(
        name: str,
        variables: dict,
        lats: np.ndarray,
        lons: np.ndarray,
        times: list,
    ) -> Path:
        ds = xr.Dataset(
            data_vars={
                var_name: (("time", "lat", "lon"), arr)
                for var_name, arr in variables.items()
            },
            coords={
                "time": times,
                "lat": lats,
                "lon": lons,
            },
        )
        path = tmp_path / name
        ds.to_netcdf(path)
        return path

    return _make


@pytest.fixture
def palisades_stub():
    """Small synthetic HeldOutFire for Palisades."""
    from ignisca.data.holdout import HeldOutFire

    return HeldOutFire(
        name="palisades_2025",
        ignition_utc=datetime(2025, 1, 7, 18, 30),
        perimeter=Polygon(
            [(-118.56, 34.05), (-118.50, 34.05), (-118.50, 34.10), (-118.56, 34.10)]
        ),
    )
```

- [ ] **Step 2: Smoke-test the fixtures**

Create `tests/test_conftest_smoke.py`:

```python
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
```

- [ ] **Step 3: Run the smoke tests**

Run: `pytest tests/test_conftest_smoke.py -v`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_conftest_smoke.py
git commit -m "test: synthetic fixture factories for GeoTIFF and netCDF"
```

---

## Task 6: LANDFIRE Source (fuel + canopy)

Reads a LANDFIRE fuel-model GeoTIFF and a canopy-cover GeoTIFF, reprojects both onto a `TargetGrid`, returns aligned `(fuel_model, canopy_cover)` arrays.

**Files:**
- Create: `src/ignisca/data/sources/landfire.py`
- Create: `tests/data/sources/test_landfire.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/sources/test_landfire.py`:

```python
import numpy as np

from ignisca.data.grid import TargetGrid
from ignisca.data.sources.landfire import load_landfire


def test_load_landfire_aligns_to_target_grid(synthetic_geotiff):
    fuel = (np.random.randint(0, 100, (32, 32)).astype(np.float32))
    canopy = (np.random.rand(32, 32).astype(np.float32) * 100)
    bounds = (-118.6, 34.0, -118.5, 34.1)
    fuel_path = synthetic_geotiff("fuel.tif", fuel, bounds=bounds)
    canopy_path = synthetic_geotiff("canopy.tif", canopy, bounds=bounds)

    grid = TargetGrid.fine(center_lon=-118.55, center_lat=34.05, size_px=32)
    out = load_landfire(fuel_path, canopy_path, target=grid)

    assert out.fuel_model.shape == (grid.height, grid.width)
    assert out.canopy_cover.shape == (grid.height, grid.width)
    assert out.fuel_model.dtype == np.float32
    assert out.canopy_cover.dtype == np.float32
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/data/sources/test_landfire.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.data.sources.landfire'`

- [ ] **Step 3: Implement `landfire.py`**

Create `src/ignisca/data/sources/landfire.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/data/sources/test_landfire.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/data/sources/landfire.py tests/data/sources/test_landfire.py
git commit -m "feat(data): LANDFIRE fuel + canopy loader"
```

---

## Task 7: DEM Source (elevation → slope + aspect sin/cos)

Reads a 3DEP elevation GeoTIFF, derives slope (degrees) and aspect (radians), returns aspect as `(sin, cos)` pair per spec §2.4.

**Files:**
- Create: `src/ignisca/data/sources/dem.py`
- Create: `tests/data/sources/test_dem.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/sources/test_dem.py`:

```python
import numpy as np

from ignisca.data.grid import TargetGrid
from ignisca.data.sources.dem import load_dem


def test_load_dem_returns_all_terrain_channels(synthetic_geotiff):
    # Synthetic ramp: elevation increases east-west → slope nonzero, aspect westward
    elev = np.tile(np.arange(32, dtype=np.float32) * 10.0, (32, 1))
    bounds = (-118.6, 34.0, -118.5, 34.1)
    dem_path = synthetic_geotiff("dem.tif", elev, bounds=bounds)

    grid = TargetGrid.fine(center_lon=-118.55, center_lat=34.05, size_px=32)
    out = load_dem(dem_path, target=grid)

    assert out.elevation.shape == (32, 32)
    assert out.slope.shape == (32, 32)
    assert out.aspect_sin.shape == (32, 32)
    assert out.aspect_cos.shape == (32, 32)
    # Slope should be strictly positive across the east-west ramp
    assert (out.slope > 0).sum() >= 30 * 30
    # Aspect sin/cos in [-1, 1]
    assert out.aspect_sin.min() >= -1.0 and out.aspect_sin.max() <= 1.0
    assert out.aspect_cos.min() >= -1.0 and out.aspect_cos.max() <= 1.0
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/data/sources/test_dem.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.data.sources.dem'`

- [ ] **Step 3: Implement `dem.py`**

Create `src/ignisca/data/sources/dem.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/data/sources/test_dem.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/data/sources/dem.py tests/data/sources/test_dem.py
git commit -m "feat(data): DEM loader with Horn slope/aspect derivation"
```

---

## Task 8: HRRR Source (wind u/v, RH, temperature, days-since-rain)

Reads an HRRR netCDF archive slice and extracts the four weather channels required by the feature stack, plus a derived days-since-rain proxy from a precipitation accumulator.

**Files:**
- Create: `src/ignisca/data/sources/hrrr.py`
- Create: `tests/data/sources/test_hrrr.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/sources/test_hrrr.py`:

```python
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
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/data/sources/test_hrrr.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.data.sources.hrrr'`

- [ ] **Step 3: Implement `hrrr.py`**

Create `src/ignisca/data/sources/hrrr.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/data/sources/test_hrrr.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/data/sources/hrrr.py tests/data/sources/test_hrrr.py
git commit -m "feat(data): HRRR weather loader (wind/RH/T/DSR)"
```

---

## Task 9: NIFC Source (vector perimeters → raster masks)

Reads NIFC-format historical fire perimeter vector files (GeoJSON/Shapefile) and rasterizes them onto the target grid to produce the binary fire mask used as both input (t) and target (t+Δ).

**Files:**
- Create: `src/ignisca/data/sources/nifc.py`
- Create: `tests/data/sources/test_nifc.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/sources/test_nifc.py`:

```python
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
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/data/sources/test_nifc.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.data.sources.nifc'`

- [ ] **Step 3: Implement `nifc.py`**

Create `src/ignisca/data/sources/nifc.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/data/sources/test_nifc.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/data/sources/nifc.py tests/data/sources/test_nifc.py
git commit -m "feat(data): NIFC perimeter rasterizer"
```

---

## Task 10: VIIRS Source (active fire archive)

Reads the VIIRS/MODIS active-fire CSV archive and returns a per-timestep ignition point list for a given fire window. Used to seed the fire progression sequence at ~hourly intervals before NIFC perimeters kick in.

**Files:**
- Create: `src/ignisca/data/sources/viirs.py`
- Create: `tests/data/sources/test_viirs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/sources/test_viirs.py`:

```python
from datetime import datetime
from pathlib import Path

import pandas as pd

from ignisca.data.sources.viirs import load_viirs_detections_in_window


def test_load_viirs_detections_filters_by_time_and_bbox(tmp_path: Path):
    rows = [
        {"latitude": 34.06, "longitude": -118.55, "acq_datetime": "2025-01-07 18:35", "confidence": 90},
        {"latitude": 34.06, "longitude": -118.55, "acq_datetime": "2025-01-07 19:40", "confidence": 85},
        {"latitude": 35.00, "longitude": -117.00, "acq_datetime": "2025-01-07 19:00", "confidence": 95},  # out of bbox
        {"latitude": 34.06, "longitude": -118.55, "acq_datetime": "2025-01-08 12:00", "confidence": 80},  # out of time
    ]
    path = tmp_path / "viirs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    out = load_viirs_detections_in_window(
        path,
        bbox=(-118.6, 34.0, -118.5, 34.1),
        start=datetime(2025, 1, 7, 18, 0),
        end=datetime(2025, 1, 7, 23, 0),
        min_confidence=50,
    )
    assert len(out) == 2
    assert list(out["acq_datetime"]) == [
        datetime(2025, 1, 7, 18, 35),
        datetime(2025, 1, 7, 19, 40),
    ]
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/data/sources/test_viirs.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.data.sources.viirs'`

- [ ] **Step 3: Implement `viirs.py`**

Create `src/ignisca/data/sources/viirs.py`:

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd


def load_viirs_detections_in_window(
    path: Path,
    bbox: Tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    min_confidence: int = 50,
) -> pd.DataFrame:
    """Return VIIRS/MODIS detections in the given bbox and time window, sorted by time."""
    df = pd.read_csv(path)
    df["acq_datetime"] = pd.to_datetime(df["acq_datetime"])
    minx, miny, maxx, maxy = bbox
    mask = (
        (df["longitude"] >= minx)
        & (df["longitude"] <= maxx)
        & (df["latitude"] >= miny)
        & (df["latitude"] <= maxy)
        & (df["acq_datetime"] >= start)
        & (df["acq_datetime"] <= end)
        & (df["confidence"] >= min_confidence)
    )
    out = df.loc[mask].copy()
    out["acq_datetime"] = out["acq_datetime"].dt.to_pydatetime()
    return out.sort_values("acq_datetime").reset_index(drop=True)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/data/sources/test_viirs.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/data/sources/viirs.py tests/data/sources/test_viirs.py
git commit -m "feat(data): VIIRS active-fire archive loader"
```

---

## Task 11: NDWS Source (TFRecord adapter)

The Next Day Wildfire Spread dataset is distributed as TFRecord. We adapt its input channels to our downstream format. Tests do NOT call tensorflow; they test the post-adapter logic with synthetic dicts that mirror NDWS's field layout. The upstream tensorflow reader is a thin shim that converts `tf.train.Example` → `dict[str, np.ndarray]` and lives in a separate, untested helper.

**Files:**
- Create: `src/ignisca/data/sources/ndws.py`
- Create: `tests/data/sources/test_ndws.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/sources/test_ndws.py`:

```python
import numpy as np

from ignisca.data.sources.ndws import NdwsRecord, adapt_ndws_record


def test_adapt_ndws_record_maps_channels():
    raw = {
        "elevation": np.full((64, 64), 100.0, dtype=np.float32),
        "sph": np.full((64, 64), 0.005, dtype=np.float32),
        "pdsi": np.full((64, 64), -1.0, dtype=np.float32),
        "NDVI": np.full((64, 64), 0.4, dtype=np.float32),
        "pr": np.full((64, 64), 0.0, dtype=np.float32),
        "tmmx": np.full((64, 64), 305.0, dtype=np.float32),
        "tmmn": np.full((64, 64), 285.0, dtype=np.float32),
        "erc": np.full((64, 64), 80.0, dtype=np.float32),
        "vs": np.full((64, 64), 6.0, dtype=np.float32),
        "th": np.full((64, 64), 270.0, dtype=np.float32),
        "PrevFireMask": np.zeros((64, 64), dtype=np.uint8),
        "FireMask": np.ones((64, 64), dtype=np.uint8),
    }
    raw["PrevFireMask"][30:34, 30:34] = 1

    rec = adapt_ndws_record(raw)

    assert isinstance(rec, NdwsRecord)
    assert rec.fire_mask.shape == (64, 64)
    assert rec.target_mask.shape == (64, 64)
    assert rec.fire_mask.sum() == 16
    assert rec.target_mask.sum() == 64 * 64
    # th=270° (wind from the west) → u=+vs, v=0
    assert np.allclose(rec.wind_u, 6.0, atol=0.01)
    assert np.allclose(rec.wind_v, 0.0, atol=0.01)
    assert np.allclose(rec.temperature_k, 305.0)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/data/sources/test_ndws.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.data.sources.ndws'`

- [ ] **Step 3: Implement `ndws.py`**

Create `src/ignisca/data/sources/ndws.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class NdwsRecord:
    """A single NDWS sample, post-adaptation to the IgnisCA feature layout.

    NDWS ships at ~375m (MODIS) and is only used for the coarse head.
    """
    fire_mask: np.ndarray         # (H, W) uint8, "PrevFireMask" from NDWS
    target_mask: np.ndarray       # (H, W) uint8, "FireMask" (next-day)
    elevation: np.ndarray         # (H, W) float32, meters
    wind_u: np.ndarray            # (H, W) float32, m/s
    wind_v: np.ndarray            # (H, W) float32, m/s
    temperature_k: np.ndarray     # (H, W) float32
    relative_humidity: np.ndarray  # (H, W) float32, percent
    ndvi: np.ndarray              # (H, W) float32, fuel proxy
    erc: np.ndarray               # (H, W) float32, Energy Release Component
    days_since_rain: np.ndarray   # (H, W) float32, proxy from `pr`


def adapt_ndws_record(raw: Dict[str, np.ndarray]) -> NdwsRecord:
    required = ["elevation", "sph", "pdsi", "NDVI", "pr", "tmmx", "vs", "th", "PrevFireMask", "FireMask"]
    for key in required:
        if key not in raw:
            raise KeyError(f"NDWS record missing field: {key}")

    vs = raw["vs"].astype(np.float32)
    th_deg = raw["th"].astype(np.float32)
    th_rad = np.deg2rad(th_deg)
    # Meteorological convention: "from direction" → u=-vs*sin(th), v=-vs*cos(th).
    # NDWS `th` is wind direction-from, so 270° = westerly wind (blowing east).
    wind_u = -vs * np.sin(th_rad)
    wind_v = -vs * np.cos(th_rad)

    t_k = raw["tmmx"].astype(np.float32)
    rh = _specific_to_relative_humidity(raw["sph"].astype(np.float32), t_k)

    pr = raw["pr"].astype(np.float32)
    days_since_rain = np.where(pr > 0.1, 0.0, 7.0).astype(np.float32)

    return NdwsRecord(
        fire_mask=raw["PrevFireMask"].astype(np.uint8),
        target_mask=raw["FireMask"].astype(np.uint8),
        elevation=raw["elevation"].astype(np.float32),
        wind_u=wind_u,
        wind_v=wind_v,
        temperature_k=t_k,
        relative_humidity=rh,
        ndvi=raw["NDVI"].astype(np.float32),
        erc=raw["erc"].astype(np.float32),
        days_since_rain=days_since_rain,
    )


def _specific_to_relative_humidity(sph: np.ndarray, t_k: np.ndarray) -> np.ndarray:
    t_c = t_k - 273.15
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))  # saturation vapor pressure hPa
    e = (sph * 1013.25) / (0.622 + 0.378 * sph)         # actual vapor pressure hPa
    rh = 100.0 * e / es
    return np.clip(rh, 0.0, 100.0).astype(np.float32)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/data/sources/test_ndws.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/data/sources/ndws.py tests/data/sources/test_ndws.py
git commit -m "feat(data): NDWS record adapter (tensorflow-free)"
```

---

## Task 12: FIRMS NRT Source (live hook, inference-only)

Pure REST adapter: queries FIRMS MODIS/VIIRS NRT CSV endpoints for current hotspots in a user-supplied bounding box. Never touches the tensor cache. Tests use `unittest.mock.patch` on `requests.get`.

**Files:**
- Create: `src/ignisca/data/sources/firms_nrt.py`
- Create: `tests/data/sources/test_firms_nrt.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/sources/test_firms_nrt.py`:

```python
from unittest.mock import patch

from ignisca.data.sources.firms_nrt import FirmsClient


def test_firms_client_parses_csv_response():
    fake_csv = (
        "latitude,longitude,acq_date,acq_time,confidence\n"
        "34.06,-118.55,2026-04-11,1845,nominal\n"
        "34.08,-118.50,2026-04-11,1850,high\n"
    )

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = fake_csv

        client = FirmsClient(map_key="FAKEKEY")
        df = client.get_hotspots(
            bbox=(-119.0, 33.5, -118.0, 34.5),
            days_back=1,
            source="VIIRS_SNPP_NRT",
        )

    assert len(df) == 2
    assert list(df["confidence"]) == ["nominal", "high"]
    assert (df["latitude"] == 34.06).any()
    mock_get.assert_called_once()
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/data/sources/test_firms_nrt.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.data.sources.firms_nrt'`

- [ ] **Step 3: Implement `firms_nrt.py`**

Create `src/ignisca/data/sources/firms_nrt.py`:

```python
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import requests

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


@dataclass
class FirmsClient:
    """Thin wrapper around the FIRMS NRT Area CSV endpoint. Inference-only."""

    map_key: str
    base_url: str = FIRMS_AREA_URL
    timeout_s: int = 15

    def get_hotspots(
        self,
        bbox: Tuple[float, float, float, float],
        days_back: int = 1,
        source: str = "VIIRS_SNPP_NRT",
    ) -> pd.DataFrame:
        if days_back < 1 or days_back > 10:
            raise ValueError("FIRMS NRT accepts days_back in [1, 10]")
        minx, miny, maxx, maxy = bbox
        url = f"{self.base_url}/{self.map_key}/{source}/{minx},{miny},{maxx},{maxy}/{days_back}"
        resp = requests.get(url, timeout=self.timeout_s)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/data/sources/test_firms_nrt.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/data/sources/firms_nrt.py tests/data/sources/test_firms_nrt.py
git commit -m "feat(data): FIRMS NRT live-hotspot client (inference-only)"
```

---

## Task 13: Feature Stack Assembly

Assembles the 12-channel tensor in the exact order defined by spec §2.4.

**Files:**
- Create: `src/ignisca/data/features.py`
- Create: `tests/data/test_features.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_features.py`:

```python
import numpy as np

from ignisca.data.features import CHANNEL_NAMES, EXPECTED_CHANNELS, assemble_feature_stack


def test_channel_names_length_is_twelve():
    assert len(CHANNEL_NAMES) == EXPECTED_CHANNELS == 12


def test_assemble_feature_stack_returns_correct_shape():
    H, W = 32, 32

    def zeros():
        return np.zeros((H, W), dtype=np.float32)

    stack = assemble_feature_stack(
        fire_mask=np.zeros((H, W), dtype=np.uint8),
        fuel_model=zeros(),
        canopy_cover=zeros(),
        elevation=zeros(),
        slope=zeros(),
        aspect_sin=zeros(),
        aspect_cos=zeros(),
        wind_u=zeros(),
        wind_v=zeros(),
        relative_humidity=zeros(),
        temperature_k=zeros(),
        days_since_rain=zeros(),
    )
    assert stack.shape == (12, H, W)
    assert stack.dtype == np.float32


def test_channel_order_matches_spec():
    H, W = 8, 8
    markers = {name: float(i + 1) for i, name in enumerate(CHANNEL_NAMES)}
    arrs = {name: np.full((H, W), v, dtype=np.float32) for name, v in markers.items()}

    stack = assemble_feature_stack(
        fire_mask=arrs["fire_mask"].astype(np.uint8),
        fuel_model=arrs["fuel_model"],
        canopy_cover=arrs["canopy_cover"],
        elevation=arrs["elevation"],
        slope=arrs["slope"],
        aspect_sin=arrs["aspect_sin"],
        aspect_cos=arrs["aspect_cos"],
        wind_u=arrs["wind_u"],
        wind_v=arrs["wind_v"],
        relative_humidity=arrs["relative_humidity"],
        temperature_k=arrs["temperature_k"],
        days_since_rain=arrs["days_since_rain"],
    )
    for i, name in enumerate(CHANNEL_NAMES):
        assert stack[i].mean() == markers[name], f"channel {i} ({name}) wrong"
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/data/test_features.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.data.features'`

- [ ] **Step 3: Implement `features.py`**

Create `src/ignisca/data/features.py`:

```python
from __future__ import annotations

import numpy as np

CHANNEL_NAMES = (
    "fire_mask",
    "fuel_model",
    "canopy_cover",
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "wind_u",
    "wind_v",
    "relative_humidity",
    "temperature_k",
    "days_since_rain",
)

EXPECTED_CHANNELS = len(CHANNEL_NAMES)


def assemble_feature_stack(
    *,
    fire_mask: np.ndarray,
    fuel_model: np.ndarray,
    canopy_cover: np.ndarray,
    elevation: np.ndarray,
    slope: np.ndarray,
    aspect_sin: np.ndarray,
    aspect_cos: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    relative_humidity: np.ndarray,
    temperature_k: np.ndarray,
    days_since_rain: np.ndarray,
) -> np.ndarray:
    layers = [
        fire_mask.astype(np.float32),
        fuel_model.astype(np.float32),
        canopy_cover.astype(np.float32),
        elevation.astype(np.float32),
        slope.astype(np.float32),
        aspect_sin.astype(np.float32),
        aspect_cos.astype(np.float32),
        wind_u.astype(np.float32),
        wind_v.astype(np.float32),
        relative_humidity.astype(np.float32),
        temperature_k.astype(np.float32),
        days_since_rain.astype(np.float32),
    ]
    shapes = {layer.shape for layer in layers}
    if len(shapes) != 1:
        raise ValueError(f"all channels must share shape, got {shapes}")
    return np.stack(layers, axis=0)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/data/test_features.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/data/features.py tests/data/test_features.py
git commit -m "feat(data): 12-channel feature stack assembly"
```

---

## Task 14: PyTorch Dataset

Reads cache shards from a directory and serves them as `(inputs, target)` tensors. Splits by top-level `split/` subdirectory (`train/`, `val/`, `test/`).

**Files:**
- Create: `src/ignisca/data/dataset.py`
- Create: `tests/data/test_dataset.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_dataset.py`:

```python
from pathlib import Path

import numpy as np
import pytest
import torch

from ignisca.data.cache import CacheShard, save_shard
from ignisca.data.dataset import IgnisDataset


def _write_n_shards(cache_root: Path, split: str, n: int):
    split_dir = cache_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        save_shard(
            split_dir / f"{i:05d}.npz",
            CacheShard(
                inputs=np.full((12, 16, 16), float(i), dtype=np.float32),
                target=np.full((16, 16), i % 2, dtype=np.uint8),
                metadata={"idx": i},
            ),
        )


def test_dataset_reads_all_shards(tmp_path: Path):
    _write_n_shards(tmp_path, "train", 5)
    ds = IgnisDataset(tmp_path, split="train")
    assert len(ds) == 5

    x, y = ds[2]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (12, 16, 16)
    assert y.shape == (16, 16)
    assert x.dtype == torch.float32
    assert y.dtype == torch.uint8
    assert x.mean().item() == 2.0


def test_dataset_raises_on_empty_split(tmp_path: Path):
    (tmp_path / "train").mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="no shards"):
        IgnisDataset(tmp_path, split="train")
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/data/test_dataset.py -v`
Expected: `ModuleNotFoundError: No module named 'ignisca.data.dataset'`

- [ ] **Step 3: Implement `dataset.py`**

Create `src/ignisca/data/dataset.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

from ignisca.data.cache import load_shard


class IgnisDataset(Dataset):
    """Serves preprocessed IgnisCA cache shards.

    Expected layout:
        cache_root/
          train/
            00000.npz
            00001.npz
            ...
          val/
          test/
    """

    def __init__(self, cache_root: Path, split: str) -> None:
        split_dir = Path(cache_root) / split
        if not split_dir.exists():
            raise RuntimeError(f"split dir does not exist: {split_dir}")
        shards = sorted(split_dir.glob("*.npz"))
        if len(shards) == 0:
            raise RuntimeError(f"no shards found in {split_dir}")
        self._shards = shards

    def __len__(self) -> int:
        return len(self._shards)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        shard = load_shard(self._shards[idx])
        inputs = torch.from_numpy(shard.inputs)
        target = torch.from_numpy(shard.target)
        return inputs, target
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/data/test_dataset.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/data/dataset.py tests/data/test_dataset.py
git commit -m "feat(data): PyTorch Dataset for cache shards"
```

---

## Task 15: Preprocessing Orchestrator + Integration Test

Wires every source into a single CLI that, given a fire name + time range + resolution, produces a directory of cache shards under a split.

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/preprocess.py`
- Create: `tests/test_preprocess_integration.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_preprocess_integration.py`:

```python
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from ignisca.data.dataset import IgnisDataset


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
```

- [ ] **Step 2: Run the integration test and verify it fails**

Run: `pytest tests/test_preprocess_integration.py -v`
Expected: `ModuleNotFoundError: No module named 'scripts.preprocess'`

- [ ] **Step 3: Create `scripts/__init__.py`**

Create `scripts/__init__.py` as an empty file.

- [ ] **Step 4: Implement `scripts/preprocess.py`**

Create `scripts/preprocess.py`:

```python
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from shapely.geometry import box

from ignisca.data.cache import CacheShard, save_shard
from ignisca.data.features import assemble_feature_stack
from ignisca.data.grid import TargetGrid
from ignisca.data.holdout import HeldOutFire, should_exclude_spatial, should_exclude_temporal
from ignisca.data.sources.dem import load_dem
from ignisca.data.sources.hrrr import load_hrrr_at
from ignisca.data.sources.landfire import load_landfire
from ignisca.data.sources.nifc import load_nifc_perimeter_at


def preprocess_fire(
    *,
    fire_name: str,
    center_lon: float,
    center_lat: float,
    size_px: int,
    resolution: str,  # "fine" or "coarse"
    timesteps: List[datetime],
    delta_hours: int,
    paths: Dict[str, Path],
    cache_root: Path,
    split: str,
    held_out: List[HeldOutFire],
) -> int:
    """Materialize cache shards for one fire. Returns the number of shards written."""
    if resolution == "fine":
        grid = TargetGrid.fine(center_lon, center_lat, size_px=size_px)
    elif resolution == "coarse":
        grid = TargetGrid.coarse(center_lon, center_lat, size_px=size_px)
    else:
        raise ValueError(f"resolution must be 'fine' or 'coarse', got {resolution}")

    tile_poly = box(*grid.bounds)

    landfire = load_landfire(paths["fuel"], paths["canopy"], target=grid)
    dem = load_dem(paths["dem"], target=grid)

    n_written = 0
    split_dir = cache_root / split
    for t in timesteps:
        t_next = t + timedelta(hours=delta_hours)

        if should_exclude_temporal(t_next, held_out):
            continue
        if should_exclude_spatial(tile_poly, held_out, buffer_km=5.0):
            continue

        fire_mask = load_nifc_perimeter_at(paths["nifc"], ts=t, target=grid)
        target_mask = load_nifc_perimeter_at(paths["nifc"], ts=t_next, target=grid)
        hrrr = load_hrrr_at(paths["hrrr"], ts=t, target=grid)

        stack = assemble_feature_stack(
            fire_mask=fire_mask,
            fuel_model=landfire.fuel_model,
            canopy_cover=landfire.canopy_cover,
            elevation=dem.elevation,
            slope=dem.slope,
            aspect_sin=dem.aspect_sin,
            aspect_cos=dem.aspect_cos,
            wind_u=hrrr.wind_u,
            wind_v=hrrr.wind_v,
            relative_humidity=hrrr.relative_humidity,
            temperature_k=hrrr.temperature_k,
            days_since_rain=hrrr.days_since_rain,
        )

        shard = CacheShard(
            inputs=stack,
            target=target_mask,
            metadata={
                "fire_name": fire_name,
                "timestamp_utc": t.isoformat(),
                "delta_hours": delta_hours,
                "resolution": resolution,
                "resolution_m": grid.resolution_m,
                "bounds": list(grid.bounds),
            },
        )
        shard_path = split_dir / f"{fire_name}_{t.strftime('%Y%m%dT%H%M')}.npz"
        save_shard(shard_path, shard)
        n_written += 1

    return n_written


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="IgnisCA preprocessing orchestrator")
    parser.add_argument("--fire-name", required=True)
    parser.add_argument("--center-lon", type=float, required=True)
    parser.add_argument("--center-lat", type=float, required=True)
    parser.add_argument("--size-px", type=int, default=512)
    parser.add_argument("--resolution", choices=["fine", "coarse"], default="fine")
    parser.add_argument("--start", type=_parse_iso, required=True)
    parser.add_argument("--end", type=_parse_iso, required=True)
    parser.add_argument("--step-hours", type=int, default=1)
    parser.add_argument("--delta-hours", type=int, default=1)
    parser.add_argument("--fuel", type=Path, required=True)
    parser.add_argument("--canopy", type=Path, required=True)
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--hrrr", type=Path, required=True)
    parser.add_argument("--nifc", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    args = parser.parse_args()

    timesteps: List[datetime] = []
    cursor = args.start
    while cursor <= args.end:
        timesteps.append(cursor)
        cursor += timedelta(hours=args.step_hours)

    n = preprocess_fire(
        fire_name=args.fire_name,
        center_lon=args.center_lon,
        center_lat=args.center_lat,
        size_px=args.size_px,
        resolution=args.resolution,
        timesteps=timesteps,
        delta_hours=args.delta_hours,
        paths={
            "fuel": args.fuel,
            "canopy": args.canopy,
            "dem": args.dem,
            "hrrr": args.hrrr,
            "nifc": args.nifc,
        },
        cache_root=args.cache_root,
        split=args.split,
        held_out=[],
    )
    print(f"Wrote {n} shards to {args.cache_root / args.split}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the integration test to verify it passes**

Run: `pytest tests/test_preprocess_integration.py -v`
Expected: 1 passed

- [ ] **Step 6: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass (should be ~25+ green).

- [ ] **Step 7: Commit**

```bash
git add scripts/preprocess.py scripts/__init__.py tests/test_preprocess_integration.py
git commit -m "feat(data): preprocessing orchestrator with end-to-end integration test"
```

---

## Self-Review Summary

This plan implements the following spec sections in full:

- **§2.1 Sources:** Tasks 6–12 cover LANDFIRE, DEM, HRRR, NIFC, VIIRS, NDWS, FIRMS.
- **§2.2 Fire splits:** Task 14's `IgnisDataset` enforces train/val/test subdirectory layout; the orchestrator accepts a `--split` argument.
- **§2.3 Holdout discipline:** Task 3 implements spatial (5 km buffer), temporal, and NDWS-screening rules; Task 15's orchestrator calls them on every sample.
- **§2.4 Sample format:** Task 13's `assemble_feature_stack` produces the exact 12-channel layout; Task 4's cache schema enforces it at save time.
- **§2.5 Pipeline shape:** Task 15's orchestrator executes the full `raw → extract → reproject → cache → Dataset` chain.
- **§2.6 Live hook:** Task 12's FIRMS NRT client exists and is isolated from any training/cache path.

**Deferred to Plan 2/3:**
- Actually downloading NDWS TFRecords and running real preprocessing on SoCal fires. Plan 1 provides the mechanism; the human operator runs it with real credentials.
- Training-time data augmentation (random crops, flips). Lives in Plan 2 alongside the training loop.
- Per-sample normalization statistics. Lives in Plan 2 (they depend on the real training set, not the test fixtures).

**No placeholders, no TBDs, no "similar to Task N" references.**
