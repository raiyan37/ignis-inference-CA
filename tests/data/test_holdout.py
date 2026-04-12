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
