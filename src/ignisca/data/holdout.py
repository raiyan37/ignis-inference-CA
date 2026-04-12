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
