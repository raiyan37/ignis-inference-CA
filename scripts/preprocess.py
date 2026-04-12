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
