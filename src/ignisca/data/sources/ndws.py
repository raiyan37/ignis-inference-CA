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
    required = [
        "elevation", "sph", "pdsi", "NDVI", "pr", "tmmx", "erc",
        "vs", "th", "PrevFireMask", "FireMask",
    ]
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
