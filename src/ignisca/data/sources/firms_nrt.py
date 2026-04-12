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
