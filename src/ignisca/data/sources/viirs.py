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
    out["acq_datetime"] = [d.to_pydatetime() for d in out["acq_datetime"]]
    return out.sort_values("acq_datetime").reset_index(drop=True)
