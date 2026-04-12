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
