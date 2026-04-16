from __future__ import annotations

import json
from pathlib import Path
from typing import List


def rank_failures(
    sample_metrics_jsonl: Path,
    k: int = 10,
    metric: str = "iou",
    mode: str = "worst",
) -> List[dict]:
    """Return the top-k samples by the given metric.

    ``mode="worst"`` sorts ascending (low IoU is worst); ``mode="best"`` sorts
    descending. Missing metric keys raise ``KeyError`` loudly rather than
    silently dropping rows.
    """
    if mode not in ("worst", "best"):
        raise ValueError(f"mode must be 'worst' or 'best', got {mode!r}")
    rows = [
        json.loads(line) for line in Path(sample_metrics_jsonl).read_text().splitlines() if line.strip()
    ]
    if not rows:
        return []
    missing = [i for i, r in enumerate(rows) if metric not in r]
    if missing:
        raise KeyError(
            f"rank_failures: {len(missing)} rows missing metric {metric!r} (first at idx {missing[0]})"
        )
    reverse = mode == "best"
    rows.sort(key=lambda r: r[metric], reverse=reverse)
    return rows[:k]
