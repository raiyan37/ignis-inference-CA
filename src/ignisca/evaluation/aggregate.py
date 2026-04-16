from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

from ignisca.evaluation.runner import EvalResult

_METRICS: tuple[str, ...] = (
    "iou",
    "precision",
    "recall",
    "auc_pr",
    "ece",
    "growth_rate_mae",
    "mean_mc_variance",
)


@dataclass
class AggregatedRow:
    fire_id: str
    cell: str
    metrics: dict[str, Tuple[float, float]] = field(default_factory=dict)
    n_seeds: int = 0


def aggregate_cell(
    *,
    cell: str,
    fire_id: str,
    results: Iterable[EvalResult],
) -> AggregatedRow:
    """Collapse a list of per-seed EvalResults into a single AggregatedRow.

    Raises ValueError loudly on heterogeneous inputs — empty result list,
    mixed cells, or mixed fire_ids. We do NOT silently drop mismatches.
    """
    results = list(results)
    if not results:
        raise ValueError("aggregate_cell: empty results list")
    for r in results:
        if r.cell != cell:
            raise ValueError(f"aggregate_cell: result cell={r.cell!r} != {cell!r}")
        if r.fire_id != fire_id:
            raise ValueError(
                f"aggregate_cell: result fire_id={r.fire_id!r} != {fire_id!r}"
            )

    metrics: dict[str, Tuple[float, float]] = {}
    for metric in _METRICS:
        values = [float(getattr(r, metric)) for r in results]
        mean = statistics.fmean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        metrics[metric] = (mean, std)
    return AggregatedRow(
        fire_id=fire_id,
        cell=cell,
        metrics=metrics,
        n_seeds=len(results),
    )


def collect_runs(runs_root: Path) -> List[EvalResult]:
    """Walk runs_root looking for */eval.json and load them into EvalResults.

    The loaded EvalResults carry empty ``slices`` and empty path fields — this
    function is a thin adapter over the JSON files written by the runner, not
    a full round-trip. It is sufficient for ``aggregate_cell`` because
    aggregation only reads the top-level metric fields.
    """
    runs_root = Path(runs_root)
    results: List[EvalResult] = []
    for eval_json in sorted(runs_root.glob("*/eval.json")):
        payload = json.loads(eval_json.read_text())
        for fire in payload.get("fires", []):
            results.append(
                EvalResult(
                    run_name=payload["run_name"],
                    cell=payload["cell"],
                    seed=int(payload["seed"]),
                    fire_id=fire["fire_id"],
                    iou=float(fire["iou"]),
                    precision=float(fire["precision"]),
                    recall=float(fire["recall"]),
                    auc_pr=float(fire["auc_pr"]),
                    ece=float(fire["ece"]),
                    growth_rate_mae=float(fire["growth_rate_mae"]),
                    mean_mc_variance=float(fire["mean_mc_variance"]),
                    slices=fire.get("slices", {}),
                    n_samples=int(fire.get("n_samples", 0)),
                )
            )
    return results
