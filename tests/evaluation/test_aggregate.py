import json
import math
from pathlib import Path

import pytest

from ignisca.evaluation.aggregate import AggregatedRow, aggregate_cell, collect_runs
from ignisca.evaluation.runner import EvalResult


def _eval_result(seed: int, iou: float, fire_id: str = "palisades_2025") -> EvalResult:
    return EvalResult(
        run_name=f"cell_A1_seed{seed}",
        cell="A1",
        seed=seed,
        fire_id=fire_id,
        iou=iou,
        precision=iou,
        recall=iou,
        auc_pr=iou,
        ece=0.1,
        growth_rate_mae=0.5,
        mean_mc_variance=0.02,
        slices={},
        predictions_path=Path(""),
        sample_metrics_path=Path(""),
        n_samples=10,
    )


def test_aggregate_cell_computes_mean_and_std():
    results = [_eval_result(0, 0.60), _eval_result(1, 0.62), _eval_result(2, 0.58)]
    row = aggregate_cell(cell="A1", fire_id="palisades_2025", results=results)
    assert isinstance(row, AggregatedRow)
    assert row.cell == "A1"
    assert row.fire_id == "palisades_2025"
    assert row.n_seeds == 3
    iou_mean, iou_std = row.metrics["iou"]
    assert math.isclose(iou_mean, 0.60, abs_tol=1e-6)
    assert iou_std > 0
    ece_mean, ece_std = row.metrics["ece"]
    assert math.isclose(ece_mean, 0.1, abs_tol=1e-6)
    assert math.isclose(ece_std, 0.0, abs_tol=1e-6)


def test_aggregate_cell_rejects_mixed_cells():
    a = _eval_result(0, 0.6)
    b = _eval_result(1, 0.6)
    b.cell = "A2"
    with pytest.raises(ValueError, match="cell"):
        aggregate_cell(cell="A1", fire_id="palisades_2025", results=[a, b])


def test_aggregate_cell_rejects_mixed_fire_ids():
    a = _eval_result(0, 0.6)
    b = _eval_result(1, 0.6, fire_id="thomas_2017")
    with pytest.raises(ValueError, match="fire_id"):
        aggregate_cell(cell="A1", fire_id="palisades_2025", results=[a, b])


def test_aggregate_cell_rejects_empty_results():
    with pytest.raises(ValueError, match="empty"):
        aggregate_cell(cell="A1", fire_id="palisades_2025", results=[])


def test_collect_runs_reads_eval_json(tmp_path: Path):
    run_dir = tmp_path / "cell_A1_seed0"
    run_dir.mkdir()
    (run_dir / "eval.json").write_text(json.dumps({
        "run_name": "cell_A1_seed0",
        "cell": "A1",
        "seed": 0,
        "fires": [{
            "fire_id": "palisades_2025",
            "iou": 0.61, "precision": 0.58, "recall": 0.63,
            "auc_pr": 0.71, "ece": 0.09, "growth_rate_mae": 0.42,
            "mean_mc_variance": 0.018, "slices": {}, "n_samples": 100,
        }],
        "mc_dropout": {"n_samples": 20},
        "artifacts": {},
    }))
    results = collect_runs(tmp_path)
    assert len(results) == 1
    r = results[0]
    assert r.run_name == "cell_A1_seed0"
    assert r.cell == "A1"
    assert r.seed == 0
    assert r.fire_id == "palisades_2025"
    assert math.isclose(r.iou, 0.61)
