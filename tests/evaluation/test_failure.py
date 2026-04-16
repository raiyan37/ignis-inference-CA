import json
from pathlib import Path

import pytest

from ignisca.evaluation.failure import rank_failures


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_rank_failures_picks_worst_by_iou(tmp_path: Path):
    jsonl = tmp_path / "sample_metrics_palisades_2025.jsonl"
    _write_jsonl(jsonl, [
        {"sample_idx": 0, "iou": 0.80, "fire_id": "palisades_2025"},
        {"sample_idx": 1, "iou": 0.10, "fire_id": "palisades_2025"},
        {"sample_idx": 2, "iou": 0.40, "fire_id": "palisades_2025"},
        {"sample_idx": 3, "iou": 0.05, "fire_id": "palisades_2025"},
    ])
    ranked = rank_failures(jsonl, k=2)
    assert len(ranked) == 2
    assert ranked[0]["sample_idx"] == 3  # lowest IoU first
    assert ranked[1]["sample_idx"] == 1


def test_rank_failures_best_mode_returns_highest(tmp_path: Path):
    jsonl = tmp_path / "sm.jsonl"
    _write_jsonl(jsonl, [
        {"sample_idx": 0, "iou": 0.80},
        {"sample_idx": 1, "iou": 0.10},
    ])
    ranked = rank_failures(jsonl, k=1, mode="best")
    assert ranked[0]["sample_idx"] == 0


def test_rank_failures_k_exceeds_rows_returns_all(tmp_path: Path):
    jsonl = tmp_path / "sm.jsonl"
    _write_jsonl(jsonl, [{"sample_idx": 0, "iou": 0.5}])
    assert len(rank_failures(jsonl, k=100)) == 1


def test_rank_failures_supports_alternate_metric(tmp_path: Path):
    jsonl = tmp_path / "sm.jsonl"
    _write_jsonl(jsonl, [
        {"sample_idx": 0, "iou": 0.9, "growth_rate_abs_err_km2_hr": 2.0},
        {"sample_idx": 1, "iou": 0.1, "growth_rate_abs_err_km2_hr": 0.1},
    ])
    ranked = rank_failures(jsonl, k=1, metric="growth_rate_abs_err_km2_hr", mode="best")
    assert ranked[0]["sample_idx"] == 0


def test_rank_failures_missing_metric_raises(tmp_path: Path):
    jsonl = tmp_path / "sm.jsonl"
    _write_jsonl(jsonl, [{"sample_idx": 0, "iou": 0.5}])
    with pytest.raises(KeyError):
        rank_failures(jsonl, k=1, metric="nonexistent")
