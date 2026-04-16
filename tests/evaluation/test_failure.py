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


@pytest.mark.viz
def test_render_failure_case_writes_png(tmp_path: Path):
    import numpy as np

    matplotlib = pytest.importorskip("matplotlib")

    from ignisca.evaluation.failure import render_failure_case

    npz_path = tmp_path / "predictions_palisades_2025.npz"
    np.savez(
        npz_path,
        mean=np.random.rand(4, 16, 16).astype(np.float16),
        variance=(np.random.rand(4, 16, 16) * 0.1).astype(np.float16),
        target=(np.random.rand(4, 16, 16) > 0.5).astype(np.uint8),
        input_mask=(np.random.rand(4, 16, 16) > 0.8).astype(np.uint8),
    )

    out_path = tmp_path / "failure_2.png"
    render_failure_case(npz_path=npz_path, sample_idx=2, out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


@pytest.mark.viz
def test_render_failure_case_rejects_out_of_range_idx(tmp_path: Path):
    import numpy as np

    pytest.importorskip("matplotlib")

    from ignisca.evaluation.failure import render_failure_case

    npz_path = tmp_path / "predictions_palisades_2025.npz"
    np.savez(
        npz_path,
        mean=np.zeros((2, 4, 4), dtype=np.float16),
        variance=np.zeros((2, 4, 4), dtype=np.float16),
        target=np.zeros((2, 4, 4), dtype=np.uint8),
        input_mask=np.zeros((2, 4, 4), dtype=np.uint8),
    )
    out_path = tmp_path / "out.png"
    with pytest.raises(IndexError):
        render_failure_case(npz_path=npz_path, sample_idx=5, out_path=out_path)
