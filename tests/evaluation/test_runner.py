import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ignisca.evaluation.runner import EvalResult, evaluate_run
from tests.evaluation.fake_dataset import FakeFireDataset


def _build_loader() -> DataLoader:
    return DataLoader(FakeFireDataset(n_samples=8, hw=32), batch_size=2, shuffle=False)


def test_evaluate_run_writes_all_artifacts(tmp_path: Path, tiny_checkpoint: Path):
    loader = _build_loader()
    run_dir = tmp_path / "cell_A1_seed0"
    run_dir.mkdir()

    result = evaluate_run(
        run_dir=run_dir,
        checkpoint_path=tiny_checkpoint,
        loader=loader,
        cell="A1",
        seed=0,
        fire_id="palisades_2025",
        pixel_area_km2=0.140625,
        mc_samples=3,
    )

    assert isinstance(result, EvalResult)
    assert result.run_name == run_dir.name
    assert result.cell == "A1"
    assert result.seed == 0
    assert result.fire_id == "palisades_2025"
    assert result.n_samples == 8
    assert 0.0 <= result.iou <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.auc_pr <= 1.0
    assert 0.0 <= result.ece <= 1.0
    assert result.growth_rate_mae >= 0.0
    assert result.mean_mc_variance >= 0.0
    assert set(result.slices.keys()) == {"santa_ana", "non_santa_ana", "early", "mature"}
    for slice_metrics in result.slices.values():
        assert set(slice_metrics.keys()) >= {
            "iou", "precision", "recall", "auc_pr", "ece", "growth_rate_mae", "mean_mc_variance"
        }

    eval_json = run_dir / "eval.json"
    assert eval_json.exists()
    payload = json.loads(eval_json.read_text())
    assert payload["run_name"] == run_dir.name
    assert payload["cell"] == "A1"
    assert payload["seed"] == 0
    assert len(payload["fires"]) == 1
    assert payload["fires"][0]["fire_id"] == "palisades_2025"
    assert payload["mc_dropout"]["n_samples"] == 3

    sample_jsonl = run_dir / "sample_metrics_palisades_2025.jsonl"
    assert sample_jsonl.exists()
    lines = sample_jsonl.read_text().strip().splitlines()
    assert len(lines) == 8
    record = json.loads(lines[0])
    assert {"sample_idx", "fire_id", "iou", "precision", "recall", "ece",
            "growth_rate_abs_err_km2_hr", "mean_mc_variance",
            "santa_ana", "is_early_fire"} <= set(record.keys())

    predictions_npz = run_dir / "predictions_palisades_2025.npz"
    assert predictions_npz.exists()
    with np.load(predictions_npz) as data:
        assert data["mean"].dtype == np.float16
        assert data["variance"].dtype == np.float16
        assert data["target"].dtype == np.uint8
        assert data["input_mask"].dtype == np.uint8
        assert data["mean"].shape == (8, 32, 32)


def test_evaluate_run_slice_metrics_reflect_fake_dataset(tmp_path: Path, tiny_checkpoint: Path):
    """FakeFireDataset has 4 santa_ana / 4 non_santa_ana and 4 early / 4 mature.

    The runner must compute at least IoU on each slice (may be 0.0 for the
    tiny random model, but must be present in the structure).
    """
    loader = _build_loader()
    run_dir = tmp_path / "cell_B2_seed1"
    run_dir.mkdir()
    result = evaluate_run(
        run_dir=run_dir,
        checkpoint_path=tiny_checkpoint,
        loader=loader,
        cell="B2",
        seed=1,
        fire_id="thomas_2017",
        pixel_area_km2=0.140625,
        mc_samples=2,
    )
    for name in ("santa_ana", "non_santa_ana", "early", "mature"):
        assert "iou" in result.slices[name]
        assert 0.0 <= result.slices[name]["iou"] <= 1.0
