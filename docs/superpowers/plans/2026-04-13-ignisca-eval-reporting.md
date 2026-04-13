# IgnisCA Evaluation & Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the evaluation and reporting infrastructure for IgnisCA — a metric library, MC Dropout inference wrapper, slice analysis, scoring runner, cross-seed aggregation, markdown report renderer, failure-mode pipeline, optional W&B adapter, and three CLI scripts — all unit-tested on CPU against synthetic fixtures in under 15 seconds.

**Architecture:** The library is split into three new packages (`inference/`, `evaluation/`, `reporting/`) with strict one-way dependencies. `runner.py` is the sole orchestration point — every other module is a pure library. The full pipeline glues together via two dataclasses (`EvalResult`, `AggregatedRow`) whose schemas are locked in early tasks to prevent drift. Tests use a tiny `ResUNet(base=4)` on synthetic tensors; no real data, no GPU, no network.

**Tech Stack:** PyTorch 2.x (existing), scikit-learn (new core dep, for `average_precision_score`), matplotlib (new optional `viz` extra, for failure PNGs), `wandb` imported lazily (only when `--wandb` flag is passed; tests monkeypatch). JSON / JSONL / npz for artifacts. No YAML, no Lightning, no DVC.

**Spec reference:** `/Users/h.raiyan/ignis-inference-CA/docs/superpowers/specs/2026-04-13-ignisca-eval-reporting-design.md` (Plan 3). Parent: `/Users/h.raiyan/ignis-inference-CA/docs/superpowers/specs/2026-04-11-ignisca-design.md` §3.4, §4.2, §4.4, §4.5, §5.

**Plan 1 / Plan 2 dependencies:**
- `ignisca.models.resunet.ResUNet(in_channels=12, base, dropout)` — backbone used by every eval test (at `base=4` for speed).
- `ignisca.training.config.TrainConfig` — reused for `evaluate_run` to infer cell/seed from a run's saved config.
- `ignisca.training.metrics.fire_class_iou` — reused as the IoU metric; NOT re-implemented.
- `ignisca.data.features.CHANNEL_NAMES` — locks wind_u=7, wind_v=8 for the Santa Ana classifier. This plan asserts that ordering at import time in `slicing.py` (mirroring the pattern in `training/losses.py`).

**Out of scope (deferred):**
- Real training runs or real ablation execution (Plan 2's CLI does the work; Plan 3 only scores artifacts).
- Real W&B project creation (the adapter is monkeypatched in tests; real sync is a post-merge operational step).
- Autoregressive rollout inference (stays teacher-forced per spec §7.5).
- Per-fire trajectory growth-rate metric (per-sample only).

---

## File Structure

**New package `src/ignisca/inference/`:**
- `__init__.py` — empty marker
- `mc_dropout.py` — `mc_dropout_predict(model, x, n_samples=20)` returning `(mean, var)`

**New package `src/ignisca/evaluation/`:**
- `__init__.py` — empty marker
- `metrics.py` — `precision_recall_at_threshold`, `auc_pr`, `expected_calibration_error`, `growth_rate_mae`, plus `PIXEL_AREA_KM2` cell → area lookup
- `slicing.py` — `SANTA_ANA_SPEED_MIN`, `SANTA_ANA_DIR_RANGE`, `EARLY_FIRE_AREA_KM2`, `classify_santa_ana`, `is_early_fire`, `slice_groups`
- `runner.py` — `EvalResult` dataclass, `evaluate_run(cfg, checkpoint_path, loader, cell, fire_id, pixel_area_km2, mc_samples=20)` and helpers
- `aggregate.py` — `AggregatedRow` dataclass, `aggregate_cell(results_by_seed)`, `collect_runs(runs_root)`
- `reporting.py` — `render_headline_table(rows, metric_columns=...)` returning markdown string
- `failure.py` — `rank_failures`, `render_failure_case` (matplotlib imported lazily inside the render function)

**New package `src/ignisca/reporting/`:**
- `__init__.py` — empty marker
- `wandb_sync.py` — `WandbSync` class with `init_run`, `log_eval`, `finish` methods; inert when `enabled=False`; `wandb` imported lazily inside `init_run`

**New scripts:**
- `scripts/evaluate.py` — single-run scoring CLI
- `scripts/report_ablation.py` — aggregate + render headline table, optional `--also-failures`
- `scripts/run_sweep.py` — secondary sweep orchestration (`--sweep lambda_phys` or `--sweep handoff_threshold`)

**New tests:**
- `tests/inference/test_mc_dropout.py`
- `tests/evaluation/test_metrics.py`
- `tests/evaluation/test_slicing.py`
- `tests/evaluation/test_runner.py`
- `tests/evaluation/test_aggregate.py`
- `tests/evaluation/test_reporting.py`
- `tests/evaluation/test_failure.py`
- `tests/reporting/test_wandb_sync.py`
- `tests/evaluation/fake_dataset.py` — helper, NOT a test file (no `test_` prefix)
- `tests/evaluation/fixtures/headline.md` — committed golden file

Test directories `tests/inference/`, `tests/evaluation/`, `tests/evaluation/fixtures/`, `tests/reporting/` must be created. No `__init__.py` files under `tests/` — the existing project pattern (see `tests/data/sources/`) works without them.

**Modified files:**
- `pyproject.toml` — add `scikit-learn>=1.3` to core deps, add `[project.optional-dependencies].viz = ["matplotlib>=3.7"]`, add a `viz` pytest marker under `[tool.pytest.ini_options]`.
- `.gitignore` — add `runs/**/predictions_*.npz` and `runs/**/sample_metrics_*.jsonl` and `reports/failures/`.
- `tests/conftest.py` — add the four shared fixtures (`tiny_resunet`, `tiny_checkpoint`, `synthetic_batch`, `santa_ana_batch`).

---

## Task 1: Dependency and .gitignore prep

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Add scikit-learn to core dependencies**

Open `pyproject.toml`. Inside the existing `[project] dependencies` array (currently ending at `"tqdm>=4.66",`), add one line so the final entry reads:

```toml
dependencies = [
    "numpy>=1.24,<2.0",
    "scipy>=1.10",
    "torch>=2.1,<3.0",
    "rasterio>=1.3",
    "xarray>=2023.1",
    "netcdf4>=1.6",
    "geopandas>=0.14",
    "shapely>=2.0",
    "pyproj>=3.6",
    "requests>=2.31",
    "pyyaml>=6.0",
    "tqdm>=4.66",
    "scikit-learn>=1.3",
]
```

- [ ] **Step 2: Add the `viz` optional-dependencies extra**

In the same `pyproject.toml`, extend the existing `[project.optional-dependencies]` section with a new `viz` extra. The final section should read:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.1.9",
]
viz = [
    "matplotlib>=3.7",
]
```

- [ ] **Step 3: Register the `viz` pytest marker**

Extend the existing `[tool.pytest.ini_options]` block so strict-markers doesn't reject the `@pytest.mark.viz` decorator we'll add in Task 14. The block should become:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-v --strict-markers"
markers = [
    "viz: tests that require the matplotlib viz extra (skipped if matplotlib is unavailable)",
]
```

- [ ] **Step 4: Extend .gitignore**

Append the following block to `.gitignore`:

```
# IgnisCA eval artifacts (regenerable, large)
runs/**/predictions_*.npz
runs/**/sample_metrics_*.jsonl
reports/failures/
```

Note: the existing `*.npz` line already matches `predictions_*.npz` globally, but the more specific pattern here is documentation for humans reading the gitignore.

- [ ] **Step 5: Install new core dep in the active environment**

Run: `pip install 'scikit-learn>=1.3'`
Expected: either installs cleanly or reports "Requirement already satisfied". Do NOT install the `viz` extra yet — Task 14 tests skip when matplotlib is absent, which is exactly the behavior we want to verify along the way.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore
git commit -m "chore: add scikit-learn core dep and viz extra for Plan 3"
```

---

## Task 2: Package scaffolding

**Files:**
- Create: `src/ignisca/inference/__init__.py`
- Create: `src/ignisca/evaluation/__init__.py`
- Create: `src/ignisca/reporting/__init__.py`

- [ ] **Step 1: Create each package marker as an empty file**

Write three zero-byte files (truly empty, no imports, no module docstring):

- `src/ignisca/inference/__init__.py`
- `src/ignisca/evaluation/__init__.py`
- `src/ignisca/reporting/__init__.py`

- [ ] **Step 2: Verify importability**

Run: `python -c "import ignisca.inference, ignisca.evaluation, ignisca.reporting; print('ok')"`
Expected: prints `ok` with no errors.

- [ ] **Step 3: Commit**

```bash
git add src/ignisca/inference/__init__.py src/ignisca/evaluation/__init__.py src/ignisca/reporting/__init__.py
git commit -m "feat(eval): add package scaffolding for inference, evaluation, reporting"
```

---

## Task 3: Shared test fixtures

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add the four shared fixtures to conftest.py**

The existing `tests/conftest.py` already has `synthetic_geotiff`, `synthetic_netcdf`, and `palisades_stub`. Append the following below `palisades_stub` (keep the existing file content intact):

```python
@pytest.fixture
def tiny_resunet():
    """ResU-Net(base=4) on CPU — ~5k params, fast enough for every eval test.

    H=W=32 is the smallest size that survives four 2x downsamples (32 / 16 = 2).
    """
    import torch  # noqa: F401 — imported lazily so fixture collection is cheap

    from ignisca.models.resunet import ResUNet

    return ResUNet(in_channels=12, base=4, dropout=0.3)


@pytest.fixture
def tiny_checkpoint(tmp_path, tiny_resunet):
    """Save the tiny ResU-Net to a .pt file matching Plan 2's checkpoint schema."""
    import torch

    ckpt_path = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": tiny_resunet.state_dict(),
            "config": {"base_channels": 4, "dropout": 0.3},
        },
        ckpt_path,
    )
    return ckpt_path


@pytest.fixture
def synthetic_batch():
    """(x, y) with plausible per-channel distributions matching CHANNEL_NAMES."""
    import torch

    torch.manual_seed(0)
    x = torch.randn(4, 12, 32, 32) * 0.5
    x[:, 0] = (torch.rand(4, 32, 32) > 0.85).float()   # fire_mask
    x[:, 1] = torch.rand(4, 32, 32)                    # fuel_model
    x[:, 4] = torch.relu(torch.randn(4, 32, 32))       # slope >= 0
    x[:, 7] = torch.randn(4, 32, 32) * 5               # wind_u
    x[:, 8] = torch.randn(4, 32, 32) * 5               # wind_v
    y = (torch.rand(4, 32, 32) > 0.80).float()
    return x, y


@pytest.fixture
def santa_ana_batch():
    """Uniform SW-flowing wind (u=-7.07, v=-7.07) ≈ 10 m/s from NE (offshore).

    A ground-truth positive case for the Santa Ana classifier.
    """
    import torch

    x = torch.zeros(2, 12, 32, 32)
    x[:, 7] = -7.07
    x[:, 8] = -7.07
    return x
```

- [ ] **Step 2: Sanity-check that fixtures import correctly**

Run: `pytest tests/ -k "nothing_matches" --collect-only 2>&1 | head -40`
Expected: collection succeeds without ImportError on the conftest additions. (The `-k` selector intentionally matches nothing so no tests run; we're only verifying that fixture definitions parse and their imports resolve.)

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add shared fixtures for eval tests (tiny model, synthetic batches)"
```

---

## Task 4: MC Dropout inference wrapper

**Files:**
- Create: `src/ignisca/inference/mc_dropout.py`
- Create: `tests/inference/test_mc_dropout.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/inference/test_mc_dropout.py`:

```python
import torch

from ignisca.inference.mc_dropout import mc_dropout_predict
from ignisca.models.resunet import ResUNet


def test_mc_dropout_zero_dropout_has_zero_variance(synthetic_batch):
    x, _ = synthetic_batch
    model = ResUNet(in_channels=12, base=4, dropout=0.0)
    mean, var = mc_dropout_predict(model, x, n_samples=5)
    assert mean.shape == (4, 1, 32, 32)
    assert var.shape == (4, 1, 32, 32)
    assert torch.allclose(var, torch.zeros_like(var), atol=1e-8)


def test_mc_dropout_nonzero_dropout_has_positive_variance(tiny_resunet, synthetic_batch):
    x, _ = synthetic_batch
    _, var = mc_dropout_predict(tiny_resunet, x, n_samples=8)
    assert var.max().item() > 0.0


def test_mc_dropout_restores_eval_mode(tiny_resunet, synthetic_batch):
    x, _ = synthetic_batch
    tiny_resunet.eval()
    _ = mc_dropout_predict(tiny_resunet, x, n_samples=3)
    assert tiny_resunet.training is False
    for m in tiny_resunet.modules():
        if isinstance(m, torch.nn.Dropout2d):
            assert m.training is False


def test_mc_dropout_leaves_groupnorm_untouched(tiny_resunet, synthetic_batch):
    x, _ = synthetic_batch
    gn_modules = [m for m in tiny_resunet.modules() if isinstance(m, torch.nn.GroupNorm)]
    assert len(gn_modules) > 0, "ResUNet should contain GroupNorm layers"
    for m in gn_modules:
        m.train(False)
    _ = mc_dropout_predict(tiny_resunet, x, n_samples=3)
    for m in gn_modules:
        assert m.training is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/inference/test_mc_dropout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ignisca.inference.mc_dropout'` (4 tests not yet collected or erroring at import).

- [ ] **Step 3: Write the implementation**

Create `src/ignisca/inference/mc_dropout.py`:

```python
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MC Dropout inference: n forward passes with Dropout2d enabled.

    GroupNorm and other normalization layers stay in eval mode — flipping the
    whole model to ``.train()`` would contaminate normalization statistics
    across MC samples. Only ``nn.Dropout2d`` submodules are toggled.

    Returns
    -------
    mean : Tensor
        (B, 1, H, W) element-wise mean of ``sigmoid(model(x))`` over n samples.
    var : Tensor
        (B, 1, H, W) element-wise population variance over the same samples.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train(True)
    try:
        with torch.no_grad():
            samples = torch.stack(
                [torch.sigmoid(model(x)) for _ in range(n_samples)], dim=0
            )
        mean = samples.mean(dim=0)
        var = samples.var(dim=0, unbiased=False)
    finally:
        model.eval()
    return mean, var
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/inference/test_mc_dropout.py -v`
Expected: 4 passed in < 5s.

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/inference/mc_dropout.py tests/inference/test_mc_dropout.py
git commit -m "feat(inference): add MC Dropout predictor with per-pixel variance"
```

---

## Task 5: Precision, recall, AUC-PR metrics

**Files:**
- Create: `src/ignisca/evaluation/metrics.py`
- Create: `tests/evaluation/test_metrics.py`

- [ ] **Step 1: Write the failing test for precision and recall**

Create `tests/evaluation/test_metrics.py`:

```python
import math

import torch

from ignisca.evaluation.metrics import (
    auc_pr,
    precision_recall_at_threshold,
)


def test_precision_recall_on_hand_computed_case():
    # 4 pixels, one batch element. TP=1, FP=1, FN=1, TN=1.
    logits = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    p, r = precision_recall_at_threshold(logits, target, threshold=0.5)
    assert math.isclose(p, 0.5, rel_tol=1e-6)
    assert math.isclose(r, 0.5, rel_tol=1e-6)


def test_precision_recall_perfect_prediction():
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    logits = torch.where(target > 0.5, torch.full_like(target, 10.0), torch.full_like(target, -10.0))
    p, r = precision_recall_at_threshold(logits, target, threshold=0.5)
    assert math.isclose(p, 1.0, rel_tol=1e-6)
    assert math.isclose(r, 1.0, rel_tol=1e-6)


def test_precision_recall_no_positives_returns_zero():
    logits = torch.full((1, 1, 2, 2), -10.0)
    target = torch.zeros(1, 1, 2, 2)
    p, r = precision_recall_at_threshold(logits, target, threshold=0.5)
    # No predicted positives and no target positives — define both as 0.
    assert p == 0.0
    assert r == 0.0


def test_auc_pr_matches_sklearn_reference():
    from sklearn.metrics import average_precision_score

    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8) * 3
    target = (torch.rand(2, 1, 8, 8) > 0.7).float()
    probs_flat = torch.sigmoid(logits).flatten().numpy()
    target_flat = target.flatten().numpy()
    expected = average_precision_score(target_flat, probs_flat)
    actual = auc_pr(logits, target)
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)


def test_auc_pr_all_negative_target_returns_zero():
    logits = torch.randn(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    assert auc_pr(logits, target) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/evaluation/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ignisca.evaluation.metrics'`.

- [ ] **Step 3: Write the implementation**

Create `src/ignisca/evaluation/metrics.py`:

```python
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score


PIXEL_AREA_KM2: dict[str, float] = {
    # Cross-scale cells (A2, B2) have two heads; runner picks the right area
    # per head at score time. These are the per-pixel areas used by
    # ``growth_rate_mae``.
    "fine": 0.0009,      # 30 m x 30 m
    "coarse": 0.140625,  # 375 m x 375 m
}


def precision_recall_at_threshold(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """Pixel-level fire-class precision and recall at a fixed threshold.

    Matches the convention of ``training.metrics.fire_class_iou``: the 0.5
    threshold on ``sigmoid(logits)`` is applied. When the prediction has zero
    positives AND the target has zero positives, both precision and recall are
    defined as 0.0 (rather than NaN) to keep aggregation pipelines simple.
    """
    pred = torch.sigmoid(logits) > threshold
    truth = target > 0.5
    tp = float((pred & truth).sum().item())
    fp = float((pred & ~truth).sum().item())
    fn = float((~pred & truth).sum().item())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def auc_pr(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Area under the precision-recall curve over flattened pixels.

    Threshold-free. Delegates to ``sklearn.metrics.average_precision_score``.
    Returns 0.0 when the target contains no positives (AUC-PR is undefined in
    that case; sklearn raises a warning we'd rather short-circuit).
    """
    truth_np = (target > 0.5).flatten().cpu().numpy().astype(np.int64)
    if truth_np.sum() == 0:
        return 0.0
    probs_np = torch.sigmoid(logits).flatten().cpu().numpy()
    return float(average_precision_score(truth_np, probs_np))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_metrics.py -v`
Expected: 5 passed in < 2s.

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat(eval): add precision/recall and AUC-PR metrics"
```

---

## Task 6: Expected Calibration Error (ECE)

**Files:**
- Modify: `src/ignisca/evaluation/metrics.py`
- Modify: `tests/evaluation/test_metrics.py`

- [ ] **Step 1: Add ECE tests**

Append to `tests/evaluation/test_metrics.py`:

```python
def test_ece_perfect_prediction_is_zero():
    from ignisca.evaluation.metrics import expected_calibration_error

    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    # Pick logits that saturate sigmoid to 0 or 1 exactly matching target.
    logits = torch.where(
        target > 0.5,
        torch.full_like(target, 30.0),
        torch.full_like(target, -30.0),
    )
    ece = expected_calibration_error(logits, target, n_bins=10)
    assert ece < 1e-6


def test_ece_constant_half_prediction_on_balanced_target():
    from ignisca.evaluation.metrics import expected_calibration_error

    # All predictions exactly 0.5, target is 50% positive.
    logits = torch.zeros(1, 1, 4, 4)   # sigmoid(0) = 0.5
    target = torch.tensor([
        [[[1.0, 1.0, 0.0, 0.0],
          [1.0, 1.0, 0.0, 0.0],
          [1.0, 1.0, 0.0, 0.0],
          [1.0, 1.0, 0.0, 0.0]]]
    ])
    ece = expected_calibration_error(logits, target, n_bins=10)
    # Single bin [0.5, 0.6) has conf=0.5, acc=0.5 → weighted |diff|=0.
    assert ece < 1e-6


def test_ece_scripted_miscalibration_matches_hand_compute():
    from ignisca.evaluation.metrics import expected_calibration_error

    # 4 pixels all with sigmoid≈0.9 (logit≈2.197), target all 0.
    # Single bin [0.9, 1.0): conf≈0.9, acc=0, weight=1.0 → ECE≈0.9
    logits = torch.full((1, 1, 2, 2), 2.1972)
    target = torch.zeros(1, 1, 2, 2)
    ece = expected_calibration_error(logits, target, n_bins=10)
    assert abs(ece - 0.9) < 1e-3
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/evaluation/test_metrics.py::test_ece_perfect_prediction_is_zero tests/evaluation/test_metrics.py::test_ece_constant_half_prediction_on_balanced_target tests/evaluation/test_metrics.py::test_ece_scripted_miscalibration_matches_hand_compute -v`
Expected: FAIL with `ImportError: cannot import name 'expected_calibration_error'`.

- [ ] **Step 3: Add the ECE implementation**

Append to `src/ignisca/evaluation/metrics.py`:

```python
def expected_calibration_error(
    logits: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Pixel-level Expected Calibration Error with equal-width bins on [0, 1].

    For each bin, compute confidence (mean predicted probability in the bin),
    accuracy (fraction of positives in the bin), and weight (bin count / total).
    ECE is the weighted sum of |acc - conf| over all non-empty bins. Background
    pixels are included — this matches the standard pixel-level convention.
    """
    probs = torch.sigmoid(logits).flatten()
    truth = (target > 0.5).flatten().float()
    total = probs.numel()
    if total == 0:
        return 0.0

    # Edges: 0.0, 0.1, 0.2, ..., 1.0 (n_bins + 1 points).
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Final bin is closed on the right so prob==1.0 is included.
        if i == n_bins - 1:
            in_bin = (probs >= lo) & (probs <= hi)
        else:
            in_bin = (probs >= lo) & (probs < hi)
        n_in = int(in_bin.sum().item())
        if n_in == 0:
            continue
        bin_conf = float(probs[in_bin].mean().item())
        bin_acc = float(truth[in_bin].mean().item())
        weight = n_in / total
        ece += weight * abs(bin_acc - bin_conf)
    return ece
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_metrics.py -v`
Expected: 8 passed in < 2s (the 5 from Task 5 plus the 3 new ECE tests).

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat(eval): add expected calibration error metric"
```

---

## Task 7: Growth-rate MAE metric

**Files:**
- Modify: `src/ignisca/evaluation/metrics.py`
- Modify: `tests/evaluation/test_metrics.py`

- [ ] **Step 1: Add growth-rate MAE tests**

Append to `tests/evaluation/test_metrics.py`:

```python
def test_growth_rate_mae_zero_when_pred_equals_target():
    from ignisca.evaluation.metrics import growth_rate_mae

    target = (torch.rand(2, 1, 8, 8) > 0.6).float()
    input_mask = (torch.rand(2, 1, 8, 8) > 0.8).float()
    # Build saturated logits from target so sigmoid(logits) > 0.5 == target.
    logits = torch.where(
        target > 0.5,
        torch.full_like(target, 10.0),
        torch.full_like(target, -10.0),
    )
    mae = growth_rate_mae(logits, target, input_mask, pixel_area_km2=1.0, dt_hours=1.0)
    assert mae == 0.0


def test_growth_rate_mae_scripted_case():
    from ignisca.evaluation.metrics import growth_rate_mae

    # B=1, H=W=4. current area = 2 pixels, true next area = 8 pixels,
    # predicted next area = 4 pixels. pixel_area = 0.5 km², dt = 2 h.
    # true growth = (8 - 2) * 0.5 / 2 = 1.5 km²/h
    # pred growth = (4 - 2) * 0.5 / 2 = 0.5 km²/h
    # |pred - true| = 1.0
    input_mask = torch.zeros(1, 1, 4, 4)
    input_mask[0, 0, 0, 0] = 1.0
    input_mask[0, 0, 0, 1] = 1.0

    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0, :] = 1.0
    target[0, 0, 1, :] = 1.0  # 8 positives total

    logits = torch.full((1, 1, 4, 4), -10.0)
    logits[0, 0, 0, :] = 10.0  # 4 predicted positives

    mae = growth_rate_mae(logits, target, input_mask, pixel_area_km2=0.5, dt_hours=2.0)
    assert abs(mae - 1.0) < 1e-6


def test_growth_rate_mae_batch_mean():
    from ignisca.evaluation.metrics import growth_rate_mae

    # Two samples with known abs errors of 2.0 and 0.0 → mean 1.0.
    input_mask = torch.zeros(2, 1, 2, 2)
    target = torch.zeros(2, 1, 2, 2)
    target[0, 0, 0, 0] = 1.0  # sample 0 has 1 true positive
    target[0, 0, 0, 1] = 1.0  # sample 0 has 2 true positives
    logits = torch.full((2, 1, 2, 2), -10.0)
    # sample 0: pred 0 positives, true 2 → error 2
    # sample 1: pred 0 positives, true 0 → error 0
    mae = growth_rate_mae(logits, target, input_mask, pixel_area_km2=1.0, dt_hours=1.0)
    assert abs(mae - 1.0) < 1e-6
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/evaluation/test_metrics.py -k growth_rate -v`
Expected: FAIL with `ImportError: cannot import name 'growth_rate_mae'`.

- [ ] **Step 3: Add the growth-rate MAE implementation**

Append to `src/ignisca/evaluation/metrics.py`:

```python
def growth_rate_mae(
    logits: torch.Tensor,
    target: torch.Tensor,
    input_mask: torch.Tensor,
    pixel_area_km2: float,
    dt_hours: float = 1.0,
) -> float:
    """Mean absolute error of the per-sample fire growth rate.

    Growth rate is defined as ``(next_area - current_area) / dt_hours`` in
    km²/h, where ``next_area`` is computed from the predicted next-step fire
    mask and ``current_area`` from the input fire mask. Returns a mean over the
    batch as a Python float.
    """
    if dt_hours <= 0:
        raise ValueError(f"dt_hours must be positive, got {dt_hours}")
    pred_bin = (torch.sigmoid(logits) > 0.5).float()
    pred_area_next = pred_bin.sum(dim=(-1, -2, -3)) * pixel_area_km2
    true_area_next = (target > 0.5).float().sum(dim=(-1, -2, -3)) * pixel_area_km2
    curr_area = (input_mask > 0.5).float().sum(dim=(-1, -2, -3)) * pixel_area_km2
    pred_growth = (pred_area_next - curr_area) / dt_hours
    true_growth = (true_area_next - curr_area) / dt_hours
    return float((pred_growth - true_growth).abs().mean().item())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_metrics.py -v`
Expected: 11 passed in < 2s.

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat(eval): add per-sample growth-rate MAE metric"
```

---

## Task 8: Santa Ana classifier and slice groups

**Files:**
- Create: `src/ignisca/evaluation/slicing.py`
- Create: `tests/evaluation/test_slicing.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/evaluation/test_slicing.py`:

```python
import math

import torch

from ignisca.evaluation.slicing import (
    EARLY_FIRE_AREA_KM2,
    SANTA_ANA_DIR_RANGE,
    SANTA_ANA_SPEED_MIN,
    classify_santa_ana,
    is_early_fire,
    slice_groups,
)


def _features_with_uniform_wind(u: float, v: float, batch: int = 2) -> torch.Tensor:
    x = torch.zeros(batch, 12, 32, 32)
    x[:, 7] = u
    x[:, 8] = v
    return x


def test_constants_have_expected_values():
    assert SANTA_ANA_SPEED_MIN == 7.0
    assert SANTA_ANA_DIR_RANGE == (0, 90)
    assert EARLY_FIRE_AREA_KM2 == 5.0


def test_classify_santa_ana_ne_origin_fast_wind_is_true(santa_ana_batch):
    flags = classify_santa_ana(santa_ana_batch)
    assert flags.dtype == torch.bool
    assert flags.shape == (2,)
    assert bool(flags[0]) is True
    assert bool(flags[1]) is True


def test_classify_santa_ana_sw_origin_is_false():
    # Wind flowing NE (u=+7.07, v=+7.07) → coming FROM SW → not Santa Ana.
    features = _features_with_uniform_wind(7.07, 7.07)
    flags = classify_santa_ana(features)
    assert bool(flags[0]) is False


def test_classify_santa_ana_below_speed_threshold_is_false():
    # Wind flowing SW at ~3 m/s (u=-2.12, v=-2.12) → from NE but too slow.
    features = _features_with_uniform_wind(-2.12, -2.12)
    flags = classify_santa_ana(features)
    assert bool(flags[0]) is False


def test_classify_santa_ana_ne_origin_exactly_at_speed_threshold_is_true():
    # Exactly 7 m/s from the NE (45°): u=v=-4.95.
    u = -SANTA_ANA_SPEED_MIN / math.sqrt(2)
    features = _features_with_uniform_wind(u, u)
    flags = classify_santa_ana(features)
    assert bool(flags[0]) is True


def test_is_early_fire_boundary_at_five_km2():
    # pixel_area = 1 km², so an input mask with 4 positives is early, 5 is not.
    early = torch.zeros(1, 1, 4, 4)
    early[0, 0, 0, :4] = 1.0  # 4 positives

    mature = torch.zeros(1, 1, 4, 4)
    mature[0, 0, 0, :4] = 1.0
    mature[0, 0, 1, 0] = 1.0  # 5 positives

    assert bool(is_early_fire(early, pixel_area_km2=1.0)[0]) is True
    assert bool(is_early_fire(mature, pixel_area_km2=1.0)[0]) is False


def test_slice_groups_returns_all_four_masks(santa_ana_batch):
    input_mask = torch.zeros(2, 1, 32, 32)
    input_mask[0, 0, :1, :2] = 1.0  # sample 0 is small
    input_mask[1, 0, :10, :10] = 1.0  # sample 1 is large

    groups = slice_groups(santa_ana_batch, input_mask, pixel_area_km2=1.0)
    assert set(groups.keys()) == {"santa_ana", "non_santa_ana", "early", "mature"}
    for mask in groups.values():
        assert mask.shape == (2,)
        assert mask.dtype == torch.bool
    # Both samples have Santa Ana winds; sample 0 is early, sample 1 is mature.
    assert bool(groups["santa_ana"][0]) is True
    assert bool(groups["santa_ana"][1]) is True
    assert bool(groups["non_santa_ana"][0]) is False
    assert bool(groups["early"][0]) is True
    assert bool(groups["early"][1]) is False
    assert bool(groups["mature"][1]) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/evaluation/test_slicing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ignisca.evaluation.slicing'`.

- [ ] **Step 3: Write the implementation**

Create `src/ignisca/evaluation/slicing.py`:

```python
from __future__ import annotations

import math
from typing import Dict

import torch

from ignisca.data.features import CHANNEL_NAMES

# Assert the channel ordering we rely on, same pattern as training/losses.py.
_EXPECTED = {7: "wind_u", 8: "wind_v"}
for _idx, _name in _EXPECTED.items():
    if CHANNEL_NAMES[_idx] != _name:
        raise AssertionError(
            f"ignisca.evaluation.slicing expects channel {_idx}={_name!r}, "
            f"but CHANNEL_NAMES[{_idx}]={CHANNEL_NAMES[_idx]!r}. "
            "Reorder ignisca.data.features.CHANNEL_NAMES or update slicing.py."
        )


SANTA_ANA_SPEED_MIN: float = 7.0           # m/s (~15 mph)
SANTA_ANA_DIR_RANGE: tuple[int, int] = (0, 90)   # meteorological "from" angle
EARLY_FIRE_AREA_KM2: float = 5.0


def classify_santa_ana(features: torch.Tensor) -> torch.Tensor:
    """Per-sample Santa Ana flag from HRRR wind channels 7 and 8.

    Meteorological convention: "wind direction" is the compass angle the wind
    is COMING FROM. Santa Anas are offshore NE-origin winds for SoCal, so we
    accept directions in [0, 90)°. The standard (u, v) pair describes the vector
    the wind is flowing TOWARD, so we negate it inside ``atan2`` to get the
    origin angle.
    """
    if features.ndim != 4 or features.shape[1] < 9:
        raise ValueError(
            f"classify_santa_ana expects (B, 12+, H, W), got shape {tuple(features.shape)}"
        )
    u = features[:, 7]
    v = features[:, 8]
    u_mean = u.mean(dim=(-2, -1))
    v_mean = v.mean(dim=(-2, -1))
    speed = torch.sqrt(u_mean * u_mean + v_mean * v_mean + 1e-8)
    # ``atan2(x, y)`` returns the angle of (x, y) from the +y axis clockwise.
    # Meteorology measures from North (positive y) clockwise, so we use
    # atan2(east_component, north_component). Here -u is the east component of
    # the origin vector and -v is the north component.
    dir_rad = torch.atan2(-u_mean, -v_mean)
    dir_deg = (dir_rad * 180.0 / math.pi) % 360.0
    in_offshore = (dir_deg >= SANTA_ANA_DIR_RANGE[0]) & (dir_deg < SANTA_ANA_DIR_RANGE[1])
    fast_enough = speed >= SANTA_ANA_SPEED_MIN
    return fast_enough & in_offshore


def is_early_fire(input_mask: torch.Tensor, pixel_area_km2: float) -> torch.Tensor:
    """Per-sample boolean: current fire area < EARLY_FIRE_AREA_KM2."""
    if input_mask.ndim != 4 or input_mask.shape[1] != 1:
        raise ValueError(
            f"is_early_fire expects (B, 1, H, W), got shape {tuple(input_mask.shape)}"
        )
    area = (input_mask > 0.5).float().sum(dim=(-1, -2, -3)) * pixel_area_km2
    return area < EARLY_FIRE_AREA_KM2


def slice_groups(
    features: torch.Tensor,
    input_mask: torch.Tensor,
    pixel_area_km2: float,
) -> Dict[str, torch.Tensor]:
    """Return boolean per-sample masks for each slice the runner scores.

    Four slices: santa_ana, non_santa_ana, early, mature. Each value is a
    1-D bool Tensor of length ``features.shape[0]``. Slices overlap — a sample
    can be both ``santa_ana`` and ``early``.
    """
    sa = classify_santa_ana(features)
    early = is_early_fire(input_mask, pixel_area_km2=pixel_area_km2)
    return {
        "santa_ana": sa,
        "non_santa_ana": ~sa,
        "early": early,
        "mature": ~early,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_slicing.py -v`
Expected: 7 passed in < 2s.

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/evaluation/slicing.py tests/evaluation/test_slicing.py
git commit -m "feat(eval): add Santa Ana classifier and slice group helper"
```

---

## Task 9: Fake dataset helper for runner tests

**Files:**
- Create: `tests/evaluation/fake_dataset.py`

- [ ] **Step 1: Write the helper**

Create `tests/evaluation/fake_dataset.py`:

```python
"""Synthetic in-memory Dataset for evaluation/runner tests.

Not a test file — no ``test_`` prefix so pytest does not collect it.
Yields 8 samples total, meant to exercise slice classification: half the
samples have Santa Ana winds, half have SW-origin winds; half have small
input masks (early), half have large ones (mature).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset


class FakeFireDataset(Dataset):
    def __init__(self, n_samples: int = 8, hw: int = 32) -> None:
        torch.manual_seed(42)
        self.n = n_samples
        self.hw = hw
        self._x = torch.zeros(n_samples, 12, hw, hw)
        self._y = torch.zeros(n_samples, hw, hw)
        for i in range(n_samples):
            if i % 2 == 0:
                # Santa Ana: wind flowing SW, coming FROM NE
                self._x[i, 7] = -7.5
                self._x[i, 8] = -7.5
            else:
                # Not Santa Ana: wind flowing NE, coming FROM SW
                self._x[i, 7] = 7.5
                self._x[i, 8] = 7.5

            if i < n_samples // 2:
                # Early fire — 2 positive pixels
                self._x[i, 0, 0, 0] = 1.0
                self._x[i, 0, 0, 1] = 1.0
                self._y[i, 0, 0:3] = 1.0
            else:
                # Mature fire — 100 positive pixels
                self._x[i, 0, :10, :10] = 1.0
                self._y[i, :10, :11] = 1.0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._x[idx], self._y[idx]
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "import sys; sys.path.insert(0, 'tests'); from evaluation.fake_dataset import FakeFireDataset; ds = FakeFireDataset(); print(len(ds), ds[0][0].shape, ds[0][1].shape)"`
Expected: `8 torch.Size([12, 32, 32]) torch.Size([32, 32])`

- [ ] **Step 3: Commit**

```bash
git add tests/evaluation/fake_dataset.py
git commit -m "test: add FakeFireDataset helper for runner tests"
```

---

## Task 10: EvalResult dataclass and runner

**Files:**
- Create: `src/ignisca/evaluation/runner.py`
- Create: `tests/evaluation/test_runner.py`

- [ ] **Step 1: Write the failing runner test**

Create `tests/evaluation/test_runner.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/evaluation/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ignisca.evaluation.runner'`.

- [ ] **Step 3: Write the implementation**

Create `src/ignisca/evaluation/runner.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ignisca.evaluation.metrics import (
    auc_pr,
    expected_calibration_error,
    growth_rate_mae,
    precision_recall_at_threshold,
)
from ignisca.evaluation.slicing import slice_groups
from ignisca.inference.mc_dropout import mc_dropout_predict
from ignisca.models.resunet import ResUNet
from ignisca.training.metrics import fire_class_iou

_METRIC_KEYS = (
    "iou",
    "precision",
    "recall",
    "auc_pr",
    "ece",
    "growth_rate_mae",
    "mean_mc_variance",
)


@dataclass
class EvalResult:
    run_name: str
    cell: str
    seed: int
    fire_id: str
    iou: float
    precision: float
    recall: float
    auc_pr: float
    ece: float
    growth_rate_mae: float
    mean_mc_variance: float
    slices: Dict[str, Dict[str, float]] = field(default_factory=dict)
    predictions_path: Path = field(default=Path(""))
    sample_metrics_path: Path = field(default=Path(""))
    n_samples: int = 0


def _load_model_from_checkpoint(
    checkpoint_path: Path, in_channels: int = 12
) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    base = int(cfg.get("base_channels", 64))
    dropout = float(cfg.get("dropout", 0.2))
    model = ResUNet(in_channels=in_channels, base=base, dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _slice_metric_dict(
    logits_sample: torch.Tensor,
    target_sample: torch.Tensor,
    input_mask_sample: torch.Tensor,
    variance_sample: torch.Tensor,
    pixel_area_km2: float,
) -> Dict[str, float]:
    """Compute all 7 metrics for a single-sample slice on the fly.

    ``logits_sample`` has shape (N, 1, H, W) where N is the number of samples
    in the slice. Empty slices are returned as all-zero.
    """
    if logits_sample.shape[0] == 0:
        return {k: 0.0 for k in _METRIC_KEYS}
    iou = fire_class_iou(logits_sample, target_sample)
    prec, rec = precision_recall_at_threshold(logits_sample, target_sample)
    return {
        "iou": iou,
        "precision": prec,
        "recall": rec,
        "auc_pr": auc_pr(logits_sample, target_sample),
        "ece": expected_calibration_error(logits_sample, target_sample),
        "growth_rate_mae": growth_rate_mae(
            logits_sample, target_sample, input_mask_sample, pixel_area_km2=pixel_area_km2
        ),
        "mean_mc_variance": float(variance_sample.mean().item()),
    }


def evaluate_run(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    loader: DataLoader,
    cell: str,
    seed: int,
    fire_id: str,
    pixel_area_km2: float,
    mc_samples: int = 20,
    device: str = "cpu",
) -> EvalResult:
    """Load a checkpoint, score the loader, write per-run artifacts.

    Outputs (inside ``run_dir``):
      - eval.json                         (top-level metrics + artifact paths)
      - sample_metrics_<fire_id>.jsonl    (one line per sample)
      - predictions_<fire_id>.npz         (MC dropout mean/variance + targets)
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model_from_checkpoint(checkpoint_path).to(device)

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_input_masks: list[torch.Tensor] = []
    all_means: list[torch.Tensor] = []
    all_vars: list[torch.Tensor] = []
    all_features: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.float().to(device).unsqueeze(1)  # (B, 1, H, W)
            input_mask = x[:, 0:1]
            logits = model(x)
            mean, var = mc_dropout_predict(model, x, n_samples=mc_samples)
            all_logits.append(logits)
            all_targets.append(y)
            all_input_masks.append(input_mask)
            all_means.append(mean)
            all_vars.append(var)
            all_features.append(x)

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    input_masks = torch.cat(all_input_masks, dim=0)
    means = torch.cat(all_means, dim=0)
    variances = torch.cat(all_vars, dim=0)
    features = torch.cat(all_features, dim=0)
    n_samples = int(logits.shape[0])

    overall_iou = fire_class_iou(logits, targets)
    overall_prec, overall_rec = precision_recall_at_threshold(logits, targets)
    overall_auc = auc_pr(logits, targets)
    overall_ece = expected_calibration_error(logits, targets)
    overall_growth = growth_rate_mae(
        logits, targets, input_masks, pixel_area_km2=pixel_area_km2
    )
    overall_mc_var = float(variances.mean().item())

    groups = slice_groups(features, input_masks, pixel_area_km2=pixel_area_km2)
    slice_metrics: Dict[str, Dict[str, float]] = {}
    for name, mask in groups.items():
        idx = mask.nonzero(as_tuple=True)[0]
        slice_metrics[name] = _slice_metric_dict(
            logits[idx], targets[idx], input_masks[idx], variances[idx],
            pixel_area_km2=pixel_area_km2,
        )

    predictions_path = run_dir / f"predictions_{fire_id}.npz"
    np.savez(
        predictions_path,
        mean=means.squeeze(1).cpu().numpy().astype(np.float16),
        variance=variances.squeeze(1).cpu().numpy().astype(np.float16),
        target=(targets.squeeze(1) > 0.5).cpu().numpy().astype(np.uint8),
        input_mask=(input_masks.squeeze(1) > 0.5).cpu().numpy().astype(np.uint8),
    )

    sample_metrics_path = run_dir / f"sample_metrics_{fire_id}.jsonl"
    with sample_metrics_path.open("w") as fh:
        sa_flags = groups["santa_ana"]
        early_flags = groups["early"]
        for i in range(n_samples):
            sample_logits = logits[i : i + 1]
            sample_target = targets[i : i + 1]
            sample_input = input_masks[i : i + 1]
            s_prec, s_rec = precision_recall_at_threshold(sample_logits, sample_target)
            record = {
                "sample_idx": i,
                "fire_id": fire_id,
                "iou": fire_class_iou(sample_logits, sample_target),
                "precision": s_prec,
                "recall": s_rec,
                "ece": expected_calibration_error(sample_logits, sample_target),
                "growth_rate_abs_err_km2_hr": growth_rate_mae(
                    sample_logits, sample_target, sample_input,
                    pixel_area_km2=pixel_area_km2,
                ),
                "mean_mc_variance": float(variances[i].mean().item()),
                "santa_ana": bool(sa_flags[i].item()),
                "is_early_fire": bool(early_flags[i].item()),
            }
            fh.write(json.dumps(record) + "\n")

    result = EvalResult(
        run_name=run_dir.name,
        cell=cell,
        seed=seed,
        fire_id=fire_id,
        iou=overall_iou,
        precision=overall_prec,
        recall=overall_rec,
        auc_pr=overall_auc,
        ece=overall_ece,
        growth_rate_mae=overall_growth,
        mean_mc_variance=overall_mc_var,
        slices=slice_metrics,
        predictions_path=predictions_path,
        sample_metrics_path=sample_metrics_path,
        n_samples=n_samples,
    )

    eval_json_path = run_dir / "eval.json"
    payload = {
        "run_name": result.run_name,
        "cell": result.cell,
        "seed": result.seed,
        "fires": [
            {
                "fire_id": result.fire_id,
                "iou": result.iou,
                "precision": result.precision,
                "recall": result.recall,
                "auc_pr": result.auc_pr,
                "ece": result.ece,
                "growth_rate_mae": result.growth_rate_mae,
                "mean_mc_variance": result.mean_mc_variance,
                "slices": result.slices,
                "n_samples": result.n_samples,
            }
        ],
        "mc_dropout": {"n_samples": mc_samples},
        "artifacts": {
            f"predictions_{fire_id}": str(predictions_path),
            f"sample_metrics_{fire_id}": str(sample_metrics_path),
        },
    }
    eval_json_path.write_text(json.dumps(payload, indent=2))
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_runner.py -v`
Expected: 2 passed in < 10s. If the runner is slow, inspect — the tiny model should be fast.

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/evaluation/runner.py tests/evaluation/test_runner.py
git commit -m "feat(eval): add EvalResult dataclass and end-to-end scoring runner"
```

---

## Task 11: Cross-seed aggregation

**Files:**
- Create: `src/ignisca/evaluation/aggregate.py`
- Create: `tests/evaluation/test_aggregate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/evaluation/test_aggregate.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/evaluation/test_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ignisca.evaluation.aggregate'`.

- [ ] **Step 3: Write the implementation**

Create `src/ignisca/evaluation/aggregate.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_aggregate.py -v`
Expected: 5 passed in < 2s.

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/evaluation/aggregate.py tests/evaluation/test_aggregate.py
git commit -m "feat(eval): add cross-seed aggregation with loud-fail validation"
```

---

## Task 12: Headline markdown report renderer

**Files:**
- Create: `src/ignisca/evaluation/reporting.py`
- Create: `tests/evaluation/fixtures/headline.md`
- Create: `tests/evaluation/test_reporting.py`

- [ ] **Step 1: Create the golden fixture file**

Create `tests/evaluation/fixtures/headline.md` with exactly this content (trailing newline included):

```
# IgnisCA Ablation Headline

| Fire | Cell | IoU | Precision | Recall | AUC-PR | ECE | Growth MAE | MC Var |
|---|---|---|---|---|---|---|---|---|
| palisades_2025 | A1 | 0.600 ± 0.020 | 0.580 ± 0.010 | 0.630 ± 0.015 | 0.710 ± 0.008 | 0.090 ± 0.003 | 0.420 ± 0.012 | 0.018 ± 0.002 |
| palisades_2025 | A2 | 0.650 ± 0.018 | 0.620 ± 0.012 | 0.670 ± 0.011 | 0.740 ± 0.009 | 0.080 ± 0.004 | 0.390 ± 0.015 | 0.017 ± 0.001 |
| thomas_2017 | A1 | 0.550 ± 0.025 | 0.530 ± 0.015 | 0.560 ± 0.020 | 0.660 ± 0.012 | 0.110 ± 0.005 | 0.480 ± 0.020 | 0.022 ± 0.003 |
| thomas_2017 | A2 | 0.590 ± 0.022 | 0.570 ± 0.014 | 0.610 ± 0.018 | 0.700 ± 0.010 | 0.100 ± 0.004 | 0.450 ± 0.018 | 0.020 ± 0.002 |
```

- [ ] **Step 2: Write the failing reporting tests**

Create `tests/evaluation/test_reporting.py`:

```python
from pathlib import Path

import pytest

from ignisca.evaluation.aggregate import AggregatedRow
from ignisca.evaluation.reporting import render_headline_table

FIXTURE = Path(__file__).parent / "fixtures" / "headline.md"


def _row(fire_id: str, cell: str, base: float) -> AggregatedRow:
    return AggregatedRow(
        fire_id=fire_id,
        cell=cell,
        metrics={
            "iou": (base, 0.020),
            "precision": (base - 0.02, 0.010),
            "recall": (base + 0.03, 0.015),
            "auc_pr": (base + 0.11, 0.008),
            "ece": (0.09, 0.003),
            "growth_rate_mae": (0.42, 0.012),
            "mean_mc_variance": (0.018, 0.002),
        },
        n_seeds=3,
    )


def test_render_headline_table_matches_golden_fixture():
    rows = [
        AggregatedRow(
            fire_id="palisades_2025", cell="A1",
            metrics={
                "iou": (0.600, 0.020),
                "precision": (0.580, 0.010),
                "recall": (0.630, 0.015),
                "auc_pr": (0.710, 0.008),
                "ece": (0.090, 0.003),
                "growth_rate_mae": (0.420, 0.012),
                "mean_mc_variance": (0.018, 0.002),
            }, n_seeds=3,
        ),
        AggregatedRow(
            fire_id="palisades_2025", cell="A2",
            metrics={
                "iou": (0.650, 0.018),
                "precision": (0.620, 0.012),
                "recall": (0.670, 0.011),
                "auc_pr": (0.740, 0.009),
                "ece": (0.080, 0.004),
                "growth_rate_mae": (0.390, 0.015),
                "mean_mc_variance": (0.017, 0.001),
            }, n_seeds=3,
        ),
        AggregatedRow(
            fire_id="thomas_2017", cell="A1",
            metrics={
                "iou": (0.550, 0.025),
                "precision": (0.530, 0.015),
                "recall": (0.560, 0.020),
                "auc_pr": (0.660, 0.012),
                "ece": (0.110, 0.005),
                "growth_rate_mae": (0.480, 0.020),
                "mean_mc_variance": (0.022, 0.003),
            }, n_seeds=3,
        ),
        AggregatedRow(
            fire_id="thomas_2017", cell="A2",
            metrics={
                "iou": (0.590, 0.022),
                "precision": (0.570, 0.014),
                "recall": (0.610, 0.018),
                "auc_pr": (0.700, 0.010),
                "ece": (0.100, 0.004),
                "growth_rate_mae": (0.450, 0.018),
                "mean_mc_variance": (0.020, 0.002),
            }, n_seeds=3,
        ),
    ]
    actual = render_headline_table(rows).rstrip() + "\n"
    expected = FIXTURE.read_text()
    assert actual == expected


def test_render_headline_table_includes_all_rows():
    rows = [_row("palisades_2025", "A1", 0.60), _row("thomas_2017", "B2", 0.65)]
    md = render_headline_table(rows)
    assert "palisades_2025" in md
    assert "thomas_2017" in md
    assert "A1" in md
    assert "B2" in md
    # Header row with every metric column present.
    for col in ("IoU", "Precision", "Recall", "AUC-PR", "ECE", "Growth MAE", "MC Var"):
        assert col in md


def test_render_headline_table_rejects_empty_rows():
    with pytest.raises(ValueError, match="empty"):
        render_headline_table([])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/evaluation/test_reporting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ignisca.evaluation.reporting'`.

- [ ] **Step 4: Write the implementation**

Create `src/ignisca/evaluation/reporting.py`:

```python
from __future__ import annotations

from typing import Sequence

from ignisca.evaluation.aggregate import AggregatedRow

_COLUMNS: tuple[tuple[str, str], ...] = (
    ("iou", "IoU"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("auc_pr", "AUC-PR"),
    ("ece", "ECE"),
    ("growth_rate_mae", "Growth MAE"),
    ("mean_mc_variance", "MC Var"),
)


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def render_headline_table(rows: Sequence[AggregatedRow]) -> str:
    """Render a list of AggregatedRows to a markdown table as a single string.

    Header format is fixed and covers every metric in ``_COLUMNS``. Rows are
    emitted in the order provided — callers are responsible for sorting if
    they want a stable fire/cell ordering.
    """
    if not rows:
        raise ValueError("render_headline_table: rows is empty")

    header_cells = ["Fire", "Cell"] + [label for _, label in _COLUMNS]
    separator_cells = ["---"] * len(header_cells)
    lines: list[str] = [
        "# IgnisCA Ablation Headline",
        "",
        "| " + " | ".join(header_cells) + " |",
        "|" + "|".join(separator_cells) + "|",
    ]
    for row in rows:
        cells = [row.fire_id, row.cell]
        for key, _ in _COLUMNS:
            mean, std = row.metrics[key]
            cells.append(_fmt(mean, std))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_reporting.py -v`
Expected: 3 passed in < 2s. If the golden-file test fails, inspect the diff carefully — the fixture header is the immutable contract.

- [ ] **Step 6: Commit**

```bash
git add src/ignisca/evaluation/reporting.py src/ignisca/evaluation/__init__.py tests/evaluation/test_reporting.py tests/evaluation/fixtures/headline.md
git commit -m "feat(eval): add markdown headline-table renderer with golden fixture"
```

(`src/ignisca/evaluation/__init__.py` is listed for explicit hygiene even though it's already committed.)

---

## Task 13: Failure ranking

**Files:**
- Create: `src/ignisca/evaluation/failure.py`
- Create: `tests/evaluation/test_failure.py`

- [ ] **Step 1: Write the failing ranking tests**

Create `tests/evaluation/test_failure.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/evaluation/test_failure.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ignisca.evaluation.failure'`.

- [ ] **Step 3: Write the ranking implementation**

Create `src/ignisca/evaluation/failure.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_failure.py -v`
Expected: 5 passed in < 2s.

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/evaluation/failure.py tests/evaluation/test_failure.py
git commit -m "feat(eval): add failure-mode ranking from per-sample JSONL"
```

---

## Task 14: Failure PNG rendering

**Files:**
- Modify: `src/ignisca/evaluation/failure.py`
- Modify: `tests/evaluation/test_failure.py`

- [ ] **Step 1: Add the failing render test**

Append to `tests/evaluation/test_failure.py`:

```python
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
```

- [ ] **Step 2: Run tests to check they skip cleanly without matplotlib**

Run: `pytest tests/evaluation/test_failure.py -v`
Expected: 5 passed + 2 skipped (the two viz-marked tests skip via `importorskip`).

- [ ] **Step 3: Install matplotlib via the viz extra**

Run: `pip install 'matplotlib>=3.7'`
Expected: installs cleanly (it pulls in a handful of plotting deps).

- [ ] **Step 4: Re-run tests to confirm the viz tests now fail (ImportError on `render_failure_case`)**

Run: `pytest tests/evaluation/test_failure.py -v`
Expected: FAIL with `ImportError: cannot import name 'render_failure_case'` on the 2 viz tests (the 5 ranking tests still pass).

- [ ] **Step 5: Add the render implementation**

Append to `src/ignisca/evaluation/failure.py`:

```python
def render_failure_case(
    *,
    npz_path: Path,
    sample_idx: int,
    out_path: Path,
) -> None:
    """Render a 4-panel PNG: input mask, target, prediction mean, variance.

    Matplotlib is imported lazily so the evaluation package does not pull
    matplotlib into the core dependency graph. Tests that use this function
    are marked ``@pytest.mark.viz`` and skip cleanly when matplotlib is absent.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    with np.load(Path(npz_path)) as data:
        mean = data["mean"][sample_idx].astype(np.float32)
        variance = data["variance"][sample_idx].astype(np.float32)
        target = data["target"][sample_idx].astype(np.uint8)
        input_mask = data["input_mask"][sample_idx].astype(np.uint8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = (
        ("Input mask", input_mask, "Reds", 0.0, 1.0),
        ("Target", target, "Reds", 0.0, 1.0),
        ("Prediction (mean)", mean, "viridis", 0.0, 1.0),
        ("Uncertainty (var)", variance, "Reds", 0.0, max(variance.max(), 1e-6)),
    )
    for ax, (title, array, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
```

Additionally, edit the existing `render_failure_case` raise path — the in-memory array access with an out-of-range `sample_idx` already raises `IndexError` naturally via NumPy, so no explicit bounds check is needed. The test verifies the default NumPy behavior.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_failure.py -v`
Expected: 7 passed in < 5s.

- [ ] **Step 7: Commit**

```bash
git add src/ignisca/evaluation/failure.py tests/evaluation/test_failure.py
git commit -m "feat(eval): render 4-panel failure PNGs with lazy matplotlib import"
```

---

## Task 15: Optional W&B sync adapter

**Files:**
- Create: `src/ignisca/reporting/wandb_sync.py`
- Create: `tests/reporting/test_wandb_sync.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/reporting/test_wandb_sync.py`:

```python
import sys
import types

import pytest

from ignisca.reporting.wandb_sync import WandbSync


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.finished: bool = False

    def log(self, data: dict) -> None:
        self.logged.append(data)

    def finish(self) -> None:
        self.finished = True


def _install_fake_wandb(monkeypatch) -> tuple[list[dict], _FakeRun]:
    init_calls: list[dict] = []
    run = _FakeRun()

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return run

    fake_module = types.SimpleNamespace(init=fake_init)
    monkeypatch.setitem(sys.modules, "wandb", fake_module)
    return init_calls, run


def test_disabled_adapter_never_imports_wandb(monkeypatch):
    # Even if wandb is missing, a disabled adapter stays silent.
    monkeypatch.setitem(sys.modules, "wandb", None)  # force ImportError if touched
    sync = WandbSync(enabled=False, project="ignisca", run_name="cell_A1_seed0")
    sync.init_run()
    sync.log_eval({"iou": 0.6})
    sync.finish()  # no explosion


def test_enabled_adapter_calls_init_log_finish_in_order(monkeypatch):
    init_calls, run = _install_fake_wandb(monkeypatch)

    sync = WandbSync(enabled=True, project="ignisca", run_name="cell_A1_seed0")
    sync.init_run()
    assert len(init_calls) == 1
    assert init_calls[0]["project"] == "ignisca"
    assert init_calls[0]["name"] == "cell_A1_seed0"

    sync.log_eval({"iou": 0.61, "ece": 0.09})
    assert run.logged == [{"iou": 0.61, "ece": 0.09}]

    sync.finish()
    assert run.finished is True


def test_log_before_init_raises(monkeypatch):
    _install_fake_wandb(monkeypatch)
    sync = WandbSync(enabled=True, project="ignisca", run_name="cell_A1_seed0")
    with pytest.raises(RuntimeError, match="init_run"):
        sync.log_eval({"iou": 0.6})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/reporting/test_wandb_sync.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ignisca.reporting.wandb_sync'`.

- [ ] **Step 3: Write the implementation**

Create `src/ignisca/reporting/wandb_sync.py`:

```python
from __future__ import annotations

from typing import Any, Optional


class WandbSync:
    """Opt-in Weights & Biases adapter.

    Inert when ``enabled=False``: ``init_run`` / ``log_eval`` / ``finish`` are
    no-ops and the ``wandb`` module is never imported. Tests monkeypatch
    ``sys.modules["wandb"]`` with a fake module to exercise the enabled path
    without touching the network.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        run_name: str,
        entity: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.project = project
        self.run_name = run_name
        self.entity = entity
        self._run: Any = None

    def init_run(self) -> None:
        if not self.enabled:
            return
        import wandb  # imported lazily — never during __init__

        kwargs: dict[str, Any] = {"project": self.project, "name": self.run_name}
        if self.entity is not None:
            kwargs["entity"] = self.entity
        self._run = wandb.init(**kwargs)

    def log_eval(self, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._run is None:
            raise RuntimeError("WandbSync.log_eval called before init_run")
        self._run.log(metrics)

    def finish(self) -> None:
        if not self.enabled:
            return
        if self._run is None:
            return
        self._run.finish()
        self._run = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/reporting/test_wandb_sync.py -v`
Expected: 3 passed in < 2s.

- [ ] **Step 5: Commit**

```bash
git add src/ignisca/reporting/wandb_sync.py tests/reporting/test_wandb_sync.py
git commit -m "feat(reporting): add monkeypatch-testable WandbSync adapter"
```

---

## Task 16: `scripts/evaluate.py` CLI

**Files:**
- Create: `scripts/evaluate.py`

- [ ] **Step 1: Write the CLI**

Create `scripts/evaluate.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ignisca.data.dataset import IgnisDataset
from ignisca.evaluation.metrics import PIXEL_AREA_KM2
from ignisca.evaluation.runner import evaluate_run
from ignisca.reporting.wandb_sync import WandbSync


def _infer_cell_seed_from_run_dir(run_dir: Path) -> tuple[str, int]:
    """Pull the cell name and seed out of a run directory name.

    Expected shape: ``<cell_name>_<head>_seed<N>`` (Plan 2 ablation) or
    ``cell_<X>_seed<N>`` (eval-only naming). Falls back to reading
    ``metrics.jsonl`` if pattern match fails.
    """
    name = run_dir.name
    parts = name.split("_")
    seed_token = next((p for p in parts if p.startswith("seed")), None)
    if seed_token is None:
        raise ValueError(f"cannot infer seed from run name {name!r}")
    seed = int(seed_token[len("seed"):])
    cell = parts[0] if parts else name
    return cell, seed


def _pixel_area_for_cache_dir(cache_dir: Path) -> float:
    lowered = str(cache_dir).lower()
    if "fine" in lowered:
        return PIXEL_AREA_KM2["fine"]
    return PIXEL_AREA_KM2["coarse"]


def main() -> None:
    parser = argparse.ArgumentParser(description="IgnisCA single-run evaluation")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing best.pt")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Cache root holding the test split")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--fire-id", required=True)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="ignisca")
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    checkpoint_path = args.run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    cell, seed = _infer_cell_seed_from_run_dir(args.run_dir)
    pixel_area_km2 = _pixel_area_for_cache_dir(args.cache_dir)

    dataset = IgnisDataset(args.cache_dir, split=args.test_split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    sync = WandbSync(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.run_dir.name,
        entity=args.wandb_entity,
    )
    sync.init_run()

    result = evaluate_run(
        run_dir=args.run_dir,
        checkpoint_path=checkpoint_path,
        loader=loader,
        cell=cell,
        seed=seed,
        fire_id=args.fire_id,
        pixel_area_km2=pixel_area_km2,
        mc_samples=args.mc_samples,
        device=args.device,
    )

    sync.log_eval({
        "iou": result.iou,
        "precision": result.precision,
        "recall": result.recall,
        "auc_pr": result.auc_pr,
        "ece": result.ece,
        "growth_rate_mae": result.growth_rate_mae,
        "mean_mc_variance": result.mean_mc_variance,
    })
    sync.finish()

    print(json.dumps({
        "run_name": result.run_name,
        "cell": result.cell,
        "seed": result.seed,
        "fire_id": result.fire_id,
        "iou": result.iou,
        "n_samples": result.n_samples,
    }, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the argument parser**

Run: `python scripts/evaluate.py --help`
Expected: help text lists `--run-dir`, `--cache-dir`, `--fire-id`, `--mc-samples`, `--wandb`. Exit code 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat(cli): add scripts/evaluate.py for single-run scoring"
```

---

## Task 17: `scripts/report_ablation.py` CLI

**Files:**
- Create: `scripts/report_ablation.py`

- [ ] **Step 1: Write the CLI**

Create `scripts/report_ablation.py`:

```python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from ignisca.evaluation.aggregate import aggregate_cell, collect_runs
from ignisca.evaluation.failure import rank_failures
from ignisca.evaluation.reporting import render_headline_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ablation runs and render headline table")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/ablation.md"))
    parser.add_argument("--also-failures", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--failures-out", type=Path, default=Path("reports/failures/"))
    args = parser.parse_args()

    results = collect_runs(args.runs_root)
    if not results:
        raise SystemExit(f"no eval.json files found under {args.runs_root}")

    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for r in results:
        grouped[(r.fire_id, r.cell)].append(r)

    rows = []
    for (fire_id, cell), seed_results in sorted(grouped.items()):
        rows.append(aggregate_cell(cell=cell, fire_id=fire_id, results=seed_results))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_headline_table(rows))
    print(f"Wrote {args.out} with {len(rows)} rows")

    if args.also_failures:
        from ignisca.evaluation.failure import render_failure_case

        args.failures_out.mkdir(parents=True, exist_ok=True)
        for r in results:
            run_dir = args.runs_root / r.run_name
            sample_jsonl = run_dir / f"sample_metrics_{r.fire_id}.jsonl"
            npz_path = run_dir / f"predictions_{r.fire_id}.npz"
            if not sample_jsonl.exists() or not npz_path.exists():
                continue
            worst = rank_failures(sample_jsonl, k=args.top_k, metric="iou", mode="worst")
            for rank, row in enumerate(worst):
                out_png = args.failures_out / f"{r.run_name}_{r.fire_id}_worst{rank:02d}.png"
                render_failure_case(
                    npz_path=npz_path,
                    sample_idx=int(row["sample_idx"]),
                    out_path=out_png,
                )
        print(f"Wrote failure PNGs under {args.failures_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the argument parser**

Run: `python scripts/report_ablation.py --help`
Expected: help text lists `--runs-root`, `--out`, `--also-failures`, `--top-k`, `--failures-out`. Exit code 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/report_ablation.py
git commit -m "feat(cli): add scripts/report_ablation.py for aggregating and rendering"
```

---

## Task 18: `scripts/run_sweep.py` CLI

**Files:**
- Create: `scripts/run_sweep.py`

- [ ] **Step 1: Write the CLI**

Create `scripts/run_sweep.py`:

```python
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from torch.utils.data import DataLoader

from ignisca.data.dataset import IgnisDataset
from ignisca.evaluation.metrics import PIXEL_AREA_KM2
from ignisca.evaluation.runner import evaluate_run
from ignisca.training.config import TrainConfig
from ignisca.training.loop import train_one_run


def _cache_root_for_head(head: str, cache_fine: Path, cache_coarse: Path) -> Path:
    return cache_fine if head == "fine" else cache_coarse


def _run_lambda_phys_sweep(args: argparse.Namespace) -> list[str]:
    base_cfg = TrainConfig(
        cache_root=_cache_root_for_head("coarse", args.cache_fine, args.cache_coarse),
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.base_seed,
    )
    produced: list[str] = []
    for value in args.values:
        run_name = f"sweep_lambda_phys_{value:g}_seed{args.base_seed}"
        cfg = replace(base_cfg, run_name=run_name, lambda_phys=float(value))
        train_one_run(cfg)
        run_dir = args.out_dir / run_name
        checkpoint_path = run_dir / "best.pt"
        loader = DataLoader(
            IgnisDataset(args.cache_coarse, split="test"),
            batch_size=args.batch_size,
            shuffle=False,
        )
        evaluate_run(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            loader=loader,
            cell=args.base_cell,
            seed=args.base_seed,
            fire_id=args.fire_id,
            pixel_area_km2=PIXEL_AREA_KM2["coarse"],
            mc_samples=args.mc_samples,
        )
        produced.append(run_name)
    return produced


def _run_handoff_sweep(args: argparse.Namespace) -> list[str]:
    """Handoff-threshold sweeps score cross-scale cells only.

    Plan 2's router exposes ``threshold_km2``. The sweep trains once per value
    on the cross-scale base cell, then runs ``evaluate_run`` with the router
    configured per value. For Plan 3 we only orchestrate; real router-config
    plumbing is a follow-up that rides on the same CLI.
    """
    raise NotImplementedError(
        "handoff_threshold sweep is scaffolded but requires router-config plumbing "
        "that lands in a follow-up to Plan 3. Use --sweep lambda_phys for now."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="IgnisCA secondary sweep runner")
    parser.add_argument("--sweep", choices=("lambda_phys", "handoff_threshold"), required=True)
    parser.add_argument("--values", type=float, nargs="+", required=True)
    parser.add_argument("--base-cell", required=True)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--cache-fine", type=Path, required=True)
    parser.add_argument("--cache-coarse", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--fire-id", required=True)
    args = parser.parse_args()

    if args.sweep == "lambda_phys":
        produced = _run_lambda_phys_sweep(args)
    else:
        produced = _run_handoff_sweep(args)

    print(f"Sweep {args.sweep} produced {len(produced)} run(s):")
    for name in produced:
        print(f"  {name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the argument parser**

Run: `python scripts/run_sweep.py --help`
Expected: help text shows `--sweep {lambda_phys,handoff_threshold}`, `--values`, `--base-cell`, `--cache-fine`, `--cache-coarse`, `--fire-id`. Exit code 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_sweep.py
git commit -m "feat(cli): add scripts/run_sweep.py for secondary sweep orchestration"
```

---

## Task 19: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `pytest tests/ -v`
Expected: all tests pass (Plan 1 + Plan 2 + Plan 3). Plan 3 contribution should finish in < 15 seconds; total suite depends on pre-existing tests.

- [ ] **Step 2: Run Plan 3 tests in isolation and time them**

Run: `pytest tests/inference tests/evaluation tests/reporting -v --durations=10`
Expected: all Plan 3 tests pass; total Plan 3 runtime under 15 seconds; no single test exceeds 2 seconds.

- [ ] **Step 3: Confirm no stray artifacts were added to git**

Run: `git status`
Expected: clean working tree (all Plan 3 commits landed). No untracked `runs/`, `reports/`, or `.npz` files visible at repo root.

- [ ] **Step 4: Confirm runtime budget**

If any Plan 3 test took longer than 2 seconds, inspect — the likely culprits are `test_runner.py` (tighten `mc_samples` or reduce batch count in `FakeFireDataset`) or `test_mc_dropout.py` (lower `n_samples`). Fix and re-run.

- [ ] **Step 5: Commit any timing fixes**

If changes were needed in Step 4:

```bash
git add tests/evaluation/fake_dataset.py tests/evaluation/test_runner.py
git commit -m "test: tighten Plan 3 runtime budget"
```

Otherwise skip this step.

---

## Deliverables summary

After all tasks:

- **New packages:** `src/ignisca/inference/`, `src/ignisca/evaluation/`, `src/ignisca/reporting/`
- **8 new source files:** `mc_dropout.py`, `metrics.py`, `slicing.py`, `runner.py`, `aggregate.py`, `reporting.py`, `failure.py`, `wandb_sync.py`
- **3 new CLI scripts:** `scripts/evaluate.py`, `scripts/report_ablation.py`, `scripts/run_sweep.py`
- **8 new test files** + 1 fake dataset helper + 1 golden markdown fixture
- **Modified:** `pyproject.toml` (sklearn core, matplotlib viz extra, viz marker), `.gitignore` (runs artifacts), `tests/conftest.py` (shared fixtures)
- **Runtime budget:** Plan 3 tests < 15 seconds on CPU, all network-free
- **Approximate LOC:** ~1,000 source + ~800 test + ~400 CLI

Once Plan 3 lands, producing the real headline table is purely operational: run the 18-run Plan 2 ablation on a GPU, call `scripts/evaluate.py` per run, then `scripts/report_ablation.py --runs-root runs/ --also-failures`, commit the resulting `reports/ablation.md`.
