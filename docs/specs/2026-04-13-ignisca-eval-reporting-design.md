# IgnisCA Plan 3 — Evaluation & Reporting Design

**Date:** 2026-04-13
**Status:** Approved, ready for implementation planning
**Parent spec:** `docs/superpowers/specs/2026-04-11-ignisca-design.md` §3.4, §4.2, §4.4, §4.5, §5
**Depends on:** Plan 1 (data pipeline, merged), Plan 2 (model + training, merged)

---

## 1. Goal & Scope

Build the evaluation and reporting infrastructure for IgnisCA as a self-contained,
offline-testable library. Mirrors the Plan 1/2 pattern: every module is unit-tested
on CPU with synthetic data, and no real training or real-data scoring happens
inside Plan 3. Real runs are a followup once the infra lands.

### In scope

1. **Metric library**: fire-class IoU (reused from `training/metrics.py`), precision,
   recall, AUC-PR, ECE (10 bins, pixel-level), growth-rate MAE (per-sample, km²/hr).
2. **MC Dropout inference wrapper** (`inference/mc_dropout.py`): N=20 forward passes
   with only `Dropout2d` submodules in train mode, returns mean + per-pixel variance.
3. **Slice analysis**: per-sample Santa Ana classification from HRRR wind channels 7/8
   (offshore NE quadrant + ≥ 7 m/s speed threshold).
4. **Scoring runner**: `evaluate_run(cfg, checkpoint)` loads a checkpoint, runs test
   fires, computes all metrics, writes per-run JSON + per-sample JSONL + predictions `.npz`.
5. **Cross-seed aggregation**: collapse 3 seeds per ablation cell into mean ± std.
6. **Headline markdown report**: 8 rows (2 held-out fires × 4 cells) × 7 metric columns,
   rendered to `reports/ablation.md`.
7. **Failure-mode pipeline**: auto-rank worst-K cases per run, dump PNG grids;
   single-case escape-hatch renderer (hybrid auto + manual approach).
8. **Optional W&B sync** behind a `--wandb` flag. Local JSONs remain source of truth.
9. **CLI scripts**: `scripts/evaluate.py`, `scripts/report_ablation.py`, `scripts/run_sweep.py`.
10. **Secondary sweep orchestration**: config + runner support for λ_phys and
    handoff-threshold sweeps. Orchestration only — sweeps are not executed in this plan.

### Out of scope (explicit)

- Real training runs or real ablation execution.
- Real W&B project creation (user creates account when ready to run real ablation).
- Rollout-mode / autoregressive inference (stays teacher-forced).
- Interactive dashboards beyond the markdown report + optional W&B.
- Per-fire trajectory growth-rate metric (per-sample only for Plan 3).
- Serving layer or saved-model export.

### Testing philosophy

Every module unit-tested against synthetic tensors and a tiny `ResUNet(base=4)` on
CPU. Failure-mode rendering verified for file creation + image shape, not visual
content. W&B adapter tested via a monkeypatched `wandb` module — network never
touched. Target full-suite runtime: **< 15 seconds on CPU**.

---

## 2. File Structure & Module Boundaries

```
src/ignisca/
├── inference/
│   ├── __init__.py                    (new)
│   └── mc_dropout.py                  (new) — mc_dropout_predict(model, x, n=20)
├── evaluation/
│   ├── __init__.py                    (new)
│   ├── metrics.py                     (new) — precision, recall, auc_pr, ece,
│   │                                          growth_rate_mae
│   ├── slicing.py                     (new) — Santa Ana classifier, slice groups
│   ├── runner.py                      (new) — EvalResult dataclass;
│   │                                          evaluate_run(cfg, ckpt, loader)
│   ├── aggregate.py                   (new) — aggregate_cell(results_by_seed)
│   ├── reporting.py                   (new) — render_headline_table(...)
│   └── failure.py                     (new) — rank_failures, render_failure_case
├── reporting/
│   ├── __init__.py                    (new)
│   └── wandb_sync.py                  (new) — WandbSync adapter, opt-in
└── training/metrics.py                (existing) — fire_class_iou reused

scripts/
├── evaluate.py                        (new) — single-run scoring CLI
├── report_ablation.py                 (new) — aggregate + render headline table
└── run_sweep.py                       (new) — λ_phys / handoff-threshold sweep runner

tests/
├── inference/test_mc_dropout.py
├── evaluation/test_metrics.py
├── evaluation/test_slicing.py
├── evaluation/test_runner.py
├── evaluation/test_aggregate.py
├── evaluation/test_reporting.py
├── evaluation/test_failure.py
├── evaluation/fixtures/headline.md    (golden file)
├── evaluation/fake_dataset.py         (synthetic in-memory Dataset)
└── reporting/test_wandb_sync.py
```

**Module naming note:** The package is named `evaluation/` (not `eval/`) to avoid
shadowing Python's builtin `eval` identifier. Scripts use the verb "evaluate" in
CLI and function names, but the module lives at `ignisca.evaluation`.

**Dependency direction** (strictly one-way):

```
metrics.py ─┐
slicing.py ─┼─→ runner.py ─→ aggregate.py ─→ reporting.py
            │                                    ↑
mc_dropout.py ─→ runner.py                       │
                                          wandb_sync.py (optional adapter)
failure.py ─→ runner.py (reads EvalResult, writes PNGs)
```

No file imports downstream. `runner.py` is the single orchestration point; every
other file is a pure library testable in isolation.

**Module responsibilities:**

- `metrics.py`: pure functions on tensors, no I/O, no model loading.
- `slicing.py`: pure functions on tensors + module-level threshold constants.
- `mc_dropout.py`: inference utility, takes a loaded model + batch.
- `runner.py`: the only module that loads checkpoints, reads datasets, writes files.
- `aggregate.py`: pure functions on lists of `EvalResult`.
- `reporting.py`: pure functions, takes `AggregatedRow`s → markdown string.
- `failure.py`: reads the JSONL + .npz artifacts written by `runner.py`, writes PNGs.
- `wandb_sync.py`: optional adapter, inert unless `enabled=True`.

---

## 3. Data Contracts

The library is glued together by two schemas: `EvalResult` (per-run) and
`AggregatedRow` (per-cell). Locking these down prevents drift between tasks.

### 3.1 `EvalResult` dataclass

Defined in `src/ignisca/evaluation/runner.py`:

```python
@dataclass
class EvalResult:
    run_name: str
    cell: str                 # "A1" | "A2" | "B1" | "B2"
    seed: int
    fire_id: str              # "palisades_2025" | "thomas_2017"
    iou: float
    precision: float
    recall: float
    auc_pr: float
    ece: float
    growth_rate_mae: float    # km²/hr, mean over samples
    mean_mc_variance: float   # mean per-pixel variance over samples
    slices: dict[str, dict[str, float]]
    # e.g. {"santa_ana": {"iou": ..., ...}, "non_santa_ana": {...},
    #       "early": {...}, "mature": {...}}
    predictions_path: Path
    sample_metrics_path: Path
    n_samples: int
```

### 3.2 Per-run JSON schema

Written to `runs/<run_name>/eval.json`:

```json
{
  "run_name": "cell_A1_seed0",
  "cell": "A1",
  "seed": 0,
  "fires": [
    {
      "fire_id": "palisades_2025",
      "iou": 0.621,
      "precision": 0.58,
      "recall": 0.63,
      "auc_pr": 0.71,
      "ece": 0.09,
      "growth_rate_mae": 0.42,
      "mean_mc_variance": 0.018,
      "slices": {
        "santa_ana": { "iou": 0.59, "precision": 0.0, "recall": 0.0, "auc_pr": 0.0, "ece": 0.0, "growth_rate_mae": 0.0, "mean_mc_variance": 0.0 },
        "non_santa_ana": { "iou": 0.0, "precision": 0.0, "recall": 0.0, "auc_pr": 0.0, "ece": 0.0, "growth_rate_mae": 0.0, "mean_mc_variance": 0.0 },
        "early": { "iou": 0.0, "precision": 0.0, "recall": 0.0, "auc_pr": 0.0, "ece": 0.0, "growth_rate_mae": 0.0, "mean_mc_variance": 0.0 },
        "mature": { "iou": 0.0, "precision": 0.0, "recall": 0.0, "auc_pr": 0.0, "ece": 0.0, "growth_rate_mae": 0.0, "mean_mc_variance": 0.0 }
      },
      "n_samples": 212
    }
  ],
  "mc_dropout": { "n_samples": 20 },
  "artifacts": {
    "predictions_palisades_2025": "runs/cell_A1_seed0/predictions_palisades_2025.npz",
    "sample_metrics_palisades_2025": "runs/cell_A1_seed0/sample_metrics_palisades_2025.jsonl"
  }
}
```

### 3.3 Per-sample JSONL schema

One line per test sample, written to
`runs/<run_name>/sample_metrics_<fire_id>.jsonl`:

```json
{"sample_idx": 0, "fire_id": "palisades_2025", "timestep": "2025-01-07T14:00Z",
 "iou": 0.58, "precision": 0.61, "recall": 0.55, "ece": 0.12,
 "growth_rate_abs_err_km2_hr": 0.34,
 "mean_mc_variance": 0.022,
 "santa_ana": false, "is_early_fire": true}
```

Sample metrics are the input for `failure.py`'s worst-K ranking — no re-running
inference is needed.

### 3.4 Predictions .npz schema

Written to `runs/<run_name>/predictions_<fire_id>.npz`:

```
mean:        (N, H, W) float16   # MC Dropout mean of sigmoid(logits)
variance:    (N, H, W) float16   # MC Dropout per-pixel variance
target:      (N, H, W) uint8     # ground-truth next-step mask
input_mask:  (N, H, W) uint8     # current fire mask (channel 0 of the input stack)
```

Float16 to keep size bounded; the failure renderer upcasts on load. These files
are NOT committed to git (added to `.gitignore`).

### 3.5 Aggregated row

Defined in `src/ignisca/evaluation/aggregate.py`:

```python
@dataclass
class AggregatedRow:
    fire_id: str
    cell: str
    metrics: dict[str, tuple[float, float]]  # metric_name -> (mean, std) across seeds
    n_seeds: int
```

The headline table is `list[AggregatedRow]` — exactly 8 rows for 2 fires × 4 cells.

---

## 4. Key Algorithms

### 4.1 MC Dropout wrapper

`src/ignisca/inference/mc_dropout.py`:

```python
def mc_dropout_predict(model, x, n_samples=20):
    model.eval()
    # Enable ONLY Dropout2d submodules; leave GroupNorm in inference mode.
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train(True)
    with torch.no_grad():
        samples = torch.stack([torch.sigmoid(model(x)) for _ in range(n_samples)])
        # samples: (n, B, 1, H, W)
    mean = samples.mean(dim=0)
    var = samples.var(dim=0, unbiased=False)
    model.eval()  # restore
    return mean, var
```

Keeping GroupNorm in inference mode is critical: flipping the whole model to
`.train()` would contaminate normalization statistics across MC samples. The
function walks `model.modules()` and only toggles `nn.Dropout2d` instances.

**Tests assert:**
- `model.training is False` after the call returns.
- Variance is identically 0 when `dropout=0` (passed via a fresh `ResUNet(dropout=0)`).
- Variance is strictly positive in at least one pixel when `dropout>0`.
- Output shapes: `mean.shape == var.shape == (B, 1, H, W)`.

### 4.2 Precision, recall, AUC-PR

`src/ignisca/evaluation/metrics.py`:

- **Precision / recall** at the fixed 0.5 threshold (matches IoU for consistency).
  Tested against a hand-computed 4-pixel case.
- **AUC-PR** is threshold-free. Uses `sklearn.metrics.average_precision_score` on
  flattened sigmoid probabilities vs. flattened target. Sklearn is added as an
  explicit core dependency (see §7 Risks).

### 4.3 ECE

`src/ignisca/evaluation/metrics.py`:

10 equal-width bins over `[0, 1]`. For each bin `[i/10, (i+1)/10)`:

```
bin_conf = mean of predicted sigmoid probs in bin
bin_acc  = fraction of targets == 1 in bin
weight   = |bin| / total_pixels
ece += weight * |bin_acc - bin_conf|
```

Pixel-level, background included. Standard practice.

**Tests assert:**
- Perfect calibration (`pred == target.float()`) → ECE ≈ 0 (< 1e-6).
- Uniformly-0.5 prediction against a 50/50 target → ECE = 0.
- A scripted miscalibrated case matches a hand-computed ECE value.

### 4.4 Growth-rate MAE

`src/ignisca/evaluation/metrics.py`:

```python
def growth_rate_mae(pred_logits, target, input_mask, pixel_area_km2, dt_hours=1.0):
    pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
    pred_area_next = pred_bin.sum(dim=(-2, -1)) * pixel_area_km2
    true_area_next = target.sum(dim=(-2, -1)) * pixel_area_km2
    curr_area = input_mask.sum(dim=(-2, -1)) * pixel_area_km2
    pred_growth = (pred_area_next - curr_area) / dt_hours
    true_growth = (true_area_next - curr_area) / dt_hours
    return (pred_growth - true_growth).abs().mean().item()
```

`pixel_area_km2` per cell:
- Fine (30 m): `0.0009` km²/pixel
- Coarse (375 m): `0.140625` km²/pixel

Runner looks up the correct value per cell from a small constant map keyed on
`cell` ∈ {`A1`, `A2`, `B1`, `B2`}.

**Tests assert:**
- `growth_rate_mae(target, target, input_mask, ...) == 0.0`.
- A scripted case (known predicted area + known true area) returns the hand-computed value.

### 4.5 Santa Ana classifier

`src/ignisca/evaluation/slicing.py`:

```python
SANTA_ANA_SPEED_MIN = 7.0       # m/s (~15 mph)
SANTA_ANA_DIR_RANGE = (0, 90)   # met convention: wind FROM N to E (offshore for SoCal)

def classify_santa_ana(features):
    # features: (B, 12, H, W); wind_u ch7, wind_v ch8
    u = features[:, 7]
    v = features[:, 8]
    u_mean = u.mean(dim=(-2, -1))
    v_mean = v.mean(dim=(-2, -1))
    speed = torch.sqrt(u_mean ** 2 + v_mean ** 2 + 1e-8)
    # Meteorological "from" direction: wind vector (u, v) flows toward;
    # negate to get the origin direction.
    dir_deg = (torch.atan2(-u_mean, -v_mean) * 180 / math.pi) % 360
    in_offshore = (dir_deg >= SANTA_ANA_DIR_RANGE[0]) & (dir_deg < SANTA_ANA_DIR_RANGE[1])
    return (speed >= SANTA_ANA_SPEED_MIN) & in_offshore   # bool (B,)
```

Santa Anas blow from the NE quadrant, which in meteorological convention is
"direction 0°–90°". In standard (u, v) that's a wind vector pointing SW (toward
negative u, negative v). The `-u, -v` in `atan2` converts the "flowing-to" vector
into the "coming-from" angle.

**Also defined:**

- `is_early_fire(input_mask, pixel_area_km2) -> bool`: `current_area_km2 < 5.0`.
- `slice_groups(features, input_mask, pixel_area_km2) -> dict[str, Tensor]`:
  returns bool masks for `{"santa_ana", "non_santa_ana", "early", "mature"}`.

**Tests assert:**
- NE-origin 10 m/s wind (u=-7.07, v=-7.07) → True.
- SW-origin 10 m/s wind (u=+7.07, v=+7.07) → False.
- NE-origin 3 m/s wind → False (below speed threshold).
- `is_early_fire` True ↔ area < 5 km², tested at the boundary.

### 4.6 Failure ranking

`src/ignisca/evaluation/failure.py`:

```python
def rank_failures(sample_metrics_jsonl, k=10, metric="iou", mode="worst"):
    rows = [json.loads(l) for l in open(sample_metrics_jsonl)]
    reverse = (mode == "best")
    rows.sort(key=lambda r: r[metric], reverse=reverse)
    return rows[:k]
```

Consumer then calls `render_failure_case(npz_path, sample_idx, out_path)` which
pulls arrays from `predictions_<fire>.npz` and writes a 4-panel PNG:
`(input_mask | target | prediction mean | variance heatmap)` via matplotlib with
a fixed viridis + reds colormap pair. Matplotlib uses the Agg backend (no GUI).

The single-case escape hatch is the same `render_failure_case` function — no
separate API. The hybrid flow: auto-rank surfaces candidates, the user picks the
most *interesting* ones (which may not be the strictly worst) by passing a specific
`sample_idx` directly.

---

## 5. Testing Strategy

**Core principle:** every module testable on CPU in < 5 seconds using synthetic
tensors or a tiny on-the-fly model.

### 5.1 Shared fixtures

Added to `tests/conftest.py`:

```python
@pytest.fixture
def tiny_resunet():
    # base=4 -> ~5k params, H=W=32 (32 % 16 == 0)
    return ResUNet(in_channels=12, base=4, dropout=0.3)

@pytest.fixture
def tiny_checkpoint(tmp_path, tiny_resunet):
    ckpt_path = tmp_path / "best.pt"
    torch.save({"epoch": 0, "model_state_dict": tiny_resunet.state_dict(),
                "config": {}}, ckpt_path)
    return ckpt_path

@pytest.fixture
def synthetic_batch():
    # 4 samples, 12 channels, 32x32, plausible per-channel distributions
    torch.manual_seed(0)
    x = torch.randn(4, 12, 32, 32) * 0.5
    x[:, 0] = (torch.rand(4, 32, 32) > 0.85).float()    # fire mask
    x[:, 1] = torch.rand(4, 32, 32)                     # fuel in [0,1]
    x[:, 4] = torch.relu(torch.randn(4, 32, 32))        # slope >= 0
    x[:, 7] = torch.randn(4, 32, 32) * 5                # wind_u
    x[:, 8] = torch.randn(4, 32, 32) * 5                # wind_v
    y = (torch.rand(4, 32, 32) > 0.80).float()
    return x, y

@pytest.fixture
def santa_ana_batch():
    # Uniform SW-flowing wind ~10 m/s -> "from NE" in met convention
    x = torch.zeros(2, 12, 32, 32)
    x[:, 7] = -7.07   # u
    x[:, 8] = -7.07   # v
    return x
```

### 5.2 Per-module test plan

| Module | Key tests | Fixtures |
|---|---|---|
| `mc_dropout.py` | var == 0 when dropout=0; var > 0 when dropout>0; `model.training is False` after call; output shapes (B,1,H,W) | `tiny_resunet`, `synthetic_batch` |
| `metrics.py` | precision/recall on 4-pixel case; AUC-PR matches sklearn reference; ECE ~ 0 for perfect pred; growth-rate MAE = 0 when pred == target; scripted case | synthetic tensors |
| `slicing.py` | NE-origin 10 m/s → True; SW-origin 10 m/s → False; NE-origin 3 m/s → False; `is_early_fire` at 5 km² boundary | `santa_ana_batch` |
| `runner.py` | End-to-end with tiny checkpoint + synthetic dataset: writes `eval.json` with expected schema; JSONL + .npz written; slice metrics present | `tiny_checkpoint`, fake dataset |
| `aggregate.py` | 3 `EvalResult`s → `AggregatedRow` with correct mean/std; missing seed fails loud (not silent) | hand-built EvalResults |
| `reporting.py` | Markdown contains every row + metric; golden-file test against `tests/evaluation/fixtures/headline.md` | hand-built AggregatedRows |
| `failure.py` | Ranking picks lowest-IoU sample; `render_failure_case` writes PNG of expected dimensions; Agg backend set at module top | synthetic JSONL + .npz |
| `wandb_sync.py` | Monkeypatched `wandb.init/log/finish` as no-ops; adapter records calls in right order; `enabled=False` → zero `wandb.*` calls | `monkeypatch` |

### 5.3 Fake dataset for the runner

`tests/evaluation/fake_dataset.py` provides a tiny in-memory `Dataset` yielding
`(x, y, fire_id, timestep_str)` for 8 fake samples split across two fake fires,
with metadata sufficient to exercise slice classification.

### 5.4 Golden-file test for reporting

`tests/evaluation/fixtures/headline.md` is committed alongside the test. The test
normalizes trailing whitespace and asserts `actual == expected`. Easy to update
when format changes, clear diff on break.

### 5.5 What is NOT tested

- Actual W&B network calls (only adapter control flow).
- Actual model quality (tiny model's outputs are near-random; we check plumbing).
- Real GPU code paths (CPU-only).
- Visual fidelity of failure PNGs (only valid PNG of the right shape).

### 5.6 Runtime budget

Full Plan 3 test suite: **< 15 seconds on CPU**. If any single test approaches 2s,
split it.

---

## 6. CLI Surface

### 6.1 `scripts/evaluate.py` — score one run

```bash
python scripts/evaluate.py \
  --run-dir runs/cell_A1_seed0 \
  --cache-fine data/cache/fine \
  --cache-coarse data/cache/coarse \
  --test-fires palisades_2025 thomas_2017 \
  --mc-samples 20 \
  --out runs/cell_A1_seed0/eval.json \
  [--wandb] [--wandb-project ignisca]
```

Loads `runs/<name>/best.pt`, infers the cell from the run's saved config, picks
the appropriate cache, runs scoring on each test fire, writes `eval.json` +
per-sample JSONL + predictions `.npz`. `--wandb` is inert unless passed; when set,
the `WandbSync` adapter uploads the JSON.

### 6.2 `scripts/report_ablation.py` — aggregate + render the headline table

```bash
python scripts/report_ablation.py \
  --runs-root runs/ \
  --cells A1 A2 B1 B2 \
  --seeds 0 1 2 \
  --test-fires palisades_2025 thomas_2017 \
  --out reports/ablation.md \
  [--also-failures --top-k 10 --failures-out reports/failures/]
```

Reads every `runs/<cell>_seed<N>/eval.json`, runs `aggregate_cell` per cell,
hands results to `render_headline_table`, writes `reports/ablation.md`. With
`--also-failures`, walks each run's sample JSONL, grabs the worst-K, and writes
`reports/failures/<cell>_seed<N>_<rank>.png` plus a companion markdown index.

### 6.3 `scripts/run_sweep.py` — secondary sweep orchestration

```bash
python scripts/run_sweep.py \
  --sweep lambda_phys \
  --values 0.01 0.05 0.1 0.3 \
  --base-cell B2 \
  --base-seed 0 \
  --cache-fine ... --cache-coarse ... \
  --out-dir runs/sweep_lambda_phys/ \
  --epochs 30
```

Wraps the existing `train_one_run` from Plan 2. Two `--sweep` modes:

- `lambda_phys`: varies `TrainConfig.lambda_phys` across the provided `--values`.
- `handoff_threshold`: varies the router's `threshold_km2` (cross-scale cells only).

Each sweep value = one training run, writes to its own subdirectory, then runs
scoring on it by calling `evaluate_run()` directly (not shell-invoking `evaluate.py`).

The sweep is parameterized on a *base cell* (`--base-cell`). Plan 3 does not know
which cell wins the primary ablation — that's a human decision post-ablation.

**Tests assert:** CLI parses correctly, orchestrates 2 tiny runs against fixtures,
produces an `eval.json` per sweep value. Does NOT actually execute real training
beyond the fixture smoke test.

---

## 7. Risks & Open Questions

### 7.1 Matplotlib as a new hard dependency

Plans 1 & 2 don't need matplotlib. **Resolution:** add `matplotlib` to
`[project.optional-dependencies].viz` in `pyproject.toml`, and make `failure.py`
import it lazily inside the render function. Tests opt into `viz` via a pytest marker.

### 7.2 sklearn for AUC-PR

Already transitively present via scipy, but not an explicit dependency.
**Resolution:** add `scikit-learn` as an explicit core dependency in `pyproject.toml`.
One line, maintained forever, 10 MB is fine for a research project.

### 7.3 Santa Ana threshold calibration

The 7 m/s speed and 0°–90° direction thresholds are defensible but arbitrary.
**Resolution:** expose as module-level constants in `slicing.py`
(`SANTA_ANA_SPEED_MIN`, `SANTA_ANA_DIR_RANGE`). Document the met convention in
a docstring. Retuning is a followup if the SA/non-SA split is uneven on real test fires.

### 7.4 `predictions_<fire>.npz` disk usage

Float16 × 2 arrays × ~200 samples × 512×512 ≈ 200 MB per fire. For 18 runs × 2
fires: ~7 GB. **Resolution:** `.gitignore` entry for `runs/**/predictions_*.npz`
and `runs/**/sample_metrics_*.jsonl`. Documented in the plan's setup step.

### 7.5 Teacher-forced scoring only

The runner feeds true input masks at each step, not its own previous predictions.
This is what the parent spec requests for Plan 3. **Resolution:** document clearly
in the README section introducing the headline table, so a reader doesn't mistake
the numbers for autoregressive rollout accuracy.

### 7.6 MC Dropout cost

20× inference per sample × 2 fires × 18 runs is the dominant scoring-time cost.
Estimate on a single A100 at batch=4 float16: ~45 min total. No mitigation needed,
just flagged for planning.

### 7.7 W&B account provisioning

The user does NOT need a W&B account for Plan 3 — the adapter is fully testable
via monkeypatching. **When to provision:** right before kicking off the real 18-run
ablation (post-Plan 3 merge). 60-second signup, free tier is sufficient.

---

## 8. Deliverables Summary

- **8 new source files** under `src/ignisca/{evaluation,inference,reporting}/`
- **3 new CLI scripts** under `scripts/`
- **8 new test files + 1 fake dataset + 1 golden fixture** under `tests/`
- **Updated `pyproject.toml`**: add `scikit-learn` (core) and `matplotlib` (viz extra)
- **Updated `.gitignore`**: exclude `runs/**/predictions_*.npz` and `runs/**/sample_metrics_*.jsonl`
- **Approximate LOC**: ~1,000 source + ~800 test
- **All Plan 3 tests CPU-only, network-free, < 15 seconds total**

Once Plan 3 lands, the outstanding items for producing the headline table are
purely operational: provision W&B (optional), run the 18-run ablation from Plan 2
CLI, run `scripts/report_ablation.py`, commit the resulting `reports/ablation.md`.
