# IgnisCA — `ignis-inference-CA`

**A cross-scale, physics-informed wildfire spread forecaster for Southern California Santa Ana events.**

IgnisCA predicts the next-timestep fire perimeter for a given SoCal wildfire using a ResU-Net with an optional physics-informed loss term. The model is pretrained on the public **NDWS** (Next Day Wildfire Spread) dataset and fine-tuned on curated historical SoCal Santa Ana fires, then evaluated against two held-out events: **Palisades 2025** and **Thomas 2017**. The central research question is whether the physics term earns its keep *specifically* under Santa Ana wind conditions.

> **Research prototype.** Not operational software. Not a CAL FIRE deliverable. Not a reproduction of the PolyU cross-scale PINN. The physics term is a regularizer, not a forward simulation.

---

## Table of contents

1. [What it is](#what-it-is)
2. [Data sources](#data-sources)
3. [Model](#model)
4. [Physics-informed loss](#physics-informed-loss)
5. [Ablation design](#ablation-design)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Project layout](#project-layout)
9. [Testing & linting](#testing--linting)
10. [Status](#status)
11. [License](#license)

---

## What it is

| | |
|---|---|
| **Task** | Binary semantic segmentation of the next-timestep fire mask |
| **Input** | 12-channel raster stack (fire mask + fuel + terrain + wind/RH/T + drought) on a 512×512 tile |
| **Output** | Per-pixel fire probability (sigmoid), with MC Dropout uncertainty |
| **Backbone** | ResU-Net, 64→128→256→512 channels, spatial dropout `p=0.2` |
| **Losses** | Binary cross-entropy, optionally + level-set residual derived from a simplified Rothermel spread-rate field |
| **Scales** | 30 m fine head (LANDFIRE-native) and 375 m coarse head (MODIS-native), routed at inference by predicted fire area |
| **Evaluation** | Next-timestep fire-class IoU and AUC-PR on two fully held-out fires, reported as `mean ± std` across three seeds |

**Non-goals.** No 5 m-resolution claims (no public fuel data at that scale for CA). No live RAWS / SDG&E / CAL FIRE FHSZ integrations beyond a single FIRMS NRT hook. No operational hardening, no dashboard, no mobile delivery.

---

## Data sources

All training and evaluation data is archival. The live FIRMS hook is inference-only and never touches the training cache.

| Source | Purpose | Native resolution | Module |
|---|---|---|---|
| [NDWS](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread) | Coarse-head pretrain (~18k CA-heavy samples) | ~375 m | `data/sources/ndws.py` |
| [LANDFIRE 2022](https://landfire.gov/) | Fuel model, canopy cover, canopy height | 30 m | `data/sources/landfire.py` |
| [USGS 3DEP DEM](https://www.usgs.gov/3d-elevation-program) | Elevation → slope, aspect | 10 m → resampled | `data/sources/dem.py` |
| [NOAA HRRR archive](https://rapidrefresh.noaa.gov/hrrr/) | Wind *(u, v)*, RH, temperature | 3 km hourly | `data/sources/hrrr.py` |
| [NIFC perimeters](https://data-nifc.opendata.arcgis.com/) | Ground-truth fire polygons | vector | `data/sources/nifc.py` |
| [VIIRS active fire](https://firms.modaps.eosdis.nasa.gov/) | Inner fire progression series | ~375 m | `data/sources/viirs.py` |
| [NASA FIRMS NRT](https://firms.modaps.eosdis.nasa.gov/active_fire/) | Live ignition hook (inference only) | ~375 m | `data/sources/firms_nrt.py` |

**Input channel stack**

| Channel | Contents |
|---|---|
| 0 | Current fire mask (binary) |
| 1 | Fuel model |
| 2 | Canopy cover |
| 3 | Elevation |
| 4 | Slope |
| 5–6 | Aspect (sin, cos) |
| 7–8 | Wind *u*, *v* |
| 9 | Relative humidity |
| 10 | Temperature |
| 11 | Days-since-rain proxy |

**Fire splits (strict, never mixed at the sample level).** Test set (touched once, at the end): **Palisades 2025**, **Thomas 2017**. Inner-validation fire for checkpoint selection: **Saddleridge 2019**. Training pool: Woolsey, Bobcat, Bond, Getty, Sand, Springs, Holy, Mountain.

**Holdout discipline.** Any tile whose centroid falls within 5 km of the eventual Palisades or Thomas perimeter is dropped from training, even from other events. Any training sample timestamped on or after a held-out fire's ignition is dropped. NDWS samples intersecting the held-out bounding boxes are removed from the pretrain set. These rules apply uniformly across all ablation cells.

---

## Model

A ResU-Net: U-Net with residual blocks in encoder and decoder, chosen for well-understood training dynamics, single-GPU feasibility, and direct comparability to published NDWS baselines.

| Stage | Config |
|---|---|
| Encoder | 4 residual downsampling stages, 64 → 128 → 256 → 512 |
| Bottleneck | 2 residual blocks at 512 |
| Decoder | 4 residual upsampling stages with skip connections, 512 → 256 → 128 → 64 |
| Head | 1-channel sigmoid |
| Dropout | 2D spatial dropout, `p=0.2`, kept active at inference for MC Dropout |

**Cross-scale routing.** Cross-scale is two training runs of the same backbone, *not* a single joint architecture. At inference, a router (`models/router.py`) selects between them based on the current predicted fire area:

- **Fine head** — 30 m grid, trained on SoCal fine-tune fires. Active when predicted fire area **< 5 km²**.
- **Coarse head** — 375 m grid, NDWS-pretrained then fine-tuned on SoCal fires resampled to 375 m. Active when fire area **≥ 5 km²**.

The 5 km² handoff is a hyperparameter swept during ablation (see below).

**Inference-time uncertainty.** Dropout stays active. Each prediction runs `N = 20` forward passes; the per-pixel mean is the predicted perimeter at a 0.5 threshold, the per-pixel variance is the uncertainty field.

---

## Physics-informed loss

The physics-informed variant adds a **level-set residual** to the training loss. The fire boundary is treated as the zero level-set of a signed-distance function `φ`, and the Hamilton–Jacobi fire spread equation is penalized as a soft constraint:

```
total_loss  =  λ_data · BCE(pred, truth)
             + λ_phys · mean( (∂φ/∂t + F · |∇φ|)² )
```

- `φ` — signed distance transform of the predicted fire mask
- `F` — per-pixel spread rate from a simplified **Rothermel** model using fuel, slope, and wind channels already in the input stack
- `∂φ/∂t` — finite differences between input and predicted fire masks
- `|∇φ|` — Sobel gradient magnitude
- Defaults: `λ_data = 1.0`, `λ_phys = 0.1`. `λ_phys` is swept over `{0.01, 0.05, 0.1, 0.3}`.

This is "physics-informed" in the literal published sense — a PDE residual in the loss. The backbone is still a fully trainable CNN. There is no collocation sampling, no coordinate MLP, no PINN-specific training pathology.

---

## Ablation design

The central experiment is a **four-cell ablation**: scale × loss.

|  | Single-scale (375 m only) | Cross-scale (fine + coarse) |
|---|---|---|
| **Data-only loss** | A₁ | A₂ |
| **Physics-informed loss** | B₁ | B₂ |

Four cells × three seeds ⇒ **12 reported models**, plus ~6 secondary sweep runs on the winning cells (λ_phys, handoff threshold) ⇒ ~18 total training runs.

**Headline table (pending results)**

```
                        Palisades 2025          Thomas 2017
                        IoU    AUC-PR           IoU    AUC-PR
                        ─────────────────────────────────────
Single-scale, data      _____  ______           _____  ______
Single-scale, PI        _____  ______           _____  ______
Cross-scale, data       _____  ______           _____  ______
Cross-scale, PI         _____  ______           _____  ______
```

**Slice analysis** drives the scientific story: (1) Santa Ana vs. non-Santa Ana timesteps — does the physics term help *specifically* in extreme wind?; (2) early fire (< 5 km²) vs. mature fire — does cross-scale help *specifically* in the fine-head regime?; (3) terrain complexity binned by local slope variance. At least three documented failure modes with visualizations.

---

## Installation

Python **3.11 or 3.12** (pinned in `.python-version`). GPU strongly recommended for training; CPU is fine for the test suite.

```bash
git clone https://github.com/raiyan37/ignis-inference-CA.git
cd ignis-inference-CA

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,viz]"
```

Optional extras:

- `dev` — `pytest`, `pytest-cov`, `ruff`
- `viz` — `matplotlib` (only needed for visualizations, gated by the `viz` pytest marker)

---

## Usage

### 1. Preprocess a fire into cache shards

```bash
python -m scripts.preprocess \
  --fire-name Woolsey \
  --center-lon -118.75 --center-lat 34.20 \
  --resolution fine \
  --start 2018-11-08T00:00 --end 2018-11-12T00:00 \
  --step-hours 1 --delta-hours 1 \
  --fuel    data/landfire/fbfm40.tif \
  --canopy  data/landfire/cc.tif \
  --dem     data/dem/socal_10m.tif \
  --hrrr    data/hrrr/2018-11/ \
  --nifc    data/nifc/woolsey.geojson \
  --cache-root cache/ \
  --split train \
  --held-out-fires configs/held_out.geojson
```

Outputs `.npz` shards under `cache/train/`. Held-out Palisades/Thomas tiles are filtered at the dataloader level via spatial (5 km buffer) and temporal rules. Use `--resolution coarse` for the 375 m head; preprocess fine and coarse caches separately.

### 2. Train a single run

```bash
python -m scripts.train \
  --cache-root cache/ \
  --run-name cross-scale-PI-seed0 \
  --epochs 30 \
  --batch-size 4 \
  --lr 3e-4 \
  --lambda-data 1.0 \
  --lambda-phys 0.1 \
  --dropout 0.2 \
  --seed 0 \
  --device cuda
```

Writes the best-IoU checkpoint and a JSONL training log under `runs/<run-name>/`. Set `--lambda-phys 0.0` for the data-only variant.

### 3. Run the full 18-run ablation grid

```bash
python -m scripts.run_ablation \
  --cache-fine   cache-fine/ \
  --cache-coarse cache-coarse/ \
  --out-dir runs/ \
  --epochs 30 \
  --results-json runs/ablation_results.json
```

Enumerates the four primary cells × three seeds and writes a consolidated JSON with per-run `best_val_iou`. At ~2–4 hours per run on a consumer GPU, the full grid is 1.5–3 days of wall-clock training.

---

## Project layout

```
src/ignisca/
├── data/
│   ├── sources/        # NDWS, HRRR, LANDFIRE, DEM, NIFC, VIIRS, FIRMS NRT loaders
│   ├── grid.py         # TargetGrid (fine=30m, coarse=375m, EPSG:3857)
│   ├── features.py     # 12-channel feature stack assembly
│   ├── holdout.py      # spatial + temporal holdout predicates
│   ├── cache.py        # .npz shard reader/writer
│   └── dataset.py      # torch Dataset over cache shards
├── models/
│   ├── resunet.py      # ResU-Net backbone
│   └── router.py       # cross-scale inference router (5 km² default handoff)
├── training/
│   ├── config.py       # TrainConfig dataclass
│   ├── losses.py       # BCE, Rothermel field, level-set residual, IgnisLoss
│   ├── metrics.py      # fire-class IoU
│   ├── loop.py         # training loop, best-IoU checkpointing, JSONL log
│   └── ablation.py     # 18-run ablation grid (4 cells × 3 seeds + head count)
├── inference/
│   └── mc_dropout.py   # MC Dropout predictor, per-pixel mean + variance
├── evaluation/
│   └── metrics.py      # IoU, precision, recall, AUC-PR
└── reporting/          # ablation tables and slice analyses

scripts/
├── preprocess.py       # orchestrator CLI
├── train.py            # single-run CLI
└── run_ablation.py     # full-grid CLI

tests/                  # mirrors src/ignisca/ layout
docs/superpowers/       # design spec and implementation plans
```

---

## Testing & linting

```bash
pytest                          # full suite
pytest tests/models/            # only model tests
pytest -k resunet               # single test by expression
pytest -m viz                   # matplotlib-gated visualization tests

ruff check src tests scripts    # lint
ruff format src tests scripts   # format
```

Tests that require the `viz` extra are gated by the `viz` marker and skipped cleanly if `matplotlib` is not installed. Pytest is configured in `pyproject.toml` with `pythonpath = ["src"]`, so tests import the package without an editable install in CI.

---

## Status

- **Done** — Data pipeline: sources, grid, features, holdout, cache, dataset
- **Done** — Models: ResU-Net backbone, cross-scale router
- **Done** — Losses: Sobel gradient, Rothermel spread-rate field, level-set residual, `IgnisLoss`
- **Done** — Training: `TrainConfig`, fire-class IoU metric, training loop, checkpoint selection
- **Done** — Ablation: 18-run grid runner, single-run + full-grid CLIs
- **Done** — Evaluation: precision, recall, AUC-PR; MC Dropout predictor
- **Pending** — Reporting: ablation table + slice analyses
- **Pending** — Trained model checkpoints, headline IoU numbers, failure-mode visualizations

The detailed design spec lives at [`docs/superpowers/specs/2026-04-11-ignisca-design.md`](docs/superpowers/specs/2026-04-11-ignisca-design.md); implementation plans are under `docs/superpowers/plans/`.

---

## License

See [`LICENSE`](LICENSE).
