# IgnisCA

**A cross-scale, physics-informed wildfire spread forecaster for Southern California Santa Ana events.**

April 2026 · Research prototype · [repo pending]

---

## Overview

IgnisCA is a research prototype that predicts the next-timestep fire perimeter for a given Southern California wildfire, using a cross-scale U-Net with an optional physics-informed loss term. The model is trained on a mixture of NDWS and curated historical SoCal Santa Ana fires, and evaluated against two held-out events (Palisades 2025, Thomas 2017). The primary scientific contribution is a four-cell ablation — scale (single vs. cross) crossed with loss (data-only vs. physics-informed) — with a particular focus on whether the physics term earns its keep during Santa Ana wind conditions.

This document specifies the methodology only. The visualization layer is a separate concern and is not documented here.

## Contents

1. [Scope & success criteria](#1-scope--success-criteria)
2. [Data architecture](#2-data-architecture)
3. [Model architecture](#3-model-architecture)
4. [Training & ablation protocol](#4-training--ablation-protocol)
5. [Evaluation](#5-evaluation)

---

## 1. Scope & success criteria

IgnisCA is scoped as a research prototype targeting a technical/research audience and a portfolio piece. It is not operational software, not a reproduction of the PolyU cross-scale PINN, and not a CAL FIRE deliverable. The core artifact is a defensible next-timestep fire perimeter prediction on historical SoCal Santa Ana fires, reported as an ablation table with held-out IoU and AUC-PR.

### 1.1 Deliverables

1. A trained cross-scale ResU-Net with two heads (30m fine, 375m coarse) and a physics-informed ablation variant.
2. A four-cell ablation table on held-out Palisades 2025 and Thomas 2017, reported as `mean ± std` across three seeds per cell.
3. An MC Dropout uncertainty map rendered per prediction.
4. A single NASA FIRMS near-real-time hook so the inference pipeline can be pointed at a current hotspot.
5. A methodology-focused README with the ablation table and slice analysis.

### 1.2 Non-goals

- Reproducing the PolyU cross-scale PINN architecture verbatim.
- Fire prediction at 5m resolution. No public fuel data exists at that scale for California; any claim at that resolution would be scientifically indefensible.
- Live AlertCalifornia, RAWS, SDG&E, or CAL FIRE FHSZ API integrations beyond the single FIRMS NRT hook.
- Operational hardening, authentication, alerting, or mobile delivery.
- The dashboard visualization layer. Treated as a separate project concern.

### 1.3 What "done" means

A working repository containing: the four trained model cells, the preprocessing pipeline, the ablation table as described in [§5](#5-evaluation), per-fire slice analysis, at least three documented failure modes with visualizations, and a methodology README. Any cell of the ablation table that is missing or marked `—` is an explicit non-completion.

---

## 2. Data architecture

All training and evaluation data is archival. The live FIRMS hook exists only to point the inference path at a current fire; it is never used for training or validation.

### 2.1 Sources

| Source | Purpose | Native resolution | Notes |
|---|---|---|---|
| NDWS (Next Day Wildfire Spread) | Bulk pretrain | ~375m (MODIS) | Public TFRecord dataset, ~18k CA-heavy samples |
| LANDFIRE 2022 | Fine-scale fuel raster | 30m | Fuel model, canopy cover, canopy height |
| USGS 3DEP DEM | Terrain | 10m → resampled | Source of slope, aspect, elevation |
| HRRR reanalysis archive | Wind, humidity, temperature | 3km hourly | Resampled to target grid |
| NIFC historical perimeters | Ground truth | vector polygons | Rasterized to target grid |
| VIIRS/MODIS active fire archive | Ignition + hourly progression | ~375m | Source of inner fire progression series |

### 2.2 Fire splits

The training, validation, and test partitions are defined at the level of individual fires, never mixed.

**Test set (held out; touched exactly once, at the end):** Palisades 2025, Thomas 2017.

**Training set:** Woolsey, Bobcat, Bond, Getty, Saddleridge, Sand, Springs, Holy, Mountain. All SoCal Santa Ana-driven events from 2015–2024.

**Inner validation:** Saddleridge 2019 is held out of training and used as the inner-validation fire for checkpoint selection across every ablation cell and seed. The remaining eight fires form the fine-tune training pool. Leave-one-out cross-validation is not used; it would multiply the training budget by 9 without materially changing the held-out test numbers.

### 2.3 Holdout discipline

1. **Spatial holdout.** Any tile whose centroid falls within 5 km of the eventual Palisades or Thomas perimeter is dropped from training, even if it belongs to a different event.
2. **Temporal holdout.** Any training sample timestamped on or after the held-out fire's ignition is dropped.
3. **NDWS screening.** NDWS samples that intersect the held-out fire bounding boxes are removed from the NDWS pretrain set.

These rules apply uniformly across all four ablation cells.

### 2.4 Sample format

Each training sample is a stack of rasters at a chosen target resolution (30m for the fine head, 375m for the coarse head), cropped to a 512×512 tile centered on the active fire front.

| Channel | Contents |
|---|---|
| 0 | Current fire mask (binary) |
| 1 | Fuel model (categorical, one-hot or learned embedding) |
| 2 | Canopy cover |
| 3 | Elevation |
| 4 | Slope |
| 5–6 | Aspect (sin, cos split) |
| 7 | Wind u |
| 8 | Wind v |
| 9 | Relative humidity |
| 10 | Temperature |
| 11 | Days-since-rain proxy |

Target: binary fire mask at the next timestep.

### 2.5 Pipeline shape

```
raw archives ──▶ feature extraction ──▶ tensor cache (.npz shards)
                      │                         │
                      ▼                         ▼
               spatial alignment           torch Dataset
               (EPSG:3857 reproject)       (train/val/test split)
```

Preprocessing runs once into an on-disk cache. Training reads only from the cache, never from the raw archives. This decouples preprocessing-era debugging from training-era iteration.

### 2.6 Live inference hook

NASA FIRMS NRT (VIIRS hotspot feed) is polled on demand to produce a current fire ignition point and a bounding box. This path is inference-only; it does not mutate the tensor cache and is not used by any training or validation run.

---

## 3. Model architecture

### 3.1 Backbone

A ResU-Net: a U-Net with residual blocks in encoder and decoder. Chosen for its well-understood training dynamics, single-GPU feasibility, and direct comparability to published NDWS baselines.

| Stage | Blocks | Channels |
|---|---|---|
| Encoder | 4 downsampling stages, residual | 64 → 128 → 256 → 512 |
| Bottleneck | 2 residual blocks | 512 |
| Decoder | 4 upsampling stages, skip connections | 512 → 256 → 128 → 64 |

Input: 12 channels as specified in [§2.4](#24-sample-format). Output: 1 channel, sigmoid-activated, interpreted as next-timestep fire probability per pixel. 2D spatial dropout with `p=0.2` is applied after each residual block and is required for MC Dropout inference ([§3.4](#34-inference-time-uncertainty)).

### 3.2 Cross-scale routing

Cross-scale operates as two training runs of the same backbone, not a single joint architecture. At inference, a router selects between them based on the current predicted fire area.

| Head | Target resolution | Training samples | Active when |
|---|---|---|---|
| Fine | 30m (LANDFIRE-native) | SoCal fine-tune fires only, 512×512 tiles | Predicted fire area < 5 km² |
| Coarse | 375m (MODIS-native) | NDWS pretrain + SoCal fires resampled to 375m | Predicted fire area ≥ 5 km² |

The 5 km² handoff threshold is a hyperparameter swept during ablation ([§4.4](#44-sweeps)).

### 3.3 Physics-informed loss

The physics-informed variant adds a level-set residual term to the training loss. The fire boundary is treated as the zero level-set of a signed-distance function φ, and the Hamilton-Jacobi fire spread equation is penalized as a soft constraint.

```
total_loss = λ_data · BCE(pred, truth)
           + λ_phys · mean( (∂φ/∂t + F · |∇φ|)² )
```

- φ is the signed distance transform of the predicted fire mask.
- `F` is a per-pixel spread-rate field derived from a simplified Rothermel model using the fuel, slope, and wind channels already in the input stack.
- `∂φ/∂t` is approximated by finite differences between the input fire mask and the prediction.
- `|∇φ|` is computed via a Sobel filter.
- Initial weights: `λ_data = 1.0`, `λ_phys = 0.1`. Swept in [§4.4](#44-sweeps).

This is "physics-informed" in the literal published sense — a PDE residual in the loss. The backbone is still a fully trainable CNN. There is no collocation sampling, no coordinate MLP, and no PINN-specific training pathology.

### 3.4 Inference-time uncertainty

Dropout layers remain active at inference. Each prediction runs `N = 20` forward passes. The per-pixel mean becomes the predicted perimeter at a 0.5 threshold; the per-pixel variance becomes the uncertainty field consumed by downstream visualization.

---

## 4. Training & ablation protocol

### 4.1 Environment

Single consumer GPU, 16–24GB VRAM (RTX 3090/4090 or Colab A100). PyTorch 2.x with Lightning for training loops, rasterio and xarray for data preparation, Weights & Biases for run tracking.

### 4.2 Per-cell pipeline

1. **Preprocess once.** Materialize the feature tensor cache ([§2.5](#25-pipeline-shape)) into `.npz` shards. Separate caches for 30m and 375m. Held-out Palisades and Thomas data is materialized into a dedicated `test/` directory inaccessible to the training loop.
2. **Pretrain on NDWS.** Coarse-head runs and the single-scale baseline pretrain for ~50 epochs, BCE loss, Adam `lr=1e-4`, cosine decay. Sanity check: target IoU on NDWS val ≥ 0.30, consistent with published NDWS results. If this threshold is not met, halt and debug the data pipeline.
3. **Fine-tune on SoCal.** Both heads fine-tune on the eight SoCal training fires (Saddleridge excluded for inner validation) for ~30 epochs at `lr=1e-5`. Holdout rules ([§2.3](#23-holdout-discipline)) enforced at the dataloader level.
4. **Physics-informed variant.** Identical schedule, identical data, loss augmented with the level-set residual term.
5. **Checkpoint selection.** The reported checkpoint for each cell is the one with the best inner-validation IoU on Saddleridge. Palisades and Thomas are touched exactly once, at the end, for the final ablation.

### 4.3 The four ablation cells

|  | Single-scale (375m) | Cross-scale (fine + coarse) |
|---|---|---|
| **Data-only loss** | A₁ | A₂ |
| **Physics-informed loss** | B₁ | B₂ |

Four cells, three seeds per cell → twelve reported models. The single-scale cells omit the fine head entirely and use only the coarse head at 375m for all samples.

### 4.4 Sweeps

| Axis | Values | Scope |
|---|---|---|
| Scale | single / cross | primary, four-cell |
| Loss | data-only / physics-informed | primary, four-cell |
| `λ_phys` | 0.01, 0.05, 0.1, 0.3 | secondary, winning physics-informed cell only |
| Handoff threshold | 2 km² / 5 km² / 10 km² | secondary, winning cross-scale cell only |
| MC Dropout N | 20 | fixed |

Total training runs: 4 primary cells × 3 seeds + ~6 secondary sweep runs ≈ 18 runs. At ~2–4 hours per run on a consumer GPU, this is 1.5–3 days of wall-clock training time.

### 4.5 Reproducibility

All runs use fixed seeds across NumPy, PyTorch, and CUDA. Dataset preprocessing is pinned to a single committed script version. Per-run hyperparameters live in YAML configs in the repo. Every W&B run URL is recorded in the final report.

---

## 5. Evaluation

### 5.1 Primary metric

Next-timestep IoU on the fire class, computed at each available ground-truth perimeter timestep for each held-out fire. Background pixels are excluded from the IoU denominator; otherwise any model trivially scores above 0.99.

### 5.2 Secondary metrics

| Metric | Why reported |
|---|---|
| Precision / recall on fire pixels | Disentangles under- from over-prediction |
| AUC-PR | Threshold-free, matches NDWS convention |
| Mean per-pixel variance (MC Dropout) | Does the model know when it is uncertain |
| Expected calibration error | Is the 0.5 threshold actually calibrated |
| Perimeter growth-rate error (km²/hr) | The number an emergency manager would quote |

### 5.3 Headline table

Reported in the README exactly as shown, with `mean ± std` across three seeds:

```
                        Palisades 2025          Thomas 2017
                        IoU    AUC-PR           IoU    AUC-PR
                        ─────────────────────────────────────
Single-scale, data      _____  ______           _____  ______
Single-scale, PI        _____  ______           _____  ______
Cross-scale, data       _____  ______           _____  ______
Cross-scale, PI         _____  ______           _____  ______
```

### 5.4 Slice analysis

The slice analysis is where the research story lives. For each slice, per-cell IoU is reported alongside the headline number.

1. **Santa Ana vs. non-Santa Ana timesteps.** Slices are assigned based on wind speed and relative humidity at each timestep. The scientific claim we are testing: the physics-informed variant outperforms the data-only variant *specifically* under Santa Ana conditions.
2. **Early fire (< 5 km²) vs. mature fire (≥ 5 km²).** Tests whether the cross-scale model outperforms the single-scale baseline *specifically* in the early-fire regime where the fine head is active.
3. **Terrain complexity.** Timesteps binned by local slope variance. Tests whether either variant handles steep chaparral canyons better.

### 5.5 Failure modes

At least three explicit "the model got this wrong" cases are documented with visualizations. Candidates: a Thomas night-spread event, a Palisades wind-shift reversal, a terrain-channeled run the model missed. Honest failure-mode reporting is what separates a credible research prototype from a demo.

### 5.6 Claims and disclaimers

IgnisCA does not claim operational readiness. It does not claim to reproduce the PolyU cross-scale PINN. It does not claim physical correctness; the physics term is a regularizer, not a forward simulation. Calibrated uncertainty is claimed only if ECE is empirically low.

---

*End of spec.*
