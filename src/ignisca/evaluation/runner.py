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
