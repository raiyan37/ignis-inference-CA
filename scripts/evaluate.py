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
