from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List

from ignisca.training.config import TrainConfig
from ignisca.training.loop import train_one_run


ABLATION_CELLS: List[Dict[str, object]] = [
    {"name": "A1_single_data", "scale": "single", "lambda_phys": 0.0},
    {"name": "A2_cross_data", "scale": "cross", "lambda_phys": 0.0},
    {"name": "B1_single_phys", "scale": "single", "lambda_phys": 0.1},
    {"name": "B2_cross_phys", "scale": "cross", "lambda_phys": 0.1},
]

SEEDS: tuple[int, ...] = (0, 1, 2)


def _heads_for(scale: str) -> List[str]:
    if scale == "single":
        return ["coarse"]
    if scale == "cross":
        return ["fine", "coarse"]
    raise ValueError(f"unknown scale: {scale}")


def run_ablation(
    base_cfg: TrainConfig,
    cache_fine: Path,
    cache_coarse: Path,
) -> List[Dict[str, object]]:
    """Run the full 4-cell × 3-seed ablation.

    Single-scale cells train only the coarse (375m) head. Cross-scale cells
    train both a fine (30m) and a coarse (375m) head as two independent runs,
    to be selected by the inference-time router in
    ``ignisca.models.router.select_head``.

    Per design spec §4.3 this produces 18 total runs: 2 single-scale × 1 head ×
    3 seeds + 2 cross-scale × 2 heads × 3 seeds = 18.
    """
    results: List[Dict[str, object]] = []
    for cell in ABLATION_CELLS:
        for seed in SEEDS:
            for head in _heads_for(str(cell["scale"])):
                run_name = f"{cell['name']}_{head}_seed{seed}"
                cache_root = cache_fine if head == "fine" else cache_coarse
                cfg = replace(
                    base_cfg,
                    cache_root=cache_root,
                    run_name=run_name,
                    seed=seed,
                    lambda_phys=float(cell["lambda_phys"]),
                )
                metrics = train_one_run(cfg)
                results.append(
                    {
                        "run_name": run_name,
                        "cell": cell["name"],
                        "head": head,
                        "seed": seed,
                        "best_val_iou": metrics["best_val_iou"],
                        "best_epoch": metrics["best_epoch"],
                    }
                )
    return results
