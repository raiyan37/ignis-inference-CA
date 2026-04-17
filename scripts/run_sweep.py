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
