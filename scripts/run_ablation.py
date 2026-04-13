from __future__ import annotations

import argparse
import json
from pathlib import Path

from ignisca.training.ablation import run_ablation
from ignisca.training.config import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="IgnisCA ablation runner")
    parser.add_argument("--cache-fine", type=Path, required=True)
    parser.add_argument("--cache-coarse", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Optional path to dump the per-run results list as JSON",
    )
    args = parser.parse_args()

    base = TrainConfig(
        cache_root=args.cache_coarse,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        base_channels=args.base_channels,
        num_workers=args.num_workers,
        device=args.device,
    )
    results = run_ablation(base, cache_fine=args.cache_fine, cache_coarse=args.cache_coarse)

    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(results, indent=2))

    print(f"Ablation finished: {len(results)} runs")
    for r in results:
        print(
            f"  {r['run_name']}: iou={r['best_val_iou']:.4f} "
            f"(cell={r['cell']}, head={r['head']}, seed={r['seed']})"
        )


if __name__ == "__main__":
    main()
