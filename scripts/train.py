from __future__ import annotations

import argparse
from pathlib import Path

from ignisca.training.config import TrainConfig
from ignisca.training.loop import train_one_run


def main() -> None:
    parser = argparse.ArgumentParser(description="IgnisCA single training run")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda-data", type=float, default=1.0)
    parser.add_argument("--lambda-phys", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = TrainConfig(
        cache_root=args.cache_root,
        out_dir=args.out_dir,
        run_name=args.run_name,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_data=args.lambda_data,
        lambda_phys=args.lambda_phys,
        dropout=args.dropout,
        base_channels=args.base_channels,
        num_workers=args.num_workers,
        device=args.device,
    )
    result = train_one_run(cfg)
    print(
        f"Run {cfg.run_name} finished: best_val_iou={result['best_val_iou']:.4f} "
        f"at epoch {int(result['best_epoch'])}"
    )


if __name__ == "__main__":
    main()
