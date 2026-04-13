from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ignisca.data.dataset import IgnisDataset
from ignisca.models.resunet import ResUNet
from ignisca.training.config import TrainConfig
from ignisca.training.losses import IgnisLoss
from ignisca.training.metrics import fire_class_iou


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def train_one_run(cfg: TrainConfig) -> Dict[str, float]:
    """Train a single model end-to-end.

    Writes a best-IoU checkpoint to ``{out_dir}/{run_name}/best.pt`` and a
    per-epoch JSONL log to ``{out_dir}/{run_name}/metrics.jsonl``. Returns a
    dict with ``best_val_iou`` and ``best_epoch``.
    """
    set_seed(cfg.seed)
    device = _resolve_device(cfg.device)

    train_ds = IgnisDataset(cfg.cache_root, split=cfg.train_split)
    val_ds = IgnisDataset(cfg.cache_root, split=cfg.val_split)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = ResUNet(in_channels=12, base=cfg.base_channels, dropout=cfg.dropout).to(device)
    loss_fn = IgnisLoss(lambda_data=cfg.lambda_data, lambda_phys=cfg.lambda_phys).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    run_dir = Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "metrics.jsonl"
    ckpt_path = run_dir / "best.pt"
    log_path.write_text("")

    best_iou = -1.0
    best_epoch = -1

    for epoch in range(cfg.epochs):
        model.train(True)
        train_loss_sum = 0.0
        n_train_batches = 0
        for x, y in train_loader:
            x = x.to(device).float()
            y = y.to(device).float().unsqueeze(1)
            logits = model(x)
            loss = loss_fn(logits, y, features=x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            n_train_batches += 1

        model.train(False)
        val_iou_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device).float()
                y = y.to(device).float().unsqueeze(1)
                logits = model(x)
                val_iou_sum += fire_class_iou(logits, y)
                n_val_batches += 1

        train_loss = train_loss_sum / max(n_train_batches, 1)
        val_iou = val_iou_sum / max(n_val_batches, 1)

        with log_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_iou": val_iou,
                        "run_name": cfg.run_name,
                    }
                )
                + "\n"
            )

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in cfg.__dict__.items()
                    },
                },
                ckpt_path,
            )

    return {"best_val_iou": float(best_iou), "best_epoch": float(best_epoch)}
