from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    cache_root: Path
    out_dir: Path = field(default_factory=lambda: Path("runs"))
    run_name: str = "default"
    train_split: str = "train"
    val_split: str = "val"
    seed: int = 0
    epochs: int = 30
    batch_size: int = 4
    lr: float = 1e-5
    lambda_data: float = 1.0
    lambda_phys: float = 0.0
    dropout: float = 0.2
    base_channels: int = 64
    num_workers: int = 0
    device: str = "cuda"
