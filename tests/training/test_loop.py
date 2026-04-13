import json
from pathlib import Path

import numpy as np
import torch

from ignisca.data.cache import CacheShard, save_shard
from ignisca.training.config import TrainConfig
from ignisca.training.loop import set_seed, train_one_run


def _write_tiny_cache(root: Path) -> None:
    """Two training shards and one validation shard at 32x32, 12 channels."""
    rng = np.random.default_rng(0)
    for split, n in [("train", 2), ("val", 1)]:
        (root / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            inputs = rng.standard_normal((12, 32, 32)).astype(np.float32)
            target = np.zeros((32, 32), dtype=np.uint8)
            target[10:20, 10:20] = 1
            save_shard(
                root / split / f"{i:05d}.npz",
                CacheShard(inputs=inputs, target=target, metadata={"idx": i}),
            )


def test_set_seed_makes_torch_deterministic():
    set_seed(123)
    a = torch.randn(4)
    set_seed(123)
    b = torch.randn(4)
    assert torch.allclose(a, b)


def test_train_one_run_writes_checkpoint_and_log(tmp_path: Path):
    _write_tiny_cache(tmp_path)
    cfg = TrainConfig(
        cache_root=tmp_path,
        out_dir=tmp_path / "runs",
        run_name="smoke",
        epochs=2,
        batch_size=1,
        lr=1e-3,
        base_channels=8,
        device="cpu",
    )
    result = train_one_run(cfg)

    ckpt = tmp_path / "runs" / "smoke" / "best.pt"
    log = tmp_path / "runs" / "smoke" / "metrics.jsonl"
    assert ckpt.exists(), "best checkpoint was not written"
    assert log.exists(), "metrics log was not written"

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        entry = json.loads(line)
        assert {"epoch", "train_loss", "val_iou", "run_name"} <= entry.keys()

    assert "best_val_iou" in result
    assert "best_epoch" in result


def test_train_one_run_with_physics_loss_runs(tmp_path: Path):
    _write_tiny_cache(tmp_path)
    cfg = TrainConfig(
        cache_root=tmp_path,
        out_dir=tmp_path / "runs",
        run_name="smoke_phys",
        epochs=1,
        batch_size=1,
        lr=1e-3,
        base_channels=8,
        lambda_phys=0.1,
        device="cpu",
    )
    result = train_one_run(cfg)
    assert (tmp_path / "runs" / "smoke_phys" / "best.pt").exists()
    assert result["best_val_iou"] >= 0.0
