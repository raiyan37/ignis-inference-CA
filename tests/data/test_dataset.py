from pathlib import Path

import numpy as np
import pytest
import torch

from ignisca.data.cache import CacheShard, save_shard
from ignisca.data.dataset import IgnisDataset


def _write_n_shards(cache_root: Path, split: str, n: int):
    split_dir = cache_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        save_shard(
            split_dir / f"{i:05d}.npz",
            CacheShard(
                inputs=np.full((12, 16, 16), float(i), dtype=np.float32),
                target=np.full((16, 16), i % 2, dtype=np.uint8),
                metadata={"idx": i},
            ),
        )


def test_dataset_reads_all_shards(tmp_path: Path):
    _write_n_shards(tmp_path, "train", 5)
    ds = IgnisDataset(tmp_path, split="train")
    assert len(ds) == 5

    x, y = ds[2]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (12, 16, 16)
    assert y.shape == (16, 16)
    assert x.dtype == torch.float32
    assert y.dtype == torch.uint8
    assert x.mean().item() == 2.0


def test_dataset_raises_on_empty_split(tmp_path: Path):
    (tmp_path / "train").mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="no shards"):
        IgnisDataset(tmp_path, split="train")
