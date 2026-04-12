from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

from ignisca.data.cache import load_shard


class IgnisDataset(Dataset):
    """Serves preprocessed IgnisCA cache shards.

    Expected layout:
        cache_root/
          train/
            00000.npz
            00001.npz
            ...
          val/
          test/
    """

    def __init__(self, cache_root: Path, split: str) -> None:
        split_dir = Path(cache_root) / split
        if not split_dir.exists():
            raise RuntimeError(f"split dir does not exist: {split_dir}")
        shards = sorted(split_dir.glob("*.npz"))
        if len(shards) == 0:
            raise RuntimeError(f"no shards found in {split_dir}")
        self._shards = shards

    def __len__(self) -> int:
        return len(self._shards)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        shard = load_shard(self._shards[idx])
        inputs = torch.from_numpy(shard.inputs)
        target = torch.from_numpy(shard.target)
        return inputs, target
