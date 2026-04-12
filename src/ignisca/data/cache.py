from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

EXPECTED_CHANNELS = 12


@dataclass
class CacheShard:
    inputs: np.ndarray   # (C, H, W) float32
    target: np.ndarray   # (H, W) uint8
    metadata: Dict[str, Any]


def save_shard(path: Path, shard: CacheShard) -> None:
    if shard.inputs.ndim != 3 or shard.inputs.shape[0] != EXPECTED_CHANNELS:
        raise ValueError(f"inputs must have 12 channels, got shape {shard.inputs.shape}")
    if shard.target.ndim != 2:
        raise ValueError(f"target must be 2-D, got shape {shard.target.shape}")
    if shard.inputs.shape[1:] != shard.target.shape:
        raise ValueError(
            f"inputs spatial shape {shard.inputs.shape[1:]} != target shape {shard.target.shape}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        inputs=shard.inputs.astype(np.float32, copy=False),
        target=shard.target.astype(np.uint8, copy=False),
        metadata=np.array(json.dumps(shard.metadata)),
    )


def load_shard(path: Path) -> CacheShard:
    # np.load defaults to a safe loader since numpy 1.16.3; JSON metadata is a plain string array.
    with np.load(Path(path)) as data:
        inputs = np.asarray(data["inputs"], dtype=np.float32)
        target = np.asarray(data["target"], dtype=np.uint8)
        metadata = json.loads(str(data["metadata"]))
    return CacheShard(inputs=inputs, target=target, metadata=metadata)
