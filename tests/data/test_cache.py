from pathlib import Path

import numpy as np

from ignisca.data.cache import CacheShard, load_shard, save_shard


def test_round_trip(tmp_path: Path):
    shard = CacheShard(
        inputs=np.random.rand(12, 32, 32).astype(np.float32),
        target=(np.random.rand(32, 32) > 0.5).astype(np.uint8),
        metadata={
            "fire_name": "palisades_2025",
            "timestamp_utc": "2025-01-07T19:00:00",
            "resolution_m": 30.0,
            "bounds": [-118.56, 34.05, -118.50, 34.10],
        },
    )

    path = tmp_path / "shard_0001.npz"
    save_shard(path, shard)
    assert path.exists()

    loaded = load_shard(path)
    np.testing.assert_array_equal(loaded.inputs, shard.inputs)
    np.testing.assert_array_equal(loaded.target, shard.target)
    assert loaded.metadata == shard.metadata


def test_save_validates_input_shapes(tmp_path: Path):
    import pytest
    bad = CacheShard(
        inputs=np.zeros((11, 32, 32), dtype=np.float32),  # wrong channel count
        target=np.zeros((32, 32), dtype=np.uint8),
        metadata={},
    )
    with pytest.raises(ValueError, match="12 channels"):
        save_shard(tmp_path / "bad.npz", bad)
