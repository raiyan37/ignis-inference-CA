from pathlib import Path

import numpy as np

from ignisca.data.cache import CacheShard, save_shard
from ignisca.training.ablation import ABLATION_CELLS, SEEDS, run_ablation
from ignisca.training.config import TrainConfig


def _write_tiny_cache(root: Path) -> None:
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


def test_ablation_cells_cover_the_full_grid():
    names = {cell["name"] for cell in ABLATION_CELLS}
    assert names == {"A1_single_data", "A2_cross_data", "B1_single_phys", "B2_cross_phys"}


def test_ablation_has_three_seeds():
    assert len(SEEDS) == 3
    assert set(SEEDS) == {0, 1, 2}


def test_run_ablation_produces_one_result_per_cell_and_head_and_seed(tmp_path: Path):
    """Single-scale cells train one head; cross-scale cells train two (fine + coarse).
    Expected total = (2 single × 1 head + 2 cross × 2 heads) × 3 seeds = 18 runs.
    """
    _write_tiny_cache(tmp_path)
    base = TrainConfig(
        cache_root=tmp_path,
        out_dir=tmp_path / "runs",
        epochs=1,
        batch_size=1,
        lr=1e-3,
        base_channels=8,
        device="cpu",
    )
    results = run_ablation(base, cache_fine=tmp_path, cache_coarse=tmp_path)
    assert len(results) == 18

    for r in results:
        assert {"run_name", "cell", "head", "seed", "best_val_iou"} <= r.keys()

    for r in results:
        ckpt = tmp_path / "runs" / r["run_name"] / "best.pt"
        assert ckpt.exists(), f"missing checkpoint for {r['run_name']}"
