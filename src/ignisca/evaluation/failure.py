from __future__ import annotations

import json
from pathlib import Path
from typing import List


def rank_failures(
    sample_metrics_jsonl: Path,
    k: int = 10,
    metric: str = "iou",
    mode: str = "worst",
) -> List[dict]:
    """Return the top-k samples by the given metric.

    ``mode="worst"`` sorts ascending (low IoU is worst); ``mode="best"`` sorts
    descending. Missing metric keys raise ``KeyError`` loudly rather than
    silently dropping rows.
    """
    if mode not in ("worst", "best"):
        raise ValueError(f"mode must be 'worst' or 'best', got {mode!r}")
    rows = [
        json.loads(line) for line in Path(sample_metrics_jsonl).read_text().splitlines() if line.strip()
    ]
    if not rows:
        return []
    missing = [i for i, r in enumerate(rows) if metric not in r]
    if missing:
        raise KeyError(
            f"rank_failures: {len(missing)} rows missing metric {metric!r} (first at idx {missing[0]})"
        )
    reverse = mode == "best"
    rows.sort(key=lambda r: r[metric], reverse=reverse)
    return rows[:k]


def render_failure_case(
    *,
    npz_path: Path,
    sample_idx: int,
    out_path: Path,
) -> None:
    """Render a 4-panel PNG: input mask, target, prediction mean, variance.

    Matplotlib is imported lazily so the evaluation package does not pull
    matplotlib into the core dependency graph. Tests that use this function
    are marked ``@pytest.mark.viz`` and skip cleanly when matplotlib is absent.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    with np.load(Path(npz_path)) as data:
        mean = data["mean"][sample_idx].astype(np.float32)
        variance = data["variance"][sample_idx].astype(np.float32)
        target = data["target"][sample_idx].astype(np.uint8)
        input_mask = data["input_mask"][sample_idx].astype(np.uint8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = (
        ("Input mask", input_mask, "Reds", 0.0, 1.0),
        ("Target", target, "Reds", 0.0, 1.0),
        ("Prediction (mean)", mean, "viridis", 0.0, 1.0),
        ("Uncertainty (var)", variance, "Reds", 0.0, max(variance.max(), 1e-6)),
    )
    for ax, (title, array, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
