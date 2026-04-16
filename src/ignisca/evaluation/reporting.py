from __future__ import annotations

from typing import Sequence

from ignisca.evaluation.aggregate import AggregatedRow

_COLUMNS: tuple[tuple[str, str], ...] = (
    ("iou", "IoU"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("auc_pr", "AUC-PR"),
    ("ece", "ECE"),
    ("growth_rate_mae", "Growth MAE"),
    ("mean_mc_variance", "MC Var"),
)


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def render_headline_table(rows: Sequence[AggregatedRow]) -> str:
    """Render a list of AggregatedRows to a markdown table as a single string.

    Header format is fixed and covers every metric in ``_COLUMNS``. Rows are
    emitted in the order provided — callers are responsible for sorting if
    they want a stable fire/cell ordering.
    """
    if not rows:
        raise ValueError("render_headline_table: rows is empty")

    header_cells = ["Fire", "Cell"] + [label for _, label in _COLUMNS]
    separator_cells = ["---"] * len(header_cells)
    lines: list[str] = [
        "# IgnisCA Ablation Headline",
        "",
        "| " + " | ".join(header_cells) + " |",
        "|" + "|".join(separator_cells) + "|",
    ]
    for row in rows:
        cells = [row.fire_id, row.cell]
        for key, _ in _COLUMNS:
            mean, std = row.metrics[key]
            cells.append(_fmt(mean, std))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
