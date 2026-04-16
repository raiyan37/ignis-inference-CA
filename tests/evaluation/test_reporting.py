from pathlib import Path

import pytest

from ignisca.evaluation.aggregate import AggregatedRow
from ignisca.evaluation.reporting import render_headline_table

FIXTURE = Path(__file__).parent / "fixtures" / "headline.md"


def _row(fire_id: str, cell: str, base: float) -> AggregatedRow:
    return AggregatedRow(
        fire_id=fire_id,
        cell=cell,
        metrics={
            "iou": (base, 0.020),
            "precision": (base - 0.02, 0.010),
            "recall": (base + 0.03, 0.015),
            "auc_pr": (base + 0.11, 0.008),
            "ece": (0.09, 0.003),
            "growth_rate_mae": (0.42, 0.012),
            "mean_mc_variance": (0.018, 0.002),
        },
        n_seeds=3,
    )


def test_render_headline_table_matches_golden_fixture():
    rows = [
        AggregatedRow(
            fire_id="palisades_2025", cell="A1",
            metrics={
                "iou": (0.600, 0.020),
                "precision": (0.580, 0.010),
                "recall": (0.630, 0.015),
                "auc_pr": (0.710, 0.008),
                "ece": (0.090, 0.003),
                "growth_rate_mae": (0.420, 0.012),
                "mean_mc_variance": (0.018, 0.002),
            }, n_seeds=3,
        ),
        AggregatedRow(
            fire_id="palisades_2025", cell="A2",
            metrics={
                "iou": (0.650, 0.018),
                "precision": (0.620, 0.012),
                "recall": (0.670, 0.011),
                "auc_pr": (0.740, 0.009),
                "ece": (0.080, 0.004),
                "growth_rate_mae": (0.390, 0.015),
                "mean_mc_variance": (0.017, 0.001),
            }, n_seeds=3,
        ),
        AggregatedRow(
            fire_id="thomas_2017", cell="A1",
            metrics={
                "iou": (0.550, 0.025),
                "precision": (0.530, 0.015),
                "recall": (0.560, 0.020),
                "auc_pr": (0.660, 0.012),
                "ece": (0.110, 0.005),
                "growth_rate_mae": (0.480, 0.020),
                "mean_mc_variance": (0.022, 0.003),
            }, n_seeds=3,
        ),
        AggregatedRow(
            fire_id="thomas_2017", cell="A2",
            metrics={
                "iou": (0.590, 0.022),
                "precision": (0.570, 0.014),
                "recall": (0.610, 0.018),
                "auc_pr": (0.700, 0.010),
                "ece": (0.100, 0.004),
                "growth_rate_mae": (0.450, 0.018),
                "mean_mc_variance": (0.020, 0.002),
            }, n_seeds=3,
        ),
    ]
    actual = render_headline_table(rows).rstrip() + "\n"
    expected = FIXTURE.read_text()
    assert actual == expected


def test_render_headline_table_includes_all_rows():
    rows = [_row("palisades_2025", "A1", 0.60), _row("thomas_2017", "B2", 0.65)]
    md = render_headline_table(rows)
    assert "palisades_2025" in md
    assert "thomas_2017" in md
    assert "A1" in md
    assert "B2" in md
    # Header row with every metric column present.
    for col in ("IoU", "Precision", "Recall", "AUC-PR", "ECE", "Growth MAE", "MC Var"):
        assert col in md


def test_render_headline_table_rejects_empty_rows():
    with pytest.raises(ValueError, match="empty"):
        render_headline_table([])
