from __future__ import annotations


def select_head(current_fire_area_km2: float, threshold_km2: float = 5.0) -> str:
    """Cross-scale inference router.

    Returns ``"fine"`` for small fires (use the 30m checkpoint) or ``"coarse"``
    once the predicted burn area reaches the handoff threshold (use the 375m
    checkpoint). The threshold defaults to 5 km² per design spec §3.2 and is
    swept in the secondary ablation grid.
    """
    if current_fire_area_km2 < threshold_km2:
        return "fine"
    return "coarse"
