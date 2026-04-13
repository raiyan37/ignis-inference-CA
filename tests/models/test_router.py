from ignisca.models.router import select_head


def test_router_picks_fine_below_threshold():
    assert select_head(current_fire_area_km2=1.0, threshold_km2=5.0) == "fine"


def test_router_picks_coarse_at_or_above_threshold():
    assert select_head(current_fire_area_km2=5.0, threshold_km2=5.0) == "coarse"
    assert select_head(current_fire_area_km2=25.0, threshold_km2=5.0) == "coarse"


def test_router_default_threshold_is_five_sqkm():
    assert select_head(current_fire_area_km2=4.9) == "fine"
    assert select_head(current_fire_area_km2=5.1) == "coarse"
