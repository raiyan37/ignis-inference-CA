import numpy as np

from ignisca.data.features import CHANNEL_NAMES, EXPECTED_CHANNELS, assemble_feature_stack


def test_channel_names_length_is_twelve():
    assert len(CHANNEL_NAMES) == EXPECTED_CHANNELS == 12


def test_assemble_feature_stack_returns_correct_shape():
    H, W = 32, 32

    def zeros():
        return np.zeros((H, W), dtype=np.float32)

    stack = assemble_feature_stack(
        fire_mask=np.zeros((H, W), dtype=np.uint8),
        fuel_model=zeros(),
        canopy_cover=zeros(),
        elevation=zeros(),
        slope=zeros(),
        aspect_sin=zeros(),
        aspect_cos=zeros(),
        wind_u=zeros(),
        wind_v=zeros(),
        relative_humidity=zeros(),
        temperature_k=zeros(),
        days_since_rain=zeros(),
    )
    assert stack.shape == (12, H, W)
    assert stack.dtype == np.float32


def test_channel_order_matches_spec():
    H, W = 8, 8
    markers = {name: float(i + 1) for i, name in enumerate(CHANNEL_NAMES)}
    arrs = {name: np.full((H, W), v, dtype=np.float32) for name, v in markers.items()}

    stack = assemble_feature_stack(
        fire_mask=arrs["fire_mask"].astype(np.uint8),
        fuel_model=arrs["fuel_model"],
        canopy_cover=arrs["canopy_cover"],
        elevation=arrs["elevation"],
        slope=arrs["slope"],
        aspect_sin=arrs["aspect_sin"],
        aspect_cos=arrs["aspect_cos"],
        wind_u=arrs["wind_u"],
        wind_v=arrs["wind_v"],
        relative_humidity=arrs["relative_humidity"],
        temperature_k=arrs["temperature_k"],
        days_since_rain=arrs["days_since_rain"],
    )
    for i, name in enumerate(CHANNEL_NAMES):
        assert stack[i].mean() == markers[name], f"channel {i} ({name}) wrong"
