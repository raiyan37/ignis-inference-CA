from __future__ import annotations

import numpy as np

CHANNEL_NAMES = (
    "fire_mask",
    "fuel_model",
    "canopy_cover",
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "wind_u",
    "wind_v",
    "relative_humidity",
    "temperature_k",
    "days_since_rain",
)

EXPECTED_CHANNELS = len(CHANNEL_NAMES)


def assemble_feature_stack(
    *,
    fire_mask: np.ndarray,
    fuel_model: np.ndarray,
    canopy_cover: np.ndarray,
    elevation: np.ndarray,
    slope: np.ndarray,
    aspect_sin: np.ndarray,
    aspect_cos: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    relative_humidity: np.ndarray,
    temperature_k: np.ndarray,
    days_since_rain: np.ndarray,
) -> np.ndarray:
    layers = [
        fire_mask.astype(np.float32),
        fuel_model.astype(np.float32),
        canopy_cover.astype(np.float32),
        elevation.astype(np.float32),
        slope.astype(np.float32),
        aspect_sin.astype(np.float32),
        aspect_cos.astype(np.float32),
        wind_u.astype(np.float32),
        wind_v.astype(np.float32),
        relative_humidity.astype(np.float32),
        temperature_k.astype(np.float32),
        days_since_rain.astype(np.float32),
    ]
    shapes = {layer.shape for layer in layers}
    if len(shapes) != 1:
        raise ValueError(f"all channels must share shape, got {shapes}")
    return np.stack(layers, axis=0)
