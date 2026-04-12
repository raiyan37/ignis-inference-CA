import numpy as np

from ignisca.data.sources.ndws import NdwsRecord, adapt_ndws_record


def test_adapt_ndws_record_maps_channels():
    raw = {
        "elevation": np.full((64, 64), 100.0, dtype=np.float32),
        "sph": np.full((64, 64), 0.005, dtype=np.float32),
        "pdsi": np.full((64, 64), -1.0, dtype=np.float32),
        "NDVI": np.full((64, 64), 0.4, dtype=np.float32),
        "pr": np.full((64, 64), 0.0, dtype=np.float32),
        "tmmx": np.full((64, 64), 305.0, dtype=np.float32),
        "tmmn": np.full((64, 64), 285.0, dtype=np.float32),
        "erc": np.full((64, 64), 80.0, dtype=np.float32),
        "vs": np.full((64, 64), 6.0, dtype=np.float32),
        "th": np.full((64, 64), 270.0, dtype=np.float32),
        "PrevFireMask": np.zeros((64, 64), dtype=np.uint8),
        "FireMask": np.ones((64, 64), dtype=np.uint8),
    }
    raw["PrevFireMask"][30:34, 30:34] = 1

    rec = adapt_ndws_record(raw)

    assert isinstance(rec, NdwsRecord)
    assert rec.fire_mask.shape == (64, 64)
    assert rec.target_mask.shape == (64, 64)
    assert rec.fire_mask.sum() == 16
    assert rec.target_mask.sum() == 64 * 64
    # th=270° (wind from the west) → u=+vs, v=0
    assert np.allclose(rec.wind_u, 6.0, atol=0.01)
    assert np.allclose(rec.wind_v, 0.0, atol=0.01)
    assert np.allclose(rec.temperature_k, 305.0)
