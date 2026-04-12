from unittest.mock import patch

from ignisca.data.sources.firms_nrt import FirmsClient


def test_firms_client_parses_csv_response():
    fake_csv = (
        "latitude,longitude,acq_date,acq_time,confidence\n"
        "34.06,-118.55,2026-04-11,1845,nominal\n"
        "34.08,-118.50,2026-04-11,1850,high\n"
    )

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = fake_csv

        client = FirmsClient(map_key="FAKEKEY")
        df = client.get_hotspots(
            bbox=(-119.0, 33.5, -118.0, 34.5),
            days_back=1,
            source="VIIRS_SNPP_NRT",
        )

    assert len(df) == 2
    assert list(df["confidence"]) == ["nominal", "high"]
    assert (df["latitude"] == 34.06).any()
    mock_get.assert_called_once()
