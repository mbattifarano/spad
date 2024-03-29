from spad import etl

import numpy as np


def test_read_csv(mock_gps_data):
    df = etl.read_csv(mock_gps_data)
    assert tuple(df.index.names) == ("driver_id", "shift_id", "timestamp")
    assert df.index.get_level_values("shift_id").dtype == np.int64
    assert df.activity_type.dtype == np.int64
    assert df.activity_type.min() >= 1
    assert df.activity_type.max() <= 8
    assert df.activity_confidence.min() >= 0.0
    assert df.activity_confidence.max() <= 1.0
    assert df.index.get_level_values("timestamp").dtype == np.dtype('<M8[ns]')
    assert not df.isna().any().any()
