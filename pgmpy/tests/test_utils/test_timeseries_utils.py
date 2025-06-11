import pandas as pd
import numpy as np

from pgmpy.utils.timeseries import from_sktime_to_dbn, from_dbn_to_sktime


def test_round_trip_multiindex():
    idx = pd.MultiIndex.from_product([[0, 1], range(3)], names=["instance", "time"])
    df = pd.DataFrame({"A": range(6), "B": list(range(5, -1, -1))}, index=idx)
    dbn_df = from_sktime_to_dbn(df)
    expected_cols = pd.MultiIndex.from_tuples([
        ("A", 0), ("B", 0), ("A", 1), ("B", 1), ("A", 2), ("B", 2)
    ])
    assert list(dbn_df.columns) == list(expected_cols)
    assert dbn_df.shape == (2, 6)
    df_conv = from_dbn_to_sktime(dbn_df)
    pd.testing.assert_frame_equal(df.sort_index(), df_conv.sort_index())


def test_single_instance_datetime_index():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame({"A": np.arange(4)}, index=dates)
    dbn_df = from_sktime_to_dbn(df)
    assert dbn_df.shape == (1, 4)
    df_back = from_dbn_to_sktime(dbn_df)
    expected_index = pd.MultiIndex.from_product([[0], range(4)], names=["instance", "time"])
    assert list(df_back.index) == list(expected_index)
    pd.testing.assert_series_equal(df_back["A"], pd.Series(df["A"].to_list()*1, index=expected_index))