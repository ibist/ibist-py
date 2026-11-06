import numpy as np
import pandas as pd
import pytest

from ibist import or_2x2_ci


def test_multiple_method_table_preserves_values_and_labels():
    result = or_2x2_ci([[7, 27], [1, 33]], method=["wald", "bp"], midp=True)
    frame = result.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.index.tolist() == ["wald", "bp-midp"]
    assert frame.columns.tolist() == ["lower", "upper"]
    np.testing.assert_array_equal(frame.to_numpy(), result.conf_int)
    assert frame.attrs["estimate"] == result.estimate
    assert frame.attrs["conf_level"] == .95


def test_summary_rounds_presentation_only():
    result = or_2x2_ci([[7, 27], [1, 33]], method=["wald", "bp"])
    before = result.conf_int.copy()
    summary = result.summary()
    assert "8.556" in summary and "bp-exact" in summary
    np.testing.assert_array_equal(before, result.conf_int)
    copy = result.conf_int
    copy[0, 0] = -1
    np.testing.assert_array_equal(before, result.conf_int)


def test_tiny_limits_use_scientific_not_zero():
    result = or_2x2_ci([[1, 10000], [10000, 1]])
    assert "e-" in result.summary()
    with pytest.raises(ValueError, match="digits"):
        result.summary(-1)
