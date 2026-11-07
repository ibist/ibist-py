"""R comparisons supplement the independent method and coverage checks."""

import json
from pathlib import Path
import warnings

import numpy as np
import pytest

import ibist


PATH = Path(__file__).parent / "reference" / "remaining_ci_r.json"
CASES = json.loads(PATH.read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_remaining_ci_r_reference(case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = getattr(ibist, case["function_name"])(**case["options"])
    expected_estimate = case["estimate"]
    if expected_estimate == "NA":
        assert np.isnan(result.estimate)
    else:
        assert result.estimate == pytest.approx(float(expected_estimate),
                                                rel=2e-9, abs=1e-10)
    assert result.side == case["side"]
    assert result.methods == tuple(case["methods"])
    expected = np.array(case["conf_int"], dtype=float)
    for index, name in enumerate(result.methods):
        # The accurate Python constrained fit differs from R's coarse fit.
        if case["function_name"] == "prop_paired_ci" and name == "score":
            np.testing.assert_allclose(result.conf_int[index], expected[index],
                                       rtol=0, atol=3e-4)
        elif name == "wang":
            np.testing.assert_array_equal(result.conf_int[index],
                                           expected[index])
        else:
            np.testing.assert_allclose(result.conf_int[index], expected[index],
                                       rtol=2e-7, atol=2e-8)
