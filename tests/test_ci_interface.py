"""Shared CI arguments and result metadata, plus R numerical comparisons."""

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm, poisson

import ibist


FUNCTIONS = {
    "prop_1s_ci": dict(x=5, n=40),
    "prop_2s_ci": dict(x=[12, 4], n=[40, 40]),
    "rr_2x2_ci": dict(x=[[7, 27], [1, 33]]),
    "or_2x2_ci": dict(x=[[7, 27], [1, 33]]),
    "rate_1s_ci": dict(x=5, exposure=10),
    "rate_2s_ci": dict(x=[9, 12], exposure=[2, 7]),
    "or_paired_ci": dict(x=[[51, 30], [10, 41]]),
    "prop_paired_ci": dict(b=8, c=25, n=180),
    "or_common_ci": dict(x=[[[7, 3], [2, 8]], [[4, 6], [1, 9]]]),
}


@pytest.mark.parametrize("name,args", FUNCTIONS.items())
def test_shared_ci_interface(name, args):
    function = getattr(ibist, name)
    signature = inspect.signature(function)
    assert signature.parameters["conf_level"].default == 0.95
    assert signature.parameters["side"].default is None
    assert signature.parameters["alternative"].default is None
    assert function(**args).side == "two.sided"
    for side, alias in [("two.sided", "two.sided"), ("lower", "greater"),
                        ("upper", "less")]:
        result = function(**args, side=side)
        assert result.side == side
        assert result.to_frame().attrs["side"] == side
        np.testing.assert_array_equal(
            result.conf_int, function(**args, alternative=alias).conf_int)
        np.testing.assert_array_equal(result.conf_int, function(
            **args, side=side, alternative=alias).conf_int)
    for side, alias in [("two.sided", "greater"), ("lower", "less")]:
        with pytest.raises(ValueError, match="conflicting"):
            function(**args, side=side, alternative=alias)
    for bad in ["greater", "less", "two-sided", True, [], ["lower"], np.nan]:
        with pytest.raises(ValueError, match="side"):
            function(**args, side=bad)


@pytest.mark.parametrize("level", [0.2, 0.8, 0.95, 0.99])
def test_one_sided_poisson_and_log_bounds(level):
    low = ibist.rate_1s_ci(5, 10, conf_level=level, side="lower").conf_int[0]
    high = ibist.rate_1s_ci(5, 10, conf_level=level, side="upper").conf_int[0]
    assert poisson.sf(4, low[0] * 10) == pytest.approx(1 - level)
    assert poisson.cdf(5, high[1] * 10) == pytest.approx(1 - level)
    assert low[1] == np.inf
    assert high[0] == 0
    result = ibist.or_2x2_ci([[7, 27], [1, 33]], conf_level=level,
                             side="lower")
    expected = (7 * 33 / 27) * np.exp(-norm.ppf(level) *
                                      np.sqrt(1/7 + 1/27 + 1 + 1/33))
    assert result.conf_int[0, 0] == pytest.approx(expected)


REFERENCE = json.loads((Path(__file__).parent / "reference" /
                        "ci_interface_r.json").read_text())
CASES = REFERENCE["cases"]


@pytest.mark.parametrize("case", CASES)
def test_r_ci_interface_reference(case):
    result = getattr(ibist, case["function_name"])(**case["options"])
    assert result.side == case["side"]
    assert result.conf_level == case["conf_level"]
    assert result.methods == tuple(case["methods"])
    assert result.estimate == pytest.approx(float(case["estimate"]))
    # R Cornfield uses tolerance on the OR scale; Python uses the log scale.
    tolerance = 1e-7 if case["function_name"] == "or_2x2_ci" else 3e-8
    np.testing.assert_allclose(result.conf_int,
                               np.array(case["conf_int"], dtype=float),
                               rtol=tolerance, atol=2e-8)


@pytest.mark.parametrize("name,args", [(n, a) for n, a in FUNCTIONS.items()
                                       if n != "or_common_ci"])
def test_ci_method_repetitions_match_r(name, args):
    function = getattr(ibist, name)
    method = function(**args).methods[0].split("-cc")[0]
    result = function(**args, method=[method, method], side="lower")
    assert result.methods == (method, method)
    np.testing.assert_array_equal(result.conf_int[0], result.conf_int[1])


@pytest.mark.parametrize("case", REFERENCE["defaults"])
def test_all_ci_default_metadata_matches_r(case):
    result = getattr(ibist, case["function_name"])(**case["options"])
    assert result.methods == (case["method"],)
    assert result.side == case["side"]
    assert result.conf_level == case["conf_level"]
    assert result.parameter == case["parameter"]
    assert result.null_value == case["null_value"]
