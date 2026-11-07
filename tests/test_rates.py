import json
from pathlib import Path
import warnings

import numpy as np
import pytest
from scipy.stats import binomtest, norm, poisson

from ibist import rate_1s_ci, rate_2s_ci, rate_test


@pytest.mark.parametrize("x", [0, 1, 5, 411])
@pytest.mark.parametrize("alternative", ["two.sided", "less", "greater"])
def test_garwood_limits_invert_poisson_tails(x, alternative):
    result = rate_1s_ci(x, 10, alternative=alternative)
    tail = 0.025 if alternative == "two.sided" else 0.05
    lower, upper = result.conf_int[0] * 10
    if lower:
        assert poisson.sf(x - 1, lower) == pytest.approx(tail, abs=1e-12)
    if np.isfinite(upper):
        assert poisson.cdf(x, upper) == pytest.approx(tail, abs=1e-12)


@pytest.mark.parametrize("correct", [False, True])
def test_score_limits_satisfy_corrected_count_equations(correct):
    x, exposure, z = 5, 10, norm.isf(0.025)
    result = rate_1s_ci(x, exposure, method="score", correct=correct)
    lower, upper = result.conf_int[0] * exposure
    xl, xu = (4.5, 5.5) if correct else (5, 5)
    assert (xl - lower) / np.sqrt(lower) == pytest.approx(z)
    assert (xu - upper) / np.sqrt(upper) == pytest.approx(-z)


def test_wh_negative_limit_is_clipped_and_other_zero_rules_match_r():
    wh = rate_1s_ci(0, 10, method="wh")
    np.testing.assert_allclose(wh.conf_int[0], [0, 0.24639362099673709])
    log = rate_1s_ci(0, 10, method="log")
    wald = rate_1s_ci(0, 10, method="wald", correct=False)
    np.testing.assert_allclose(log.conf_int[0], [0, -np.log(0.025) / 10])
    np.testing.assert_array_equal(log.conf_int, wald.conf_int)
    assert rate_1s_ci(0, method="wald").conf_int[0, 1] != log.conf_int[0, 1]


@pytest.mark.parametrize("method", ["exact", "score", "wh", "wald", "log"])
def test_one_rate_scale_and_one_sided_equivalence(method):
    result = rate_1s_ci(5, 10, method=method, conf_level=0.9)
    lower = rate_1s_ci(5, 10, method=method, alternative="greater")
    upper = rate_1s_ci(5, 10, method=method, alternative="less")
    np.testing.assert_allclose([lower.conf_int[0, 0], upper.conf_int[0, 1]],
                               result.conf_int[0], rtol=1e-12)
    scaled = rate_1s_ci(5, 70, method=method, conf_level=0.9)
    np.testing.assert_allclose(scaled.conf_int, result.conf_int / 7)


@pytest.mark.parametrize("x", [[9, 12], [0, 2], [2, 0]])
@pytest.mark.parametrize("method", ["score", "exact"])
def test_rate_ratio_transforms_independent_binomial_limits(x, method):
    exposure = np.array([1817.6, 7496.3])
    result = rate_2s_ci(x, exposure, method=method)
    ci = binomtest(x[0], sum(x)).proportion_ci(
        method="wilson" if method == "score" else "exact")
    with np.errstate(divide="ignore"):
        expected = (np.array([ci.low, ci.high]) /
                    (1 - np.array([ci.low, ci.high])) *
                    exposure[1] / exposure[0])
    np.testing.assert_allclose(result.conf_int[0], expected, rtol=1e-10)


@pytest.mark.parametrize("method", ["log", "score", "exact"])
def test_rate_ratio_reciprocity_scaling_and_one_sided_limits(method):
    result = rate_2s_ci([9, 12], [2, 7], method=method, conf_level=0.9)
    reverse = rate_2s_ci([12, 9], [7, 2], method=method, conf_level=0.9)
    np.testing.assert_allclose(result.conf_int[0],
                               1 / reverse.conf_int[0, ::-1])
    scaled = rate_2s_ci([9, 12], [20, 70], method=method, conf_level=0.9)
    np.testing.assert_allclose(scaled.conf_int, result.conf_int)
    lower = rate_2s_ci([9, 12], [2, 7], method=method,
                       alternative="greater")
    upper = rate_2s_ci([9, 12], [2, 7], method=method, alternative="less")
    np.testing.assert_allclose([lower.conf_int[0, 0], upper.conf_int[0, 1]],
                               result.conf_int[0])


def test_rate_tests_match_normal_and_conditional_score_formulas():
    result = rate_test(411, 25800, r=0.0119, correct=False)
    z = (411 - 0.0119 * 25800) / np.sqrt(0.0119 * 25800)
    assert result.statistic == pytest.approx(z)
    assert result.p_value == pytest.approx(2 * norm.sf(abs(z)))
    two = rate_test([12, 5], [100, 80], correct=False)
    z = (12 - 17 * 100 / 180) / np.sqrt(17 * 100 / 180 * 80 / 180)
    assert two.statistic == pytest.approx(z**2)
    assert two.p_value == pytest.approx(2 * norm.sf(abs(z)))
    assert two.estimate == pytest.approx(1.92)
    assert "Conditional odds ratio" not in two.summary()
    assert "Rate ratio" in two.summary()
    with pytest.warns(UserWarning, match="approximation"):
        capped = rate_test([5, 5], [1, 1.01])
    assert capped.statistic == 0
    assert capped.p_value == 1


def test_rate_result_metadata_and_summary_preserve_precision():
    result = rate_2s_ci([9, 12], [2, 7], method=["log", "exact", "log"])
    assert np.ndim(result.estimate) == 0
    assert result.methods == ("log", "exact", "log")
    np.testing.assert_array_equal(result.sample_rates, [4.5, 12 / 7])
    samples = result.samples
    samples.iloc[0, 0] = -1
    assert result.samples.iloc[0, 0] == 9
    frame = result.to_frame()
    assert frame.attrs["samples"] == ((9, 2.0), (12, 7.0))
    stored = result.conf_int.copy()
    result.summary(digits=2)
    np.testing.assert_array_equal(stored, result.conf_int)
    with pytest.warns(UserWarning, match="approximation"):
        test = rate_test([9, 12], [2, 7])
    assert test.samples.columns.tolist() == ["x", "exposure"]
    np.testing.assert_array_equal(test.sample_rates, result.sample_rates)
    tiny = rate_test(1, 10**12, r=1e-12)
    assert "1.000e-12" in tiny.summary()


def test_rate_ratio_avoids_overflowing_intermediate_exposure_ratios():
    for method in ("score", "exact"):
        result = rate_2s_ci([0, 2], [1e-300, 1e300], method=method)
        assert result.estimate == 0
        np.testing.assert_array_equal(result.conf_int[0], [0, np.inf])
    result = rate_2s_ci([100, 100], [1e-310, 1e-310], method="log")
    assert result.estimate == pytest.approx(1)
    assert np.all(np.isfinite(result.conf_int))


@pytest.mark.parametrize("bad", [-1, 0.5, np.nan, np.inf, True, "2"])
def test_invalid_rate_counts_are_rejected(bad):
    with pytest.raises(ValueError):
        rate_1s_ci(bad)
    with pytest.raises(ValueError):
        rate_2s_ci([bad, 2])
    with pytest.raises(ValueError):
        rate_test(bad)


def test_rate_options_zero_counts_and_lengths_are_validated():
    for options in [dict(exposure=0), dict(conf_level=np.nan),
                    dict(correct=1), dict(method=[]),
                    dict(alternative="unknown")]:
        with pytest.raises(ValueError):
            rate_1s_ci(5, **options)
    for x, time in [([1, 2], 1), ([0, 0], [1, 1]), ([1, 2], [1, 0])]:
        with pytest.raises(ValueError):
            rate_2s_ci(x, time)
    with pytest.raises(ValueError):
        rate_2s_ci([1, 2], [True, 1])
    with pytest.raises(ValueError, match="positive event counts"):
        rate_test([0, 2], [1, 1])
    with pytest.raises(ValueError):
        rate_test(1, r=0)
    with pytest.warns(UserWarning, match="approximation"):
        rate_test([1, 1], [1, 1])


CASES = json.loads((Path(__file__).parent / "reference" /
                    "rates_r.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_rate_r_regression(case):
    function = {"rate_1s_ci": rate_1s_ci, "rate_2s_ci": rate_2s_ci,
                "rate_test": rate_test}[case["function_name"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = function(case["x"], case["exposure"], **case["options"])
    np.testing.assert_allclose(result.estimate, float(case["estimate"]))
    expected = np.array(case["conf_int"], dtype=float)
    if case.get("negative_wh_lower"):
        # Preserve raw R references; this intentional difference is reviewed.
        expected = np.maximum(expected, 0)
    actual = result.conf_int
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
    if case["function_name"] == "rate_test":
        assert result.statistic == pytest.approx(case["statistic"], abs=1e-12)
        assert result.p_value == pytest.approx(case["p_value"], abs=1e-12)
