import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import xlogy
from scipy.stats import binomtest, norm

from ibist import CIResult, prop_2s_ci


METHODS = ("newcombe", "wald", "miettinen-nurminen", "mee",
           "agresti-caffo", "brown-li")


def test_counts_and_table_inputs_have_identical_orientation():
    x, n = [38, 267], [100, 984]
    result = prop_2s_ci(x, n, method=METHODS)
    table = pd.DataFrame([[38, 62], [267, 717]], index=["first", "second"])
    equivalent = prop_2s_ci(table, method=METHODS)
    assert isinstance(result, CIResult)
    assert result.estimate == 38 / 100 - 267 / 984
    assert result.null_value == 0
    np.testing.assert_array_equal(result.sample_proportions, [38 / 100, 267 / 984])
    np.testing.assert_array_equal(result.conf_int, equivalent.conf_int)
    pd.testing.assert_frame_equal(result.samples, equivalent.samples)
    assert result.methods == METHODS
    assert prop_2s_ci(x, n).methods == ("newcombe",)
    assert prop_2s_ci(x, n).inputs is None


@pytest.mark.parametrize("correct", [False, True])
@pytest.mark.parametrize("alternative", ["two.sided", "less", "greater"])
def test_newcombe_combines_separate_wilson_distances(correct, alternative):
    x, n = np.array([1366, 383]), np.array([15571, 5366])
    level = 0.95
    p = x / n
    tail = (1 - level) / (2 if alternative == "two.sided" else 1)
    z = norm.isf(tail)
    # Match each untruncated component's nominal central tail probability.
    name = "wilsoncc" if correct else "wilson"
    first = binomtest(int(x[0]), int(n[0])).proportion_ci(
        confidence_level=1 - 2 * tail, method=name
    )
    second = binomtest(int(x[1]), int(n[1])).proportion_ci(
        confidence_level=1 - 2 * tail, method=name
    )
    one, two = np.array([first.low, first.high]), np.array([second.low, second.high])
    expected = p[0] - p[1] + np.array([-1, 1]) * np.sqrt(
        (one - p[0]) ** 2 + (two[::-1] - p[1]) ** 2
    )
    if alternative == "less":
        expected[0] = -1
    if alternative == "greater":
        expected[1] = 1
    result = prop_2s_ci(x, n, correct=correct, alternative=alternative)
    np.testing.assert_allclose(result.conf_int[0], expected, rtol=1e-12)


@pytest.mark.parametrize("method", ["wald", "agresti-caffo", "brown-li"])
def test_normal_approximations_match_direct_formulas(method):
    x, n = np.array([9, 3]), np.array([10, 10])
    z = norm.isf(0.025)
    if method == "agresti-caffo":
        center, denominator = (x + 1) / (n + 2), n + 2
    elif method == "brown-li":
        center, denominator = (x + 0.5) / (n + 1), n
    else:
        center, denominator = x / n, n
    margin = z * np.sqrt(np.sum(center * (1 - center) / denominator))
    expected = np.clip(center[0] - center[1] + np.array([-margin, margin]), -1, 1)
    result = prop_2s_ci(x, n, method=method)
    np.testing.assert_allclose(result.conf_int[0], expected, rtol=1e-12)
    assert result.estimate == x[0] / n[0] - x[1] / n[1]
    if method == "wald":
        corrected = prop_2s_ci(x, n, method=method, correct=True)
        expected = np.clip(expected + np.array([-1, 1]) * np.sum(0.5 / n), -1, 1)
        np.testing.assert_allclose(corrected.conf_int[0], expected)


@pytest.mark.parametrize("method", ["mee", "miettinen-nurminen"])
@pytest.mark.parametrize("alternative", ["two.sided", "less", "greater"])
def test_score_limits_satisfy_independently_optimized_likelihood_equations(method, alternative):
    x, n = np.array([12, 4]), np.array([40, 30])
    result = prop_2s_ci(x, n, method=method, alternative=alternative, tol=1e-10)
    z = norm.isf(0.025 if alternative == "two.sided" else 0.05)
    factor = n.sum() / (n.sum() - 1) if method == "miettinen-nurminen" else 1
    for i, delta in enumerate(result.conf_int[0]):
        if delta in (-1, 1):
            continue
        # Direct likelihood maximization in p2, not the production derivative
        # or logit-based nuisance fit.
        def negative_loglik(q):
            p = np.array([q + delta, q])
            return -float(np.sum(xlogy(x, p) + xlogy(n - x, 1 - p)))

        domain = [max(0, -delta), min(1, 1 - delta)]
        fit = minimize_scalar(negative_loglik, bounds=domain, method="bounded",
                              options={"xatol": 1e-14})
        candidates = [domain[0], domain[1], fit.x]
        q = min(candidates, key=negative_loglik)
        p = np.array([q + delta, q])
        signed_score = (result.estimate - delta) / np.sqrt(
            factor * np.sum(p * (1 - p) / n)
        )
        assert signed_score == pytest.approx(z if i == 0 else -z, abs=2e-7)


@pytest.mark.parametrize("n", [[3, 5], [10 ** 12, 2 * 10 ** 12]])
def test_zero_event_score_limits_have_closed_form_and_distinct_variance_factor(n):
    z2 = norm.isf(0.025) ** 2
    for method in ("mee", "miettinen-nurminen"):
        factor = sum(n) / (sum(n) - 1) if method == "miettinen-nurminen" else 1
        expected = [-z2 * factor / (n[1] + z2 * factor),
                     z2 * factor / (n[0] + z2 * factor)]
        result = prop_2s_ci([0, 0], n, method=method)
        np.testing.assert_allclose(result.conf_int[0], expected, rtol=1e-8, atol=0)
        all_successes = prop_2s_ci(n, n, method=method)
        np.testing.assert_allclose(all_successes.conf_int[0], -np.array(expected)[::-1],
                                   rtol=1e-8, atol=0)
    result = prop_2s_ci([12, 4], [40, 30], method=["mee", "miettinen-nurminen"])
    assert result.conf_int[1, 0] < result.conf_int[0, 0]
    assert result.conf_int[1, 1] > result.conf_int[0, 1]


@pytest.mark.parametrize("method", METHODS)
def test_boundary_reversal_complement_and_one_sided_tail_levels(method):
    x_cases = ([0, 0], [0, 5], [3, 0], [3, 5], [2, 4])
    n = np.array([3, 5])
    for x in x_cases:
        x = np.array(x)
        result = prop_2s_ci(x, n, method=method, conf_level=0.9)
        reversed_result = prop_2s_ci(x[::-1], n[::-1], method=method, conf_level=0.9)
        complemented = prop_2s_ci(n - x, n, method=method, conf_level=0.9)
        np.testing.assert_allclose(result.conf_int[0], -reversed_result.conf_int[0, ::-1],
                                   atol=1e-9, rtol=1e-8)
        np.testing.assert_allclose(result.conf_int[0], -complemented.conf_int[0, ::-1],
                                   atol=1e-9, rtol=1e-8)
        lower = prop_2s_ci(x, n, method=method, alternative="greater")
        upper = prop_2s_ci(x, n, method=method, alternative="less")
        assert lower.conf_int[0, 1] == 1
        assert upper.conf_int[0, 0] == -1
        assert lower.conf_int[0, 0] == pytest.approx(result.conf_int[0, 0], abs=1e-9)
        assert upper.conf_int[0, 1] == pytest.approx(result.conf_int[0, 1], abs=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_low_confidence_one_sided_bounds_match_signed_direction(method):
    x, n = [4, 1], [6, 5]
    for alternative in ("less", "greater"):
        result = prop_2s_ci(x, n, method=method, conf_level=0.2,
                            alternative=alternative)
        if alternative == "greater":
            assert result.conf_int[0, 0] > result.estimate
        else:
            assert result.conf_int[0, 1] < result.estimate
    if method in ("mee", "miettinen-nurminen"):
        result = prop_2s_ci(x, n, method=method, conf_level=0.5, alternative="greater")
        assert result.conf_int[0, 0] == result.estimate


def test_method_order_flags_and_copied_sample_metadata():
    result = prop_2s_ci([9, 3], [10, 10], method=["brown-li", "wald", "brown-li"],
                        correct=True, alternative="greater")
    assert result.methods == ("brown-li", "wald-cc", "brown-li")
    frame = result.to_frame()
    assert frame.index.name == "method" and list(frame.columns) == ["lower", "upper"]
    assert frame.attrs["alternative"] == "greater"
    assert "Bound: lower" in result.summary()
    assert "Data: x=(9, 3), n=(10, 10)" in result.summary()
    assert "brown-li" in repr(result)
    before = result.conf_int.copy()
    samples, proportions = result.samples, result.sample_proportions
    samples.iloc[0, 0] = -1
    proportions[0] = -1
    frame.iloc[0, 0] = -1
    assert result.samples.iloc[0, 0] == 9
    assert result.sample_proportions[0] == 0.9
    np.testing.assert_array_equal(result.conf_int, before)
    assert len(pd.concat([result.to_frame(), result.to_frame()])) == 6


@pytest.mark.parametrize("x,n", [([1, 2], None), ([1, 2], 3), (1, [2, 3]),
                                  ([1, 2, 3], [4, 4, 4]), ([1, 2], [3]),
                                  ([[1, 2], [3, 4]], [3, 7]),
                                  ([[1, 2, 3], [2, 3, 4]], None),
                                  ([[0, 0], [2, 3]], None),
                                  ([1, 2], [0, 3]), ([1, 4], [3, 3]),
                                  ([True, False], [3, 3]),
                                  ([1, 2], [np.inf, 3]), ([1, 2], [2, 2 ** 53]),
                                  ([np.nan, 1], [3, 3]), ([1.5, 2], [3, 3])])
def test_invalid_counts_and_ambiguous_table_arguments(x, n):
    with pytest.raises(ValueError):
        prop_2s_ci(x, n)


@pytest.mark.parametrize("kwargs", [dict(nuisance_tol=0),
                                    dict(method="unknown"),
                                    dict(method=[]), dict(method=None),
                                    dict(method="mee", correct=True),
                                    dict(method="brown-li", correct=True),
                                    dict(correct=1), dict(tol=0), dict(tol=True),
                                    dict(tol=1), dict(conf_level=np.nan),
                                    dict(conf_level=True), dict(conf_level=1),
                                    dict(alternative="two-sided")])
def test_invalid_or_unimplemented_options_are_explicit(kwargs):
    with pytest.raises(ValueError):
        prop_2s_ci([1, 2], [3, 3], **kwargs)


def test_empty_columns_and_large_individual_sample_sizes_are_valid():
    for table in ([[0, 3], [0, 5]], [[3, 0], [5, 0]]):
        assert np.all(np.isfinite(prop_2s_ci(table, method=METHODS).conf_int))
    n = 2 ** 52 + 1
    a = prop_2s_ci([1, 2], [n, n], method="wald")
    b = prop_2s_ci([[1, n - 1], [2, n - 2]], method="wald")
    np.testing.assert_array_equal(a.conf_int, b.conf_int)


def test_r_reference_cases():
    cases = json.loads((Path(__file__).parent / "reference/prop_2s_r.json").read_text())["cases"]
    for case in cases:
        result = prop_2s_ci(case["x"], case["n"], **case["options"])
        assert result.estimate == pytest.approx(case["estimate"], abs=1e-15)
        np.testing.assert_allclose(result.conf_int, case["conf_int"], atol=2e-8, rtol=1e-10)
        np.testing.assert_allclose(result.sample_proportions, case["sample_proportions"],
                                   atol=0, rtol=1e-14)
        assert result.methods == tuple(case["methods"])
