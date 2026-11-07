import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import binom, norm

from ibist import CIResult, rr_2x2_ci


def test_katz_matches_published_sas_example_and_direct_formula():
    # SAS/STAT 14.2 FREQ Output 41.5.3, p. 2861; shared specification.
    table = np.array([[11, 4], [2, 6]])
    result = rr_2x2_ci(table, method="katz")
    assert isinstance(result, CIResult)
    assert result.estimate == pytest.approx(2.933333333333333)
    np.testing.assert_array_equal(np.round(result.conf_int[0], 4),
                                  [0.8502, 10.1204])
    reverse_event = rr_2x2_ci(table[:, ::-1], method="katz")
    np.testing.assert_array_equal(np.round(reverse_event.conf_int[0], 4),
                                  [0.1403, 0.9009])
    variance = 1 / 11 - 1 / 15 + 1 / 2 - 1 / 8
    expected = result.estimate * np.exp(
        np.array([-1, 1]) * norm.isf(0.025) * np.sqrt(variance))
    np.testing.assert_allclose(result.conf_int[0], expected, rtol=1e-12)


@pytest.mark.parametrize("level", [0.8, 0.95, 0.99])
@pytest.mark.parametrize("alternative", ["two.sided", "less", "greater"])
def test_koopman_limits_satisfy_independent_fit(level, alternative):
    table = np.array([[38, 62], [267, 717]])
    x, n = table[:, 0], table.sum(axis=1)
    result = rr_2x2_ci(table, conf_level=level, alternative=alternative,
                       tol=1e-10)
    target = norm.isf((1 - level) /
                      (2 if alternative == "two.sided" else 1))**2
    for ratio in result.conf_int[0]:
        if ratio == 0 or not np.isfinite(ratio):
            continue

        def negative_loglik(q):
            return -np.sum(binom.logpmf(x, n, [ratio * q, q]))

        upper = min(1, 1 / ratio)
        fit = minimize_scalar(negative_loglik, bounds=(0, upper),
                               method="bounded", options={"xatol": 1e-14})
        q = min([0, fit.x, upper], key=negative_loglik)
        p = np.array([ratio * q, q])
        score = np.sum((x - n * p)**2 / (n * p * (1 - p)))
        assert score == pytest.approx(target, rel=2e-6)


def test_boundary_limits_have_closed_forms_and_no_pseudocounts():
    z2 = norm.isf(0.025)**2
    result = rr_2x2_ci([[3, 0], [5, 0]])
    np.testing.assert_allclose(result.conf_int[0],
                               [3 / (3 + z2), (5 + z2) / 5], rtol=1e-8)
    zero = rr_2x2_ci([[0, 3], [5, 0]])
    np.testing.assert_allclose(zero.conf_int[0], [0, z2 / (3 + z2)],
                               rtol=1e-8)
    empty = rr_2x2_ci([[0, 3], [0, 5]])
    assert np.isnan(empty.estimate)
    np.testing.assert_array_equal(empty.conf_int[0], [0, np.inf])
    certain = rr_2x2_ci([[10**12, 0], [10**12, 0]], tol=1e-13)
    np.testing.assert_allclose(certain.conf_int[0],
                               [10**12 / (10**12 + z2),
                                (10**12 + z2) / 10**12], atol=2e-14, rtol=0)


@pytest.mark.parametrize("method", ["koopman", "katz"])
def test_reciprocity_one_sided_bounds_and_metadata(method):
    table = np.array([[38, 62], [267, 717]])
    result = rr_2x2_ci(table, method=[method, method], conf_level=0.9)
    swapped = rr_2x2_ci(table[::-1], method=method, conf_level=0.9)
    np.testing.assert_allclose(result.conf_int[0],
                               1 / swapped.conf_int[0, ::-1], rtol=1e-8)
    lower = rr_2x2_ci(table, method=method, alternative="greater")
    upper = rr_2x2_ci(table, method=method, alternative="less")
    np.testing.assert_allclose([lower.conf_int[0, 0], upper.conf_int[0, 1]],
                               result.conf_int[0], rtol=1e-8)
    assert result.methods == (method, method)
    assert lower.conf_int[0, 1] == np.inf
    assert upper.conf_int[0, 0] == 0
    np.testing.assert_array_equal(result.sample_proportions,
                                  [0.38, 267 / 984])
    frame = result.samples
    frame.iloc[0, 0] = -1
    assert result.samples.iloc[0, 0] == 38
    stored = result.conf_int.copy()
    assert "risk ratio" in result.summary()
    np.testing.assert_array_equal(result.conf_int, stored)


def test_risk_ratio_validation():
    invalid = [[1, 2, 3, 4], [[0, 0], [2, 3]], [[0.5, 2], [3, 4]],
               [[True, False], [True, True]], [[True, 2], [3, 4]],
               [[np.inf, 2], [3, 4]]]
    for value in invalid:
        with pytest.raises(ValueError):
            rr_2x2_ci(value)
    table = [[2, 3], [4, 5]]
    for options in [dict(method=[]), dict(method="unknown"),
                    dict(alternative="unknown"), dict(conf_level=True),
                    dict(tol=0), dict(tol=True), dict(tol=np.nan)]:
        with pytest.raises(ValueError):
            rr_2x2_ci(table, **options)
    with pytest.raises(ValueError, match="positive event counts"):
        rr_2x2_ci([[0, 3], [2, 5]], method="katz")


CASES = json.loads((Path(__file__).parent / "reference" /
                    "rr_2x2_r.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_risk_ratio_r_regression(case):
    result = rr_2x2_ci(case["x"], **case["options"])
    expected = (float(case["estimate"]) if case["estimate"] not in (None, "NA")
                else np.nan)
    np.testing.assert_allclose(result.estimate, expected, equal_nan=True)
    np.testing.assert_allclose(result.conf_int,
                               np.array(case["conf_int"], dtype=float),
                               rtol=3e-8, atol=2e-8)
    np.testing.assert_allclose(result.sample_proportions,
                               case["sample_proportions"])
