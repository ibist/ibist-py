"""Independent numerical checks and comparisons with the R implementation."""

from itertools import product
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import fisher_exact, nchypergeom_fisher, norm
from scipy.stats.contingency import odds_ratio

from ibist import CIResult, or_2x2_ci
from ibist.odds_ratio import _ConditionalTable


def test_wald_and_gart_against_direct_formulas():
    table = np.array([[7, 27], [1, 33]])
    result = or_2x2_ci(table, method=["wald", "gart"])
    assert isinstance(result, CIResult)
    assert result.estimate == pytest.approx(7 * 33 / 27)
    for i, adjustment in enumerate([0, 0.5]):
        a, b, c, d = (table + adjustment).ravel()
        theta = a * d / (b * c)
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        offset = np.array([-1, 1]) * norm.isf(.025) * se
        expected = np.exp(math.log(theta) + offset)
        np.testing.assert_allclose(result.conf_int[i], expected, rtol=1e-13)
    assert result.methods == ("wald", "gart")


def test_baptista_pike_published_fagerland_example():
    # Fagerland (2012), Stata Journal 12, 505--514, worked mercii example.
    # Printed endpoints reflect the article's numerical inversion precision.
    exact = or_2x2_ci([[7, 27], [1, 33]], method="bp")
    midp = or_2x2_ci([[7, 27], [1, 33]], method="bp", midp=True)
    np.testing.assert_allclose(exact.conf_int[0], [1.000, 195.495], atol=.004)
    np.testing.assert_allclose(midp.conf_int[0], [1.328, 98.838], atol=.003)


def test_cornfield_against_scipy_for_all_small_nondegenerate_tables():
    # Exhaustive cells 0--3 include every possible location of zero counts.
    for cells in product(range(4), repeat=4):
        table = np.array(cells).reshape(2, 2)
        if np.any(table.sum(axis=0) == 0) or np.any(table.sum(axis=1) == 0):
            continue
        expected = odds_ratio(table).confidence_interval()
        result = or_2x2_ci(table, method="cornfield")
        np.testing.assert_allclose(result.conf_int[0], expected, rtol=2e-8,
                                   atol=1e-10, err_msg=str(cells))


@pytest.mark.parametrize("midp", [False, True])
def test_zero_cell_cornfield_limit_against_closed_form(midp):
    # For margins (2,2) and (2,2), weights are 1, 4*theta, theta**2.
    # With observed upper-left count zero, solve P(X=0) = alpha/2
    # (exact) or P(X=0)/2 = alpha/2 (mid-p).
    target = .05 if midp else .025
    expected_upper = -2 + math.sqrt(3 + 1 / target)
    result = or_2x2_ci([[0, 2], [2, 0]], method="cornfield", midp=midp)
    expected = [0, expected_upper]
    np.testing.assert_allclose(result.conf_int[0], expected, rtol=1e-8)


@pytest.mark.parametrize("midp", [False, True])
def test_sparse_cornfield_roots_and_midp_ties(midp):
    # The upper limit is precisely one for the mid-p 3/3 antidiagonal table.
    result = or_2x2_ci([[0, 3], [3, 0]], method="cornfield", midp=midp)
    if midp:
        assert result.intervals[0].upper == pytest.approx(1)
    conditional = _ConditionalTable(np.array([[0, 2], [2, 0]]))
    expected = 1 / 6 if midp else fisher_exact([[0, 2], [2, 0]]).pvalue
    assert conditional.pvalue(0, "bp", midp) == pytest.approx(expected)


def test_conditional_probabilities_against_scipy():
    table = np.array([[7, 27], [1, 33]])
    model = _ConditionalTable(table)
    for theta in [1e-6, .1, 1, 8.5, 100, 1e6]:
        expected = nchypergeom_fisher.pmf(model.support, 68, 34, 8, theta)
        probabilities = np.exp(model.log_probabilities(math.log(theta)))
        np.testing.assert_allclose(probabilities, expected,
                                   rtol=1e-11, atol=1e-15)
    expected_p = fisher_exact(table).pvalue
    assert model.pvalue(0, "bp", False) == pytest.approx(expected_p)


def test_huge_margins_with_small_support_against_exact_integer_weights():
    b = 10 ** 12
    model = _ConditionalTable(np.array([[1, b], [2, b]]))
    weights = [math.comb(b + 1, k) * math.comb(b + 2, 3 - k)
               for k in range(4)]
    for theta in [Fraction(1, 10), Fraction(1), Fraction(10)]:
        tilted = [w * theta ** k for k, w in enumerate(weights)]
        expected = np.array([float(w / sum(tilted)) for w in tilted])
        actual = np.exp(model.log_probabilities(math.log(float(theta))))
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)


def test_exact_conditional_coverage_by_enumerating_tables():
    # Fixed margins give X~noncentral hypergeometric. Sum probabilities of
    # every table whose interval covers theta; no simulated coverage estimate.
    support = np.arange(5)
    intervals = {}
    for method in ["cornfield", "bp"]:
        intervals[method] = np.array([
            or_2x2_ci([[int(x), int(4 - x)], [int(4 - x), int(1 + x)]],
                      method=method).conf_int[0] for x in support
        ])
    for theta in [.001, .01, .1, .5, 1, 2, 10, 100, 1000]:
        prob = nchypergeom_fisher.pmf(support, 9, 4, 4, theta)
        for method, ci in intervals.items():
            covered = (ci[:, 0] <= theta * (1 + 1e-7)) & (
                ci[:, 1] >= theta * (1 - 1e-7))
            assert prob[covered].sum() >= .95 - 1e-12, (method, theta)


def test_cell_rearrangements_reciprocate_limits():
    table = np.array([[7, 27], [1, 33]])
    for midp in [False, True]:
        for method in ["wald", "gart", "cornfield", "bp"]:
            result = or_2x2_ci(table, method=method, midp=midp)
            transpose = or_2x2_ci(table.T, method=method, midp=midp)
            reverse = or_2x2_ci(table[::-1], method=method, midp=midp)
            np.testing.assert_allclose(result.conf_int, transpose.conf_int,
                                       rtol=1e-7)
            reciprocal = 1 / result.conf_int[0, ::-1]
            np.testing.assert_allclose(reverse.conf_int[0], reciprocal,
                                       rtol=1e-7)


REFERENCE_PATH = Path(__file__).parent / "reference" / "or_2x2_r.json"
REFERENCE = json.loads(REFERENCE_PATH.read_text())


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case:
                         f"{case['name']}-{case['conf_level']}-{case['midp']}")
def test_matches_r_reference_cases(case):
    methods = [ci["method"].split("-")[0] for ci in case["intervals"]]
    result = or_2x2_ci(case["table"], conf_level=case["conf_level"],
                       method=methods, midp=case["midp"])
    expected = [[float(ci["lower"]), float(ci["upper"])]
                for ci in case["intervals"]]
    assert result.estimate == pytest.approx(float(case["estimate"]))
    assert result.methods == tuple(ci["method"] for ci in case["intervals"])
    # R uses an absolute tolerance in theta; Python solves in log(theta).
    np.testing.assert_allclose(result.conf_int, expected, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize("bad", [[], [1, 2, 3, 4], [[1, 2, 3], [4, 5, 6]],
                                 [[True, False], [True, True]],
                                 [["1", "2"], ["3", "4"]]])
def test_rejects_non_numeric_or_wrong_shape(bad):
    with pytest.raises(ValueError, match="2 x 2 numeric"):
        or_2x2_ci(bad)


@pytest.mark.parametrize("value", [-1, .5, np.nan, np.inf])
def test_rejects_invalid_counts(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        or_2x2_ci([[1, 2], [3, value]])


@pytest.mark.parametrize("level",
                         [0, 1, -1, np.nan, np.inf, True, [.95], "0.95"])
def test_rejects_invalid_confidence_levels(level):
    with pytest.raises(ValueError, match="conf_level"):
        or_2x2_ci([[1, 2], [3, 4]], conf_level=level)


@pytest.mark.parametrize("method",
                         ["adjusted", "unknown", [], ["wald", 2], None])
def test_rejects_invalid_methods(method):
    with pytest.raises(ValueError, match="method"):
        or_2x2_ci([[1, 2], [3, 4]], method=method)


def test_boundary_estimates_and_degenerate_tables():
    assert or_2x2_ci([[0, 2], [3, 4]], method="bp").estimate == 0
    assert math.isinf(or_2x2_ci([[1, 0], [3, 4]], method="bp").estimate)
    assert math.isnan(or_2x2_ci([[0, 0], [3, 4]], method="gart").estimate)
    with pytest.raises(ValueError, match="positive cells"):
        or_2x2_ci([[0, 2], [3, 4]])
    for method in ["cornfield", "bp"]:
        with pytest.raises(ValueError, match="positive row and column"):
            or_2x2_ci([[0, 0], [3, 4]], method=method)


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, True, "0.01"])
def test_rejects_invalid_tolerances(value):
    with pytest.raises(ValueError, match="tol"):
        or_2x2_ci([[1, 2], [3, 4]], tol=value)


@pytest.mark.parametrize("value", [None, 0, 1, "yes", [True]])
def test_rejects_invalid_midp(value):
    with pytest.raises(ValueError, match="midp"):
        or_2x2_ci([[1, 2], [3, 4]], midp=value)


def test_low_confidence_midp_and_large_wald_counts():
    ci = or_2x2_ci([[7, 27], [1, 33]], method="cornfield", midp=True,
                   conf_level=.01).conf_int[0]
    assert ci[0] < ci[1]
    result = or_2x2_ci([[100000, 200000], [300000, 400000]])
    assert result.estimate == pytest.approx(2 / 3)
    lower, upper = result.conf_int[0]
    assert lower < result.estimate < upper
