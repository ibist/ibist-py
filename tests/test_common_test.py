"""Common-OR tests checked by direct enumeration and exact tail equations."""

from itertools import product
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import hypergeom, nchypergeom_fisher

from ibist import TestResult as InferenceResult, or_common_ci, or_common_test


TABS = [[[7, 3], [2, 8]], [[4, 6], [1, 9]]]


def direct_distribution(tables, theta):
    """Enumerate full allocations, independently of the network engine."""
    supports, masses = [], []
    for raw in tables:
        table = np.array(raw)
        r1, r2 = (int(n) for n in table.sum(axis=1))
        c1 = int(table[:, 0].sum())
        support = list(range(max(0, c1 - r2), min(r1, c1) + 1))
        weights = [math.comb(r1, k) * math.comb(r2, c1 - k) * theta**k
                   for k in support]
        supports.append(support)
        masses.append(np.array(weights) / sum(weights))
    totals, probabilities, allocations = [], [], []
    for indices in product(*(range(len(s)) for s in supports)):
        values = [supports[j][i] for j, i in enumerate(indices)]
        totals.append(sum(values))
        probabilities.append(math.prod(
            masses[j][i] for j, i in enumerate(indices)))
        allocation = []
        for raw, value in zip(tables, values):
            table = np.array(raw)
            r1, r2 = (int(n) for n in table.sum(axis=1))
            c1 = int(table[:, 0].sum())
            allocation.append([[value, r1 - value],
                               [c1 - value, r2 - c1 + value]])
        allocations.append(allocation)
    return np.array(totals), np.array(probabilities), allocations


@pytest.mark.parametrize("theta", [0.25, 1, 4])
@pytest.mark.parametrize("alternative", ["two.sided", "greater", "less"])
@pytest.mark.parametrize("tables", [TABS,
    [[[8, 2], [3, 1]], [[9, 1], [4, 2]]]])
def test_common_test_matches_independent_full_enumeration(theta, alternative,
                                                          tables):
    result = or_common_test(tables, theta=theta, alternative=alternative)
    observed = sum(table[0][0] for table in tables)
    totals, mass, _ = direct_distribution(tables, theta)
    less = mass[totals <= observed].sum()
    greater = mass[totals >= observed].sum()
    expected = {"less": less, "greater": greater,
                "two.sided": min(1, 2 * min(less, greater))}[alternative]
    assert result.p_value == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert result.statistic == observed
    assert isinstance(result, InferenceResult)
    assert result.df is None
    assert result.n_strata == len(tables)
    assert result.null_value == theta
    assert result.conf_int is None
    assert result.conf_level is None
    assert result.estimate_name == "Common odds ratio"
    assert result.estimate == pytest.approx(or_common_ci(tables).estimate,
                                            rel=2e-10)


def test_two_sided_rule_is_doubled_smaller_inclusive_tail():
    tables = [[[3, 1], [1, 8]]]
    result = or_common_test(tables)
    # Support weights: 126, 336, 216, 36, 1; total weight 715.
    assert result.p_value == pytest.approx(74 / 715)
    assert or_common_test(tables, alternative="greater").p_value == (
        pytest.approx(37 / 715))
    assert or_common_test(tables, alternative="less").p_value == (
        pytest.approx(714 / 715))
    # Probability ordering would count just the two least-probable tables.
    assert result.p_value != pytest.approx(37 / 715)
    assert or_common_test([[[1, 1], [1, 1]]]).p_value == 1


@pytest.mark.parametrize("theta", [0.3, 1, 3])
def test_single_stratum_one_sided_tails_match_hypergeometric(theta):
    tables = [[[7, 27], [1, 33]]]
    lower = or_common_test(tables, theta=theta, alternative="less")
    upper = or_common_test(tables, theta=theta, alternative="greater")
    distribution = nchypergeom_fisher(68, 34, 8, theta)
    assert lower.p_value == pytest.approx(distribution.cdf(7), abs=1e-12)
    assert upper.p_value == pytest.approx(distribution.sf(6), abs=1e-12)
    if theta == 1:
        assert upper.p_value == pytest.approx(hypergeom.sf(6, 68, 34, 8))
    assert lower.p_value + upper.p_value == pytest.approx(
        1 + distribution.pmf(7))


def test_jung_published_example_matches_primary_paper():
    # Jung (2014), Table 2, DOI 10.1002/bimj.201300048.
    tables = [[[10, 1], [12, 1]], [[9, 0], [11, 1]], [[8, 0], [7, 3]]]
    result = or_common_test(tables, alternative="greater")
    assert round(result.p_value, 4) == 0.1563


@pytest.mark.parametrize("theta", [0.25, 1, 4])
@pytest.mark.parametrize("alternative", ["two.sided", "greater", "less"])
def test_exact_null_rejection_probability_is_bounded(theta, alternative):
    tables = [[[2, 1], [1, 2]], [[1, 2], [2, 1]]]
    _, masses, allocations = direct_distribution(tables, theta)
    pvalues = np.array([or_common_test(x, theta=theta,
                                      alternative=alternative).p_value
                        for x in allocations])
    for alpha in [0.05, 0.1, 0.25]:
        assert masses[pvalues <= alpha].sum() <= alpha + 1e-14


@pytest.mark.parametrize("side,alternative", [("lower", "greater"),
                                              ("upper", "less")])
@pytest.mark.parametrize("level", [0.8, 0.95, 0.99])
def test_test_and_interval_invert_the_same_inclusive_tails(side, alternative,
                                                          level):
    ci = or_common_ci(TABS, conf_level=level, side=side)
    theta = ci.conf_int[0, 0 if side == "lower" else 1]
    result = or_common_test(TABS, theta=theta, alternative=alternative)
    assert result.p_value == pytest.approx(1 - level, abs=1e-10)
    central = or_common_ci(TABS, conf_level=level)
    theta = central.conf_int[0, 0 if side == "lower" else 1]
    result = or_common_test(TABS, theta=theta)
    assert result.p_value == pytest.approx(1 - level, abs=1e-10)


def test_invariance_reversal_and_conditional_mle():
    result = or_common_test(TABS, theta=4, alternative="greater")
    array = np.moveaxis(np.array(TABS), 0, -1)
    array_result = or_common_test(array, theta=4, alternative="greater")
    assert array_result.statistic == result.statistic
    assert array_result.p_value == pytest.approx(result.p_value, rel=1e-12)
    assert array_result.estimate == pytest.approx(result.estimate, rel=2e-10)
    assert or_common_test(TABS[::-1], theta=4,
                          alternative="greater").p_value == pytest.approx(
                              result.p_value)
    transposed = [np.array(tab).T for tab in TABS]
    assert or_common_test(transposed, theta=4,
                          alternative="greater").p_value == pytest.approx(
                              result.p_value)
    reversed_rows = [np.array(tab)[::-1, :] for tab in TABS]
    reverse = or_common_test(reversed_rows, theta=0.25, alternative="less")
    assert reverse.p_value == pytest.approx(result.p_value)
    assert reverse.estimate == pytest.approx(1 / result.estimate)
    assert or_common_test(TABS, theta=1).estimate == pytest.approx(
        result.estimate, rel=2e-10)
    original = result.p_value
    result.summary(digits=2)
    assert result.p_value == original


@pytest.mark.parametrize("tables", [[[[0, 0], [0, 0]]],
                                    [[[5, 0], [0, 0]]]])
def test_deterministic_strata_have_unit_pvalues_and_r_estimate(tables):
    for theta in [0.25, 1, 4]:
        for alternative in ["two.sided", "greater", "less"]:
            result = or_common_test(tables, theta=theta,
                                     alternative=alternative)
            assert result.p_value == 1
            assert result.estimate == 0
    original = or_common_test(TABS)
    with_deterministic = or_common_test(TABS + tables)
    assert with_deterministic.p_value == pytest.approx(original.p_value)


def test_upper_left_count_statistic_retains_integer_precision():
    count = 2**53 - 1
    result = or_common_test([[[count, 0], [0, 0]]] * 3)
    assert isinstance(result.statistic, int)
    assert result.statistic == count * 3


@pytest.mark.parametrize("alternative", ["two.sided", "greater", "less"])
def test_tiny_tail_probabilities_are_retained(alternative):
    if alternative == "less":
        tables, theta = [[[0, 1], [1, 0]]] * 2, 1e150
        expected = (1 / (1 + theta))**2
    else:
        tables, theta = [[[1, 0], [0, 1]]] * 2, 1e-150
        expected = (theta / (1 + theta))**2
        if alternative == "two.sided":
            expected *= 2
    result = or_common_test(tables, theta=theta, alternative=alternative)
    assert result.p_value > 0
    assert result.p_value == pytest.approx(expected, rel=1e-12, abs=0)


@pytest.mark.parametrize("theta", [0, -1, True, None, "1", [1], np.nan,
                                    np.inf, 1 + 0j, 10**1000])
def test_common_null_parameter_is_validated(theta):
    with pytest.raises(ValueError, match="theta"):
        or_common_test(TABS, theta=theta)


@pytest.mark.parametrize("alternative", [None, "two-sided", "lower", True,
                                          ["greater"]])
def test_common_alternative_is_validated(alternative):
    with pytest.raises(ValueError, match="alternative"):
        or_common_test(TABS, alternative=alternative)


@pytest.mark.parametrize("tables", [[], np.ones((2, 2)), [[1, 2], [3, 4]],
                                     [[[True, 0], [0, 1]]],
                                     [[[0.5, 0], [0, 1]]],
                                     [[[1, 0], [0, -1]]]])
def test_common_tables_are_validated(tables):
    with pytest.raises(ValueError):
        or_common_test(tables)
    with pytest.raises(TypeError):
        or_common_test(TABS, side="lower")


CASES = json.loads((Path(__file__).parent / "reference" /
                    "common_test_r.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_common_test_r_reference(case):
    result = or_common_test(case["x"], theta=case["theta"],
                            alternative=case["alternative"])
    assert result.statistic == case["statistic"]
    assert result.p_value == pytest.approx(case["p_value"], rel=2e-10, abs=0)
    assert result.null_value == case["null_value"] == case["parameter"]
    assert result.n_strata == case["n_strata"]
    assert result.method == case["method"]
    assert result.df is None
    assert result.estimate == pytest.approx(float(case["estimate"]),
                                            rel=2e-9, abs=1e-10)
