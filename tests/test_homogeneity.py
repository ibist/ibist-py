import warnings

import numpy as np
import pytest

from ibist import or_homogeneity_test


def test_common_odds_ratio_one_has_analytic_statistic():
    tabs = [[[30, 20], [20, 30]], [[20, 30], [30, 20]]]
    for correct in (False, True):
        result = or_homogeneity_test(tabs, method="breslow-day", correct=correct)
        assert result.statistic == pytest.approx(8)
        assert result.df == 1
    result = or_homogeneity_test(tabs, method="woolf")
    assert result.statistic == pytest.approx(12 * np.log(2.25) ** 2)


def test_woolf_zero_adjustment_affects_all_strata():
    tabs = np.array([[[0, 10], [20, 30]], [[5, 15], [25, 35]]], dtype=float)
    adjusted = tabs + 0.5
    logs = np.log(adjusted[:, 0, 0] * adjusted[:, 1, 1]
                   / (adjusted[:, 0, 1] * adjusted[:, 1, 0]))
    variances = np.sum(1 / adjusted, axis=(1, 2))
    expected = (logs[0] - logs[1]) ** 2 / variances.sum()
    with pytest.warns(RuntimeWarning, match="approximation"):
        result = or_homogeneity_test(list(tabs.astype(int)), method="woolf")
    assert result.statistic == pytest.approx(expected)


@pytest.mark.parametrize("method", ["breslow-day", "woolf", "zelen"])
def test_stratum_order_orientation_and_array_input(method):
    tabs = np.array([[[8, 9], [4, 12]], [[6, 10], [5, 11]]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = or_homogeneity_test(list(tabs), method=method)
        for variant in (list(tabs[::-1]), list(tabs.transpose(0, 2, 1)),
                        list(tabs[:, ::-1]), np.moveaxis(tabs, 0, 2)):
            actual = or_homogeneity_test(variant, method=method)
            if method != "zelen":
                assert actual.statistic == pytest.approx(result.statistic)
            assert actual.p_value == pytest.approx(result.p_value)


def test_zelen_matches_published_statxact_example():
    tabs = [[[8, 1], [11, 1]], [[14, 2], [18, 0]],
            [[25, 3], [42, 2]], [[39, 3], [22, 6]]]
    # Table 10.5, Applied Nonparametric Statistical Methods, third edition.
    result = or_homogeneity_test(tabs)
    assert round(result.p_value, 4) == 0.0689
    assert result.df is None and result.n_strata == 4
    assert or_homogeneity_test([tabs[0]]).p_value == 1
    assert or_homogeneity_test([[[0, 0], [0, 0]]]).p_value == 1


def test_zelen_conditional_weights_by_hand():
    tabs = [[[2, 3], [1, 4]], [[1, 2], [3, 2]]]
    # With total upper-left count 3, allocations are (0,3),(1,2),(2,1),(3,0).
    from math import comb
    weights = np.array([comb(5, a) * comb(5, 3 - a)
                         * comb(3, 3 - a) * comb(5, 1 + a) for a in range(4)])
    expected = weights[weights <= weights[2]].sum() / weights.sum()
    assert or_homogeneity_test(tabs).p_value == pytest.approx(expected)


def test_homogeneity_invalid_inputs_methods_and_resource_limits():
    tab = [[10, 20], [30, 40]]
    for method in ("breslow-day", "woolf"):
        with pytest.raises(ValueError, match="at least two"):
            or_homogeneity_test([tab], method=method)
        with pytest.raises(ValueError, match="positive"):
            or_homogeneity_test([tab, [[0, 0], [1, 2]]], method=method)
    for method in ("zelen", "woolf"):
        with pytest.raises(ValueError, match="only for Breslow-Day"):
            or_homogeneity_test([tab, tab], method=method, correct=True)
    for bad in ([], tab, np.ones((3, 2, 2)), [[[-1, 1], [2, 3]]]):
        with pytest.raises(ValueError):
            or_homogeneity_test(bad)
    with pytest.raises(ValueError, match="finite positive"):
        or_homogeneity_test([[[0, 10], [20, 30]], [[0, 20], [30, 40]]],
                             method="breslow-day")
    with pytest.raises(ValueError, match="max_tables"):
        or_homogeneity_test([tab, tab], max_tables=1)



def test_zelen_uses_relative_weights_for_huge_margins():
    n = 10 ** 12
    tabs = [[[1, 0], [n, 2 * n]], [[0, 1], [2 * n, n]]]
    # Conditional on total one, compare allocations (1,0) and (0,1).
    expected = (n + 1) ** 2 / ((n + 1) ** 2 + 4 * n ** 2)
    assert or_homogeneity_test(tabs).p_value == pytest.approx(expected, abs=1e-12)
