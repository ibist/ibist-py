"""Common odds-ratio limits checked by direct fixed-margin enumeration."""

from itertools import product
import math

import numpy as np
import pytest
from scipy.stats import nchypergeom_fisher

from ibist import or_common_ci


TABS = [[[7, 3], [2, 8]], [[4, 6], [1, 9]]]


def direct_distribution(tables, theta):
    supports, masses = [], []
    for raw in tables:
        tab = np.array(raw)
        r1, r2 = tab.sum(axis=1)
        c1 = tab[:, 0].sum()
        support = list(range(max(0, c1-r2), min(r1, c1)+1))
        weights = [math.comb(r1, k) * math.comb(r2, c1-k) * theta**k
                   for k in support]
        supports.append(support)
        masses.append(np.array(weights)/sum(weights))
    totals, probabilities = [], []
    for indices in product(*(range(len(s)) for s in supports)):
        totals.append(sum(supports[j][i] for j, i in enumerate(indices)))
        probabilities.append(math.prod(
            masses[j][i] for j, i in enumerate(indices)))
    return np.array(totals), np.array(probabilities)


@pytest.mark.parametrize("algorithm", ["network", "convolution"])
@pytest.mark.parametrize("side", ["two.sided", "lower", "upper"])
@pytest.mark.parametrize("level", [0.2, 0.8, 0.95])
def test_common_exact_tail_and_mle_equations(algorithm, side, level):
    result = or_common_ci(TABS, algorithm=algorithm, side=side,
                           conf_level=level)
    tail = (1-level) / (2 if side == "two.sided" else 1)
    observed = sum(t[0][0] for t in TABS)
    for index, theta in enumerate(result.conf_int[0]):
        if not 0 < theta < np.inf:
            continue
        totals, mass = direct_distribution(TABS, theta)
        select = totals >= observed if index == 0 else totals <= observed
        assert mass[select].sum() == pytest.approx(tail, abs=2e-10)
    totals, mass = direct_distribution(TABS, result.estimate)
    assert totals @ mass == pytest.approx(observed, abs=1e-9)
    assert result.methods == ("exact",)


def test_common_single_stratum_matches_conditional_hypergeometric():
    tab = [[7, 27], [1, 33]]
    result = or_common_ci([tab])
    lower, upper = result.conf_int[0]
    assert nchypergeom_fisher.sf(6, 68, 34, 8, lower) == pytest.approx(0.025)
    assert nchypergeom_fisher.cdf(7, 68, 34, 8, upper) == pytest.approx(0.025)
    mean = nchypergeom_fisher.mean(68, 34, 8, result.estimate)
    assert mean == pytest.approx(7)


def test_common_strata_invariance_boundaries_and_empty_margins():
    result = or_common_ci(TABS)
    array = np.moveaxis(np.array(TABS), 0, -1)
    np.testing.assert_allclose(result.conf_int, or_common_ci(array).conf_int)
    permuted = or_common_ci(TABS[::-1])
    np.testing.assert_allclose(result.conf_int, permuted.conf_int)
    reversed_tabs = [np.array(tab)[::-1, :] for tab in TABS]
    reverse = or_common_ci(reversed_tabs)
    np.testing.assert_allclose(result.conf_int, 1/reverse.conf_int[:, ::-1])
    deterministic = TABS + [[[0, 0], [7, 8]]]
    deterministic_ci = or_common_ci(deterministic).conf_int
    np.testing.assert_allclose(result.conf_int, deterministic_ci)
    none = or_common_ci([[[0, 0], [0, 0]]])
    assert none.estimate == 0
    np.testing.assert_array_equal(none.conf_int, [[0, np.inf]])
    zero = or_common_ci([[[0, 3], [2, 0]]])
    assert zero.estimate == 0
    assert zero.conf_int[0, 0] == 0
    infinity = or_common_ci([[[3, 0], [0, 2]]])
    assert infinity.estimate == np.inf
    assert infinity.conf_int[0, 1] == np.inf


@pytest.mark.parametrize("value", [[], [[1, 2], [3, 4]], np.ones((2, 2)),
                                    [[[1, 2], [3, -1]]], [[[True, 2], [3, 4]]]])
def test_common_rejects_invalid_tables(value):
    with pytest.raises(ValueError):
        or_common_ci(value)


def test_common_aliases_and_invalid_settings():
    result = or_common_ci(TABS, side="lower")
    assert result.side == "lower"
    np.testing.assert_array_equal(result.conf_int,
        or_common_ci(TABS, alternative="greater").conf_int)
    for options in [dict(algorithm="unknown"), dict(conf_level=True),
                    dict(side="greater"),
                    dict(side="lower", alternative="less")]:
        with pytest.raises(ValueError):
            or_common_ci(TABS, **options)
