"""Exact difference intervals: coverage, tail equations and disconnection."""

from itertools import product

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import binom

from ibist import prop_2s_ci


def test_chan_zhang_complete_separation_has_analytic_worst_case():
    for sizes in [[3, 5], [10, 20]]:
        n1, n2 = sizes
        result = prop_2s_ci([n1, 0], sizes, method="chan-zhang")

        def probability(delta):
            q = np.clip((n1 - n2*delta)/(n1+n2), 0, 1-delta)
            return (q+delta)**n1 * (1-q)**n2

        exact = brentq(lambda d: probability(d)-0.025, 0, 1)
        assert result.conf_int[0, 0] == pytest.approx(exact, abs=2e-8)
        assert result.conf_int[0, 1] == 1


@pytest.mark.parametrize("sizes", [[1, 1], [2, 1], [2, 2]])
def test_chan_zhang_finite_sample_coverage(sizes):
    n1, n2 = sizes
    counts = np.array(list(product(range(n1+1), range(n2+1))))
    for level in [0.8, 0.95]:
        limits = np.array([prop_2s_ci(x, sizes, method="chan-zhang",
                                      conf_level=level, tol=1e-6).conf_int[0]
                           for x in counts])
        for p1 in np.linspace(0, 1, 11):
            for p2 in np.linspace(0, 1, 11):
                mass = binom.pmf(counts[:, 0], n1, p1) * binom.pmf(
                    counts[:, 1], n2, p2)
                covered = (limits[:, 0] <= p1-p2) & (p1-p2 <= limits[:, 1])
                assert mass[covered].sum() >= level - 1e-12


def independent_tail_order(delta, sizes, observed):
    """Fit each constrained likelihood independently of production code."""
    states = np.array(list(product(range(sizes[0]+1), range(sizes[1]+1))))
    scores = []
    for x in states:
        def loglik(q):
            return binom.logpmf(x, sizes, [q+delta, q]).sum()

        low, high = max(0, -delta), min(1, 1-delta)
        fit = minimize_scalar(lambda q: -loglik(q), bounds=(low, high),
                               method="bounded", options={"xatol": 1e-13})
        q = max([low, high, fit.x], key=loglik)
        p = np.array([q+delta, q])
        variance = (p*(1-p)/sizes).sum()
        scores.append((x[0]/sizes[0]-x[1]/sizes[1]-delta)/np.sqrt(variance))
    observed_index = np.flatnonzero(np.all(states == observed, axis=1))[0]
    selected = np.array(scores) >= scores[observed_index] - 1e-8
    return states, selected


def test_chan_zhang_retains_disconnected_acceptance():
    sizes, observed = np.array([9, 19]), np.array([5, 10])
    states, selected = independent_tail_order(-0.095, sizes, observed)
    mass = binom.pmf(states[:, 0], sizes[0], 0) * binom.pmf(
        states[:, 1], sizes[1], 0.095)
    # q=0.095 gives p1=0, p2=0.095 under delta=-0.095.
    assert mass[selected].sum() > 0.44
    states, selected = independent_tail_order(-0.09, sizes, observed)
    values = []
    for q in np.linspace(0.09, 1, 31):
        mass = binom.pmf(states[:, 0], sizes[0], q-0.09) * binom.pmf(
            states[:, 1], sizes[1], q)
        values.append(mass[selected].sum())
    assert max(values) < 0.44
    result = prop_2s_ci(observed, sizes, method="chan-zhang", conf_level=0.12)
    assert result.conf_int[0, 0] < -0.095


def test_chan_zhang_reversal_one_sided_and_mixed_methods():
    result = prop_2s_ci([4, 1], [6, 5], method=["mee", "chan-zhang"])
    reverse = prop_2s_ci([1, 4], [5, 6], method="chan-zhang")
    np.testing.assert_allclose(result.conf_int[1], -reverse.conf_int[0, ::-1],
                               atol=1e-8)
    lower = prop_2s_ci([4, 1], [6, 5], method="chan-zhang",
                        conf_level=0.975, side="lower")
    assert lower.conf_int[0, 0] == result.conf_int[1, 0]
    assert lower.conf_int[0, 1] == 1
    low_level = prop_2s_ci([4, 1], [6, 5], method="chan-zhang",
                            conf_level=0.2, side="lower")
    assert low_level.conf_int[0, 0] > low_level.estimate
    assert result.methods == ("mee", "chan-zhang")
    for tol in [0, 1, True, np.inf, np.nan]:
        with pytest.raises(ValueError):
            prop_2s_ci([4, 1], [6, 5], method="chan-zhang", nuisance_tol=tol)
    with pytest.raises(ValueError, match="1,000,000"):
        prop_2s_ci([1, 2], [1000, 1000], method="chan-zhang")
