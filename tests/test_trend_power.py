"""Focused statistical and cross-language checks for trend and power APIs."""

from itertools import product
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from ibist import PowerResult, power_p1s_test, power_p2s_test, trend_test


def test_trend_normal_statistic_and_empty_row_scores():
    table = np.array([[8, 12], [0, 0], [16, 4]])
    scores = np.arange(1, 4)
    totals, events = table.sum(axis=1), table[:, 0]
    p = events.sum() / table.sum()
    center = totals @ scores / table.sum()
    expected = sum(scores * (events - totals * p)) / math.sqrt(
        p * (1 - p) * sum(totals * (scores - center)**2))
    result = trend_test(table)
    assert result.statistic == pytest.approx(expected)
    assert result.chisq == pytest.approx(expected**2)
    assert result.score_statistic == scores @ events
    assert result.p_value == pytest.approx(2 * norm.sf(abs(expected)),
                                           rel=1e-12, abs=0)
    assert result.estimate is None and result.conf_int is None


def test_exact_trend_matches_direct_combinatorial_enumeration():
    table = [[2, 0], [1, 1], [0, 2]]
    scores = [0, 1, 3]
    allocations = [k for k in product(range(3), repeat=3) if sum(k) == 3]
    masses = [math.prod(math.comb(2, value) for value in k) / math.comb(6, 3)
              for k in allocations]
    observed = 1
    lower = sum(p for p, k in zip(masses, allocations)
                if sum(s * v for s, v in zip(scores, k)) <= observed)
    upper = sum(p for p, k in zip(masses, allocations)
                if sum(s * v for s, v in zip(scores, k)) >= observed)
    for alt, expected in [("less", lower), ("greater", upper),
                          ("two.sided", min(1, 2 * min(lower, upper)))]:
        result = trend_test(table, scores=scores, alternative=alt, exact=True)
        assert result.p_value == pytest.approx(expected, rel=1e-12, abs=0)
        shifted = trend_test(table, scores=[0.1 + 0.2 * s for s in scores],
                              alternative=alt, exact=True)
        assert shifted.p_value == pytest.approx(result.p_value)


def test_normal_power_designs_match_closed_form_sample_sizes():
    target, alpha, p0, p1 = 0.8, 0.05, 0.1, 0.25
    expected = ((norm.isf(alpha) * math.sqrt(p0 * (1 - p0)) +
                 norm.ppf(target) * math.sqrt(p1 * (1 - p1))) / (p1 - p0))**2
    result = power_p1s_test(p0=p0, p1=p1, power=target, alternative="greater")
    assert isinstance(result, PowerResult)
    assert result.n == pytest.approx(expected, rel=1e-9)
    a, b = 0.3, 0.5
    pooled = (a + b) / 2
    expected = ((norm.isf(alpha / 2) * math.sqrt(2 * pooled * (1 - pooled)) +
                 norm.ppf(target) * math.sqrt(a * (1 - a) + b * (1 - b))) /
                (b - a))**2
    assert power_p2s_test(p1=a, p2=b, power=target).n == pytest.approx(
        expected, abs=2e-4)
    assert power_p1s_test(n=10.1, p0=p0, p1=p1).n == 11
    assert power_p2s_test(n=10.1, p1=a, p2=b).n == 10.1


def test_exact_power_uses_full_null_and_alternative_rejection_probabilities():
    n, p0, p1, alpha = 12, 0.3, 0.6, 0.05
    x = np.arange(n + 1)
    pmf0 = np.array([math.comb(n, k) * p0**k * (1 - p0)**(n - k) for k in x])
    pmf1 = np.array([math.comb(n, k) * p1**k * (1 - p1)**(n - k) for k in x])
    cdf, sf = pmf0.cumsum(), pmf0[::-1].cumsum()[::-1]
    low = next(k for k in x if cdf[k] >= alpha / 2) - 1
    high = next(k for k in x if cdf[k] >= 1 - alpha / 2) + 1
    quantile = ((x <= low) | (x >= high)).astype(float)
    randomized = ((x < low) | (x > high)).astype(float)
    randomized += 0.5 * ((x == low) | (x == high))
    rules = {"quantile": quantile, "midp-rand": randomized,
             "cp": (2 * np.minimum(cdf, sf) <= alpha).astype(float),
             "midp": (2 * np.minimum(cdf - pmf0 / 2, sf - pmf0 / 2) <= alpha)}
    for method, reject in rules.items():
        result = power_p1s_test(n=n, p0=p0, p1=p1, exact=True,
                                 exact_method=method)
        assert result.power == pytest.approx(pmf1 @ reject, rel=1e-12)
        assert result.achieved_sig_level == pytest.approx(pmf0 @ reject,
                                                          rel=1e-12, abs=0)


def test_exact_sample_size_starts_at_one_and_never_exceeds_its_bound():
    result = power_p1s_test(p0=.01, p1=.99, power=.8, alternative="greater",
                             exact=True, size_rule="minimal", max_n=100)
    assert result.n == 1
    assert result.power == pytest.approx(.99)
    with pytest.raises(ValueError, match="max_n"):
        power_p1s_test(p0=.5, p1=.99, power=.5, alternative="greater",
                        exact=True, size_rule="minimal", max_n=4)


def test_invalid_inputs_and_power_result_display():
    for kwargs in [dict(n=20, p0=.1, p1=.2, power=.8),
                   dict(n=20, p0=-.1, p1=.2),
                   dict(n=20, p0=.1, p1=.2, exact_method="bad"),
                   dict(n=20, p0=.1, p1=.2, correct=None)]:
        with pytest.raises(ValueError):
            power_p1s_test(**kwargs)
    with pytest.raises(ValueError):
        power_p2s_test(n=20, p1=.1, p2=.2, group_rate=0)
    with pytest.raises(ValueError):
        trend_test([[1, 2], [3, 4]], scores=[2, 1])
    with pytest.raises(ValueError):
        trend_test([[1, 0], [2, 0]])
    result = power_p1s_test(n=50, p0=.1, p1=.25)
    before = result.power
    assert str(result) == repr(result)
    result.summary(digits=1)
    assert result.power == before


CASES = json.loads((Path(__file__).parent / "reference" /
                    "trend_power_r.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_representative_r_reference(case):
    function = {"trend": trend_test, "one": power_p1s_test,
                "two": power_p2s_test}[case["kind"]]
    result = function(**case["args"])
    for name, expected in case["result"].items():
        actual = getattr(result, name)
        if isinstance(expected, str):
            assert actual == expected
        else:
            parameters = ("n", "p0", "p1", "p2", "sig_level")
            atol = (2e-4 if case["kind"] == "two" else 2e-8) if (
                name in parameters) else 1e-11
            if name == "achieved_sig_level" and case["args"].get("p0") is None:
                # A slightly different fitted p0 changes the derived null size.
                atol = 5e-8
            assert actual == pytest.approx(expected, rel=2e-8, abs=atol)
