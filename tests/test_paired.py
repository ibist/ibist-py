"""Paired intervals: independent formulas, finite coverage and edge cases."""

import hashlib
import json
import math
from importlib.resources import files

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import xlogy
from scipy.stats import binom, norm

from ibist import or_paired_ci, prop_paired_ci
from ibist.paired import _paired_score


@pytest.mark.parametrize("side", ["two.sided", "lower", "upper"])
def test_paired_odds_wald_and_conditional_midp(side):
    tab = [[51, 30], [10, 41]]
    level = 0.95
    tail = (1 - level) / (2 if side == "two.sided" else 1)
    result = or_paired_ci(tab, side=side, method=["wald", "midp", "wald"])
    assert result.estimate == 3
    assert result.methods == ("wald", "midp", "wald")
    z = norm.isf(tail)
    expected = 3 * np.exp(np.array([-1, 1]) * z * np.sqrt(1/30 + 1/10))
    if side == "lower":
        expected[1] = np.inf
    if side == "upper":
        expected[0] = 0
    np.testing.assert_allclose(result.conf_int[0], expected)
    for index, odds in enumerate(result.conf_int[1]):
        if not 0 < odds < np.inf:
            continue
        p = odds / (1 + odds)
        probability = (binom.sf(29, 40, p) if index == 0 else
                       binom.cdf(30, 40, p)) - 0.5 * binom.pmf(30, 40, p)
        assert probability == pytest.approx(tail, abs=1e-8)


def test_paired_odds_concordance_reversal_and_no_discordance():
    methods = ["wilson", "clopper-pearson", "midp", "blaker"]
    first = or_paired_ci([[1, 30], [10, 2]], method=methods)
    second = or_paired_ci([[100, 30], [10, 200]], method=methods)
    reverse = or_paired_ci([[2, 10], [30, 1]], method=methods)
    np.testing.assert_array_equal(first.conf_int, second.conf_int)
    np.testing.assert_allclose(first.conf_int, 1 / reverse.conf_int[:, ::-1],
                               rtol=2e-7)
    zero = or_paired_ci([[1, 0], [3, 2]], method=methods)
    assert zero.estimate == 0
    assert np.all(zero.conf_int[:, 0] == 0)
    with pytest.warns(RuntimeWarning, match="unidentified"):
        none = or_paired_ci([[1, 0], [0, 2]], method=methods)
    assert np.isnan(none.estimate)
    np.testing.assert_array_equal(none.conf_int, [[0, np.inf]] * 4)
    with pytest.raises(ValueError, match="discordant"):
        or_paired_ci([[1, 0], [0, 2]])


@pytest.mark.parametrize("side", ["two.sided", "lower", "upper"])
@pytest.mark.parametrize("level", [0.2, 0.8, 0.95])
def test_paired_wald_family_uses_paired_variance(side, level):
    b, c, n = 8, 25, 180
    result = prop_paired_ci(b, c, n, side=side, conf_level=level,
                            method=["wald", "waldcc", "agresti-min"])
    z = norm.isf((1 - level) / (2 if side == "two.sided" else 1))
    for i, name in enumerate(result.methods):
        bb, cc, nn = ((b + 0.5, c + 0.5, n + 2) if name == "agresti-min"
                      else (b, c, n))
        center = (bb - cc) / nn
        variance = (bb + cc - (bb - cc)**2 / nn) / nn**2
        margin = z * np.sqrt(variance) + (1/n if name == "waldcc" else 0)
        expected = np.clip([center - margin, center + margin], -1, 1)
        if side == "lower":
            expected[1] = 1
        if side == "upper":
            expected[0] = -1
        np.testing.assert_allclose(result.conf_int[i], expected)
    assert result.estimate == (b - c) / n


@pytest.mark.parametrize("counts", [(8, 25, 180), (7, 3, 20), (0, 0, 8),
                                    (8, 0, 8), (0, 8, 8)])
def test_paired_score_matches_independent_likelihood_fit(counts):
    b, c, n = counts
    result = prop_paired_ci(b, c, n)
    for delta in result.conf_int[0]:
        if abs(delta) == 1:
            continue
        a = n - b - c

        def negative_loglik(q):
            return -float(xlogy([b, c, a],
                                [q + delta, q, 1 - 2*q - delta]).sum())

        low, high = max(0, -delta), (1 - delta) / 2
        fit = minimize_scalar(negative_loglik, bounds=(low, high),
                               method="bounded", options={"xatol": 1e-14})
        q = min([low, high, fit.x], key=negative_loglik)
        variance = (2*q + delta - delta**2) / n
        signed = ((b - c) / n - delta) / np.sqrt(variance)
        assert abs(signed) == pytest.approx(norm.ppf(0.975), rel=2e-7)
        assert signed == pytest.approx(_paired_score(b, c, n, delta), rel=1e-6)


def test_score_boundary_closed_forms_and_low_confidence():
    z2 = norm.ppf(0.975)**2
    np.testing.assert_allclose(prop_paired_ci(8, 0, 8).conf_int[0],
                               [(8-z2)/(8+z2), 1], atol=1e-8)
    np.testing.assert_allclose(prop_paired_ci(0, 0, 8).conf_int[0],
                               [-z2/(8+z2), z2/(8+z2)], atol=1e-8)
    result = prop_paired_ci(7, 3, 20, conf_level=0.2, side="lower")
    assert result.conf_int[0, 0] > result.estimate
    assert _paired_score(7, 3, 20, result.conf_int[0, 0]) == pytest.approx(
        norm.ppf(0.2), abs=1e-7)


def test_wang_bundle_has_matching_hash_and_finite_sample_coverage():
    directory = files("ibist").joinpath("_native")
    manifest = json.loads(directory.joinpath("wang_core.json").read_text())
    content = directory.joinpath("wang_core.h").read_bytes()
    assert hashlib.md5(content).hexdigest() == manifest["md5"]
    n = 3
    states = [(b, c, n-b-c) for b in range(n+1) for c in range(n-b+1)]
    limits = np.array([prop_paired_ci(b, c, n, method="wang",
                                      precision=0.0001).conf_int[0]
                       for b, c, _ in states])
    for p10 in np.linspace(0, 1, 11):
        for p01 in np.linspace(0, 1-p10, 11):
            delta = p10 - p01
            mass = []
            for b, c, t in states:
                coefficient = math.comb(n, b) * math.comb(n-b, c)
                mass.append(coefficient * p10**b * p01**c * (1-p10-p01)**t)
            covered = (limits[:, 0] <= delta) & (delta <= limits[:, 1])
            assert np.array(mass)[covered].sum() >= 0.95 - 1e-4


@pytest.mark.parametrize("side", ["two.sided", "lower", "upper"])
def test_paired_interfaces_aliases_metadata_and_precision(side):
    aliases = {"two.sided": "two.sided", "lower": "greater", "upper": "less"}
    alias = aliases[side]
    for function, args in [(or_paired_ci, ([[51, 30], [10, 41]],)),
                            (prop_paired_ci, (8, 25, 180))]:
        result = function(*args, side=side)
        np.testing.assert_array_equal(result.conf_int,
            function(*args, alternative=alias).conf_int)
        assert result.side == side
        assert result.to_frame().attrs["side"] == side
        before = result.conf_int.copy()
        result.summary(digits=2)
        np.testing.assert_array_equal(result.conf_int, before)
        with pytest.raises(ValueError, match="conflicting"):
            function(*args, side="lower", alternative="less")


@pytest.mark.parametrize("options", [dict(b=True), dict(c=-1), dict(n=0),
                                      dict(n=1), dict(b=0.5), dict(method=[]),
                                      dict(tol=0), dict(precision=np.nan),
                                      dict(grid_one=1), dict(grid_two=True)])
def test_paired_invalid_inputs(options):
    args = dict(b=3, c=1, n=8)
    args.update(options)
    with pytest.raises(ValueError):
        prop_paired_ci(**args)
    with pytest.raises(ValueError):
        prop_paired_ci(1, 1, 2000, method="wang")
