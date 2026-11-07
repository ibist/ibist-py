"""Exact and asymptotic tests of homogeneity of stratum odds ratios."""

import math
import warnings

import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2

from .contingency import TestResult, _flag, _positive_integer, _table
from .odds_ratio import _ConditionalTable


def or_homogeneity_test(x, *, method="zelen", correct=False,
                         max_tables=1000000):
    """Test whether independent 2 by 2 strata share an unspecified odds ratio.

    Parameters
    ----------
    x : list of 2 by 2 tables, or array of shape (2, 2, K)
        Nonnegative integer counts. As in R, a list is interpreted as one
        table per item; a three-dimensional NumPy array uses the last axis
        for strata. Wrap a single table in a list.
    method : {"zelen", "breslow-day", "woolf"}, default "zelen"
        Zelen's exact conditional test, or a chi-square asymptotic test.
    correct : bool, default False
        Tarone's adjustment for Breslow--Day only. This is not a continuity
        correction; it is an error to request it for Zelen or Woolf.
    max_tables : int, default 1000000
        Limit on the exact search nodes. Exceeding it raises an error;
        computation never silently switches to an approximate method.

    Returns
    -------
    TestResult
        p_value, statistic, method, and n_strata. Asymptotic methods also
        return df=K-1. Zelen's statistic is the sum of upper-left counts;
        it is not compared with a chi-square distribution.

    Notes
    -----
    The null is equality of the stratum odds ratios, not that a common
    odds ratio equals one. Zelen conditions on each stratum's margins and
    the sum of upper-left cells, removing the nuisance common odds ratio.
    It sums probabilities no larger than the observed allocation probability
    with the R implementation's absolute 1e-12 tolerance on log weights.
    The exact enumeration can be expensive for wide supports or many strata.
    A single stratum gives p=1; empty margins are allowed for the exact test.

    Breslow--Day estimates the common odds ratio by Mantel--Haenszel and
    fits each table with its observed margins. Its statistic is the sum of
    squared upper-left residuals divided by fitted variances; each variance
    is 1/sum(1/fitted cells). Tarone subtracts the squared sum of residuals
    divided by the sum of these variances. A finite positive MH estimate
    and positive margins are required, with at least two strata.

    Woolf computes inverse-variance weighted dispersion of log odds ratios.
    If any cell is zero, add 0.5 to all cells in all strata, matching R ibist
    and DescTools. This automatic adjustment is independent of correct.
    Asymptotic methods warn if a fitted (Breslow--Day) or adjusted (Woolf)
    cell is below five. Only display output is rounded.

    References
    ----------
    Breslow and Day (1980), Statistical Methods in Cancer Research, vol. I,
    IARC Scientific Publications 32.
    Tarone (1985), Biometrika 72, 91--95, doi:10.1093/biomet/72.1.91.
    Woolf (1955), Annals of Human Genetics 19, 251--253.
    Zelen (1971), Biometrika 58, 129--137, doi:10.1093/biomet/58.1.129.

    Examples
    --------
    >>> tabs = [[[30, 20], [20, 30]], [[20, 30], [30, 20]]]
    >>> or_homogeneity_test(tabs, method="breslow-day").statistic
    8.0
    """
    if not isinstance(method, str) or method not in ("zelen", "breslow-day", "woolf"):
        raise ValueError("method must be zelen, breslow-day, or woolf.")
    correct = _flag(correct, "correct")
    if correct and method != "breslow-day":
        raise ValueError("correct=True is available only for Breslow-Day.")
    max_tables = _positive_integer(max_tables, "max_tables")
    if isinstance(x, np.ndarray) and x.ndim == 3 and x.shape[:2] == (2, 2):
        items = [x[:, :, i] for i in range(x.shape[2])]
    elif isinstance(x, (list, tuple)):
        items = list(x)
    else:
        raise ValueError("x must be a list of 2 x 2 tables or a 2 x 2 x K array.")
    if not items:
        raise ValueError("x must contain at least one stratum.")
    tabs = [_table(item, positive_margins=method != "zelen", shape=(2, 2))[0]
            for item in items]
    k = len(tabs)
    if method == "zelen":
        observed = sum(int(t[0, 0]) for t in tabs)
        return TestResult(float(observed), _zelen(tabs, observed, max_tables),
                          "Zelen's exact test of homogeneity of odds ratios",
                          n_strata=k)
    if k < 2:
        raise ValueError("Asymptotic tests require at least two strata.")
    counts = np.stack(tabs).astype(float)
    if method == "woolf":
        if np.any(counts == 0):
            counts += 0.5
        logs = (np.log(counts[:, 0, 0]) + np.log(counts[:, 1, 1])
                - np.log(counts[:, 0, 1]) - np.log(counts[:, 1, 0]))
        weights = 1 / np.sum(1 / counts, axis=(1, 2))
        pooled = np.sum(weights * logs) / np.sum(weights)
        statistic = float(np.sum(weights * (logs - pooled) ** 2))
        sparse = np.any(counts < 5)
        description = "Woolf test of homogeneity of odds ratios"
    else:
        totals = counts.sum(axis=(1, 2))
        numerator = np.sum(counts[:, 0, 0] * counts[:, 1, 1] / totals)
        denominator = np.sum(counts[:, 0, 1] * counts[:, 1, 0] / totals)
        theta = numerator / denominator if denominator else math.inf
        if not math.isfinite(theta) or theta <= 0:
            raise ValueError("Breslow-Day requires a finite positive Mantel-Haenszel "
                             "odds ratio; consider method='zelen'.")
        fitted = []
        for table, total in zip(counts, totals):
            r, c = table[0].sum() / total, table[:, 0].sum() / total
            if theta == 1:
                a = r * c
            else:
                b = 1 + (theta - 1) * (r + c)
                discriminant = max(0.0, b * b + 4 * (1 - theta) * theta * r * c)
                q = -0.5 * (b + math.copysign(math.sqrt(discriminant), b))
                roots = [q / (1 - theta)]
                if q:
                    roots.append(-theta * r * c / q)
                feasible = [v for v in roots if math.isfinite(v)
                            and max(0, r + c - 1) < v < min(r, c)]
                if len(feasible) != 1:
                    raise ValueError("Could not fit positive Breslow-Day cell counts.")
                a = feasible[0]
            fitted.append(np.array([[a, r - a], [c - a, 1 - r - c + a]]) * total)
        fitted = np.stack(fitted)
        if np.any(fitted <= 0) or not np.all(np.isfinite(fitted)):
            raise ValueError("Could not fit positive Breslow-Day cell counts.")
        variance = 1 / np.sum(1 / fitted, axis=(1, 2))
        difference = counts[:, 0, 0] - fitted[:, 0, 0]
        statistic = float(np.sum(difference ** 2 / variance))
        if correct:
            statistic -= float(difference.sum() ** 2 / variance.sum())
        sparse = np.any(fitted < 5)
        description = "Breslow-Day test of homogeneity of odds ratios"
        if correct:
            description += " with Tarone adjustment"
    if sparse:
        warnings.warn("Chi-squared approximation may be incorrect: "
                      "some cell counts are below 5.", RuntimeWarning, stacklevel=2)
    statistic = max(0.0, statistic)
    return TestResult(statistic, float(chi2.sf(statistic, k - 1)), description,
                      df=k - 1, n_strata=k)


def _zelen(tables, observed, max_tables):
    supports, coefficients = [], []
    observed_weight = 0.0
    for table in tables:
        r1, r2 = (int(v) for v in table.sum(axis=1))
        c1 = int(table[:, 0].sum())
        lower, upper = max(0, c1 - r2), min(r1, c1)
        if upper - lower + 1 > max_tables:
            raise ValueError("Zelen support exceeds max_tables.")
        # Relative adjacent weights preserve precision for huge margins with
        # narrow supports. Per-stratum normalizing constants cancel.
        if lower == upper:
            support, weights = np.array([lower]), np.array([0.0])
        else:
            conditional = _ConditionalTable(table)
            support, weights = conditional.support, conditional.base
        observed_weight += float(weights[int(table[0, 0]) - lower])
        supports.append(support)
        coefficients.append(weights)
    lows = np.cumsum([int(s[0]) for s in supports][::-1])[::-1]
    highs = np.cumsum([int(s[-1]) for s in supports][::-1])[::-1]
    weights = []
    visited = 0

    def visit(i, remaining, weight):
        nonlocal visited
        visited += 1
        if visited > max_tables:
            raise ValueError("Zelen enumeration exceeded max_tables.")
        if i == len(tables):
            if remaining == 0:
                weights.append(weight)
            return
        if i == len(tables) - 1:
            low = int(supports[i][0])
            if low <= remaining <= int(supports[i][-1]):
                weights.append(weight + float(coefficients[i][remaining - low]))
            return
        for value, coefficient in zip(supports[i], coefficients[i]):
            rest = remaining - int(value)
            if lows[i + 1] <= rest <= highs[i + 1]:
                visit(i + 1, rest, weight + float(coefficient))

    visit(0, observed, 0.0)
    weights = np.array(weights)
    return min(1.0, float(np.exp(logsumexp(weights[weights <= observed_weight + 1e-12])
                                - logsumexp(weights))))
