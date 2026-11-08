"""Cochran--Armitage tests for ordered-group event proportions."""

import math

import numpy as np
from scipy.special import betaln, logsumexp
from scipy.stats import norm

from .contingency import TestResult, _flag, _table


def trend_test(x, *, scores=None, alternative="two.sided", exact=False):
    """Test a linear trend in event proportions across ordered groups.

    Parameters
    ----------
    x : array_like or pandas.DataFrame, shape (R, 2)
        Nonnegative integer counts. Ordered groups are rows; the first
        column contains events. Both columns and at least two rows must
        have positive totals. Counts and total must not exceed 2**53-1.
    scores : one-dimensional array_like, optional
        Finite strictly increasing scores for the original rows. Defaults
        to 1, ..., R, retaining positions when empty rows are dropped.
    alternative : {"two.sided", "greater", "less"}, default "two.sided"
        Greater means increasing event proportions as group scores increase.
    exact : bool, default False
        Compute a fixed-margin exact conditional p-value instead of the
        asymptotic normal p-value.

    Returns
    -------
    TestResult
        Signed Cochran--Armitage statistic, chisq (its square), score_statistic
        (sum of scores times events), p_value, alternative, and method.
        No coefficient estimate, CI, or degrees of freedom are fabricated.

    Raises
    ------
    ValueError
        If inputs are invalid or Python's exact calculation exceeds one
        million intermediate states. It never silently uses an approximation.

    See Also
    --------
    assoc_test : Association tests including CMH linear-by-linear inference.

    Notes
    -----
    Matches R trend.test() and its supported ca.trend.test() backend. Exact
    inference conditions on both margins. Tails include the observed score;
    two-sided p-values double the smaller tail, capped at one. The reported
    Z always uses the asymptotic Cochran--Armitage variance, including when
    its p-value is exact. This variance differs from a finite-population
    correction used in some conditional score tests.

    Dynamic programming merges scores at 15 significant digits and uses R's
    100*machine-epsilon relative score tolerance for inclusive tails. Exact
    computation can be expensive. Python limits intermediate states to one
    million; R's legacy backend has no such resource limit. Neither scores
    nor category labels are sorted automatically.

    References
    ----------
    Cochran (1954), Biometrics 10(4), 417--451, doi:10.2307/3001616.
    Armitage (1955), Biometrics 11(3), 375--386, doi:10.2307/3001775.

    Examples
    --------
    >>> tab = [[8, 12], [12, 8], [16, 4]]
    >>> trend_test(tab, scores=[1, 2, 4]).statistic > 0
    True
    >>> trend_test([[2, 0], [1, 1], [0, 2]], exact=True).estimate is None
    True
    """
    exact = _flag(exact, "exact")
    if not isinstance(alternative, str) or alternative not in (
            "two.sided", "greater", "less"):
        raise ValueError("alternative must be two.sided, greater, or less.")
    table, _ = _table(x, positive_margins=False)
    if table.shape[1] != 2:
        raise ValueError("x must be an R-by-2 count table.")
    if scores is None:
        scores = np.arange(1, len(table) + 1, dtype=float)
    else:
        scores = np.asarray(scores)
        if (scores.ndim != 1 or len(scores) != len(table)
                or scores.dtype.kind not in "iuf"
                or not np.all(np.isfinite(scores))
                or not np.all(scores[1:] > scores[:-1])):
            raise ValueError("scores must be finite and strictly increasing, "
                             "one per original row.")
        scores = scores.astype(float)
    keep = table.sum(axis=1) > 0
    table, scores = table[keep], scores[keep]
    if len(table) < 2 or np.any(table.sum(axis=0) == 0):
        raise ValueError("At least two rows and both columns must be nonempty.")
    rows, events = table.sum(axis=1), table[:, 0]
    n = int(rows.sum())
    proportion = int(events.sum()) / n
    center = (rows @ scores) / n
    variance = proportion * (1 - proportion) * np.sum(
        rows * (scores - center)**2)
    numerator = np.sum(scores * (events - rows * proportion))
    z = float(numerator / math.sqrt(variance)) if variance else 0.0
    statistic = float(scores @ events)
    if exact:
        support, mass = _distribution(rows, scores - scores[0],
                                       int(events.sum()))
        observed = float((scores - scores[0]) @ events)
        tolerance = 100 * np.finfo(float).eps * max(1, max(abs(support)))
        lower = float(mass[support <= observed + tolerance].sum())
        upper = float(mass[support >= observed - tolerance].sum())
        p_value = {"less": lower, "greater": upper,
                   "two.sided": min(1.0, 2 * min(lower, upper))}[alternative]
    else:
        p_value = (2 * norm.sf(abs(z)) if alternative == "two.sided" else
                   norm.sf(z) if alternative == "greater" else norm.cdf(z))
    method = ("Exact conditional Cochran-Armitage trend test" if exact else
              "Asymptotic Cochran-Armitage trend test")
    return TestResult(z, float(p_value), method, alternative=alternative,
                      chisq=z**2, score_statistic=statistic)


def _distribution(rows, scores, events):
    states = {(0, 0.0): 0.0}
    later = int(rows.sum())
    for size, score in zip(rows, scores):
        size = int(size)
        later -= size
        updated = {}
        for (cases, total), weight in states.items():
            low = max(0, events - cases - later)
            high = min(size, events - cases)
            if high - low + 1 > 1000000:
                raise ValueError("Exact trend calculation exceeds "
                                 "1,000,000 states.")
            for value in range(low, high + 1):
                total_score = float(format(total + score * value, ".15g"))
                key = (cases + value, total_score)
                choose = (-math.log(size + 1) -
                          betaln(value + 1, size - value + 1))
                candidate = weight + choose
                updated[key] = (np.logaddexp(updated[key], candidate)
                                if key in updated else candidate)
                if len(updated) > 1000000:
                    raise ValueError("Exact trend calculation exceeds "
                                     "1,000,000 states.")
        states = updated
    support = np.array([total for _, total in states])
    weights = np.array(list(states.values()))
    return support, np.exp(weights - logsumexp(weights))
