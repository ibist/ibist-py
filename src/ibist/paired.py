"""Confidence intervals for paired binary outcomes."""

import math
from numbers import Real
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .odds_ratio import _exp, _probability, _validate_table
from .proportions import prop_1s_ci
from .results import CIResult, ConfidenceInterval, _ci_orientation
from .risk_ratio import _methods
from ._wang import _wang_lower


def or_paired_ci(x, *, conf_level=0.95, method="wald", side=None,
                 alternative=None) -> CIResult:
    """Estimate the conditional matched-pair odds ratio b/c.

    Parameters
    ----------
    x : array_like, shape (2, 2)
        Nonnegative integer pair counts. Rows describe the first member,
        columns the second, with the positive category first in both.
        Thus b=x[0, 1] is positive only for the first member and c=x[1, 0]
        only for the second. For matched case-control data, put cases in
        rows, controls in columns, and exposure present first.
    conf_level : float, default 0.95
        Confidence level strictly between zero and one.
    method : str or sequence of str, default "wald"
        "wald", "wilson", "clopper-pearson", "midp", or "blaker".
        Requested order and repetitions are retained. Wald requires both
        discordant counts positive; the remaining methods permit zeros.
    side : {"two.sided", "lower", "upper"}, optional
        Omitted or None selects two-sided limits. One-sided bounds use the
        full error probability. Lower bounds have upper endpoint inf;
        upper bounds have lower endpoint 0.
    alternative : {"two.sided", "greater", "less"}, optional
        Compatibility alias: greater selects lower, less selects upper.
        Conflicting side and alternative values raise an error.

    Returns
    -------
    CIResult
        Unadjusted ratio b/c in estimate, one conf_int row per requested
        method, method labels, side, conf_level, and null_value=1.
        Summaries round display values only.

    Raises
    ------
    ValueError
        If counts or options are invalid, or Wald is requested with a
        zero discordant count.

    Warns
    -----
    RuntimeWarning
        When no discordant pairs exist and a transformed method is used;
        the odds ratio is unidentified.

    See Also
    --------
    prop_paired_ci : Confidence intervals for a paired proportion difference.
    prop_1s_ci : Binomial intervals underlying the transformed methods.

    Notes
    -----
    Concordant pairs contain no information about this conditional OR.
    Transposing the table or reversing both category orders reciprocates
    b/c. Reversing just one category order changes the discordant pairs.
    This is a conditional matched-pair OR, not a marginal OR or the
    cross-product ratio of the pair-count table.

    Wald uses exp(log(b/c) +/- z*sqrt(1/b+1/c)). Other methods transform
    binomial limits for b successes in b+c discordant pairs by p/(1-p).
    Clopper--Pearson and two-sided Blaker intervals have at least nominal
    conditional coverage; Wald, Wilson, and mid-p lack that guarantee.
    One-sided Blaker bounds use inclusive exact binomial tails, matching R.
    With one nonzero discordant count, the estimate is zero or infinity.
    With no discordant pairs, transformed methods warn and return nan with
    [0, inf]; requesting Wald raises an error. Definitions follow R
    or.paired.ci().

    References
    ----------
    Fagerland, Lydersen and Laake (2014), Statistics in Medicine 33,
    2850--2875, doi:10.1002/sim.6148.
    Blaker (2000), Canadian Journal of Statistics 28, 783--798,
    doi:10.2307/3315916.

    Examples
    --------
    >>> twins = [[51, 30], [10, 41]]
    >>> or_paired_ci(twins).estimate
    3.0
    >>> result = or_paired_ci(twins, method=["wilson", "clopper-pearson"])
    >>> result.methods
    ('wilson', 'clopper-pearson')
    >>> or_paired_ci([[8, 4], [0, 6]], method="wilson").estimate
    inf
    """
    side, alternative = _ci_orientation(side, alternative)
    tab = _validate_table(x)
    if any(isinstance(v, (bool, np.bool_))
           for v in np.asarray(x, dtype=object).flat):
        raise ValueError("Tables must contain counts, not booleans.")
    conf_level = _probability(conf_level, "conf_level")
    methods = _methods(method, ("wald", "wilson", "clopper-pearson",
                                "midp", "blaker"))
    b, c = int(tab[0, 1]), int(tab[1, 0])
    if "wald" in methods and (not b or not c):
        raise ValueError("Wald requires both discordant counts positive.")
    if not b + c:
        warnings.warn("No discordant pairs: matched-pair odds ratio is "
                      "unidentified.", RuntimeWarning, stacklevel=2)
    estimate = b / c if c else math.inf if b else math.nan
    tail = (1 - conf_level) / (2 if side == "two.sided" else 1)
    z = float(norm.isf(tail))
    intervals = []
    for name in methods:
        if not b + c:
            lower, upper = 0.0, math.inf
        elif name == "wald":
            margin = z * math.sqrt(1 / b + 1 / c)
            lower, upper = _exp(math.log(b / c) - margin), _exp(
                math.log(b / c) + margin)
        else:
            binomial = "clopper-pearson" if name == "midp" else name
            probability = prop_1s_ci(b, b + c, conf_level=conf_level,
                                      method=binomial, midp=name == "midp",
                                      side=side, tol=1e-12).conf_int[0]
            lower = _odds(probability[0])
            upper = _odds(probability[1])
        if side == "lower":
            upper = math.inf
        if side == "upper":
            lower = 0.0
        intervals.append(ConfidenceInterval(name, lower, upper))
    return CIResult(estimate, tuple(intervals), conf_level,
                    "matched-pair odds ratio", null_value=1.0,
                    alternative=alternative)


def prop_paired_ci(b, c, n, *, conf_level=0.95, method="score", side=None,
                   alternative=None, tol=1e-8, precision=1e-5, grid_one=30,
                   grid_two=20) -> CIResult:
    """Calculate intervals for the paired proportion difference (b-c)/n.

    Parameters
    ----------
    b : nonnegative integer
        Discordant pairs positive only for group 1.
    c : nonnegative integer
        Discordant pairs positive only for group 2.
    n : positive integer
        Total number of pairs, including concordant pairs. Require b+c <= n
        and each count <= 2**53-1.
    conf_level : float, default 0.95
        Confidence level strictly between zero and one.
    method : str or sequence of str, default "score"
        "score" (Tango), "wald", "waldcc", "agresti-min", or "wang".
        Requested order and repetitions are retained.
    side : {"two.sided", "lower", "upper"}, optional
        Omitted or None selects two-sided limits. One-sided bounds use the
        full error probability. Lower bounds have upper endpoint 1;
        upper bounds have lower endpoint -1.
    alternative : {"two.sided", "greater", "less"}, optional
        Compatibility alias: greater selects lower, less selects upper.
        Conflicting side and alternative values raise an error.
    tol : positive finite float, default 1e-8
        Root-finding tolerance for score limits.
    precision : positive finite float, default 1e-5
        Endpoint precision for Wang's method.
    grid_one : integer >= 2, default 30
        First nuisance-parameter search grid size for Wang's method.
    grid_two : integer >= 2, default 20
        Second nuisance-parameter search grid size for Wang's method.

    Returns
    -------
    CIResult
        Observed scalar difference (b-c)/n in estimate, one conf_int row per
        requested method, method labels, side, conf_level, and null_value=0.
        All limits are restricted to [-1, 1].

    Raises
    ------
    ValueError
        If counts or options are invalid, b+c exceeds n, or Wang
        enumeration exceeds one million states.

    See Also
    --------
    or_paired_ci : Confidence intervals for a conditional matched-pair OR.
    prop_2s_ci : Confidence intervals for independent proportion differences.

    Notes
    -----
    Wald uses the paired multinomial variance; Waldcc adds 1/n to its
    margin. Agresti--Min computes a Wald interval using b+0.5, c+0.5, and
    n+2 internally, retaining (b-c)/n as the reported estimate. Tango score
    inversion fits the multinomial likelihood under a candidate difference.
    Python's constrained fits are more accurate than R's current coarse
    optimization, so score limits can differ slightly.

    Wang uses the C++ core shared with R and its numerical inductive-order
    construction. Finite nuisance grids approximate the supremum; their
    settings do not certify exact nominal coverage. Enumeration uses O(n^2)
    states and can be expensive even for moderate n. At most one million
    states are permitted. Wang rounds endpoints at precision, as R does;
    other methods retain full numerical precision. Definitions follow R
    prop.paired.ci().

    References
    ----------
    Tango (1998), Statistics in Medicine 17, 891--908.
    Agresti and Min (2005), Statistics in Medicine 24, 729--740.
    Wang (2012), Statistics & Probability Letters 82, 1623--1628.

    Examples
    --------
    >>> result = prop_paired_ci(8, 25, 180)
    >>> round(result.estimate, 4)
    -0.0944
    >>> prop_paired_ci(8, 25, 180, method=["waldcc", "agresti-min"]).methods
    ('waldcc', 'agresti-min')
    >>> result = prop_paired_ci(3, 0, 4, method="wang", precision=0.0001)
    >>> result.conf_int.shape
    (1, 2)
    """
    side, alternative = _ci_orientation(side, alternative)
    b, c, n = (_count(value, name) for value, name in
               [(b, "b"), (c, "c"), (n, "n")])
    if n == 0 or b + c > n:
        raise ValueError("n must be positive and b+c must not exceed n.")
    conf_level = _probability(conf_level, "conf_level")
    methods = _methods(method, ("score", "wald", "waldcc", "agresti-min",
                                "wang"))
    tol = _positive(tol, "tol")
    precision = _positive(precision, "precision")
    grid_one = _grid(grid_one, "grid_one")
    grid_two = _grid(grid_two, "grid_two")
    estimate = (b - c) / n
    tail = (1 - conf_level) / (2 if side == "two.sided" else 1)
    z = float(norm.isf(tail))
    intervals = []
    for name in methods:
        if name == "wang":
            level = 1 - tail
            lower = (-1.0 if side == "upper" else _wang_lower(
                b, n - b - c, c, level, precision, grid_one, grid_two))
            upper = (1.0 if side == "lower" else -_wang_lower(
                c, n - b - c, b, level, precision, grid_one, grid_two))
        elif name == "score":
            lower = _paired_score_limit(b, c, n, z, tol)
            upper = _paired_score_limit(b, c, n, -z, tol)
        else:
            bb, cc, nn = ((b + 0.5, c + 0.5, n + 2) if
                          name == "agresti-min" else (b, c, n))
            center = (bb - cc) / nn
            variance = max(0.0, (bb + cc - (bb - cc)**2 / nn) / nn**2)
            margin = z * math.sqrt(variance) + (1 / n if name == "waldcc"
                                                else 0)
            lower, upper = center - margin, center + margin
        lower = float(np.clip(lower, -1, 1))
        upper = float(np.clip(upper, -1, 1))
        if side == "lower":
            upper = 1.0
        if side == "upper":
            lower = -1.0
        intervals.append(ConfidenceInterval(name, lower, upper))
    return CIResult(estimate, tuple(intervals), conf_level,
                    "paired proportion difference", null_value=0.0,
                    alternative=alternative)


def _paired_score(b, c, n, delta):
    """Signed Tango score using the concave constrained multinomial fit."""
    low, high = max(0.0, -delta), (1 - delta) / 2
    a = n - b - c

    def slope(q):
        p, r = q + delta, 1 - 2 * q - delta
        return ((b / p if b and p > 0 else math.inf if b else 0)
                + (c / q if c and q > 0 else math.inf if c else 0)
                - (2 * a / r if a and r > 0 else math.inf if a else 0))

    # Endpoints are included: zero cells often put the MLE on the boundary.
    if slope(low) <= 0:
        q = low
    elif slope(high) >= 0:
        q = high
    else:
        q = brentq(slope, low, high, xtol=1e-15, rtol=1e-14)
    difference = (b - c) / n - delta
    variance = max(0.0, (2 * q + delta - delta * delta) / n)
    if difference == 0:
        return 0.0
    return difference / math.sqrt(variance) if variance else math.copysign(
        math.inf, difference)


def _paired_score_limit(b, c, n, target, tol):
    def objective(delta):
        return _paired_score(b, c, n, delta) - target

    # The inward endpoints avoid a degenerate constrained likelihood at +/-1.
    edge = np.finfo(float).eps * 4
    low, high = -1 + edge, 1 - edge
    if objective(low) <= 0:
        return -1.0
    if objective(high) >= 0:
        return 1.0
    return float(brentq(objective, low, high, xtol=tol, rtol=1e-14))


def _odds(p):
    return float(p / (1 - p)) if p < 1 else math.inf


def _count(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0 or value != int(value)
            or value > 2**53 - 1):
        raise ValueError(f"{name} must be a nonnegative integer <= 2**53-1.")
    return int(value)


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a positive finite number.")
    return float(value)


def _grid(value, name):
    value = _count(value, name)
    if value < 2:
        raise ValueError(f"{name} must be an integer >= 2.")
    return value
