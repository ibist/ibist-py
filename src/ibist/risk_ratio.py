"""Confidence intervals for the ratio of two independent binomial risks."""

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .odds_ratio import _exp, _probability, _validate_table
from .proportions import _independent_counts
from .results import CIResult, ConfidenceInterval, _ci_orientation


def rr_2x2_ci(x, *, conf_level=0.95, method="koopman",
              alternative=None, side=None, tol=1e-8) -> CIResult:
    """Calculate confidence intervals for the first risk divided by the second.

    Parameters
    ----------
    x : array_like, shape (2, 2)
        Nonnegative integer counts. Rows are independent samples; columns
        contain events and non-events. Both row totals must be positive.
    conf_level : float, default 0.95
        Confidence level strictly between zero and one.
    method : str or sequence of str, default "koopman"
        "koopman" inverts the constrained Pearson score statistic. "katz"
        exponentiates log-Wald limits and requires positive event counts.
        Requested order and repeated methods are retained, matching R.
    side : {"two.sided", "lower", "upper"}, optional
        Omitted or None selects two-sided limits.
    alternative : {"two.sided", "less", "greater"}, optional
        Compatibility alias; conflicting values with side are rejected.
        Two limits, an upper bound with lower limit zero, or a lower bound
        with upper limit infinity. One-sided bounds use the full tail error.
    tol : float, default 1e-8
        Positive absolute root tolerance on the log risk-ratio scale.

    Returns
    -------
    CIResult
        Scalar observed risk ratio, method-specific intervals, side,
        and copied group metadata in samples and sample_proportions.

    Notes
    -----
    Katz uses variance 1/x1 - 1/n1 + 1/x2 - 1/n2 on the log scale, without
    pseudo-counts. Koopman fits binomial probabilities under p1=r*p2 and
    inverts their Pearson statistic, without a finite-sample factor or
    continuity correction. Reciprocal row reversal stabilizes inversion.
    A zero event count gives a boundary estimate and Koopman limit. With
    zero events in both rows the estimate is nan and limits are (0, inf).
    No risk difference, odds ratio, or Poisson rate ratio is substituted.

    References
    ----------
    Katz, Baptista, Azen, and Pike (1978). Obtaining confidence intervals
    for the risk ratio in cohort studies. Biometrics 34, 469--474.
    Koopman (1984). Confidence intervals for the ratio of two binomial
    proportions. Biometrics 40, 513--517.

    Examples
    --------
    >>> result = rr_2x2_ci([[38, 62], [267, 717]], method=["katz", "koopman"])
    >>> round(result.estimate, 3)
    1.4
    """
    side, alternative = _ci_orientation(side, alternative)
    table = _validate_table(x)
    if any(isinstance(v, (bool, np.bool_))
           for v in np.asarray(x, dtype=object).flat):
        raise ValueError("x must contain integer counts, not booleans.")
    counts, sizes = _independent_counts(table, None)
    conf_level = _probability(conf_level, "conf_level")
    _alternative(alternative)
    if (isinstance(tol, (bool, np.bool_)) or not isinstance(tol, Real)
            or not math.isfinite(tol) or tol <= 0):
        raise ValueError("tol must be a positive finite number.")
    methods = _methods(method, ("koopman", "katz"))
    proportions = counts / sizes
    estimate = (float(proportions[0] / proportions[1])
                if counts[1] else math.inf if counts[0] else math.nan)
    tail = (1 - conf_level) / (2 if alternative == "two.sided" else 1)
    z = float(norm.isf(tail))
    intervals = []
    for name in methods:
        if name == "katz":
            if np.any(counts == 0):
                raise ValueError("Katz requires positive event counts.")
            # This form retains small failure counts in near-certain risks.
            variance = float(np.sum((sizes - counts) / counts / sizes))
            center = math.log(estimate)
            lower, upper = (_exp(center - z * math.sqrt(variance)),
                            _exp(center + z * math.sqrt(variance)))
        else:
            lower = _koopman_lower(counts, sizes, z, float(tol))
            opposite = _koopman_lower(counts[::-1], sizes[::-1], z,
                                      float(tol))
            upper = math.inf if opposite == 0 else 1 / opposite
        if alternative == "less":
            lower = 0.0
        if alternative == "greater":
            upper = math.inf
        intervals.append(ConfidenceInterval(name, lower, upper))
    samples = tuple((int(a), int(b)) for a, b in zip(counts, sizes))
    return CIResult(estimate, tuple(intervals), conf_level, "risk ratio",
                    null_value=1.0, alternative=alternative,
                    _sample_counts=samples)


def _methods(method, choices):
    if isinstance(method, str):
        selected = [method]
    elif isinstance(method, Sequence):
        selected = list(method)
    else:
        raise ValueError("method must be a name or sequence of names.")
    if not selected or any(not isinstance(m, str) or m not in choices
                           for m in selected):
        raise ValueError(f"method must select one or more of {choices}.")
    return selected


def _alternative(value):
    if not isinstance(value, str) or value not in ("two.sided", "less",
                                                  "greater"):
        raise ValueError("alternative must be two.sided, less, or greater.")
    return value


def _koopman_score(eta, counts, sizes):
    if eta > 0:
        return -_koopman_score(-eta, counts[::-1], sizes[::-1])
    ratio = math.exp(eta)
    x1, x2 = (float(v) for v in counts)
    n1, n2 = (float(v) for v in sizes)
    total = x1 + x2
    if total == 0:
        return 0.0
    b = ratio * (n1 + x2) + n2 + x1
    disc = max(0.0, b * b - 4 * ratio * (n1 + n2) * total)
    q = min(1.0, 2 * total / (b + math.sqrt(disc)))
    # Solve for 1-q separately to avoid cancellation near complete success.
    a = ratio * (n1 + n2)
    bt = b - 2 * a
    ct = (1 - ratio) * (n2 - x2)
    root = math.sqrt(max(0.0, bt * bt + 4 * a * ct))
    if bt >= 0:
        t = 2 * ct / (bt + root) if bt + root else 0.0
    else:
        t = (root - bt) / (2 * a)
    t = min(1.0, max(0.0, t))
    probabilities = (ratio * q, q)
    failures = ((1 - ratio) + ratio * t, t)
    statistic = 0.0
    for x, n, p, f in zip((x1, x2), (n1, n2), probabilities, failures):
        residual = x * f - (n - x) * p
        variance = n * p * f
        if variance > 0:
            statistic += residual * residual / variance
        elif abs(residual) > 1e-10:
            statistic = math.inf
    difference = x1 / n1 - ratio * x2 / n2
    if difference == 0:
        return 0.0
    return math.copysign(math.sqrt(statistic), difference)


def _koopman_lower(counts, sizes, z, tol):
    if not np.any(counts) or (counts[0] == 0 and z >= 0):
        return 0.0

    def objective(eta):
        return _koopman_score(eta, counts, sizes) - z

    lower, upper = -1.0, 1.0
    while objective(lower) < 0 and lower > -700:
        lower = max(-700.0, lower * 2)
    while objective(upper) > 0 and upper < 700:
        upper = min(700.0, upper * 2)
    if objective(lower) <= 0:
        return 0.0
    if objective(upper) >= 0:
        return math.inf
    return math.exp(brentq(objective, lower, upper, xtol=tol))
