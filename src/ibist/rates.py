"""Intervals and normal-approximation tests for Poisson rates."""

import math
from numbers import Real
import warnings

import numpy as np
from scipy.stats import chi2, norm

from .contingency import TestResult, _flag
from .odds_ratio import _exp, _probability
from .proportions import prop_1s_ci
from .results import CIResult, ConfidenceInterval, _ci_orientation
from .risk_ratio import _alternative, _methods


def rate_1s_ci(x, exposure=1.0, *, conf_level=0.95, method="exact",
               correct=True, alternative=None, side=None,
               asymcheck=False) -> CIResult:
    """Estimate a Poisson rate and return one or more confidence intervals.

    Parameters
    ----------
    x : nonnegative integer
        Observed event count. Zero events are allowed.
    exposure : positive finite scalar, default 1
        Known person-time or other exposure. Exposure one estimates the mean.
    conf_level : float, default 0.95
        Confidence level strictly between zero and one.
    method : str or sequence of str, default "exact"
        "exact" (Garwood), "score", "wh" (Wilson--Hilferty), "wald", or
        "log". Method order and repetitions are retained.
    correct : bool, default True
        Count-scale corrections for score, WH, and Wald. Exact and log
        intervals are unchanged. This follows the R rate-function defaults.
    side : {"two.sided", "lower", "upper"}, optional
        Interval form; omitted or None selects two-sided limits. One-sided
        bounds use the full error probability.
    alternative : {"two.sided", "less", "greater"}, optional
        Compatibility alias: greater selects lower, less selects upper.
        Conflicting side and alternative values raise an error.
    asymcheck : bool, default False
        Attach shared diagnostics in result.diagnostics. Approximate
        methods report events and plug-in skewness without a universal
        accuracy cutoff. Exact methods are marked not applicable.

    Returns
    -------
    CIResult
        Scalar rate estimate, intervals, and counts/exposure metadata in
        samples. sample_rates returns a fresh array of the observed rates.

    Raises
    ------
    ValueError
        If the count, exposure, confidence level, or method is invalid.

    See Also
    --------
    rate_2s_ci : Confidence intervals for a ratio of independent rates.
    rate_test : Normal-approximation test for one or two rates.

    Notes
    -----
    Score corrects the lower and upper counts by -0.5 and +0.5, with the
    lower count restricted to zero. Corrected WH uses x+0.5 at both limits;
    uncorrected WH uses x and x+1. Wald corrects the counts before applying
    their square-root standard errors. At zero events, uncorrected Wald
    and log use the exact upper-limit fallback -log(tail)/exposure from R.
    All limits are restricted to nonnegative rates. In particular, Python
    clips a negative corrected WH lower limit at zero; current R can return
    a negative value at zero events. The R review note records this case.
    Every method supports the same side argument as the R counterpart.

    References
    ----------
    Garwood (1936). Fiducial limits for the Poisson distribution.
    Biometrika 28, 437--442.
    Wilson and Hilferty (1931). The distribution of chi-square.
    Proceedings of the National Academy of Sciences 17, 684--688.
    Correction formulas follow ibist-R rate.1s.ci(), not a new method.

    Examples
    --------
    >>> rate_1s_ci(5, 10).estimate
    0.5
    """
    side, alternative = _ci_orientation(side, alternative)
    counts, exposures = _rate_inputs(x, exposure, lengths=(1,))
    conf_level = _probability(conf_level, "conf_level")
    correct = _flag(correct, "correct")
    asymcheck = _flag(asymcheck, "asymcheck")
    _alternative(alternative)
    methods = _methods(method, ("exact", "score", "wh", "wald", "log"))
    count, time = int(counts[0]), float(exposures[0])
    tail = (1 - conf_level) / (2 if alternative == "two.sided" else 1)
    z = float(norm.isf(tail))
    intervals = []
    for name in methods:
        limits = _one_rate_limits(count, time, name, tail, z, correct)
        lower, upper = _bounds(limits, alternative)
        intervals.append(ConfidenceInterval(name, lower, upper))
    diagnostics = None
    if asymcheck:
        from .asymcheck import _rate
        diagnostics = _rate(count, time, conf_level, methods, correct,
                             alternative, side)
    return CIResult(count / time, tuple(intervals), conf_level,
                    "Poisson rate", alternative=alternative, correct=correct,
                    _rate_data=((count, time),), diagnostics=diagnostics)


def rate_2s_ci(x, exposure=(1.0, 1.0), *, conf_level=0.95, method="log",
               alternative=None, side=None) -> CIResult:
    """Estimate a ratio of two independent Poisson rates.

    Parameters
    ----------
    x : length-two array_like of nonnegative integers
        Independent event counts. At least one must be positive; counts
        and their total must not exceed 2**53-1.
    exposure : length-two array_like, default (1.0, 1.0)
        Known positive finite exposures for the two groups. Scalar exposure
        is not recycled. Use matching units, such as person-years.
    conf_level : float, default 0.95
        Confidence level strictly between zero and one.
    method : str or sequence of str, default "log"
        "log" (log-Wald), "score" (transformed Wilson), or "exact"
        (transformed Clopper--Pearson). Log requires both counts positive.
        Requested order and repetitions are retained.
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
        Scalar ratio (x1/T1)/(x2/T2) in estimate, one conf_int row per
        method, method labels, side, conf_level, and null_value=None.
        samples contains copied counts and exposures; sample_rates contains
        the two observed rates. R stores all three estimates in one vector.

    Raises
    ------
    ValueError
        If inputs or options are invalid, both event counts are zero,
        or log-Wald is requested with a zero event count.

    See Also
    --------
    rate_1s_ci : Confidence intervals for a single Poisson rate.
    rate_test : Normal-approximation test for one or two Poisson rates.

    Notes
    -----
    Independent counts satisfy Xi ~ Poisson(lambda_i*Ti). Log-Wald limits
    exponentiate log(estimate) +/- z*sqrt(1/x1+1/x2). Conditional methods
    use x1 given x1+x2 as a binomial count, then transform probability p
    into the rate ratio p*T2/((1-p)*T1). They permit a zero count, with a
    boundary estimate and limits at zero or infinity. Definitions follow
    R rate.2s.ci(). No continuity correction is applied.

    References
    ----------
    Clopper and Pearson (1934), Biometrika 26, 404--413, for the conditional
    exact binomial bounds. Wilson (1927), JASA 22, 209--212, for the score
    bounds. The exposure transformation follows ibist-R rate.2s.ci().

    Examples
    --------
    >>> result = rate_2s_ci([12, 5], [100, 80])
    >>> round(result.estimate, 2)
    1.92
    >>> rate_2s_ci([12, 5], [100, 80], method=["log", "score"]).methods
    ('log', 'score')
    >>> result = rate_2s_ci([0, 5], [100, 80], method="exact")
    >>> float(result.conf_int[0, 0])
    0.0
    """
    side, alternative = _ci_orientation(side, alternative)
    counts, exposures = _rate_inputs(x, exposure, lengths=(2,))
    if not np.any(counts):
        raise ValueError("At least one event is required.")
    conf_level = _probability(conf_level, "conf_level")
    _alternative(alternative)
    methods = _methods(method, ("log", "score", "exact"))
    estimate = _rate_ratio(counts, exposures)
    tail = (1 - conf_level) / (2 if alternative == "two.sided" else 1)
    z = float(norm.isf(tail))
    intervals = []
    for name in methods:
        if name == "log":
            limits = _log_ratio(counts, exposures, z)
        else:
            binomial = "wilson" if name == "score" else "clopper-pearson"
            total = int(counts[0]) + int(counts[1])
            first = prop_1s_ci(int(counts[0]), total, method=binomial,
                               conf_level=conf_level,
                               alternative=alternative).conf_int[0]
            other = {"less": "greater", "greater": "less"}.get(
                alternative, "two.sided")
            second = prop_1s_ci(int(counts[1]), total, method=binomial,
                                conf_level=conf_level,
                                alternative=other).conf_int[0]
            log_scale = math.log(exposures[1]) - math.log(exposures[0])
            lower = _scaled_odds(first[0], log_scale)
            upper = _scaled_odds(second[0], log_scale, reciprocal=True)
            limits = lower, upper
        lower, upper = _bounds(limits, alternative)
        intervals.append(ConfidenceInterval(name, lower, upper))
    return CIResult(estimate, tuple(intervals), conf_level,
                    "Poisson rate ratio",
                    alternative=alternative,
                    _rate_data=_metadata(counts, exposures))


def rate_test(x, exposure=1.0, *, r=1.0, alternative="two.sided",
              conf_level=0.95, correct=True) -> TestResult:
    """Normal-approximation test for one or two Poisson rates.

    Parameters
    ----------
    x : nonnegative integer or length-one/two array_like
        Event counts. One count specifies a one-sample test; two counts
        specify independent samples. Counts and their total must not exceed
        2**53-1. Two-sample inference requires both counts positive because
        the reported log-Wald interval needs them.
    exposure : positive finite scalar or length-one/two array_like, default 1
        Known exposures with the same length as x. Scalar exposure is not
        recycled for two counts. All exposures must use matching units.
    r : positive finite float, default 1
        Null rate per exposure unit for one sample. Ignored for two samples,
        whose null hypothesis is equality of rates (rate ratio one).
    alternative : {"two.sided", "greater", "less"}, default "two.sided"
        Alternative for the unit rate or the first rate divided by the
        second. Also determines the direction of the reported interval.
    conf_level : float, default 0.95
        Confidence level strictly between zero and one.
    correct : bool, default True
        Apply the directional half-count correction for one sample or
        capped Yates correction for the two-sample conditional test.

    Returns
    -------
    TestResult
        statistic is signed Z for one sample and chi-square for two;
        df=1 follows R in both cases. Also includes p_value, method,
        alternative, scalar estimate, null_value, conf_int, and conf_level.
        The estimate and interval concern the unit rate for one sample or
        rate ratio for two. samples and sample_rates expose copied group
        metadata; R's two-sample estimate vector combines both group rates
        and their ratio, whereas Python keeps the scalar target separate.

    Raises
    ------
    ValueError
        If inputs or options are invalid, including a zero event count
        in a two-sample request requiring the log-Wald interval.

    Warns
    -----
    UserWarning
        For two samples when a null expected binomial cell count is
        below five, following R prop.test().

    See Also
    --------
    rate_1s_ci : Single-rate intervals, including Garwood exact limits.
    rate_2s_ci : Rate-ratio intervals, including conditional exact limits.

    Notes
    -----
    One sample uses Z=(x-cc-r*T)/sqrt(r*T) and a matching score interval.
    The directional half-count correction is not capped, following R
    rate.test(). Two samples use the Pearson statistic for the conditional
    binomial count x1 given x1+x2, with null probability T1/(T1+T2).
    Small expected binomial cell counts trigger a warning, matching R.

    The two-sample interval is log-Wald for the rate ratio; correction
    affects the test but not this interval. Test and interval decisions can
    therefore differ. This is a normal approximation, not an exact Poisson
    test. Statistical formulas and correction rules follow R rate.test().

    References
    ----------
    R ibist rate.test(), Details; R stats prop.test(), Details, for the
    conditional binomial score statistic and its continuity correction.

    Examples
    --------
    >>> result = rate_test(411, 25800, r=0.0119, alternative="greater")
    >>> result.alternative
    'greater'
    >>> round(rate_test([12, 5], [100, 80]).estimate, 2)
    1.92
    """
    counts, exposures = _rate_inputs(x, exposure)
    conf_level = _probability(conf_level, "conf_level")
    correct = _flag(correct, "correct")
    _alternative(alternative)
    tail = (1 - conf_level) / (2 if alternative == "two.sided" else 1)
    zq = float(norm.isf(tail))
    if len(counts) == 1:
        null = _positive(r, "r")
        count, time = int(counts[0]), float(exposures[0])
        mean = null * time
        if not math.isfinite(mean) or mean <= 0:
            raise ValueError("The null mean r*exposure must be finite "
                             "and positive.")
        direction = {"greater": 1, "less": -1}.get(
            alternative, float(np.sign(count - mean)))
        cc = 0.5 * direction if correct else 0.0
        signed = (count - cc - mean) / math.sqrt(mean)
        statistic = signed
        estimate, name = count / time, "Rate"
        limits = _score_limits(count, time, zq, correct)
        description = "Normal approximation test for a Poisson rate"
    else:
        limits = _log_ratio(counts, exposures, zq)
        total = int(counts[0]) + int(counts[1])
        p0 = float(exposures[0] / exposures.sum())
        if not 0 < p0 < 1:
            raise ValueError("Exposures must define a probability in (0, 1).")
        expected = total * p0
        difference = counts[0] - expected
        cc = min(0.5, abs(difference)) if correct else 0.0
        signed = math.copysign(abs(difference) - cc, difference)
        signed /= math.sqrt(total * p0 * (1 - p0))
        statistic = signed * signed
        estimate, name, null = _rate_ratio(counts, exposures), "Rate ratio", 1.0
        description = ("Two-sample test for equality of Poisson rates "
                       "(conditional binomial normal approximation); "
                       "log-Wald rate-ratio interval")
        if min(expected, total - expected) < 5:
            warnings.warn("Chi-squared approximation may be incorrect.",
                          UserWarning, stacklevel=2)
    p_value = (2 * norm.sf(abs(signed)) if alternative == "two.sided"
               else norm.cdf(signed) if alternative == "less"
               else norm.sf(signed))
    return TestResult(float(statistic), float(p_value), description, df=1,
                      alternative=alternative, estimate=float(estimate),
                      conf_int=_bounds(limits, alternative),
                      conf_level=conf_level, null_value=null,
                      estimate_name=name,
                      _rate_data=_metadata(counts, exposures))


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a positive finite number.")
    return float(value)


def _rate_inputs(x, exposure, lengths=(1, 2)):
    try:
        counts, times = np.asarray(x), np.asarray(exposure)
        bool_counts = any(isinstance(v, (bool, np.bool_))
                          for v in np.asarray(x, dtype=object).flat)
        bool_times = any(isinstance(v, (bool, np.bool_))
                         for v in np.asarray(exposure, dtype=object).flat)
    except (TypeError, ValueError):
        raise ValueError("x and exposure must be numeric counts "
                         "and times.") from None
    if (bool_counts or counts.ndim > 1 or counts.size not in lengths
            or counts.dtype.kind not in "iuf"
            or not np.all(np.isfinite(counts)) or np.any(counts < 0)
            or np.any(counts != np.floor(counts))):
        raise ValueError("x must contain finite nonnegative integer counts.")
    if (np.any(counts > 2**53 - 1)
            or float(counts.sum(dtype=float)) > 2**53 - 1):
        raise ValueError("The total count must not exceed 2**53-1.")
    if (bool_times or times.ndim > 1 or times.size != counts.size
            or times.dtype.kind not in "iuf"
            or not np.all(np.isfinite(times)) or np.any(times <= 0)):
        raise ValueError("exposure must have the same length as x "
                         "and be positive.")
    if not math.isfinite(float(times.sum(dtype=float))):
        raise ValueError("Total exposure must be finite.")
    return counts.reshape(-1).astype(np.int64), times.reshape(-1).astype(float)


def _metadata(counts, exposure):
    return tuple((int(x), float(t)) for x, t in zip(counts, exposure))


def _bounds(limits, alternative):
    lower, upper = (max(0.0, float(v)) for v in limits)
    return (0.0 if alternative == "less" else lower,
            math.inf if alternative == "greater" else upper)


def _score_limits(x, exposure, z, correct):
    lower_x = max(0.0, x - 0.5) if correct else float(x)
    upper_x = x + 0.5 if correct else float(x)
    first = math.sqrt(lower_x + z * z / 4)
    second = math.sqrt(upper_x + z * z / 4)
    lower_root = (lower_x / (first + z / 2) if z >= 0 and first + z / 2
                  else first - z / 2)
    upper_root = (upper_x / (second - z / 2) if z < 0 and second - z / 2
                  else second + z / 2)
    return lower_root**2 / exposure, upper_root**2 / exposure


def _one_rate_limits(x, exposure, method, tail, z, correct):
    if method == "exact":
        lower = chi2.ppf(tail, 2 * x) / 2 if x else 0.0
        upper = float(chi2.isf(tail, 2 * (x + 1))) / 2
        return lower / exposure, upper / exposure
    if method == "score":
        return _score_limits(x, exposure, z, correct)
    if method == "wh":
        a, b = (x + 0.5, x + 0.5) if correct else (x, x + 1)
        lower = a * (1 - 1 / (9 * a) - z / (3 * math.sqrt(a)))**3 if a else 0
        upper = b * (1 - 1 / (9 * b) + z / (3 * math.sqrt(b)))**3
        return max(0.0, lower) / exposure, max(0.0, upper) / exposure
    if not x and (method == "log" or not correct):
        return 0.0, -math.log(tail) / exposure
    if method == "log":
        log_rate = math.log(x) - math.log(exposure)
        return (_exp(log_rate - z / math.sqrt(x)),
                _exp(log_rate + z / math.sqrt(x)))
    a, b = (max(0, x - 0.5), x + 0.5) if correct else (x, x)
    return max(0.0, a - z * math.sqrt(a)) / exposure, (
        max(0.0, b + z * math.sqrt(b)) / exposure)


def _rate_ratio(counts, exposure):
    if counts[1] == 0:
        return math.inf if counts[0] else math.nan
    if counts[0] == 0:
        return 0.0
    return _exp(math.log(int(counts[0])) - math.log(exposure[0])
                - math.log(int(counts[1])) + math.log(exposure[1]))


def _scaled_odds(p, log_scale, reciprocal=False):
    if p <= 0:
        return math.inf if reciprocal else 0.0
    if p >= 1:
        return 0.0 if reciprocal else math.inf
    log_odds = math.log(p) - math.log1p(-p)
    return _exp(log_scale - log_odds if reciprocal else log_scale + log_odds)


def _log_ratio(counts, exposure, z):
    if np.any(counts == 0):
        raise ValueError("Log-Wald requires positive event counts "
                         "in both samples.")
    log_ratio = (math.log(int(counts[0])) - math.log(exposure[0])
                 - math.log(int(counts[1])) + math.log(exposure[1]))
    margin = z * math.sqrt(float(np.sum(1 / counts)))
    return _exp(log_ratio - margin), _exp(log_ratio + margin)
