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
               correct=True, alternative=None, side=None) -> CIResult:
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
        Compatibility alias. Conflicting side and alternative values fail.

    Returns
    -------
    CIResult
        Scalar rate estimate, intervals, and counts/exposure metadata in
        samples. sample_rates returns a fresh array of the observed rates.

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
    return CIResult(count / time, tuple(intervals), conf_level,
                    "Poisson rate", alternative=alternative, correct=correct,
                    _rate_data=((count, time),))


def rate_2s_ci(x, exposure=(1.0, 1.0), *, conf_level=0.95, method="log",
               alternative=None, side=None) -> CIResult:
    """Estimate a ratio of two independent Poisson rates.

    Counts and exposures must be length-two vectors. Methods are "log"
    (default, log-Wald, requiring positive counts), "score" (transformed
    Wilson), and "exact" (transformed Clopper--Pearson). Request a string
    or a sequence of methods; order and repetitions are retained.

    Both exposures must be positive and at least one count must be positive.
    Scalar exposures are not recycled. Conditional methods use the binomial
    count x1 given x1+x2, then transform its probability p into a rate ratio
    p*T2/((1-p)*T1). They allow a zero count and give boundary limits of zero
    or infinity. All counts and their total must not exceed 2**53-1.

    conf_level defaults to 0.95. side is "two.sided" (default), "lower",
    or "upper". One-sided bounds use the full error probability. None selects
    two-sided limits. alternative is a compatibility alias: "greater" selects
    lower, and "less" selects upper; conflicting arguments are errors.

    Returns a CIResult with a scalar ratio estimate, sample_rates, and
    samples containing x and exposure. This separates the inference target
    from the two group rates; R stores all three in its estimate vector.

    References
    ----------
    Clopper and Pearson (1934), Biometrika 26, 404--413, for the conditional
    exact binomial bounds. Wilson (1927), JASA 22, 209--212, for the score
    bounds. The exposure transformation follows ibist-R rate.2s.ci().
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

    x and exposure are matching scalars or length-one/two vectors. r is the
    positive null unit rate for one sample, default 1; it is ignored for two
    samples, whose null is equal rates. conf_level defaults to 0.95, correct
    to True, and alternative to "two.sided"; "less" and "greater" are also
    supported. Counts and their total must not exceed 2**53-1.

    One sample uses Z=(x-cc-r*T)/sqrt(r*T), with a directional half-count
    correction when requested, and reports a matching score interval.
    The correction is not capped, following R rate.test(). Two samples
    use the conditional binomial Pearson statistic with the same capped
    Yates correction as R prop.test(). Their interval is log-Wald for the
    rate ratio and requires positive event counts in both groups. Correction
    affects the two-sample test but not that interval. Test and interval
    decisions can therefore differ. Small expected binomial cell counts
    trigger a warning, matching R.

    Returns TestResult with statistic (signed Z for one sample, chi-square
    for two), p_value, df=1, scalar estimate, null_value, conf_int tuple,
    conf_level, alternative, and fresh samples/sample_rates metadata.
    No exact Poisson hypothesis test is substituted for the approximation.
    The statistical formulas and correction rules follow ibist-R rate.test().
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
