"""Confidence intervals for binomial probabilities."""

from collections.abc import Sequence
import math
from numbers import Real

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import beta, binom, norm

from .contingency import _flag
from .odds_ratio import _probability
from .results import CIResult, ConfidenceInterval


_METHODS = ("wilson", "wald", "agresti-coull", "clopper-pearson", "blaker")


def prop_1s_ci(x, n, *, conf_level=0.95, method="wilson", correct=False,
               midp=False, alternative="two.sided", tol=1e-8) -> CIResult:
    """Estimate binomial probabilities with one or more interval constructions.

    Parameters
    ----------
    x : integer or one-dimensional array_like
        Nonnegative success counts. A vector can describe separate samples
        or enumerate possible counts for a coverage calculation.
    n : integer or one-dimensional array_like
        Positive trial counts. A scalar (or length-one vector) is recycled;
        otherwise its length must equal that of x. Require x <= n and
        n <= 2**53-1 to preserve integer precision in numerical calculations.
    conf_level : float, default 0.95
        Confidence level strictly between zero and one.
    method : str or sequence of str, default "wilson"
        "wilson", "wald", "agresti-coull", "clopper-pearson", or "blaker".
        Requested order and repetitions are retained, matching R prop.1s.ci.
    correct : bool, default False
        Full half-count continuity correction for Wilson and Wald only.
    midp : bool, default False
        Use mid-p tails for Clopper--Pearson only. Other methods in a mixed
        request are unchanged. Mid-p is not coverage-exact.
    alternative : {"two.sided", "less", "greater"}, default "two.sided"
        Two limits, an upper bound with lower limit zero, or a lower bound
        with upper limit one. A one-sided bound uses the full tail error.
    tol : float, default 1e-8
        Positive numerical endpoint tolerance, below one. Mid-p uses this
        tolerance on the logit scale, implying a tighter absolute probability
        tolerance. Closed-form limits do not use numerical inversion.

    Returns
    -------
    CIResult
        The observed estimate x/n (a float or one-dimensional NumPy array),
        full-precision conf_int, method labels, alternative, correct, midp,
        and per-interval inputs. Rows follow input order and then method order.
        Scalar to_frame() output retains the method index; vectors have a
        zero-based (input, method) MultiIndex. No hypothesis test is fabricated.

    Notes
    -----
    Wilson inverts the score equation. Correction uses (x-0.5)/n for the
    lower limit and (x+0.5)/n for the upper limit, clipped to [0,1]. It uses
    the full correction independently of any null probability; R prop.test
    can cap its correction according to its null hypothesis. Wald adds
    0.5/n to its margin when corrected. Agresti--Coull uses n+z**2 trials and
    x+z**2/2 successes internally, retaining x/n as the reported estimate.
    All limits are restricted to [0,1].

    Clopper--Pearson inverts inclusive binomial tails. Mid-p subtracts half
    the observed-count probability from each tail. Blaker encloses the
    two-sided acceptance set from smaller-inclusive-tail ordering; the
    numerical search partitions at changes in that ordering and does not
    assume a continuous two-sided p-value. A one-sided Blaker bound is the
    corresponding exact binomial-tail bound, not half a two-sided Blaker
    interval. Numerical Blaker endpoints are rounded outward by tol to keep
    accepted ordering ties inside the closed interval. Two-sided Blaker
    computation can be slower for large samples;
    an inversion requiring more than 1,000,000 ordering changes is rejected.
    Prefer Wilson for large coverage simulations. Repeated count/method
    combinations are cached within a call.

    A flag affecting none of the selected methods is an error. Request
    mid-p with method="clopper-pearson", not method="midp"; request corrected
    Wilson/Wald via correct=True, not alternative method names.

    References
    ----------
    Wilson (1927), JASA 22, 209--212.
    Clopper and Pearson (1934), Biometrika 26, 404--413.
    Agresti and Coull (1998), The American Statistician 52, 119--126.
    Blaker (2000), Canadian Journal of Statistics 28, 783--798;
    correction (2001), 29, 681.
    Brown, Cai, and DasGupta (2001), Statistical Science 16, 101--133.

    Examples
    --------
    >>> result = prop_1s_ci(5, 40, method=["wilson", "clopper-pearson"])
    >>> result.estimate
    0.125
    >>> prop_1s_ci([0, 5, 40], 40).inputs["input"].tolist()
    [0, 1, 2]
    """
    successes, trials = _binomial_counts(x, n)
    conf_level = _probability(conf_level, "conf_level")
    correct, midp = _flag(correct, "correct"), _flag(midp, "midp")
    if not isinstance(alternative, str) or alternative not in ("two.sided", "less", "greater"):
        raise ValueError("alternative must be two.sided, less, or greater.")
    if (isinstance(tol, (bool, np.bool_)) or not isinstance(tol, Real)
            or not math.isfinite(tol) or not 0 < tol < 1):
        raise ValueError("tol must be a finite number between zero and one.")
    if isinstance(method, str):
        methods = [method]
    elif isinstance(method, Sequence):
        methods = list(method)
    else:
        raise ValueError("method must be a name or sequence of names.")
    if not methods or any(not isinstance(m, str) or m not in _METHODS for m in methods):
        raise ValueError(f"method must select one or more of {_METHODS}.")
    if correct and not any(m in ("wald", "wilson") for m in methods):
        raise ValueError("correct applies only to Wald and Wilson intervals.")
    if midp and "clopper-pearson" not in methods:
        raise ValueError("midp applies only to Clopper-Pearson intervals.")
    tail = (1 - conf_level) / (2 if alternative == "two.sided" else 1)
    z = float(norm.isf(tail))
    intervals, inputs, cache = [], [], {}
    for i, (count, size) in enumerate(zip(successes, trials)):
        count, size = int(count), int(size)
        for name in methods:
            key = (count, size, name)
            if key not in cache:
                cache[key] = _binomial_limits(count, size, name, tail, z,
                                              correct, midp, alternative, float(tol))
            lower, upper = cache[key]
            label = name
            if correct and name in ("wald", "wilson"):
                label += "-cc"
            if midp and name == "clopper-pearson":
                label += "-midp"
            intervals.append(ConfidenceInterval(label, lower, upper))
            inputs.append((i, count, size))
    estimates = successes / trials
    estimate = float(estimates[0]) if len(estimates) == 1 else estimates.copy()
    return CIResult(estimate, tuple(intervals), conf_level, "binomial proportion",
                    alternative=alternative, correct=correct, midp=midp,
                    _input_data=tuple(inputs))


def _binomial_counts(x, n):
    def counts(value, name, positive=False):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be scalar or one-dimensional integer counts.") from None
        if (array.ndim > 1 or array.size == 0 or array.dtype.kind not in "iuf"
                or not np.all(np.isfinite(array)) or np.any(array < 0)
                or np.any(array != np.floor(array))
                or (positive and np.any(array == 0))):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} must contain finite {qualifier} integer "
                             "counts in a scalar or one-dimensional array.")
        if np.any(array > 2 ** 53 - 1):
            raise ValueError(f"{name} counts must not exceed 2**53-1.")
        return array.reshape(-1).astype(np.int64)

    successes, trials = counts(x, "x"), counts(n, "n", positive=True)
    if len(trials) not in (1, len(successes)):
        raise ValueError("n must be scalar or have the same length as x.")
    if len(trials) == 1:
        trials = np.repeat(trials, len(successes))
    if np.any(successes > trials):
        raise ValueError("x must not exceed n.")
    return successes, trials


def _binomial_limits(x, n, method, tail, z, correct, midp, alternative, tol):
    p = x / n
    if method == "wilson":
        offset = 0.5 / n if correct else 0

        def endpoint(q, direction):
            return (q + z * z / (2 * n) + direction * z
                    * math.sqrt(q * (1 - q) / n + z * z / (4 * n * n))) / (1 + z * z / n)

        lower = 0.0 if x == 0 else endpoint(max(0.0, p - offset), -1)
        upper = 1.0 if x == n else endpoint(min(1.0, p + offset), 1)
    elif method in ("wald", "agresti-coull"):
        size = n + z * z if method == "agresti-coull" else n
        center = (x + z * z / 2) / size if method == "agresti-coull" else p
        margin = z * math.sqrt(center * (1 - center) / size)
        if correct and method == "wald":
            margin += 0.5 / n
        lower, upper = center - margin, center + margin
    elif method == "blaker" and alternative == "two.sided":
        lower = _blaker_lower(x, n, 2 * tail, tol)
        upper = 1 - _blaker_lower(n - x, n, 2 * tail, tol)
    elif method == "clopper-pearson" and midp:
        lower, upper = _midp_limits(x, n, tail, tol)
    else:
        lower = 0.0 if x == 0 else float(beta.ppf(tail, x, n - x + 1))
        upper = 1.0 if x == n else float(beta.isf(tail, x + 1, n - x))
    lower, upper = max(0.0, min(1.0, lower)), max(0.0, min(1.0, upper))
    if alternative == "greater":
        upper = 1.0
    if alternative == "less":
        lower = 0.0
    return float(lower), float(upper)


def _midp_limits(x, n, tail, tol):
    def lower_tail(eta):
        p = expit(eta)
        return float(binom.sf(x, n, p) + 0.5 * binom.pmf(x, n, p) - tail)

    def upper_tail(eta):
        p = expit(eta)
        return float(binom.cdf(x - 1, n, p) + 0.5 * binom.pmf(x, n, p) - tail)

    if x == 0:
        lower = 0.0
    elif lower_tail(750) <= 0:
        lower = 1.0
    else:
        lower = float(expit(brentq(lower_tail, -750, 750, xtol=tol, rtol=1e-14)))
    if x == n:
        upper = 1.0
    elif upper_tail(-750) <= 0:
        upper = 0.0
    else:
        upper = float(expit(brentq(upper_tail, -750, 750, xtol=tol, rtol=1e-14)))
    return lower, upper


def _blaker_lower(x, n, alpha, tol):
    if x == 0:
        return 0.0
    lower = float(beta.ppf(alpha / 2, x, n - x + 1))
    upper = min(x / n, float(beta.ppf(alpha, x, n - x + 1)))
    def outward(value):
        # Keep accepted jump points inside the closed interval, even when
        # a numerical root rounds just above the true lower boundary.
        return max(0.0, float(np.nextafter(value - tol, -math.inf)))

    if lower == upper:
        return outward(lower)

    def right(p):
        return float(binom.sf(x - 1, n, p))

    first = max(0, int(binom.ppf(right(lower), n, lower)) - 1)
    last = min(x - 1, int(binom.ppf(right(upper), n, upper)))
    if last - first + 1 > 1000000:
        raise ValueError("Blaker inversion exceeds 1,000,000 ordering changes; "
                         "use Wilson for large samples.")
    breaks = [lower, upper]
    for k in range(first, last + 1):
        def crossing(p):
            return float(binom.cdf(k, n, p)) - right(p)

        if crossing(lower) > 0 and crossing(upper) < 0:
            breaks.append(brentq(crossing, lower, upper, xtol=tol / 10, rtol=1e-14))
    breaks = sorted(set(breaks))
    for left, right_bound in zip(breaks[:-1], breaks[1:]):
        middle = (left + right_bound) / 2
        k = min(x - 1, int(binom.ppf(right(middle), n, middle)))
        if binom.cdf(k, n, middle) > right(middle):
            k -= 1

        def acceptability(p):
            return right(p) + float(binom.cdf(k, n, p)) - alpha

        if acceptability(left) >= -1e-14:
            return outward(left)
        if acceptability(right_bound) >= 0:
            root = brentq(acceptability, left, right_bound, xtol=tol, rtol=1e-14)
            return outward(root)
    return outward(upper)
