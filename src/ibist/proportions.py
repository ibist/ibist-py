"""Confidence intervals for binomial probabilities."""

from collections.abc import Sequence
import math
from numbers import Real

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import beta, binom, norm

from .contingency import _flag
from .odds_ratio import _probability, _validate_table
from .results import CIResult, ConfidenceInterval, _ci_orientation


_METHODS = ("wilson", "wald", "agresti-coull", "clopper-pearson", "blaker")


def prop_1s_ci(x, n, *, conf_level=0.95, method="wilson", correct=False,
               midp=False, alternative=None, side=None, tol=1e-8,
               asymcheck=False) -> CIResult:
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
    side : {"two.sided", "lower", "upper"}, optional
        Omitted or None selects two-sided limits. One-sided bounds use the
        full error probability. Lower bounds have upper endpoint 1;
        upper bounds have lower endpoint 0.
    alternative : {"two.sided", "less", "greater"}, optional
        Compatibility alias: greater selects lower, less selects upper.
        Conflicting side and alternative values are rejected.
    tol : float, default 1e-8
        Positive numerical endpoint tolerance, below one. Mid-p uses this
        tolerance on the logit scale, implying a tighter absolute probability
        tolerance. Closed-form limits do not use numerical inversion.

    asymcheck : bool, default False
        Attach the book's approximation checks in result.diagnostics.
        See asymcheck() for criteria, sources, and interpretation.

    Returns
    -------
    CIResult
        The observed estimate x/n (a float or one-dimensional NumPy array),
        full-precision conf_int, method labels, side, correct, midp,
        and per-interval inputs. Rows follow input order and then method order.
        Scalar to_frame() output retains the method index; vectors have a
        zero-based (input, method) MultiIndex. No hypothesis test is fabricated.

    Raises
    ------
    ValueError
        If counts, options, or method-specific flags are invalid, or a
        Blaker inversion exceeds one million ordering changes.

    See Also
    --------
    prop_2s_ci : Intervals for an independent proportion difference.
    prop_paired_ci : Intervals for a paired proportion difference.

    Notes
    -----
    The book's Chapter 11 binomial-normal guideline checks
    n*p_hat*(1-p_hat) >= 5 using original counts. Corrections and pseudo-counts
    do not increase that quantity. Passing does not guarantee coverage;
    Wald coverage can be poor even when the guideline passes. Exact and mid-p
    methods need no asymptotic check; mid-p does not guarantee coverage.

    Original counts and settings are retained for asymcheck(result), even
    when attached diagnostics are not requested. Pre-checks run no inference;
    diagnostics do not change methods, p-values, or confidence limits.

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
    asymcheck = _flag(asymcheck, "asymcheck")
    side, alternative = _ci_orientation(side, alternative)
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
    result = CIResult(estimate, tuple(intervals), conf_level,
                      "binomial proportion",
                    alternative=alternative, correct=correct, midp=midp,
                    _input_data=tuple(inputs))
    from ._asymcheck_book import _attach
    return _attach(result, "p1", asymcheck, x=successes, n=trials,
                   method=methods, conf_level=conf_level, correct=correct,
                   midp=midp, side=side, tol=tol)


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


_DIFFERENCE_METHODS = ("newcombe", "wald", "miettinen-nurminen", "mee",
                       "agresti-caffo", "brown-li", "chan-zhang")


def prop_2s_ci(x, n=None, *, conf_level=0.95, method="newcombe", correct=False,
               alternative=None, side=None, tol=1e-8,
               nuisance_tol=1e-10, asymcheck=False) -> CIResult:
    """Estimate the first independent population proportion minus the second.

    Parameters
    ----------
    x : length-two array_like, or 2 by 2 count table
        Two success counts, or a table with independent samples in rows and
        successes/failures in columns. Empty event columns are allowed.
    n : length-two array_like, optional
        Positive trial counts for success-vector input; omit for a table.
        A scalar is not recycled for two independent samples. Each trial
        count must not exceed 2**53-1, as for prop_1s_ci().
    conf_level : float, default 0.95
        Confidence level strictly between zero and one.
    method : str or sequence of str, default "newcombe"
        "newcombe", "wald", "miettinen-nurminen", "mee", "agresti-caffo",
        "brown-li", or "chan-zhang". Preserve order and repetitions.
    correct : bool, default False
        Full continuity correction for Newcombe and Wald only. Other
        selected methods are unchanged. An entirely inapplicable flag is
        an error; this flag never switches Mee and Miettinen--Nurminen.
    side : {"two.sided", "lower", "upper"}, optional
        Omitted or None selects two-sided limits. One-sided bounds use the
        full error probability. Lower bounds have upper endpoint 1;
        upper bounds have lower endpoint -1.
    alternative : {"two.sided", "less", "greater"}, optional
        Compatibility alias: greater selects lower, less selects upper.
        Conflicting side and alternative values are rejected.
    tol : float, default 1e-8
        Positive numerical tolerance below one. Score inversion uses this
        tolerance on a logit-scaled distance to the observed difference.
        This improves accuracy for narrow intervals; attainable endpoint
        accuracy is also limited by floating-point precision. Constrained
        likelihood fits use a tighter tolerance.

    nuisance_tol : float, default 1e-10
        Probability tolerance for Chan--Zhang global nuisance maximization.
        Ambiguous regions are retained conservatively. Enumeration is limited
        to one million possible tables and may be computationally expensive.
        The outer search encloses disconnected acceptance sets; it does not
        assume monotone tails or use a fixed nuisance grid.

    asymcheck : bool, default False
        Attach the book's approximation checks in result.diagnostics.
        See asymcheck() for criteria, sources, and interpretation.

    Returns
    -------
    CIResult
        The observed proportion difference, full-precision conf_int,
        method labels, side, correct, null_value=0, and fresh
        sample_proportions and samples tables. Each interval row represents
        one requested method. No statistic or p-value is manufactured.

    Raises
    ------
    ValueError
        If counts or options are invalid, or Chan--Zhang enumeration
        exceeds one million possible tables.

    See Also
    --------
    prop_1s_ci : Intervals for one binomial probability.
    prop_paired_ci : Intervals for a paired proportion difference.
    rr_2x2_ci : Intervals for a ratio of independent binomial risks.

    Notes
    -----
    Approximation checks use the book's Chapter 12 guideline:
    n_i*p_hat_i*(1-p_hat_i) >= 5 in each original sample. Pseudo-counts and
    continuity corrections never increase these quantities. Passing is not
    a coverage guarantee. Chan--Zhang needs no asymptotic check.

    Original counts and settings are retained for asymcheck(result), even
    when attached diagnostics are not requested. Pre-checks run no inference;
    diagnostics do not change methods, p-values, or confidence limits.

    The six approximate methods and Chan--Zhang exact unconditional tail
    intervals match R. Newcombe combines separate Wilson distances in
    quadrature, using corrected Wilson limits if requested. It does not
    invert the pooled score test for equality. Wald uses the unpooled
    binomial variance; correction adds 1/(2*n1)+1/(2*n2) to the margin.

    Mee and Miettinen--Nurminen invert the score statistic using binomial
    likelihood estimates constrained by p1-p2=delta. Only Miettinen--Nurminen
    multiplies the constrained variance by (n1+n2)/(n1+n2-1). This is a
    finite-sample factor, not the continuity-correction option.

    Agresti--Caffo adds one success and one failure to each sample before
    forming a Wald interval. Brown--Li uses adjusted proportions
    (x_i+0.5)/(n_i+1) and a normal margin with variance denominators equal
    to the ORIGINAL trial counts n_i. It is not a posterior-quantile
    interval. The common reported estimate remains the observed difference.
    All limits are restricted to [-1,1]. One-sided bounds use the full tail
    error probability. A 2 by 2 table is converted to per-row success/trial
    counts without requiring positive column totals.

    References
    ----------
    Newcombe (1998), Statistics in Medicine 17, 873--890.
    Miettinen and Nurminen (1985), Statistics in Medicine 4, 213--226.
    Mee (1984), Biometrics 40, 1175--1176.
    Agresti and Caffo (2000), The American Statistician 54, 280--288.
    Chan and Zhang (1999), Biometrics 55, 1202--1209.
    Brown and Li (2005), Journal of Statistical Planning and Inference 130,
    359--375, doi:10.1016/j.jspi.2003.09.039.

    Examples
    --------
    >>> result = prop_2s_ci([9, 3], [10, 10], method=["newcombe", "brown-li"])
    >>> round(result.estimate, 3)
    0.6
    >>> prop_2s_ci([[9, 1], [3, 7]]).sample_proportions.tolist()
    [0.9, 0.3]
    """
    asymcheck = _flag(asymcheck, "asymcheck")
    side, alternative = _ci_orientation(side, alternative)
    counts, sizes = _independent_counts(x, n)
    conf_level = _probability(conf_level, "conf_level")
    correct = _flag(correct, "correct")
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
    if not methods or any(not isinstance(m, str) or m not in _DIFFERENCE_METHODS
                          for m in methods):
        raise ValueError(f"method must select one or more of {_DIFFERENCE_METHODS}.")
    if correct and not any(m in ("newcombe", "wald") for m in methods):
        raise ValueError("correct applies only to Newcombe and Wald intervals.")
    nuisance_tol = _probability(nuisance_tol, "nuisance_tol")
    proportions = counts / sizes
    estimate = float(proportions[0] - proportions[1])
    tail = (1 - conf_level) / (2 if alternative == "two.sided" else 1)
    z = float(norm.isf(tail))
    intervals = []
    for name in methods:
        if name == "chan-zhang":
            from ._independent_exact import _independent_exact_limits
            limits = _independent_exact_limits(counts, sizes, tail,
                                                alternative, tol,
                                                nuisance_tol)
        elif name == "newcombe":
            one = np.array(_binomial_limits(int(counts[0]), int(sizes[0]), "wilson",
                                             tail, z, correct, False, "two.sided", tol))
            two = np.array(_binomial_limits(int(counts[1]), int(sizes[1]), "wilson",
                                             tail, z, correct, False, "two.sided", tol))
            distances = np.sqrt((one - proportions[0]) ** 2
                                 + (two[::-1] - proportions[1]) ** 2)
            limits = estimate + np.array([-1, 1]) * np.sign(z) * distances
        elif name in ("miettinen-nurminen", "mee"):
            total = int(sizes[0]) + int(sizes[1])
            factor = total / (total - 1) if name == "miettinen-nurminen" else 1.0
            limits = _independent_score_limits(counts, sizes, z, tol, factor)
        else:
            if name == "brown-li":
                centers = (counts.astype(float) + 0.5) / (sizes.astype(float) + 1)
                variance_sizes = sizes
            elif name == "agresti-caffo":
                variance_sizes = sizes.astype(float) + 2
                centers = (counts.astype(float) + 1) / variance_sizes
            else:
                variance_sizes, centers = sizes, proportions
            margin = z * float(np.sqrt(np.sum(centers * (1 - centers) / variance_sizes)))
            if correct and name == "wald":
                margin += float(np.sum(0.5 / sizes))
            center = float(centers[0] - centers[1])
            limits = np.array([center - margin, center + margin])
        lower, upper = np.clip(limits, -1.0, 1.0)
        if alternative == "greater":
            upper = 1.0
        if alternative == "less":
            lower = -1.0
        label = name + ("-cc" if correct and name in ("newcombe", "wald") else "")
        intervals.append(ConfidenceInterval(label, float(lower), float(upper)))
    samples = tuple((int(a), int(b)) for a, b in zip(counts, sizes))
    result = CIResult(estimate, tuple(intervals), conf_level,
                    "difference of independent proportions", null_value=0.0,
                    alternative=alternative, correct=correct, _sample_counts=samples)
    from ._asymcheck_book import _attach
    return _attach(result, "p2", asymcheck, x=counts, n=sizes,
                   method=methods, conf_level=conf_level, correct=correct,
                   side=side, tol=tol, nuisance_tol=nuisance_tol)


def _independent_counts(x, n):
    try:
        raw = np.asarray(x)
    except (TypeError, ValueError):
        raise ValueError("x must contain two counts or be a 2 x 2 table.") from None
    if raw.ndim == 2:
        if n is not None:
            raise ValueError("Omit n when x is a 2 x 2 table.")
        table = _validate_table(x)
        counts, sizes = table[:, 0], table.sum(axis=1)
    else:
        counts, sizes = x, n
    if np.ndim(counts) != 1 or np.ndim(sizes) != 1:
        raise ValueError("x and n must each contain two counts.")
    if len(counts) != 2 or len(sizes) != 2:
        raise ValueError("x and n must each contain two counts.")
    return _binomial_counts(counts, sizes)


def _constrained_probabilities(counts, sizes, delta, tol):
    # Interpolate successes and failures separately, avoiding loss of tiny
    # failure probabilities when a fitted success probability rounds to one.
    width = 1 - abs(delta)

    def probabilities(u, v):
        if delta >= 0:
            return (np.array([delta + width * u, width * u]),
                    np.array([width * v, delta + width * v]))
        return (np.array([width * u, -delta + width * u]),
                np.array([-delta + width * v, width * v]))

    failures = sizes - counts

    def derivative(u, v):
        p, f = probabilities(u, v)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            success_terms = np.divide(counts, p, out=np.zeros(2), where=counts > 0)
            failure_terms = np.divide(failures, f, out=np.zeros(2), where=failures > 0)
        return float(success_terms.sum() - failure_terms.sum())

    if derivative(0.0, 1.0) <= 0:
        return probabilities(0.0, 1.0)
    if derivative(1.0, 0.0) >= 0:
        return probabilities(1.0, 0.0)
    eta = brentq(lambda t: derivative(expit(t), expit(-t)), -750, 750,
                 xtol=tol / 100, rtol=1e-14, maxiter=200)
    return probabilities(expit(eta), expit(-eta))


def _independent_score_limits(counts, sizes, z, tol, variance_factor):
    estimate = float(counts[0] / sizes[0] - counts[1] / sizes[1])

    def score(delta):
        if delta <= -1:
            return 0.0 if estimate == -1 else math.inf
        if delta >= 1:
            return 0.0 if estimate == 1 else -math.inf
        p, failure = _constrained_probabilities(counts, sizes, delta, tol)
        variance = float(np.sum(p * failure / sizes)) * variance_factor
        if variance <= 0:
            return 0.0 if estimate == delta else math.copysign(math.inf, estimate - delta)
        return (estimate - delta) / math.sqrt(variance)

    def limit(target):
        if target == 0:
            return estimate
        direction = -1 if target > 0 else 1
        distance = estimate + 1 if direction == -1 else 1 - estimate
        if distance == 0:
            return estimate

        def difference(eta):
            fraction = expit(eta)
            if fraction == 1:
                return float(direction)
            return estimate + direction * distance * fraction

        eta = brentq(lambda t: score(difference(t)) - target, -750, 750,
                     xtol=tol, rtol=1e-14, maxiter=200)
        return difference(eta)

    return np.array([limit(z), limit(-z)])
