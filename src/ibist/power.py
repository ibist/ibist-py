"""R-compatible power and design calculations for binomial proportions."""

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Real
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom, norm

from .contingency import _digits, _flag
from .odds_ratio import _probability


@dataclass(frozen=True, repr=False)
class PowerResult:
    """Resolved design parameters corresponding to R's power.htest.

    Attributes
    ----------
    n, power, sig_level : float
        Sample size, power, and nominal significance level. For two samples,
        n is the first group size. Normal designs return continuous n.
    alternative, method : str
        Alternative hypothesis and calculation description.
    p0, p1, p2 : float or None
        One-sample null/alternative proportions or two-sample proportions.
    achieved_sig_level : float or None
        Exact null rejection probability; None for normal calculations.
    note : str or None
        Two-sample allocation/correction description, matching R.

    Notes
    -----
    Power is computed when omitted. For inverse calculations it records the
    requested target, except exact sample-size designs return achieved power,
    as in R. Stored values retain full precision; summary() rounds display.
    """

    n: float
    power: float
    sig_level: float
    alternative: str
    method: str
    p0: float | None = None
    p1: float | None = None
    p2: float | None = None
    achieved_sig_level: float | None = None
    note: str | None = None

    def summary(self, digits=5):
        """Display resolved parameters without changing their precision."""
        _digits(digits)
        fields = ("n", "p0", "p1", "p2", "sig_level", "achieved_sig_level",
                  "power", "alternative")
        lines = [self.method]
        for name in fields:
            value = getattr(self, name)
            if value is not None:
                text = (value if isinstance(value, str) else
                        f"{value:.{digits}g}")
                lines.append(f"{name}: {text}")
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.summary()


def power_p1s_test(n=None, p0=None, p1=None, power=None, *, sig_level=0.05,
                   alternative="two.sided", correct=False, exact=False,
                   exact_method="quantile", strict=False,
                   tol=np.finfo(float).eps**0.5, max_n=10000000,
                   size_rule="close", alpha_min_frac=0.9):
    """Compute one-sample proportion power or solve one design parameter.

    Parameters
    ----------
    n : positive float, optional
        Supplied sample size is rounded up, matching R. Normal sample-size
        solutions remain continuous; exact solutions are integers.
    p0, p1 : float, optional
        Null and alternative proportions, strictly between zero and one.
    power : float, optional
        Target power in (0, 1). Omit to compute power. Exactly one of n, p0,
        p1, power, sig_level must be None.
    sig_level : float or None, default 0.05
        Nominal significance level in (0, 1); None requests inversion.
    alternative : {"two.sided", "less", "greater"}, default "two.sided"
        Direction relative to the null proportion.
    correct : bool, default False
        Half-count continuity correction for normal calculations.
    exact : bool, default False
        Use a binomial rejection region instead of a normal approximation.
    exact_method : {"quantile", "midp", "midp-rand", "cp"}, default "quantile"
        Quantile critical counts, deterministic central mid-p rejection,
        randomized half-mass boundaries, or inclusive exact binomial tails.
    strict : bool, default False
        Include both tails of two-sided normal power; otherwise use the
        dominant tail. Exact methods always use their specified regions.
    tol : positive float, default sqrt(machine epsilon)
        Root-finding tolerance, matching R.
    max_n : positive number, default 10000000
        Integer bound for exact sample-size search. As in R, the normal
        solver uses this as its initial upper bracket and can extend it.
    size_rule : {"close", "minimal"}, default "close"
        Exact design requires actual size <= sig_level and sufficient power.
        Close also requires size >= alpha_min_frac*sig_level.
    alpha_min_frac : float, default 0.9
        Fraction in (0, 1) defining the close rule.

    Returns
    -------
    PowerResult
        Resolved parameters and procedure description, mirroring R's
        power.htest; exact results also include achieved_sig_level.

    Raises
    ------
    ValueError
        For invalid inputs, an unbracketed inverse, or no feasible exact
        design up to max_n. One-sample integer sizes are limited to 2**31-1,
        following R's integer representation.

    Warns
    -----
    UserWarning
        If a nondefault exact_method is supplied with exact=False.

    See Also
    --------
    power_p2s_test : Two-sample power with unequal group sizes.

    Notes
    -----
    Matches R power.p1s.test(). Quantile critical counts are qbinom(tail)-1
    and qbinom(1-tail)+1. Midp-rand assigns half rejection probability at
    those boundaries; deterministic midp instead rejects on its mid-p value.
    CP uses inclusive binomial tails. Actual size is always the full null
    rejection probability, including randomized boundary contributions.
    Mid-p does not guarantee nominal size control. Exact design searches
    therefore check actual size explicitly and scan n in order: discrete
    power and size are not monotone. Large searches can be expensive.

    Two-sided p1 inversion first seeks p1>p0; p0 inversion first seeks p0<p1.
    Exact inversions for p0 or sig_level are discontinuous and can locate a
    rejection-region boundary without attaining the target exactly. The
    returned power follows R's target convention; recompute with power=None
    to obtain achieved power at the returned parameters. Normal upper tails
    use survival functions to preserve small probabilities.

    References
    ----------
    R ibist power.p1s.test(), Details and exact rejection-region definitions.
    R stats power.prop.test(), Details, for normal power conventions.
    Clopper and Pearson (1934), Biometrika 26, 404--413.

    Examples
    --------
    >>> round(power_p1s_test(n=50, p0=0.1, p1=0.25).power, 7)
    0.8624926
    >>> power_p1s_test(p0=.01, p1=.99, power=.8, alternative="greater",
    ...                exact=True, size_rule="minimal", max_n=100).n
    1
    """
    values = dict(n=n, p0=p0, p1=p1, power=power, sig_level=sig_level)
    missing = _validate(values)
    correct, exact, strict = (_flag(value, name) for value, name in
                              [(correct, "correct"), (exact, "exact"),
                               (strict, "strict")])
    alternative = _choice(alternative, ("two.sided", "less", "greater"),
                           "alternative")
    exact_method = _choice(
        exact_method, ("quantile", "midp", "midp-rand", "cp"), "exact_method")
    size_rule = _choice(size_rule, ("close", "minimal"), "size_rule")
    tol = _positive(tol, "tol")
    max_n = math.floor(_positive(max_n, "max_n"))
    if not 1 <= max_n <= 2**31 - 1:
        raise ValueError("max_n must be between 1 and 2**31-1.")
    alpha_min_frac = _probability(alpha_min_frac, "alpha_min_frac")
    if n is not None:
        values["n"] = math.ceil(n)
        if values["n"] > 2**31 - 1:
            raise ValueError("n must not exceed R's integer limit 2**31-1.")
    if not exact and exact_method != "quantile":
        warnings.warn("exact_method ignored when exact=False.", UserWarning,
                      stacklevel=2)

    def evaluate(**change):
        args = {**values, **change}
        nn, null, alt, alpha = (args[key] for key in
                                ("n", "p0", "p1", "sig_level"))
        if exact:
            return _exact_power(int(nn), null, alt, alpha, alternative,
                                 exact_method)
        se0 = math.sqrt(null * (1 - null) / nn)
        se1 = math.sqrt(alt * (1 - alt) / nn)
        cc = 0.5 / nn if correct else 0
        z = norm.isf(alpha / (2 if alternative == "two.sided" else 1))
        lower = norm.cdf((null - z * se0 - cc - alt) / se1)
        upper = norm.sf((null + z * se0 + cc - alt) / se1)
        value = (upper if alternative == "greater" else
                 lower if alternative == "less" else
                 lower + upper if strict else upper if alt >= null else lower)
        return float(value), alpha

    if missing == "power":
        values["power"] = evaluate()[0]
    elif missing == "n" and exact:
        null, alt, alpha, target = (values[k] for k in
                                    ("p0", "p1", "sig_level", "power"))
        impossible = (null == alt or alternative == "greater" and alt < null
                      or alternative == "less" and alt > null)
        if impossible and target > alpha:
            raise ValueError("Target power is unattainable in this direction.")
        for nn in range(1, max_n + 1):
            actual, size = evaluate(n=nn)
            if size <= alpha and actual >= target and (
                    size_rule == "minimal" or size >= alpha_min_frac * alpha):
                values["n"], values["power"] = nn, actual
                break
        else:
            raise ValueError("No feasible n up to max_n; consider minimal "
                             "or a smaller alpha_min_frac.")
    elif missing == "n":
        values["n"] = _positive_root(
            lambda nn: evaluate(n=nn)[0] - power, max_n, tol)
    else:
        if missing == "p1":
            intervals = ([(p0 + tol, 1 - tol)] if alternative == "greater" else
                         [(tol, p0 - tol)] if alternative == "less" else
                         [(p0 + tol, 1 - tol), (tol, p0 - tol)])
        elif missing == "p0":
            intervals = ([(tol, p1 - tol)] if alternative == "greater" else
                         [(p1 + tol, 1 - tol)] if alternative == "less" else
                         [(tol, p1 - tol), (p1 + tol, 1 - tol)])
        else:
            intervals = [(tol, 1 - tol)]
        values[missing] = _root(lambda v: evaluate(**{missing: v})[0] - power,
                                intervals, tol, missing)
    size = evaluate()[1] if exact else None
    suffix = {"quantile": "exact binomial", "cp": "exact Clopper--Pearson",
              "midp": "deterministic mid-p test",
              "midp-rand": "randomized mid-p boundaries"}
    label = suffix[exact_method] if exact else "normal approximation"
    method = f"One-sample proportion power calculation ({label})"
    return PowerResult(**values, alternative=alternative, method=method,
                       achieved_sig_level=size)


def power_p2s_test(n=None, p1=None, p2=None, *, sig_level=0.05, power=None,
                   group_rate=1, alternative="two.sided", correct=False,
                   strict=False, tol=np.finfo(float).eps**0.25):
    """Compute two-sample proportion power with unequal group sizes.

    Parameters
    ----------
    n : positive float, optional
        First-group size; second-group size is n*group_rate. Values and
        normal sample-size solutions are continuous, matching R.
    p1, p2 : float, optional
        Group proportions in (0, 1).
    sig_level : float or None, default 0.05
        Nominal level in (0, 1). Exactly one of n, p1, p2, power, sig_level
        must be None; the missing quantity is solved.
    power : float, optional
        Target power in (0, 1); omit to compute power.
    group_rate : positive float, default 1
        Allocation ratio n2/n1, not a ratio of event rates.
    alternative : {"two.sided", "one.sided"}, default "two.sided"
        R's power convention, using the absolute difference between groups.
    correct : bool, default False
        Add 0.5*(1/n1+1/n2) to the rejection threshold.
    strict : bool, default False
        Include both normal tails for a two-sided test; otherwise use the
        dominant-tail approximation, as in stats::power.prop.test().
    tol : positive float, default machine epsilon**0.25
        Root-finding tolerance, matching R.

    Returns
    -------
    PowerResult
        Resolved parameters, method, and note identifying n as first-group
        size. Power records the requested target for inverse calculations.

    Raises
    ------
    ValueError
        For invalid inputs or an unattainable inverse within valid parameter
        ranges. Missing p1 is sought in [0, p2], missing p2 in [p1, 1].

    See Also
    --------
    power_p1s_test : One-sample normal and exact binomial power.

    Notes
    -----
    Matches R power.p2s.test(). The normal rejection threshold uses the pooled
    null variance; power uses the unpooled alternative variance. Group sizes
    are not rounded. There is no exact two-sample option in this interface.
    At equal allocation, defaults agree with R stats::power.prop.test().
    Probability inversions stay within [0, 1]; infeasible requests raise an
    error rather than extending a search to inadmissible probabilities.

    References
    ----------
    R ibist power.p2s.test(), Details; R stats power.prop.test(), Details.

    Examples
    --------
    >>> round(power_p2s_test(n=100, p1=.3, p2=.5, group_rate=2).power, 7)
    0.9192899
    """
    values = dict(n=n, p1=p1, p2=p2, power=power, sig_level=sig_level)
    missing = _validate(values)
    group_rate = _positive(group_rate, "group_rate")
    tol = _positive(tol, "tol")
    correct, strict = _flag(correct, "correct"), _flag(strict, "strict")
    alternative = _choice(alternative, ("two.sided", "one.sided"),
                           "alternative")

    def evaluate(**change):
        args = {**values, **change}
        nn, a, b, alpha = (args[k] for k in ("n", "p1", "p2", "sig_level"))
        n2 = nn * group_rate
        pooled = (nn * a + n2 * b) / (nn + n2)
        se0 = math.sqrt(pooled * (1 - pooled) * (1 / nn + 1 / n2))
        se1 = math.sqrt(a * (1 - a) / nn + b * (1 - b) / n2)
        cc = 0.5 * (1 / nn + 1 / n2) if correct else 0
        z = norm.isf(alpha / (2 if alternative == "two.sided" else 1))
        distance = abs(a - b)
        value = norm.cdf((distance - z * se0 - cc) / se1)
        if strict and alternative == "two.sided":
            value += norm.sf((distance + z * se0 + cc) / se1)
        return float(value)

    if missing == "power":
        values["power"] = evaluate()
    elif missing == "n":
        values["n"] = _positive_root(lambda nn: evaluate(n=nn) - power,
                                      10000000, tol)
    else:
        interval = ((0, p2) if missing == "p1" else (p1, 1)
                    if missing == "p2" else (1e-10, 1 - 1e-10))
        values[missing] = _root(lambda v: evaluate(**{missing: v}) - power,
                                [interval], tol, missing)
    note = ("n is number in the 1st group; " +
            ("continuity correction applied" if correct else
             "no continuity correction"))
    return PowerResult(**values, alternative=alternative, note=note,
                       method="Two-sample proportions power calculation")


def _validate(values):
    missing = [key for key, value in values.items() if value is None]
    if len(missing) != 1:
        raise ValueError("Exactly one design parameter must be None.")
    for key, value in values.items():
        if value is not None:
            _positive(value, key) if key == "n" else _probability(value, key)
    return missing[0]


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a positive finite number.")
    return float(value)


def _choice(value, choices, name):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {choices}.")
    return value


def _root(function, intervals, tol, name):
    for low, high in intervals:
        if low >= high:
            continue
        try:
            return brentq(function, low, high, xtol=tol,
                          rtol=4 * np.finfo(float).eps)
        except (ValueError, ZeroDivisionError):
            pass
    raise ValueError(f"Could not solve for {name} with supplied parameters.")


def _positive_root(function, upper, tol):
    low = 1.0
    for _ in range(100):
        if function(low) <= 0:
            break
        low /= 2
    for _ in range(100):
        if function(upper) >= 0:
            break
        upper *= 2
    return _root(function, [(low, upper)], tol, "n")


@lru_cache(maxsize=2048)
def _region(n, p0, alpha, alternative, method):
    tail = alpha / (2 if alternative == "two.sided" else 1)
    if method in ("quantile", "midp-rand"):
        low = int(binom.ppf(tail, n, p0)) - 1
        high = int(binom.ppf(1 - tail, n, p0)) + 1
        return low, high

    def lower(x):
        return (binom.cdf(x - 1, n, p0) + 0.5 * binom.pmf(x, n, p0)
                if method == "midp" else binom.cdf(x, n, p0))

    def upper(x):
        return (binom.sf(x, n, p0) + 0.5 * binom.pmf(x, n, p0)
                if method == "midp" else binom.sf(x - 1, n, p0))

    a, b = -1, n + 1
    while b - a > 1:
        mid = (a + b) // 2
        if lower(mid) <= tail:
            a = mid
        else:
            b = mid
    low = a
    a, b = -1, n + 1
    while b - a > 1:
        mid = (a + b) // 2
        if upper(mid) <= tail:
            b = mid
        else:
            a = mid
    return low, b


def _exact_power(n, p0, p1, alpha, alternative, method):
    low, high = _region(n, p0, alpha, alternative, method)

    def rejection(p):
        if method == "midp-rand":
            lower = binom.cdf(low - 1, n, p) + 0.5 * binom.pmf(low, n, p)
            upper = binom.sf(high, n, p) + 0.5 * binom.pmf(high, n, p)
        else:
            lower, upper = binom.cdf(low, n, p), binom.sf(high - 1, n, p)
        return float(upper if alternative == "greater" else
                     lower if alternative == "less" else lower + upper)

    return rejection(p1), rejection(p0)
