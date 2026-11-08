"""Normal power for one- and two-sample Poisson score tests."""

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import norm

from .contingency import _flag
from .power import PowerResult, _choice, _positive


def power_r1s_test(n=None, r0=None, r1=None, power=None, *, sig_level=0.05,
                   exposure=1, alternative="two.sided", correct=False,
                   strict=False, tol=np.finfo(float).eps**0.5):
    """Compute one-sample Poisson rate power or solve one design quantity.

    Parameters
    ----------
    n : positive float, optional
        Sample size. Total exposure is n*exposure. Supplied and solved sizes
        remain continuous; round up for subject recruitment.
    r0, r1 : positive float, optional
        Null and alternative rates per unit exposure.
    power : float, optional
        Target in (0, 1); omit to calculate power. Exactly one of n, r0, r1,
        power, sig_level must be None.
    sig_level : float or None, default 0.05
        Significance level in (0, 1); None requests inversion.
    exposure : positive float, default 1
        Fixed exposure per subject, in the units used for the rates.
    alternative : {"two.sided", "less", "greater"}, default "two.sided"
        Direction relative to r0.
    correct : bool, default False
        Apply a half-count continuity correction, as in rate_test().
    strict : bool, default False
        Include both two-sided tails; otherwise use the dominant tail,
        following proportion power conventions.
    tol : positive float, default sqrt(machine epsilon)
        Root tolerance on the log or logit scale.

    Returns
    -------
    PowerResult
        Resolved n, rates, level, power, exposure, alternative, method, note.
        Inverse calls retain the requested power, as in R's power.htest.

    Raises
    ------
    ValueError
        For invalid inputs, equal rates when solving n, or an unattainable
        target within the numerical search bounds.

    See Also
    --------
    rate_test : Normal score tests of Poisson rates.
    power_r2s_test : Two-sample rate power with unequal allocation.
    power_p1s_test : One-sample proportion power.

    Notes
    -----
    Matches R power.r1s.test(). For T=n*exposure, the rate estimator has
    approximate mean r1 and variance r1/T. Score boundaries are r0 +/-
    (z*sqrt(r0/T)+c), where c=0.5/T if corrected, else zero. Critical z
    uses sig_level/2 for two-sided tests. Power uses the alternative normal
    distribution. This is large-sample planning power, not exact Poisson
    power; accuracy may be poor with few expected events.

    Two-sided inversion prefers r1>r0 or r0<r1, then the opposite branch.
    Directional inversion prefers the branch matching the alternative.
    Positive roots are searched within 100 doublings or halvings of 1 for
    n, or the other rate for rate inversion. Level inversion uses logit
    coordinates of magnitude at most 36. Exposure is not a solved quantity.

    References
    ----------
    Minitab, LLC. Example of Power and Sample Size for 1-Sample Poisson
    Rate; Methods and Formulas for the same procedure. Minitab Support,
    accessed 2026-11-08. https://support.minitab.com
    Published powers .842947 and .898200 match for n=25 and 30, r0=15,
    r1=13, level .05, and alternative="less", without correction.

    R Core Team. The Poisson Distribution. R stats documentation.
    https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Poisson.html
    Documents the Poisson mean and variance used in the derivation above.

    Examples
    --------
    >>> from ibist import power_r1s_test
    >>> design = power_r1s_test(r0=.1, r1=.2, power=.8, exposure=2)
    >>> round(design.n, 2)
    49.62
    """
    alternative = _choice(alternative, ("two.sided", "less", "greater"),
                          "alternative")
    values = dict(n=n, r0=r0, r1=r1, power=power, sig_level=sig_level)
    _validate(values, correct, strict, tol)
    exposure = _positive(exposure, "exposure")

    def evaluate(v):
        total = v["n"] * exposure
        delta = v["r1"] - v["r0"]
        if alternative == "less":
            delta = -delta
        if alternative == "two.sided":
            delta = abs(delta)
        boundary = norm.isf(v["sig_level"] / (
            2 if alternative == "two.sided" else 1)) * math.sqrt(v["r0"]/total)
        if correct:
            boundary += 0.5 / total
        se = math.sqrt(v["r1"] / total)
        answer = norm.cdf((delta - boundary) / se)
        if strict and alternative == "two.sided":
            answer += norm.cdf((-delta - boundary) / se)
        return float(answer)

    missing = next(key for key, value in values.items() if value is None)
    other = r1 if missing == "r0" else r0 if missing == "r1" else 1
    prefer = -1 if missing == "r0" else 1
    if alternative == "less":
        prefer = -prefer
    if missing == "n" and r0 == r1:
        raise ValueError("Equal rates cannot determine a sample size.")
    values = _solve(values, evaluate, other, prefer, tol)
    return PowerResult(**values, exposure=exposure, alternative=alternative,
                       method="One-sample Poisson rate power calculation",
                       note=_note(correct))


def power_r2s_test(n=None, r1=None, r2=None, *, sig_level=0.05, power=None,
                   group_rate=1, exposure=1, alternative="two.sided",
                   correct=False, strict=False,
                   tol=np.finfo(float).eps**0.5):
    """Compute power or solve a design quantity for two Poisson rates.

    Parameters
    ----------
    n : positive float, optional
        First group size; second size is n*group_rate. Sizes stay continuous.
    r1, r2 : positive float, optional
        Rates per unit exposure in groups one and two.
    sig_level : float or None, default 0.05
        Significance level in (0, 1); None requests inversion.
    power : float, optional
        Target in (0, 1); omit to calculate power. Exactly one of n, r1, r2,
        power, sig_level must be None.
    group_rate : positive float, default 1
        Sample-size ratio n2/n1, independent of follow-up duration.
    exposure : positive float or pair of float, default 1
        Fixed exposure per subject, shared scalar or separate group values.
    alternative : {"two.sided", "one.sided"}, default "two.sided"
        One-sided power uses the supplied difference's direction, matching
        power_p2s_test().
    correct : bool, default False
        Apply the count-scale continuity correction used by rate_test().
    strict : bool, default False
        Include both tails for two-sided power, else only the dominant tail.
    tol : positive float, default sqrt(machine epsilon)
        Root tolerance on the log or logit scale.

    Returns
    -------
    PowerResult
        Resolved design with both rates, exposure pair, and group_rate.
        Inverse calls retain the target power.

    Raises
    ------
    ValueError
        For invalid inputs, equal rates when solving n, or no bracketed root.

    See Also
    --------
    power_r1s_test : One-sample power and numerical search bounds.
    rate_test : Normal score tests of Poisson rates.
    power_p2s_test : Two-sample proportion power with unequal allocation.

    Notes
    -----
    Matches R power.r2s.test(). Set T1=n*exposure[0] and
    T2=n*group_rate*exposure[1]. The expected pooled rate is
    rbar=(r1*T1+r2*T2)/(T1+T2). Null and alternative standard errors are
    s0=sqrt(rbar*(1/T1+1/T2)) and s1=sqrt(r1/T1+r2/T2). Dominant-tail
    power is Phi((abs(r1-r2)-z*s0-c)/s1), with
    c=0.5*(1/T1+1/T2) if corrected, else zero. Strict two-sided power adds
    Phi((-abs(r1-r2)-z*s0-c)/s1). Critical z uses sig_level/2 for two sides.

    The random pooled variance of rate_test() is replaced by its expected
    value. This is large-sample planning power, not exact conditional power;
    it can be inaccurate with few expected events. Rate inversion first
    seeks r1<r2 or r2>r1, then the other branch. Search bounds follow
    power_r1s_test(). Round n up for subject recruitment and recompute power.

    References
    ----------
    Minitab, LLC. Example of Power and Sample Size for 2-Sample Poisson
    Rate; Methods and Formulas for the same procedure. Minitab Support,
    accessed 2026-11-08. https://support.minitab.com
    With strict=True, baseline rate 35, level .05, and unit exposures,
    rates 32 and 38 give .902793 and .902550 power at n=79 and 86 per
    group. Target power .9 gives those rounded sample sizes.

    R Core Team. The Poisson Distribution. R stats documentation.
    https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Poisson.html
    Documents the Poisson mean and variance used in the derivation above.

    Examples
    --------
    >>> from ibist import power_r2s_test
    >>> design = power_r2s_test(r1=.1, r2=.2, power=.8, group_rate=2,
    ...                         exposure=(2, 3))
    >>> round(design.n, 2)
    83.25
    """
    alternative = _choice(alternative, ("two.sided", "one.sided"),
                          "alternative")
    values = dict(n=n, r1=r1, r2=r2, power=power, sig_level=sig_level)
    _validate(values, correct, strict, tol)
    group_rate = _positive(group_rate, "group_rate")
    try:
        raw = np.asarray(exposure, dtype=object)
        if raw.ndim == 0 or raw.shape == (1,):
            raw = np.repeat(raw, 2)
        if raw.shape != (2,):
            raise ValueError
        exposure = tuple(_positive(x, "exposure") for x in raw)
    except (ValueError, TypeError):
        message = "exposure must be a positive scalar or pair."
        raise ValueError(message) from None

    def evaluate(v):
        t1 = v["n"] * exposure[0]
        t2 = v["n"] * group_rate * exposure[1]
        pooled = v["r1"] * (t1/(t1+t2)) + v["r2"] * (t2/(t1+t2))
        se0 = math.sqrt(pooled * (1/t1 + 1/t2))
        se1 = math.sqrt(v["r1"]/t1 + v["r2"]/t2)
        delta = abs(v["r1"] - v["r2"])
        boundary = norm.isf(v["sig_level"] / (
            2 if alternative == "two.sided" else 1)) * se0
        if correct:
            boundary += 0.5 * (1/t1 + 1/t2)
        answer = norm.cdf((delta - boundary) / se1)
        if strict and alternative == "two.sided":
            answer += norm.cdf((-delta - boundary) / se1)
        return float(answer)

    missing = next(key for key, value in values.items() if value is None)
    other = r2 if missing == "r1" else r1 if missing == "r2" else 1
    prefer = -1 if missing == "r1" else 1
    if missing == "n" and r1 == r2:
        raise ValueError("Equal rates cannot determine a sample size.")
    values = _solve(values, evaluate, other, prefer, tol)
    return PowerResult(**values, exposure=exposure, group_rate=group_rate,
                       alternative=alternative,
                       method="Two-sample Poisson rates power calculation",
                       note="n is number in the 1st group; " + _note(correct))


def _validate(values, correct, strict, tol):
    if sum(value is None for value in values.values()) != 1:
        raise ValueError("Exactly one design parameter must be None.")
    for name, value in values.items():
        if value is not None:
            _positive(value, name)
            if name in ("power", "sig_level") and value >= 1:
                raise ValueError(f"{name} must be in (0, 1).")
    _flag(correct, "correct")
    _flag(strict, "strict")
    _positive(tol, "tol")


def _note(correct):
    return ("continuity correction applied" if correct else
            "no continuity correction")


def _solve(values, evaluate, reference, prefer, tol):
    missing = next(key for key, value in values.items() if value is None)
    if missing == "power":
        return dict(values, power=evaluate(values))
    transform = expit if missing == "sig_level" else math.exp

    def objective(x):
        v = dict(values)
        v[missing] = float(transform(x))
        return evaluate(v) - values["power"]

    center = 0.0 if missing == "sig_level" else math.log(reference)
    start = objective(center)
    if math.isfinite(start) and start == 0:
        values[missing] = float(transform(center))
        return values
    for direction in (prefer, -prefer):
        previous, previous_value = center, start
        for step in range(1, 101):
            endpoint = center + direction * step * math.log(2)
            if missing == "sig_level" and abs(endpoint) > 36:
                break
            value = objective(endpoint)
            if (math.isfinite(value) and math.isfinite(previous_value) and
                    (value == 0 or (value > 0) != (previous_value > 0))):
                low, high = sorted((previous, endpoint))
                root = brentq(objective, low, high, xtol=tol,
                              rtol=4*np.finfo(float).eps)
                values[missing] = float(transform(root))
                return values
            previous, previous_value = endpoint, value
    raise ValueError(f"Could not solve for {missing} with supplied parameters.")
