"""Exact conditional tests and confidence intervals for a common odds ratio."""

import math
from numbers import Real

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from .contingency import TestResult
from .odds_ratio import _ConditionalTable, _exp, _probability, _validate_table
from .results import CIResult, ConfidenceInterval, _ci_orientation


def or_common_ci(x, *, conf_level=0.95, algorithm="network", side=None,
                 alternative=None) -> CIResult:
    """Estimate a common odds ratio and invert exact conditional tails.

    x is a list of 2 by 2 tables or a NumPy array of shape (2,2,K), with
    independent strata on the last axis. Wrap a single table in a list.
    Both algorithm="network" (default, log-space dynamic programming) and
    "convolution" (direct probability convolution) combine the stratum
    noncentral hypergeometric distributions conditional on their margins.
    The estimate is the conditional MLE, not Mantel--Haenszel's estimate.

    conf_level defaults to 0.95. side is two.sided (default), lower, or upper;
    None selects the default. alternative is a compatibility alias (greater
    means lower and less means upper); conflicting values are errors.
    Inclusive tails use half the error probability for central limits and
    the full error probability for one-sided bounds. Limits are 0/inf at
    support boundaries. Deterministic strata, including empty margins, are
    retained. If the total support is a singleton, all ORs are unidentified;
    R's convention reports estimate 0 and interval [0,inf].

    Returns CIResult with the conditional scalar estimate, one exact
    interval, side, conf_level, and null_value=1. Network calculation is
    stable in log space; convolution can underflow in extreme tails. An
    allocation exceeding one million total support points raises an error.
    There is no R runtime dependency. Definitions follow or.common.ci().

    References
    ----------
    Mehta, Patel and Gray (1985), Journal of the American Statistical
    Association 80, 969--973, doi:10.1080/01621459.1985.10478212.
    """
    side, alternative = _ci_orientation(side, alternative)
    conf_level = _probability(conf_level, "conf_level")
    if not isinstance(algorithm, str) or algorithm not in ("network",
                                                         "convolution"):
        raise ValueError("algorithm must be network or convolution.")
    distribution = _CommonOdds(x)
    observed = distribution.observed
    estimate = distribution.mle()
    tail = (1 - conf_level) / (2 if side == "two.sided" else 1)

    def limit(lower):
        if lower and (observed == 0 or side == "upper"):
            return 0.0
        if not lower and (observed == distribution.width or side == "lower"):
            return math.inf

        def objective(eta):
            prob = distribution.probabilities(eta, algorithm)
            return float(prob[observed:].sum() if lower else
                         prob[:observed + 1].sum()) - tail

        return _exp(brentq(objective, -700, 700, xtol=1e-10, rtol=1e-14))

    interval = ConfidenceInterval("exact", limit(True), limit(False))
    return CIResult(estimate, (interval,), conf_level, "common odds ratio",
                    null_value=1.0, alternative=alternative)


def or_common_test(x, *, theta=1.0, alternative="two.sided") -> TestResult:
    """Exact conditional test of a specified common null odds ratio.

    Parameters
    ----------
    x : list of 2 by 2 tables or NumPy array of shape (2,2,K)
        Independent strata, with groups in rows and outcomes in columns.
        Wrap a single table in a list. Counts are nonnegative integers and
        deterministic strata, including empty margins, are retained.
    theta : positive finite number, default 1
        Null odds ratio in every stratum. Unit odds ratios test conditional
        independence of groups and outcomes within strata.
    alternative : {"two.sided", "greater", "less"}, default "two.sided"
        Inclusive upper or lower tail of the sum of upper-left counts.
        Two-sided inference doubles the smaller tail and caps it at one.

    Returns
    -------
    TestResult
        statistic is the integer sum of observed upper-left counts; p_value
        is exact conditional on every stratum's margins. estimate is the
        conditional common-OR MLE, independent of theta. null_value stores
        theta, also reported as parameter in R. df is None; n_strata records
        the number of strata. Obtain confidence limits with or_common_ci().
        R's convention reports estimate zero for a singleton total support,
        where the common OR is unidentified and every p-value is one.

    Notes
    -----
    This follows ibist-R or.common.test(). The network algorithm combines
    fixed-margin noncentral hypergeometric distributions. Log-space tails
    retain very small representable p-values. At most one million total
    support points are allowed; exceeding this limit raises an error.

    The doubled-smaller-tail rule can give a different two-sided p-value
    from probability-ordered Fisher inference in indep_test(). With one
    stratum, the one-sided tails agree with Fisher's conditional test.
    Homogeneity of unspecified stratum ORs is assessed by
    or_homogeneity_test(); this function tests their specified null value.
    At non-unit theta, ibist uses the noncentral hypergeometric extension.

    References
    ----------
    Jung, S.-H. (2014), Biometrical Journal 56(1), 129--140,
    doi:10.1002/bimj.201300048. Stratified Fisher testing and tail rules.

    Examples
    --------
    >>> result = or_common_test([[[3, 1], [1, 8]]])
    >>> round(result.p_value, 4)
    0.1035
    """
    if isinstance(theta, (bool, np.bool_)) or not isinstance(theta, Real):
        raise ValueError("theta must be a positive finite number.")
    try:
        theta = float(theta)
    except (OverflowError, ValueError):
        raise ValueError("theta must be a positive finite number.") from None
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be a positive finite number.")
    if not isinstance(alternative, str) or alternative not in (
            "two.sided", "greater", "less"):
        raise ValueError("alternative must be two.sided, greater, or less.")
    distribution = _CommonOdds(x)
    observed = distribution.observed
    log_prob = distribution.log_probabilities(math.log(theta))
    less = min(0.0, float(logsumexp(log_prob[:observed + 1])))
    greater = min(0.0, float(logsumexp(log_prob[observed:])))
    log_value = {"less": less, "greater": greater,
                 "two.sided": min(0.0, math.log(2) + min(less, greater))}[
                     alternative]
    return TestResult(
        statistic=distribution.minimum + observed, p_value=_exp(log_value),
        method="Exact conditional test for a common odds ratio",
        alternative=alternative, estimate=distribution.mle(), null_value=theta,
        n_strata=len(distribution.bases), estimate_name="Common odds ratio")


class _CommonOdds:
    def __init__(self, x):
        if isinstance(x, np.ndarray) and x.ndim == 3 and x.shape[:2] == (2, 2):
            items = [x[:, :, k] for k in range(x.shape[2])]
        elif isinstance(x, (list, tuple)):
            items = list(x)
        else:
            raise ValueError("x must be a list of 2 x 2 tables or "
                             "2 x 2 x K array.")
        if not items:
            raise ValueError("x must contain at least one stratum.")
        self.bases = []
        self.minimum, self.observed, self.width = 0, 0, 0
        for item in items:
            tab = _validate_table(item)
            if any(isinstance(v, (bool, np.bool_))
                   for v in np.asarray(item, dtype=object).flat):
                raise ValueError("Tables must contain counts, not booleans.")
            r1, r2 = (int(value) for value in tab.sum(axis=1))
            c1 = int(tab[:, 0].sum())
            low, high = max(0, c1 - r2), min(r1, c1)
            self.width += high - low
            if self.width + 1 > 1000000:
                raise ValueError("Common OR support exceeds 1,000,000 points.")
            self.minimum += low
            self.observed += int(tab[0, 0]) - low
            base = (np.array([0.0]) if low == high else
                    _ConditionalTable(tab).base.copy())
            # Each stratum's arbitrary normalization cancels from the tilt.
            base -= base.max()
            self.bases.append(base)
        self.coefficients = None

    def mle(self):
        if self.observed == 0:
            return 0.0
        if self.observed == self.width:
            return math.inf
        eta = brentq(lambda value: self.mean(value) - self.observed,
                     -700, 700, xtol=1e-10, rtol=1e-14)
        return _exp(eta)

    def mean(self, eta):
        value = 0.0
        for base in self.bases:
            support = np.arange(len(base))
            log_weights = base + support * eta
            prob = np.exp(log_weights - logsumexp(log_weights))
            value += float(support @ prob)
        return value

    def probabilities(self, eta, algorithm="network"):
        if algorithm == "convolution":
            prob = np.array([1.0])
            for base in self.bases:
                weights = base + np.arange(len(base)) * eta
                prob = np.convolve(prob, np.exp(weights - logsumexp(weights)))
            return prob / prob.sum()
        return np.exp(self.log_probabilities(eta))

    def log_probabilities(self, eta):
        if self.coefficients is None:
            coefficients = np.array([0.0])
            for base in self.bases:
                combined = np.full(len(coefficients) + len(base) - 1, -np.inf)
                for index, value in enumerate(base):
                    part = combined[index:index + len(coefficients)]
                    np.logaddexp(part, coefficients + value, out=part)
                coefficients = combined - combined.max()
            self.coefficients = coefficients
        weights = self.coefficients + np.arange(self.width + 1) * eta
        return weights - logsumexp(weights)
