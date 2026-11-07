"""Exact conditional confidence intervals for a common odds ratio."""

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

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
    if observed == 0:
        estimate = 0.0
    elif observed == distribution.width:
        estimate = math.inf
    else:
        eta = brentq(lambda value: distribution.mean(value) - observed,
                     -700, 700, xtol=1e-10, rtol=1e-14)
        estimate = _exp(eta)
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
        self.observed, self.width = 0, 0
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
            self.observed += int(tab[0, 0]) - low
            base = (np.array([0.0]) if low == high else
                    _ConditionalTable(tab).base.copy())
            # Each stratum's arbitrary normalization cancels from the tilt.
            base -= base.max()
            self.bases.append(base)
        self.coefficients = None

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
        return np.exp(weights - logsumexp(weights))
