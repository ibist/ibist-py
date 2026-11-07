"""Confidence intervals for the odds ratio in a 2 x 2 count table."""

from collections.abc import Sequence
import math
from numbers import Real

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

from .results import CIResult, ConfidenceInterval, _ci_orientation


def or_2x2_ci(x, *, conf_level: float = 0.95,
              method: str | Sequence[str] = "wald", midp: bool = False,
              tol: float = 1e-8, side=None, alternative=None) -> CIResult:
    """Estimate an odds ratio and calculate one or more confidence intervals.

    Parameters
    ----------
    x : array_like, shape (2, 2)
        Non-negative integer counts arranged as ``[[a, b], [c, d]]``.
        The common reported estimate is the sample odds ratio ``a*d/(b*c)``.
    conf_level : float, default 0.95
        Confidence level strictly between zero and one.
    method : str or sequence of str, default "wald"
        ``"wald"``: log-Wald interval, requiring positive cells.
        ``"gart"``: log-Wald interval after adding 0.5 to every cell.
        ``"cornfield"``: invert the two one-sided conditional tails.
        ``"bp"``: invert the probability-ordered two-sided conditional test.
    midp : bool, default False
        Modify conditional intervals using mid-p probabilities. Cornfield
        subtracts half the observed-table probability from each tail. BP
        includes half the mass of the entire group tied with the observed
        table. This option does not change Wald or Gart intervals.
    side : {"two.sided", "lower", "upper"}, optional
        Interval form; omitted or None selects two-sided limits. One-sided
        BP uses the same ordered conditional tail as Cornfield.
    alternative : {"two.sided", "less", "greater"}, optional
        Compatibility alias; greater selects lower, less selects upper.
        Conflicting side and alternative values are rejected.
    tol : float, default 1e-8
        Absolute root-finding tolerance on the log odds-ratio scale. BP
        limits can occur at jumps in the probability-ordered p-value.

    Returns
    -------
    CIResult
        Sample estimate and a confidence interval for each requested method.
        Gart uses the adjusted estimate to construct its limits, but the
        common reported estimate remains the unadjusted sample odds ratio.

    Notes
    -----
    The 0.5 adjustment is the Haldane--Anscombe correction, used here with
    Gart's interval construction. Conditional methods use Fisher's noncentral
    hypergeometric distribution with fixed margins. They require positive
    row and column totals. Exact intervals can be conservative; mid-p limits
    do not guarantee at least nominal coverage. A zero sample odds ratio has
    conditional lower limit zero; an infinite estimate has upper limit inf.
    With an empty margin the sample estimate is nan; Gart remains available.
    Conditional computation enumerates the possible upper-left counts and
    can be expensive for wide supports. At very low confidence levels, BP
    mid-p inversion may have no accepted interior point; a ValueError is
    raised if an interval cannot be bracketed.

    References
    ----------
    Haldane (1956), Annals of Human Genetics 20, 309--311.
    Anscombe (1956), Biometrika 43, 461--464.
    Gart (1966), JRSS B 28, 164--179.
    Baptista and Pike (1977), Applied Statistics 26, 214--220.
    Fagerland (2012), The Stata Journal 12, 505--514.

    Examples
    --------
    >>> result = or_2x2_ci([[7, 27], [1, 33]], method=["wald", "cornfield"])
    >>> round(result.estimate, 3)
    8.556
    """
    side, alternative = _ci_orientation(side, alternative)
    tab = _validate_table(x)
    conf_level = _probability(conf_level, "conf_level")
    if not isinstance(midp, (bool, np.bool_)):
        raise ValueError("midp must be True or False.")
    if isinstance(tol, (bool, np.bool_)) or not isinstance(tol, Real):
        raise ValueError("tol must be a positive finite number.")
    try:
        tol = float(tol)
    except (TypeError, ValueError):
        raise ValueError("tol must be a positive finite number.") from None
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a positive finite number.")
    if isinstance(method, str):
        methods = [method]
    elif isinstance(method, Sequence):
        methods = list(method)
    else:
        raise ValueError("method must be a method name or a sequence of names.")
    valid = ("wald", "gart", "cornfield", "bp")
    if not methods or any(not isinstance(m, str) or m not in valid
                          for m in methods):
        raise ValueError(f"method must contain one or more of {valid}.")
    a, b, c, d = (float(v) for v in tab.ravel())
    numerator, denominator = a * d, b * c
    estimate = (numerator / denominator if denominator else
                math.inf if numerator else math.nan)
    z = norm.isf((1 - conf_level) / (2 if side == "two.sided" else 1))
    conditional = None
    intervals = []
    for name in methods:
        if name in ("wald", "gart"):
            cells = tab.astype(float) + (0.5 if name == "gart" else 0)
            if np.any(cells == 0):
                raise ValueError("Wald interval requires positive cells.")
            log_or = np.log(cells[0, 0]) + np.log(cells[1, 1])
            log_or -= np.log(cells[0, 1]) + np.log(cells[1, 0])
            se = np.sqrt(np.sum(1 / cells))
            lower, upper = (_exp(log_or - z * se), _exp(log_or + z * se))
            label = name
        else:
            if conditional is None:
                conditional = _ConditionalTable(tab)
            lower, upper = conditional.interval(
                name, conf_level, midp, tol, side=side)
            label = f"{name}-{'midp' if midp else 'exact'}"
        if side == "lower":
            upper = math.inf
        if side == "upper":
            lower = 0.0
        intervals.append(ConfidenceInterval(label, float(lower), float(upper)))
    return CIResult(estimate, tuple(intervals), conf_level, "odds ratio", 1.0,
                    midp=bool(midp), alternative=alternative)


def _validate_table(x):
    try:
        tab = np.asarray(x)
    except (TypeError, ValueError):
        raise ValueError("x must be a 2 x 2 numeric table.") from None
    if tab.shape != (2, 2) or tab.dtype.kind not in "iuf":
        raise ValueError("x must be a 2 x 2 numeric table.")
    if (not np.all(np.isfinite(tab)) or np.any(tab < 0)
            or np.any(tab != np.floor(tab))):
        raise ValueError("x must contain non-negative integer counts.")
    if np.any(tab > 2 ** 53 - 1):
        raise ValueError("Cell counts must not exceed 2**53 - 1.")
    return tab.astype(np.int64)


def _probability(value, name):
    message = f"{name} must be a single number between zero and one."
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(message)
    return value


def _exp(value):
    if value > math.log(np.finfo(float).max):
        return math.inf
    return math.exp(value)


class _ConditionalTable:
    def __init__(self, tab):
        row1, row2 = (int(v) for v in tab.sum(axis=1))
        col1, col2 = (int(v) for v in tab.sum(axis=0))
        if min(row1, row2, col1, col2) == 0:
            raise ValueError("Need positive row and column totals.")
        self.observed = int(tab[0, 0])
        self.support = np.arange(max(0, col1 - row2), min(row1, col1) + 1)
        # Adjacent weight ratios avoid cancellation in differences of large
        # log factorials, including huge margins with only a few events.
        s = self.support[:-1]
        log_ratio = (np.log(row1 - s) + np.log(col1 - s)
                     - np.log(s + 1) - np.log(row2 - col1 + s + 1))
        self.base = np.concatenate(([0.0], np.cumsum(log_ratio)))
        self.index = self.observed - int(self.support[0])
        self.base -= self.base[self.index]
        cells = tab.ravel().astype(float)
        self.center = (float(np.log(cells[[0, 3]]).sum()
                             - np.log(cells[[1, 2]]).sum())
                       if np.all(cells > 0) else 0.0)

    def log_probabilities(self, eta):
        weights = self.base + (self.support - self.observed) * eta
        return weights - logsumexp(weights)

    def pvalue(self, eta, method, midp, lower=True):
        log_prob = self.log_probabilities(eta)
        prob = np.exp(log_prob)
        if method == "cornfield":
            selected = (self.support >= self.observed if lower else
                        self.support <= self.observed)
            return float(prob[selected].sum() - (0.5 * prob[self.index]
                                                 if midp else 0.0))
        difference = log_prob - log_prob[self.index]
        tied = np.abs(difference) <= 1e-12
        return float(prob[(difference < 0) & ~tied].sum()
                     + (0.5 if midp else 1.0) * prob[tied].sum())

    def interval(self, method, conf_level, midp, tol, side="two.sided"):
        if side != "two.sided":
            method = "cornfield"
        target = (1 - conf_level) / (2 if method == "cornfield" else 1)
        if side != "two.sided":
            target = 1 - conf_level
        limits = []
        for lower in (True, False):
            if (lower and side == "upper") or (not lower and side == "lower"):
                limits.append(0.0 if lower else math.inf)
                continue
            if lower and self.observed == self.support[0]:
                limits.append(0.0)
                continue
            if not lower and self.observed == self.support[-1]:
                limits.append(math.inf)
                continue

            def objective(eta):
                return self.pvalue(eta, method, midp, lower) - target

            center = self.center
            if method == "cornfield":
                # Tail roots need not lie on the expected side of the sample
                # estimate, particularly with sparse counts and mid-p limits.
                sign = 1 if lower else -1
                left = right = center
                while sign * objective(left) > 0 and left > -750:
                    left -= math.log(10)
                while sign * objective(right) < 0 and right < 750:
                    right += math.log(10)
                root = brentq(objective, left, right, xtol=tol, rtol=1e-14)
                limits.append(_exp(root))
                continue
            if objective(center) < 0:
                # Locate a tilt where the observed table is a conditional mode.
                left = (self.base[self.index - 1] if self.index else -math.inf)
                right = (-self.base[self.index + 1]
                         if self.index + 1 < len(self.support) else math.inf)
                if math.isfinite(left) and math.isfinite(right):
                    candidates = [left + fraction * (right - left)
                                  for fraction in (0.5, 1e-6, 1 - 1e-6)]
                else:
                    boundary = left + 1 if math.isfinite(left) else right - 1
                    candidates = [boundary]
                center = max(candidates, key=objective)
                if objective(center) < 0:
                    raise ValueError("Cannot bracket BP at this level.")
            inside = center
            direction = -1 if lower else 1
            outside = inside + direction * math.log(10)
            while objective(outside) > 0 and abs(outside) < 750:
                outside += direction * math.log(10)
            if objective(outside) > 0:
                limits.append(0.0 if lower else math.inf)
                continue
            bracket = sorted((inside, outside))
            root = brentq(objective, *bracket, xtol=tol, rtol=1e-14)
            limits.append(_exp(root))
        return tuple(limits)
