"""Asymptotic inference for two-rater agreement beyond chance."""

import math

import numpy as np
from scipy.stats import norm

from .contingency import TestResult, _table
from .odds_ratio import _probability


def agreement_test(x, *, weights="unweighted", conf_level=0.95,
                   alternative="two.sided", test_variance="fce") -> TestResult:
    """Test agreement beyond chance using Cohen's kappa or weighted kappa.

    Parameters
    ----------
    x : array_like or pandas.DataFrame, shape (K, K)
        Nonnegative integer counts for paired ratings by two raters, with
        at least two categories and positive total. Rows and columns must
        represent the same categories in the same order; nothing is sorted.
        Counts and their total must not exceed 2**53-1. Empty categories are
        retained because they affect positional weights.
    weights : {"unweighted", "linear", "quadratic"} or array_like, optional
        Default "unweighted" uses identity agreement weights. Linear and
        quadratic weights use category positions, not numeric labels.
        A custom numeric matrix must match x, be symmetric, have diagonal
        ones, and have finite entries in [0, 1]. R's mean-relative symmetry
        tolerance sqrt(machine epsilon) is used; weights are not symmetrized.
    conf_level : float, default 0.95
        Confidence level strictly between zero and one.
    alternative : {"two.sided", "greater", "less"}, default "two.sided"
        Alternative to zero kappa; also determines interval direction.
    test_variance : {"fce", "delta"}, default "fce"
        FCE uses the Fleiss--Cohen--Everitt null variance for testing; delta
        uses the multinomial variance evaluated at observed proportions.
        Both choices use the estimation variance for confidence limits.

    Returns
    -------
    TestResult
        Scalar kappa estimate, zero null_value, signed Z statistic, p_value,
        conf_int tuple, conf_level, alternative, and method description.
        std_error is the CI standard error. null_std_error stores the
        selected test standard error: a null SE for fce, an estimation SE
        for delta. Also includes test_variance, weight_method, n, and a
        weights property returning a fresh NumPy matrix. df is None.

    Raises
    ------
    ValueError
        If counts, weights, or options are invalid, chance agreement is one,
        or a valid positive test variance cannot be computed.

    See Also
    --------
    assoc_test : Association tests, distinct from matching-category agreement.

    Notes
    -----
    Definitions and defaults follow R agreement.test(), which delegates to
    the supported cohen.kappa.test() entry point. For observed weighted
    agreement Po and chance agreement Pe from independent margins,
    kappa=(Po-Pe)/(1-Pe). Linear weights are 1-abs(i-j)/(K-1); quadratic
    Fleiss--Cohen weights are 1-((i-j)/(K-1))**2. Relabeling categories does
    not change these weights; reordering categories can change them.

    The FCE estimation variance equals the observed multinomial delta-method
    variance. The default FCE null variance is distinct, so default tests
    need not agree with interval decisions. Selecting delta changes the test
    standard error, not the estimate or interval. The historical R field
    null.std.error corresponds to null_std_error even when delta is chosen.

    Two-sided intervals are untruncated Wald intervals. One-sided finite
    bounds use the full error probability, are clipped to [-1, 1], and have
    the other endpoint at 1 (greater) or -1 (less). A nonpositive selected
    test variance is an error. Only asymptotic two-rater inference is
    implemented; exact tests and multi-rater kappas are not substituted.
    Agreement and association answer different questions: raters can be
    strongly associated while consistently choosing different categories.
    Stored values retain full precision; summaries round display values only.

    References
    ----------
    Fleiss, Cohen, and Everitt (1969). Large sample standard errors of kappa
    and weighted kappa. Psychological Bulletin 72(5), 323--327,
    doi:10.1037/h0028106.
    SAS Institute (2015), SAS/STAT 14.1 User's Guide, The FREQ Procedure,
    Tests and Measures of Agreement.
    https://support.sas.com/documentation/onlinedoc/stat/141/freq.pdf

    Examples
    --------
    >>> ratings = [[9, 1], [2, 8]]
    >>> result = agreement_test(ratings)
    >>> round(result.estimate, 3)
    0.7
    >>> result.weight_method
    'unweighted'
    >>> agreement_test(ratings, weights="quadratic").weight_method
    'quadratic'
    >>> agreement_test(ratings, test_variance="delta").test_variance
    'delta'
    """
    conf_level = _probability(conf_level, "conf_level")
    if not isinstance(alternative, str) or alternative not in (
            "two.sided", "greater", "less"):
        raise ValueError("alternative must be two.sided, greater, or less.")
    if not isinstance(test_variance, str) or test_variance not in (
            "fce", "delta"):
        raise ValueError("test_variance must be fce or delta.")
    table, _ = _table(x, positive_margins=False)
    if table.shape[0] != table.shape[1]:
        raise ValueError("x must be a square table of paired-rating counts.")
    n = int(table.sum())
    if not n:
        raise ValueError("x must have a positive total count.")
    weight, weight_method = _weights(weights, table.shape[0])
    p = table / n
    row = np.array([_sum(values) for values in p])
    col = np.array([_sum(values) for values in p.T])
    independent = np.outer(row, col)
    observed = _sum(p * weight)
    chance = _sum(independent * weight)
    denominator = 1 - chance
    if denominator <= 0:
        raise ValueError("Kappa is undefined because chance agreement is one.")
    estimate = (observed - chance) / denominator
    row_weight = weight @ col
    col_weight = row @ weight
    combined = row_weight[:, None] + col_weight[None, :]
    variance = (_sum(p * (weight - combined * (1 - estimate))**2)
                - (estimate - chance * (1 - estimate))**2)
    variance /= denominator**2 * n
    null_variance = (_sum(independent * (weight - combined)**2) - chance**2)
    null_variance /= denominator**2 * n
    tolerance = 100 * np.finfo(float).eps
    if -tolerance < variance < 0:
        variance = 0.0
    if -tolerance < null_variance < 0:
        null_variance = 0.0
    if (not math.isfinite(variance) or variance < 0
            or not math.isfinite(null_variance) or null_variance <= 0):
        raise ValueError("A valid kappa variance could not be computed for x.")
    selected = null_variance if test_variance == "fce" else variance
    if selected <= 0:
        raise ValueError("The selected test variance must be positive.")
    std_error, null_std_error = math.sqrt(variance), math.sqrt(selected)
    z = estimate / null_std_error
    p_value = (2 * norm.sf(abs(z)) if alternative == "two.sided" else
               norm.sf(z) if alternative == "greater" else norm.cdf(z))
    tail = (1 - conf_level) / (2 if alternative == "two.sided" else 1)
    margin = norm.isf(tail) * std_error
    interval = (estimate - margin, estimate + margin)
    if alternative == "greater":
        interval = (float(np.clip(interval[0], -1, 1)), 1.0)
    elif alternative == "less":
        interval = (-1.0, float(np.clip(interval[1], -1, 1)))
    weighted = weight_method != "unweighted"
    method = ("Asymptotic test for weighted Cohen's kappa" if weighted else
              "Asymptotic test for Cohen's kappa")
    return TestResult(
        z, float(p_value), method, alternative=alternative, estimate=estimate,
        null_value=0.0, conf_int=interval, conf_level=conf_level,
        estimate_name="weighted kappa" if weighted else "kappa",
        std_error=std_error, null_std_error=null_std_error,
        test_variance=test_variance, weight_method=weight_method, n=n,
        _weight_data=tuple(tuple(float(value) for value in row)
                           for row in weight))


def _sum(values):
    return math.fsum(float(value) for value in values.flat)


def _weights(weights, size):
    if isinstance(weights, str):
        if weights == "unweighted":
            return np.eye(size), weights
        if weights not in ("linear", "quadratic"):
            raise ValueError("weights must be unweighted, linear, "
                             "or quadratic.")
        category = np.arange(size)
        distance = abs(category[:, None] - category[None, :]) / (size - 1)
        power = 2 if weights == "quadratic" else 1
        return 1 - distance**power, weights
    try:
        values = np.asarray(weights)
    except (TypeError, ValueError):
        raise ValueError("weights must be a name or numeric matrix.") from None
    if (values.shape != (size, size) or values.dtype.kind not in "iuf"
            or not np.all(np.isfinite(values)) or np.any(values < 0)
            or np.any(values > 1) or np.any(np.diag(values) != 1)):
        raise ValueError("Custom weights must match x, have diagonal ones, "
                         "and contain finite numbers in [0, 1].")
    values = values.astype(float, copy=True)
    different = values != values.T
    if np.any(different):
        target, current = values[different], values.T[different]
        tolerance = math.sqrt(np.finfo(float).eps)
        scale = _sum(abs(target)) / len(target)
        if scale <= tolerance:
            scale = 1.0
        difference = _sum(abs(target - current)) / (len(target) * scale)
        if difference > tolerance:
            raise ValueError("Custom weights must be symmetric.")
    return values, "custom"
