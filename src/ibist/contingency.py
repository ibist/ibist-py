"""Independence tests and nominal association for count tables."""

from dataclasses import dataclass
import math
from numbers import Integral
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import chi2, hypergeom, random_table
from scipy.stats.contingency import odds_ratio

from .odds_ratio import _probability


@dataclass(frozen=True)
class TestResult:
    """A test with native fields and a readable summary.

    Observed counts, expected counts, and residual diagnostics are DataFrames;
    input DataFrame labels are preserved. Fisher has no chi-square statistic,
    degrees of freedom, or Pearson residuals. Confidence limits, when present,
    refer to the conditional odds ratio for Fisher tests. Rate tests use
    estimate_name to identify their rate or rate ratio and expose copied
    samples/sample_rates metadata. Stored values retain full precision.
    Association wrappers add measure and test_method identifiers, std_error
    for multinomial estimation, and chisq for CMH's squared signed statistic.
    Unsupported method-specific fields remain None.
    Agreement tests add null_std_error (the selected test standard error),
    test_variance, weight_method, n, and a copied weights array.
    """

    statistic: float | None
    p_value: float
    method: str
    df: int | None = None
    alternative: str = "two.sided"
    observed: pd.DataFrame | None = None
    expected: pd.DataFrame | None = None
    residuals: pd.DataFrame | None = None
    stdres: pd.DataFrame | None = None
    estimate: float | None = None
    conf_int: tuple[float, float] | None = None
    conf_level: float | None = None
    null_value: float | None = None
    n_strata: int | None = None
    n_resamples: int | None = None
    estimate_name: str = "Conditional odds ratio"
    _rate_data: tuple[tuple[int, float], ...] = ()
    measure: str | None = None
    test_method: str | None = None
    std_error: float | None = None
    chisq: float | None = None
    null_std_error: float | None = None
    test_variance: str | None = None
    weight_method: str | None = None
    n: int | None = None
    _weight_data: tuple[tuple[float, ...], ...] = ()

    @property
    def weights(self):
        """Fresh agreement-weight matrix, or None for other tests."""
        return np.array(self._weight_data) if self._weight_data else None

    @property
    def samples(self):
        """Fresh event counts and exposures for a rate test, otherwise None."""
        if not self._rate_data:
            return None
        frame = pd.DataFrame(self._rate_data, columns=["x", "exposure"])
        frame.index.name = "sample"
        return frame

    @property
    def sample_rates(self):
        """Fresh sample rates for a Poisson test, otherwise None."""
        if not self._rate_data:
            return None
        return np.array([x / t for x, t in self._rate_data])

    def summary(self, digits: int = 3) -> str:
        """Round only the displayed summary; retain tiny p-values."""
        _digits(digits)
        lines = [self.method]
        if self.statistic is not None:
            lines.append(f"Statistic: {self.statistic:.{digits}f}")
        if self.df is not None:
            lines.append(f"df: {self.df}")
        if self.n_strata is not None:
            lines.append(f"Strata: {self.n_strata}")
        lines.append(f"p-value: {self.p_value:.{max(1, digits)}g}")
        if self.estimate is not None:
            estimate = f"{self.estimate:.{digits}f}"
            if (self._rate_data and np.isfinite(self.estimate)
                    and 0 < abs(self.estimate) < 0.5 * 10**-digits):
                estimate = f"{self.estimate:.{digits}e}"
            lines.append(f"{self.estimate_name}: {estimate}")
        if self.conf_int is not None:
            lower, upper = self.conf_int
            lines.append(f"{100 * self.conf_level:g}% confidence interval: "
                         f"({lower:.{digits}g}, {upper:.{digits}g})")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.summary()


@dataclass(frozen=True)
class NominalAssociationResult:
    """Descriptive nominal association estimates, without tests or intervals."""

    estimate: pd.Series
    statistic: float
    n: int
    observed: pd.DataFrame
    expected: pd.DataFrame

    @property
    def measures(self):
        return tuple(self.estimate.index)

    def to_frame(self):
        """Return a fresh table of full-precision point estimates."""
        return self.estimate.rename("estimate").to_frame().copy()

    def summary(self, digits: int = 3):
        _digits(digits)
        frame = self.to_frame()
        return "Nominal association measures\n\n" + frame.to_string(
            float_format=lambda value: f"{value:.{digits}f}"
        )

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.summary()


def _digits(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError("digits must be a non-negative integer.")


def _flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be True or False.")
    return bool(value)


def _table(x, *, positive_margins=True, shape=None):
    try:
        values = np.asarray(x)
    except (TypeError, ValueError):
        raise ValueError("x must be a two-dimensional numeric count table.") from None
    if (values.ndim != 2 or min(values.shape) < 2
            or values.dtype.kind not in "iuf"
            or (shape is not None and values.shape != shape)):
        raise ValueError("x must be a two-dimensional numeric count table "
                         "with at least two rows and columns.")
    if (not np.all(np.isfinite(values)) or np.any(values < 0)
            or np.any(values != np.floor(values))):
        raise ValueError("x must contain finite non-negative integer counts.")
    if np.any(values > 2 ** 53 - 1) or float(values.sum(dtype=float)) > 2 ** 53 - 1:
        raise ValueError("The total count must not exceed 2**53 - 1.")
    values = values.astype(np.int64)
    if positive_margins and (np.any(values.sum(axis=0) == 0)
                             or np.any(values.sum(axis=1) == 0)):
        raise ValueError("x must have positive row and column totals.")
    if isinstance(x, pd.DataFrame):
        frame = pd.DataFrame(values, index=x.index.copy(), columns=x.columns.copy())
    else:
        frame = pd.DataFrame(values)
    return values, frame


def _expected(tab):
    return np.outer(tab.sum(axis=1).astype(float), tab.sum(axis=0)) / tab.sum()


def indep_test(x, *, method="pearson", correct=False, **options):
    """Test row/column independence in a two-dimensional integer count table.

    Parameters
    ----------
    x : array_like or pandas.DataFrame
        Nonnegative integer counts, with positive row and column margins.
    method : {"pearson", "lr", "fisher"}, default "pearson"
        Pearson chi-square, likelihood-ratio G-square, or conditional Fisher.
    correct : bool, default False
        R's capped Yates correction, only for Pearson on a 2 by 2 table.
    **options
        Fisher only: ``alternative`` ("two.sided", "less", "greater"),
        ``conf_int`` (True), ``conf_level`` (0.95), ``simulate_p_value``
        (False), ``n_resamples`` (2000), ``random_state`` (None), and
        ``max_tables`` (1,000,000 exact enumeration nodes). One-sided Fisher
        inference requires 2 by 2 counts. Simulation affects larger tables
        only and uses fixed margins and the plus-one p-value convention.

    Returns
    -------
    TestResult
        Statistic, df, p_value, expected/observed tables, and Pearson
        diagnostics for asymptotic tests. Fisher additionally reports the
        conditional maximum likelihood odds ratio and optional exact
        confidence limits for 2 by 2 counts; these are not proportion limits.

    Raises
    ------
    ValueError
        If counts, margins, or options are invalid, or exact Fisher
        enumeration exceeds max_tables.

    Warns
    -----
    RuntimeWarning
        For Pearson and likelihood-ratio tests when any expected count
        is below five.

    See Also
    --------
    nominal_assoc : Descriptive nominal association measures.
    or_2x2_ci : Odds-ratio intervals with several constructions.
    or_common_test : Exact conditional test of a specified common null OR.

    Notes
    -----
    Pearson and LR warn if any expected count is below five. LR uses
    ``2*sum(O*log(O/E))``, with zero contributions from zero cells. P-values
    use the chi-square survival function, retaining tiny upper tails.
    Fisher orders tables by conditional probability, not Pearson statistics.
    General exact computation enumerates fixed-margin tables and can be
    expensive; exceeding max_tables raises an error and never silently
    switches to Monte Carlo. Explicit simulation uses SciPy's Patefield
    sampler, available with the declared SciPy >= 1.10 dependency. A seed
    reproduces Python samples, not R's random stream. Probability ties use
    the relative tolerance 1e-7, matching R's 2 by 2 convention.
    Empty margins, fractional counts, and totals above 2**53-1 are rejected.

    References
    ----------
    Fisher (1935), JRSS 98, 39--82, doi:10.2307/2342435.
    Patefield (1981), Applied Statistics 30, 91--97, doi:10.2307/2346669.
    SAS Institute (2016), SAS/STAT 14.2 User's Guide, FREQ, pp. 2759--2763.

    Examples
    --------
    >>> result = indep_test([[12, 8], [5, 15]], method="lr")
    >>> result.df
    1
    """
    if not isinstance(method, str) or method not in ("pearson", "lr", "fisher"):
        raise ValueError("method must be pearson, lr, or fisher.")
    correct = _flag(correct, "correct")
    tab, observed = _table(x)
    if correct and (method != "pearson" or tab.shape != (2, 2)):
        raise ValueError("correct=True is only available for Pearson on a 2 x 2 table.")
    expected = _expected(tab)
    expected_frame = pd.DataFrame(expected, index=observed.index,
                                  columns=observed.columns)
    if method == "fisher":
        return _fisher(tab, observed, expected_frame, options)
    if options:
        raise ValueError("Additional options are available only for Fisher's method.")
    if np.any(expected < 5):
        warnings.warn("Chi-squared approximation may be incorrect: expected "
                      "counts are below 5.", RuntimeWarning, stacklevel=2)
    residuals = (tab - expected) / np.sqrt(expected)
    row_fraction = tab.sum(axis=1)[:, None] / tab.sum()
    col_fraction = tab.sum(axis=0)[None, :] / tab.sum()
    stdres = (tab - expected) / np.sqrt(expected * (1 - row_fraction)
                                       * (1 - col_fraction))
    deviation = np.abs(tab - expected)
    if correct:
        deviation -= min(0.5, float(deviation.min()))
    if method == "pearson":
        statistic = float(np.sum(deviation ** 2 / expected))
        description = "Pearson's chi-squared test of independence"
        if correct:
            description += " with Yates' continuity correction"
    else:
        positive = tab > 0
        statistic = max(0.0, float(2 * np.sum(tab[positive]
                                             * np.log(tab[positive] / expected[positive]))))
        description = "Likelihood-ratio G-squared test of independence"
    df = (tab.shape[0] - 1) * (tab.shape[1] - 1)
    return TestResult(statistic, float(chi2.sf(statistic, df)), description, df,
                      observed=observed, expected=expected_frame,
                      residuals=pd.DataFrame(residuals, index=observed.index,
                                              columns=observed.columns),
                      stdres=pd.DataFrame(stdres, index=observed.index,
                                          columns=observed.columns))


def nominal_assoc(x, *, measure=("phi", "contingency", "cramer-v")):
    """Compute unadjusted Phi, Pearson contingency coefficient, and Cramer's V.

    Parameters
    ----------
    x : array_like or pandas.DataFrame
        Two-dimensional nonnegative integer count table with at least two
        rows and columns and positive margins. Totals above 2**53-1 are
        rejected. DataFrame row and column labels are preserved.
    measure : str or sequence of str, default all three measures
        "phi", "contingency", or "cramer-v". Requested order is retained
        and duplicates are removed. Category ordering has no effect.

    Returns
    -------
    NominalAssociationResult
        estimate is a named pandas Series; measures records its order.
        Also includes uncorrected Pearson statistic, total n, and observed
        and expected DataFrames. to_frame() returns a fresh estimate table;
        summary() rounds display values without changing stored precision.

    Raises
    ------
    ValueError
        If the count table, margins, or requested measure is invalid.

    See Also
    --------
    indep_test : Independence tests and Pearson residual diagnostics.

    Notes
    -----
    For Pearson X2, total n, and d=min(rows, columns), the estimates are
    sqrt(X2/n), sqrt(X2/(X2+n)), and sqrt(X2/(n*(d-1))), respectively.
    Phi is nonnegative even for binary tables and can exceed one for d>2;
    its maximum is sqrt(d-1). The contingency coefficient has maximum
    sqrt((d-1)/d). Cramer's V lies in [0, 1] and equals Phi when d=2.

    These unadjusted definitions match R nominal.assoc() and DescTools'
    defaults. The cited SAS formulas retain the sign of Phi and V for
    2 by 2 tables; ibist returns their absolute values. Empty margins are
    rejected, not dropped. No confidence intervals, hypothesis tests, bias
    corrections, or sparse-count warnings are produced. Use indep_test()
    for independence tests and chi-square approximation warnings.

    References
    ----------
    SAS Institute Inc. (2016). SAS/STAT 14.2 User's Guide, The FREQ
    Procedure, p. 2764: Phi Coefficient, Contingency Coefficient, and
    Cramer's V.
    https://support.sas.com/documentation/onlinedoc/stat/142/freq.pdf

    Examples
    --------
    >>> tumor = [[109, 68, 14], [72, 79, 37], [9, 42, 139]]
    >>> nominal_assoc(tumor).measures
    ('phi', 'contingency', 'cramer-v')
    >>> nominal_assoc(tumor, measure="cramer-v").measures
    ('cramer-v',)
    """
    tab, observed = _table(x)
    try:
        names = [measure] if isinstance(measure, str) else list(measure)
    except TypeError:
        raise ValueError("measure must be a name or sequence of names.") from None
    allowed = ("phi", "contingency", "cramer-v")
    if not names or any(not isinstance(m, str) or m not in allowed for m in names):
        raise ValueError(f"measure must select one or more of {allowed}.")
    names = list(dict.fromkeys(names))
    expected = _expected(tab)
    statistic = float(np.sum((tab - expected) ** 2 / expected))
    n = int(tab.sum())
    estimates = dict(phi=math.sqrt(statistic / n),
                     contingency=math.sqrt(statistic / (statistic + n)),
                     **{"cramer-v": math.sqrt(statistic / (n * (min(tab.shape) - 1)))})
    return NominalAssociationResult(
        pd.Series([estimates[m] for m in names], index=names, dtype=float),
        statistic, n, observed,
        pd.DataFrame(expected, index=observed.index, columns=observed.columns)
    )


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _fisher(tab, observed, expected, options):
    allowed = {"alternative", "conf_int", "conf_level", "simulate_p_value",
               "n_resamples", "random_state", "max_tables"}
    if options.keys() - allowed:
        raise ValueError("Unknown Fisher options: " + ", ".join(sorted(options.keys() - allowed)))
    alternative = options.get("alternative", "two.sided")
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError("alternative must be two.sided, less, or greater.")
    ci = _flag(options.get("conf_int", True), "conf_int")
    level = _probability(options.get("conf_level", 0.95), "conf_level")
    simulate = _flag(options.get("simulate_p_value", False), "simulate_p_value")
    b = _positive_integer(options.get("n_resamples", 2000), "n_resamples")
    max_tables = _positive_integer(options.get("max_tables", 1000000), "max_tables")
    if tab.shape == (2, 2):
        n = int(tab.sum()); r = int(tab[0].sum()); c = int(tab[:, 0].sum())
        a = int(tab[0, 0])
        if alternative == "less":
            p_value = float(hypergeom.cdf(a, n, r, c))
        elif alternative == "greater":
            p_value = float(hypergeom.sf(a - 1, n, r, c))
        else:
            low, high = max(0, r + c - n), min(r, c)
            if high - low + 1 > max_tables:
                raise ValueError("Fisher support exceeds max_tables.")
            support = np.arange(low, high + 1)
            logprob = hypergeom.logpmf(support, n, r, c)
            threshold = hypergeom.logpmf(a, n, r, c) + np.log1p(1e-7)
            p_value = float(np.exp(logsumexp(logprob[logprob <= threshold])))
        fitted = odds_ratio(tab, kind="conditional")
        limits = None
        if ci:
            bounds = fitted.confidence_interval(confidence_level=level,
                                                 alternative=alternative.replace(".", "-"))
            limits = (float(bounds.low), float(bounds.high))
        return TestResult(None, min(1.0, p_value), "Fisher's exact test of independence",
                          alternative=alternative, observed=observed, expected=expected,
                          estimate=float(fitted.statistic), conf_int=limits,
                          conf_level=level if ci else None, null_value=1.0)
    if alternative != "two.sided":
        raise ValueError("One-sided Fisher alternatives require a 2 x 2 table.")
    rows, cols = tab.sum(axis=1), tab.sum(axis=0)
    threshold = float(_fixed_margin_logpmf(tab)) + np.log1p(1e-7)
    if simulate:
        rng = np.random.default_rng(options.get("random_state"))
        extreme = 0
        for start in range(0, b, 1000):
            samples = random_table.rvs(rows, cols, size=min(1000, b - start),
                                        method="patefield", random_state=rng)
            weights = _fixed_margin_logpmf(samples)
            extreme += int(np.sum(weights <= threshold))
        description = ("Fisher's test of independence with simulated p-value "
                       f"(based on {b} replicates)")
        return TestResult(None, (extreme + 1) / (b + 1), description,
                          observed=observed, expected=expected, n_resamples=b)
    p_value = _fisher_enumerate(rows, cols, threshold, max_tables)
    return TestResult(None, p_value, "Fisher's exact test of independence",
                      observed=observed, expected=expected)


def _row_allocations(total, columns):
    """Yield bounded compositions of one row under the remaining margins."""
    if len(columns) == 1:
        if 0 <= total <= columns[0]:
            yield (total,)
        return
    low = max(0, total - sum(columns[1:]))
    high = min(total, columns[0])
    for first in range(low, high + 1):
        for remainder in _row_allocations(total - first, columns[1:]):
            yield (first,) + remainder


def _fisher_enumerate(rows, columns, threshold, max_tables):
    # Transpose to minimize the number of rows explored recursively.
    if len(rows) > len(columns):
        rows, columns = columns, rows
    rows = tuple(int(v) for v in rows)
    columns = tuple(int(v) for v in columns)
    terms = []
    visited = 0

    def visit(row, remaining, weight):
        nonlocal visited
        visited += 1
        if visited > max_tables:
            raise ValueError("Exact Fisher enumeration exceeded max_tables; "
                             "request simulate_p_value=True or increase max_tables.")
        if row == len(rows) - 1:
            if weight <= threshold:
                terms.append(weight)
            return
        for cells in _row_allocations(rows[row], remaining):
            visit(row + 1, tuple(a - b for a, b in zip(remaining, cells)),
                  weight + float(_row_logpmf(np.array(cells),
                                             np.array(remaining))))

    visit(0, columns, 0.0)
    return min(1.0, float(np.exp(logsumexp(terms))))



def _row_logpmf(cells, columns):
    """Sequential hypergeometric weights avoid subtracting huge log factorials."""
    population = columns.sum(axis=-1)
    draws = cells.sum(axis=-1)
    result = np.zeros_like(population, dtype=float)
    for j in range(cells.shape[-1] - 1):
        # A zero remaining population is deterministic, with probability one.
        result += hypergeom.logpmf(cells[..., j], np.maximum(population, 1),
                                    columns[..., j], draws)
        population = population - columns[..., j]
        draws = draws - cells[..., j]
    return result


def _fixed_margin_logpmf(tables):
    columns = tables.sum(axis=-2)
    result = np.zeros(tables.shape[:-2], dtype=float)
    for i in range(tables.shape[-2] - 1):
        result += _row_logpmf(tables[..., i, :], columns)
        columns = columns - tables[..., i, :]
    return result
