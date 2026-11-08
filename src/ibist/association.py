"""Common tests of nominal and ordinal association for count tables."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from .contingency import TestResult, _digits, _flag, indep_test, nominal_assoc
from .odds_ratio import _probability
from .ordinal import _MEASURES, ordinal_assoc


_NOMINAL = ("phi", "contingency", "cramer-v")


@dataclass(frozen=True, repr=False)
class AssociationTestResults(Sequence):
    """Ordered collection of TestResult objects from a multi-measure call.

    Attributes
    ----------
    results : tuple of TestResult
        Individual full-precision tests in requested order, with repetitions.
        Indexing and iteration also expose these results.
    shared_tests : tuple of tuples of str
        Requested measure groups sharing one statistical test.

    Notes
    -----
    Corresponds to R's ibist_assoc_tests. measures exposes coefficient names;
    to_frame() returns a fresh combined table, and summary() rounds only for
    display. Tests use the same data and p-values are unadjusted.
    """

    results: tuple[TestResult, ...]
    shared_tests: tuple[tuple[str, ...], ...] = ()

    def __len__(self):
        return len(self.results)

    def __getitem__(self, item):
        return self.results[item]

    @property
    def measures(self):
        """Selected measure names in requested order, including repetitions."""
        return tuple(result.measure for result in self.results)

    def to_frame(self):
        """Return a fresh full-precision table of estimates and tests."""
        rows = []
        for result in self.results:
            lower, upper = result.conf_int or (math.nan, math.nan)
            rows.append((result.estimate, result.std_error, lower, upper,
                         result.statistic, result.df, result.p_value))
        return pd.DataFrame(rows, index=pd.Index(self.measures, name="measure"),
                            columns=["estimate", "std_error", "lower", "upper",
                                     "statistic", "df", "p_value"],
                            dtype=float)

    def summary(self, digits=3):
        """Format results, shared-test groups, and unadjusted p-values."""
        _digits(digits)
        table = self.to_frame().to_string(
            float_format=lambda value: f"{value:.{digits}f}",
            formatters={"p_value": lambda value:
                        f"{value:.{max(1, digits)}g}"})
        notes = ["Shared test: " + ", ".join(group)
                 for group in self.shared_tests]
        notes.append("Tests use the same data; p-values are unadjusted.")
        return (f"Association tests ({self.results[0].test_method})\n\n"
                + table + "\n\n" + "\n".join(notes))

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.summary()


def assoc_test(x, *, measure="pearson", method="multinomial",
               alternative="two.sided", conf_level=0.95, row_scores=None,
               col_scores=None, correct=False):
    """Test nominal or ordinal association using one or more measures.

    Parameters
    ----------
    x : array_like or pandas.DataFrame
        Two-dimensional nonnegative integer count table. Counts and total
        must not exceed 2**53-1. Ordinal rows/columns must already be ordered.
    measure : str or sequence of str, default "pearson"
        Nominal: "phi", "contingency", "cramer-v". Ordinal: "gamma", "tau-b",
        "tau-c", "somers-c|r", "somers-r|c", "pearson", "spearman".
        Requested order and repetitions are retained.
    method : {"multinomial", "pearson", "cmh"}, default "multinomial"
        One procedure applies to the whole call. Multinomial supports all
        ordinal measures; Pearson independence supports nominal measures
        only; CMH linear-by-linear supports Pearson/Spearman only.
    alternative : {"two.sided", "greater", "less"}, default "two.sided"
        Direction for tests of zero ordinal association. Nominal independence
        tests require two.sided. Multinomial intervals follow this direction.
    conf_level : float, default 0.95
        Confidence level strictly between zero and one. Used for multinomial
        intervals only, but validated for all methods.
    row_scores, col_scores : one-dimensional array_like, optional
        Pearson category scores, following ordinal_assoc(). For CMH, supply
        both or neither; in a mixed call they apply to Pearson only.
        Scores are rejected for nominal tests and CMH Spearman-only calls.
    correct : bool, default False
        Capped Yates correction for nominal Pearson tests of 2 by 2 tables
        only. Nominal coefficient estimates remain uncorrected.

    Returns
    -------
    TestResult or AssociationTestResults
        A single measure returns TestResult with its estimate, zero null,
        statistic, p_value, alternative, method description, measure, and
        test_method. Multinomial adds std_error, conf_int, and conf_level;
        nominal Pearson retains df and independence diagnostics; CMH adds
        chisq (the square of its signed Z). Unsupported intervals are None.
        Multiple measures return an ordered sequence of TestResult objects
        with measures, shared_tests, to_frame(), and a combined summary().

    Raises
    ------
    ValueError
        If inputs are invalid, a measure/method combination is unsupported,
        or method-specific arguments are incompatible. Mixed nominal and
        ordinal measure requests are rejected rather than switching methods.

    Warns
    -----
    RuntimeWarning
        For nominal Pearson inference when an expected count is below five.

    See Also
    --------
    ordinal_assoc : Multi-measure ordinal estimation and multinomial tests.
    nominal_assoc : Descriptive nominal association measures.
    indep_test : Independence tests, including likelihood-ratio and Fisher.

    Notes
    -----
    Definitions follow R assoc.test(). Nominal measures share the Pearson
    independence test; no coefficient confidence intervals are supplied.
    Empty margins are rejected for nominal tests and dropped for ordinal
    methods. The five concordance measures share a multinomial test, while
    Pearson and Spearman use distinct multinomial null variances.

    CMH uses Z=r*sqrt(n-1), with chisq=Z**2 and signed normal tails.
    It differs from multinomial inference for the same correlation and
    supplies no confidence interval. Multinomial intervals use estimation
    variances, not null variances; they need not invert their tests.
    Two-sided Wald intervals are untruncated; one-sided finite bounds are
    clipped to [-1, 1]. See ordinal_assoc() for score and rank definitions.

    Multiple measures use the same observations. Shared tests are identified
    explicitly; p-values are not adjusted for multiple comparisons. Full
    numerical precision is retained and rounding affects summaries only.

    References
    ----------
    Brown and Benedetti (1977), JASA 72(358), 309--315,
    doi:10.1080/01621459.1977.10480995.
    Mantel (1963), JASA 58(303), 690--700,
    doi:10.1080/01621459.1963.10500879.
    SAS Institute (2016), SAS/STAT 14.2 User's Guide, The FREQ Procedure,
    Chi-Square Tests and Statistics; Measures of Association.

    Examples
    --------
    >>> table = [[12, 3, 1], [4, 9, 2], [1, 5, 13]]
    >>> assoc_test(table).measure
    'pearson'
    >>> results = assoc_test(table, measure=["gamma", "tau-b"])
    >>> results.measures
    ('gamma', 'tau-b')
    >>> results.shared_tests
    (('gamma', 'tau-b'),)
    >>> assoc_test(table, measure="cramer-v", method="pearson").df
    4
    """
    try:
        measures = [measure] if isinstance(measure, str) else list(measure)
    except TypeError:
        raise ValueError("measure must contain valid measure names.") from None
    if not measures or any(not isinstance(name, str)
                           or name not in _NOMINAL + _MEASURES
                           for name in measures):
        raise ValueError("measure must contain valid measure names.")
    choices = {"multinomial": _MEASURES, "pearson": _NOMINAL,
               "cmh": ("pearson", "spearman")}
    if not isinstance(method, str) or method not in choices:
        raise ValueError("method must be multinomial, pearson, or cmh.")
    if any(name not in choices[method] for name in measures):
        raise ValueError("Selected measures are incompatible with method.")
    if not isinstance(alternative, str) or alternative not in (
            "two.sided", "greater", "less"):
        raise ValueError("alternative must be two.sided, greater, or less.")
    conf_level = _probability(conf_level, "conf_level")
    correct = _flag(correct, "correct")
    if correct and method != "pearson":
        raise ValueError("correct=True is only for nominal Pearson tests.")
    scores = row_scores is not None or col_scores is not None
    if method == "pearson" and (scores or alternative != "two.sided"):
        raise ValueError("Nominal tests require two.sided and no scores.")
    if method == "cmh" and (
            (row_scores is None) != (col_scores is None)
            or (scores and "pearson" not in measures)):
        raise ValueError("CMH scores require both axes and a Pearson request.")
    results = []
    if method == "pearson":
        fit = indep_test(x, method="pearson", correct=correct)
        estimates = nominal_assoc(x, measure=measures).estimate
        for name in measures:
            description = f"Pearson independence test: {name}"
            if correct:
                description += "; Yates correction"
            results.append(replace(
                deepcopy(fit), estimate=float(estimates[name]), null_value=0.0,
                alternative=alternative, method=description, measure=name,
                test_method=method, estimate_name=name))
    else:
        fit = ordinal_assoc(x, measure=measures, conf_level=conf_level,
                            alternative=alternative, row_scores=row_scores,
                            col_scores=col_scores)
        for i, name in enumerate(measures):
            estimate = float(fit.estimate.iloc[i])
            if method == "multinomial":
                result = TestResult(
                    float(fit.statistic.iloc[i]), float(fit.p_value.iloc[i]),
                    f"Multinomial test of zero {name} association",
                    alternative=alternative, estimate=estimate, null_value=0.0,
                    conf_int=tuple(fit.conf_int.iloc[i]), conf_level=conf_level,
                    std_error=float(fit.std_error.iloc[i]))
            else:
                z = estimate * math.sqrt(fit.n - 1)
                p_value = (2 * norm.sf(abs(z)) if alternative == "two.sided"
                           else norm.sf(z) if alternative == "greater"
                           else norm.cdf(z))
                result = TestResult(
                    z, float(p_value), f"CMH linear-by-linear test: {name}",
                    alternative=alternative,
                    estimate=estimate, null_value=0.0, chisq=z**2)
            results.append(replace(result, measure=name, test_method=method,
                                   estimate_name=name))
    if len(results) == 1:
        return results[0]
    group = (measures if method == "pearson" else
             [name for name in measures if name in _MEASURES[:5]]
             if method == "multinomial" else [])
    shared = (tuple(group),) if len(set(group)) > 1 else ()
    return AssociationTestResults(tuple(results), shared)
