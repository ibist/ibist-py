"""Shared pre- and post-inference approximation diagnostics."""

from copy import deepcopy
from dataclasses import dataclass
import inspect
import math

import numpy as np
import pandas as pd


_CHECK_COLUMNS = ("method", "criterion", "value", "threshold", "comparison",
                  "condition_met")


@dataclass(frozen=True, repr=False)
class AsymCheckResult:
    """Method-specific approximation diagnostics, matching ibist_asymcheck.

    procedure, method identify the inference; applicable is false for exact
    procedures. checks returns a fresh DataFrame with method, criterion,
    value, threshold, comparison, condition_met. details returns copied
    supporting quantities. notes explains limitations. condition_met
    summarizes assessed criteria only, or is None when none was assessed.
    This is not a hypothesis-test result or a guarantee of valid inference.
    """

    procedure: str
    method: str | tuple[str, ...]
    applicable: bool
    notes: tuple[str, ...]
    _checks: tuple
    _details: dict
    _check_columns: tuple[str, ...] = _CHECK_COLUMNS

    @property
    def checks(self):
        """Fresh criterion table, with None for unassessed conditions."""
        return pd.DataFrame(self._checks, columns=self._check_columns)

    @property
    def details(self):
        """Copied expected counts, variances, or interval settings."""
        return deepcopy(self._details)

    @property
    def condition_met(self):
        """All assessed criteria met, or None when none was assessed."""
        assessed = [row[5] for row in self._checks if row[5] is not None]
        return all(assessed) if assessed else None

    def summary(self, digits=4):
        """Display diagnostics without changing stored precision."""
        from .contingency import _digits
        _digits(digits)
        table = self.checks.to_string(index=False,
                                     float_format=lambda x: f"{x:.{digits}g}")
        return (f"Asymptotic diagnostics: {self.procedure}\n{table}\n" +
                "\n".join(self.notes))

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.summary()


def asymcheck(object, *args, method=None, **kwargs):
    """Check approximation diagnostics before or after inference.

    Parameters
    ----------
    object : callable, supported result, or count data
        indep_test, proportion/risk-ratio/odds-ratio CI functions, rate_1s_ci,
        rate_2s_ci, rate_test, or glm_gof for pre-checks; their results for
        post-checks, including GLMGOFResults collections.
        Raw data require an explicit method. No inference is run by a
        pre-check, and result data names are never evaluated.
    *args, **kwargs
        Usual inference arguments for function pre-checks. Post-checks
        accept no additional inputs or overrides. Raw one-rate counts
        accept rate_1s_ci arguments. Extra Fisher options are unsupported
        in pre-checks; the diagnostic does not depend on tail enumeration.
    method : str or sequence of str, optional
        For raw counts: pearson, lr, fisher, mantel-haenszel, or one-rate
        interval methods exact, score, wh, wald, log. For a callable, its
        own method argument. Omit when checking a result.

    Returns
    -------
    AsymCheckResult
        Procedure, method, applicability, criterion table, details, notes,
        and condition_met. Unassessed criteria have condition_met=None.

    Raises
    ------
    ValueError
        For unsupported functions/results, missing retained inputs,
        invalid counts/settings, or post-check input overrides.

    See Also
    --------
    indep_test : Independence inference with optional attached diagnostics.
    rate_1s_ci : Rate intervals with optional attached diagnostics.

    Notes
    -----
    Matches R asymcheck(). Following the book, 2x2 tests require all expected
    counts >=5; larger tables use minimum >=1 and fraction >=5 of at least .8.
    These Pearson heuristics do not prove
    G-squared accuracy. Mantel-Haenszel retains R's legacy sum of conditional
    hypergeometric variances >=5 heuristic, with empty strata contributing
    zero. Raw stratified inputs are lists of 2x2 tables or 2x2xK arrays.

    Proportion, RR, and OR approximate intervals check the book's original
    sample quantity n_i*p_hat_i*(1-p_hat_i) >=5 in each sample. Corrections
    and pseudo-counts never increase it. Exact/mid-p methods need no
    asymptotic check; mid-p does not guarantee coverage. CI criterion rows
    identify zero-based input/sample indices, one-based in R. Original
    counts and settings are retained even without attached diagnostics.

    One-rate tests check null expected events r*T >=10. Two-rate equality
    tests check n*pi0*(1-pi0) >=5 with n=x1+x2, pi0=T1/(T1+T2). The check
    concerns the test, not its accompanying interval. Rate-ratio log-Wald
    intervals check both observed counts >10; conditional score uses the
    binomial CI guideline after conditioning on the total.

    One-rate intervals follow book Ch. 21: Wald observed x >=50; log-Wald
    and uncorrected score x >=10. WH is recommended even for small counts;
    no binary cutoff is invented for WH or corrected score. Skewness
    1/sqrt(x) is descriptive, not a pass/fail approximation criterion.
    Exact procedures and zero-event log/uncorrected Wald exact fallbacks
    are not applicable. Multiple CI methods retain order and repetitions.
    Passing criteria does not assess independence, common-effect
    homogeneity, or all approximation assumptions. No p-value or interval
    decision is used and the selected inference method is never changed.
    All GLM GOF methods have asymptotic reference laws. Their pre-checks
    report nonpositive df or degenerate standardization as failed criteria;
    inference rejects these unestimable tests. Details retain convergence,
    rank, and fitted-value boundary distance.

    References
    ----------
    R Core Team. Fisher's Exact Test for Count Data, Details (Cochran
    conditions). R stats documentation, https://stat.ethz.ch/R-manual/
    NIST/SEMATECH e-Handbook, 1.3.6.6.19, Poisson Distribution (skewness).
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda366j.htm
    The variance-sum threshold preserves ibist-R's existing diagnostic.
    Companion textbook, Chapters 11--12 (proportions), 14 (contingency tables),
    15 (RR/OR and MH), and 21--22 (rates), inferential summaries and practical
    notes. Book guidelines do not guarantee coverage or approximation accuracy.

    Examples
    --------
    >>> from ibist import asymcheck, indep_test, rate_1s_ci
    >>> asymcheck(indep_test, [[12, 8], [5, 15]]).condition_met
    True
    >>> fit = rate_1s_ci(12, 100, method="wald", asymcheck=True)
    >>> asymcheck(fit).applicable
    True
    """
    from .contingency import TestResult, indep_test
    from .rates import rate_1s_ci
    from .results import CIResult
    from .glm_gof import GLMGOFResults, glm_gof, _combine, _precheck
    from .proportions import prop_1s_ci, prop_2s_ci
    from .risk_ratio import rr_2x2_ci
    from .odds_ratio import or_2x2_ci
    from .rates import rate_2s_ci, rate_test
    from ._asymcheck_book import _book_ci, _rate_test
    adapters = {prop_1s_ci: "p1", prop_2s_ci: "p2", rr_2x2_ci: "rr",
                or_2x2_ci: "or", rate_2s_ci: "rate2"}

    if callable(object):
        if object is glm_gof:
            if method is not None:
                kwargs["method"] = method
            try:
                bound = inspect.signature(object).bind(*args, **kwargs)
            except TypeError as exc:
                raise ValueError(str(exc)) from None
            bound.apply_defaults()
            return _precheck(**bound.arguments)
        if object not in (*adapters, indep_test, rate_1s_ci, rate_test):
            raise ValueError("Unsupported inference function.")
        if method is not None:
            kwargs["method"] = method
        try:
            bound = inspect.signature(object).bind(*args, **kwargs)
        except TypeError as exc:
            raise ValueError(str(exc)) from None
        bound.apply_defaults()
        values = bound.arguments
        from .contingency import _flag
        _flag(values.pop("asymcheck"), "asymcheck")
        if object in adapters:
            return _book_ci(adapters[object], **values)
        if object is rate_test:
            return _rate_test(**values)
        if object is indep_test:
            if values.pop("options"):
                raise ValueError("Extra Fisher options are unsupported here.")
            return _indep(**values)
        return _rate(**values)
    if isinstance(object, GLMGOFResults):
        if args or kwargs or method is not None:
            raise ValueError("Post-checks do not accept input overrides.")
        return _combine([result._gof_diagnostics for result in object])
    if isinstance(object, (TestResult, CIResult)):
        if args or kwargs or method is not None:
            raise ValueError("Post-checks do not accept input overrides.")
        if (isinstance(object, TestResult) and
                object._gof_diagnostics is not None):
            return object._gof_diagnostics
        if object._asymcheck_context:
            context = dict(object._asymcheck_context)
            family = context.pop("family")
            if family == "rate-test":
                return _rate_test(**context)
            return _book_ci(family, **context)
        if isinstance(object, TestResult) and object.observed is not None:
            label = object.method.lower()
            selected = ("fisher" if "fisher" in label else
                        "lr" if "g-squared" in label else
                        "pearson" if "pearson" in label else None)
            if selected is not None:
                return _indep(object.observed, selected, "yates" in label)
        if (isinstance(object, CIResult) and
                object.parameter == "Poisson rate" and
                len(object._rate_data) == 1):
            x, exposure = object._rate_data[0]
            return _rate(x, exposure, object.conf_level, object.methods,
                         object.correct, object.alternative, object.side)
        raise ValueError("Unsupported result or missing retained inputs; "
                         "use a pre-check.")
    if method is None:
        raise ValueError("Raw inputs require an explicit method.")
    if isinstance(method, str) and method == "mantel-haenszel":
        if args or kwargs:
            raise ValueError("Raw Mantel-Haenszel checks accept counts only.")
        return _mh(object)
    function = (indep_test if isinstance(method, str) and
                method in ("pearson", "lr", "fisher") else rate_1s_ci)
    return asymcheck(function, object, *args, method=method, **kwargs)


def _result(procedure, method, applicable, rows, details, notes,
            columns=_CHECK_COLUMNS):
    notes = tuple(notes) + ("Criteria do not verify all sampling assumptions.",)
    return AsymCheckResult(procedure, method, bool(applicable), notes,
                           tuple(rows), deepcopy(details), tuple(columns))


def _indep(x, method="pearson", correct=False):
    from .contingency import _expected, _flag, _table
    if not isinstance(method, str) or method not in ("pearson", "lr", "fisher"):
        raise ValueError("method must be pearson, lr, or fisher.")
    correct = _flag(correct, "correct")
    tab, observed = _table(x)
    if correct and (method != "pearson" or tab.shape != (2, 2)):
        raise ValueError("Correction requires a 2 x 2 Pearson test.")
    expected = _expected(tab)
    applicable = method != "fisher"
    thresholds = (5., 1.) if tab.shape == (2, 2) else (1., .8)
    rows = []
    for criterion, value, threshold in [
        ("minimum expected count", float(expected.min()), thresholds[0]),
        ("fraction expected >= 5", float(np.mean(expected >= 5)),
         thresholds[1]),
    ]:
        rows.append((method, criterion, value,
                     threshold if applicable else None,
                     ">=" if applicable else None,
                     bool(value >= threshold) if applicable else None))
    notes = (["Expected-count rules are approximation screening heuristics."]
             if applicable else
             ["Conditional Fisher inference needs no asymptotic check."])
    if method == "lr":
        notes.append("Pearson screening rules do not establish G-squared "
                     "accuracy.")
    frame = pd.DataFrame(expected, index=observed.index,
                         columns=observed.columns)
    return _result("independence", method, applicable, rows,
                   dict(expected=frame, correct=correct), notes)


def _mh(x):
    from .contingency import _table
    if isinstance(x, np.ndarray) and x.ndim == 3 and x.shape[:2] == (2, 2):
        tables = [x[:, :, i] for i in range(x.shape[2])]
    elif isinstance(x, (list, tuple)):
        tables = list(x)
    else:
        raise ValueError("Use a list or 2 x 2 x K array.")
    if not tables:
        raise ValueError("At least one stratum is required.")
    variances = []
    for table in tables:
        tab, _ = _table(table, positive_margins=False, shape=(2, 2))
        n = float(tab.sum())
        variance = (float(np.prod(np.r_[tab.sum(axis=0),
                                       tab.sum(axis=1)].astype(float))) /
                    ((n-1)*n*n) if n > 1 else 0.)
        variances.append(variance)
    total = math.fsum(variances)
    rows = [("mantel-haenszel", "sum of variances", total,
             5., ">=", total >= 5)]
    variance = pd.Series(variances,
                         index=[f"Stratum {i+1}" for i in range(len(tables))])
    return _result("stratified independence", "mantel-haenszel", True, rows,
                   dict(variance=variance, variance_sum=total),
                   ["Legacy variance-sum heuristic; common-effect homogeneity "
                    "is unchecked."])


def _rate(x, exposure=1., conf_level=.95, method="exact", correct=True,
          alternative=None, side=None):
    from .contingency import _flag
    from .odds_ratio import _probability
    from .rates import _rate_inputs
    from .results import _ci_orientation
    from .risk_ratio import _methods
    counts, exposures = _rate_inputs(x, exposure, lengths=(1,))
    x, exposure = int(counts[0]), float(exposures[0])
    conf_level = _probability(conf_level, "conf_level")
    correct = _flag(correct, "correct")
    side, _ = _ci_orientation(side, alternative)
    methods = tuple(_methods(method, ("exact", "score", "wh", "wald", "log")))
    fallback = tuple(x == 0 and (m == "log" or
                     (m == "wald" and not correct)) for m in methods)
    approximate = tuple(m != "exact" and not f
                        for m, f in zip(methods, fallback))
    rows = []
    for m, approx in zip(methods, approximate):
        threshold = (50. if m == "wald" else 10. if m == "log" or
                     (m == "score" and not correct) else None)
        if not approx:
            threshold = None
        rows += [(m, "observed events", float(x), threshold,
                  ">=" if threshold is not None else None,
                  bool(x >= threshold) if threshold is not None else None),
                 (m, "plug-in skewness", 1/math.sqrt(x) if x and approx
                  else None, None, None, None)]
    notes = ["Book Ch. 21: observed-count guidance, not a coverage guarantee.",
             "Wald: y >= 50; log-Wald and uncorrected score: y >= 10.",
             "WH is recommended even for small counts; no binary cutoff is "
             "given.",
             "Corrected score/WH have no separate observed-count cutoff in "
             "the book."]
    if not x:
        notes.append("Zero events: plug-in skewness is undefined.")
    if any(fallback):
        notes.append("Zero-event log/uncorrected Wald uses exact fallback.")
    if not any(approximate):
        notes.append("No asymptotic interval is being assessed.")
    return _result("one-sample Poisson rate interval",
                   methods[0] if len(methods) == 1 else methods,
                   any(approximate), rows,
                   dict(x=x, exposure=exposure, conf_level=conf_level,
                        side=side, correct=correct, approximate=approximate,
                        exact_fallback=fallback), notes)
