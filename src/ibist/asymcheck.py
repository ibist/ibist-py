"""Shared pre- and post-inference approximation diagnostics."""

from copy import deepcopy
from dataclasses import dataclass
import inspect
import math

import numpy as np
import pandas as pd


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

    @property
    def checks(self):
        """Fresh criterion table, with None for unassessed conditions."""
        return pd.DataFrame(self._checks, columns=(
            "method", "criterion", "value", "threshold", "comparison",
            "condition_met"))

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
        indep_test or rate_1s_ci for pre-checks; their results for post-checks.
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
    Matches R asymcheck(). Expected-count screening reports minimum >=1
    and fraction >=5 of at least .8. These Pearson heuristics do not prove
    G-squared accuracy. Mantel-Haenszel retains R's legacy sum of conditional
    hypergeometric variances >=5 heuristic, with empty strata contributing
    zero. Raw stratified inputs are lists of 2x2 tables or 2x2xK arrays.

    Approximate one-rate intervals report events and plug-in Poisson
    skewness 1/sqrt(x), undefined at zero, without a universal cutoff.
    Exact procedures and zero-event log/uncorrected Wald exact fallbacks
    are not applicable. Multiple CI methods retain order and repetitions.
    Passing criteria does not assess independence, common-effect
    homogeneity, or all approximation assumptions. No p-value or interval
    decision is used and the selected inference method is never changed.

    References
    ----------
    R Core Team. Fisher's Exact Test for Count Data, Details (Cochran
    conditions). R stats documentation, https://stat.ethz.ch/R-manual/
    NIST/SEMATECH e-Handbook, 1.3.6.6.19, Poisson Distribution (skewness).
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda366j.htm
    The variance-sum threshold preserves ibist-R's existing diagnostic.

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

    if callable(object):
        if object is not indep_test and object is not rate_1s_ci:
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
        if object is indep_test:
            if values.pop("options"):
                raise ValueError("Extra Fisher options are unsupported here.")
            return _indep(**values)
        return _rate(**values)
    if isinstance(object, (TestResult, CIResult)):
        if args or kwargs or method is not None:
            raise ValueError("Post-checks do not accept input overrides.")
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


def _result(procedure, method, applicable, rows, details, notes):
    notes = tuple(notes) + ("Criteria do not verify all sampling assumptions.",)
    return AsymCheckResult(procedure, method, bool(applicable), notes,
                           tuple(rows), deepcopy(details))


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
    rows = []
    for criterion, value, threshold in [
        ("minimum expected count", float(expected.min()), 1.),
        ("fraction expected >= 5", float(np.mean(expected >= 5)), .8),
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
        rows += [(m, "observed events", float(x), None, None, None),
                 (m, "plug-in skewness", 1/math.sqrt(x) if x and approx
                  else None, None, None, None)]
    notes = ["No universal event-count cutoff is asserted for interval "
             "accuracy."]
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
