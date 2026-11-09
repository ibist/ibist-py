"""Book approximation guidelines, separate from input validation."""

from dataclasses import replace
import math

import numpy as np
import pandas as pd

from .asymcheck import _CHECK_COLUMNS, _result
from .contingency import _flag
from .odds_ratio import _probability, _validate_table
from .results import _ci_orientation


def _freeze(value):
    if isinstance(value, np.ndarray):
        return _freeze(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value.item() if isinstance(value, np.generic) else value


def _attach(result, family, attach, **settings):
    values = {k: _freeze(v) for k, v in settings.items()}
    context = dict(family=family, **values)
    check = (_rate_test(**settings) if family == "rate-test" else
             _book_ci(family, **settings)) if attach else None
    return replace(result, _asymcheck_context=tuple(context.items()),
                   diagnostics=check)


def _book_ci(family, x, n=None, exposure=None, conf_level=.95, method="wilson",
             correct=None, midp=False, alternative=None, side=None, tol=1e-8,
             nuisance_tol=None):
    from .proportions import (_METHODS, _DIFFERENCE_METHODS, _binomial_counts,
                              _independent_counts)
    from .rates import _positive, _rate_inputs
    from .risk_ratio import _methods

    choices = dict(p1=_METHODS, p2=_DIFFERENCE_METHODS, rr=("koopman", "katz"),
                   or_=("wald", "gart", "cornfield", "bp"),
                   rate2=("log", "score", "exact"))
    key = "or_" if family == "or" else family
    methods = tuple(_methods(method, choices[key]))
    side, alternative = _ci_orientation(side, alternative)
    conf_level = _probability(conf_level, "conf_level")
    if family != "or":
        tol = _probability(tol, "tol")
    if family in ("p1", "p2"):
        correct = _flag(correct, "correct")
        correctable = ("wald", "wilson") if family == "p1" else (
            "wald", "newcombe")
        if correct and not any(m in correctable for m in methods):
            raise ValueError("correct applies to none of the requested methods")
    midp = _flag(midp, "midp")
    if family == "p1" and midp and "clopper-pearson" not in methods:
        raise ValueError("midp applies only to Clopper-Pearson")
    if family == "p2":
        nuisance_tol = _probability(nuisance_tol, "nuisance_tol")
    exact = tuple(m in ("clopper-pearson", "blaker", "chan-zhang",
                        "cornfield", "bp", "exact") for m in methods)
    if family == "or" and any(exact):
        tol = _positive(tol, "tol")
    original_events = None
    if family == "rate2":
        events, exposure = _rate_inputs(x, exposure, lengths=(2,))
        total = int(events[0])+int(events[1])
        if not total:
            raise ValueError("at least one event is required for a "
                             "rate-ratio CI")
        counts, sizes = np.array([events[0]]), np.array([total])
        original_events = events
    elif family == "p1":
        counts, sizes = _binomial_counts(x, n)
    else:
        if family in ("or", "rr"):
            x = _validate_table(x)
        counts, sizes = _independent_counts(x, n)
    p = counts/sizes
    values = sizes*p*(1-p)
    rows = []
    inputs = range(len(counts)) if family == "p1" else range(1)
    for input_index in inputs:
        for name, is_exact in zip(methods, exact):
            log_rate = family == "rate2" and name == "log"
            use = [input_index] if family == "p1" else range(
                len(original_events) if log_rate else len(values))
            for sample, i in enumerate(use):
                value = float(original_events[i] if log_rate else values[i])
                threshold = (10. if log_rate else 5.) if not is_exact else None
                comparison = None if is_exact else ">" if log_rate else ">="
                met = (value > 10 if log_rate else value >= 5
                       ) if not is_exact else None
                rows.append((name, "observed events" if log_rate else
                             "n * p.hat * (1-p.hat)", value, threshold,
                             comparison, met, input_index, sample))
    titles = dict(p1="one-proportion", p2="independent proportion-difference",
                  rr="risk-ratio", or_="odds-ratio", rate2="Poisson rate-ratio")
    chapters = dict(p1="11", p2="12", rr="15", or_="15",
                    rate2="22 (and 11 for conditional score)")
    key = "or_" if family == "or" else family
    notes = ["The book's approximation guidelines are not coverage guarantees.",
             "Criteria use original observations; pseudo-counts are not "
             "sample size."]
    if family == "p1":
        notes.append("Ch. 11 binomial-normal guideline; Wald can under-cover "
                     "even if met.")
    if family in ("p2", "rr", "or"):
        notes.append("Check n_i * p.hat_i * (1-p.hat_i) >= 5 in each original "
                     "sample.")
    if family == "rate2":
        notes.append("Log-Wald: both y_i > 10; score: conditional-binomial CI "
                     "guideline.")
    if any(exact):
        notes.append("Exact methods need no asymptotic approximation check.")
    if midp:
        notes.append("Mid-p does not guarantee nominal coverage.")
    details = dict(counts=pd.DataFrame(dict(x=counts, n=sizes)),
                   sample_proportions=p, methods=pd.DataFrame(dict(
                       method=methods, applicable=[not v for v in exact])),
                   conf_level=conf_level, side=side, alternative=alternative,
                   correct=correct, midp=midp, tol=tol,
                   nuisance_tol=nuisance_tol, exposure=exposure,
                   original_events=original_events,
                   source="Book chapter "+chapters[key])
    return _result(titles[key]+" interval",
                   methods[0] if len(methods) == 1 else methods,
                   any(not v for v in exact), rows, details, notes,
                   columns=_CHECK_COLUMNS + ("input", "sample"))


def _rate_test(x, exposure=1., r=1., alternative="two.sided", conf_level=.95,
               correct=True):
    from .rates import _positive, _rate_inputs
    from .risk_ratio import _alternative
    counts, exposure = _rate_inputs(x, exposure)
    _probability(conf_level, "conf_level")
    alternative = _alternative(alternative)
    correct = _flag(correct, "correct")
    if len(counts) == 1:
        r = _positive(r, "r")
        value = r*float(exposure[0])
        if not math.isfinite(value):
            raise ValueError("invalid null rate")
        threshold, criterion, p0 = 10., "null expected events", None
    else:
        p0 = float(exposure[0]/exposure.sum())
        value = float(counts.sum())*p0*(1-p0)
        threshold, criterion = 5., "n * pi0 * (1-pi0)"
    return _result("Poisson rate test", "normal", True,
                   [("normal", criterion, value, threshold, ">=",
                     value >= threshold)],
                   dict(x=counts, exposure=exposure, r=r, pi0=p0,
                        correct=correct, alternative=alternative,
                        source="Book Ch. 21" if len(counts) == 1 else
                        "Book Ch. 22"),
                   ["Book null-distribution guideline; not an accuracy "
                    "guarantee.", "This checks the test approximation, not "
                    "its accompanying interval."])
