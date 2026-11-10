"""Family-specific goodness-of-fit tests for fitted statsmodels GLMs."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
import warnings

import numpy as np
import pandas as pd
from scipy.special import logit, xlogy
from scipy.stats import chi2

from .asymcheck import _result
from .contingency import TestResult, _digits, _flag


@dataclass(frozen=True, repr=False)
class GLMGOFResults(Sequence):
    """Ordered TestResult collection, corresponding to ibist_glm_gof_tests.

    results retains requested order and repetitions. methods identifies the
    procedures. to_frame() returns a fresh table; p-values are unadjusted.
    """

    results: tuple[TestResult, ...]

    def __len__(self):
        return len(self.results)

    def __getitem__(self, item):
        return self.results[item]

    @property
    def methods(self):
        return tuple(result.test_method for result in self.results)

    def to_frame(self):
        """Fresh full-precision table with one row per requested method."""
        return pd.DataFrame([(r.statistic, r.df, r.p_value) for r in self],
                            index=pd.Index(self.methods, name="method"),
                            columns=("statistic", "df", "p_value"))

    def summary(self, digits=4):
        """Format a combined table without changing stored precision."""
        _digits(digits)
        return (self.to_frame().to_string(float_format=lambda x:
                                         f"{x:.{digits}g}") +
                "\nTests use the same fit; p-values are unadjusted.")

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.summary()


def glm_gof(fit, *, method="pearson", g=10, asymcheck=False):
    """Test goodness of fit of a fitted binomial or Poisson GLM.

    Parameters
    ----------
    fit : fitted statsmodels GLM result
        Converged, unpenalized Binomial or Poisson model with dispersion one.
        Binomial endog may be binary, proportions with integer trial weights,
        or success/failure columns. Frequency weights must be integer counts.
        Analytic/survey weights, constrained fits, and other families are
        unsupported. Poisson variance weights must all be one.
    method : str or sequence of str, default "pearson"
        pearson, deviance, hosmer-lemeshow, osius-rojek, or stukel. Last three
        require a Binomial logit model. Order and repetitions are retained.
    g : int, default 10
        Requested Hosmer-Lemeshow groups, >=3. Probability ties stay together;
        actual groups may be fewer. Ties use 14 significant digits.
        Nondefault g requires that method.
    asymcheck : bool, default False
        Attach method-specific approximation diagnostics. All methods here
        are asymptotic. Adequacy is unassessed pending stabilized R checks.

    Returns
    -------
    TestResult or GLMGOFResults
        One method returns statistic, df, p_value, family, link, test_method,
        and copied groups. Multiple methods return an ordered collection.
        Diagnostics needed for post-checks are retained regardless of flag.
        Results do not retain the fitted model. P-values are unadjusted.

    Raises
    ------
    ValueError
        For unsupported fits, invalid counts/settings, boundary fitted means,
        nonconvergence, nonpositive df, or singular test standardization.

    Warns
    -----
    RuntimeWarning
        If a stated expected-count approximation criterion is not met.

    See Also
    --------
    asymcheck : Standalone pre- and post-inference approximation diagnostics.

    Notes
    -----
    Matches R glm.gof(). Pearson and deviance aggregate identical design rows
    AND offsets, preserving fitted expected totals, and use J-rank df. There
    is no automatic pooling. For individual Poisson outcomes, aggregation
    checks grouped means rather than all aspects of individual dispersion.

    Hosmer-Lemeshow uses SAS's target-frequency grouping, always combining
    probability ties, with actual group count minus two df. An intercept is
    required. Quantile-based implementations can produce different groups.

    Osius-Rojek uses the SAS variant centered at J, not J-rank. Its variance
    is 2*sum(1-1/n) plus squared residual norm after projecting
    (1-2*p)/sqrt(n*p*(1-p)) onto the weighted model space. The returned
    statistic is the squared standardized statistic, on one df.

    Stukel tests eta**2 times the positive/negative eta indicators using an
    efficient score test, with offsets included in eta. Nuisance projection
    removes original model terms. df is the estimable increment, one or two;
    zero increment is an error. No augmented fit is run. Column-scaled SVD
    projections use relative tolerance 1e-10. Projection noise below
    1e-10*max(original column norm, 1) is discarded; standardizing variance
    <=1e-12 is numerically degenerate. Diagnostics report condition_met=None
    for all methods until the R checks stabilize. Inference validation
    remains in place.

    References
    ----------
    SAS Institute. SAS/STAT User's Guide, LOGISTIC Procedure, Goodness-of-Fit
    Tests, chapter 79, pp. 6050-6053. https://support.sas.com/documentation/
    Hosmer and Lemeshow (1980). Communications in Statistics A9, 1043-1069.
    Osius and Rojek (1992). JASA 87, 1145-1152.
    Stukel (1988). JASA 83, 426-431.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> y = np.array([0, 0, 0, 7, 6, 17])
    >>> n = np.array([14, 20, 19, 18, 12, 17])
    >>> x = sm.add_constant(np.arange(25, 76, 10))
    >>> fit = sm.GLM(np.column_stack((y, n-y)), x,
    ...              family=sm.families.Binomial()).fit()
    >>> result = glm_gof(fit, method="osius-rojek")
    >>> result.df
    1
    """
    asymcheck = _flag(asymcheck, "asymcheck")
    data = _data(fit)
    methods = _methods(method, g, data)
    results = []
    for name in methods:
        parts = _parts(data, name, g)
        check = _check(data, name, parts)
        if check.condition_met is False:
            warnings.warn(name + ": approximation criterion not met; inspect "
                          "diagnostics", RuntimeWarning, stacklevel=2)
        statistic = _stat(data, name, parts)
        groups = parts["groups"]
        observed = groups[["observed"]].copy()
        expected = groups[["expected"]].copy()
        results.append(TestResult(
            statistic, float(chi2.sf(statistic, parts["df"])),
            f"GLM {name} goodness-of-fit test", df=parts["df"],
            observed=observed, expected=expected, test_method=name,
            family=data["family"], link=data["link"],
            diagnostics=check if asymcheck else None, _gof_diagnostics=check))
    return results[0] if len(results) == 1 else GLMGOFResults(tuple(results))


def _basis(x):
    if not x.shape[1]:
        return np.zeros((len(x), 0))
    scale = np.sqrt(np.sum(x*x, axis=0))
    x = x[:, scale > 0]
    if not x.shape[1]:
        return np.zeros((len(x), 0))
    u, singular, _ = np.linalg.svd(x/scale[scale > 0], full_matrices=False)
    return u[:, singular > 1e-10*singular.max()]


def _data(fit):
    from statsmodels.genmod.generalized_linear_model import (
        GLMResults, GLMResultsWrapper)
    from statsmodels.genmod.families import Binomial, Poisson
    if (not isinstance(fit, (GLMResults, GLMResultsWrapper)) or
            not getattr(fit, "converged", False)):
        raise ValueError("fit must be a converged statsmodels GLM result.")
    if hasattr(fit, "constraints"):
        raise ValueError("Constrained fits are unsupported.")
    model = fit.model
    fam = model.family
    if (type(fam) not in (Binomial, Poisson) or
            not np.isclose(fit.scale, 1, rtol=1e-12, atol=0)):
        raise ValueError("Only Binomial/Poisson GLMs with dispersion one "
                         "are supported.")
    family = "binomial" if type(fam) is Binomial else "poisson"
    link = fam.link.__class__.__name__.lower()
    if model.exog is None or model.endog is None:
        raise ValueError("fit must retain its design and response.")
    x = np.asarray(model.exog, dtype=float)
    y = np.asarray(model.endog, dtype=float)
    freq = np.broadcast_to(np.asarray(model.freq_weights, float), y.shape)
    var = np.broadcast_to(np.asarray(model.var_weights, float), y.shape)
    trials = np.broadcast_to(np.asarray(model.n_trials, float), y.shape)
    if np.any(freq < 0) or np.any(np.abs(freq-np.round(freq)) > 1e-7):
        raise ValueError("Frequency weights must be integer counts.")
    if family == "poisson" and np.any(var != 1):
        raise ValueError("Poisson analytic variance weights are unsupported.")
    mu = np.asarray(fit.fittedvalues, dtype=float)
    offset = np.broadcast_to(np.asarray(getattr(model, "_offset_exposure", 0),
                                        dtype=float), y.shape)
    n = freq * var * trials if family == "binomial" else freq
    observed = y*n
    counts = (np.r_[n, observed, n-observed] if family == "binomial" else
              np.r_[y, n])
    active_mu = mu[n > 0]
    if (not all(np.isfinite(a).all() for a in (x, y, n, active_mu, offset)) or
            np.any(active_mu <= 0) or
            (family == "binomial" and np.any(active_mu >= 1))):
        raise ValueError("Invalid retained inputs or boundary fitted means.")
    if (np.any(counts < 0) or np.any(np.abs(counts-np.round(counts)) > 1e-7)
            or n.sum() > 2**53-1 or observed.sum() > 2**53-1):
        raise ValueError("Weights must encode integer counts/frequencies.")
    keep = n > 0
    x, offset, n, observed = x[keep], offset[keep], n[keep], observed[keep]
    expected = n*mu[keep]
    if not len(n):
        raise ValueError("No positive-weight observations.")
    index, first, ids = {}, [], []
    for i, row in enumerate(np.column_stack((x, offset))):
        key = tuple(row)
        if key not in index:
            index[key] = len(first)
            first.append(i)
        ids.append(index[key])
    groups = len(first)
    aggregate = lambda z: np.bincount(ids, weights=z, minlength=groups)
    data = dict(family=family, link=link, x=x[first], n=aggregate(n),
                observed=aggregate(observed), expected=aggregate(expected))
    data["mu"] = data["expected"]/data["n"]
    data["variance"] = (data["expected"]*(1-data["mu"])
                        if family == "binomial" else data["expected"])
    data["q"] = _basis(np.sqrt(data["variance"])[:, None]*data["x"])
    data["rank"] = data["q"].shape[1]
    return data


def _methods(method, g, data):
    from .risk_ratio import _methods as choose
    methods = tuple(choose(method, ("pearson", "deviance", "hosmer-lemeshow",
                                    "osius-rojek", "stukel")))
    if (isinstance(g, (bool, np.bool_)) or not isinstance(g, Real) or
            not np.isfinite(g) or g < 3 or int(g) != g):
        raise ValueError("g must be an integer >=3.")
    if g != 10 and "hosmer-lemeshow" not in methods:
        raise ValueError("g applies only to Hosmer-Lemeshow.")
    if (any(m not in ("pearson", "deviance") for m in methods) and
            (data["family"] != "binomial" or data["link"] != "logit")):
        raise ValueError("Requested method requires a Binomial logit model.")
    return methods


def _hl(data, g):
    q = _basis(data["x"])
    if np.max(np.abs(1 - q@q.sum(axis=0))) > 1e-7:
        raise ValueError("Hosmer-Lemeshow requires an intercept.")
    order = np.argsort(data["mu"], kind="stable")
    p = np.array([float(format(value, ".14g")) for value in data["mu"][order]])
    _, ids = np.unique(p, return_inverse=True)
    blocks = np.column_stack([np.bincount(ids, weights=data[k][order])
                              for k in ("observed", "expected", "n")])
    target = np.floor(blocks[:, 2].sum()/g + .5)
    group, current, size = [], 0, 0
    for i, row in enumerate(blocks):
        f = row[2]
        if (i > 0 and current < g-1 and
                not (size < target and size + np.floor(f/2) <= target)):
            current += 1
            size = 0
        group.append(current)
        size += f
    group = np.asarray(group)
    if current > 0 and size < blocks[:, 2].sum()/(2*g):
        group[group == current] = current-1
    values = np.column_stack([np.bincount(group, weights=blocks[:, j])
                              for j in range(3)])
    return pd.DataFrame(values, columns=("observed", "expected", "n"))


def _parts(data, method, g, validate=True):
    groups = pd.DataFrame({k: data[k] for k in ("observed", "expected", "n")})
    parts = dict(groups=groups, df=len(groups)-data["rank"])
    if method == "hosmer-lemeshow":
        parts["groups"] = _hl(data, g)
        parts["df"] = len(parts["groups"])-2
    if method == "osius-rojek":
        v = (1-2*data["mu"])/np.sqrt(data["variance"])
        residual = v-data["q"]@(data["q"].T@v)
        parts["variance"] = float(2*np.sum(1-1/data["n"])+residual@residual)
        if validate and (not np.isfinite(parts["variance"]) or
                         parts["variance"] <= 1e-12):
            raise ValueError("Osius-Rojek variance is zero or unstable.")
        parts["df"] = 1
    if method == "stukel":
        eta = logit(data["mu"])
        z = np.column_stack((eta**2*(eta >= 0), eta**2*(eta < 0)))
        original = np.sqrt(data["variance"])[:, None]*z
        a = original-data["q"]@(data["q"].T@original)
        small = np.linalg.norm(a, axis=0) <= (
            1e-10*np.maximum(np.linalg.norm(original, axis=0), 1))
        a[:, small] = 0
        parts["q"] = _basis(a)
        parts["df"] = parts["q"].shape[1]
    if validate and parts["df"] <= 0:
        raise ValueError("No positive residual or added-term degrees of "
                         "freedom.")
    return parts


def _stat(data, method, parts):
    tab = parts["groups"]
    if method == "stukel":
        residual = (data["observed"]-data["expected"])/np.sqrt(data["variance"])
        return float(np.sum((parts["q"].T@residual)**2))
    o, e, n = (tab[k].to_numpy() for k in ("observed", "expected", "n"))
    variance = e*(1-e/n) if data["family"] == "binomial" else e
    pearson = float(np.sum((o-e)**2/variance))
    if method == "osius-rojek":
        return (pearson-len(tab))**2/parts["variance"]
    if method != "deviance":
        return pearson
    value = np.sum(xlogy(o, o/e))
    value += (np.sum(xlogy(n-o, (n-o)/(n-e))) if data["family"] == "binomial"
              else -np.sum(o-e))
    return max(0., float(2*value))


def _check(data, method, parts):
    tab = parts["groups"]
    rows = [(method, "asymptotic adequacy", None, None, None, None)]
    notes = ["Adequacy is unassessed pending stabilization of R checks.",
             "Computability does not establish approximation accuracy."]
    return _result("GLM goodness of fit", method, True, rows,
                   dict(family=data["family"], link=data["link"],
                        rank=data["rank"], converged=True,
                        boundary_fitted=False,
                        groups=tab, minimum_fitted_margin=float(np.min(
                            np.minimum(data["mu"], 1-data["mu"])
                            if data["family"] == "binomial" else data["mu"]))),
                   notes)


def _combine(checks):
    if len(checks) == 1:
        return checks[0]
    notes = tuple(dict.fromkeys(n for c in checks for n in c.notes[:-1]))
    return _result("GLM goodness of fit", tuple(c.method for c in checks), True,
                   [row for c in checks for row in c._checks],
                   dict(methods=tuple(c.details for c in checks)), notes)


def _precheck(fit, *, method="pearson", g=10, asymcheck=False):
    _flag(asymcheck, "asymcheck")
    data = _data(fit)
    return _combine([_check(data, m, _parts(data, m, g, validate=False))
                     for m in _methods(method, g, data)])
