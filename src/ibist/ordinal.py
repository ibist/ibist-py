"""Ordinal association and multinomial asymptotic inference for count tables."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from .contingency import _digits, _table
from .odds_ratio import _probability


_MEASURES = ("gamma", "tau-b", "tau-c", "somers-c|r", "somers-r|c",
             "pearson", "spearman")


@dataclass(frozen=True, repr=False)
class OrdinalAssociationResult:
    """Estimates, confidence limits, and tests for ordinal association.

    Attributes
    ----------
    estimate, std_error, statistic, p_value : pandas.Series
        Full-precision results indexed by selected measure, including any
        repeated requests. Statistics test zero association.
    conf_int : pandas.DataFrame
        Lower and upper confidence limits indexed by measure.
    conf_level : float
        Confidence level used for all intervals.
    alternative : str
        Alternative hypothesis, also determining interval direction.
    S, var_S : int, float
        Concordance contrast and its multinomial null variance.
    concordant, discordant, n : int
        Unordered concordant/discordant pair counts and total sample size.
    method : str
        Statistical procedure description.

    Notes
    -----
    This corresponds to R's ibist_ordinal_assoc. Dotted fields use underscores
    in Python. measures reports the selected names; to_frame() returns a new
    table. Formatting changes only displayed values.
    """

    estimate: pd.Series
    std_error: pd.Series
    conf_int: pd.DataFrame
    statistic: pd.Series
    p_value: pd.Series
    conf_level: float
    alternative: str
    S: int
    var_S: float
    concordant: int
    discordant: int
    n: int
    method: str = "Asymptotic ordinal association measures and tests"

    @property
    def measures(self):
        """Selected measure names, preserving order and repetitions."""
        return tuple(self.estimate.index)

    def to_frame(self):
        """Return a fresh table of estimates, intervals, and test results."""
        return pd.DataFrame({
            "estimate": self.estimate.to_numpy(copy=True),
            "std_error": self.std_error.to_numpy(copy=True),
            "lower": self.conf_int["lower"].to_numpy(copy=True),
            "upper": self.conf_int["upper"].to_numpy(copy=True),
            "statistic": self.statistic.to_numpy(copy=True),
            "p_value": self.p_value.to_numpy(copy=True)},
            index=self.estimate.index.copy())

    def summary(self, digits=3):
        """Format results while retaining precision and tiny p-values."""
        _digits(digits)
        frame = self.to_frame()
        text = frame.to_string(
            float_format=lambda value: f"{value:.{digits}f}",
            formatters={"p_value": lambda value:
                        f"{value:.{max(1, digits)}g}"})
        note = ("\nConcordance-based measures share the same test."
                if sum(name in _MEASURES[:5] for name in self.measures) > 1
                else "")
        return (f"{self.method}\nConfidence level: {100 * self.conf_level:g}%"
                f"\nAlternative: {self.alternative}\n\n{text}{note}")

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.summary()


def ordinal_assoc(x, *, measure=_MEASURES, conf_level=0.95,
                  alternative="two.sided", row_scores=None, col_scores=None):
    """Estimate ordinal associations with asymptotic intervals and tests.

    Parameters
    ----------
    x : array_like or pandas.DataFrame
        Two-dimensional nonnegative integer count table, with rows and
        columns already ordered from lowest to highest category. Empty
        margins are dropped; at least two nonempty rows and columns must
        remain. Counts and their total must not exceed 2**53-1.
    measure : str or sequence of str, default all seven measures
        "gamma", "tau-b", "tau-c", "somers-c|r", "somers-r|c", "pearson",
        or "spearman". Requested order and repetitions are retained, as in R.
    conf_level : float, default 0.95
        Confidence level strictly between zero and one.
    alternative : {"two.sided", "greater", "less"}, default "two.sided"
        Alternative for tests of zero association. Also determines interval
        direction. Positive association links higher rows to higher columns.
    row_scores, col_scores : one-dimensional array_like, optional
        Finite numeric category scores for Pearson correlation, each with
        at least two distinct values. Either may be supplied independently;
        lengths must match the original table before empty margins drop.
        Defaults use numeric DataFrame labels if all labels on that axis
        are numeric; otherwise use 1, ..., number of retained categories.
        Array input uses category positions. Spearman always uses midranks.

    Returns
    -------
    OrdinalAssociationResult
        Named estimate, std_error, statistic, and p_value Series, and a
        conf_int DataFrame with lower/upper columns. Includes conf_level,
        alternative, method, measures, S, var_S, concordant, discordant,
        and n. S is the difference between unordered pair counts; var_S
        is its null variance. to_frame() returns a fresh combined table.

    Raises
    ------
    ValueError
        If counts, measures, scores, or options are invalid, fewer than two
        nonempty categories remain on either axis, or retained scores have
        zero marginal variance. As in R, score validation applies even when
        Pearson is not selected.

    See Also
    --------
    nominal_assoc : Descriptive measures ignoring category ordering.
    indep_test : Tests of row/column independence.

    Notes
    -----
    Definitions follow R ordinal.assoc(). Gamma excludes ties on either
    variable; tau-b adjusts for row and column ties, and tau-c adjusts for
    table dimensions. Somers C|R treats rows as explanatory and columns as
    response; R|C reverses these roles. Category order is never inferred by
    sorting labels or scores. Custom scores affect Pearson only.

    Let Aij and Dij count observations concordant and discordant with cell
    (i,j), and S=C-D. The five concordance-based tests share the signed
    statistic S/sqrt(var_S), where var_S=sum(nij*(Aij-Dij)**2)-4*S**2/n.
    Pearson and Spearman use distinct multinomial null variances; these are
    not the usual continuous-normal correlation test or the CMH
    linear-by-linear test. Spearman uses marginal midranks and its variance
    accounts for estimating those ranks from the margins.

    Intervals use measure-specific multinomial estimation variances, which
    differ from null variances. Two-sided intervals are untruncated Wald
    intervals and can exceed [-1, 1]. One-sided finite bounds are clipped
    to [-1, 1], with the other endpoint at 1 (greater) or -1 (less).
    Thus intervals need not invert their tests. A nonpositive null variance
    gives nan statistics and p-values, rather than a fabricated test.
    All calculations retain full precision; only display values are rounded.

    References
    ----------
    Brown and Benedetti (1977), JASA 72(358), 309--315,
    doi:10.1080/01621459.1977.10480995.
    Goodman and Kruskal (1954), JASA 49(268), 732--764.
    Kendall (1970), Rank Correlation Methods, 4th ed., Griffin.
    Somers (1962), American Sociological Review 27(6), 799--811.
    Stuart (1953), Biometrika 40(1--2), 105--110.
    SAS Institute, Base SAS 9.4 Procedures Guide: Statistical Procedures,
    Measures of Association (multinomial variances and null tests).

    Examples
    --------
    >>> table = [[12, 3, 1], [4, 9, 2], [1, 5, 13]]
    >>> result = ordinal_assoc(table, measure=["tau-b", "spearman"])
    >>> result.measures
    ('tau-b', 'spearman')
    >>> result.conf_int.shape
    (2, 2)
    """
    conf_level = _probability(conf_level, "conf_level")
    if not isinstance(alternative, str) or alternative not in (
            "two.sided", "greater", "less"):
        raise ValueError("alternative must be two.sided, greater, or less.")
    try:
        measures = [measure] if isinstance(measure, str) else list(measure)
    except TypeError:
        raise ValueError("measure must contain valid measure names.") from None
    if not measures or any(not isinstance(name, str) or name not in _MEASURES
                           for name in measures):
        raise ValueError(f"measure must contain one or more of {_MEASURES}.")
    counts, frame = _table(x, positive_margins=False)
    keep_rows, keep_cols = counts.sum(axis=1) > 0, counts.sum(axis=0) > 0
    rows = _scores(row_scores, counts.shape[0], "row_scores")
    cols = _scores(col_scores, counts.shape[1], "col_scores")
    counts = counts[np.ix_(keep_rows, keep_cols)]
    if min(counts.shape) < 2:
        raise ValueError("x must have two nonempty rows and columns.")
    rows = (rows[keep_rows] if rows is not None else _default_scores(
        frame.index[keep_rows] if isinstance(x, pd.DataFrame) else None,
        counts.shape[0]))
    cols = (cols[keep_cols] if cols is not None else _default_scores(
        frame.columns[keep_cols] if isinstance(x, pd.DataFrame) else None,
        counts.shape[1]))
    n = int(counts.sum())
    prob = counts / n
    A, D = _quadrants(counts)
    concordant = sum(int(v) * int(a) for v, a in zip(counts.flat, A.flat)) // 2
    discordant = sum(int(v) * int(d) for v, d in zip(counts.flat, D.flat)) // 2
    S = concordant - discordant
    contrast = (A - D).astype(float)
    var_S = float(np.sum(counts * (contrast - 2 * S / n)**2))
    r, c = counts.sum(axis=1), counts.sum(axis=0)
    ur = sum(int(v) * (n - int(v)) for v in r) / (2 * n * n)
    uc = sum(int(v) * (n - int(v)) for v in c) / (2 * n * n)
    u = math.sqrt(ur * uc)
    s, pair_mass = S / (n * n), (concordant + discordant) / (n * n)
    delta = contrast / n
    d = min(counts.shape)
    estimates = [s / pair_mass if pair_mass else math.nan, s / u,
                 2 * d * s / (d - 1), s / ur, s / uc]
    rprob, cprob = r / n, c / n
    gradients = [
        (delta * pair_mass - s * (A + D) / n) / pair_mass**2 if pair_mass else
        np.full(counts.shape, np.nan),
        delta / u + estimates[1] * (
            rprob[:, None] / ur + cprob[None, :] / uc) / 2,
        2 * d * delta / (d - 1),
        delta / ur + s * rprob[:, None] / ur**2,
        delta / uc + s * cprob[None, :] / uc**2]
    variances = [_variance(prob, value) / n for value in gradients]
    pearson = _score_correlation(prob, rows, cols, n)
    spearman = _spearman(prob, counts, n)
    estimates.extend([pearson[0], spearman[0]])
    variances.extend([pearson[1], spearman[1]])
    shared_z = S / math.sqrt(var_S) if var_S > 0 else math.nan
    statistics = [shared_z] * 5 + [
        estimate / math.sqrt(null) if null > 0 else math.nan
        for estimate, _, null in (pearson, spearman)]
    select = [_MEASURES.index(name) for name in measures]
    estimate = np.array(estimates)[select]
    std_error = np.sqrt(np.array(variances)[select])
    statistic = np.array(statistics)[select]
    tail = (1 - conf_level) / (2 if alternative == "two.sided" else 1)
    z = norm.isf(tail)
    limits = np.column_stack((estimate - z * std_error,
                              estimate + z * std_error))
    usable = np.isfinite(estimate) & np.isfinite(std_error)
    if alternative == "greater":
        limits[usable, 0] = np.clip(limits[usable, 0], -1, 1)
        limits[usable, 1] = 1
    elif alternative == "less":
        limits[usable, 0] = -1
        limits[usable, 1] = np.clip(limits[usable, 1], -1, 1)
    p_value = (2 * norm.sf(abs(statistic)) if alternative == "two.sided" else
               norm.sf(statistic) if alternative == "greater" else
               norm.cdf(statistic))
    index = pd.Index(measures, name="measure")
    return OrdinalAssociationResult(
        pd.Series(estimate, index=index, name="estimate"),
        pd.Series(std_error, index=index, name="std_error"),
        pd.DataFrame(limits, index=index, columns=["lower", "upper"]),
        pd.Series(statistic, index=index, name="statistic"),
        pd.Series(p_value, index=index, name="p_value"), conf_level,
        alternative, S, var_S, concordant, discordant, n)


def _scores(scores, size, name):
    if scores is None:
        return None
    try:
        values = np.asarray(scores)
    except (TypeError, ValueError):
        raise ValueError(f"{name} needs finite numeric scores.") from None
    if (values.ndim != 1 or len(values) != size
            or values.dtype.kind not in "iuf"
            or not np.all(np.isfinite(values)) or len(np.unique(values)) < 2):
        raise ValueError(f"{name} must contain {size} finite numeric scores "
                         "with at least two distinct values.")
    return values.astype(float)


def _default_scores(labels, size):
    if labels is not None:
        try:
            scores = np.array([float(str(label)) for label in labels])
            if np.all(np.isfinite(scores)):
                return scores
        except (TypeError, ValueError, OverflowError):
            pass
    return np.arange(1, size + 1, dtype=float)


def _quadrants(counts):
    prefix = np.pad(counts.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    total = int(prefix[-1, -1])
    A = (prefix[:-1, :-1] + total - prefix[1:, -1, None]
         - prefix[None, -1, 1:] + prefix[1:, 1:])
    D = (prefix[:-1, -1, None] - prefix[:-1, 1:]
         + prefix[None, -1, :-1] - prefix[1:, :-1])
    return A, D


def _variance(prob, values):
    centered = values - np.sum(prob * values)
    return float(np.sum(prob * centered**2))


def _score_correlation(prob, rows, cols, n):
    # Rescaling scores avoids overflow without changing the correlation.
    row_scale, col_scale = np.max(abs(rows)), np.max(abs(cols))
    if row_scale == 0 or col_scale == 0:
        raise ValueError("Category scores need positive marginal variance.")
    rows, cols = rows / row_scale, cols / col_scale
    rows = rows - prob.sum(axis=1) @ rows
    cols = cols - prob.sum(axis=0) @ cols
    row_var = float(prob.sum(axis=1) @ rows**2)
    col_var = float(prob.sum(axis=0) @ cols**2)
    if row_var <= 0 or col_var <= 0:
        raise ValueError("Category scores need positive marginal variance.")
    scale = math.sqrt(row_var * col_var)
    cross = np.outer(rows, cols)
    estimate = float(np.sum(prob * cross) / scale)
    influence = cross / scale - estimate * (
        rows[:, None]**2 / row_var + cols[None, :]**2 / col_var) / 2
    return (estimate, _variance(prob, influence) / n,
            _variance(prob, cross) / (n * row_var * col_var))


def _spearman(prob, counts, n):
    r, c = counts.sum(axis=1), counts.sum(axis=0)
    rows = (r.cumsum() - r / 2) / n - 0.5
    cols = (c.cumsum() - c / 2) / n - 0.5
    F = (n**3 - sum(int(v)**3 for v in r)) / n**3
    G = (n**3 - sum(int(v)**3 for v in c)) / n**3
    scale = math.sqrt(F * G) / 12
    cross = np.outer(rows, cols)
    v = float(np.sum(prob * cross))
    row_weight, col_weight = prob @ cols, rows @ prob
    row_tail = np.cumsum(row_weight[::-1])[::-1] - row_weight
    col_tail = np.cumsum(col_weight[::-1])[::-1] - col_weight
    vij = (cross + row_weight[:, None] / 2 + col_weight[None, :] / 2
           + row_tail[:, None] + col_tail[None, :])
    wij = -(G * (r / n)[:, None]**2 + F * (c / n)[None, :]**2) / (96 * scale)
    influence = vij / scale - v * wij / scale**2
    return (v / scale, _variance(prob, influence) / n,
            _variance(prob, vij) / (n * scale**2))
