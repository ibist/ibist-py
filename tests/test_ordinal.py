"""Independent pair, delta-method, published, and R ordinal comparisons."""

from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau, norm, pearsonr, somersd, spearmanr

from ibist import OrdinalAssociationResult, ordinal_assoc


TABLE = np.array([[12, 3, 1], [4, 9, 2], [1, 5, 13]])
MEASURES = ("gamma", "tau-b", "tau-c", "somers-c|r", "somers-r|c",
            "pearson", "spearman")


def observations(table):
    return np.repeat(np.array(list(np.ndindex(table.shape))), table.flat,
                     axis=0).T


def estimands(prob):
    """Direct probability definitions, separate from the package engine."""
    row, col = prob.sum(axis=1), prob.sum(axis=0)
    concordant = discordant = 0.0
    for i, j in np.ndindex(prob.shape):
        for k, l in np.ndindex(prob.shape):
            if i < k and j < l:
                concordant += prob[i, j] * prob[k, l]
            elif i < k and j > l:
                discordant += prob[i, j] * prob[k, l]
    s = concordant - discordant
    ur = sum(a * b for a, b in combinations(row, 2))
    uc = sum(a * b for a, b in combinations(col, 2))
    d = min(prob.shape)

    def score_moments(rows, cols):
        rows = rows - row @ rows
        cols = cols - col @ cols
        covariance = sum(prob[i, j] * rows[i] * cols[j]
                         for i, j in np.ndindex(prob.shape))
        scale = math.sqrt((row @ rows**2) * (col @ cols**2))
        return covariance / scale, covariance, scale

    pearson = score_moments(np.arange(len(row)), np.arange(len(col)))[0]
    spearman, rank_cov, rank_scale = score_moments(
        row.cumsum() - row / 2, col.cumsum() - col / 2)
    return np.array([s / (concordant + discordant), s / math.sqrt(ur * uc),
                     2 * d * s / (d - 1), s / ur, s / uc, pearson, spearman,
                     rank_cov, rank_scale])


@pytest.mark.parametrize("seed", range(10))
def test_estimates_and_shared_test_match_expanded_pairs(seed):
    table = np.random.default_rng(seed).integers(0, 5, size=(3, 4))
    result = ordinal_assoc(table)
    x, y = observations(table)
    signs = [np.sign((int(x[i]) - int(x[j])) * (int(y[i]) - int(y[j])))
             for i, j in combinations(range(len(x)), 2)]
    c, d = signs.count(1), signs.count(-1)
    assert result.concordant == c
    assert result.discordant == d
    assert result.S == c - d
    assert result.n == len(x)
    assert result.estimate["gamma"] == pytest.approx((c - d) / (c + d))
    assert result.estimate["tau-b"] == pytest.approx(kendalltau(x, y).statistic)
    assert result.estimate["tau-c"] == pytest.approx(
        kendalltau(x, y, variant="c").statistic)
    assert result.estimate["pearson"] == pytest.approx(pearsonr(x, y).statistic)
    assert result.estimate["spearman"] == pytest.approx(
        spearmanr(x, y).statistic)
    assert result.estimate["somers-c|r"] == pytest.approx(
        somersd(table).statistic)
    assert result.estimate["somers-r|c"] == pytest.approx(
        somersd(table.T).statistic)
    contrast = np.array([sum(np.sign((int(x[i]) - int(x[j])) *
                                     (int(y[i]) - int(y[j])))
                              for j in range(len(x)))
                         for i in range(len(x))])
    variance = np.sum((contrast - contrast.mean())**2)
    assert result.var_S == pytest.approx(variance)
    assert result.statistic.iloc[:5].to_numpy() == pytest.approx(
        np.full(5, (c - d) / math.sqrt(variance)))
    assert result.p_value.iloc[:5].to_numpy() == pytest.approx(
        np.full(5, 2 * norm.sf(abs((c - d) / math.sqrt(variance)))))


@pytest.mark.parametrize("seed", range(6))
def test_standard_errors_match_independent_multinomial_delta_method(seed):
    table = np.random.default_rng(seed).integers(2, 30, size=(3, 4))
    result = ordinal_assoc(table)
    n = table.sum()
    p = table.astype(float).ravel() / n
    gradient = np.zeros((len(p), 9))
    step = 1e-6
    for i in range(len(p) - 1):
        direction = np.zeros(len(p))
        direction[i], direction[-1] = step, -step
        difference = (estimands((p + direction).reshape(table.shape)) -
                      estimands((p - direction).reshape(table.shape)))
        gradient[i] = difference / (2 * step)
    covariance = np.diag(p) - np.outer(p, p)
    variances = np.diag(gradient.T @ covariance @ gradient) / n
    assert result.estimate.to_numpy() == pytest.approx(
        estimands(table / n)[:7], rel=1e-10, abs=1e-14)
    assert result.std_error.to_numpy()**2 == pytest.approx(
        variances[:7], rel=2e-7, abs=1e-12)
    # The Spearman null uses the rank-covariance derivative, not the full
    # correlation derivative; marginal ranks are estimated, not fixed.
    expected_null = variances[7] / estimands(table / n)[8]**2
    assert result.statistic["spearman"] == pytest.approx(
        result.estimate["spearman"] / math.sqrt(expected_null), rel=2e-7)
    x, y = observations(table)
    product = (x - x.mean()) * (y - y.mean())
    null = product.var() / (n * x.var() * y.var())
    assert result.statistic["pearson"] == pytest.approx(
        result.estimate["pearson"] / math.sqrt(null))


def test_primary_sas_pain_example_matches_all_seven_measures():
    # SAS Base 9.4, Example 3.8, Output 3.8.3 (primary source).
    table = [[26, 26, 23, 18, 9], [6, 7, 9, 14, 23]]
    result = ordinal_assoc(table)
    expected = np.array([
        [0.5313, 0.0935, 0.3480, 0.7146],
        [0.3373, 0.0642, 0.2114, 0.4631],
        [0.4111, 0.0798, 0.2547, 0.5675],
        [0.4427, 0.0837, 0.2786, 0.6068],
        [0.2569, 0.0499, 0.1592, 0.3547],
        [0.3776, 0.0714, 0.2378, 0.5175],
        [0.3771, 0.0718, 0.2363, 0.5178]])
    actual = result.to_frame()[["estimate", "std_error", "lower", "upper"]]
    np.testing.assert_array_equal(actual.to_numpy().round(4), expected)
    assert round(result.statistic["somers-r|c"], 4) == 5.1511


@pytest.mark.parametrize("alternative", ["two.sided", "greater", "less"])
def test_intervals_use_estimation_variances_and_requested_tail(alternative):
    result = ordinal_assoc(TABLE, conf_level=0.9, alternative=alternative)
    z = norm.isf(0.05 if alternative == "two.sided" else 0.1)
    lower = result.estimate - z * result.std_error
    upper = result.estimate + z * result.std_error
    if alternative == "greater":
        lower, upper = np.clip(lower, -1, 1), np.ones(7)
    elif alternative == "less":
        lower, upper = -np.ones(7), np.clip(upper, -1, 1)
    assert result.conf_int["lower"].to_numpy() == pytest.approx(lower)
    assert result.conf_int["upper"].to_numpy() == pytest.approx(upper)
    expected = (2 * norm.sf(abs(result.statistic)) if alternative == "two.sided"
                else norm.sf(result.statistic) if alternative == "greater"
                else norm.cdf(result.statistic))
    assert result.p_value.to_numpy() == pytest.approx(expected, abs=0)


def test_transposition_reversal_and_uniform_replication():
    result = ordinal_assoc(TABLE)
    transposed = ordinal_assoc(TABLE.T)
    order = [0, 1, 2, 4, 3, 5, 6]
    assert transposed.to_frame().to_numpy() == pytest.approx(
        result.to_frame().iloc[order].to_numpy())
    reversed_rows = ordinal_assoc(TABLE[::-1])
    assert reversed_rows.estimate.to_numpy() == pytest.approx(
        -result.estimate.to_numpy())
    assert reversed_rows.std_error.to_numpy() == pytest.approx(
        result.std_error.to_numpy())
    assert reversed_rows.p_value.to_numpy() == pytest.approx(
        result.p_value.to_numpy(), abs=0)
    scaled = ordinal_assoc(TABLE * 4)
    assert scaled.estimate.to_numpy() == pytest.approx(result.estimate)
    assert scaled.std_error.to_numpy() == pytest.approx(result.std_error / 2)
    assert scaled.var_S == pytest.approx(result.var_S * 64)
    assert scaled.concordant == result.concordant * 16


def test_scores_labels_and_empty_margins_follow_r():
    table = pd.DataFrame([[8, 1, 4], [2, 5, 9]],
                         index=["1", "2"], columns=["1", "2", "10"])
    default = ordinal_assoc(table)
    explicit = ordinal_assoc(table.to_numpy(), row_scores=[1, 2],
                              col_scores=[1, 2, 10])
    assert default.to_frame().to_numpy() == pytest.approx(
        explicit.to_frame().to_numpy())
    ordinal = ordinal_assoc(table.to_numpy())
    assert default.estimate["pearson"] != pytest.approx(
        ordinal.estimate["pearson"])
    assert default.estimate["spearman"] == pytest.approx(
        ordinal.estimate["spearman"])
    named = table.copy()
    named.columns = ["low", "medium", "high"]
    assert ordinal_assoc(named).to_frame().to_numpy() == pytest.approx(
        ordinal.to_frame().to_numpy())
    expanded = np.pad(TABLE, ((1, 1), (1, 1)))
    assert ordinal_assoc(expanded).to_frame().to_numpy() == pytest.approx(
        ordinal_assoc(TABLE).to_frame().to_numpy())
    # Validate scores before dropping empty rows and columns, then subset.
    scored = ordinal_assoc(expanded, row_scores=[0, 1, 2, 5, 9],
                            col_scores=[0, 2, 3, 10, 12])
    retained = ordinal_assoc(TABLE, row_scores=[1, 2, 5], col_scores=[2, 3, 10])
    assert scored.to_frame().to_numpy() == pytest.approx(
        retained.to_frame().to_numpy())


def test_duplicate_requests_and_result_views_preserve_precision():
    result = ordinal_assoc(TABLE, measure=["spearman", "gamma", "spearman"])
    assert isinstance(result, OrdinalAssociationResult)
    assert result.measures == ("spearman", "gamma", "spearman")
    assert list(result.conf_int.columns) == ["lower", "upper"]
    before = result.to_frame()
    frame = result.to_frame()
    frame.iloc[0, 0] = -999
    pd.testing.assert_frame_equal(result.to_frame(), before)
    assert str(result) == repr(result)
    result.summary(digits=1)
    pd.testing.assert_frame_equal(result.to_frame(), before)
    assert "share the same test" in ordinal_assoc(TABLE).summary()
    with pytest.raises(ValueError, match="digits"):
        result.summary(digits=-1)


def test_degenerate_null_variances_produce_missing_tests():
    result = ordinal_assoc([[4, 0], [0, 4]])
    assert np.all(result.estimate == 1)
    assert np.allclose(result.std_error, 0, atol=1e-14)
    assert result.var_S == 0
    assert result.statistic.isna().all()
    assert result.p_value.isna().all()


def test_small_p_values_are_not_lost_to_cdf_subtraction_or_display():
    result = ordinal_assoc(TABLE * 10)
    assert 0 < result.p_value["gamma"] < 1e-100
    assert "e-" in result.summary()


def test_large_counts_retain_exact_pair_counts_without_expanding_samples():
    count = 10**12
    result = ordinal_assoc(TABLE * count)
    reference = ordinal_assoc(TABLE)
    assert isinstance(result.concordant, int)
    assert result.concordant == reference.concordant * count**2
    assert result.discordant == reference.discordant * count**2
    assert result.S == reference.S * count**2
    assert result.estimate.to_numpy() == pytest.approx(reference.estimate)
    assert result.std_error.to_numpy() == pytest.approx(
        reference.std_error / math.sqrt(count))


@pytest.mark.parametrize("scale", [1e-250, 1e250])
def test_score_rescaling_preserves_estimates_and_inference(scale):
    result = ordinal_assoc(TABLE, row_scores=np.array([-1., 0, 2]) * scale,
                            col_scores=np.array([-2., 0, 1]) * scale)
    reference = ordinal_assoc(TABLE, row_scores=[-1, 0, 2],
                               col_scores=[-2, 0, 1])
    assert result.to_frame().to_numpy() == pytest.approx(
        reference.to_frame().to_numpy())


@pytest.mark.parametrize("x", [[], [[0, 0], [0, 0]], [[1, 2], [0, 0]],
                             [[True, False], [False, True]],
                             [[1, -1], [2, 3]], [[1.5, 0], [2, 3]],
                             [[np.nan, 0], [2, 3]], np.ones((2, 2, 2)),
                             [[2**53, 0], [0, 1]]])
def test_invalid_tables_are_rejected(x):
    with pytest.raises(ValueError):
        ordinal_assoc(x)


@pytest.mark.parametrize("options", [
    {"measure": []}, {"measure": None}, {"measure": "unknown"},
    {"measure": [True]}, {"alternative": "lower"},
    {"alternative": ["less"]}, {"conf_level": 1}, {"conf_level": True},
    {"row_scores": [1]}, {"row_scores": [1, 1, 1]},
    {"col_scores": [1, np.inf, 3]}, {"row_scores": ["1", "2", "3"]},
    {"col_scores": [[1, 2, 3]]}, {"row_scores": [True, False, True]}])
def test_invalid_options_are_rejected(options):
    with pytest.raises(ValueError):
        ordinal_assoc(TABLE, **options)


def test_scores_that_only_vary_in_dropped_categories_are_rejected():
    table = [[1, 2], [0, 0], [2, 3]]
    with pytest.raises(ValueError, match="variance"):
        ordinal_assoc(table, row_scores=[1, 2, 1], measure="tau-b")


CASES = json.loads((Path(__file__).parent / "reference" /
                    "ordinal_r.json").read_text())["cases"]


def numeric(values):
    return np.array([math.nan if v == "NA" else float(v) for v in values])


@pytest.mark.parametrize("case", CASES)
def test_r_references_match_estimates_intervals_and_tests(case):
    x = case["x"]
    if "row_labels" in case or "col_labels" in case:
        x = pd.DataFrame(x, index=case.get("row_labels"),
                         columns=case.get("col_labels"))
    options = {name: case[name] for name in
               ["row_scores", "col_scores", "measure"] if name in case}
    result = ordinal_assoc(x, conf_level=case["conf_level"],
                            alternative=case["alternative"], **options)
    assert result.measures == tuple(case["measures"])
    for name in ["estimate", "std_error", "statistic", "p_value"]:
        actual = getattr(result, name).to_numpy()
        expected = numeric(case[name])
        if name == "std_error":
            # R subtracts near-equal terms at perfect-association boundaries.
            boundary = expected < 1e-7
            np.testing.assert_allclose(actual[boundary], expected[boundary],
                                       atol=1e-8, rtol=0, equal_nan=True)
            actual, expected = actual[~boundary], expected[~boundary]
        tolerance = 0 if name == "p_value" else 1e-11
        np.testing.assert_allclose(actual, expected, rtol=2e-9,
                                   atol=tolerance, equal_nan=True)
    for name in ["lower", "upper"]:
        tolerance = 3e-8 if np.any(numeric(case["std_error"]) < 1e-7) else 1e-11
        np.testing.assert_allclose(result.conf_int[name], numeric(case[name]),
                                   rtol=2e-9, atol=tolerance, equal_nan=True)
    for name in ["S", "var_S", "concordant", "discordant", "n"]:
        assert getattr(result, name) == pytest.approx(case[name], rel=1e-12,
                                                     abs=1e-10)
    assert result.method == case["method"]
