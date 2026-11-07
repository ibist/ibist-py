import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln
from scipy.stats import chi2, chi2_contingency, fisher_exact, norm

from ibist import indep_test, nominal_assoc, or_homogeneity_test


def test_pearson_matches_scipy_and_preserves_dataframe_labels():
    tab = pd.DataFrame([[12, 8], [5, 15]], index=["A", "B"],
                       columns=["yes", "no"])
    result = indep_test(tab)
    reference = chi2_contingency(tab, correction=False)
    assert result.statistic == pytest.approx(reference.statistic)
    assert result.p_value == pytest.approx(reference.pvalue)
    assert result.df == 1
    pd.testing.assert_frame_equal(result.observed, tab)
    assert result.expected.index.equals(tab.index)
    assert result.expected.columns.equals(tab.columns)
    np.testing.assert_allclose(result.expected, reference.expected_freq)
    np.testing.assert_allclose(result.residuals,
                               (tab - result.expected) / np.sqrt(result.expected))
    rows = tab.sum(axis=1).to_numpy()[:, None] / tab.to_numpy().sum()
    cols = tab.sum(axis=0).to_numpy()[None, :] / tab.to_numpy().sum()
    np.testing.assert_allclose(result.stdres, (tab - result.expected)
                               / np.sqrt(result.expected * (1 - rows) * (1 - cols)))


def test_yates_cap_does_not_overcorrect_near_independence():
    tab = [[11, 10], [10, 10]]
    result = indep_test(tab, correct=True)
    assert result.statistic == pytest.approx(0, abs=1e-25)
    assert result.p_value == 1
    tab = [[12, 8], [5, 15]]
    ref = chi2_contingency(tab, correction=True)
    assert indep_test(tab, correct=True).statistic == pytest.approx(ref.statistic)


def test_lr_uses_zero_cell_convention_and_independent_normal_tail():
    result = indep_test([[0, 20], [10, 10]], method="lr")
    statistic = 2 * (20 * np.log(4 / 3) + 10 * np.log(2) + 10 * np.log(2 / 3))
    assert result.statistic == pytest.approx(statistic)
    assert result.p_value == pytest.approx(2 * norm.sf(np.sqrt(statistic)))
    np.testing.assert_allclose(result.expected, [[5, 15], [5, 15]])


def test_lr_matches_poisson_loglinear_deviance_and_retains_small_pvalue():
    tab = np.array([[109, 68, 14], [72, 79, 37], [9, 42, 139]])
    result = indep_test(tab, method="lr")
    expected = tab.sum(axis=1)[:, None] * tab.sum(axis=0)[None, :] / tab.sum()
    # This is the Poisson independence MLE; saturated minus fitted loglik.
    saturated = np.sum(tab * np.log(tab) - tab - gammaln(tab + 1))
    fitted = np.sum(tab * np.log(expected) - expected - gammaln(tab + 1))
    assert result.statistic == pytest.approx(2 * (saturated - fitted))
    assert result.p_value > 0
    assert result.p_value == pytest.approx(
        np.exp(-result.statistic / 2) * (1 + result.statistic / 2),
        rel=1e-12, abs=0
    )
    assert "e-" in result.summary()


@pytest.mark.parametrize("method", ["pearson", "lr"])
def test_asymptotic_symmetries_sparse_warnings_and_independence(method):
    independent = [[10, 20, 30], [20, 40, 60]]
    result = indep_test(independent, method=method)
    assert result.statistic == pytest.approx(0, abs=1e-12)
    assert result.p_value == 1
    tab = np.array([[30, 20, 10], [10, 20, 50]])
    reference = indep_test(tab, method=method)
    for variant in (tab.T, tab[::-1, ::-1]):
        actual = indep_test(variant, method=method)
        assert actual.statistic == pytest.approx(reference.statistic)
        assert actual.p_value == pytest.approx(reference.p_value)
    with pytest.warns(RuntimeWarning, match="approximation"):
        indep_test([[0, 2], [3, 0]], method=method)


@pytest.mark.parametrize("alternative", ["two.sided", "less", "greater"])
def test_binary_fisher_pvalues_and_conditional_estimate(alternative):
    tab = [[7, 77], [2, 90]]
    result = indep_test(tab, method="fisher", alternative=alternative)
    ref = fisher_exact(tab, alternative=alternative.replace(".", "-"))
    assert result.p_value == pytest.approx(ref.pvalue)
    assert result.estimate == pytest.approx(4.060981065910681)
    assert result.estimate != pytest.approx(7 * 90 / (77 * 2))
    assert result.statistic is None and result.df is None
    assert result.residuals is None and result.stdres is None
    assert result.null_value == 1
    if alternative == "greater":
        assert np.isinf(result.conf_int[1])
    if alternative == "less":
        assert result.conf_int[0] == 0
    assert indep_test(tab, method="fisher", conf_int=False).conf_int is None
    if alternative == "two.sided":
        assert indep_test(tab, method="fisher", simulate_p_value=True).p_value == result.p_value


def test_general_fisher_matches_direct_small_sample_enumeration():
    # For margins (3,3) and (2,2,2), six permutations of (2,1,0)
    # have probability 1/10 each; (1,1,1) has probability 4/10.
    tab = np.array([[2, 1, 0], [0, 1, 2]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = indep_test(tab, method="fisher")
    assert result.p_value == pytest.approx(0.6)
    assert indep_test(tab.T, method="fisher").p_value == pytest.approx(0.6)
    assert result.estimate is None and result.conf_int is None
    # Equal margins of two: six doubled permutation tables, each 1/90.
    assert indep_test(2 * np.eye(3, dtype=int), method="fisher").p_value == pytest.approx(1 / 15)


def test_fisher_monte_carlo_is_seeded_and_uses_plus_one():
    tab = [[2, 1, 0], [0, 1, 2]]
    opts = dict(method="fisher", simulate_p_value=True, n_resamples=20000,
                random_state=42)
    result = indep_test(tab, **opts)
    assert result.p_value == indep_test(tab, **opts).p_value
    assert result.n_resamples == 20000
    assert "20000 replicates" in result.method
    assert result.p_value == pytest.approx(0.6, abs=0.015)
    assert result.p_value >= 1 / 20001
    assert (result.p_value * 20001) == pytest.approx(round(result.p_value * 20001))


def test_fisher_exhaustion_and_option_validation():
    tab = [[2, 1, 0], [0, 1, 2]]
    with pytest.raises(ValueError, match="max_tables"):
        indep_test(tab, method="fisher", max_tables=1)
    for opts in (dict(alternative="greater"), dict(correct=True),
                 dict(n_resamples=0), dict(conf_level=True),
                 dict(simulate_p_value=1), dict(unknown=True)):
        with pytest.raises(ValueError):
            indep_test(tab, method="fisher", **opts)
    with pytest.raises(ValueError, match="only for Fisher"):
        indep_test(tab, conf_int=False)


@pytest.mark.parametrize("bad", [[], [1, 2], [[1, 2]], [[[1, 2], [3, 4]]],
                                 [[1, -1], [2, 3]], [[1, 0.5], [2, 3]],
                                 [[1, np.nan], [2, 3]], [[1, np.inf], [2, 3]],
                                 [[0, 0], [2, 3]], [[1, 0], [2, 0]],
                                 [[True, False], [False, True]],
                                 [["1", "2"], ["3", "4"]], [[2 ** 53, 1], [1, 1]]])
def test_invalid_count_tables_are_rejected(bad):
    for function in (indep_test, nominal_assoc):
        with pytest.raises(ValueError):
            function(bad)


def test_nominal_formulas_and_maxima():
    result = nominal_assoc([[30, 10], [20, 40]])
    phi = abs(30 * 40 - 10 * 20) / np.sqrt(40 * 60 * 50 * 50)
    np.testing.assert_allclose(result.estimate,
                               [phi, phi / np.sqrt(1 + phi * phi), phi])
    perfect = nominal_assoc(np.diag([10, 20, 30]))
    np.testing.assert_allclose(perfect.estimate, [np.sqrt(2), np.sqrt(2 / 3), 1])
    assert nominal_assoc([[10, 20, 30], [20, 40, 60]]).estimate.max() == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert nominal_assoc([[0, 2], [3, 0]]).estimate["phi"] == 1


def test_nominal_selection_and_printing_preserve_precision():
    tab = np.array([[109, 68, 14], [72, 79, 37], [9, 42, 139]])
    result = nominal_assoc(tab, measure=["cramer-v", "phi", "cramer-v"])
    assert result.measures == ("cramer-v", "phi")
    for variant in (tab.T, tab[::-1, ::-1], 7 * tab):
        np.testing.assert_allclose(nominal_assoc(variant).estimate,
                                   nominal_assoc(tab).estimate)
    before = result.estimate.copy()
    assert "0.451" in result.summary()
    assert "Nominal association" in repr(result)
    pd.testing.assert_series_equal(before, result.estimate)
    frame = result.to_frame()
    frame.iloc[0, 0] = -1
    pd.testing.assert_series_equal(before, result.estimate)
    for invalid in ([], None, 1, "signed-phi", ["phi", "unknown"]):
        with pytest.raises(ValueError):
            nominal_assoc(tab, measure=invalid)


def test_r_reference_cases():
    path = Path(__file__).parent / "reference" / "contingency_r.json"
    with path.open() as file:
        cases = json.load(file)["cases"]
    for case in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            kind = case["kind"]
            if kind == "nominal":
                result = nominal_assoc(case["table"])
                np.testing.assert_allclose(result.estimate, case["estimate"], rtol=1e-11)
            elif kind == "homogeneity":
                result = or_homogeneity_test(case["tables"], method=case["method"],
                                             correct=case.get("correct", False))
                assert result.statistic == pytest.approx(case["statistic"], abs=1e-9)
                assert result.p_value == pytest.approx(case["p_value"], abs=1e-11)
            else:
                opts = case.get("options", {})
                result = indep_test(case["table"], method=case["method"], **opts)
                if "statistic" in case:
                    assert result.statistic == pytest.approx(case["statistic"], abs=1e-9)
                assert result.p_value == pytest.approx(case["p_value"], rel=1e-10, abs=0)
                if "expected" in case:
                    np.testing.assert_allclose(result.expected, case["expected"], rtol=1e-12)
                if "residuals" in case:
                    np.testing.assert_allclose(result.residuals, case["residuals"],
                                               atol=1e-12)
                    np.testing.assert_allclose(result.stdres, case["stdres"],
                                               atol=1e-12)
                # R fisher.test uses coarser root-finding tolerances. The
                # conditional mean and tail equations are checked separately.
                if "estimate" in case:
                    assert result.estimate == pytest.approx(float(case["estimate"]), rel=2e-4)
                if "conf_int" in case:
                    expected = [float(v) for v in case["conf_int"]]
                    np.testing.assert_allclose(result.conf_int, expected, rtol=6e-4)


@pytest.mark.parametrize("alternative", ["two.sided", "less", "greater"])
def test_fisher_inverts_the_inclusive_tails_and_fits_the_conditional_mean(alternative):
    # Independent adjacent combinatorial weights from the existing OR engine
    # verify SciPy's conditional estimate and limits used by this interface.
    from ibist.odds_ratio import _ConditionalTable
    table = np.array([[7, 77], [2, 90]])
    conditional = _ConditionalTable(table)
    result = indep_test(table, method="fisher", alternative=alternative)
    probabilities = np.exp(conditional.log_probabilities(np.log(result.estimate)))
    assert probabilities @ conditional.support == pytest.approx(table[0, 0], abs=1e-10)
    alpha = (1 - result.conf_level) / (2 if alternative == "two.sided" else 1)
    if alternative != "less":
        probability = conditional.pvalue(np.log(result.conf_int[0]), "cornfield",
                                           False, lower=True)
        assert probability == pytest.approx(alpha, abs=1e-10)
    if alternative != "greater":
        probability = conditional.pvalue(np.log(result.conf_int[1]), "cornfield",
                                           False, lower=False)
        assert probability == pytest.approx(alpha, abs=1e-10)



def test_general_fisher_keeps_precision_with_huge_margins_and_narrow_support():
    n = 10 ** 12
    # Only three possible locations for the single observation in row one.
    result = indep_test([[1, 0, 0], [n, 2 * n, 3 * n]], method="fisher")
    assert result.p_value == pytest.approx((n + 1) / (6 * n + 1), abs=1e-12)
