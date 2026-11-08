"""Association APIs checked against independent tests and R references."""

import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, norm, pearsonr, spearmanr

from ibist import (AssociationTestResults, TestResult as InferenceResult,
                   assoc_test, ordinal_assoc)


TABLE = np.array([[12, 3, 1], [4, 9, 2], [1, 5, 13]])
ORDINAL = ("gamma", "tau-b", "tau-c", "somers-c|r", "somers-r|c",
           "pearson", "spearman")
NOMINAL = ("phi", "contingency", "cramer-v")


def test_default_and_sequence_return_native_results_with_named_measures():
    single = assoc_test(TABLE)
    assert isinstance(single, InferenceResult)
    assert single.measure == "pearson"
    assert single.test_method == "multinomial"
    assert single.null_value == 0
    names = ("tau-b", "gamma", "tau-b", "spearman")
    result = assoc_test(TABLE, measure=names)
    assert isinstance(result, AssociationTestResults)
    assert result.measures == names
    assert len(result) == 4
    assert all(isinstance(item, InferenceResult) for item in result)
    assert result[0] == result[2]
    assert result[:2] == result.results[:2]
    assert result.shared_tests == (names[:3],)
    assert list(result.to_frame().index) == list(names)
    assert "Shared test:" in result.summary()
    assert "p-values are unadjusted" in result.summary()
    assert repr(result) == str(result)
    before = result.to_frame()
    view = result.to_frame()
    view.iloc[0, 0] = -999
    result.summary(digits=1)
    pd.testing.assert_frame_equal(before, result.to_frame())
    with pytest.raises(ValueError, match="digits"):
        result.summary(digits=-1)
    # A one-element sequence follows the same single-result rule.
    assert assoc_test(TABLE, measure=["gamma"]).measure == "gamma"


@pytest.mark.parametrize("alternative", ["two.sided", "greater", "less"])
def test_multinomial_preserves_all_ordinal_inference(alternative):
    result = assoc_test(TABLE, measure=ORDINAL, alternative=alternative,
                         conf_level=0.9)
    old = ordinal_assoc(TABLE, alternative=alternative, conf_level=0.9)
    assert result.shared_tests == (ORDINAL[:5],)
    for i, item in enumerate(result):
        assert item.estimate == old.estimate.iloc[i]
        assert item.statistic == old.statistic.iloc[i]
        assert item.p_value == old.p_value.iloc[i]
        assert item.std_error == old.std_error.iloc[i]
        assert item.conf_int == tuple(old.conf_int.iloc[i])
        assert item.conf_level == 0.9
        assert item.df is None
        assert item.alternative == alternative
        assert item.estimate_name == ORDINAL[i]


@pytest.mark.parametrize("correct", [False, True])
def test_nominal_coefficients_stay_uncorrected_with_corrected_tests(correct):
    table = np.array([[40, 10], [15, 35]])
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    raw = np.sum((table - expected)**2 / expected)
    distance = abs(table - expected)
    if correct:
        distance = np.maximum(0, distance - 0.5)
    statistic = np.sum(distance**2 / expected)
    result = assoc_test(table, measure=NOMINAL, method="pearson",
                         correct=correct)
    assert result.shared_tests == (NOMINAL,)
    for fit in result:
        assert fit.statistic == pytest.approx(statistic)
        assert fit.p_value == pytest.approx(chi2.sf(statistic, 1),
                                             rel=1e-12, abs=0)
        assert fit.df == 1
        assert fit.conf_int is None
        assert fit.conf_level is None
        assert fit.std_error is None
        assert fit.null_value == 0
        np.testing.assert_array_equal(fit.observed, table)
    assert result[0].estimate == pytest.approx(math.sqrt(raw / table.sum()))
    assert result[1].estimate == pytest.approx(
        math.sqrt(raw / (raw + table.sum())))
    assert result[2].estimate == pytest.approx(result[0].estimate)
    if correct:
        assert "Yates" in result[0].method


@pytest.mark.parametrize("alternative", ["two.sided", "greater", "less"])
@pytest.mark.parametrize("measure", ["pearson", "spearman"])
def test_cmh_matches_expanded_score_correlation_and_normal_tails(alternative,
                                                               measure):
    cells = np.array(list(np.ndindex(TABLE.shape)))
    x, y = np.repeat(cells, TABLE.flat, axis=0).T
    rho = (pearsonr(x, y).statistic if measure == "pearson" else
           spearmanr(x, y).statistic)
    z = rho * math.sqrt(TABLE.sum() - 1)
    fit = assoc_test(TABLE, measure=measure, method="cmh",
                      alternative=alternative)
    assert fit.estimate == pytest.approx(rho)
    assert fit.statistic == pytest.approx(z)
    assert fit.chisq == pytest.approx(z**2)
    expected = (2 * norm.sf(abs(z)) if alternative == "two.sided" else
                norm.sf(z) if alternative == "greater" else norm.cdf(z))
    assert fit.p_value == pytest.approx(expected, rel=1e-12, abs=0)
    assert fit.df is None
    assert fit.conf_int is None
    assert fit.conf_level is None
    assert fit.std_error is None
    assert fit.statistic != pytest.approx(
        assoc_test(TABLE, measure=measure).statistic)


def test_cmh_mixed_scores_apply_only_to_pearson_and_empty_margins_drop():
    scores = {"row_scores": [1, 2, 10], "col_scores": [2, 3, 9]}
    result = assoc_test(TABLE, measure=["pearson", "spearman"], method="cmh",
                         **scores)
    assert result[0] == assoc_test(TABLE, method="cmh", **scores)
    assert result[1] == assoc_test(TABLE, measure="spearman", method="cmh")
    assert not result.shared_tests
    expanded = np.pad(TABLE, ((1, 1), (1, 1)))
    fit = assoc_test(expanded, method="cmh")
    assert fit.estimate == pytest.approx(
        assoc_test(TABLE, method="cmh").estimate)
    with pytest.raises(ValueError, match="totals"):
        assoc_test(expanded, measure="phi", method="pearson")


def test_degenerate_multinomial_test_does_not_turn_into_a_cmh_test():
    table = [[4, 0], [0, 4]]
    multinomial = assoc_test(table)
    assert math.isnan(multinomial.p_value)
    assert math.isnan(multinomial.statistic)
    assert multinomial.conf_int == pytest.approx((1, 1))
    cmh = assoc_test(table, method="cmh")
    assert cmh.statistic == pytest.approx(math.sqrt(7))
    assert cmh.p_value == pytest.approx(chi2.sf(7, 1), rel=1e-12, abs=0)


def test_published_sas_concordance_test_and_tiny_p_values():
    pain = [[26, 26, 23, 18, 9], [6, 7, 9, 14, 23]]
    fit = assoc_test(pain, measure="somers-r|c")
    assert round(fit.statistic, 4) == 5.1511
    assert round(fit.estimate, 4) == 0.2569
    multiple = assoc_test(TABLE * 10, measure=["gamma", "tau-b"])
    assert 0 < multiple[0].p_value < 1e-100
    assert "e-" in multiple.summary()


def test_nominal_warning_occurs_once_and_dataframe_labels_survive():
    table = pd.DataFrame([[1, 2], [3, 4]], index=["A", "B"], columns=["X", "Y"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = assoc_test(table, measure=NOMINAL, method="pearson")
    assert len(caught) == 1
    assert issubclass(caught[0].category, RuntimeWarning)
    assert list(result[0].observed.index) == ["A", "B"]
    assert list(result[0].observed.columns) == ["X", "Y"]
    result[0].observed.iloc[0, 0] = 99
    assert result[1].observed.iloc[0, 0] == 1


@pytest.mark.parametrize("options", [
    {"measure": []}, {"measure": None}, {"measure": [None]},
    {"measure": "cramer-v"}, {"measure": ["gamma", "phi"]},
    {"measure": "gamma", "method": "cmh"},
    {"measure": "spearman", "method": "pearson"},
    {"method": "invalid"}, {"method": None}, {"method": ["cmh"]},
    {"alternative": "lower"}, {"alternative": None},
    {"conf_level": True}, {"conf_level": 1}, {"correct": None},
    {"correct": True}, {"method": "cmh", "correct": True},
    {"measure": "phi", "method": "pearson", "alternative": "greater"},
    {"measure": "phi", "method": "pearson", "row_scores": [1, 2, 3]},
    {"method": "cmh", "row_scores": [1, 2, 3]},
    {"measure": "spearman", "method": "cmh", "row_scores": [1, 2, 3],
     "col_scores": [1, 2, 3]},
    {"measure": "phi", "method": "pearson", "correct": True}])
def test_invalid_combinations_are_errors(options):
    with pytest.raises(ValueError):
        assoc_test(TABLE, **options)


CASES = json.loads((Path(__file__).parent / "reference" /
                    "assoc_test_r.json").read_text())["cases"]


def number(value):
    return math.nan if value == "NA" else float(value)


@pytest.mark.parametrize("case", CASES)
def test_r_reference_collection_and_each_native_test(case):
    x = case["x"]
    if "row_labels" in case or "col_labels" in case:
        x = pd.DataFrame(x, index=case.get("row_labels"),
                         columns=case.get("col_labels"))
    options = {key: case[key] for key in
               ["measure", "method", "alternative", "conf_level", "correct",
                "row_scores", "col_scores"] if key in case}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = assoc_test(x, **options)
    assert result.measures == tuple(case["measure"])
    assert len(result) == len(case["results"])
    for actual, expected in zip(result, case["results"]):
        for name in ["method", "measure", "test_method", "alternative"]:
            assert getattr(actual, name) == expected[name]
        for name in ["statistic", "p_value", "estimate", "null_value",
                     "std_error", "chisq"]:
            value = expected[name]
            if value is None:
                assert getattr(actual, name) is None
            else:
                value = number(value)
                atol = (1e-8 if name == "std_error" and value < 1e-7 else
                        0 if name == "p_value" else 1e-11)
                np.testing.assert_allclose(getattr(actual, name), value,
                                           rtol=2e-9, atol=atol, equal_nan=True)
        assert actual.df == expected["parameter"]
        assert actual.conf_level == expected["conf_level"]
        if expected["conf_int"] is None:
            assert actual.conf_int is None
        else:
            limits = [number(v) for v in expected["conf_int"]]
            np.testing.assert_allclose(actual.conf_int,
                                       limits,
                                       rtol=2e-9, atol=3e-8, equal_nan=True)
