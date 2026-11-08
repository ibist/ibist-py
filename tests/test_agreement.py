"""Agreement inference checked independently and against R's public APIs."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from ibist import TestResult as InferenceResult, agreement_test


SKIN = np.array([[10, 4, 1, 0], [5, 10, 12, 2],
                 [2, 4, 12, 5], [0, 2, 6, 13]])


def coefficient(p, weights):
    """Direct definition on cell probabilities, not package internals."""
    observed = sum(p[i, j] * weights[i, j] for i, j in np.ndindex(p.shape))
    row, col = p.sum(axis=1), p.sum(axis=0)
    chance = sum(row[i] * col[j] * weights[i, j]
                 for i, j in np.ndindex(p.shape))
    return (observed - chance) / (1 - chance)


def delta_variance(p, weights, n):
    """Differentiate on the probability simplex and apply multinomial Cov."""
    flat = p.ravel()
    gradient = np.zeros(len(flat))
    step = 1e-6
    for i in range(len(flat) - 1):
        direction = np.zeros(len(flat))
        direction[i], direction[-1] = step, -step
        high = coefficient((flat + direction).reshape(p.shape), weights)
        low = coefficient((flat - direction).reshape(p.shape), weights)
        gradient[i] = (high - low) / (2 * step)
    covariance = np.diag(flat) - np.outer(flat, flat)
    return gradient @ covariance @ gradient / n


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("weight_method", ["unweighted", "linear", "quadratic",
                                           "custom"])
def test_variances_match_independent_delta_method(seed, weight_method):
    table = np.random.default_rng(seed).integers(2, 30, size=(3, 3))
    custom = [[1, 0.2, 0.1], [0.2, 1, 0.6], [0.1, 0.6, 1]]
    choice = custom if weight_method == "custom" else weight_method
    result = agreement_test(table, weights=choice)
    p = table / table.sum()
    weights = result.weights
    expected = coefficient(p, weights)
    independent = np.outer(p.sum(axis=1), p.sum(axis=0))
    observed_variance = delta_variance(p, weights, table.sum())
    null_variance = delta_variance(independent, weights, table.sum())
    assert result.estimate == pytest.approx(expected, abs=1e-14)
    assert result.std_error**2 == pytest.approx(observed_variance,
                                                rel=3e-7, abs=1e-12)
    assert result.null_std_error**2 == pytest.approx(null_variance,
                                                     rel=3e-7, abs=1e-12)
    delta = agreement_test(table, weights=choice, test_variance="delta")
    assert delta.std_error == result.std_error
    assert delta.null_std_error == delta.std_error
    assert delta.estimate == result.estimate
    assert delta.conf_int == result.conf_int
    assert delta.statistic == pytest.approx(expected / math.sqrt(
        observed_variance), rel=3e-7, abs=1e-10)


@pytest.mark.parametrize("weights", ["unweighted", "linear", "quadratic"])
@pytest.mark.parametrize("test_variance", ["fce", "delta"])
@pytest.mark.parametrize("alternative", ["two.sided", "greater", "less"])
def test_balanced_binary_closed_form_variances_and_tails(weights, test_variance,
                                                        alternative):
    table = [[30, 10], [10, 30]]
    result = agreement_test(table, weights=weights, test_variance=test_variance,
                            alternative=alternative, conf_level=0.9)
    estimate, n = 0.5, 80
    variance = (1 - estimate**2) / n
    selected = 1 / n if test_variance == "fce" else variance
    z = estimate / math.sqrt(selected)
    p = (2 * norm.sf(abs(z)) if alternative == "two.sided" else
         norm.sf(z) if alternative == "greater" else norm.cdf(z))
    assert result.estimate == estimate
    assert result.std_error == pytest.approx(math.sqrt(variance))
    assert result.null_std_error == pytest.approx(math.sqrt(selected))
    assert result.statistic == pytest.approx(z)
    assert result.p_value == pytest.approx(p, rel=1e-12, abs=0)
    margin = norm.isf(0.05 if alternative == "two.sided" else 0.1)
    margin *= math.sqrt(variance)
    interval = (estimate - margin, estimate + margin)
    if alternative == "greater":
        interval = (interval[0], 1)
    elif alternative == "less":
        interval = (-1, interval[1])
    assert result.conf_int == pytest.approx(interval)


def test_primary_sas_skin_example_matches_published_output():
    # SAS FREQ, Agreement Study, Figure 3.10; primary SAS documentation.
    result = agreement_test(SKIN)
    assert round(result.estimate, 4) == 0.3449
    assert round(result.std_error, 4) == 0.0724
    assert round(result.null_std_error, 4) == 0.0612
    assert round(result.statistic, 4) == 5.6366
    assert np.round(result.conf_int, 4).tolist() == [0.203, 0.4868]
    assert result.p_value < 1e-4
    assert result.n == 88


def test_empty_categories_and_numeric_labels_do_not_change_weight_geometry():
    table = [[4, 0, 1], [0, 0, 0], [2, 0, 5]]
    result = agreement_test(table, weights="linear")
    np.testing.assert_array_equal(result.weights,
                                  [[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]])
    named = pd.DataFrame(table, index=["1", "2", "10"],
                          columns=["1", "2", "10"])
    renamed = agreement_test(named, weights="linear")
    assert renamed.estimate == result.estimate
    assert renamed.conf_int == result.conf_int
    assert result.n == 12


def test_named_and_custom_weight_permutations_and_transposition():
    base = agreement_test(SKIN)
    perm = [2, 0, 3, 1]
    permuted = agreement_test(SKIN[np.ix_(perm, perm)])
    assert permuted.estimate == pytest.approx(base.estimate)
    assert permuted.std_error == pytest.approx(base.std_error)
    weighted = agreement_test(SKIN, weights="linear")
    transpose = agreement_test(SKIN.T, weights="linear")
    reverse = agreement_test(SKIN[::-1, ::-1], weights="linear")
    custom = agreement_test(SKIN[np.ix_(perm, perm)],
                            weights=weighted.weights[np.ix_(perm, perm)])
    for result in [transpose, reverse, custom]:
        assert result.estimate == pytest.approx(weighted.estimate)
        assert result.std_error == pytest.approx(weighted.std_error)
        assert result.null_std_error == pytest.approx(weighted.null_std_error)
        assert result.p_value == pytest.approx(weighted.p_value,
                                                rel=1e-12, abs=0)


def test_replication_and_tiny_normal_tails_preserve_precision():
    result = agreement_test(SKIN)
    scaled = agreement_test(SKIN * 4)
    assert scaled.estimate == result.estimate
    assert scaled.std_error == pytest.approx(result.std_error / 2)
    assert scaled.null_std_error == pytest.approx(result.null_std_error / 2)
    assert scaled.statistic == pytest.approx(result.statistic * 2)
    tiny = agreement_test([[1500, 500], [500, 1500]])
    assert 0 < tiny.p_value < 1e-100
    assert tiny.p_value == pytest.approx(2 * norm.sf(math.sqrt(1000)),
                                         rel=1e-12, abs=0)
    assert "e-" in tiny.summary()


def test_native_result_metadata_and_fresh_weights_do_not_mutate_inputs():
    weights = np.array([[1., 0.2], [0.2, 1.]])
    result = agreement_test([[9, 1], [2, 8]], weights=weights)
    assert isinstance(result, InferenceResult)
    assert result.weight_method == "custom"
    assert result.test_variance == "fce"
    assert result.df is None
    assert result.null_value == 0
    assert result.conf_level == 0.95
    assert result.estimate_name == "weighted kappa"
    before = result.weights
    weights[0, 1] = 99
    view = result.weights
    view[0, 1] = -999
    np.testing.assert_array_equal(result.weights, before)
    result.summary(digits=1)
    np.testing.assert_array_equal(result.weights, before)


def test_untruncated_and_one_sided_clipped_intervals():
    table = [[3, 0], [1, 3]]
    two = agreement_test(table)
    lower = agreement_test(table, alternative="greater")
    upper = agreement_test(table, alternative="less")
    assert two.conf_int[1] > 1
    assert lower.conf_int[1] == 1
    assert upper.conf_int[0] == -1
    assert -1 <= lower.conf_int[0] <= 1
    assert -1 <= upper.conf_int[1] <= 1


def test_custom_symmetry_tolerance_follows_r_mean_relative_comparison():
    table = [[9, 1], [2, 8]]
    close = [[1, 0.5], [0.5 + 1e-10, 1]]
    assert agreement_test(table, weights=close).weight_method == "custom"
    with pytest.raises(ValueError, match="symmetric"):
        agreement_test(table, weights=[[1, 0.5], [0.5 + 1e-6, 1]])


@pytest.mark.parametrize("table", [[], [[0, 0], [0, 0]], [[1, 0], [0, 0]],
                                 [[1, 2, 3], [2, 3, 4]],
                                 [[1, -1], [2, 3]], [[1.5, 0], [2, 3]],
                                 [[np.nan, 0], [2, 3]],
                                 [[True, False], [False, True]],
                                 [[2**53, 0], [0, 1]]])
def test_invalid_or_undefined_tables_are_rejected(table):
    with pytest.raises(ValueError):
        agreement_test(table)


@pytest.mark.parametrize("weights", ["other", None, True, [1, 2],
                                   [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   [[0, 0], [0, 1]], [[1, 2], [2, 1]],
                                   [[1, -0.1], [-0.1, 1]],
                                   [[1, np.nan], [np.nan, 1]],
                                   [[1, 1], [1, 1]],
                                   [[True, False], [False, True]]])
def test_invalid_or_uninformative_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        agreement_test([[9, 1], [2, 8]], weights=weights)


@pytest.mark.parametrize("options", [{"conf_level": True}, {"conf_level": 1},
                                    {"alternative": "lower"},
                                    {"alternative": ["greater"]},
                                    {"test_variance": "other"},
                                    {"test_variance": ["fce"]}])
def test_invalid_inference_options_are_rejected(options):
    with pytest.raises(ValueError):
        agreement_test(SKIN, **options)


def test_perfect_agreement_has_zero_ci_error_but_delta_test_is_undefined():
    result = agreement_test([[4, 0], [0, 4]])
    assert result.estimate == 1
    assert result.std_error == 0
    assert result.conf_int == (1, 1)
    assert result.statistic == pytest.approx(math.sqrt(8))
    with pytest.raises(ValueError, match="selected test variance"):
        agreement_test([[4, 0], [0, 4]], test_variance="delta")


CASES = json.loads((Path(__file__).parent / "reference" /
                    "agreement_r.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_r_reference_estimates_weights_errors_and_inference(case):
    options = {key: case[key] for key in ["weights", "conf_level",
                                         "alternative", "test_variance"]}
    if case.get("error"):
        with pytest.raises(ValueError):
            agreement_test(case["x"], **options)
        return
    result = agreement_test(case["x"], **options)
    for name in ["weight_method", "method", "estimate_name", "n", "null_value"]:
        assert getattr(result, name) == case[name]
    assert result.test_variance == case["test_variance"]
    assert result.alternative == case["alternative"]
    assert result.conf_level == case["conf_level"]
    for name in ["estimate", "statistic", "p_value", "std_error",
                 "null_std_error"]:
        atol = 1e-8 if name == "std_error" and case[name] < 1e-7 else (
            0 if name == "p_value" else 1e-11)
        np.testing.assert_allclose(getattr(result, name), case[name],
                                   rtol=2e-9, atol=atol)
    np.testing.assert_allclose(result.conf_int, case["conf_int"],
                               rtol=2e-9, atol=3e-8)
    # Portable JSON serialization can round a repeating fraction's last bit.
    np.testing.assert_allclose(result.weights, case["weight_matrix"],
                               rtol=0, atol=1e-15)
