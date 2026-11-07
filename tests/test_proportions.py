import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta, binom, binomtest, norm

from ibist import CIResult, or_2x2_ci, prop_1s_ci


@pytest.mark.parametrize("method, expected", [
    ("wilson", [0.4181, 0.4773]),
    ("agresti-coull", [0.4181, 0.4773]),
    ("clopper-pearson", [0.4174, 0.4779]),
])
def test_published_sas_brown_eye_example(method, expected):
    # SAS/STAT 14.2 FREQ, Output 41.4.1, p. 2857: 341 of 762 at 90%.
    result = prop_1s_ci(341, 762, conf_level=0.9, method=method)
    np.testing.assert_allclose(np.round(result.conf_int[0], 4), expected)
    assert result.estimate == 341 / 762


@pytest.mark.parametrize("n", [1, 2, 10, 40])
@pytest.mark.parametrize("alternative", ["two.sided", "greater", "less"])
@pytest.mark.parametrize("conf_level", [0.2, 0.9, 0.99])
def test_wilson_and_exact_agree_with_scipy_including_boundaries(n, alternative, conf_level):
    for x in set([0, 1, n // 2, n]):
        reference = binomtest(x, n, alternative=alternative.replace(".", "-"))
        for method, correct, scipy_method in [
            ("wilson", False, "wilson"), ("wilson", True, "wilsoncc"),
            ("clopper-pearson", False, "exact"),
        ]:
            result = prop_1s_ci(x, n, method=method, correct=correct,
                                conf_level=conf_level, alternative=alternative)
            limits = reference.proportion_ci(confidence_level=conf_level,
                                             method=scipy_method)
            np.testing.assert_allclose(result.conf_int[0], [limits.low, limits.high],
                                       atol=1e-11, rtol=1e-10)


def test_approximate_methods_match_independent_defining_equations():
    x, n = 5, 40
    p, z = x / n, norm.isf(0.025)
    for correct in (False, True):
        result = prop_1s_ci(x, n, method=["wilson", "wald"], correct=correct)
        for direction, endpoint in zip((-1, 1), result.conf_int[0]):
            adjusted = (x + direction * (0.5 if correct else 0)) / n
            assert abs(endpoint - adjusted) == pytest.approx(
                z * np.sqrt(endpoint * (1 - endpoint) / n)
            )
        margin = z * np.sqrt(p * (1 - p) / n) + (0.5 / n if correct else 0)
        np.testing.assert_allclose(result.conf_int[1], [p - margin, p + margin])
    adjusted_n = n + z * z
    adjusted_p = (x + z * z / 2) / adjusted_n
    margin = z * np.sqrt(adjusted_p * (1 - adjusted_p) / adjusted_n)
    result = prop_1s_ci(x, n, method="agresti-coull")
    assert result.estimate == p
    np.testing.assert_allclose(result.conf_int[0], [adjusted_p - margin, adjusted_p + margin])


@pytest.mark.parametrize("alternative", ["two.sided", "greater", "less"])
def test_midp_endpoints_invert_inclusive_half_mass_tails(alternative):
    result = prop_1s_ci(5, 40, method="clopper-pearson", midp=True,
                        alternative=alternative, tol=1e-11)
    alpha = 0.05 / (2 if alternative == "two.sided" else 1)
    lower, upper = result.conf_int[0]
    if alternative != "less":
        probability = binom.sf(5, 40, lower) + 0.5 * binom.pmf(5, 40, lower)
        assert probability == pytest.approx(alpha, abs=1e-11)
    if alternative != "greater":
        probability = binom.cdf(4, 40, upper) + 0.5 * binom.pmf(5, 40, upper)
        assert probability == pytest.approx(alpha, abs=1e-11)


def test_midp_boundaries_and_huge_narrow_intervals_are_stable():
    for n in (1, 10, 10 ** 12):
        expected = -np.expm1(np.log(0.05) / n)
        zero = prop_1s_ci(0, n, method="clopper-pearson", midp=True)
        assert zero.conf_int[0, 0] == 0
        assert zero.conf_int[0, 1] == pytest.approx(expected, rel=1e-8, abs=0)
        all_successes = prop_1s_ci(n, n, method="clopper-pearson", midp=True)
        np.testing.assert_allclose(all_successes.conf_int[0],
                                   [1 - expected, 1], atol=1e-12)
    zero = prop_1s_ci(0, 5, method="clopper-pearson", midp=True,
                      conf_level=0.2, alternative="less")
    assert zero.conf_int[0, 1] == 0
    all_successes = prop_1s_ci(5, 5, method="clopper-pearson", midp=True,
                               conf_level=0.2, alternative="greater")
    assert all_successes.conf_int[0, 0] == 1


@pytest.mark.parametrize("level", [0.5, 0.8, 0.95, 0.99])
def test_blaker_encloses_the_directly_enumerated_acceptance_set(level):
    for n in range(1, 9):
        support = np.arange(n + 1)
        result = prop_1s_ci(support, n, method="blaker", conf_level=level, tol=1e-11)
        for p in np.linspace(0.005, 0.995, 41):
            mass = binom.pmf(support, n, p)
            ordering = np.minimum(binom.cdf(support, n, p), binom.sf(support - 1, n, p))
            accepted = np.array([mass[ordering <= value + 1e-14].sum() > 1 - level
                                 for value in ordering])
            included = (result.conf_int[:, 0] <= p) & (result.conf_int[:, 1] >= p)
            assert np.all(included[accepted])
            assert mass[included].sum() >= level - 1e-12


@pytest.mark.parametrize("alternative", ["less", "greater"])
def test_one_sided_blaker_is_exact_binomial_tail_inversion(alternative):
    for n in (1, 5, 20):
        result = prop_1s_ci(np.arange(n + 1), n, method="blaker",
                            alternative=alternative, conf_level=0.9)
        reference = prop_1s_ci(np.arange(n + 1), n, method="clopper-pearson",
                               alternative=alternative, conf_level=0.9)
        np.testing.assert_allclose(result.conf_int, reference.conf_int, atol=1e-13)


def test_vector_result_order_metadata_and_scalar_compatibility():
    result = prop_1s_ci([0, 5, 40], 40, method=["wald", "wilson"])
    assert isinstance(result, CIResult)
    np.testing.assert_array_equal(result.estimate, [0, 0.125, 1])
    assert result.methods == ("wald", "wilson") * 3
    assert result.inputs["input"].tolist() == [0, 0, 1, 1, 2, 2]
    assert result.inputs["x"].tolist() == [0, 0, 5, 5, 40, 40]
    assert result.inputs["n"].tolist() == [40] * 6
    frame = result.to_frame()
    assert frame.index.names == ["input", "method"]
    assert list(frame.columns) == ["lower", "upper"]
    np.testing.assert_array_equal(frame.to_numpy(), result.conf_int)
    assert len(pd.concat([frame, frame])) == 2 * len(frame)
    for i, x in enumerate((0, 5, 40)):
        scalar = prop_1s_ci(x, 40, method=["wald", "wilson"])
        assert isinstance(scalar.estimate, float)
        np.testing.assert_array_equal(result.conf_int[2 * i:2 * i + 2], scalar.conf_int)
        assert scalar.to_frame().index.name == "method"
    scalar = or_2x2_ci([[7, 27], [1, 33]])
    assert scalar.inputs is None
    assert scalar.to_frame().index.name == "method"


def test_vector_trial_counts_and_duplicate_method_requests():
    result = prop_1s_ci([0, 3, 12], [1, 4, 12], method=["wilson", "wilson"])
    assert result.inputs["n"].tolist() == [1, 1, 4, 4, 12, 12]
    np.testing.assert_array_equal(result.conf_int[::2], result.conf_int[1::2])
    np.testing.assert_array_equal(result.estimate, [0, 0.75, 1])


def test_correction_and_midp_flags_affect_only_applicable_rows():
    result = prop_1s_ci(5, 40, method=["wilson", "wald", "clopper-pearson", "blaker"],
                        correct=True, midp=True)
    assert result.methods == ("wilson-cc", "wald-cc", "clopper-pearson-midp", "blaker")
    for i, (method, correct, midp) in enumerate([
        ("wilson", True, False), ("wald", True, False),
        ("clopper-pearson", False, True), ("blaker", False, False),
    ]):
        np.testing.assert_array_equal(result.conf_int[i], prop_1s_ci(
            5, 40, method=method, correct=correct, midp=midp
        ).conf_int[0])
    for kwargs in (dict(method="blaker", correct=True), dict(midp=True),
                   dict(method="midp"), dict(method="wilsoncc"), dict(method=[])):
        with pytest.raises(ValueError):
            prop_1s_ci(5, 40, **kwargs)


def test_printing_and_returned_tables_do_not_modify_stored_values():
    assert "Data: x=5, n=40" in prop_1s_ci(5, 40).summary()
    result = prop_1s_ci([0, 5, 40], 40, method=["wilson", "blaker"],
                        alternative="greater")
    before, estimates = result.conf_int.copy(), result.estimate.copy()
    summary = result.summary()
    assert "Bound: lower" in summary and "0.125" in summary
    assert "estimate" in summary and "input" in summary
    assert str(result) == repr(result)
    inputs, frame, limits = result.inputs, result.to_frame(), result.conf_int
    inputs.iloc[0, 1] = 99
    frame.iloc[0, 0] = -1
    frame.attrs["estimate"] = (99, 99, 99)
    limits[0, 0] = -1
    assert result.inputs.iloc[0, 1] == 0
    np.testing.assert_array_equal(result.conf_int, before)
    np.testing.assert_array_equal(result.estimate, estimates)


def test_blaker_includes_the_symmetric_acceptance_jump():
    # With n=7, p=0.5, the two extreme counts have a combined mass of
    # 2/128 > 0.01, so both intervals must contain the accepted tie point.
    result = prop_1s_ci([0, 7], 7, method="blaker", conf_level=0.99)
    assert np.all(result.conf_int[:, 0] <= 0.5)
    assert np.all(result.conf_int[:, 1] >= 0.5)


@pytest.mark.parametrize("x,n", [([], 2), (1, []), ([[1]], 2), (1, [[2]]),
                                  ([1, 2], [3, 4, 5]), (1, [2, 3]),
                                  (True, 2), (1, False), (1.5, 2), (-1, 2),
                                  (1, 0), (3, 2), (1, np.inf), (np.nan, 2),
                                  (1, 2 ** 53), ("1", 2), (1, "2")])
def test_invalid_counts_are_rejected(x, n):
    with pytest.raises(ValueError):
        prop_1s_ci(x, n)


@pytest.mark.parametrize("options", [dict(conf_level=True), dict(conf_level=1),
                                      dict(conf_level=np.nan), dict(tol=True),
                                      dict(tol=0), dict(tol=1), dict(tol=np.inf),
                                      dict(correct=1), dict(midp=1),
                                      dict(alternative="two-sided"),
                                      dict(alternative=None), dict(method=None)])
def test_invalid_options_are_rejected(options):
    with pytest.raises(ValueError):
        prop_1s_ci(5, 40, **options)


def test_reflection_and_one_sided_tail_levels():
    for method in ("wilson", "wald", "agresti-coull", "clopper-pearson", "blaker"):
        result = prop_1s_ci([0, 2, 7, 10], 10, method=method)
        reflected = prop_1s_ci([10, 8, 3, 0], 10, method=method)
        np.testing.assert_allclose(result.conf_int, 1 - reflected.conf_int[:, ::-1], atol=1e-8)
        if method != "blaker":
            lower = prop_1s_ci(3, 10, method=method, alternative="greater",
                               conf_level=0.975)
            upper = prop_1s_ci(3, 10, method=method, alternative="less",
                               conf_level=0.975)
            both = prop_1s_ci(3, 10, method=method)
            assert lower.conf_int[0, 0] == pytest.approx(both.conf_int[0, 0])
            assert upper.conf_int[0, 1] == pytest.approx(both.conf_int[0, 1])


def test_r_reference_cases():
    path = Path(__file__).parent / "reference" / "prop_1s_r.json"
    cases = json.loads(path.read_text())["cases"]
    for case in cases:
        result = prop_1s_ci(case["x"], case["n"], **case["options"])
        np.testing.assert_allclose(result.estimate, case["estimate"], atol=0, rtol=1e-14)
        np.testing.assert_allclose(result.conf_int, case["conf_int"], atol=2e-8, rtol=1e-10)
        assert result.methods == tuple(case["methods"])
        pd.testing.assert_frame_equal(result.inputs, pd.DataFrame(case["inputs"]),
                                      check_dtype=False)
