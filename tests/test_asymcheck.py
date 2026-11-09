"""Focused statistical and interface checks for shared diagnostics."""

import math
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from ibist import AsymCheckResult, asymcheck, indep_test, rate_1s_ci


def test_independence_pre_attached_post_and_copy_safety():
    tab = [[12, 8], [5, 15]]
    pre = asymcheck(indep_test, tab)
    fit = indep_test(tab, asymcheck=True)
    for check in (fit.diagnostics, asymcheck(fit),
                  asymcheck(tab, method="pearson")):
        assert check.checks.equals(pre.checks)
        assert check.details["expected"].equals(pre.details["expected"])
    expected = np.outer([20, 20], [17, 23]) / 40
    np.testing.assert_allclose(pre.details["expected"], expected)
    assert pre.condition_met is True
    assert indep_test(tab).diagnostics is None
    assert indep_test(tab).p_value == fit.p_value
    copied = pre.checks
    copied.iloc[0, 2] = -1
    details = pre.details
    details["expected"].iloc[0, 0] = -1
    assert pre.checks.iloc[0, 2] > 0
    assert pre.details["expected"].iloc[0, 0] > 0


def test_expected_count_boundaries_and_fisher_nonapplicability():
    table = [[1, 5, 5, 5, 5], [1, 5, 5, 5, 5]]
    with warnings.catch_warnings(record=True) as caught:
        check = asymcheck(indep_test, table)
    assert not caught
    assert list(check.checks.value) == [1, .8]
    assert check.condition_met is True
    assert asymcheck([[0, 1], [1, 20]], method="lr").condition_met is False
    exact = indep_test([[12, 8], [5, 15]], method="fisher", asymcheck=True)
    assert exact.diagnostics.applicable is False
    assert asymcheck(exact).condition_met is None


def test_mh_hypergeometric_formula_and_book_example():
    tables = np.array([88, 469, 48, 653, 27, 480, 34, 1324]).reshape(
        (2, 2, 2), order="F")
    check = asymcheck(tables, method="mantel-haenszel")
    assert check.details["variance_sum"] == pytest.approx(41.6369241, rel=1e-8)
    listed = asymcheck([tables[:, :, i] for i in range(2)],
                       method="mantel-haenszel")
    assert listed.checks.equals(check.checks)
    simple = asymcheck([np.ones((2, 2))], method="mantel-haenszel")
    assert simple.details["variance_sum"] == pytest.approx(1/3)
    assert simple.condition_met is False
    assert asymcheck([np.zeros((2, 2))],
                     method="mantel-haenszel").condition_met is False


def test_rate_pre_attached_post_with_multiple_methods_and_settings():
    methods = ["wald", "exact", "wald", "log"]
    pre = asymcheck(rate_1s_ci, 12, 100, method=methods,
                    side="lower", correct=False)
    fit = rate_1s_ci(12, 100, method=methods, side="lower", correct=False,
                     asymcheck=True)
    assert isinstance(pre, AsymCheckResult)
    assert pre.checks.equals(fit.diagnostics.checks)
    assert pre.checks.equals(asymcheck(fit).checks)
    assert pre.details == fit.diagnostics.details
    assert pre.details == asymcheck(fit).details
    assert pre.method == tuple(methods)
    assert pre.condition_met is False
    assert pre.checks.value.iloc[1] == pytest.approx(1/math.sqrt(12))
    assert not asymcheck(rate_1s_ci, 0, method="log").applicable
    assert not asymcheck(rate_1s_ci, 0, method="wald", correct=False).applicable
    assert asymcheck(rate_1s_ci, 0, method="wald").applicable
    assert "Book Ch. 21" in str(pre)


def test_prechecks_do_not_run_inference(monkeypatch):
    import ibist.rates
    import ibist.contingency

    def fail(*args, **kwargs):
        raise AssertionError("Inference ran during a pre-check")

    monkeypatch.setattr(ibist.rates, "_one_rate_limits", fail)
    monkeypatch.setattr(ibist.contingency, "_fisher", fail)
    assert asymcheck(rate_1s_ci, 12, method="wald").applicable
    assert not asymcheck(indep_test, [[1, 1], [1, 1]],
                         method="fisher").applicable


def test_invalid_inputs_and_postcheck_overrides():
    with pytest.raises(ValueError, match="Unsupported"):
        asymcheck(sum, [1, 2])
    with pytest.raises(ValueError, match="explicit method"):
        asymcheck([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        asymcheck(indep_test, [[0, 0], [1, 2]])
    with pytest.raises(ValueError):
        asymcheck(rate_1s_ci, 2, exposure=-1)
    with pytest.raises(ValueError, match="overrides"):
        asymcheck(rate_1s_ci(2), method="wald")
    with pytest.raises(ValueError):
        indep_test([[1, 1], [1, 1]], asymcheck=1)


CASES = json.loads((Path(__file__).parent / "reference" /
                    "asymcheck_r.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_representative_r_diagnostics(case):
    args = case["args"]
    if case["kind"] == "mh":
        check = asymcheck(args["x"], method="mantel-haenszel")
    else:
        fun = indep_test if case["kind"] == "indep" else rate_1s_ci
        check = asymcheck(fun, **args)
    expected = case["result"]
    assert check.procedure == expected["procedure"]
    method = expected["method"]
    expected_method = tuple(method) if isinstance(method, list) else method
    assert check.method == expected_method
    assert check.applicable == expected["applicable"]
    assert check.condition_met == expected["condition_met"]
    assert list(check.notes) == expected["notes"]
    assert len(check._checks) == len(expected["checks"])
    for actual, reference in zip(check._checks, expected["checks"]):
        for value, name in zip(actual, check.checks.columns):
            target = reference[name]
            numeric = isinstance(target, (int, float))
            if numeric and not isinstance(target, bool):
                assert value == pytest.approx(target, rel=1e-12, abs=1e-14)
            else:
                assert value == target
    if "expected" in expected:
        np.testing.assert_allclose(check.details["expected"],
                                   expected["expected"], rtol=1e-12)
    if "variance_sum" in expected:
        assert check.details["variance_sum"] == pytest.approx(
            expected["variance_sum"], rel=1e-12)
