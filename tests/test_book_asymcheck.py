"""Book criteria, workflow equivalence, and representative R parity."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ibist import (asymcheck, or_2x2_ci, prop_1s_ci, prop_2s_ci, rate_1s_ci,
                   rate_2s_ci, rate_test, rr_2x2_ci)


FUNCTIONS = dict(p1=prop_1s_ci, p2=prop_2s_ci, rr=rr_2x2_ci,
                 or_=or_2x2_ci, rate1=rate_1s_ci, rate2=rate_2s_ci,
                 rate_test=rate_test)


def test_original_book_quantities_determine_adequacy():
    assert asymcheck(prop_1s_ci, 16, 40).checks.value.iloc[0] == 9.6
    assert asymcheck(prop_1s_ci, 10, 20).condition_met is True
    # Positive cells or adjusted variance do not satisfy the book criterion.
    for function, method in [(or_2x2_ci, "gart"), (rr_2x2_ci, "katz"),
                              (prop_2s_ci, "agresti-caffo")]:
        check = asymcheck(function, [[2, 2], [2, 2]], method=method)
        assert check.checks.value.tolist() == [1, 1]
        assert check.condition_met is False
    assert asymcheck(or_2x2_ci, [[0, 40], [16, 24]],
                     method="gart").condition_met is False
    null = asymcheck(rate_test, 411, 25800, r=.0119)
    assert null.checks.value.iloc[0] == pytest.approx(307.02)
    assert asymcheck(rate_test, 411, 1, r=9).condition_met is False
    assert asymcheck(rate_test, 411, 1, r=10).condition_met is True
    low_hdl = asymcheck(rate_test, [9, 12], [1817.6, 7496.3])
    p0 = 1817.6/(1817.6+7496.3)
    assert low_hdl.checks.value.iloc[0] == pytest.approx(21*p0*(1-p0))
    assert low_hdl.condition_met is False
    for count, expected in [(49, False), (50, True)]:
        check = asymcheck(rate_1s_ci, count, method="wald")
        assert check.condition_met is expected
    for count, expected in [(9, False), (10, True)]:
        check = asymcheck(rate_1s_ci, count, method="log")
        assert check.condition_met is expected
    assert asymcheck(rate_2s_ci, [10, 11]).condition_met is False
    assert asymcheck(rate_2s_ci, [11, 11]).condition_met is True
    assert asymcheck(rate_1s_ci, 3, method="wh").condition_met is None


def test_pre_attached_post_preserve_inference_and_copy_inputs():
    vf = [[38, 62], [267, 717]]
    requests = [
        (prop_1s_ci, dict(x=[0, 16], n=40, method=["wald", "wilson", "wald"],
                         side="lower", correct=True)),
        (prop_2s_ci, dict(x=vf, method=["newcombe", "agresti-caffo"],
                         side="upper")),
        (rr_2x2_ci, dict(x=vf, method=["katz", "koopman"])),
        (or_2x2_ci, dict(x=vf, method=["gart", "cornfield"], midp=True,
                        side="upper", tol=1e-7)),
        (rate_2s_ci, dict(x=[151, 55], exposure=[57518.1, 74573.5],
                         method=["log", "score", "exact"])),
        (rate_test, dict(x=411, exposure=25800, r=.0119)),
    ]
    for function, args in requests:
        pre = asymcheck(function, **args)
        fit = function(**args, asymcheck=True)
        plain = function(**args)
        assert plain.diagnostics is None
        for check in (fit.diagnostics, asymcheck(fit), asymcheck(plain)):
            pd.testing.assert_frame_equal(pre.checks, check.checks)
            assert pre.notes == check.notes
            assert pre.condition_met == check.condition_met
        np.testing.assert_array_equal(fit.conf_int, plain.conf_int)
        if function is rate_test:
            assert fit.p_value == plain.p_value
    methods, table = ["wald", "gart"], np.array(vf)
    fit = or_2x2_ci(table, method=methods)
    table[0, 0], methods[0] = 99, "changed"
    check = asymcheck(fit)
    assert check.method == ("wald", "gart")
    assert check.details["counts"].x.tolist() == [38, 267]
    copied = check.details
    copied["counts"].iloc[0, 0] = 99
    assert check.details["counts"].x.iloc[0] == 38


def test_prechecks_run_no_inference_and_validate_options(monkeypatch):
    import ibist.proportions
    import ibist.odds_ratio
    import ibist.risk_ratio

    def fail(*args, **kwargs):
        raise AssertionError("Inference ran during a pre-check")

    monkeypatch.setattr(ibist.proportions, "_binomial_limits", fail)
    monkeypatch.setattr(ibist.odds_ratio._ConditionalTable, "interval", fail)
    monkeypatch.setattr(ibist.risk_ratio, "_koopman_lower", fail)
    assert not asymcheck(prop_1s_ci, 0, 40, method="blaker").applicable
    assert not asymcheck(or_2x2_ci, [[0, 40], [16, 24]],
                         method="bp").applicable
    assert asymcheck(rr_2x2_ci, [[38, 62], [267, 717]]).condition_met
    with pytest.raises(ValueError):
        asymcheck(prop_1s_ci, 2, 10, midp=True)
    with pytest.raises(ValueError):
        asymcheck(rate_test, 1, asymcheck=1)
    with pytest.raises(ValueError):
        asymcheck(rr_2x2_ci, [[2, 2], [2, 2]], tol=1)


CASES = json.loads((Path(__file__).parent / "reference" /
                    "book_asymcheck_r.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_book_criteria_match_r(case):
    key = case["kind"].replace("-", "_")
    function = FUNCTIONS["or_" if key == "or" else key]
    check = asymcheck(function, **case["args"])
    expected = case["result"]
    method = expected["method"]
    assert check.method == (tuple(method) if isinstance(method, list)
                            else method)
    assert check.procedure == expected["procedure"]
    assert check.applicable == expected["applicable"]
    assert check.condition_met == expected["condition_met"]
    assert list(check.notes) == expected["notes"]
    pd.testing.assert_frame_equal(check.checks,
                                  pd.DataFrame(expected["checks"]),
                                  check_dtype=False, rtol=1e-12, atol=1e-14)
    for name, reference in expected["details"].items():
        value = check.details[name]
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(value, pd.DataFrame(reference),
                                          check_dtype=False)
        elif isinstance(value, (np.ndarray, tuple)):
            np.testing.assert_allclose(value, reference, rtol=1e-12, atol=1e-14)
        elif isinstance(reference, float):
            assert value == pytest.approx(reference, rel=1e-12, abs=1e-14)
        else:
            assert value == reference
