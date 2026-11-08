"""Focused formula, inverse, invariance, and R parity checks for rate power."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from ibist import PowerResult, power_r1s_test, power_r2s_test


def test_published_minitab_rate_power_examples():
    # Fixed external values: ibist-R/RATE_POWER_VALIDATION.md lists sources.
    for n, expected in [(25, .842947), (30, .898200)]:
        one = power_r1s_test(n=n, r0=15, r1=13, alternative="less")
        assert one.power == pytest.approx(expected, rel=0, abs=5e-7)
    for rate, n, expected in [(32, 79, .902793), (38, 86, .902550)]:
        two = power_r2s_test(n=n, r1=rate, r2=35, strict=True)
        assert two.power == pytest.approx(expected, rel=0, abs=5e-7)
        design = power_r2s_test(r1=rate, r2=35, power=.9, strict=True)
        assert math.ceil(design.n) == n
        assert power_r2s_test(n=n-1, r1=rate, r2=35, strict=True).power < .9


def test_rate_designs_match_closed_form_sample_sizes():
    expected = ((norm.isf(.05)*math.sqrt(.1) +
                 norm.ppf(.8)*math.sqrt(.2)) / (.2-.1))**2 / 2
    one = power_r1s_test(r0=.1, r1=.2, power=.8, exposure=2,
                         alternative="greater")
    assert isinstance(one, PowerResult)
    assert one.n == pytest.approx(expected, rel=1e-8)
    pooled = (.1*2 + .2*6) / 8
    expected = ((norm.isf(.025)*math.sqrt(pooled*(1/2 + 1/6)) +
                 norm.ppf(.8)*math.sqrt(.1/2 + .2/6)) / (.2-.1))**2
    two = power_r2s_test(r1=.1, r2=.2, power=.8, group_rate=2,
                         exposure=(2, 3))
    assert two.n == pytest.approx(expected, rel=1e-8)
    assert two.exposure == (2., 3.) and two.group_rate == 2
    # With correction, d*n - A*sqrt(n) - c = 0 is quadratic in sqrt(n).
    a = math.sqrt(expected) * (.2-.1)
    c = .5 * (1/2 + 1/6)
    corrected_n = ((a + math.sqrt(a*a + 4*(.2-.1)*c)) / (2*(.2-.1)))**2
    corrected = power_r2s_test(r1=.1, r2=.2, power=.8, group_rate=2,
                               exposure=(2, 3), correct=True)
    assert corrected.n == pytest.approx(corrected_n, rel=1e-8)


def test_rate_inverses_round_trip_each_design_quantity():
    for fun, args in [
        (power_r1s_test, dict(n=100, r0=.1, r1=.2, sig_level=.05)),
        (power_r2s_test, dict(n=100, r1=.1, r2=.2, sig_level=.05)),
    ]:
        target = fun(**args).power
        for name, expected in args.items():
            inverse = dict(args, power=target)
            inverse[name] = None
            result = fun(**inverse)
            assert getattr(result, name) == pytest.approx(expected, rel=1e-7)


def test_units_allocation_and_replication_preserve_power():
    base = power_r2s_test(n=100, r1=.1, r2=.2, group_rate=2,
                          exposure=(2, 3), correct=True, strict=True)
    swapped = power_r2s_test(n=200, r1=.2, r2=.1, group_rate=.5,
                             exposure=(3, 2), correct=True, strict=True)
    units = power_r2s_test(n=100, r1=10, r2=20, group_rate=2,
                           exposure=(.02, .03), correct=True, strict=True)
    replicated = power_r2s_test(n=50, r1=.1, r2=.2, group_rate=2,
                                exposure=(4, 6), correct=True, strict=True)
    for result in (swapped, units, replicated):
        assert result.power == pytest.approx(base.power, rel=1e-12)
    assert "exposure: 2, 3" in base.summary()
    assert base.n == 100


def test_directions_tails_correction_and_null_power():
    greater = power_r1s_test(n=50, r0=.1, r1=.2, alternative="greater")
    wrong = power_r1s_test(n=50, r0=.1, r1=.2, alternative="less")
    corrected = power_r1s_test(n=50, r0=.1, r1=.2, alternative="greater",
                               correct=True)
    assert wrong.power < corrected.power < greater.power
    for fun, rates in [(power_r1s_test, dict(r0=.2, r1=.2)),
                       (power_r2s_test, dict(r1=.2, r2=.2))]:
        assert fun(n=20, **rates, strict=True).power == pytest.approx(.05)
        assert fun(n=20, **rates).power == pytest.approx(.025)
        assert fun(n=20.1, **rates).n == 20.1


def test_invalid_or_unattainable_rate_designs():
    for kwargs in [dict(n=10, r0=.1), dict(n=10, r0=0, r1=.2),
                   dict(n=10, r0=.1, r1=.2, power=.8),
                   dict(n=10, r0=True, r1=.2),
                   dict(n=10, r0=.1, r1=.2, correct=None)]:
        with pytest.raises(ValueError):
            power_r1s_test(**kwargs)
    for exposure in [(1, 2, 3), [[1, 2]], (1, 0), True, (1, np.nan)]:
        with pytest.raises(ValueError):
            power_r2s_test(n=10, r1=.1, r2=.2, exposure=exposure)
    with pytest.raises(ValueError, match="Equal rates"):
        power_r2s_test(r1=.2, r2=.2, power=.8)
    with pytest.raises(ValueError, match="Could not solve"):
        power_r1s_test(r0=.2, r1=.1, power=.8, alternative="greater")


CASES = json.loads((Path(__file__).parent / "reference" /
                    "rate_power_r.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_representative_r_rate_power_reference(case):
    function = power_r1s_test if case["kind"] == "one" else power_r2s_test
    result = function(**case["args"])
    for name, expected in case["result"].items():
        actual = getattr(result, name)
        if isinstance(expected, str):
            assert actual == expected
        else:
            assert actual == pytest.approx(expected, rel=5e-8, abs=1e-10)
