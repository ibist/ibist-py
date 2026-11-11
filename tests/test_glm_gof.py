"""Published SAS results, independent formulas, and focused interface checks."""

from importlib import import_module
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.tools.tools import add_constant
from scipy.special import logit

from ibist import GLMGOFResults, asymcheck, glm_gof
from ibist import TestResult as InferenceResult


REFERENCE = Path(__file__).parent / "reference"
METHODS = ("pearson", "deviance", "hosmer-lemeshow", "osius-rojek", "stukel")


def age_fit():
    y = np.array([0, 0, 0, 7, 6, 17])
    n = np.array([14, 20, 19, 18, 12, 17])
    x = add_constant(np.arange(25, 76, 10))
    return GLM(np.c_[y, n-y], x, family=families.Binomial()).fit(
        tol=1e-10)


@pytest.fixture(scope="module")
def sas_fit():
    frame = pd.read_csv(REFERENCE / "glm_gof_sas_mroz.csv")
    x = add_constant(frame.drop(columns="observed")).to_numpy()
    return GLM(frame.observed, x, family=families.Binomial()).fit(
        tol=1e-10)


def test_published_sas_logistic_statistics(sas_fit):
    # SAS/STAT LOGISTIC, Example 79.19, Output 79.19.1 and 79.19.3.
    published = [538.4877, 606.5035, 13.2967, .8412, 2.2581]
    published_p = [.5586, .0324, .1020, .3590, .3233]
    results = glm_gof(sas_fit, method=METHODS)
    assert isinstance(results, GLMGOFResults)
    assert results.methods == METHODS
    for fit, stat, p in zip(results, published, published_p):
        assert fit.statistic == pytest.approx(stat, rel=0, abs=5.1e-5)
        assert fit.p_value == pytest.approx(p, rel=0, abs=5.1e-5)
    assert list(results[2].groups.n) == [55]*9 + [57]
    assert list(results.to_frame().df) == [544, 544, 8, 1, 2]


def test_published_grouped_binomial_and_poisson():
    result = glm_gof(age_fit(), method=["pearson", "deviance"])
    np.testing.assert_allclose([r.statistic for r in result],
                               [6.6020, 7.7756], rtol=0, atol=5e-5)
    y = np.array([42, 37, 1, 101, 73, 14])
    x = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0],
                  [1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 1]])
    fit = GLM(y, x, family=families.Poisson(),
                 exposure=[500, 1200, 100, 400, 500, 300]).fit(tol=1e-10)
    result = glm_gof(fit, method=["pearson", "deviance"])
    np.testing.assert_allclose([r.statistic for r in result],
                               [2.8416, 2.8207], rtol=0, atol=5e-5)
    assert result[0].df == 2
    assert result[0].statistic == pytest.approx(fit.pearson_chi2, rel=1e-10)
    assert result[1].statistic == pytest.approx(fit.deviance, rel=1e-10)


def test_stukel_matches_independent_statsmodels_score_test(sas_fit):
    eta = logit(np.asarray(sas_fit.fittedvalues))
    z = np.c_[eta**2*(eta >= 0), eta**2*(eta < 0)]
    score = sas_fit.score_test(exog_extra=z)
    # Statsmodels 0.15 exposes the constraint count on the result object.
    if hasattr(score, "k_constraint"):
        statistic, p_value = score
        df = score.k_constraint
    else:
        statistic, p_value, df = score
    result = glm_gof(sas_fit, method="stukel")
    assert result.statistic == pytest.approx(np.asarray(statistic).item(),
                                             rel=1e-9)
    assert result.p_value == pytest.approx(np.asarray(p_value).item(),
                                           rel=1e-9)
    assert result.df == df


def test_osius_rojek_matches_direct_variance_formula():
    fit = age_fit()
    p = np.asarray(fit.fittedvalues)
    n = np.array([14, 20, 19, 18, 12, 17])
    x = fit.model.exog
    v = n*p*(1-p)
    c = x.T@(1-2*p)
    sigma2 = 2*len(p) + np.sum((1/(p*(1-p))-6)/n)
    sigma2 -= c @ np.linalg.solve(x.T@(v[:, None]*x), c)
    pearson = np.sum((np.array([0, 0, 0, 7, 6, 17])-n*p)**2/v)
    expected = (pearson-len(p))**2/sigma2
    assert glm_gof(fit, method="osius-rojek").statistic == pytest.approx(
        expected, rel=1e-10)


def test_pre_attached_post_are_equal_without_test_calculation(monkeypatch):
    fit = age_fit()
    result = glm_gof(fit, method="stukel", asymcheck=True)
    assert isinstance(result, InferenceResult)
    assert result.diagnostics.checks.equals(asymcheck(result).checks)
    module = import_module("ibist.glm_gof")

    def fail(*args):
        raise AssertionError("GOF statistic ran during a pre-check")

    monkeypatch.setattr(module, "_stat", fail)
    pre = asymcheck(glm_gof, fit, method="stukel")
    assert pre.checks.equals(result.diagnostics.checks)
    copied = result.groups
    copied.iloc[0, 0] = -1
    assert result.groups.iloc[0, 0] >= 0


def test_multiple_methods_retain_order_and_diagnostics(sas_fit):
    result = glm_gof(sas_fit, method=["stukel", "osius-rojek", "stukel"])
    assert result.methods == ("stukel", "osius-rojek", "stukel")
    assert result[0].statistic == result[2].statistic
    pre = asymcheck(glm_gof, sas_fit,
                    method=["stukel", "osius-rojek", "stukel"])
    assert pre.checks.equals(asymcheck(result).checks)
    assert "unadjusted" in str(result)
    assert glm_gof(sas_fit, method="stukel").diagnostics is None


def test_zero_frequency_boundary_predictions_are_ignored():
    y = np.array([2, 4, 6, 7, 9, 12, 0])/20
    x = add_constant(np.array([0, 1, 2, 3, 4, 5, 1000]))
    fit = GLM(y, x, family=families.Binomial(), var_weights=np.repeat(20, 7),
              freq_weights=[1, 1, 1, 1, 1, 1, 0]).fit()
    base = GLM(y[:-1], x[:-1], family=families.Binomial(),
               var_weights=np.repeat(20, 6)).fit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert glm_gof(fit).statistic == pytest.approx(
            glm_gof(base).statistic, rel=1e-8)


def test_family_validation_ties_and_zero_variance():
    fit = age_fit()
    with pytest.raises(ValueError):
        glm_gof(fit, method="unknown")
    with pytest.raises(ValueError):
        glm_gof(fit, g=4)
    gaussian = GLM([1, 3, 2, 5], add_constant(range(4))).fit()
    with pytest.raises(ValueError, match="Binomial/Poisson"):
        glm_gof(gaussian)
    probit = GLM(fit.model.endog, fit.model.exog,
                    family=families.Binomial(families.links.Probit()),
                    var_weights=fit.model.n_trials).fit()
    with pytest.raises(ValueError, match="logit"):
        glm_gof(probit, method="stukel")
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    flat = GLM(y, add_constant(np.arange(8)-3.5),
                  family=families.Binomial()).fit()
    with pytest.raises(ValueError, match="variance"):
        glm_gof(flat, method="osius-rojek")
    with pytest.raises(ValueError, match="degrees of freedom"):
        glm_gof(flat, method="stukel")
    with pytest.raises(ValueError, match="degrees of freedom"):
        glm_gof(flat, method="hosmer-lemeshow")
    for method in ("osius-rojek", "stukel", "hosmer-lemeshow"):
        assert asymcheck(glm_gof, flat, method=method).condition_met is None


CASES = json.loads((REFERENCE / "glm_gof_r.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["label"])
def test_representative_r_glm_reference(case):
    model = case["model"]
    family = (families.Binomial(families.links.Probit())
              if model["link"] == "probit" else families.Binomial()
              if model["family"] == "binomial" else families.Poisson())
    y, n = np.asarray(model["y"]), np.asarray(model["n"])
    endog = np.c_[y, n-y] if model["family"] == "binomial" else y
    freq = np.ones(len(y)) if model["family"] == "binomial" else n
    fit = GLM(endog, model["x"], family=family, freq_weights=freq,
                     offset=model["offset"]).fit(tol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        results = glm_gof(fit, **case["args"])
    if isinstance(results, InferenceResult):
        results = [results]
    assert len(results) == len(case["result"])
    for result, expected in zip(results, case["result"]):
        assert result.test_method == expected["method"]
        assert result.df == expected["df"]
        assert result.statistic == pytest.approx(expected["statistic"],
                                                 rel=2e-7, abs=1e-9)
        assert result.p_value == pytest.approx(expected["p_value"],
                                               rel=2e-7, abs=1e-10)
        np.testing.assert_allclose(result.groups, expected["groups"], rtol=2e-7,
                                   atol=1e-9)


def test_glm_adequacy_is_unassessed_pending_r_stabilization(sas_fit):
    pre = asymcheck(glm_gof, sas_fit, method=METHODS)
    assert pre.condition_met is None
    assert pre.checks["threshold"].isna().all()
    assert pre.checks["condition_met"].isna().all()
    for method in METHODS:
        result = glm_gof(sas_fit, method=method, asymcheck=True)
        assert result.diagnostics.condition_met is None
        assert asymcheck(result).checks.equals(
            asymcheck(glm_gof, sas_fit, method=method).checks)
