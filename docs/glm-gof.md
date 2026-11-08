# Goodness of fit for fitted GLMs

`glm_gof()` mirrors R's `glm.gof()`. The first argument is a fitted
statsmodels GLM result, from either the array or formula interface.

```python
from ibist import asymcheck, glm_gof

# fit is a converged statsmodels GLM result
pre = asymcheck(glm_gof, fit, method="pearson")
result = glm_gof(fit, method="pearson", asymcheck=True)
result.diagnostics
post = asymcheck(result)

results = glm_gof(fit, method=["pearson", "deviance"])
results.to_frame()
```

R uses `glm.gof`, `TRUE`, and `result$diagnostics`. Single methods return
`TestResult` / `htest`; multiple methods return `GLMGOFResults` /
`ibist_glm_gof_tests`, preserving order and repetitions. Tests share the
same model, and their p-values are unadjusted. Python `groups` returns a
copy containing observed events, expected events, and trial/frequency n.

## Methods and assumptions

| Method | Family/link | Reference df |
| --- | --- | --- |
| `pearson` (default) | Binomial or Poisson, supported GLM links | J-rank |
| `deviance` | Binomial or Poisson, supported GLM links | J-rank |
| `hosmer-lemeshow` | Binomial logit with intercept | Actual groups minus 2 |
| `osius-rojek` | Binomial logit | 1; squared standardized Z |
| `stukel` | Binomial logit | Estimable added terms, one or two |

All methods are asymptotic. Pearson and deviance aggregate matching design
rows AND offsets. They do not pool small groups. Weights must encode
integer trials or frequencies; Poisson variance weights must be one.
Zero-weight rows are excluded. Constrained/penalized fits, quasi families,
estimated dispersion, missing retained data, and active boundary means
are unsupported. Grouped Poisson tests assess means and do not replace
individual-count dispersion diagnostics.

Hosmer-Lemeshow uses SAS target-frequency groups. Set `g` (default 10,
minimum 3) to request a group count; tied probabilities are kept together
at 14 significant digits, so actual groups may be fewer. Quantile-based
implementations can differ. Osius-Rojek uses SAS's J-centering variant.
Stukel uses efficient scores of two signed squared-linear-predictor terms,
including offsets, without an augmented fit. Function help gives formulas
and numerical tolerances.

## Approximation diagnostics

Pre-checks do not compute the GOF statistic. They can flag nonpositive df
or degenerate Osius-Rojek/Stukel calculations; inference rejects those
unestimable tests. Structural input errors remain errors. Diagnostics
retain convergence, rank, boundary distance, and grouped counts.
Pearson/deviance and Hosmer-Lemeshow screen both expected successes and
failures (or Poisson counts). Osius-Rojek checks standardizing variance;
Stukel checks estimable added directions. Sparse cells alone do not
disqualify the latter methods. Passing these checks is no guarantee of
finite-sample accuracy. Warnings do not select another method.

## Validation and compatibility

Both packages reproduce SAS's five-method Mroz example and its grouped
binomial and Poisson-offset examples. Independent formula/score checks
supplement eight representative R cases. See the authoritative
[validation report][validation] for published values and source provenance.

R's existing `logistic.gof()` retains its original output and pooling rules;
it is not silently redirected through the new, unpooled interface. Book
sources remain unchanged. Thorough checks are reserved for before a push.

[validation]:
https://github.com/ibist/ibist-R/blob/main/GLM_GOF_VALIDATION.md
