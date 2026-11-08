# Approximation diagnostics

`asymcheck()` has the same name and workflows in R and Python. It returns
method-specific diagnostics, separately from the test or interval result.

## Before and after inference

```python
from ibist import asymcheck, indep_test, rate_1s_ci

table = [[12, 8], [5, 15]]
pre = asymcheck(indep_test, table, method="pearson")
fit = indep_test(table, method="pearson", asymcheck=True)
attached = fit.diagnostics
post = asymcheck(fit)

ci = rate_1s_ci(12, 100, method="wald", asymcheck=True)
rate_pre = asymcheck(rate_1s_ci, x=12, exposure=100, method="wald")
rate_post = asymcheck(ci)
```

The corresponding R calls use `indep.test`, `rate.1s.ci`, `TRUE`, and
`fit$diagnostics`. The attached option defaults to false. Pre-checks inspect
inputs without running inference. Post-checks use retained counts and
settings; they never evaluate printed data names or use significance.

## Supported procedures

| Procedure | Diagnostic |
| --- | --- |
| Pearson independence | Minimum expected count and fraction at least 5 |
| G-squared independence | Same sparsity indicators, with an LR caveat |
| Conditional Fisher | Not applicable; no chi-square approximation |
| Mantel–Haenszel | Sum of conditional hypergeometric variances |
| One-rate intervals | Events and plug-in Poisson skewness |
| GLM goodness of fit | Expected counts, rank, df, method standardization |

Use raw stratified counts with `method="mantel-haenszel"`. Input is a list
of 2-by-2 tables or a 2-by-2-by-K NumPy array. R also accepts a function
pre-check of `stats::mantelhaen.test` with a count array; this adapter is
R-specific. Its result does not retain the inputs needed for a post-check.
R's existing `mantelhaen.asymcheck()` keeps its original output and print
method without deprecation warnings, using the shared calculation.

Use `asymcheck(glm_gof, fit, method=...)` before GLM GOF inference, or
`asymcheck(result)` afterward. It also accepts GLMGOFResults collections.
All five GLM GOF methods are asymptotic. Degenerate standardization or
nonpositive test df is reported as a failed criterion in pre-checks;
inference rejects these unestimable tests. See [GLM GOF](glm-gof.md).

Other procedures are not yet supported. Unknown functions or result objects
raise clear errors. Additional Fisher inference options are not accepted
by pre-checks; they do not affect these sparsity indicators.

## Results and interpretation

`AsymCheckResult` corresponds to R's `ibist_asymcheck`. It contains
`procedure`, `method`, `applicable`, `checks`, `details`, `notes`, and
`condition_met` (`condition.met` in R). Each check records its method,
criterion, value, threshold, comparison, and whether the criterion is met.
Python exposes fresh DataFrames and copies of supporting quantities.

The expected-count screen uses minimum >=1 and fraction >=5 of at least
.8. These Pearson heuristics do not establish G-squared accuracy. The
Mantel–Haenszel screen retains the legacy variance-sum >=5 criterion;
empty strata contribute zero. Passing a rule does not verify independence,
sampling assumptions, or common-effect homogeneity.

Approximate rate intervals report events and plug-in count skewness
`1/sqrt(x)`. No universal cutoff is asserted, so `condition_met` is `None`
(`NA` in R). At zero events, skewness is undefined. Exact intervals and
the zero-event log/uncorrected Wald exact fallbacks are marked not
applicable. Mixed method requests preserve order and repetitions. Diagnostic
checks never switch methods or change p-values or confidence limits.

## Sources and focused validation

R stats' *Fisher's Exact Test for Count Data*, Details, documents the
expected-count criteria; NIST/SEMATECH's *e-Handbook*, section 1.3.6.6.19,
documents Poisson skewness. See [R stats][rstats] and [NIST][poisson].
The Mantel–Haenszel threshold preserves the existing R heuristic.

Focused checks include exact screening boundaries, direct hypergeometric
variance, the legacy book example, pre/attached/post equivalence, exact
fallbacks, no-inference pre-checks, and 14 representative R cases. Thorough
package checks remain reserved for before a push. Book files are unchanged.

[rstats]: https://stat.ethz.ch/R-manual/
[poisson]: https://www.itl.nist.gov/div898/handbook/eda/section3/eda366j.htm
