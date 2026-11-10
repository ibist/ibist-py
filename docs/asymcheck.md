# Approximation diagnostics

`asymcheck()` checks the book's method-specific approximation guidelines.
It reports quantities, thresholds, and whether the guidelines are met.
Positive cells or nonzero variance alone do not establish adequacy.
Passing a guideline does not guarantee coverage or approximation accuracy.

## Before and after inference

```python
from ibist import asymcheck, rr_2x2_ci

table = [[38, 62], [267, 717]]
pre = asymcheck(rr_2x2_ci, table, method=["katz", "koopman"])
fit = rr_2x2_ci(table, method=["katz", "koopman"], asymcheck=True)
attached = fit.diagnostics
post = asymcheck(fit)
```

R uses `rr.2x2.ci`, `TRUE`, and `fit$diagnostics`. All attached options
are false by default. Supported new CI/test results retain original counts
and settings for post-checks even without attached diagnostics. Older saved
objects without retained inputs need pre-checks. Pre-checks run no inference;
post-checks never evaluate printed data names. Diagnostics never select a
new method or change intervals, statistics, or p-values.

## Criteria from the book

| Procedure | Approximation guideline | Book source |
| --- | --- | --- |
| 2x2 independence | All four null expected counts >=5 | Ch. 12 |
| Larger contingency tables | Minimum >=1; fraction >=5 at least .8 | Ch. 14 |
| Mantel–Haenszel | Sum of null conditional variances >=5 | Ch. 15 |
| Proportion CIs | Original n*p_hat*(1-p_hat) >=5 per sample | Ch. 11–12 |
| RR/OR approximate CIs | Same original-sample criterion | Ch. 15 |
| One-rate test | Null expected events r*T >=10 | Ch. 21 |
| Two-rate equality test | n*pi0*(1-pi0) >=5 | Ch. 22 |
| One-rate ordinary Wald CI | Observed x >=50 | Ch. 21 |
| One-rate log-Wald/uncorrected score CI | Observed x >=10 | Ch. 21 |
| Rate-ratio log-Wald CI | Both observed counts >10 | Ch. 22 |
| Score rate-ratio CI | Conditional binomial CI guideline | Ch. 11/22 |
| Exact and mid-p procedures | No asymptotic check | Exact sections |

For two-rate equality, `n=x1+x2` and `pi0=T1/(T1+T2)`. This test check is
separate from the accompanying CI's approximation criterion. For rate-ratio
conditional score CIs, `p_hat=x1/(x1+x2)` and `n=x1+x2`.

Proportion/RR/OR CI checks always use original observations. Gart,
Agresti–Coull/Caffo, Brown–Li, and continuity corrections cannot turn
inadequate original sample quantities into a passing check. In particular,
`[[2, 2], [2, 2]]` fails for Katz and Gart because each row quantity is 1,
even though all cells and adjusted variances are positive. The book also
shows that Wald coverage can be poor even when its guideline passes.

Chapter 21 recommends uncorrected Wilson–Hilferty even for small counts;
no binary cutoff is invented for WH or corrected score. Their conditions
remain unassessed. Plug-in Poisson skewness `1/sqrt(x)` is descriptive.
Exact intervals and zero-event log/uncorrected Wald exact fallbacks are
marked not applicable. Mid-p needs no asymptotic check but does not
promise nominal coverage.

## Supported interfaces

Pre/attached/post workflows support `indep_test`, `prop_1s_ci`, `prop_2s_ci`,
`rr_2x2_ci`, `or_2x2_ci`, `rate_1s_ci`, `rate_2s_ci`, and `rate_test`.
RR root tolerance now matches R: `0 < tol < 1`; former Python acceptance
of `tol >= 1` is removed.

Use raw stratified counts with `method="mantel-haenszel"`: a list of 2x2
tables or a 2x2xK NumPy array. R additionally supports pre-checks of
`stats::mantelhaen.test`, `stats::prop.test`, and `stats::binom.test`.
The base R results lack retained counts and need pre-checks. Those function
adapters are R-specific; no new Python proportion test API is introduced.
One-proportion tests use null n*pi0*(1-pi0) >=5; two-proportion equality
tests use four pooled expected counts >=5. Use callable pre-checks to
distinguish rate and proportion CI methods named `wald`.

GLM adequacy is unassessed pending stabilization and translation of the
R checks. See [GLM GOF](glm-gof.md). Unknown functions/results and post-check
input overrides raise errors.

## Results and interpretation

`AsymCheckResult` corresponds to R's `ibist_asymcheck`: `procedure`, `method`,
`applicable`, `checks`, `details`, `notes`, and `condition_met` (`condition.met`
in R). Criterion tables record method, quantity, threshold, comparison,
and result. CI checks also identify zero-based input/sample indices
(one-based in R). Fresh DataFrames and copied supporting quantities protect
stored precision and inputs. Method order, repeated methods, correction,
mid-p flags, confidence level, and interval side are retained.

`None` (`NA` in R) means no criterion was assessed, rather than a successful
check. Exact methods are marked not applicable. Input validity remains
separate from approximation adequacy. No diagnostic verifies independence,
sampling design, or common-effect homogeneity.

## Sources and focused validation

The companion textbook's inferential summaries and practical notes in
Chapters 11–12, 14–15, and 21–22 supply the criteria. [R stats][rstats]
also documents Cochran expected-count guidance; [NIST][poisson] gives the
Poisson skewness formula.

The R exporter `data-raw/export-python-book-asymcheck-reference.R` supplies
15 representative cases with revision and source hashes. Examples include
OSA 16/40, VF 38/100 versus 267/984, AF observed 411 against null expected
307.02, and the MI/low-HDL rate examples. Existing 14 independence/MH/rate
cases are regenerated with the corrected criteria. Comparisons use 1e-12
relative and 1e-14 absolute tolerances. Direct book formulas, threshold
boundaries, pre/attached/post equality, and skipped-inference checks
supplement R agreement. Thorough checks are reserved for before a push.

[rstats]: https://stat.ethz.ch/R-manual/
[poisson]: https://www.itl.nist.gov/div898/handbook/eda/section3/eda366j.htm
