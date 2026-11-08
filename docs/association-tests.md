# Association tests

`assoc_test()` mirrors R's `assoc.test()`. Both accept a contingency table,
one or more measures, and one inference method for the whole call. Defaults
are `measure="pearson"`, `method="multinomial"`,
`alternative="two.sided"`, and confidence level 0.95.

| Method | Supported measures | Inference |
| --- | --- | --- |
| `pearson` | `phi`, `contingency`, `cramer-v` | Pearson independence test |
| `multinomial` | Seven ordinal measures | Multinomial tests and Wald CIs |
| `cmh` | `pearson`, `spearman` | CMH linear-by-linear test |

Ordinal measures are `gamma`, `tau-b`, `tau-c`, `somers-c|r`, `somers-r|c`,
`pearson`, and `spearman`. Here `measure="pearson"` names the score
correlation; `method="pearson"` names the independence procedure for nominal
coefficients. Incompatible combinations are errors; procedures never switch
silently between measures.

```python
from ibist import assoc_test

table = [[12, 3, 1], [4, 9, 2], [1, 5, 13]]
single = assoc_test(table, measure="gamma")
print(single)

multiple = assoc_test(table, measure=["gamma", "tau-b", "somers-c|r"])
print(multiple)
print(multiple[0].p_value)
print(multiple.to_frame())

nominal = assoc_test(table, measure=["phi", "cramer-v"], method="pearson")
cmh = assoc_test(table, measure=["pearson", "spearman"], method="cmh")
```

R uses identical measure/method names with dotted function/argument names:

```r
assoc.test(tab, measure = c("gamma", "tau-b", "somers-c|r"))
assoc.test(tab, measure = c("phi", "cramer-v"), method = "pearson")
assoc.test(tab, measure = c("pearson", "spearman"), method = "cmh")
```

## Results

One measure, including a length-one sequence, returns Python's `TestResult`
or R's `htest`. Fields identify `measure` and `test_method` (R:
`test.method`) alongside the estimate, zero null, statistic, p-value,
alternative, and procedure description.

Multiple measures return `AssociationTestResults`, an ordered sequence of
`TestResult` objects. R returns a named list of `htest` objects of class
`ibist_assoc_tests`. Order and repetitions are preserved. Python's
`to_frame()` and R's `summary()` or `as.data.frame()` return full-precision
tables of estimates, standard errors, intervals, statistics, degrees of
freedom, and p-values. Unsupported fields are missing.

Printing rounds display values only and identifies shared tests: nominal
measures share one independence test; the five concordance measures share
one multinomial test. Pearson and Spearman have distinct multinomial null
variances. Groups appear in Python's `shared_tests` and R's `shared.tests`
attribute. All measures use the same data; p-values are unadjusted and
should not be interpreted as independent evidence.

## Method-specific behavior

Nominal coefficients use the uncorrected Pearson statistic for estimation.
`correct=True` requests capped Yates correction for the test of a 2 by 2
table; estimates remain unchanged. Other methods/table sizes reject this
flag. Empty margins are rejected; expected counts below five produce the
existing warning once per call. Nominal tests require `two.sided`, accept
no category scores, and supply no coefficient intervals.

Multinomial results follow `ordinal_assoc()` exactly. Standard errors and
intervals use estimation variances, not null variances. Two-sided Wald
limits are untruncated; one-sided finite limits are clipped to [-1, 1].
Intervals need not invert their tests.

CMH uses `Z=r*sqrt(n-1)` and returns its square in `chisq`. Two-sided normal
tails equal the one-degree-of-freedom chi-square upper tail; directional
tests use signed Z. CMH differs from multinomial inference for the same
coefficient and supplies no interval or estimation standard error. Its
signed statistic has `df=None`, matching the existing R CMH result.

Ordinal methods drop empty margins and follow the [ordinal guide's score
rules](ordinal-association.md). For CMH, supply both `row_scores` and
`col_scores` or neither. They affect Pearson only, including mixed requests;
Spearman-only CMH calls reject scores. All methods validate `conf_level`,
but only multinomial inference uses it for intervals.

## Validation and references

Independent checks cover Pearson chi-square, corrected tests with unchanged
estimates, expanded-data correlations, CMH normal tails, degenerate null
variances, tiny p-values, and the published SAS pain example. The exporter
adds 274 R comparison calls across all alternatives, scores, rectangular
tables, numeric labels, repeated measures, and empty margins.

P-values use relative tolerance `2e-9` without an absolute allowance;
ordinary estimates/statistics allow `1e-11` absolute error. Near-zero
standard errors allow `1e-8`, and intervals allow `3e-8`, following the
ordinal engine's documented perfect-association rounding differences.

References match R's help: Brown and Benedetti (1977),
DOI `10.1080/01621459.1977.10480995`; Mantel (1963),
DOI `10.1080/01621459.1963.10500879`; and SAS/STAT 14.2, FREQ.
Existing functions retain their defaults and output structures. Book
examples require no changes.
