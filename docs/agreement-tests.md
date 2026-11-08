# Agreement tests

`agreement_test()` is the Python counterpart of R's `agreement.test()`.
It provides asymptotic two-rater inference for Cohen's kappa and weighted
kappa, testing zero agreement beyond chance. It returns the existing
`TestResult`; R returns an `htest`.

```python
from ibist import agreement_test

ratings = [[9, 1], [2, 8]]
result = agreement_test(ratings)
print(result)
print(result.std_error, result.null_std_error)
print(result.weights)

weighted = agreement_test(ratings, weights="quadratic")
delta = agreement_test(ratings, weights="linear", test_variance="delta")
lower = agreement_test(ratings, alternative="greater")
```

R uses the same defaults and corresponding dotted names:

```r
agreement.test(ratings)
agreement.test(ratings, weights = "quadratic")
agreement.test(ratings, weights = "linear", test.variance = "delta")
```

## Ratings and weights

Supply a square count table with at least two categories and a positive
total. Both axes must represent the same rating categories in the same
order. Counts and their total must not exceed `2**53-1`. Categories are
neither sorted nor aligned by labels. Empty rows and columns are retained:
dropping a rating category would change positional weights.

Each call selects one weighting scheme:

| `weights` | Agreement weight for categories i and j |
| --- | --- |
| `unweighted` (default) | 1 for matching categories, 0 otherwise |
| `linear` | `1-abs(i-j)/(K-1)` |
| `quadratic` | `1-((i-j)/(K-1))**2`, Fleiss--Cohen weights |
| Custom matrix | Supplied positional agreement weights |

Named weights use category positions, not numeric labels or distances
between labels. Custom matrices must match the table, have diagonal ones,
be symmetric, and contain finite values in [0, 1]. Python follows R's
mean-relative symmetry tolerance `sqrt(machine epsilon)` and preserves
accepted weights without symmetrizing them. Inputs are copied; accessing
`result.weights` returns a fresh array.

## Estimates, intervals, and tests

Kappa is `(Po-Pe)/(1-Pe)`, where Po is observed weighted agreement and Pe
is expected agreement from independent margins. Agreement differs from
association: two raters can be strongly associated while consistently
assigning different categories. Use `assoc_test()` for association inference.

The default confidence level is 0.95 and alternative is `two.sided`.
Two-sided Wald intervals are untruncated. One-sided finite limits use the
full error probability, are clipped to [-1, 1], and set the other endpoint
to 1 for `greater` or -1 for `less`.

Confidence intervals always use the FCE estimation variance, equivalent
to the multinomial delta-method variance at observed cell proportions.
`test_variance="fce"` (default) uses the distinct FCE null variance for
testing. `test_variance="delta"` instead uses the observed estimation
variance for testing. Thus changing this option changes the test standard
error and p-value, while leaving the estimate and interval unchanged.
Default interval and test decisions need not agree exactly.

| Python field | R field | Meaning |
| --- | --- | --- |
| `estimate`, `null_value` | `estimate`, `null.value` | Kappa and zero null |
| `statistic`, `p_value` | `statistic`, `p.value` | Signed Z and normal tail |
| `conf_int`, `conf_level` | `conf.int` and its level attribute | Wald CI |
| `std_error` | `std.error` | Estimation SE used for the CI |
| `null_std_error` | `null.std.error` | Selected test SE, including delta |
| `test_variance` | `test.variance` | `fce` or `delta` |
| `weight_method`, `weights` | `weight.method`, `weights` | Scheme and matrix |
| `n` | `n` | Number of paired ratings |

The historical name `null.std.error` is retained, but for delta it contains
an estimation SE, not a null SE. The signed normal test has `df=None`.
Perfect agreement may have a zero-width CI and a valid FCE test; a zero
selected delta variance is an error. Chance agreement of one makes kappa
undefined. Invalid variances are errors. Exact tests and multi-rater kappas
are outside this implementation. Stored results retain full precision;
summaries round display values only and retain tiny normal-tail p-values.

## Compatibility and validation

The R entry point delegates to `cohen.kappa.test()` and preserves its
inference and positional arguments. The legacy function remains supported
without deprecation warnings. The book's existing calls require no changes.

Independent tests differentiate the kappa estimand under multinomial
sampling, at both observed probabilities and independent margins. Balanced
binary tables supply closed-form estimation and null variances. Checks cover
weights, empty categories, transposition, category permutations, replication,
tiny tails, invalid inputs, and copied result metadata. The published SAS
skin-condition example reproduces kappa 0.3449, estimation SE 0.0724,
test SE 0.0612, Z 5.6366, and the 95% CI (0.2030, 0.4868).

The R exporter adds 60 representative cases, including pericardial-effusion
example, all weight schemes, both variance choices, all alternatives, random
tables, and expected failures. Estimates and test statistics use `2e-9`
relative and `1e-11` absolute tolerance. P-values have no absolute allowance.
Near-zero standard errors allow `1e-8` absolute error; intervals allow `3e-8`.
Weight comparisons allow `1e-15` for portable JSON serialization of fractions.

References: Fleiss, Cohen, and Everitt (1969), Psychological Bulletin 72(5),
323--327, DOI `10.1037/h0028106`; and SAS/STAT 14.1, The FREQ Procedure,
Tests and Measures of Agreement. See the [primary SAS agreement example][sas]
and [FCE reference][fce].

[sas]:
https://support.sas.com/documentation/onlinedoc/91pdf/sasdoc_91/stat_ug_7313.pdf
[fce]: https://doi.org/10.1037/h0028106
