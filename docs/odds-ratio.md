# Odds-ratio intervals: interface and validation

```python
from ibist import or_2x2_ci

result = or_2x2_ci([[7, 27], [1, 33]],
                   method=["wald", "gart", "cornfield", "bp"])
print(result)
```

For `[[a, b], [c, d]]`, the reported estimate is `a*d/(b*c)`. Counts must be
non-negative integers. The default is a 95% Wald interval, matching R.

| R | Python | Definition |
| --- | --- | --- |
| `or.2x2.ci()` | `or_2x2_ci()` | Odds-ratio estimate and intervals |
| `conf.level` | `conf_level` | Confidence level, default 0.95 |
| `method` | `method` | One name or a sequence of names |
| `midp` | `midp` | Conditional mid-p modification, default false |
| `tol` through `...` | `tol` | Numerical inversion tolerance |
| `$estimate` | `.estimate` | Unadjusted sample odds ratio |
| `$conf.int` | `.conf_int` | Lower and upper limits, one row per method |
| `$method` | `.methods` | Method labels in requested order |

## Statistical definitions

- `wald`: exponentiate log odds ratio plus or minus the normal quantile times
  the square root of the sum of the four reciprocal counts. All cells must
  be positive.
- `gart`: use that construction after adding 0.5 to every cell. Credit for
  the correction belongs to Haldane and Anscombe; the interval construction
  follows Gart. The common reported estimate remains unadjusted, as in R.
- `cornfield`: condition on both margins and invert the two one-sided tails
  of Fisher's noncentral hypergeometric distribution at alpha/2.
- `bp`: invert the two-sided conditional test that orders possible tables
  by their probabilities under the candidate odds ratio, at alpha.

With `midp=True`, Cornfield subtracts half the observed-table probability
from each tail. BP includes half the probability of the entire group tied
with the observed table, matching the R implementation. When other tables
tie, this differs from subtracting half the observed-table mass alone. Exact
and mid-p labels distinguish the conditional intervals; Wald and Gart ignore
this option. Mid-p procedures do not guarantee at least nominal coverage.

The confidence limits are two-sided. A zero sample odds ratio has lower
conditional limit zero, and an infinite estimate has upper limit infinity.
Empty margins contain no information for conditional inference and are
rejected explicitly. Gart can still be computed for such tables, with an
undefined common estimate reported as NaN.

## Numerical implementation and validation

Python independently implements the formulas and conditional inversion using
NumPy and SciPy numerical primitives. It does not call R. Cross-products use
floating-point arithmetic to avoid integer overflow. Conditional weights
are built from adjacent weight ratios on the log scale, avoiding cancellation
between large log factorials, and normalized with log-sum-exp; combinatorial
terms are reused within a call. Ties use a relative tolerance on log
probabilities rather than R's additional absolute probability tolerance.
Root finding uses log odds ratios; `tol` is an absolute tolerance on that
scale, whereas R uses an absolute tolerance on the odds ratio itself.
BP p-values can jump when probability ordering changes, so a limit can occur
at a jump rather than at a point with p-value exactly alpha. At unusually low
confidence levels, BP mid-p inversion may not find an accepted interior
point; it then raises an explicit error. Conditional calculation enumerates
the support and may be expensive for large margins.

The checks include:

1. Wald and Gart limits against direct formulas; zero-cell Cornfield limits
   against a closed-form quadratic solution.
2. Cornfield exact limits against SciPy for all nondegenerate tables with
   cell counts 0--3; conditional probabilities against SciPy's noncentral
   hypergeometric distribution; BP's null p-value against Fisher's exact test.
   A large-margin, small-support case is also compared with exact integer
   combinatorial weights.
3. The published BP exact and mid-p example in Fagerland (2012), allowing
   for the article's numerical endpoint precision.
4. Exact coverage by enumerating every table for margins (4, 5) and (4, 5)
   across a grid of odds ratios. This is a finite validation exercise, not
   a claim of exhaustive coverage validation for all margins.
5. Sixty R comparisons spanning confidence levels 0.90, 0.95, and 0.99,
   all four methods, mid-p options, balanced and sparse counts, and zeros in
   each cell position. Row reversal and transposition also check orientation.

`tests/reference/or_2x2_r.json` contains R comparison cases. Regenerate them
explicitly with `Rscript data-raw/export-python-reference.R` from `ibist-R`,
then review the changes. The generator records the R revision and source
hashes. R output comparisons supplement the independent checks; they are
not themselves published or independent validation. Do not automatically
replace reference values when a test fails.

## References

- Anscombe, F. J. (1956). On estimating binomial response relations.
  *Biometrika*, 43, 461--464. https://doi.org/10.1093/biomet/43.3-4.461
- Baptista, J., and Pike, M. C. (1977). Algorithm AS 115: Exact two-sided
  confidence limits for the odds ratio in a 2 x 2 table.
  *Applied Statistics*, 26, 214--220. https://doi.org/10.2307/2347041
- Fagerland, M. W. (2012). Exact and mid-p confidence intervals for the odds
  ratio. *The Stata Journal*, 12, 505--514.
  https://doi.org/10.1177/1536867X1201200311
- Gart, J. J. (1966). Alternative analyses of contingency tables.
  *JRSS Series B*, 28, 164--179.
- Haldane, J. B. S. (1956). The estimation and significance of the logarithm
  of a ratio of frequencies. *Annals of Human Genetics*, 20, 309--311.
  https://doi.org/10.1111/j.1469-1809.1955.tb01285.x
