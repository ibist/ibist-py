# Contingency-table methods

These functions are implemented in the development checkout and are not in
PyPI release 0.1.0a1. They use NumPy, SciPy, and pandas; users need neither R
nor DescTools. Functions expose Python fields and preserve full precision.
`summary(digits=3)` controls displayed decimal places, with significant-digit
scientific notation for small p-values. NumPy tables are accepted; DataFrame
row and column labels are retained in observed, expected, and residual tables.

## Independence

```python
ibist.indep_test(x, method="pearson", correct=False, **options)
```

`x` must be a two-dimensional table of nonnegative integer counts with at
least two rows and columns and positive margins. Empty margins are rejected.
Counts and their total must not exceed `2**53-1` to avoid silent loss of
integer precision; this explicit bound is a Python input restriction.

- `pearson` computes Pearson's chi-square statistic. `correct=True` is
  available only for 2 by 2 counts and uses R's capped Yates correction.
- `lr` computes the likelihood-ratio G-square statistic; a zero cell
  contributes zero. Both asymptotic methods warn when expected counts are
  below five and use the chi-square survival function with `(R-1)*(C-1)` df.
- `fisher` conditions on both margins and orders tables by conditional
  probability, not Pearson's statistic. It does not issue an asymptotic
  sparse-count warning.

The return value is a `TestResult`. Asymptotic methods supply `statistic`,
`df`, `p_value`, `method`, `observed`, `expected`, `residuals`, and `stdres`.
Both residual fields are Pearson diagnostics, including for the LR method.

Fisher options are accepted only with `method="fisher"`:

| Option | Default | Meaning |
| --- | --- | --- |
| `alternative` | `"two.sided"` | `"less"` and `"greater"` require 2 by 2 counts |
| `conf_int` | `True` | Return an exact conditional odds-ratio interval for 2 by 2 counts |
| `conf_level` | `0.95` | Confidence level between zero and one |
| `simulate_p_value` | `False` | Explicit Monte Carlo option for larger tables |
| `n_resamples` | `2000` | Positive simulation size |
| `random_state` | `None` | Integer seed or NumPy Generator |
| `max_tables` | `1000000` | Positive limit on exact search nodes/support size |

For 2 by 2 tables, `estimate` is the **conditional maximum likelihood odds
ratio**, rather than the sample cross-product ratio. `conf_int` contains
odds-ratio limits and follows `alternative`; it is not a difference-of-
proportions interval. These fields come from SciPy's conditional odds-ratio
implementation, available since SciPy 1.10. No chi-square statistic, df, or
Pearson residuals are attached to Fisher results. Simulation has no effect
on the 2 by 2 calculation, matching R's convention.

For larger tables, exact computation enumerates tables satisfying the fixed
margins. This is a native enumeration, **not R's network algorithm**, and can
be slow for large margins. Sequential hypergeometric probabilities avoid
cancellation between huge log-factorials. It raises an error when
`max_tables` is exceeded;
there is no automatic switch to simulation. Relative probability ties use
`1e-7`, matching R's binary convention. For large tables, explicitly request
`simulate_p_value=True`. SciPy's Patefield sampler draws fixed-margin tables;
the p-value is `(extreme+1)/(n_resamples+1)`. A fixed seed reproduces Python
results, but not R's random-number stream. Sampling is batched to bound memory;
`n_resamples` is recorded in the result and the method description.

## Nominal association

```python
ibist.nominal_assoc(x, measure=("phi", "contingency", "cramer-v"))
```

Using the uncorrected Pearson statistic X2, n observations, and
`d=min(R,C)`, return `sqrt(X2/n)`, `sqrt(X2/(X2+n))`, and
`sqrt(X2/(n*(d-1)))`, respectively. Phi is nonnegative, even for binary
tables; it can exceed one for larger tables. No bias correction, confidence
intervals, tests, or sparse-count warnings are included.

`measure` accepts one name or a sequence, preserving order and removing
duplicates. `NominalAssociationResult.estimate` is a named pandas Series;
`to_frame()` returns a fresh table. Other fields are `statistic`, `n`,
`observed`, and `expected`.

## Odds-ratio homogeneity

```python
ibist.or_homogeneity_test(x, method="zelen", correct=False,
                         max_tables=1000000)
```

Supply a list of 2 by 2 tables or a NumPy array of shape `(2,2,K)`, matching
R's stratum axis. The null is that the stratum odds ratios are equal, without
specifying their common value. This differs from testing a common odds ratio
against one.

- `zelen` is the default exact test, conditioning on each table's margins
  and the sum of upper-left counts. Probability ordering uses an absolute
  `1e-12` tolerance on log weights, matching R ibist. It returns p=1 for
  one stratum and permits degenerate margins. Exact enumeration can be
  expensive and raises an error beyond `max_tables` search nodes. Adjacent
  log-weight ratios preserve precision with large margins and narrow supports.
- `breslow-day` uses the Mantel--Haenszel common odds-ratio estimate and
  fits each table with its original margins. `correct=True` applies Tarone's
  adjustment; this is not a continuity correction. A finite positive MH
  estimate is required.
- `woolf` measures inverse-variance weighted dispersion of log odds ratios.
  If any cell is zero, 0.5 is added to every cell in every stratum. This
  automatic adjustment is independent of `correct`.

The asymptotic methods require at least two strata and positive margins.
They use chi-square upper tails with K-1 df and warn if fitted Breslow--Day
counts or adjusted Woolf counts are below five. `correct=True` is rejected
for methods other than Breslow--Day. All methods return `TestResult` with
`statistic`, `p_value`, `method`, and `n_strata`; only the asymptotic methods
supply `df`.

## Validation and R correspondence

Independent checks cover a Poisson independence likelihood/deviance,
closed-form chi-square tails, nominal-association cross-product formulas,
perfect association, common-OR-one Breslow--Day, two-stratum Woolf, and
small-table Fisher enumeration. The Zelen example from Table 10.5 of
*Applied Nonparametric Statistical Methods*, third edition, reproduces the
reported StatXact p-value 0.0689.

The 260 R comparison cases in `tests/reference/contingency_r.json` supplement
these checks. They include both book G-tests, the tumor association table,
zero cells, capped Yates correction, one-sided Fisher results, Tarone's
adjustment, seeded random tables/strata, and degenerate Zelen cases.
Regenerate explicitly from the R repository with:

```sh
Rscript data-raw/export-python-contingency-reference.R
```

The generator records the R Git revision and source hashes. Review changes
before accepting new reference values; matching R alone is not independent
validation.

R's `fisher.test()` uses relatively coarse numerical root-finding defaults.
In these examples its conditional estimate differs from Python's by up to
approximately `1.2e-4` relatively, and confidence endpoints by `5e-4`.
Regression checks allow relative tolerances `2e-4` and `6e-4`, respectively.
Separate checks of the conditional mean and inclusive confidence-limit tails
use absolute tolerance `1e-10`, verifying the Python results independently
with adjacent combinatorial probability weights from the existing OR engine.
Both implementations use the same statistical construction.

## References

- Fisher, R. A. (1935). The logic of inductive inference. *JRSS*, 98, 39--82.
  https://doi.org/10.2307/2342435
- Breslow, N. E., and Day, N. E. (1980). *Statistical Methods in Cancer
  Research. Volume I: The Analysis of Case-Control Studies.* IARC Scientific
  Publications No. 32.
- Tarone, R. E. (1985). On heterogeneity tests based on efficient scores.
  *Biometrika*, 72, 91--95. https://doi.org/10.1093/biomet/72.1.91
- Woolf, B. (1955). On estimating the relation between blood group and disease.
  *Annals of Human Genetics*, 19, 251--253.
  https://doi.org/10.1111/j.1469-1809.1955.tb01348.x
- Zelen, M. (1971). The analysis of several 2 x 2 contingency tables.
  *Biometrika*, 58, 129--137. https://doi.org/10.1093/biomet/58.1.129
- Patefield, W. M. (1981). Algorithm AS 159: An efficient method of generating
  R x C tables with given row and column totals. *Applied Statistics*, 30,
  91--97. https://doi.org/10.2307/2346669
- SAS Institute Inc. (2016). *SAS/STAT 14.2 User's Guide*. FREQ procedure,
  pp. 2759--2764.
  https://support.sas.com/documentation/onlinedoc/stat/142/freq.pdf
