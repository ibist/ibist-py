# Risk ratios and basic Poisson rates

These native Python functions follow the R implementations and the shared
statistical specification. They require no R installation at runtime.

## Risk ratios

```python
from ibist import rr_2x2_ci

result = rr_2x2_ci([[38, 62], [267, 717]], method=["koopman", "katz"])
print(result)
print(result.sample_proportions)
```

Rows are independent samples and columns are events and non-events. The
reported risk ratio is the first probability divided by the second. Both
row totals must be positive. The default is a 95% two-sided Koopman interval.

- `koopman` maximizes binomial likelihood under `p1 = ratio*p2` and inverts
  the constrained Pearson statistic. It applies neither continuity
  correction nor the Miettinen--Nurminen finite-sample variance factor.
  Log-ratio inversion and reciprocal row reversal stabilize the computation.
- `katz` exponentiates log-Wald limits, with variance
  `1/x1 - 1/n1 + 1/x2 - 1/n2`. Both event counts must be positive; no
  pseudo-counts are added. Computing variance from failure counts avoids
  cancellation when risks are close to one.

A zero event count is allowed for Koopman and gives a boundary estimate and
limit. With zero events in both rows the estimate is `nan` and the interval
is `[0, inf]`. `side="lower"` requests a lower bound;
`side="upper"` requests an upper bound. Finite one-sided bounds use
the full error probability. The other endpoint is infinity or zero.

Results use `CIResult`: `estimate` is scalar, `conf_int` contains one row per
requested method, and `sample_proportions` and `samples` return copied group
metadata. Method order and repetitions are retained. Only summaries round.

## Rate intervals

```python
from ibist import rate_1s_ci, rate_2s_ci

rate_1s_ci(411, exposure=25800, method=["exact", "score", "log"])
rate_2s_ci([9, 12], exposure=[1817.6, 7496.3], method=["score", "exact"])
```

| Function | Default | Other methods | Correction default |
| --- | --- | --- | --- |
| `rate_1s_ci()` | `exact` | `score`, `wh`, `wald`, `log` | `True` |
| `rate_2s_ci()` | `log` | `score`, `exact` | No correction option |

The confidence level defaults to 0.95. Counts are nonnegative integers and
exposures are positive finite numbers. The total count must not exceed
`2**53-1`. Two-sample exposures must have length two; scalars are not recycled.

For one sample, exposure one estimates the Poisson mean. The exact method
uses Garwood's chi-squared quantiles. Corrected score adjusts counts by
minus/plus 0.5; corrected WH uses `x+0.5` at both limits; corrected Wald
adjusts counts before applying their square-root standard errors. The
correction flag leaves exact and log methods unchanged, matching R.
At zero events, log and uncorrected Wald use R's exact upper-limit fallback.

For two samples, the parameter is the first rate divided by the second.
Log-Wald requires both event counts to be positive. Score and exact methods
condition on their sum, use Wilson or Clopper--Pearson binomial intervals,
and transform probabilities by `p*T2 / ((1-p)*T1)`. At least one event is
required. A zero count gives the corresponding boundary limit. Reciprocal
binomial limits avoid subtracting an upper probability close to one.

Both functions use `side="two.sided"`, `"lower"`, or `"upper"`, matching
R. Omitting `side` or passing `None` selects two-sided limits. Every method
supports one-sided bounds at the full error probability. `alternative` is a
compatibility alias (`greater` means lower, `less` means upper); conflicting
arguments are errors. Hypothesis tests continue to use `alternative`.

Rate intervals return a scalar `CIResult.estimate`. `samples` has columns
`x` and `exposure`, and `sample_rates` returns the observed group rates.
This keeps the ratio separate from group metadata; R's two-rate result puts
both group rates and the ratio in its estimate vector. These properties and
interval arrays return copies; DataFrame attributes use immutable tuples.

## Rate tests

```python
from ibist import rate_test

rate_test(411, exposure=25800, r=0.0119, correct=False)
rate_test([12, 5], exposure=[100, 80], alternative="greater")
```

One sample tests a positive null unit rate `r` (default 1) with a signed
normal statistic and score interval. Its half-count correction is not
capped, matching R. Two samples test equal rates using a conditional
binomial Pearson statistic with R's capped Yates correction. The returned
rate-ratio interval is log-Wald, so it requires positive counts and is not
an inversion of that test. Correction affects the test but not that interval.
`r` is ignored for two samples. No exact Poisson test is substituted.

`correct=True`, `conf_level=0.95`, and `alternative="two.sided"` are defaults.
The result uses the existing `TestResult` class with `statistic`, `p_value`,
`df`, `estimate`, `null_value`, `conf_int`, `conf_level`, and `alternative`.
Rate-specific summaries use `estimate_name`; `samples` and `sample_rates`
provide copied counts, exposures, and rates. Two-sample tests warn when an
expected binomial cell count is below five, as R does.

## Zero-count WH review

The current R corrected WH interval for zero events and exposure 10 at 95%
is `[-0.00015611287295404339, 0.24639362099673709]`. Python clips the lower
limit to zero to respect the parameter space, preserving the upper limit.
Clipping does not validate the midpoint correction or establish coverage.
The R implementation remains unchanged. The reproducible author-review note
is `ibist-R/misc/rate-interval-review.md` in the shared workspace.

## Validation and reproducibility

Tests include published SAS Katz limits, direct log-Wald formulas, Koopman
endpoints checked by independent constrained likelihood optimization,
closed-form zero-event and all-success limits, and trillion-trial boundaries.
Rate checks include inclusive Poisson tail inversion, score equations,
conditional binomial limits, exposure scaling, reversal, one-sided bounds,
correction behavior, and invalid inputs. Metadata and display precision are
also checked.

The 336 cases in `tests/reference/rr_2x2_r.json` and 123 cases in
`tests/reference/rates_r.json` supplement independent validation. Regenerate
them from the R package root with:

```sh
Rscript data-raw/export-python-risk-rate-reference.R
```

The exporter records the R revision and source MD5 hashes and retains raw
R results, including the negative WH limit. Tests account for that declared
clipping difference. Nonfinite R results use strings (`NA`, `Inf`, `-Inf`)
to preserve their meaning in JSON. Risk-ratio comparisons allow `3e-8`
relative or `2e-8` absolute endpoint error for R's default root tolerance;
rate intervals use `2e-10` relative and absolute tolerance.

## References

- Katz, Baptista, Azen, and Pike (1978). Obtaining confidence intervals for
  the risk ratio in cohort studies. *Biometrics* 34, 469--474.
- Koopman (1984). Confidence intervals for the ratio of two binomial
  proportions. *Biometrics* 40, 513--517.
- Garwood (1936). Fiducial limits for the Poisson distribution.
  *Biometrika* 28, 437--442.
- Wilson and Hilferty (1931). The distribution of chi-square.
  *Proceedings of the National Academy of Sciences* 17, 684--688.
- Wilson (1927), *JASA* 22, 209--212; Clopper and Pearson (1934),
  *Biometrika* 26, 404--413, for the underlying binomial intervals.

These references are already used in the R documentation or book. Correction
conventions are those of the existing R implementation and are not attributed
to the original transformation papers.

Rate CI results leave `null_value=None`, matching R's unset reference value.
`rate_test()` reports its explicit null hypothesis separately.
