# Trend tests and proportion power

## Cochran--Armitage trend tests

`trend_test()` mirrors R's `trend.test()` and its existing `ca.trend.test()`
backend. Supply ordered groups as rows of an R-by-2 table; the first column
contains events. Scores default to original row positions. Supply finite,
strictly increasing scores for unequal spacing. Empty rows are dropped
after scores are assigned; both outcome columns must have positive totals.

```python
from ibist import trend_test

tab = [[8, 12], [12, 8], [16, 4]]
result = trend_test(tab, scores=[1, 2, 4], alternative="greater")
exact = trend_test([[2, 0], [1, 1], [0, 2]], exact=True)
```

The default uses the signed Cochran--Armitage normal statistic. Exact
inference conditions on both margins, includes the observed score in each
tail, and doubles the smaller tail for two-sided tests. It retains the same
reported asymptotic Z, even when the p-value is exact. `TestResult` contains
`statistic`, `chisq`, `score_statistic`, `p_value`, `alternative`, and
`method`; no estimate or CI is fabricated. R uses dotted field names.

Exact dynamic programming follows R's score merging at 15 significant
digits and its relative tie tolerance. Python limits intermediate states
to one million and raises an error without approximation; the R backend
has no matching resource limit. Counts and totals are limited to `2**53-1`
in the new entry points. The existing R entry point remains supported.

## Proportion power and sample size

`power_p1s_test()` and `power_p2s_test()` mirror R's `power.p1s.test()` and
`power.p2s.test()`. Exactly one design quantity must be omitted: `n`, a
proportion, `power`, or `sig_level`. Python's `None` corresponds to R's `NULL`.
By default, `sig_level=0.05`, `correct=False`, and `strict=False`.

```python
from ibist import power_p1s_test, power_p2s_test

power = power_p1s_test(n=50, p0=.1, p1=.25)
design = power_p1s_test(p0=.1, p1=.25, power=.8)
exact = power_p1s_test(n=35, p0=.1, p1=.25, exact=True, exact_method="midp")
unequal = power_p2s_test(p1=.3, p2=.5, power=.8, group_rate=1.5)
```

`PowerResult` corresponds to R's `power.htest`. It records `n`, relevant
proportions, `sig_level`, `power`, `alternative`, and `method`. Exact
one-sample results add `achieved_sig_level`; two-sample results add the
same descriptive `note` as R. Summaries round display values only.

One-sample supplied n is rounded up, even for normal power calculations;
a solved normal n remains continuous. Two-sample n always remains
continuous and refers to group 1; group 2 has size `n*group_rate`.
R's two-sample alternatives are `two.sided` and `one.sided`, using the
absolute group difference. One-sample alternatives are `two.sided`,
`greater`, and `less`. `strict=True` includes both tails of two-sided normal
power; otherwise the dominant-tail approximation follows base R conventions.

### Exact one-sample methods

| `exact_method` | Rejection rule |
| --- | --- |
| `quantile` | Fixed critical counts from binomial quantiles |
| `cp` | Inclusive exact binomial tails; central two-sided ordering |
| `midp` | Deterministic rejection based on central mid-p values |
| `midp-rand` | Quantile boundaries with half rejection mass at boundaries |

Power and achieved size use the full rejection probabilities under p1 and
p0, including randomized boundary contributions. Deterministic mid-p does
not guarantee nominal size control. `size_rule="minimal"` requires achieved
size no greater than nominal and sufficient power. `close` additionally
requires size at least `alpha_min_frac*sig_level`, default fraction 0.9.

Exact design scans n in order up to `max_n` because discrete power is not
monotone. The R search was corrected to guarantee the first feasible n
and respect that bound; it previously could return 5 when 1 sufficed or
when `max_n=4`. Normal one-sample root finding retains R's existing behavior:
`max_n` is an initial bracket and can be exceeded during extension.

When solving a parameter, `power` generally records the requested target;
exact sample-size designs instead return achieved power. Exact p0 and
significance-level inversions are discontinuous: a numerical root can
locate a rejection boundary without attaining the target exactly. Recompute
with `power=None` to obtain achieved power at the returned parameters.
One-sample two-sided inversion seeks p1 above p0 first, or p0 below p1
first. Two-sample p1/p2 inversion stays within [0, p2] / [p1, 1]; Python
does not extend to inadmissible probabilities. Two-sample methods are normal
approximations only; the one-sample exact options do not apply to them.

## Focused validation

Independent checks cover direct combinatorial trend tails, closed-form
normal sample sizes, all four exact rejection rules, and bounded minimal
sample-size selection. The exporter supplies 30 representative R cases,
including unequal allocation and inversions for each supported parameter.
R's two-sample root tolerance is relatively coarse (`eps**0.25`); comparison
allows `2e-4` absolute error for two-sample solved parameters and `2e-8`
for one-sample solved parameters. Exact null size derived from a
fitted p0 allows `5e-8` because it inherits that root's numerical error.
Full regression suites and package checks are reserved for before a push.

References are listed in the function help: Cochran (1954) and Armitage
(1955) for trend inference; the R package's documented rejection rules and
base R `power.prop.test()` conventions for power; Clopper and Pearson (1934)
for inclusive binomial inference. Book sources remain unchanged.
