# Binomial proportion intervals

`prop_1s_ci()` and `prop_2s_ci()` are available in the development checkout,
after release 0.1.0a1. Both are native Python implementations requiring no R.
The authoritative [shared specification](https://github.com/ibist/ibist-R/blob/main/BINOMIAL_INTERVALS.md)
defines the statistical methods; this guide documents the Python interface.

```python
ibist.prop_1s_ci(x, n, conf_level=0.95, method="wilson", correct=False,
                midp=False, side=None, alternative=None, tol=1e-8)
```

## Inputs and options

The common CI argument is `side`: `"two.sided"` (default), `"lower"`, or
`"upper"`. Omitted `side` or `None` selects two-sided limits. `alternative`
remains a compatibility alias; greater means lower and less means upper.
Conflicting arguments are errors, matching R.

`x` contains successes and `n` trials. Both can be scalars or one-dimensional
numeric arrays. Trial counts are positive integers; successes are nonnegative
integers not exceeding their trials. A scalar or length-one `n` is recycled,
otherwise `n` and `x` must have the same length. Empty arrays, nested arrays,
boolean arrays, missing values, fractions, and counts exceeding `2**53-1`
are rejected. The last restriction protects integer precision in numerical
calculations and is an explicit difference from R's input validation.

The default is a two-sided 95% Wilson interval. `method` accepts a name or
sequence; requested order and repetitions are retained, matching R's
`prop.1s.ci()`. Confidence levels and numerical tolerances must be finite and
strictly between zero and one. Method names are exact strings rather than
R-style abbreviations.

| Method | Construction | Optional modification |
| --- | --- | --- |
| `wilson` | Invert the binomial score equation | `correct=True` |
| `wald` | Observed proportion plus/minus normal margin | `correct=True` |
| `agresti-coull` | Add z-squared trials and z-squared/2 successes | None |
| `clopper-pearson` | Invert inclusive binomial tails using beta quantiles | `midp=True` |
| `blaker` | Enclose the smaller-inclusive-tail acceptance set | None |

Correction and mid-p apply only to their relevant methods in a mixed call.
Requesting a flag that affects none of the selected methods is an error.
There is no method named `midp` or `wilsoncc`; use the separate flags.
The reported point estimate is always `x/n`, including for Agresti--Coull.
All endpoints are restricted to [0,1]. Wald limits can collapse at zero or
all successes; Wilson is the default approximate method.

Wilson correction uses `(x-0.5)/n` in the lower-limit equation and
`(x+0.5)/n` in the upper-limit equation, clipping to [0,1]. Wald adds
`0.5/n` to its margin. These use the full half-count correction; they do not
apply the null-dependent correction cap used by R's `prop.test()`.

`side="lower"` returns a lower confidence bound and upper limit one;
`side="upper"` returns an upper bound and lower limit zero. Each uses
the full error probability, while two-sided central intervals divide it
between their two tails. These options change the interval direction but
supply no hypothesis test or p-value. Vector intervals are pointwise;
there is no simultaneous or multiple-comparison adjustment.

## Exact, mid-p, and numerical behavior

Clopper--Pearson inverts inclusive tails. Mid-p subtracts half the observed
count's probability from each inclusive tail; it does not guarantee nominal
coverage. Python solves mid-p roots on the logit scale to retain relative
accuracy for tiny endpoints. The `tol` setting bounds error on that scale;
the logistic transformation gives a tighter absolute probability error.
Beta upper quantiles use the survival probability directly, avoiding
subtraction of a tiny tail from one.

Blaker's two-sided interval encloses all accepted probabilities from
smaller-inclusive-tail ordering. The p-value can jump when an opposite-tail
count enters the acceptance ordering, so the search partitions at those
changes instead of applying one root finder to a presumed continuous p-value.
Numerical lower endpoints are rounded outward by `tol` (and upper endpoints
by reflection), retaining accepted jump points in the closed interval. This
can widen an endpoint relative to R's result by about the chosen tolerance.
A one-sided Blaker bound is the exact binomial-tail bound, not an endpoint
of its two-sided Blaker interval.

Blaker can be slower for large samples. Python rejects an inversion spanning
more than 1,000,000 ordering changes; R does not currently impose this guard.
Wilson is preferable for large approximate coverage studies. Repeated
count/trial/method combinations are cached within a call, so repeated
observations need not trigger repeated numerical inversions.

## Results and presentation

The result is the existing `CIResult` with:

- `estimate`: a float for one input, otherwise a NumPy vector in input order.
- `conf_int`: a fresh two-column array of full-precision lower/upper limits,
  ordered by input and then method.
- `methods`: one label per interval, including `wilson-cc`, `wald-cc`, or
  `clopper-pearson-midp` where applicable.
- `inputs`: a fresh DataFrame with one row per interval, containing the
  zero-based input index, success count `x`, and trial count `n`.
- `conf_level`, `side`, `correct`, and `midp`: the requested settings.

`to_frame()` keeps the existing method index for one input. For vectors it
uses a `(input, method)` MultiIndex and the same `lower` and `upper` columns.
DataFrame attributes contain full-precision estimates and input metadata as
plain immutable tuples for vector inputs, allowing normal pandas concatenation.
R instead uses one-based input indices and appends them to interval labels.

```python
import ibist

scalar = ibist.prop_1s_ci(5, 40, method=["wilson", "clopper-pearson", "blaker"])
print(scalar)

vector = ibist.prop_1s_ci([0, 5, 40], 40, method=["wilson", "wald"])
print(vector.to_frame())
print(vector.inputs)

midp = ibist.prop_1s_ci(5, 40, method="clopper-pearson", midp=True)
lower = ibist.prop_1s_ci(5, 40, side="lower")
```

Printing uses three decimal places by default and scientific notation for
tiny values. `summary(digits=...)` changes presentation only. Vector summaries
show counts and estimates alongside interval rows. Existing scalar odds-ratio
summaries and their DataFrame layout are preserved; their `midp` metadata now
also records the flag passed to `or_2x2_ci()`.

## Differences of independent proportions

```python
ibist.prop_2s_ci(x, n=None, conf_level=0.95, method="newcombe", correct=False,
                side=None, alternative=None, tol=1e-8, nuisance_tol=1e-10)
```

Provide two success counts `x` and two trial counts `n`. A scalar `n` is
not recycled for this interface. Alternatively, provide a 2x2 table with
independent samples in rows and successes/failures in columns, omitting `n`.
The reported estimate is the first proportion minus the second. All-zero
success or failure columns are valid; each sample must still have positive
trials. Each trial count is restricted to `2**53-1`, as for `prop_1s_ci()`.

The six approximate methods are:

| Method | Construction |
| --- | --- |
| `newcombe` (default) | Combine distances to separate Wilson bounds in quadrature |
| `wald` | Normal interval using the unpooled binomial variance |
| `agresti-caffo` | Wald after adding one success and one failure per sample |
| `miettinen-nurminen` | Constrained score inversion with variance factor N/(N-1) |
| `mee` | The same score inversion without the N/(N-1) factor |
| `brown-li` | Adjusted proportions (x+0.5)/(n+1), normal margin using original n |

Newcombe is a hybrid Wilson interval; it does not invert the pooled score
statistic for testing equality. Correction uses corrected component Wilson
bounds. Wald correction adds `1/(2*n1)+1/(2*n2)` to its margin. `correct=True`
affects only Newcombe and Wald, including in a mixed method request; it is
an error if neither method is selected. In particular, `correct` never
switches between Mee and Miettinen--Nurminen. Their fixed variance factor
is part of the method definition.

Brown--Li uses the **original** sample sizes as variance denominators, not
n+1 or posterior variances, and is not a posterior-quantile interval.
Agresti--Caffo and Brown--Li use adjusted centers internally, while reporting
the observed difference as their common point estimate.

`side="lower"` gives a lower confidence bound with upper endpoint
one; `side="upper"` gives an upper bound with lower endpoint minus
one. All limits are restricted to [-1,1]. Method sequences preserve order
and repetitions. The return value is `CIResult`, with a scalar `estimate`,
one interval per requested method, `side`, `correct`, and a
`null_value` of zero for reference. It supplies no test statistic or p-value.

`sample_proportions` is a fresh two-element NumPy array, and `samples` is a
fresh DataFrame with `x` and `n`, indexed by sample 0 and 1 in comparison
order. `to_frame()` retains its ordinary method index and lower/upper
columns; its attributes record immutable sample counts and options.
The sample metadata are distinct from `prop_1s_ci()`'s per-interval `inputs`.

```python
comparison = ibist.prop_2s_ci([12, 4], [40, 30],
                             method=["newcombe", "miettinen-nurminen", "mee"])
print(comparison)
print(comparison.sample_proportions)
print(comparison.samples)

# Equivalent orientation: independent groups in rows, events first.
comparison = ibist.prop_2s_ci([[12, 28], [4, 26]], method="mee")
```

Score inversion maximizes the constrained binomial likelihood with
`p1-p2=delta`. Python solves the nuisance equation on a logit scale and
constructs success/failure probabilities separately, preserving tiny failure
probabilities when a success probability rounds to one. The outer inversion
uses a logit-scaled distance from the observed difference to a parameter
boundary. `tol` applies on that transformed scale, yielding good relative
accuracy for narrow intervals; floating-point precision still limits the
attainable endpoint accuracy. Independent likelihood maximization and
closed-form zero-event limits check this construction.

Chan--Zhang exact intervals are available with `method="chan-zhang"`.
They invert inclusive unconditional score tails with global nuisance
maximization. An enclosing outer search retains disconnected acceptance
regions. `nuisance_tol=1e-10` controls probability ambiguity, retained
conservatively; `tol=1e-8` controls outward endpoint precision. Enumeration
requires at most one million possible tables and can be expensive. No
approximate method is substituted. See
[paired and exact intervals](paired-and-exact-intervals.md) for validation.
There is no mid-p option for independent-difference intervals.

## Validation

Independent checks include:

- SAS/STAT 14.2 FREQ Output 41.4.1 (p. 2857): 341 successes in 762 trials,
  with published 90% Wilson, Agresti--Coull, and Clopper--Pearson limits.
- SciPy binomial-test Wilson, continuity-corrected Wilson, and exact limits
  across boundaries, one-sided alternatives, and confidence levels below 50%.
- The Wilson score equations and direct Wald/Agresti--Coull formulas.
- Mid-p endpoint tail equations, boundary formulas, and tiny endpoints with
  one trillion trials.
- Direct enumeration of the Blaker acceptance set and coverage for every
  outcome at n=1 through 8, on 41 probabilities, at 50%, 80%, 95%, and 99%.
  This is a finite coverage check, not an exhaustive proof for every n and p.
- A symmetric Blaker jump at n=7 and p=0.5, where floating-point rounding must
  not exclude an accepted tie, plus reflection and one-sided exact bounds.
- Vector row ordering, input recycling, copied metadata, scalar compatibility,
  and multiple correction/mid-p settings.

The 358 cases in `tests/reference/prop_1s_r.json` supplement those independent
checks. The exporter records R's Git revision and source hashes. Regenerate
explicitly from the R repository and review differences before accepting them:

```sh
Rscript data-raw/export-python-prop-1s-reference.R
```

R comparisons allow absolute endpoint differences of `2e-8`, covering R's
inversion precision and Python's outward Blaker rounding. Closed-form and
independent defining-equation checks use tighter method-specific tolerances.
No R installation or DescTools dependency is needed to run the Python tests.

For `prop_2s_ci()`, independent checks cover direct Wald, Agresti--Caffo,
and Brown--Li formulas, Newcombe using independently obtained SciPy Wilson
bounds, and Mee/Miettinen--Nurminen score equations evaluated using a
separate constrained likelihood optimizer. Analytic zero-event limits check
the variance factor and remain accurate for trillion-trial examples.
Boundary counts, row reversal, success/failure complementation, confidence
levels below 50%, correction flags, and copied group metadata are also tested.

The additional 255 R comparisons are in `tests/reference/prop_2s_r.json`.
Regenerate with `Rscript data-raw/export-python-prop-2s-reference.R` in the R
repository. They use absolute endpoint tolerance `2e-8` for the numerically
inverted score methods. Formula-based comparisons use tighter tolerances.
These comparisons supplement the defining-equation checks rather than
establishing correctness on their own.

## References

- Wilson, E. B. (1927). Probable inference, the law of succession, and
  statistical inference. *JASA*, 22, 209--212.
- Clopper, C. J., and Pearson, E. S. (1934). The use of confidence or fiducial
  limits illustrated in the case of the binomial. *Biometrika*, 26, 404--413.
- Agresti, A., and Coull, B. A. (1998). Approximate is better than "exact" for
  interval estimation of binomial proportions. *The American Statistician*,
  52, 119--126.
- Blaker, H. (2000). Confidence curves and improved exact confidence intervals
  for discrete distributions. *Canadian Journal of Statistics*, 28, 783--798.
  Correction (2001), 29, 681.
- Brown, L. D., Cai, T. T., and DasGupta, A. (2001). Interval estimation for
  a binomial proportion. *Statistical Science*, 16, 101--133.
- SAS Institute Inc. (2016). *SAS/STAT 14.2 User's Guide*. FREQ procedure.
  https://support.sas.com/documentation/onlinedoc/stat/142/freq.pdf

- Newcombe, R. G. (1998). Interval estimation for the difference between
  independent proportions: comparison of eleven methods.
  *Statistics in Medicine*, 17, 873--890.
- Miettinen, O., and Nurminen, M. (1985). Comparative analysis of two rates.
  *Statistics in Medicine*, 4, 213--226.
- Mee, R. W. (1984). Confidence bounds for the difference between two
  probabilities. *Biometrics*, 40, 1175--1176.
- Agresti, A., and Caffo, B. (2000). Simple and effective confidence intervals
  for proportions and differences of proportions result from adding two
  successes and two failures. *The American Statistician*, 54, 280--288.
- Brown, L., and Li, X. (2005). Confidence intervals for two sample binomial
  distribution. *Journal of Statistical Planning and Inference*, 130,
  359--375. https://doi.org/10.1016/j.jspi.2003.09.039
