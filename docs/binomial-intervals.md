# Binomial proportion intervals

`prop_1s_ci()` is available in the development checkout, after release
0.1.0a1. It is a native Python implementation and does not require R.
The authoritative [shared specification](https://github.com/ibist/ibist-R/blob/main/BINOMIAL_INTERVALS.md)
defines the statistical methods; this guide documents the Python interface.

```python
ibist.prop_1s_ci(x, n, conf_level=0.95, method="wilson", correct=False,
                midp=False, alternative="two.sided", tol=1e-8)
```

## Inputs and options

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

`alternative="greater"` returns a lower confidence bound and upper limit one;
`alternative="less"` returns an upper bound and lower limit zero. Each uses
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
- `conf_level`, `alternative`, `correct`, and `midp`: the requested settings.

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
lower = ibist.prop_1s_ci(5, 40, alternative="greater")
```

Printing uses three decimal places by default and scientific notation for
tiny values. `summary(digits=...)` changes presentation only. Vector summaries
show counts and estimates alongside interval rows. Existing scalar odds-ratio
summaries and their DataFrame layout are preserved; their `midp` metadata now
also records the flag passed to `or_2x2_ci()`.

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
