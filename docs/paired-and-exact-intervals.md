# Paired and exact confidence intervals

These interfaces follow the R package. All default to 95% two-sided
intervals. `side="lower"` or `side="upper"` requests a one-sided bound at
the full error probability. Omitted `side` or `None` selects two-sided limits.
`alternative` remains a compatibility alias; conflicting values are errors.
Results use `CIResult` with scalar estimates, interval rows, and `side`
metadata. Method requests preserve order and repetitions.

## Paired data

```python
from ibist import or_paired_ci, prop_paired_ci

or_paired_ci([[51, 30], [10, 41]], method=["wald", "wilson", "blaker"])
prop_paired_ci(8, 25, 180, method=["score", "waldcc", "agresti-min"])
prop_paired_ci(3, 0, 4, method="wang", precision=0.0001)
```

`or_paired_ci()` estimates `b/c` from off-diagonal counts `b=x[0,1]` and
`c=x[1,0]`. Methods are `wald` (default), `wilson`, `clopper-pearson`, `midp`,
and `blaker`. Wald requires positive discordant counts; other methods allow
one zero count. With no discordant pairs, warn and return an undefined
estimate with `[0, inf]`. Concordant pairs do not affect conditional ORs.

`prop_paired_ci(b, c, n)` estimates `(b-c)/n` and requires positive `n` with
`b+c <= n`. Methods are `score` (default, Tango), `wald`, `waldcc`,
`agresti-min`, and `wang`. `tol=1e-8` controls score roots. Boundary fits are
included in constrained likelihood calculations. Python uses tighter fits
than R's existing optimizer; R comparison tolerance is `3e-4` absolute.
Waldcc adds `1/n`; Agresti--Min adjusts counts by 0.5 and total pairs by 2.

Wang uses the shared native C++ implementation with R's `precision=1e-5`,
`grid_one=30`, and `grid_two=20` defaults. Limits are rounded as in R.
Finite nuisance grids approximate the supremum and do not certify nominal
coverage. The paired state space grows quadratically; Python rejects more
than one million states. Increase grids for more nuisance search resolution.
The implementation preserves R's initial fixed-grid calculation as well.

## Independent exact differences

```python
from ibist import prop_2s_ci

prop_2s_ci([9, 3], [10, 10], method=["newcombe", "chan-zhang"])
```

Chan--Zhang inverts inclusive unconditional score tails. For each candidate
difference, nuisance probabilities are maximized globally. Bounds on score
orderings enclose the outer acceptance set, including disconnected regions.
This construction uses neither a monotone p-value assumption nor a fixed
nuisance grid. `nuisance_tol=1e-10` controls ambiguous probability regions,
which are retained conservatively; `tol=1e-8` controls outward endpoint
accuracy. At most `(n1+1)*(n2+1) <= 1,000,000` tables are enumerated. Even
moderate samples can be expensive; no approximate method is substituted.
Continuity correction affects only applicable methods in a mixed request.

## Common odds ratios

```python
from ibist import or_common_ci

tables = [[[7, 3], [2, 8]], [[4, 6], [1, 9]]]
or_common_ci(tables, side="lower")
```

Input is a list of tables or a NumPy array of shape `(2,2,K)`. The estimate
is the conditional MLE. Exact inclusive tails of the sum of upper-left
counts give the limits. `algorithm="network"` uses log-space dynamic
programming; `"convolution"` combines direct probabilities and can underflow
in extreme tails. Deterministic strata are retained. For a singleton total
support, R's convention is estimate zero and interval `[0, inf]`; the
parameter is unidentified. Python limits the total support to one million
points. This is inference about a common OR, separate from homogeneity tests.

## Native source and builds

The authoritative source and synchronization policy are in
[the R development principles][policy].
Python bundles the generated header and provenance in `src/ibist/_native/`.
From `ibist-R/`, regenerate with:

```sh
Rscript --vanilla data-raw/export-python-wang-core.R
Rscript --vanilla data-raw/export-python-wang-core.R --check
```

From `ibist-py/`, verify the copy with:

```sh
python tools/check_shared_core.py --r-source ../ibist-R
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
```

Wheels contain the compiled Python binding and are specific to their Python
version and platform. Source installations require a C++ compiler and Python
development headers. They require neither R nor Rcpp. The current publishing
workflows build Linux Python 3.12 binaries plus a source distribution; other
platforms/versions may need source compilation. CI tests Linux, macOS, and
Windows at Python 3.10, 3.12, and 3.14. Land the R core before dependent Python
CI changes. Header files are included in source distributions, as required
by [setuptools extension documentation][extensions].

## Validation and references

Checks include direct multinomial and binomial formulas, exact small-space
coverage, constrained likelihood fits, conditional hypergeometric tail and
MLE equations, analytic boundaries, and a disconnected Chan--Zhang acceptance
case. An R exporter provides 545 additional reference cases. Wang limits
match exactly at the recorded precision. Paired score fits allow `3e-4`
absolute error against R's coarser optimizer; other reference limits use
`2e-7` relative or `2e-8` absolute tolerance.

References follow the R manuals: Tango (1998), Statistics in Medicine 17,
891--908; Agresti and Min (2005), Statistics in Medicine 24, 729--740; Wang
(2012), Statistics & Probability Letters 82, 1623--1628; Fagerland, Lydersen
and Laake (2014), Statistics in Medicine 33, 2850--2875; Blaker (2000),
Canadian Journal of Statistics 28, 783--798; Chan and Zhang (1999), Biometrics
55, 1202--1209; Mehta, Patel and Gray (1985), JASA 80, 969--973. These methods
have distinct sampling assumptions and interval constructions.

[policy]:
  https://github.com/ibist/ibist-R/blob/main/CROSS_LANGUAGE_DEVELOPMENT.md
[extensions]:
  https://setuptools.pypa.io/en/latest/userguide/ext_modules.html
