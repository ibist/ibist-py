# Initial implementation plan

The shared
[development principles](https://github.com/ibist/ibist-R/blob/main/CROSS_LANGUAGE_DEVELOPMENT.md)
define the policy. This document tracks the Python implementation sequence.

## Package foundation

The repository uses a `src/ibist/` layout, distribution name `ibist`, and
import name `ibist`. The initial TestPyPI version was `0.1.0.dev0`; the first
production alpha is prepared as `0.1.0a1`. Packaging checks
build both source and wheel distributions and verify an installed import.
All seven prepared R datasets are bundled with metadata. The first method,
`or_2x2_ci()`, returns structured results and supports all four R interval
methods, including exact and mid-p conditional intervals.
NumPy, SciPy, and pandas are regular dependencies, so introductory examples
and DataFrame output work with a standard installation.

## First complete implementation

Dataset loading, structured interval results, and `or_2x2_ci()` are implemented.
The R `data-raw/export-python.R` step exports the existing prepared datasets
as compressed JSON column arrays with JSON metadata, without duplicating
cleaning logic. See [dataset preparation](datasets.md) for the schema and
provenance, including raw inputs missing from the current R checkout.

Validation includes direct formulas, Fagerland's published example, SciPy
conditional intervals, enumerated coverage, and 60 R comparison cases. See
[odds-ratio validation](odds-ratio.md) for the interface correspondence,
test scope, and numerical differences. The development checkout also
implements contingency-table independence, nominal association, and odds-ratio
homogeneity tests; see [contingency-table methods](contingency-tables.md).
`prop_1s_ci()` now implements the one-proportion interval family, including
vector inputs and multiple methods; see [binomial intervals](binomial-intervals.md).
`prop_2s_ci()` now provides six approximate independent-proportion interval
methods, including score inversion and one-sided bounds. `rr_2x2_ci()` now
provides Katz and Koopman risk-ratio intervals. Basic Poisson-rate intervals
and normal-approximation tests are available through `rate_1s_ci()`,
`rate_2s_ci()`, and `rate_test()`; see
[risk ratios and rates](risk-ratios-and-rates.md). Chan--Zhang exact
independent-difference intervals, paired intervals, and common odds-ratio
intervals are now implemented. Wang uses the shared C++ core with native
Python bindings. `or_common_test()` completes the common-OR test/CI pair,
including non-unit null odds ratios and all alternatives. See
[paired and exact methods](paired-and-exact-intervals.md).

## Subsequent stages

1. Complete basic odds-ratio, rate, and paired-data intervals.
2. Ordinal association is implemented; add kappa and trend tests next.
3. Add power calculations, logistic goodness of fit, and computationally
   demanding paired methods.

Add dependencies and compiled bindings when an implemented feature requires
them. Record any feature or option available in only one language. Check
PyPI name availability again before a release; the 2026-11-06 check found no
project named `ibist`, but did not reserve the name.
