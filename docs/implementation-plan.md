# Initial implementation plan

The shared
[development principles](https://github.com/ibist/ibist-R/blob/main/CROSS_LANGUAGE_DEVELOPMENT.md)
define the policy. This document tracks the Python implementation sequence.

## Package foundation

The repository uses a `src/ibist/` layout, distribution name `ibist`, and
import name `ibist`. The initial version is `0.1.0.dev0`. Packaging checks
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
test scope, and numerical differences. Additional intervals are the next
implementation work; this first step does not provide the remaining R methods.

## Subsequent stages

1. Complete basic odds-ratio, rate, and paired-data intervals.
2. Add ordinal association, kappa, trend tests, and exact stratified inference.
3. Add power calculations, logistic goodness of fit, and computationally
   demanding paired methods.

Add dependencies and compiled bindings when an implemented feature requires
them. Record any feature or option available in only one language. Check
PyPI name availability again before a release; the 2026-11-06 check found no
project named `ibist`, but did not reserve the name.
