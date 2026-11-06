# Initial implementation plan

The shared
[development principles](https://github.com/ibist/ibist-R/blob/main/CROSS_LANGUAGE_DEVELOPMENT.md)
define the policy. This document tracks the Python implementation sequence.

## Package foundation

The repository uses a `src/ibist/` layout, distribution name `ibist`, and
import name `ibist`. The initial version is `0.1.0.dev0`. Packaging checks
build both source and wheel distributions and verify an installed import.
No statistical methods or prepared datasets are included yet.

## First complete implementation

Establish dataset loading and structured result objects, then implement
`or_2x2_ci()` with the agreed R methods and mid-p options. Extend the existing
R `data-raw/` workflow to produce portable data with metadata rather than
duplicating the cleaning logic. Select and document the export format,
metadata schema, and shared validation-case format during this stage.

Check the first method against direct formulas and published reference cases,
then compare its estimates and confidence limits with `ibist::or.2x2.ci()`.
Cover sparse and zero-cell tables and exact versus mid-p behavior. Record the
correspondence between public R and Python interfaces as methods are added.

## Subsequent stages

1. Complete basic odds-ratio, rate, and paired-data intervals.
2. Add ordinal association, kappa, trend tests, and exact stratified inference.
3. Add power calculations, logistic goodness of fit, and computationally
   demanding paired methods.

Add dependencies and compiled bindings when an implemented feature requires
them. Record any feature or option available in only one language. Check
PyPI name availability again before a release; the 2026-11-06 check found no
project named `ibist`, but did not reserve the name.
