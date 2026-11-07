# Changelog

## Unreleased

- Add `prop_paired_ci()` and `or_paired_ci()` with all R methods and CI sides.
- Add `or_common_ci()` with exact limits and conditional MLEs, supporting
  network and convolution algorithms.
- Add Chan--Zhang exact intervals to `prop_2s_ci()`, including global
  nuisance maximization and disconnected-acceptance searches.
- Compile Wang's shared C++ core with thin native R/Python bindings. Bundle
  generated source and provenance; check synchronization and platform builds.
- Add independent formula/coverage checks and 545 further R reference cases.

- Align CI entry points with R: `side` selects `two.sided` (default),
  `lower`, or `upper`; preserve `alternative` as a compatibility alias and
  reject conflicting values. Expose `CIResult.side` and export it as metadata.
- Add one-sided conditional odds-ratio bounds and preserve repeated methods.
- Compare shared CI arguments, metadata, and limits with R reference cases.

- Add `rr_2x2_ci()` with Katz and Koopman intervals, one-sided bounds,
  zero-event handling, and group metadata.
- Add `rate_1s_ci()`, `rate_2s_ci()`, and `rate_test()` with the R methods
  and correction conventions. Expose one-sided intervals explicitly and
  keep scalar inference targets separate from sample rates and exposures.
- Clip negative Wilson--Hilferty rate limits to zero; record the current R
  behavior and proposed fix in a separate author-review note.
- Add independent formula and likelihood checks plus 459 R reference cases
  for risk-ratio intervals and basic rate functions.

- Add `prop_2s_ci()` with Newcombe, Wald, Agresti--Caffo,
  Miettinen--Nurminen, Mee, and Brown--Li intervals for independent samples.
  Support count vectors, 2x2 tables, one-sided bounds, and applicable
  continuity corrections. Preserve observed estimates and group metadata.
- Validate two-proportion intervals against independent formulas and
  constrained likelihood fits, analytic zero-event limits, and 255 R cases.

- Add `prop_1s_ci()` with Wilson, Wald, Agresti--Coull, Clopper--Pearson,
  mid-p, and Blaker intervals, including correction and one-sided options.
- Extend `CIResult` for vector binomial inputs with explicit input metadata
  and DataFrame indexing, preserving scalar summaries and stored precision.
- Validate binomial intervals against published SAS limits, SciPy, enumerated
  Blaker acceptance sets and coverage, and 358 R comparison cases.

- Add `indep_test()` with Pearson, likelihood-ratio, and Fisher tests,
  including general fixed-margin exact enumeration and seeded Monte Carlo.
- Add `nominal_assoc()` for Phi, contingency coefficient, and Cramer's V.
- Add `or_homogeneity_test()` with Zelen exact, Breslow--Day, Tarone-adjusted
  Breslow--Day, and Woolf tests.
- Preserve DataFrame category labels, full-precision results, and readable
  summaries. Validate with direct formulas, small exact enumerations, a
  published StatXact example, and 260 explicitly exported R comparison cases.

## 0.1.0a1

First alpha release. Public interfaces may change during development.

- Add `or_2x2_ci()` with Wald, Gart, Cornfield, and Baptista--Pike intervals,
  including exact and mid-p conditional options.
- Provide structured confidence interval results with readable summaries
  and pandas DataFrame output.
- Include all seven teaching datasets with category order, labels, units,
  missing values, and provenance from the common R preparation workflow.
- Validate interval calculations against direct formulas, published results,
  SciPy, enumerated coverage cases, and R reference cases.
- Support Python 3.10 and later, with NumPy, SciPy, and pandas as dependencies.

## 0.1.0.dev0

Initial development version published on TestPyPI and verified by installing
the published distribution into a fresh environment.
