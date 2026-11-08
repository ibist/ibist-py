# Changelog

## Unreleased

- Add `glm_gof()` matching R's `glm.gof()` for binomial/Poisson Pearson
  and deviance tests and logistic Hosmer-Lemeshow, Osius-Rojek, and Stukel
  methods. Add ordered GLMGOFResults and pre/attached/post diagnostics.
  Require statsmodels for fitted GLM objects; validate published SAS
  examples, independent formulas, and eight representative R cases.

- Add shared `asymcheck()` pre/post diagnostics and optional attachment to
  independence tests and one-rate intervals. Add stratified Mantel-Haenszel
  screening, preserve the legacy R wrapper, and compare 14 R cases.

- Add one-/two-sample Poisson rate power in step with the new R interfaces,
  including per-subject exposure, unequal allocation, solve-one-missing
  designs, optional continuity correction, and 19 R comparison cases.
  Verify four published Minitab powers and two rounded sample sizes in
  both packages; record matching settings and validation scope.

- Add `trend_test()` and one-/two-sample proportion power functions matching
  R's new trend entry point and existing power interfaces. Add `PowerResult`
  following R's `power.htest`, and fix exact sample-size minimality in R
  before translating it. Use focused formula checks and 30 R references.

- Add `agreement_test()` matching R's `agreement.test()` for unweighted and
  weighted Cohen's kappa, FCE/delta test variances, and Wald intervals.
  Extend `TestResult` with copied agreement weights and kappa metadata.
  Validate independent variances, the SAS example, and 60 R cases.

- Add `assoc_test()` matching R's `assoc.test()`, with nominal Pearson,
  ordinal multinomial, and CMH methods. Multiple measures return an ordered
  `AssociationTestResults` collection identifying shared, unadjusted tests.
  Extend `TestResult` with optional association metadata and validate
  independent formulas plus 274 R comparison calls.

- Add `ordinal_assoc()` with all seven R measures, multinomial standard
  errors, Wald confidence intervals, tests, score handling, and a native
  `OrdinalAssociationResult`. Check independent formulas, the published SAS
  example, and 222 R comparison cases.

- Standardize public statistical docstrings with parameters, return fields,
  notes, related functions, references, and executable examples. Expand
  dataset-loading help and add an API index mapping Python functions to R.

- Add `or_common_test()` with R's common-null odds ratio, inclusive one-sided
  tails, doubled-smaller-tail two-sided p-values, and conditional MLE.
  Validate using direct enumeration, exact null rejection probabilities,
  primary-paper examples, tiny tails, and 456 R comparison cases.

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
