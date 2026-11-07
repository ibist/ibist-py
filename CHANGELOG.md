# Changelog

## Unreleased

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
