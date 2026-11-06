# Python package instructions

This repository contains the native Python sister package to `ibist-R`.
Inspect Git status before editing and preserve unrelated changes.

## Shared development policy

Read the authoritative
[R and Python development principles](https://github.com/ibist/ibist-R/blob/main/CROSS_LANGUAGE_DEVELOPMENT.md)
before implementing methods or exporting datasets. In the shared workspace,
the same document is `../ibist-R/CROSS_LANGUAGE_DEVELOPMENT.md`. Keep policy
changes in that document rather than maintaining a second copy here.

Use the R `data-raw/` workflow as the common data preparation source. Preserve
variable definitions, labels, units, category order, and missing values in
portable exports and metadata. Python users must not need R at runtime.

## Python conventions

- Use snake_case for functions and arguments and four spaces for indentation.
- Use spaces after commas and around operators. Wrap at natural boundaries;
  avoid one argument per line when a compact call is readable.
- Keep statistical definitions and defaults aligned with the shared
  specification. Document intentional differences from R.
- Keep full numerical precision internally and round only for presentation.
- Validate methods independently against formulas, published results, or
  established software with matching assumptions, then compare with R.
- Add dependencies when required by implemented functionality. NumPy and
  SciPy are the planned numerical foundations.

## Changes and checks

Keep changes focused and preserve unrelated line wrapping. Verify affected
methods and package builds as appropriate. Do not stage or commit unless
asked; follow the repository's commit-message style.
