# ibist for Python

Python sister package to [ibist for R](https://github.com/ibist/ibist-R),
supporting *Biostatistical Analysis of Proportions and Rates with R* by
Elizabeth Schifano and Jun Yan.

## Status

The released alpha includes all seven teaching datasets and
`or_2x2_ci()` with Wald, Gart, Cornfield, and Baptista--Pike intervals,
including exact and mid-p conditional options. This is an early alpha;
public interfaces may change during development.
The development checkout additionally provides `prop_1s_ci()` for scalar
and vector binomial counts, `prop_2s_ci()` for independent-proportion
differences, plus `indep_test()`, `nominal_assoc()`, and
`or_homogeneity_test()` for contingency tables.
It also provides `rr_2x2_ci()` for risk ratios, `rate_1s_ci()` and
`rate_2s_ci()` for Poisson rate intervals, and `rate_test()` for one- and
two-sample normal-approximation rate tests. See the
[risk-ratio and rate guide](docs/risk-ratios-and-rates.md) for methods,
result metadata, and the documented zero-count WH difference from R.
These additions are unreleased; use the editable development installation
below to try them. See [contingency-table methods](docs/contingency-tables.md)
for options, validation, and exact-computation limits. The
[binomial interval guide](docs/binomial-intervals.md) covers proportion methods,
vector output, and correction options.
The distribution and import names are both `ibist`; the GitHub repository
remains `ibist/ibist-py`.

See the [function reference](docs/api-reference.md) for R counterparts,
shared result conventions, and access to complete help with `help()` or
`python -m pydoc`.

`ordinal_assoc()` provides seven ordinal association measures with
multinomial standard errors, confidence intervals, and tests. See
[ordinal association](docs/ordinal-association.md) for category ordering,
Pearson scores, Spearman midranks, and result fields.

`assoc_test()` combines nominal Pearson, ordinal multinomial, and CMH
association inference. Single measures return `TestResult`; multiple measures
return an ordered collection with a combined summary identifying shared
tests. See [association tests](docs/association-tests.md).

`agreement_test()` provides two-rater Cohen's kappa inference with named or
custom weights, confidence intervals, and FCE/delta test-variance choices.
See [agreement tests](docs/agreement-tests.md) for result fields and examples.

`trend_test()` provides asymptotic and exact Cochran--Armitage inference.
`power_p1s_test()` and `power_p2s_test()` compute proportion power and solve
sample sizes or other design parameters. See
[trend and power](docs/trend-and-power.md) for exact rules and allocation.
`power_r1s_test()` and `power_r2s_test()` provide normal-approximation
Poisson rate power with explicit follow-up and unequal group allocation.

`asymcheck()` provides pre- and post-inference approximation diagnostics.
Supported tests and intervals also accept `asymcheck=True` to attach them.
See [approximation diagnostics](docs/asymcheck.md) for methods and criteria.

`glm_gof(fit)` checks binomial and Poisson GLMs, with additional
Hosmer-Lemeshow, Osius-Rojek, and Stukel methods for binary logistic models.
It accepts fitted statsmodels GLM results and supports shared diagnostics.
See [GLM goodness of fit](docs/glm-gof.md) for scope and published checks.

## Installation

After publication, install the first alpha from PyPI with:

```sh
python -m pip install "ibist==0.1.0a1"
```

NumPy, SciPy, and pandas are installed automatically. See the
[changelog](https://github.com/ibist/ibist-py/blob/main/CHANGELOG.md) for the release scope.

## Development

Use Python 3.10 or later and an isolated environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The package can then be imported without an R installation:

```python
import ibist

print(ibist.__version__)

result = ibist.or_2x2_ci([[7, 27], [1, 33]],
                         method=["wald", "gart", "cornfield", "bp"])
print(result)
print(result.to_frame())

midp = ibist.or_2x2_ci([[7, 27], [1, 33]], method=["cornfield", "bp"], midp=True)
print(midp)

fluorosis = ibist.datasets.load("fluorosis")
print(fluorosis)
frame = fluorosis.to_frame()
```

The development checkout also supports:

```python
result = ibist.prop_1s_ci(5, 40, method=["wilson", "clopper-pearson", "blaker"])
print(result)
print(ibist.prop_1s_ci([0, 5, 40], 40, method=["wilson", "wald"]).to_frame())
print(ibist.prop_2s_ci([9, 3], [10, 10],
                       method=["newcombe", "mee", "brown-li"]))

table = [[109, 68, 14], [72, 79, 37], [9, 42, 139]]
print(ibist.indep_test(table, method="lr"))
print(ibist.nominal_assoc(table).to_frame())

small = [[7, 77], [2, 90]]
print(ibist.indep_test(small, method="fisher", alternative="greater"))

strata = [[[30, 20], [20, 30]], [[20, 30], [30, 20]]]
print(ibist.or_homogeneity_test(strata, method="breslow-day"))
```

NumPy, SciPy, and pandas are runtime dependencies, so a standard installation
supports all examples and DataFrame output. The `dev` extra provides testing
and packaging tools. Printed
summaries round for readability; stored values retain full precision.

To build and inspect the distribution:

```sh
python -m pip install build twine
python -m build
python -m twine check dist/*
python -m pytest -q
```

The shared
[R and Python development principles](https://github.com/ibist/ibist-R/blob/main/CROSS_LANGUAGE_DEVELOPMENT.md)
govern statistical specifications, native interfaces, common data preparation,
and validation. R is initially the reference implementation; future features
may start in either language. Dataset preparation remains in the R package's
`data-raw/` workflow; its export step reads the prepared R datasets and produces
the portable files bundled here.

See the [implementation plan](https://github.com/ibist/ibist-py/blob/main/docs/implementation-plan.md) for the first stages.
The [dataset format and preparation](https://github.com/ibist/ibist-py/blob/main/docs/datasets.md) and
[odds-ratio interface and validation](https://github.com/ibist/ibist-py/blob/main/docs/odds-ratio.md) describe this implementation.
See [TestPyPI publishing](https://github.com/ibist/ibist-py/blob/main/docs/publishing.md) for the manual release workflow
and installation checks.
The separate [PyPI publishing guide](https://github.com/ibist/ibist-py/blob/main/docs/pypi-publishing.md)
describes production releases.

## License

Copyright (C) 2026 Elizabeth Schifano and Jun Yan.

GNU General Public License, version 3 or later; see [LICENSE](https://github.com/ibist/ibist-py/blob/main/LICENSE).

CI functions share R's `side` argument: `"two.sided"` (default), `"lower"`,
or `"upper"`. Omitted `side` or `None` selects two-sided limits. For example,
`ibist.rate_1s_ci(5, exposure=10, side="lower")` returns a lower bound.
`alternative` remains available as a compatibility alias; conflicting
arguments are rejected. Hypothesis tests retain `alternative`.

Paired and stratified intervals are available as `prop_paired_ci()`,
`or_paired_ci()`, and `or_common_ci()`. `prop_2s_ci(method="chan-zhang")`
provides exact unconditional difference intervals. All use the shared `side`
convention. Wang runs the C++ core shared with R through a native binding;
source installs need a C++ compiler, and R is not required. See
[paired and exact intervals](docs/paired-and-exact-intervals.md).

`or_common_test(tables, theta=1, alternative="two.sided")` tests a specified
common null odds ratio, using the same fixed-margin distribution as
`or_common_ci()`. Its two-sided p-value doubles the smaller inclusive tail;
see [paired and exact intervals](docs/paired-and-exact-intervals.md).
