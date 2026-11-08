# Function reference

Function docstrings are the primary API reference. They describe arguments
and defaults, return fields, statistical methods, boundary cases, related
functions, references, and executable examples. Read them in Python with:

```python
import ibist

help(ibist.or_common_test)
help(ibist.prop_1s_ci)
help(ibist.datasets.load)
```

From a shell, use `python -m pydoc ibist.or_common_test`. These commands
describe the installed version; consult the changelog for release scope.

Docstrings follow [PEP 257][pep257] for general conventions and
[numpydoc][numpydoc] for scientific function documentation. Statistical
functions use Parameters, Returns, Raises, See Also, Notes, References,
and Examples, with Warns where relevant. Dataset helpers use the applicable
sections. The statistical content follows the corresponding R help pages.

## Statistical functions and R counterparts

Python names use underscores where R names use dots. Method names and
statistical defaults follow R; intentional differences are documented in
each function and its guide.

| Python function | R function | Guide |
| --- | --- | --- |
| `prop_1s_ci()` | `prop.1s.ci()` | [Binomial][binomial] |
| `prop_2s_ci()` | `prop.2s.ci()` | [Binomial][binomial] |
| `prop_paired_ci()` | `prop.paired.ci()` | [Paired and exact][paired] |
| `or_2x2_ci()` | `or.2x2.ci()` | [Odds ratios][odds] |
| `or_paired_ci()` | `or.paired.ci()` | [Paired and exact][paired] |
| `or_common_ci()` | `or.common.ci()` | [Paired and exact][paired] |
| `or_common_test()` | `or.common.test()` | [Paired and exact][paired] |
| `or_homogeneity_test()` | `or.homogeneity.test()` | [Tables][tables] |
| `rr_2x2_ci()` | `rr.2x2.ci()` | [Risk ratios and rates][rates] |
| `rate_1s_ci()` | `rate.1s.ci()` | [Risk ratios and rates][rates] |
| `rate_2s_ci()` | `rate.2s.ci()` | [Risk ratios and rates][rates] |
| `rate_test()` | `rate.test()` | [Risk ratios and rates][rates] |
| `indep_test()` | `indep.test()` | [Tables][tables] |
| `nominal_assoc()` | `nominal.assoc()` | [Tables][tables] |
| `ordinal_assoc()` | `ordinal.assoc()` | [Ordinal association][ordinal] |
| `assoc_test()` | `assoc.test()` | [Association tests][association] |
| `agreement_test()` | `agreement.test()` | [Agreement tests][agreement] |
| `trend_test()` | `trend.test()` | [Trend and power][design] |
| `power_p1s_test()` | `power.p1s.test()` | [Trend and power][design] |
| `power_p2s_test()` | `power.p2s.test()` | [Trend and power][design] |

## Shared conventions

CI functions return `CIResult`, corresponding to R's `ibist_ci`. Omitted
`side` selects `"two.sided"`; `"lower"` and `"upper"` request one-sided
bounds. `alternative` remains a compatibility alias: `"greater"` selects
`"lower"`, and `"less"` selects `"upper"`. Conflicting values are errors.
Hypothesis tests use `alternative` to specify the alternative hypothesis.

Tests return `TestResult`, corresponding to R's `htest`; nominal and ordinal
association return `NominalAssociationResult` and `OrdinalAssociationResult`.
`assoc_test()` returns one `TestResult` or an ordered `AssociationTestResults`
collection for multiple measures. Dotted fields become snake_case, such
as `p.value` to `p_value` and `conf.int` to `conf_int`. Result fields retain
full precision. Use `summary()` for display and `to_frame()` where provided
for pandas output. Function-specific Returns sections describe the fields
and any differences from R.

Power calculations return `PowerResult`, corresponding to R's `power.htest`,
with resolved design parameters and full-precision power values.

## Datasets

`ibist.datasets.available()` lists the bundled datasets.
`ibist.datasets.load(name)` returns independent arrays and metadata;
`Dataset.to_frame()` creates a pandas DataFrame. The
[dataset guide](datasets.md) explains variable labels, categories, missing
values, and the common R export workflow.

[binomial]: binomial-intervals.md
[paired]: paired-and-exact-intervals.md
[odds]: odds-ratio.md
[rates]: risk-ratios-and-rates.md
[tables]: contingency-tables.md
[ordinal]: ordinal-association.md
[association]: association-tests.md
[agreement]: agreement-tests.md
[design]: trend-and-power.md
[pep257]: https://peps.python.org/pep-0257/
[numpydoc]: https://numpydoc.readthedocs.io/en/latest/format.html
