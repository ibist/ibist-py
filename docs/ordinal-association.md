# Ordinal association

`ordinal_assoc()` is the Python counterpart of R's `ordinal.assoc()`.
It computes seven measures by default: `gamma`, `tau-b`, `tau-c`,
`somers-c|r`, `somers-r|c`, `pearson`, and `spearman`.

```python
from ibist import ordinal_assoc

table = [[12, 3, 1], [4, 9, 2], [1, 5, 13]]
result = ordinal_assoc(table)
print(result)
print(result.to_frame())

selected = ordinal_assoc(table, measure=["tau-b", "somers-c|r", "spearman"],
                         alternative="greater", conf_level=0.9)
```

## Inputs and ordering

Rows and columns must already run from the lowest to highest category.
Nothing is sorted automatically, including ordered pandas category labels.
Counts are nonnegative integers, and the total cannot exceed `2**53-1`.
Empty rows and columns are dropped; at least two nonempty categories on
each axis must remain. This follows R and differs from `nominal_assoc()`,
which rejects empty margins.

Pearson scores default to numeric DataFrame labels if every retained label
on that axis is numeric; otherwise they are consecutive category positions.
Array input uses positions after empty categories are removed. Numeric
labels such as `1, 2, 10` are used as supplied. A DataFrame's numeric index
is treated as labels, including any gaps after empty categories drop.

Supply `row_scores` and/or `col_scores` to choose Pearson scores explicitly.
Their lengths refer to the original table, before empty margins drop. Scores
need not increase, but must be finite and have at least two distinct values;
the retained scores must have positive marginal variance. Score validation
also applies when only other measures are requested, matching R. Scores
affect Pearson only; Spearman always uses marginal midranks for tied counts.

## Results and inference

`OrdinalAssociationResult` corresponds to R's `ibist_ordinal_assoc`:

| Python field | R field | Meaning |
| --- | --- | --- |
| `estimate` | `estimate` | Named association estimates |
| `std_error` | `std.error` | Estimation standard errors |
| `conf_int` | `conf.int` | Lower/upper confidence limits |
| `statistic` | `statistic` | Signed asymptotic test statistics |
| `p_value` | `p.value` | Tests of zero association |
| `S`, `var_S` | `S`, `var.S` | Concordance contrast and its null variance |
| `concordant`, `discordant` | Same names | Unordered pair counts |
| `measures` | `measure` | Selected names in requested order |

Named vectors become pandas Series; intervals are a DataFrame with columns
`lower` and `upper`. Requested order and repeated measures are preserved.
The result also records `n`, `conf_level`, `alternative`, and `method`.
`to_frame()` returns a fresh combined table; `summary(digits=3)` formats
without changing stored values and retains small p-values in scientific
notation. Pair counts and `S` are exact Python integers.

The five concordance-based measures share the test `S/sqrt(var_S)`.
Pearson and Spearman use their own multinomial null variances. These tests
are distinct from the usual continuous-normal Pearson test, SciPy's rank
test defaults, and the CMH linear-by-linear test. Somers C|R treats rows as
explanatory and columns as response; R|C reverses these roles.

Intervals use separate multinomial estimation variances. Two-sided Wald
intervals are untruncated and can exceed `[-1, 1]`. One-sided intervals
clip the finite bound to that range and set the other endpoint to 1 for
`greater` or -1 for `less`. The default is `alternative="two.sided"` and
`conf_level=0.95`. Because this function includes tests, `alternative`
specifies their hypothesis and also directs intervals, following R;
there is no separate `side` argument here.

Intervals need not invert the reported tests. Zero null variance gives
missing statistics and p-values, including some perfect-association tables.
Python computes variances from centered multinomial influences to reduce
cancellation. At perfect association, R can report a standard error near
`1e-9` when the theoretical value is zero; boundary comparisons allow this
rounding difference. Rescaling category scores internally also avoids
overflow and underflow without changing their statistical meaning.

## Validation and references

Independent checks enumerate observation pairs, compare point estimates
with expanded-data correlations and Somers' D, and differentiate the
estimands numerically under multinomial sampling. Checks cover marginal
rank uncertainty, transposition, reversal, replication, score choices,
empty margins, invalid inputs, large counts, and very small p-values.

The seven estimates, standard errors, and intervals reproduce
[SAS Example 3.8, Output 3.8.3][pain] to its four published decimal places.
R exports supply 222 additional cases across all three alternatives,
including random tables, numeric labels, explicit scores, empty margins,
duplicate requests, and perfect association. P-values use relative
tolerance `2e-9` without an absolute allowance. Other fields use that
relative tolerance and `1e-11` absolute tolerance, except near-zero standard
errors (`1e-8` absolute) and their affected intervals (`3e-8` absolute).

References match R's help: Goodman and Kruskal (1954), Kendall (1970),
Somers (1962), Stuart (1953), and Brown and Benedetti (1977).
[Brown and Benedetti][brown] describe multinomial asymptotic tests for
these ordinal association measures. The function docstring lists the
bibliographic details, and the [SAS association reference][sas] gives
the formulas and inference conventions.

[pain]:
  https://documentation.sas.com/doc/en/procstat/9.4/procstat_freq_examples08.htm
[brown]: https://doi.org/10.1080/01621459.1977.10480995
[sas]:
  https://documentation.sas.com/doc/en/procstat/9.4/procstat_freq_details22.htm
