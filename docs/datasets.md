# Teaching datasets and common preparation

`ibist.datasets.available()` lists the bundled datasets. Load one with
`ibist.datasets.load("fluorosis")`. The returned `Dataset` provides column
arrays in `.data` and definitions and provenance in `.metadata`. Every load
returns independent objects. Use `.to_frame()` for a pandas DataFrame;
pandas is included in a standard installation. The frame also retains
metadata in `frame.attrs["ibist"]`.

| Dataset | Rows | Columns |
| --- | ---: | ---: |
| `asah` | 309 | 4 |
| `depress` | 1,730 | 9 |
| `fluorosis` | 24 | 5 |
| `midbp` | 16 | 4 |
| `mihdl` | 24 | 4 |
| `nrs` | 91,795 | 6 |
| `rds` | 16 | 4 |

## Preparation and regeneration

The R package's `data-raw/` scripts remain the common cleaning workflow.
After changing a dataset, run its existing preparation script to regenerate
`ibist-R/data/<name>.rda`, then run from the R package root:

```sh
Rscript data-raw/export-python.R
```

The exporter requires R and jsonlite during development. It reads the prepared
R objects, extracts descriptions from their Rd documentation, and writes
`ibist-py/src/ibist/_data/`. An optional first argument selects a different
output directory. It does not modify the R data files. Commit R preparation
and exporter changes first, regenerate exports from that revision, then review
and commit the Python files so their provenance identifies the committed source.
Installed Python users need neither R nor a network connection to load data.

The current checkout lacks `data-raw/nrs.csv`, `data-raw/rds.tab`, and the
depression study's ZIP/SPSS inputs. These datasets are exported from the
existing prepared R objects. Their metadata explicitly record the missing
inputs; rebuilding those R objects from raw data requires recovering them.
The ASAH tables and fluorosis counts are embedded in their preparation scripts;
the DBP and HDL CSV inputs are present. No observations are reconstructed or
cleaned again by Python.

## Portable schema, version 1

`index.json` lists the dataset names. Each dataset has two files:

- `<name>.json.gz`: a gzip-compressed JSON object mapping column names to
  arrays. Missing values are JSON null; factor values are their string labels.
- `<name>.json`: row and column counts, descriptions, source references, and
  column metadata. Columns record storage type, R class, descriptions, labels,
  units where defined, factor levels and ordered status, missing counts, and
  numeric value labels. No reference category is imposed beyond the original
  factor level order; models may choose a different reference explicitly.

Provenance includes the R repository revision, an indicator for uncommitted
preparation changes, preparation and documentation paths, raw-input availability,
and MD5 content hashes for the prepared data, scripts, documentation, and
compressed export. These hashes identify content for reproducibility;
they are not security signatures. Generated files must not be edited by hand.

The exporter validates every value and missing position by reading the JSON
back into R with zero numerical tolerance. Python checks the export hashes,
dimensions, categories, missing counts, and selected published tables. Numeric
labels such as the depression dataset's gender coding remain numeric with a
label mapping; they are not silently recoded to factors. Integer columns with
missing values become pandas nullable `Int64`, while factors become categorical
columns retaining the original level order and ordered status.
