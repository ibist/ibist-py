# ibist for Python

Python sister package to [ibist for R](https://github.com/ibist/ibist-R),
supporting *Biostatistical Analysis of Proportions and Rates with R* by
Elizabeth Schifano and Jun Yan.

## Status

The first implementation includes all seven teaching datasets and
`or_2x2_ci()` with Wald, Gart, Cornfield, and Baptista--Pike intervals,
including exact and mid-p conditional options. No release has been published
on PyPI.
The distribution and import names are both `ibist`; the GitHub repository
remains `ibist/ibist-py`. PyPI had no project named `ibist` when checked on
2026-11-06; the name has not been reserved.

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

See the [implementation plan](docs/implementation-plan.md) for the first stages.
The [dataset format and preparation](docs/datasets.md) and
[odds-ratio interface and validation](docs/odds-ratio.md) describe this implementation.

## License

Copyright (C) 2026 Elizabeth Schifano and Jun Yan.

GNU General Public License, version 3 or later; see [LICENSE](LICENSE).
