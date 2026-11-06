# ibist for Python

Python sister package to [ibist for R](https://github.com/ibist/ibist-R),
supporting *Biostatistical Analysis of Proportions and Rates with R* by
Elizabeth Schifano and Jun Yan.

## Status

This repository establishes the package structure. Statistical functions and
datasets have not yet been implemented. No release has been published on PyPI.
The distribution and import names are both `ibist`; the GitHub repository
remains `ibist/ibist-py`. PyPI had no project named `ibist` when checked on
2026-11-06; the name has not been reserved.

## Development

Use Python 3.10 or later and an isolated environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The package can then be imported without an R installation:

```python
import ibist

print(ibist.__version__)
```

To build and inspect the distribution:

```sh
python -m pip install build twine
python -m build
python -m twine check dist/*
```

The shared
[R and Python development principles](https://github.com/ibist/ibist-R/blob/main/CROSS_LANGUAGE_DEVELOPMENT.md)
govern statistical specifications, native interfaces, common data preparation,
and validation. R is initially the reference implementation; future features
may start in either language. Dataset preparation remains in the R package's
`data-raw/` workflow and will generate distribution files for both packages.

See the [implementation plan](docs/implementation-plan.md) for the first stages.

## License

Copyright (C) 2026 Elizabeth Schifano and Jun Yan.

GNU General Public License, version 3 or later; see [LICENSE](LICENSE).
