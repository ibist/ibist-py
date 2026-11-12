# Production PyPI publishing

The package is in alpha development. It provides teaching datasets,
confidence intervals for proportions, rates, and odds and risk ratios,
association and agreement tests, power calculations, and GLM goodness-of-fit
tests. See the [API index](api-reference.md) and [changelog](../CHANGELOG.md) for
the development scope and release history. Public interfaces may change
during development. Production publication is a manually triggered action.

## Publisher configuration

The maintainer has configured a production PyPI account, a GitHub environment
named `pypi`, and a pending GitHub publisher using these values:

| Field | Value |
| --- | --- |
| Project name | `ibist` |
| Owner | `ibist` |
| Repository | `ibist-py` |
| Workflow filename | `publish-pypi.yml` |
| Environment | `pypi` |

The workflow `.github/workflows/publish-pypi.yml` builds the source archive
and wheel, checks their metadata, installs the wheel with its dependencies,
and runs the tests and installed examples. Only checked distributions are
passed to the publishing job, which uses Trusted Publishing and the explicit
production upload endpoint. No API token or password is needed in secrets.

Both publishers use the shared `tool.cibuildwheel` configuration in
`pyproject.toml`. It builds a CPython 3.12 Linux x86-64 wheel inside the
`manylinux_2_28` container, repairs its native-library dependencies, and
tests the installed wheel there before the runner checks it again.
Only the repaired wheel and source archive enter `dist/` for upload.
Other Python versions and platforms use the source archive and need a
C++ compiler. Broader wheel coverage remains separate release work.

## Publish a release

1. Set the version in `src/ibist/__init__.py`. `pyproject.toml` reads it
   automatically. Update the changelog and review the release scope.
2. Commit and push to `main`. Confirm the regular checks pass on Python
   3.10, 3.12, and 3.14.
3. Open [GitHub Actions](https://github.com/ibist/ibist-py/actions), select
   **Publish to PyPI**, choose **Run workflow**, select `main`, and run it.
4. If GitHub environment approval is configured, review the checks and
   approve the publishing job.
5. Confirm that the version appears on [PyPI](https://pypi.org/project/ibist/),
   then verify installation from the index as shown below.

The workflow publishes only from `main`. An existing upload is reported as
an error; distribution filenames cannot be overwritten. A pending publisher
creates the project on its first successful upload and does not reserve
the name beforehand. Keep the separate
[TestPyPI workflow](publishing.md) for rehearsing future releases.

## Verify the published alpha

After upload, use a fresh environment outside the source checkout:

```sh
python3 -m venv /tmp/ibist-pypi-alpha
source /tmp/ibist-pypi-alpha/bin/activate
python -m pip install "ibist==0.1.0a1"
python -m pip check
```

An explicit version pin permits installation of the alpha. For the latest
available prerelease, use `python -m pip install --pre ibist` instead.
NumPy, SciPy, pandas, and statsmodels are installed automatically.

```python
from importlib.metadata import version
import ibist

assert version("ibist") == ibist.__version__ == "0.1.0a1"
print(ibist.or_2x2_ci([[7, 27], [1, 33]], method="cornfield"))
for name in ibist.datasets.available():
    dataset = ibist.datasets.load(name)
    assert len(dataset.to_frame()) == len(dataset)
```

See the official
[Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
