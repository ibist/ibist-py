# TestPyPI publishing

The manually triggered workflow `.github/workflows/publish-testpypi.yml`
builds and tests the package, checks the distribution metadata, and passes
the checked wheel and source archive to a separate publishing job. It publishes
only from `main`, using the GitHub environment `testpypi` and Trusted Publishing.
The upload destination is TestPyPI; use the separate
[production workflow](pypi-publishing.md) for PyPI releases.

## Account configuration

The pending GitHub publisher in the
[TestPyPI account settings](https://test.pypi.org/manage/account/publishing/)
must have these exact values:

| Field | Value |
| --- | --- |
| Project name | `ibist` |
| Owner | `ibist` |
| Repository | `ibist-py` |
| Workflow filename | `publish-testpypi.yml` |
| Environment | `testpypi` |

The repository's GitHub environment must also be named `testpypi`. These
account and environment settings have been configured by the maintainer.
The workflow obtains short-lived publishing credentials automatically;
no API token or password is needed in repository secrets.

## Run a trial publication

1. Commit and push the workflow to `main`. Confirm the regular package checks
   pass on Python 3.10, 3.12, and 3.14.
2. Open [GitHub Actions](https://github.com/ibist/ibist-py/actions), select
   **Publish to TestPyPI**, choose **Run workflow**, select `main`, and run it.
3. If environment approval is configured in GitHub, approve the publishing job
   after reviewing the build and test results.
4. Confirm that the version appears on
   [TestPyPI](https://test.pypi.org/project/ibist/).

The first trial used `0.1.0.dev0`. The workflow reads the current version
from `ibist.__version__` through the package metadata. Changed trial builds
need a new version; existing distribution filenames cannot be overwritten.
The workflow deliberately reports an error for an existing upload instead of
silently skipping it. A pending publisher creates the project on the first
successful upload; it does not reserve the project name beforehand.

## Verify installation from TestPyPI

After publication, create a fresh environment outside the source checkout:

```sh
python3 -m venv /tmp/ibist-testpypi
source /tmp/ibist-testpypi/bin/activate
python -m pip install "numpy>=1.24" "scipy>=1.10" "pandas>=2.0"
python -m pip install --index-url https://test.pypi.org/simple/ \
    --no-deps "ibist==0.1.0.dev0"
python -m pip check
```

Dependencies come from regular PyPI because TestPyPI is a separate index.
The final installation command downloads only the named ibist trial from
TestPyPI. Adjust the pinned ibist version when testing a later upload.

```python
from importlib.metadata import version
import ibist

print(version("ibist"))
print(ibist.or_2x2_ci([[7, 27], [1, 33]], method="cornfield"))
print(ibist.datasets.load("fluorosis").to_frame().head())
```

See the official guides for
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [TestPyPI](https://packaging.python.org/en/latest/guides/using-testpypi/).
