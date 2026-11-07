"""Load teaching datasets exported by the R data-preparation workflow."""

from copy import deepcopy
from dataclasses import dataclass
import gzip
from importlib.resources import files
import json

import numpy as np
import pandas as pd


@dataclass(repr=False)
class Dataset:
    """Column arrays and metadata, including categories, labels, and provenance.

    Integer columns with missing values contain Python integers and None;
    numeric columns use NaN for missing values. Categorical values retain
    their R factor labels. ``to_frame()`` restores categorical order and
    nullable integer types, and stores the full metadata in ``frame.attrs``.
    Each load returns independent arrays and metadata.
    """

    data: dict[str, np.ndarray]
    metadata: dict

    @property
    def name(self) -> str:
        return self.metadata["name"]

    def __len__(self) -> int:
        return self.metadata["nrows"]

    def __repr__(self) -> str:
        return (f"Dataset(name={self.name!r}, nrows={len(self)}, "
                f"columns={tuple(self.data)!r})")

    def to_frame(self):
        """Return a new pandas DataFrame with column types and metadata."""
        columns = {}
        for info in self.metadata["columns"]:
            name = info["name"]
            values = self.data[name]
            if info["type"] == "category":
                levels = info["levels"]
                columns[name] = pd.Categorical(values, categories=levels,
                                               ordered=info["ordered"])
            elif info["type"] == "integer" and info["missing_count"]:
                columns[name] = pd.array(values, dtype="Int64")
            else:
                columns[name] = values.copy()
        frame = pd.DataFrame(columns)
        frame.attrs["ibist"] = deepcopy(self.metadata)
        return frame


def available() -> tuple[str, ...]:
    """List the bundled teaching datasets.

    Returns
    -------
    tuple of str
        Dataset names accepted by load(), in the bundled index order.

    Notes
    -----
    Reads the packaged index locally without R or network access. The
    datasets are exported from the R data-preparation workflow.

    See Also
    --------
    load : Load the arrays and metadata for one dataset.

    Examples
    --------
    >>> "rds" in available()
    True
    """
    resource = files("ibist").joinpath("_data", "index.json")
    return tuple(json.loads(resource.read_text(encoding="utf-8"))["datasets"])


def load(name: str) -> Dataset:
    """Load a named dataset without R or a network connection.

    Parameters
    ----------
    name : str
        One of the names returned by available(). Unknown names raise
        ValueError; files are not downloaded on demand.

    Returns
    -------
    Dataset
        data maps variable names to NumPy arrays. metadata preserves
        variable descriptions, units, factor levels, ordered status,
        numeric value labels, missing-value counts, and source hashes.
        Each load returns independent arrays and metadata.

    Notes
    -----
    Data are bundled exports of the R data-preparation workflow. Numeric
    missing values become NaN; missing integer values become None in object
    arrays. Categorical arrays retain factor labels. Dataset.to_frame()
    restores pandas categories and nullable integers, and places metadata
    in frame.attrs["ibist"]. See docs/datasets.md for provenance, source
    references, and the shared preparation workflow.

    See Also
    --------
    available : List dataset names.
    Dataset.to_frame : Convert to a pandas DataFrame with types and metadata.

    Examples
    --------
    >>> dataset = load("rds")
    >>> dataset.name
    'rds'
    >>> dataset.to_frame().attrs["ibist"]["name"]
    'rds'
    """
    if not isinstance(name, str) or name not in available():
        message = f"Unknown dataset {name!r}; choose from {available()}."
        raise ValueError(message)
    root = files("ibist").joinpath("_data")
    metadata_path = root.joinpath(f"{name}.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with root.joinpath(f"{name}.json.gz").open("rb") as raw:
        with gzip.GzipFile(fileobj=raw) as compressed:
            values = json.load(compressed)
    data = {}
    for info in metadata["columns"]:
        column = values[info["name"]]
        if info["type"] == "number":
            dtype = float
        elif info["type"] == "integer" and not info["missing_count"]:
            dtype = np.int64
        else:
            dtype = object
        data[info["name"]] = np.array(column, dtype=dtype)
    return Dataset(data, metadata)
