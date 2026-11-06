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
    """Return the names of the seven bundled teaching datasets."""
    resource = files("ibist").joinpath("_data", "index.json")
    return tuple(json.loads(resource.read_text(encoding="utf-8"))["datasets"])


def load(name: str) -> Dataset:
    """Load a named dataset without R or a network connection.

    Names are listed by ``available()``. The exported JSON data represent
    missing values as null; metadata preserve variable descriptions, units,
    factor levels, ordered status, numeric value labels, and source hashes.
    See ``docs/datasets.md`` for the common preparation workflow.
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
