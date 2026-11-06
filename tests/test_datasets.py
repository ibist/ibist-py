import gzip
import hashlib
from importlib.resources import files
import json

import numpy as np
import pandas as pd
import pytest

from ibist import datasets


SIZES = {"asah": (309, 4), "depress": (1730, 9), "fluorosis": (24, 5),
         "midbp": (16, 4), "mihdl": (24, 4), "nrs": (91795, 6), "rds": (16, 4)}


@pytest.mark.parametrize("name", SIZES)
def test_exports_load_with_metadata_and_matching_hash(name):
    dataset = datasets.load(name)
    assert (len(dataset), len(dataset.data)) == SIZES[name]
    assert dataset.metadata["schema_version"] == 1
    assert dataset.metadata["name"] == name
    columns = dataset.metadata["columns"]
    assert tuple(dataset.data) == tuple(c["name"] for c in columns)
    path = files("ibist").joinpath("_data", f"{name}.json.gz")
    payload = path.read_bytes()
    expected_hash = dataset.metadata["provenance"]["data_md5"]
    assert hashlib.md5(payload).hexdigest() == expected_hash
    raw = json.loads(gzip.decompress(payload))
    for column in dataset.metadata["columns"]:
        assert column["description"]
        values = raw[column["name"]]
        assert len(values) == len(dataset)
        assert sum(v is None for v in values) == column["missing_count"]
        if column["type"] == "category":
            assert all(v is None or v in column["levels"] for v in values)


def test_r_factor_order_and_numeric_labels_are_preserved():
    asah = datasets.load("asah").to_frame()
    assert asah["scale"].cat.categories.tolist() == ["WFNS", "HH", "PAASH"]
    assert asah["observer1"].cat.ordered
    assert asah["observer1"].cat.categories.tolist() == list("12345")
    wfns = asah[asah["scale"] == "WFNS"]
    table = pd.crosstab(wfns["observer1"], wfns["observer2"], dropna=False)
    np.testing.assert_array_equal(table, [[37, 9, 0, 0, 0], [9, 27, 1, 2, 0],
                                         [0, 2, 3, 0, 0], [0, 3, 1, 4, 0],
                                         [0, 0, 0, 1, 4]])
    gender = datasets.load("depress").metadata["columns"][1]
    assert gender["value_labels"] == [{"value": 0, "label": "Male"},
                                      {"value": 1, "label": "Female"}]
    assert gender["label"] == "A Gender"


def test_missing_values_and_nullable_integer_columns():
    nrs = datasets.load("nrs").to_frame()
    assert str(nrs["ex"].dtype) == "Int64"
    assert nrs.isna().sum().to_dict() == {"id": 0, "gender": 0, "age": 0,
                                         "ex": 1405, "pa": 1445, "nrs": 1480}
    assert len(nrs.dropna()) == 90122
    depress = datasets.load("depress").to_frame()
    assert depress["IEStert"].isna().sum() == 604
    assert depress["depress2yr"].isna().sum() == 499


def test_units_counts_and_factor_labels():
    fluorosis = datasets.load("fluorosis")
    assert fluorosis.data["count"].sum() == 308
    description = fluorosis.metadata["columns"][0]["description"]
    assert description.startswith("Otitis media")
    rds = datasets.load("rds")
    assert rds.metadata["columns"][0]["units"] == "grams"
    assert rds.metadata["columns"][0]["levels"] == [
        "500-749g", "750-999g", "1000-1249g", "1250-1500g"]
    midbp = datasets.load("midbp")
    assert set(midbp.data["dbp"]) == set(range(1, 9))
    assert midbp.metadata["columns"][-1]["units"] == "person-years"


def test_load_returns_independent_data_and_metadata():
    first = datasets.load("fluorosis")
    first.data["count"][0] = -1
    first.metadata["columns"][0]["levels"].append("changed")
    second = datasets.load("fluorosis")
    assert second.data["count"][0] == 2
    assert second.metadata["columns"][0]["levels"] == ["0", "1"]


@pytest.mark.parametrize("name", ["unknown", "../rds", "", None, 1])
def test_rejects_unknown_dataset_names(name):
    with pytest.raises(ValueError, match="Unknown dataset"):
        datasets.load(name)


def test_missing_raw_inputs_are_explicit_in_provenance():
    source = datasets.load("nrs").metadata["provenance"]
    assert source["prepared_file"] == "data/nrs.rda"
    assert source["raw_inputs"] == [
        {"path": "data-raw/nrs.csv", "available": False, "md5": None}]
