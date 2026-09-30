import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from shrecc.result_store import (
    CACHE_FORMAT,
    CACHE_VERSION,
    get_package_user_data_dir,
    load_consumption_result_cache,
    save_pickle,
    write_consumption_result_cache,
)
from shrecc.solver import solve_consumption_system


def test_save_pickle_creates_parent_directories(tmp_path):
    filename = tmp_path / "nested" / "value.pkl"

    save_pickle({"value": 1}, filename)

    assert filename.is_file()


def test_get_package_user_data_dir_creates_directory(monkeypatch, tmp_path):
    destination = tmp_path / "user-data"
    monkeypatch.setattr(
        "shrecc.result_store.appdirs.user_data_dir",
        lambda _package_name: destination,
    )

    assert get_package_user_data_dir() == destination
    assert destination.is_dir()


def _result_chunks():
    times = pd.date_range("2040-01-01", periods=4, freq="h")
    production_volume = xr.DataArray(
        [[[10.0]], [[20.0]], [[30.0]], [[40.0]]],
        dims=("time", "producer_country", "technology"),
        coords={
            "time": times,
            "producer_country": ["A"],
            "technology": ["Solar"],
        },
    )
    trade_volume = xr.DataArray(
        np.zeros((4, 1, 1)),
        dims=("time", "exporter_country", "importer_country"),
        coords={
            "time": times,
            "exporter_country": ["A"],
            "importer_country": ["A"],
        },
    )
    first, _ = solve_consumption_system(
        production_volume.isel(time=slice(0, 2)),
        trade_volume.isel(time=slice(0, 2)),
    )
    second, _ = solve_consumption_system(
        production_volume.isel(time=slice(2, 4)),
        trade_volume.isel(time=slice(2, 4)),
    )
    return first, second


def test_result_cache_writes_manifest_and_loads_selected_range(tmp_path):
    cache_dir = tmp_path / "canonical-results"
    chunks = _result_chunks()

    write_consumption_result_cache(chunks, cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    selected = load_consumption_result_cache(
        cache_dir,
        general_range=["2040-01-01 01:00", "2040-01-01 02:00"],
        variables=["consumption_volume", "consumption_mix"],
    )

    assert manifest["format"] == CACHE_FORMAT
    assert manifest["version"] == CACHE_VERSION
    assert len(manifest["chunks"]) == 2
    assert set(selected.data_vars) == {"consumption_volume", "consumption_mix"}
    assert selected.sizes["time"] == 2
    xr.testing.assert_allclose(
        selected["consumption_volume"],
        xr.concat(chunks, dim="time")["consumption_volume"].isel(time=slice(1, 3)),
    )


def test_result_cache_rejects_existing_directory(tmp_path):
    cache_dir = tmp_path / "canonical-results"
    cache_dir.mkdir()

    with pytest.raises(FileExistsError):
        write_consumption_result_cache(_result_chunks(), cache_dir)
