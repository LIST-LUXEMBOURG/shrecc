"""Chunked persistence for canonical SHRECC consumption results."""

import gzip
import json
import pickle
from pathlib import Path

import appdirs
import pandas as pd
import xarray as xr

from shrecc.solver import iter_consumption_system_results

CACHE_FORMAT = "shrecc-consumption-results"
CACHE_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
CANONICAL_CACHE_DIRNAME = f"consumption_results_v{CACHE_VERSION}"


def save_pickle(obj, filename):
    """Persist a Python object using the repository's legacy pickle format."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    """Load a Python object from the repository's legacy pickle format."""
    with Path(filename).open("rb") as handle:
        return pickle.load(handle)


def get_package_user_data_dir(package_name="shrecc"):
    """Return the package user-data directory, creating it when necessary."""
    destination = Path(appdirs.user_data_dir(package_name))
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def consumption_result_cache_path(data_root, year):
    """Return the versioned canonical cache path for a data year."""
    return Path(data_root) / str(year) / CANONICAL_CACHE_DIRNAME


def write_solved_consumption_result_cache(
    production_volume,
    trade_volume,
    cache_dir,
    *,
    consumption_volume=None,
    time_chunk_size=168,
    **solver_options,
):
    """Solve canonical inputs in chunks and write them directly to a cache."""
    chunks = iter_consumption_system_results(
        production_volume,
        trade_volume,
        consumption_volume=consumption_volume,
        time_chunk_size=time_chunk_size,
        **solver_options,
    )
    return write_consumption_result_cache(chunks, cache_dir)


def write_consumption_result_cache(result_chunks, cache_dir):
    """Write canonical result Datasets as compressed time chunks.

    Args:
        result_chunks: Iterable of xarray Datasets with a ``time`` dimension.
        cache_dir: New directory in which to create the cache.

    Returns:
        Path to the written cache directory.

    Raises:
        FileExistsError: If ``cache_dir`` already exists.
        ValueError: If chunks are empty, overlap, or have incompatible schemas.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "chunks": [],
    }
    reference_schema = None
    previous_end = None

    try:
        for chunk_number, results in enumerate(result_chunks):
            schema, times = _validate_result_chunk(results)
            if reference_schema is None:
                reference_schema = schema
                manifest["schema"] = schema
                manifest["attrs"] = dict(results.attrs)
            elif schema != reference_schema:
                raise ValueError("Consumption-result chunks have incompatible schemas")

            start = times[0]
            end = times[-1]
            if previous_end is not None and start <= previous_end:
                raise ValueError(
                    "Consumption-result chunks must be ordered and non-overlapping"
                )

            filename = f"chunk_{chunk_number:05d}.pkl.gz"
            with gzip.open(cache_dir / filename, "wb", compresslevel=4) as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

            manifest["chunks"].append(
                {
                    "file": filename,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "time_count": len(times),
                }
            )
            previous_end = end

        if not manifest["chunks"]:
            raise ValueError("result_chunks must contain at least one Dataset")

        manifest_path = cache_dir / MANIFEST_FILENAME
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
    except Exception:
        # An absent manifest makes an interrupted cache unambiguously invalid.
        raise

    return cache_dir


def iter_consumption_result_cache(
    cache_dir,
    *,
    general_range=None,
    times=None,
    variables=None,
):
    """Yield cached result chunks intersecting an optional time selection."""
    cache_dir = Path(cache_dir)
    manifest = _load_manifest(cache_dir)
    start, end, requested_times = _normalize_time_selection(general_range, times)

    selected_any = False
    for chunk in manifest["chunks"]:
        chunk_start = pd.Timestamp(chunk["start"])
        chunk_end = pd.Timestamp(chunk["end"])
        if start is not None and chunk_end < start:
            continue
        if end is not None and chunk_start > end:
            continue

        with gzip.open(cache_dir / chunk["file"], "rb") as handle:
            results = pickle.load(handle)
        if not isinstance(results, xr.Dataset):
            raise ValueError(f"Cache chunk {chunk['file']!r} is not an xarray Dataset")

        if general_range is not None:
            results = results.sel(time=slice(start, end))
        if requested_times is not None:
            results = results.sel(time=results["time"].isin(requested_times))
        if variables is not None:
            results = results[list(variables)]
        if results.sizes["time"]:
            selected_any = True
            yield results

    if not selected_any:
        raise ValueError("The requested time selection contains no cached results")


def load_consumption_result_cache(
    cache_dir,
    *,
    general_range=None,
    times=None,
    variables=None,
):
    """Load and concatenate selected chunks from a canonical result cache."""
    chunks = list(
        iter_consumption_result_cache(
            cache_dir,
            general_range=general_range,
            times=times,
            variables=variables,
        )
    )
    if len(chunks) == 1:
        return chunks[0]
    return xr.concat(chunks, dim="time")


def _validate_result_chunk(results):
    if not isinstance(results, xr.Dataset):
        raise TypeError("Each result chunk must be an xarray Dataset")
    if "time" not in results.dims:
        raise ValueError("Each result chunk must have a time dimension")
    if results.sizes["time"] == 0:
        raise ValueError("Consumption-result chunks cannot be empty")

    times = pd.DatetimeIndex(pd.to_datetime(results["time"].values))
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("Chunk times must be increasing and unique")

    schema = {
        "variables": {
            name: list(results[name].dims) for name in sorted(results.data_vars)
        },
        "non_time_sizes": {
            name: size for name, size in results.sizes.items() if name != "time"
        },
        "non_time_coordinates": {
            name: results[name].values.tolist()
            for name in sorted(results.coords)
            if name != "time"
        },
    }
    return schema, times


def _load_manifest(cache_dir):
    manifest_path = cache_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No canonical consumption-result manifest found at {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != CACHE_FORMAT:
        raise ValueError("Unsupported consumption-result cache format")
    if manifest.get("version") != CACHE_VERSION:
        raise ValueError(
            "Unsupported consumption-result cache version: "
            f"{manifest.get('version')!r}"
        )
    return manifest


def _normalize_time_selection(general_range, times):
    if general_range is not None and times is not None:
        raise ValueError("Use either general_range or times, not both")

    if general_range is not None:
        if len(general_range) != 2:
            raise ValueError("general_range must contain a start and end timestamp")
        start, end = pd.to_datetime(general_range)
        if start > end:
            raise ValueError("general_range start must not be after its end")
        return start, end, None

    if times is not None:
        requested_times = pd.DatetimeIndex(pd.to_datetime(times))
        if requested_times.empty:
            raise ValueError("times must contain at least one timestamp")
        return requested_times.min(), requested_times.max(), requested_times

    return None, None, None
