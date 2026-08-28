"""Temporal LCIA calculations for mapped SHRECC electricity mixes."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import warnings

import bw2data as bd
import numpy as np
import pandas as pd
import xarray as xr

from shrecc.database import get_country_network_activities, map_known_inputs

DEFAULT_LCIA_METHOD_FAMILY = "EF v3.1"
VALID_LCIA_ENGINES = {"linear", "multilca"}
SOURCE_SCORE_CACHE_FORMAT = "shrecc-lcia-source-scores"
SOURCE_SCORE_CACHE_VERSION = 1
SOURCE_SCORE_CACHE_DIRNAME = (
    f"lcia_source_scores_v{SOURCE_SCORE_CACHE_VERSION}"
)


@dataclass(frozen=True)
class ResolvedInventoryBasis:
    """Hourly coefficients over a fixed set of Brightway background inputs."""

    year: int
    background_database: str
    coefficients: pd.DataFrame

    def __post_init__(self):
        if not isinstance(self.coefficients.index, pd.MultiIndex):
            raise TypeError("coefficients index must identify database and code")
        if self.coefficients.index.nlevels != 2:
            raise ValueError("coefficients index must contain database and code")
        if not isinstance(self.coefficients.columns, pd.MultiIndex):
            raise TypeError("coefficients columns must identify time and country")
        if self.coefficients.columns.nlevels != 2:
            raise ValueError("coefficients columns must contain time and country")


class LCIAResults:
    """Hourly electricity impacts and their profile-weighted aggregates.

    ``hourly(year)`` returns an xarray Dataset with three labelled variables:
    ``intensity`` is the impact of consuming one kilowatt hour in each country
    and hour, ``consumption_weight`` contains the unnormalized temporal
    profile, and ``weighted_contribution`` is each hour's contribution to the
    annual result. ``monthly()`` and ``annual()`` return the corresponding
    profile-weighted intensities without recalculating Brightway.
    """

    def __init__(self, results_by_year, methods, engine):
        self._results_by_year = dict(results_by_year)
        self.methods = tuple(methods)
        self.engine = engine

    @property
    def years(self):
        """Return modeled years in calculation order."""
        return tuple(self._results_by_year)

    @property
    def impact_categories(self):
        """Return the selectable impact-category labels used by xarray."""
        if not self._results_by_year:
            return ()
        dataset = next(iter(self._results_by_year.values()))
        return tuple(map(str, dataset["impact_category"].values))

    def hourly(self, year=None):
        """Return hourly intensities, profile weights, and contributions.

        Select one time series with, for example,
        ``result["intensity"].sel(consumer_country="PT",
        impact_category=assessment.impact_categories[0])``.
        """
        return self._resolve_year(year).copy()

    def annual(self):
        """Return profile-weighted annual LCIA intensities."""
        rows = []
        for year, dataset in self._results_by_year.items():
            rows.append(
                dataset["weighted_contribution"].sum("time").expand_dims(year=[year])
            )
        return xr.concat(rows, dim="year").rename("weighted_intensity")

    def monthly(self):
        """Return profile-weighted monthly LCIA intensities."""
        rows = []
        for year, dataset in self._results_by_year.items():
            times = dataset["time"].to_index()
            periods = times.to_period("M")
            monthly = []
            for period in periods.unique():
                positions = np.flatnonzero(periods == period)
                intensity = dataset["intensity"].isel(time=positions)
                weights = dataset["consumption_weight"].isel(time=positions)
                total = weights.sum("time")
                if bool((total <= 0).any()):
                    raise ValueError(f"Consumption weights sum to zero in {period}")
                monthly.append(
                    ((intensity * weights).sum("time") / total).expand_dims(
                        time=[period.start_time]
                    )
                )
            rows.append(xr.concat(monthly, dim="time").expand_dims(year=[year]))
        return xr.concat(rows, dim="year").rename("weighted_intensity")

    def __repr__(self):
        return (
            f"LCIAResults(years={list(self.years)!r}, "
            f"impact_categories={len(self.impact_categories)}, "
            f"engine={self.engine!r})"
        )

    def _resolve_year(self, year):
        if year is None:
            if len(self._results_by_year) != 1:
                raise ValueError("year is required for multi-year LCIA results")
            year = next(iter(self._results_by_year))
        try:
            return self._results_by_year[int(year)]
        except KeyError as exc:
            raise ValueError(f"No LCIA results available for {year}") from exc


def load_source_score_cache(data_root, background_database, methods):
    """Load locally generated activity LCIA scores for one background state.

    The fingerprint includes the current Brightway project, background and
    dependency metadata, and the complete characterization methods. Cache
    files contain ecoinvent-derived results and must remain local.

    Returns:
        A ``(path, scores)`` tuple. ``scores`` maps Brightway
        ``(database, code)`` keys to arrays ordered like ``methods``.
    """
    path = _source_score_cache_path(data_root, background_database, methods)
    if not path.is_file():
        return path, {}

    try:
        with np.load(path, allow_pickle=False) as stored:
            databases = stored["database"].astype(str)
            codes = stored["code"].astype(str)
            values = stored["scores"].astype(float)
        if values.shape != (len(databases), len(methods)):
            raise ValueError("cached source-score dimensions are inconsistent")
        return path, {
            (database, code): values[position]
            for position, (database, code) in enumerate(zip(databases, codes))
        }
    except (OSError, KeyError, ValueError) as exc:
        warnings.warn(
            f"Ignoring unreadable SHRECC LCIA source-score cache {path}: {exc}",
            stacklevel=2,
        )
        return path, {}


def write_source_score_cache(path, scores, methods):
    """Atomically persist locally generated activity LCIA scores."""
    if not scores:
        return

    keys = sorted(scores)
    values = np.vstack([np.asarray(scores[key], dtype=float) for key in keys])
    if values.shape != (len(keys), len(methods)):
        raise ValueError("source scores do not match the requested LCIA methods")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(
                handle,
                database=np.asarray([key[0] for key in keys]),
                code=np.asarray([key[1] for key in keys]),
                scores=values,
            )
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _source_score_cache_path(data_root, background_database, methods):
    fingerprint = _source_score_cache_fingerprint(background_database, methods)
    return Path(data_root) / SOURCE_SCORE_CACHE_DIRNAME / f"{fingerprint}.npz"


def _source_score_cache_fingerprint(background_database, methods):
    payload = {
        "format": SOURCE_SCORE_CACHE_FORMAT,
        "version": SOURCE_SCORE_CACHE_VERSION,
        "project": str(bd.projects.current),
        "databases": _background_database_state(background_database),
        "methods": [
            {
                "name": tuple(method),
                "metadata": dict(bd.Method(method).metadata),
                "characterization_factors": bd.Method(method).load(),
            }
            for method in methods
        ],
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_cache_json_default,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _background_database_state(background_database):
    pending = [background_database]
    states = {}
    while pending:
        database_name = pending.pop()
        if database_name in states:
            continue
        metadata = dict(bd.databases[database_name])
        states[database_name] = metadata
        pending.extend(
            dependency
            for dependency in metadata.get("depends", ())
            if dependency not in states
        )
    return states


def _cache_json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def resolve_lcia_methods(methods=None):
    """Resolve requested Brightway methods, defaulting to EF v3.1."""
    if methods is None:
        resolved = sorted(
            method
            for method in bd.methods
            if len(method) > 1 and method[1] == DEFAULT_LCIA_METHOD_FAMILY
        )
        if not resolved:
            raise ValueError(
                "No EF v3.1 LCIA methods are installed in the current "
                "Brightway project. Pass methods explicitly to lcia()."
            )
        return tuple(resolved)

    if (
        isinstance(methods, tuple)
        and methods
        and all(isinstance(item, str) for item in methods)
    ):
        methods = [methods]
    try:
        resolved = tuple(tuple(method) for method in methods)
    except TypeError as exc:
        raise TypeError(
            "methods must be a Brightway method tuple or an iterable of tuples"
        ) from exc
    if not resolved:
        raise ValueError("methods cannot be empty")
    missing = [method for method in resolved if method not in bd.methods]
    if missing:
        raise ValueError(
            "LCIA methods are not installed in the current project: "
            + "; ".join(map(str, missing))
        )
    return resolved


def build_resolved_inventory_basis(
    table,
    *,
    year,
    background_database,
    include_network=True,
    strict=False,
    background_index=None,
):
    """Resolve an hourly mapped table to a fixed Brightway input basis."""
    if not isinstance(table.columns, pd.MultiIndex) or table.columns.nlevels != 2:
        raise ValueError("Hourly coefficient table columns must be (time, country)")

    include_network = include_network is True or (
        isinstance(include_network, str) and include_network.lower() == "true"
    )
    known_inputs, known_network_inputs = map_known_inputs(
        background_database,
        table,
        strict=strict,
        include_network=include_network,
        background_index=background_index,
    )
    coefficients = {}
    column_count = len(table.columns)
    for row in table.index:
        geography, activity_name, _product, unit = row
        key = known_inputs.get((geography, activity_name, unit))
        if key is None:
            continue
        values = table.loc[row].to_numpy(dtype=float)
        coefficients.setdefault(key, np.zeros(column_count, dtype=float))
        coefficients[key] += values

    if include_network:
        for column_index, column in enumerate(table.columns):
            _time, country = column
            for exchange in get_country_network_activities(
                background_database,
                country,
            ):
                key = known_network_inputs.get((exchange["loc"], exchange["name"]))
                if key is None:
                    continue
                coefficients.setdefault(
                    key,
                    np.zeros(column_count, dtype=float),
                )
                coefficients[key][column_index] += float(exchange["val"])

    if not coefficients:
        raise ValueError("No hourly assessment inputs could be resolved")

    index = pd.MultiIndex.from_tuples(
        coefficients,
        names=["database", "code"],
    )
    coefficient_table = pd.DataFrame(
        np.vstack([coefficients[key] for key in coefficients]),
        index=index,
        columns=table.columns,
    )
    coefficient_table.columns.names = ["time", "consumer_country"]
    return ResolvedInventoryBasis(
        year=int(year),
        background_database=background_database,
        coefficients=coefficient_table,
    )


def calculate_lcia(
    basis,
    methods,
    *,
    engine="linear",
    batch_size=1000,
    source_score_cache=None,
):
    """Calculate hourly LCIA intensities for a resolved inventory basis."""
    engine = str(engine).lower()
    if engine not in VALID_LCIA_ENGINES:
        raise ValueError("engine must be 'linear' or 'multilca'")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    if engine == "linear":
        source_scores = _score_unique_inputs(
            basis,
            methods,
            cache=source_score_cache,
        )
        values = basis.coefficients.to_numpy(dtype=float).T @ source_scores
    else:
        values = _score_composite_inventories(
            basis,
            methods,
            batch_size=batch_size,
        )
    return _score_array(values, basis.coefficients.columns, methods)


def consumption_profile_weights(results, consumption_profile):
    """Return unnormalized hourly weights for each consumer country."""
    mix = results["consumption_mix"]
    times = mix["time"].to_index()
    countries = mix["consumer_country"].to_index()
    if isinstance(consumption_profile, str) and consumption_profile == "flat":
        values = np.ones((len(times), len(countries)), dtype=float)
    elif (
        isinstance(consumption_profile, str)
        and consumption_profile == "national_demand"
    ):
        values = (
            results["consumption_volume"]
            .sel(time=times, consumer_country=countries)
            .to_numpy()
            .astype(float)
        )
    elif isinstance(consumption_profile, pd.Series):
        missing = times.difference(consumption_profile.index)
        if not missing.empty:
            raise ValueError(
                "Custom consumption profile is missing hourly LCIA timestamps: "
                + ", ".join(map(str, missing[:5]))
            )
        values = np.broadcast_to(
            consumption_profile.reindex(times).to_numpy(dtype=float)[:, None],
            (len(times), len(countries)),
        ).copy()
    else:
        raise ValueError(
            "consumption_profile must be 'flat', 'national_demand', "
            "or a pandas Series"
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("LCIA consumption weights must be finite and non-negative")
    totals = values.sum(axis=0)
    if (totals <= 0).any():
        missing = ", ".join(map(str, countries[totals <= 0]))
        raise ValueError(
            "LCIA consumption weights sum to zero for countries: " + missing
        )
    return xr.DataArray(
        values,
        dims=("time", "consumer_country"),
        coords={"time": times, "consumer_country": countries},
        name="consumption_weight",
    )


def build_lcia_dataset(intensity, consumption_weight, *, year, engine):
    """Combine hourly intensities and profile weights in a labelled dataset."""
    consumption_weight = consumption_weight.sel(
        time=intensity["time"],
        consumer_country=intensity["consumer_country"],
    )
    normalized_weight = consumption_weight / consumption_weight.sum("time")
    weighted_contribution = (intensity * normalized_weight).rename(
        "weighted_contribution"
    )
    intensity = intensity.copy()
    intensity.attrs.update(
        {
            "description": (
                "Impact intensity of consuming one kilowatt hour in the "
                "consumer country at this hour"
            ),
            "functional_unit": "1 kilowatt hour",
        }
    )
    consumption_weight = consumption_weight.copy()
    consumption_weight.attrs["description"] = (
        "Unnormalized temporal weights supplied by the consumption profile"
    )
    weighted_contribution.attrs.update(
        {
            "description": (
                "Hourly contribution to the profile-weighted annual intensity"
            ),
            "functional_unit": "1 kilowatt hour",
        }
    )
    return xr.Dataset(
        {
            "intensity": intensity,
            "consumption_weight": consumption_weight,
            "weighted_contribution": weighted_contribution,
        },
        attrs={
            "year": int(year),
            "engine": engine,
            "intensity_functional_unit": "1 kWh",
        },
    )


def _score_unique_inputs(basis, methods, *, cache=None):
    cache = {} if cache is None else cache
    missing_keys = [key for key in basis.coefficients.index if key not in cache]
    if missing_keys:
        labels, demands, _node_ids = _input_demands(missing_keys)
        scores = _run_fast_multilca(demands, methods)
        cache.update({key: scores[label] for key, label in zip(missing_keys, labels)})
    return np.vstack([cache[key] for key in basis.coefficients.index])


def _score_composite_inventories(basis, methods, *, batch_size):
    _labels, basis_demands, node_ids = _basis_demands(basis)
    data_objects = _multilca_data_objects(basis_demands, methods)
    coefficients = basis.coefficients.to_numpy(dtype=float).T
    output = np.empty((len(coefficients), len(methods)), dtype=float)
    for start in range(0, len(coefficients), batch_size):
        stop = min(start + batch_size, len(coefficients))
        demands = {}
        labels = []
        for position, row in enumerate(coefficients[start:stop], start=start):
            label = f"inventory:{position}"
            labels.append(label)
            demands[label] = {
                node_id: float(amount)
                for node_id, amount in zip(node_ids, row)
                if amount != 0
            }
        scores = _run_fast_multilca(
            demands,
            methods,
            data_objects=data_objects,
        )
        output[start:stop] = np.vstack([scores[label] for label in labels])
    return output


def _basis_demands(basis):
    return _input_demands(list(basis.coefficients.index))


def _input_demands(input_keys):
    labels = []
    demands = {}
    node_ids = []
    for position, (database, code) in enumerate(input_keys):
        node = bd.get_node(database=database, code=code)
        label = f"input:{position}"
        labels.append(label)
        node_ids.append(node.id)
        demands[label] = {node.id: 1.0}
    return labels, demands, node_ids


def _multilca_data_objects(demands, methods):
    method_config = {"impact_categories": list(methods)}
    return bd.get_multilca_data_objs(
        functional_units=demands,
        method_config=method_config,
    )


def _run_fast_multilca(demands, methods, *, data_objects=None):
    import bw2calc as bc

    method_config = {"impact_categories": list(methods)}
    if data_objects is None:
        data_objects = _multilca_data_objects(demands, methods)
    lca = bc.FastScoresOnlyMultiLCA(
        demands=demands,
        method_config=method_config,
        data_objs=data_objects,
    )
    lca.calculate()
    scores = np.asarray(lca.scores)
    method_labels = list(lca.scores.coords["LCIA"].to_numpy())
    demand_labels = list(lca.scores.coords["processes"].to_numpy())
    method_positions = {label: position for position, label in enumerate(method_labels)}
    ordered_method_positions = [method_positions[str(method)] for method in methods]
    demand_positions = {label: position for position, label in enumerate(demand_labels)}
    return {
        label: scores[ordered_method_positions, demand_positions[label]]
        for label in demands
    }


def _score_array(values, columns, methods):
    times = pd.DatetimeIndex(columns.get_level_values("time"))
    countries = pd.Index(columns.get_level_values("consumer_country"))
    unique_times = times.unique()
    unique_countries = countries.unique()
    expected = pd.MultiIndex.from_product(
        [unique_times, unique_countries],
        names=["time", "consumer_country"],
    )
    frame = pd.DataFrame(values, index=columns, columns=_method_labels(methods))
    frame = frame.reindex(expected)
    if frame.isna().to_numpy().any():
        raise ValueError("Hourly LCIA coefficients do not form a complete grid")
    array = frame.to_numpy().reshape(
        len(unique_times),
        len(unique_countries),
        len(methods),
    )
    return xr.DataArray(
        array,
        dims=("time", "consumer_country", "impact_category"),
        coords={
            "time": unique_times,
            "consumer_country": unique_countries,
            "impact_category": _method_labels(methods),
        },
        name="intensity",
        attrs={"unit": "LCIA score per kWh"},
    )


def _method_labels(methods):
    return [" | ".join(map(str, method)) for method in methods]
