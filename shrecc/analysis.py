"""Brightway analysis helpers for SHRECC electricity inventories."""

from collections import defaultdict
from collections.abc import Mapping
from gc import collect

import bw2data as bd
import numpy as np
import pandas as pd

DEFAULT_COUNTRY_ALIASES = {"NIE": "GB", "UK": "GB"}
DEFAULT_BACKGROUND_LABEL = "Background"

__all__ = (
    "compare_electricity_generation_mixes",
    "compare_electricity_lcia",
    "resolve_electricity_generation_shares",
)


class _ActivityResolver:
    """Cache Brightway lookups for one analysis call."""

    def __init__(self):
        self._database_indexes = {}
        self._nodes = {}
        self._metadata = {}
        self._electricity_inputs = {}

    def find_unique(self, database_name, activity_name, location):
        if database_name not in self._database_indexes:
            index = {}
            duplicates = set()
            for activity in bd.Database(database_name):
                key = (activity.get("name"), activity.get("location"))
                if key in index:
                    duplicates.add(key)
                else:
                    index[key] = activity
            self._database_indexes[database_name] = (index, duplicates)

        index, duplicates = self._database_indexes[database_name]
        key = (activity_name, location)
        return None if key in duplicates else index.get(key)

    def node(self, activity_id):
        if activity_id not in self._nodes:
            self._nodes[activity_id] = bd.get_node(id=activity_id)
        return self._nodes[activity_id]

    def metadata(self, activity_id):
        if activity_id not in self._metadata:
            activity = self.node(activity_id)
            self._metadata[activity_id] = (
                activity.get("name"),
                activity.get("location"),
            )
        return self._metadata[activity_id]

    def electricity_inputs(self, activity_id):
        if activity_id not in self._electricity_inputs:
            activity = self.node(activity_id)
            self._electricity_inputs[activity_id] = tuple(
                (exchange.input.id, float(exchange["amount"]))
                for exchange in activity.technosphere()
                if float(exchange["amount"]) > 0
                and _is_electricity_activity(exchange.input)
            )
        return self._electricity_inputs[activity_id]


def resolve_electricity_generation_shares(
    root_activity,
    *,
    tolerance=1e-10,
    max_iterations=500,
    resolver=None,
):
    """Resolve delivered electricity through wrappers to generation shares.

    Positive electricity exchanges are propagated through markets, voltage
    transformations, import activities, and IAM-region wrappers. Traversal
    stops at generating activities or at terminal activities without further
    electricity inputs. Activities with the same name are combined across
    locations so foreground and background technology families can be
    compared directly.

    Args:
        root_activity: Brightway activity supplying delivered electricity.
        tolerance: Remaining wrapper volume below which traversal is complete.
        max_iterations: Maximum wrapper-propagation iterations.
        resolver: Optional run-scoped resolver, primarily for composing
            repeated analyses.

    Returns:
        A pair containing a descending Series of normalized generation shares
        and a diagnostics dictionary.

    Raises:
        RuntimeError: If wrapper traversal does not converge.
        ValueError: If no terminal electricity supply is found.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be greater than zero")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")

    resolver = resolver or _ActivityResolver()
    terminal_volumes = defaultdict(float)
    nonstandard_terminals = defaultdict(float)
    frontier = {root_activity.id: 1.0}
    remaining_volume = 0.0

    for iteration in range(1, max_iterations + 1):
        next_frontier = defaultdict(float)
        for activity_id, required_volume in frontier.items():
            activity = resolver.node(activity_id)
            inputs = resolver.electricity_inputs(activity_id)
            generation = _is_generation_activity(activity)
            if generation or not inputs:
                name, location = resolver.metadata(activity_id)
                terminal_volumes[name] += required_volume
                if not generation:
                    nonstandard_terminals[(name, location)] += required_volume
                continue

            for input_id, amount in inputs:
                next_frontier[input_id] += required_volume * amount

        remaining_volume = sum(abs(value) for value in next_frontier.values())
        if remaining_volume <= tolerance:
            break
        frontier = dict(next_frontier)
    else:
        largest = sorted(
            (
                (*resolver.metadata(activity_id), volume)
                for activity_id, volume in frontier.items()
            ),
            key=lambda row: abs(row[-1]),
            reverse=True,
        )[:5]
        raise RuntimeError(
            "Electricity wrapper traversal did not converge. Largest "
            f"remaining activities: {largest}"
        )

    terminal_volumes = pd.Series(terminal_volumes, dtype=float)
    terminal_total = terminal_volumes.sum()
    if terminal_total <= 0:
        raise ValueError(
            f"No terminal electricity generation found below {root_activity}"
        )

    diagnostics = {
        "iterations": iteration,
        "terminal generation volume": terminal_total,
        "remaining wrapper volume": remaining_volume,
        "nonstandard terminal volume": sum(nonstandard_terminals.values()),
    }
    shares = (terminal_volumes / terminal_total).sort_values(ascending=False)
    return shares, diagnostics


def compare_electricity_generation_mixes(
    foreground_databases,
    background_databases,
    *,
    countries,
    years,
    background_label=DEFAULT_BACKGROUND_LABEL,
    country_aliases=None,
):
    """Compare foreground and background delivered-electricity compositions.

    Args:
        foreground_databases: Mapping from display label to ``{year:
            database_name}`` mappings. Foreground databases are expected to
            contain activities named ``"Electricity mix in {country}, {year}"``.
        background_databases: Mapping from year to ecoinvent or premise
            background database name.
        countries: Consumer-country codes to compare.
        years: Model years to compare.
        background_label: Display label assigned to background rows.
        country_aliases: Optional country-to-background-location aliases.

    Returns:
        Three DataFrames: long-form generation shares, missing activity roots,
        and traversal diagnostics.
    """
    years, countries, foreground_databases, background_databases = (
        _validate_comparison_inputs(
            foreground_databases,
            background_databases,
            countries,
            years,
            background_label,
        )
    )
    resolver = _ActivityResolver()
    aliases = {**DEFAULT_COUNTRY_ALIASES, **(country_aliases or {})}
    mix_rows = []
    gaps = []
    diagnostics_rows = []

    for year in years:
        for country in countries:
            roots = _comparison_roots(
                foreground_databases,
                background_databases,
                year,
                country,
                background_label,
                aliases,
                resolver,
            )
            for model, root_activity in roots.items():
                if root_activity is None:
                    gaps.append(
                        {
                            "year": year,
                            "country": country,
                            "model": model,
                            "missing": "delivered low-voltage electricity activity",
                        }
                    )
                    continue

                shares, diagnostics = resolve_electricity_generation_shares(
                    root_activity,
                    resolver=resolver,
                )
                diagnostics_rows.append(
                    {
                        "year": year,
                        "country": country,
                        "model": model,
                        "root activity": root_activity.get("name"),
                        **diagnostics,
                    }
                )
                mix_rows.extend(
                    {
                        "year": year,
                        "country": country,
                        "model": model,
                        "technology": technology,
                        "share": share,
                    }
                    for technology, share in shares.items()
                )

    return (
        pd.DataFrame(
            mix_rows,
            columns=["year", "country", "model", "technology", "share"],
        ),
        pd.DataFrame(
            gaps,
            columns=["year", "country", "model", "missing"],
        ),
        pd.DataFrame(diagnostics_rows),
    )


def compare_electricity_lcia(
    foreground_databases,
    background_databases,
    *,
    countries,
    years,
    method,
    background_label=DEFAULT_BACKGROUND_LABEL,
    country_aliases=None,
    verbose=False,
):
    """Calculate foreground and background LCIA scores per kWh.

    Calculations are batched by year to bound Brightway matrix memory.

    Args:
        foreground_databases: Mapping from display label to ``{year:
            database_name}`` mappings.
        background_databases: Mapping from year to background database name.
        countries: Consumer-country codes to compare.
        years: Model years to compare.
        method: Brightway impact-category tuple.
        background_label: Display label assigned to background rows.
        country_aliases: Optional country-to-background-location aliases.
        verbose: Print each year before its LCIA calculation.

    Returns:
        Two DataFrames: long-form LCIA scores and missing activity roots.
    """
    years, countries, foreground_databases, background_databases = (
        _validate_comparison_inputs(
            foreground_databases,
            background_databases,
            countries,
            years,
            background_label,
        )
    )
    resolver = _ActivityResolver()
    aliases = {**DEFAULT_COUNTRY_ALIASES, **(country_aliases or {})}
    rows = []
    gaps = []

    for year in years:
        if verbose:
            print(f"Calculating {method} for {year}")
        demands = {}
        metadata = {}
        for country in countries:
            roots = _comparison_roots(
                foreground_databases,
                background_databases,
                year,
                country,
                background_label,
                aliases,
                resolver,
            )
            for model, activity in roots.items():
                if activity is None:
                    gaps.append(
                        {
                            "year": year,
                            "country": country,
                            "model": model,
                        }
                    )
                    continue
                label = f"{year}:{country}:{model}"
                demands[label] = {activity.id: 1}
                metadata[label] = (country, model)

        if not demands:
            continue
        scores = _run_multilca(demands, method)
        rows.extend(
            {
                "year": year,
                "country": country,
                "model": model,
                "score": scores[label],
            }
            for label, (country, model) in metadata.items()
        )
        collect()

    return (
        pd.DataFrame(
            rows,
            columns=["year", "country", "model", "score"],
        ),
        pd.DataFrame(
            gaps,
            columns=["year", "country", "model"],
        ),
    )


def _run_multilca(demands, method):
    import bw2calc as bc

    method_config = {"impact_categories": [method]}
    data_objects = bd.get_multilca_data_objs(
        functional_units=demands,
        method_config=method_config,
    )
    lca = bc.MultiLCA(
        demands=demands,
        method_config=method_config,
        data_objs=data_objects,
    )
    lca.lci()
    lca.lcia()
    scores = {
        label: float(np.asarray(lca.scores[(method, label)]).reshape(-1)[0])
        for label in demands
    }
    del lca, data_objects
    return scores


def _comparison_roots(
    foreground_databases,
    background_databases,
    year,
    country,
    background_label,
    aliases,
    resolver,
):
    roots = {
        label: resolver.find_unique(
            databases[year],
            f"Electricity mix in {country}, {year}",
            country,
        )
        for label, databases in foreground_databases.items()
    }
    roots[background_label] = resolver.find_unique(
        background_databases[year],
        "market for electricity, low voltage",
        aliases.get(country, country),
    )
    return roots


def _validate_comparison_inputs(
    foreground_databases,
    background_databases,
    countries,
    years,
    background_label,
):
    if not isinstance(foreground_databases, Mapping) or not foreground_databases:
        raise ValueError("foreground_databases must be a non-empty mapping")
    if not isinstance(background_databases, Mapping):
        raise TypeError("background_databases must be a year mapping")
    if not isinstance(background_label, str) or not background_label:
        raise ValueError("background_label must be a non-empty string")
    if background_label in foreground_databases:
        raise ValueError("background_label must differ from foreground database labels")

    years = tuple(int(year) for year in years)
    countries = tuple(str(country) for country in countries)
    if not years:
        raise ValueError("years cannot be empty")
    if not countries:
        raise ValueError("countries cannot be empty")

    normalized_foregrounds = {}
    for label, databases in foreground_databases.items():
        if not isinstance(label, str) or not label:
            raise ValueError("foreground database labels must be non-empty strings")
        normalized_foregrounds[label] = _require_year_database_mapping(
            databases,
            years,
            f"foreground_databases[{label!r}]",
        )
    normalized_backgrounds = _require_year_database_mapping(
        background_databases,
        years,
        "background_databases",
    )
    return years, countries, normalized_foregrounds, normalized_backgrounds


def _require_year_database_mapping(databases, years, name):
    if not isinstance(databases, Mapping):
        raise TypeError(f"{name} must be a year mapping")
    missing = set(years).difference(databases)
    if missing:
        raise ValueError(
            f"{name} is missing years: " + ", ".join(map(str, sorted(missing)))
        )
    normalized = {}
    for year in years:
        database_name = databases[year]
        if not isinstance(database_name, str) or not database_name:
            raise ValueError(f"{name}[{year}] must be a non-empty string")
        normalized[year] = database_name
    return normalized


def _is_electricity_activity(activity):
    product = str(activity.get("reference product", "")).lower()
    unit = str(activity.get("unit", "")).lower()
    return "electricity" in product and unit in {"kilowatt hour", "kwh"}


def _is_generation_activity(activity):
    name = str(activity.get("name", "")).lower()
    return (
        name.startswith("electricity production")
        or "heat and power co-generation" in name
        or (name.startswith("treatment of ") and "power plant" in name)
        or "electricity generation" in name
    )
