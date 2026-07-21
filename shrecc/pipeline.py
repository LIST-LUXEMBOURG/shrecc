"""High-level orchestration for creating SHRECC electricity databases."""

from importlib.resources import files
from pathlib import Path
import shutil

import pandas as pd

from shrecc.database import apply_cutoff, create_database, load_mapping_data
from shrecc.energy_charts import (
    data_processing as process_energy_charts_data,
    energy_charts_cached_countries,
    get_data as get_energy_charts_data,
)
from shrecc.mapping import (
    activity_mix_to_database_table,
    map_consumption_mix_to_ecoinvent_activities,
)
from shrecc.premise_mapping import (
    PremiseConsumptionMixMapper,
    premise_activity_mix_to_database_table,
)
from shrecc.result_store import (
    MANIFEST_FILENAME,
    consumption_result_cache_path,
    load_consumption_result_cache,
)
from shrecc.tyndp import (
    TYNDP_SCENARIO_YEARS,
    build_z_gross_from_tyndp_scenario,
    consumption_results_from_z_gross,
    validate_tyndp_scenario,
)

PROSPECTIVE_YEARS = frozenset(
    year for years in TYNDP_SCENARIO_YEARS.values() for year in years
)
VALID_SOURCES = {"auto", "energy_charts", "tyndp"}


class NewDatabase:
    """Build mapped electricity tables and write Brightway databases.

    ``create()`` performs source acquisition, graph solving, mapping, temporal
    filtering, and cutoff application. ``write()`` is the only method that
    changes a Brightway project.

    The source defaults to ``"auto"``: TYNDP is selected for supported
    prospective scenario years and Energy Charts for other years.
    """

    def __init__(
        self,
        *,
        years,
        countries,
        project_name,
        premise_db,
        my_db_name,
        scenario=None,
        climate_year=None,
        iam="remind-eu",
        general_range=None,
        refined_range=None,
        freq=None,
        times=None,
        source="auto",
        cutoff=1e-3,
        include_cutoff=True,
        strict=False,
        network=True,
        zero_consumption="month_hour_average",
        include_consumption_mix_volume=True,
        data_dir=None,
        technology_mapping=None,
        country_mapping=None,
        ecoinvent_mapping=None,
        topology_files=None,
        download=True,
        engine="calamine",
        check=True,
        verbose=False,
    ):
        self.years = _normalize_years(years)
        self.countries = _normalize_countries(countries)
        self.project_name = _require_string(project_name, "project_name")
        self.scenario = str(scenario).upper() if scenario is not None else None
        self.climate_year = int(climate_year) if climate_year is not None else None
        self.iam = _require_string(iam, "iam")
        self.source = str(source).lower()
        if self.source not in VALID_SOURCES:
            raise ValueError(
                "source must be one of: " + ", ".join(sorted(VALID_SOURCES))
            )

        self.general_range = general_range
        self.refined_range = refined_range
        self.freq = freq
        self.times = times
        _validate_time_selection(general_range, refined_range, freq, times)

        self.cutoff = float(cutoff)
        if self.cutoff < 0:
            raise ValueError("cutoff cannot be negative")
        self.include_cutoff = bool(include_cutoff)
        self.strict = bool(strict)
        self.network = network
        self.zero_consumption = zero_consumption
        self.include_consumption_mix_volume = bool(include_consumption_mix_volume)
        self.download = bool(download)
        self.engine = engine
        self.check = bool(check)
        self.verbose = bool(verbose)

        package_data = files("shrecc.data")
        tyndp_mapping_dir = package_data / "tyndp"
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.technology_mapping = (
            technology_mapping
            if technology_mapping is not None
            else tyndp_mapping_dir / "tyndp_activities.csv"
        )
        self.country_mapping = (
            country_mapping if country_mapping is not None else tyndp_mapping_dir
        )
        self.ecoinvent_mapping = (
            ecoinvent_mapping
            if ecoinvent_mapping is not None
            else package_data / "el_map_all_norm.csv"
        )
        self.topology_files = dict(topology_files or {})
        remind_eu_topology = tyndp_mapping_dir / "remind-eu-topology.json"
        if self.iam == "remind-eu" and self.iam not in self.topology_files:
            self.topology_files[self.iam] = remind_eu_topology

        self.sources = {year: self._source_for_year(year) for year in self.years}
        if any(source_name == "tyndp" for source_name in self.sources.values()):
            if self.scenario is None:
                raise ValueError("scenario is required for TYNDP years")
            if self.climate_year is None:
                raise ValueError("climate_year is required for TYNDP years")
            for year, source_name in self.sources.items():
                if source_name == "tyndp":
                    validate_tyndp_scenario(
                        self.scenario,
                        year,
                        self.climate_year,
                    )

        self.background_databases = _resolve_year_values(
            premise_db,
            self.years,
            "premise_db",
            allow_automatic_suffix=False,
        )
        self.database_names = _resolve_year_values(
            my_db_name,
            self.years,
            "my_db_name",
            allow_automatic_suffix=True,
        )

        self.consumption_results = {}
        self.activity_mixes = {}
        self.exchange_geography_maps = {}
        self.database_tables = {}
        self.written_database_names = {}

    def create(self):
        """Create mapped and filtered database tables for every configured year."""
        self.consumption_results.clear()
        self.activity_mixes.clear()
        self.exchange_geography_maps.clear()
        self.database_tables.clear()
        self.written_database_names.clear()

        for year in self.years:
            if self.verbose:
                print(f"Creating {self.sources[year]} electricity mix for {year}")
            if self.sources[year] == "tyndp":
                self._create_tyndp_year(year)
            else:
                self._create_energy_charts_year(year)
        return self

    def write(self):
        """Write all created tables to their configured Brightway databases."""
        if set(self.database_tables) != set(self.years):
            self.create()

        for year in self.years:
            create_database(
                dataframe_filt=self.database_tables[year],
                project_name=self.project_name,
                db_name=self.database_names[year],
                eidb_name=self.background_databases[year],
                network=self.network,
                strict=self.strict,
            )
            self.written_database_names[year] = self.database_names[year]
        return self

    def table(self, year=None):
        """Return one mapped table, requiring ``year`` for multi-year runs."""
        year = self._resolve_result_year(year)
        if year not in self.database_tables:
            raise RuntimeError("create() must be called before accessing tables")
        return self.database_tables[year]

    def results(self, year=None):
        """Return one canonical result Dataset."""
        year = self._resolve_result_year(year)
        if year not in self.consumption_results:
            raise RuntimeError("create() must be called before accessing results")
        return self.consumption_results[year]

    def _create_energy_charts_year(self, year):
        data_root = self._data_root()
        cache_dir = consumption_result_cache_path(data_root, year)
        if not (cache_dir / MANIFEST_FILENAME).is_file():
            data = get_energy_charts_data(
                year,
                path_to_data=data_root,
                required_countries=self.countries,
            )
            cache_dir = process_energy_charts_data(
                data,
                year,
                path_to_data=data_root,
                include_consumption_mix_volume=False,
            )
        general_range, times = self._selection_for_year(year)
        results = load_consumption_result_cache(
            cache_dir,
            general_range=general_range,
            times=times,
        )
        missing_result_countries = set(self.countries).difference(
            results["consumption_mix"]["consumer_country"].to_index()
        )
        cached_source_countries = energy_charts_cached_countries(year, data_root)
        missing_source_countries = (
            set(self.countries).difference(cached_source_countries)
            if cached_source_countries is not None
            else set()
        )
        missing_countries = missing_result_countries | missing_source_countries
        if missing_countries:
            if self.verbose:
                print(
                    "Canonical Energy Charts cache is missing required countries "
                    f"{sorted(missing_countries)}; resuming source acquisition."
                )
            data = get_energy_charts_data(
                year,
                path_to_data=data_root,
                required_countries=self.countries,
            )
            shutil.rmtree(cache_dir)
            cache_dir = process_energy_charts_data(
                data,
                year,
                path_to_data=data_root,
                include_consumption_mix_volume=False,
            )
            results = load_consumption_result_cache(
                cache_dir,
                general_range=general_range,
                times=times,
            )
            remaining_missing = set(self.countries).difference(
                results["consumption_mix"]["consumer_country"].to_index()
            )
            if remaining_missing:
                raise ValueError(
                    "Rebuilt Energy Charts cache is still missing required "
                    "countries: " + ", ".join(sorted(remaining_missing))
                )
        if (
            self.include_consumption_mix_volume
            and "consumption_mix_volume" not in results
        ):
            results["consumption_mix_volume"] = (
                results["consumption_mix"] * results["consumption_volume"]
            ).assign_attrs(unit=results.attrs.get("volume_unit", "MWh"))

        activity_mapping = load_mapping_data(Path(self.ecoinvent_mapping))
        activity_mix = map_consumption_mix_to_ecoinvent_activities(
            results["consumption_mix"],
            activity_mapping,
            check=self.check,
        )
        table = activity_mix_to_database_table(
            activity_mix,
            countries=self.countries,
            general_range=general_range,
            refined_range=self.refined_range,
            freq=self.freq,
            times=times,
        )
        self._store_year(year, results, activity_mix, table)

    def _create_tyndp_year(self, year):
        general_range, times = self._selection_for_year(year)
        Z_gross = build_z_gross_from_tyndp_scenario(
            scenario=self.scenario,
            year=year,
            climate_year=self.climate_year,
            data_dir=self._tyndp_cache_dir(),
            technology_mapping=self.technology_mapping,
            country_mapping=self.country_mapping,
            download=self.download,
            engine=self.engine,
            verbose=self.verbose,
        )
        Z_gross = _select_dataframe_times(
            Z_gross,
            general_range=general_range,
            times=times,
        )
        results = consumption_results_from_z_gross(
            Z_gross,
            check=self.check,
            zero_consumption=self.zero_consumption,
            verbose=self.verbose,
            include_consumption_mix_volume=(self.include_consumption_mix_volume),
        )

        mapper = PremiseConsumptionMixMapper(
            technology_mapping=self.technology_mapping,
            ecoinvent_mapping=self.ecoinvent_mapping,
            iam_model=self.iam,
            topology_files=self.topology_files,
            on_missing_model="warn",
        )
        activity_mix = mapper.map_technologies(
            results["consumption_mix"],
            check=self.check,
        )
        exchange_geography_map = mapper.build_exchange_geography_map(activity_mix)
        table = premise_activity_mix_to_database_table(
            activity_mix,
            exchange_geography_map,
            countries=self.countries,
            general_range=general_range,
            refined_range=self.refined_range,
            freq=self.freq,
            times=times,
        )
        self.exchange_geography_maps[year] = exchange_geography_map
        self._store_year(year, results, activity_mix, table)

    def _store_year(self, year, results, activity_mix, table):
        self.consumption_results[year] = results
        self.activity_mixes[year] = activity_mix
        self.database_tables[year] = apply_cutoff(
            table,
            cutoff=self.cutoff,
            include_cutoff=self.include_cutoff,
        )

    def _source_for_year(self, year):
        if self.source != "auto":
            return self.source
        return "tyndp" if year in PROSPECTIVE_YEARS else "energy_charts"

    def _data_root(self):
        if self.data_dir is not None:
            return self.data_dir
        from shrecc.energy_charts import get_package_user_data_dir

        return Path(get_package_user_data_dir())

    def _tyndp_cache_dir(self):
        return self._data_root() / "tyndp" / "cache"

    def _selection_for_year(self, year):
        if self.general_range is not None:
            timestamps = pd.to_datetime(self.general_range)
            if len(self.years) == 1:
                if any(timestamp.year != year for timestamp in timestamps):
                    raise ValueError(
                        f"general_range timestamps must belong to year {year}"
                    )
            else:
                timestamps = pd.DatetimeIndex(
                    [timestamp.replace(year=year) for timestamp in timestamps]
                )
            return list(timestamps), None

        timestamps = pd.DatetimeIndex(pd.to_datetime(self.times))
        if len(self.years) == 1:
            if any(timestamp.year != year for timestamp in timestamps):
                raise ValueError(f"times must belong to year {year}")
        else:
            timestamps = pd.DatetimeIndex(
                [timestamp.replace(year=year) for timestamp in timestamps]
            )
        return None, timestamps

    def _resolve_result_year(self, year):
        if year is None:
            if len(self.years) != 1:
                raise ValueError("year is required for a multi-year NewDatabase")
            return self.years[0]
        year = int(year)
        if year not in self.years:
            raise KeyError(f"Year {year} is not configured")
        return year

    def __repr__(self):
        years = ", ".join(map(str, self.years))
        sources = ", ".join(f"{year}:{self.sources[year]}" for year in self.years)
        return f"NewDatabase(years=[{years}], sources={{{sources}}})"


def _normalize_years(years):
    if isinstance(years, int):
        normalized = (years,)
    else:
        try:
            normalized = tuple(int(year) for year in years)
        except (TypeError, ValueError) as exc:
            raise TypeError("years must be an integer or iterable of integers") from exc
    if not normalized:
        raise ValueError("years must contain at least one year")
    if len(set(normalized)) != len(normalized):
        raise ValueError("years cannot contain duplicates")
    return normalized


def _normalize_countries(countries):
    if isinstance(countries, str):
        countries = [countries]
    normalized = tuple(str(country).upper() for country in countries)
    if not normalized:
        raise ValueError("countries must contain at least one country")
    if len(set(normalized)) != len(normalized):
        raise ValueError("countries cannot contain duplicates")
    return normalized


def _require_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _resolve_year_values(value, years, name, allow_automatic_suffix):
    if isinstance(value, dict):
        missing = set(years).difference(value)
        if missing:
            raise ValueError(
                f"{name} is missing years: " + ", ".join(map(str, sorted(missing)))
            )
        return {year: _require_string(value[year], name) for year in years}

    value = _require_string(value, name)
    if len(years) == 1:
        return {years[0]: value.format(year=years[0])}
    if "{year}" in value:
        return {year: value.format(year=year) for year in years}
    if allow_automatic_suffix:
        return {year: f"{value}_{year}" for year in years}
    raise ValueError(
        f"{name} must be a year mapping or contain '{{year}}' for multiple years"
    )


def _validate_time_selection(general_range, refined_range, freq, times):
    if general_range is None and times is None:
        raise ValueError("Either general_range or times must be provided")
    if general_range is not None and times is not None:
        raise ValueError("Use either general_range or times, not both")
    if general_range is not None and len(general_range) != 2:
        raise ValueError("general_range must contain a start and end timestamp")
    if times is not None and not len(times):
        raise ValueError("times must contain at least one timestamp")
    if refined_range and general_range is None:
        raise ValueError("general_range is required when refined_range is supplied")
    if refined_range and freq is None:
        raise ValueError("freq is required when refined_range is supplied")


def _select_dataframe_times(dataframe, *, general_range, times):
    if general_range is not None:
        selected = dataframe.loc[general_range[0] : general_range[1]]
    else:
        selected = dataframe.loc[dataframe.index.isin(times)]
    if selected.empty:
        raise ValueError("The requested time selection contains no source data")
    return selected
