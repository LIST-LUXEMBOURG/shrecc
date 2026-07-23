"""High-level orchestration for creating SHRECC electricity databases."""

from importlib.resources import files
from pathlib import Path
import shutil
import warnings

import pandas as pd

from shrecc.database import apply_cutoff, create_database
from shrecc.energy_charts import (
    # Aliases distinguish orchestration dependencies from their implementations.
    data_processing as process_energy_charts_data,
    energy_charts_cached_countries,
    get_data as get_energy_charts_data,
)
from shrecc.mapping import (
    activity_mix_to_database_table,
    load_ecoinvent_mapping,
    map_consumption_mix_to_ecoinvent_activities,
    mapping_gap_to_report,
)
from shrecc.premise_mapping import (
    PremiseConsumptionMixMapper,
    premise_activity_mix_to_database_table,
)
from shrecc.result_store import (
    MANIFEST_FILENAME,
    consumption_result_cache_path,
    get_package_user_data_dir,
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
    """Create consumption-based electricity inventories for Brightway.

    In plain terms, this class answers: "Which technologies, in which
    countries, supplied the electricity consumed here during the selected
    hours?" It follows electricity through the interconnected country network,
    maps the resulting technology shares to ecoinvent or premise activities,
    and prepares one inventory for each requested consumer country.

    The workflow is deliberately split in two. :meth:`create` downloads or
    reads source data, solves the electricity system, and prepares inspectable
    in-memory tables. :meth:`write` then writes those tables to Brightway.
    Calling :meth:`create` alone does not modify a Brightway project.

    With ``source="auto"``, supported prospective years use ENTSO-E TYNDP
    scenario data and other years use historical Energy Charts data. Multi-year
    runs write one foreground database per year; each foreground activity
    records that year in its name and Brightway metadata.

    Args:
        years: One year or an iterable of years to model. Time selections are
            reused at the corresponding dates when several years are supplied.
        countries: Country codes for which consumption inventories are created,
            for example ``["DE", "FR", "NL"]``.
        project_name: Brightway project that receives the database when
            :meth:`write` is called.
        bg_db_name: Existing ecoinvent or premise-modified ecoinvent background
            database name. For several years, provide a ``{year: name}``
            mapping or a string containing ``"{year}"``.
        my_db_name: Name of the database to create. A year suffix is added
            automatically for multi-year runs unless names are supplied as a
            mapping or with a ``"{year}"`` placeholder.
        scenario: TYNDP scenario code, such as ``"DE"``. Required whenever a
            configured year uses TYNDP.
        climate_year: Weather year used by TYNDP. Required for TYNDP years.
        iam: IAM geography model used by premise, currently typically
            ``"remind-eu"``.
        time_range: Two timestamps defining an inclusive period. For a
            multi-year run, their month, day, and time are reused in each year.
        hour_range: Optional inclusive daily hour range within ``time_range``,
            for example ``[10, 14]``.
        times: Exact, potentially disconnected timestamps to select instead of
            ``time_range``.
        source: ``"auto"``, ``"energy_charts"``, or ``"tyndp"``.
        cutoff: Minimum activity share retained as an individual row.
        include_cutoff: If true, combine shares below ``cutoff`` into a
            residual high-voltage electricity activity.
        strict: If true, :meth:`write` raises when a required background
            activity cannot be matched uniquely after geographic fallbacks.
        network: Include fixed electricity transmission and distribution
            infrastructure exchanges in the written inventories.
        zero_consumption: Treatment of zero-consumption TYNDP country-hours:
            ``"keep_zero"``, ``"month_hour_average"``, or ``"raise"``.
        include_consumption_mix_volume: Retain the resolved four-dimensional
            consumption volumes as well as normalized shares. Disable this to
            reduce memory and cache size when only inventory shares are needed.
        data_dir: Optional local cache directory. The platform-specific SHRECC
            user-data directory is used by default.
        technology_mapping: Optional replacement TYNDP technology concordance.
        country_mapping: Optional replacement TYNDP node and edge mappings.
        ecoinvent_mapping: Optional replacement ecoinvent allocation table.
        topology_files: Optional IAM-to-topology-file mapping used for premise
            region lookup.
        download: Permit missing source data to be downloaded.
        engine: Spreadsheet engine used to read TYNDP workbooks.
        check: Run conservation and consistency checks during calculation.
        verbose: Print progress information.

    After creation, intermediate and final objects remain available by year in
    ``consumption_results``, ``activity_mixes``, and ``database_tables``. The
    historical pre-cutoff diagnostics are retained in ``mapping_gap_reports``;
    :meth:`mapping_report` returns a defensive copy. Successfully written
    output names are recorded in ``written_database_names``.

    Examples:
        >>> electricity = NewDatabase(
        ...     years=2025,
        ...     countries=["DE", "FR"],
        ...     project_name="my-project",
        ...     bg_db_name="ecoinvent-3.11-cutoff",
        ...     my_db_name="shrecc_2025",
        ...     time_range=[
        ...         "2025-06-01 00:00:00",
        ...         "2025-06-30 23:00:00",
        ...     ],
        ... )
        >>> electricity.create()
        NewDatabase(years=[2025], sources={2025:energy_charts})
        >>> electricity.table().columns.tolist()
        ['DE', 'FR']
    """

    def __init__(
        self,
        *,
        years,
        countries,
        project_name,
        bg_db_name,
        my_db_name,
        scenario=None,
        climate_year=None,
        iam="remind-eu",
        time_range=None,
        hour_range=None,
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

        self.time_range = time_range
        self.hour_range = hour_range
        self.times = times
        _validate_time_selection(time_range, hour_range, times)

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
            bg_db_name,
            self.years,
            "bg_db_name",
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
        self.mapping_gaps = {}
        self.mapping_gap_reports = {}
        self.database_tables = {}
        self.written_database_names = {}

    def create(self):
        """Prepare mapped inventory tables without changing Brightway.

        Source data are acquired or loaded from cache, the interconnected
        electricity system is solved, technologies are mapped to background
        activities, the requested times are aggregated, and the cutoff is
        applied. Existing results on this object are replaced, so calling this
        method again rebuilds every configured year and can fill newly cached
        source data.

        Returns:
            This ``NewDatabase`` instance, allowing method chaining.
        """
        self.consumption_results.clear()
        self.activity_mixes.clear()
        self.exchange_geography_maps.clear()
        self.mapping_gaps.clear()
        self.mapping_gap_reports.clear()
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
        """Write the prepared inventories to the configured Brightway project.

        :meth:`create` is called automatically if not all configured years have
        prepared tables. Each configured year is written to its own output
        database, replacing an existing database with the same name.
        Foreground activity names and metadata record the modeled year.

        Returns:
            This ``NewDatabase`` instance, allowing method chaining.

        Raises:
            ValueError: If ``strict=True`` and a required background activity
                cannot be matched uniquely.
        """
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
                year=year,
            )
            self.written_database_names[year] = self.database_names[year]
        return self

    def table(self, year=None):
        """Return the final database input table for one year.

        Args:
            year: Requested year. It may be omitted for a single-year object.

        Returns:
            DataFrame whose rows are background activities and whose columns
            are consumer-country inventories. Values are electricity shares
            after temporal aggregation and cutoff treatment.

        Raises:
            RuntimeError: If :meth:`create` has not prepared the table.
            ValueError: If ``year`` is omitted for a multi-year object.
        """
        year = self._resolve_result_year(year)
        if year not in self.database_tables:
            raise RuntimeError("create() must be called before accessing tables")
        return self.database_tables[year]

    def results(self, year=None):
        """Return production, trade, consumption, and mix results for one year.

        Args:
            year: Requested year. It may be omitted for a single-year object.

        Returns:
            Canonical xarray Dataset before activity mapping and cutoff.

        Raises:
            RuntimeError: If :meth:`create` has not prepared the results.
            ValueError: If ``year`` is omitted for a multi-year object.
        """
        year = self._resolve_result_year(year)
        if year not in self.consumption_results:
            raise RuntimeError("create() must be called before accessing results")
        return self.consumption_results[year]

    def mapping_report(self, year=None):
        """Return source technologies that required fallback mapping.

        The report is calculated after time selection but before cutoff. Rows
        identify ``(source_country, technology)`` and columns identify consumer
        countries. Each value is the mean share of that consumer's mix assigned
        to a source-country high-voltage production mix because no direct
        activity mapping was available.

        TYNDP mappings are validated during creation and therefore return an
        empty report instead of using this historical-data fallback.

        Args:
            year: Requested year. It may be omitted for a single-year object.

        Returns:
            DataFrame of dimensionless fallback shares. Modifying it does not
            alter the report stored on this object.

        Raises:
            RuntimeError: If :meth:`create` has not prepared the report.
            ValueError: If ``year`` is omitted for a multi-year object.
        """
        year = self._resolve_result_year(year)
        if year not in self.mapping_gap_reports:
            raise RuntimeError("create() must be called before accessing reports")
        return self.mapping_gap_reports[year].copy()

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
        time_range, times = self._selection_for_year(year)
        results = load_consumption_result_cache(
            cache_dir,
            general_range=time_range,
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
                general_range=time_range,
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

        activity_mapping = load_ecoinvent_mapping(Path(self.ecoinvent_mapping))
        activity_mix, mapping_gaps = map_consumption_mix_to_ecoinvent_activities(
            results["consumption_mix"],
            activity_mapping,
            check=self.check,
            return_mapping_gaps=True,
        )
        mapping_report = mapping_gap_to_report(
            mapping_gaps,
            countries=self.countries,
            general_range=time_range,
            refined_range=self.hour_range,
            freq="h",
            times=times,
        )
        self.mapping_gaps[year] = mapping_gaps
        self.mapping_gap_reports[year] = mapping_report
        self._warn_mapping_gaps(year, mapping_report)
        table = activity_mix_to_database_table(
            activity_mix,
            countries=self.countries,
            general_range=time_range,
            refined_range=self.hour_range,
            freq="h",
            times=times,
        )
        self._store_year(year, results, activity_mix, table)

    def _create_tyndp_year(self, year):
        time_range, times = self._selection_for_year(year)
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
            time_range=time_range,
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
            general_range=time_range,
            refined_range=self.hour_range,
            freq="h",
            times=times,
        )
        self.exchange_geography_maps[year] = exchange_geography_map
        self.mapping_gap_reports[year] = pd.DataFrame(
            columns=self.countries,
            dtype=float,
        )
        self._store_year(year, results, activity_mix, table)

    def _warn_mapping_gaps(self, year, report):
        if report.empty:
            if self.verbose:
                print(f"No activity mapping gaps found for {year}")
            return

        consumer_totals = report.sum(axis=0).sort_values(ascending=False)
        consumer_summary = ", ".join(
            f"{country} {share:.1%}"
            for country, share in consumer_totals.head(5).items()
        )
        largest_gap = report.max(axis=1).sort_values(ascending=False)
        gap_summary = ", ".join(
            f"{country}/{technology} {share:.1%}"
            for (country, technology), share in largest_gap.head(5).items()
        )
        warnings.warn(
            f"Energy Charts activity mapping gaps for {year} were assigned "
            "to source-country high-voltage production mixes. Mean fallback "
            f"share by consumer: {consumer_summary}. Largest source-technology "
            f"gaps: {gap_summary}. See mapping_report({year}) for the full "
            "report.",
            stacklevel=4,
        )

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
        return Path(get_package_user_data_dir())

    def _tyndp_cache_dir(self):
        return self._data_root() / "tyndp" / "cache"

    def _selection_for_year(self, year):
        if self.time_range is not None:
            timestamps = pd.to_datetime(self.time_range)
            if len(self.years) == 1:
                if any(timestamp.year != year for timestamp in timestamps):
                    raise ValueError(
                        f"time_range timestamps must belong to year {year}"
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


def _validate_time_selection(time_range, hour_range, times):
    if time_range is None and times is None:
        raise ValueError("Either time_range or times must be provided")
    if time_range is not None and times is not None:
        raise ValueError("Use either time_range or times, not both")
    if time_range is not None and len(time_range) != 2:
        raise ValueError("time_range must contain a start and end timestamp")
    if times is not None and not len(times):
        raise ValueError("times must contain at least one timestamp")
    if hour_range is not None and time_range is None:
        raise ValueError("time_range is required when hour_range is supplied")
    if hour_range is not None:
        if len(hour_range) != 2:
            raise ValueError("hour_range must contain a start and end hour")
        start_hour, end_hour = hour_range
        if not 0 <= start_hour <= end_hour <= 23:
            raise ValueError(
                "hour_range must contain inclusive hours between 0 and 23"
            )


def _select_dataframe_times(dataframe, *, time_range, times):
    if time_range is not None:
        selected = dataframe.loc[time_range[0] : time_range[1]]
    else:
        selected = dataframe.loc[dataframe.index.isin(times)]
    if selected.empty:
        raise ValueError("The requested time selection contains no source data")
    return selected
