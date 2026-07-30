"""High-level orchestration for creating SHRECC electricity databases."""

from importlib.resources import files
from pathlib import Path
import shutil
import warnings

import bw2data as bd
import pandas as pd
import xarray as xr

from shrecc.database import apply_cutoff, create_database
from shrecc.energy_charts import (
    # Aliases distinguish orchestration dependencies from their implementations.
    data_processing as process_energy_charts_data,
    energy_charts_cached_countries,
    get_data as get_energy_charts_data,
)
from shrecc.mapping import (
    activity_mix_to_database_table,
    aggregate_consumption_mix,
    load_ecoinvent_mapping,
    map_consumption_mix_to_ecoinvent_activities,
    mapping_gap_to_report,
    validate_consumption_profile,
    validate_inventory_resolution,
)
from shrecc.lcia import (
    LCIAResults,
    build_lcia_dataset,
    build_resolved_inventory_basis,
    calculate_lcia,
    consumption_profile_weights,
    resolve_lcia_methods,
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

    The workflow is deliberately staged. :meth:`create` downloads or reads
    source data, solves the electricity system, and prepares inspectable
    in-memory tables. :meth:`write` writes annual or monthly tables to
    Brightway. :meth:`lcia` calculates hourly impacts without writing hourly
    activities. Calling :meth:`create` alone does not modify a Brightway
    project.

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
        time_range: Two timestamps defining an inclusive period. When omitted,
            the complete first configured year is used. For a multi-year run,
            the selected month, day, and time are reused in each year. A
            February 29 boundary is clipped to February 28 in non-leap years.
            Ignored with a warning when a custom consumption Series is used.
        hour_range: Optional inclusive daily hour range within ``time_range``,
            for example ``[10, 14]``. Ignored with a custom Series.
        times: Exact, potentially disconnected timestamps to select instead of
            ``time_range``. Explicit February 29 timestamps are dropped when
            reused in non-leap years. Ignored with a custom Series.
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
        consumption_profile: Temporal weighting used to aggregate the selected
            hourly consumption mixes. Use ``"flat"`` for equal hourly
            weighting, ``"national_demand"`` for country-specific demand
            weighting, or a pandas Series of custom timestamp weights. A
            custom Series defines its own timestamps and a Series contained
            within one calendar year is reused as a template for every model
            year.
        inventory_resolution: Temporal resolution of written foreground
            activities: ``"annual"`` (default) or ``"monthly"``.
            ``"yearly"`` is accepted as an alias for ``"annual"``. Hourly
            results remain available for calculation through :meth:`lcia`.
        include_consumption_mix_volume: Retain the resolved four-dimensional
            consumption volumes as well as normalized shares. Disable this to
            reduce memory and cache size when only inventory shares are needed.
        retain_hourly_results: Keep canonical hourly result datasets on this
            object after aggregation. Disable this for memory-efficient
            database creation when only mapped tables and reports are needed.
            Hourly LCIA requires this option to remain enabled.
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

    After creation, hourly canonical results remain available in
    ``consumption_results`` unless ``retain_hourly_results=False``. The mapped
    inventories at the configured resolution and final tables are available by
    year in ``activity_mixes`` and ``database_tables``. Historical pre-cutoff
    diagnostics are retained in ``mapping_gap_reports``;
    :meth:`mapping_report` returns a defensive copy. Successfully written
    output names are recorded in ``written_database_names``.
    After :meth:`lcia`, labelled temporal assessment results are retained in
    ``lcia_results``.

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
        consumption_profile="flat",
        inventory_resolution="annual",
        include_consumption_mix_volume=True,
        retain_hourly_results=True,
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

        self.inventory_resolution = validate_inventory_resolution(
            inventory_resolution
        )
        self.consumption_profile = validate_consumption_profile(
            consumption_profile
        )
        profile_defines_time = isinstance(self.consumption_profile, pd.Series)
        if not profile_defines_time and time_range is None and times is None:
            first_year = self.years[0]
            time_range = [
                f"{first_year}-01-01 00:00:00",
                f"{first_year}-12-31 23:00:00",
            ]
        _validate_time_selection(
            time_range,
            hour_range,
            times,
            custom_profile=profile_defines_time,
        )
        if profile_defines_time:
            self.time_range = None
            self.hour_range = None
            self.times = None
        else:
            self.time_range = time_range
            self.hour_range = hour_range
            self.times = times
        self._consumption_profiles_by_year = _resolve_consumption_profiles(
            self.consumption_profile,
            self.years,
        )

        self.cutoff = float(cutoff)
        if self.cutoff < 0:
            raise ValueError("cutoff cannot be negative")
        self.include_cutoff = bool(include_cutoff)
        self.strict = bool(strict)
        self.network = network
        self.zero_consumption = zero_consumption
        self.include_consumption_mix_volume = bool(include_consumption_mix_volume)
        self.retain_hourly_results = bool(retain_hourly_results)
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
        self.lcia_results = None

    def create(self):
        """Prepare mapped inventory tables without changing Brightway.

        Source data are acquired or loaded from cache, the interconnected
        electricity system is solved, and the requested times are resolved to
        annual or monthly inventories. Technologies are then mapped to
        background activities and the cutoff is applied. Existing results
        on this object are replaced, so calling this method again rebuilds
        every configured year and can fill newly cached source data.

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
        self.lcia_results = None

        for year in self.years:
            if self.verbose:
                print(f"Creating {self.sources[year]} electricity mix for {year}")
            if self.sources[year] == "tyndp":
                self._create_tyndp_year(year)
            else:
                self._create_energy_charts_year(year)
        return self

    def lcia(self, methods=None, *, engine="linear", batch_size=1000):
        """Calculate hourly LCIA without writing hourly Brightway activities.

        The default ``linear`` engine scores each unique background input once
        and multiplies those scores by the hourly mapped coefficients. The
        ``multilca`` engine instead submits composite hourly functional units
        to Brightway in bounded batches and is primarily intended for
        validation.

        Args:
            methods: Brightway method tuple or iterable of method tuples.
                When omitted, all installed ``EF v3.1`` methods are used.
            engine: ``"linear"`` (default) or ``"multilca"``.
            batch_size: Number of composite functional units per MultiLCA
                batch. Ignored by the linear engine.

        Returns:
            :class:`shrecc.lcia.LCIAResults` with hourly intensities and
            profile-weighted annual and monthly accessors.

        Raises:
            RuntimeError: If hourly canonical results were not retained.
        """
        if not self.retain_hourly_results:
            raise RuntimeError(
                "lcia() requires retain_hourly_results=True"
            )
        if set(self.consumption_results) != set(self.years):
            self.create()

        bd.projects.set_current(self.project_name)
        resolved_methods = resolve_lcia_methods(methods)
        results_by_year = {}
        for year in self.years:
            if self.verbose:
                print(
                    f"Calculating hourly LCIA for {year} with the {engine} engine"
                )
            country_intensities = []
            source_score_cache = {}
            for country in self.countries:
                hourly_table = self._hourly_database_table(
                    year,
                    countries=[country],
                )
                basis = build_resolved_inventory_basis(
                    hourly_table,
                    year=year,
                    background_database=self.background_databases[year],
                    include_network=self.network,
                    strict=self.strict,
                )
                country_intensities.append(
                    calculate_lcia(
                        basis,
                        resolved_methods,
                        engine=engine,
                        batch_size=batch_size,
                        source_score_cache=source_score_cache,
                    )
                )
            intensity = xr.concat(
                country_intensities,
                dim="consumer_country",
            ).sel(consumer_country=list(self.countries))
            weights = consumption_profile_weights(
                self.consumption_results[year],
                self._consumption_profiles_by_year[year],
            )
            results_by_year[year] = build_lcia_dataset(
                intensity,
                weights,
                year=year,
                engine=engine,
            )

        self.lcia_results = LCIAResults(
            results_by_year,
            resolved_methods,
            engine,
        )
        return self.lcia_results

    def write(self):
        """Write the prepared inventories to the configured Brightway project.

        :meth:`create` is called automatically if not all configured years have
        prepared tables. Each configured year is written to its own output
        database, replacing an existing database with the same name.
        Foreground activity names and metadata record the modeled period and
        inventory resolution.

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
                consumption_profile=self._consumption_profile_label(),
                inventory_resolution=self.inventory_resolution,
            )
            self.written_database_names[year] = self.database_names[year]
        return self

    def table(self, year=None):
        """Return the final database input table for one year.

        Args:
            year: Requested year. It may be omitted for a single-year object.

        Returns:
            DataFrame whose rows are background activities and whose columns
            are consumer-country inventories. Monthly and hourly tables use
            ``(time, country)`` columns. Values are electricity shares after
            temporal resolution and cutoff treatment.

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
            RuntimeError: If :meth:`create` has not prepared the results or
                hourly results were not retained.
            ValueError: If ``year`` is omitted for a multi-year object.
        """
        year = self._resolve_result_year(year)
        if not self.retain_hourly_results:
            raise RuntimeError(
                "Hourly results were not retained. Initialize NewDatabase "
                "with retain_hourly_results=True to access results()."
            )
        if year not in self.consumption_results:
            raise RuntimeError("create() must be called before accessing results")
        return self.consumption_results[year]

    def mapping_report(self, year=None):
        """Return source technologies that required fallback mapping.

        The report is calculated after time selection and temporal resolution
        but before cutoff. Rows identify
        ``(source_country, technology)`` and columns identify consumer
        countries. Each value is the share of that consumer's inventory
        assigned to a source-country high-voltage production mix because no
        direct activity mapping was available.

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
        consumption_profile = self._consumption_profiles_by_year[year]
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
        _validate_custom_profile_source_coverage(
            consumption_profile,
            results["consumption_mix"]["time"].to_index(),
            source_name="Energy Charts",
            year=year,
        )
        if (
            self.include_consumption_mix_volume
            and "consumption_mix_volume" not in results
        ):
            results["consumption_mix_volume"] = (
                results["consumption_mix"] * results["consumption_volume"]
            ).assign_attrs(unit=results.attrs.get("volume_unit", "MWh"))

        aggregated_consumption_mix = aggregate_consumption_mix(
            results["consumption_mix"],
            consumption_profile=consumption_profile,
            consumption_volume=results["consumption_volume"],
            inventory_resolution=self.inventory_resolution,
            countries=self.countries,
            general_range=time_range,
            refined_range=self.hour_range,
            freq="h",
            times=times,
        )
        activity_mapping = load_ecoinvent_mapping(Path(self.ecoinvent_mapping))
        activity_mix, mapping_gaps = map_consumption_mix_to_ecoinvent_activities(
            aggregated_consumption_mix,
            activity_mapping,
            check=self.check,
            return_mapping_gaps=True,
        )
        mapping_report = mapping_gap_to_report(
            mapping_gaps,
            countries=self.countries,
        )
        self.mapping_gaps[year] = mapping_gaps
        self.mapping_gap_reports[year] = mapping_report
        self._warn_mapping_gaps(year, mapping_report)
        table = activity_mix_to_database_table(
            activity_mix,
            countries=self.countries,
            inventory_resolution=self.inventory_resolution,
        )
        self._store_year(year, results, activity_mix, table)

    def _create_tyndp_year(self, year):
        consumption_profile = self._consumption_profiles_by_year[year]
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
        _validate_tyndp_country_coverage(
            Z_gross,
            countries=self.countries,
            scenario=self.scenario,
            year=year,
            climate_year=self.climate_year,
        )
        if isinstance(consumption_profile, pd.Series):
            consumption_profile = _adapt_profile_to_tyndp_times(
                consumption_profile,
                Z_gross.index,
                year=year,
            )
            times = _positive_profile_times(consumption_profile, year)
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

        aggregated_consumption_mix = aggregate_consumption_mix(
            results["consumption_mix"],
            consumption_profile=consumption_profile,
            consumption_volume=results["consumption_volume"],
            inventory_resolution=self.inventory_resolution,
            countries=self.countries,
            general_range=time_range,
            refined_range=self.hour_range,
            freq="h",
            times=times,
        )

        mapper = PremiseConsumptionMixMapper(
            technology_mapping=self.technology_mapping,
            ecoinvent_mapping=self.ecoinvent_mapping,
            iam_model=self.iam,
            topology_files=self.topology_files,
            on_missing_model="warn",
        )
        activity_mix = mapper.map_technologies(
            aggregated_consumption_mix,
            check=self.check,
        )
        exchange_geography_map = mapper.build_exchange_geography_map(activity_mix)
        table = premise_activity_mix_to_database_table(
            activity_mix,
            exchange_geography_map,
            countries=self.countries,
            inventory_resolution=self.inventory_resolution,
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
            "to source-country high-voltage production mixes. Fallback "
            f"share by consumer: {consumer_summary}. Largest source-technology "
            f"gaps: {gap_summary}. See mapping_report({year}) for the full "
            "report.",
            stacklevel=4,
        )

    def _hourly_database_table(self, year, *, countries=None):
        """Map retained hourly results to background activity table rows."""
        countries = list(countries or self.countries)
        consumption_mix = self.consumption_results[year]["consumption_mix"].sel(
            consumer_country=countries
        )
        if self.sources[year] == "energy_charts":
            activity_mapping = load_ecoinvent_mapping(
                Path(self.ecoinvent_mapping)
            )
            activity_mix = map_consumption_mix_to_ecoinvent_activities(
                consumption_mix,
                activity_mapping,
                check=self.check,
            )
            return activity_mix_to_database_table(
                activity_mix,
                countries=countries,
                inventory_resolution="annual",
                preserve_time=True,
            )

        mapper = PremiseConsumptionMixMapper(
            technology_mapping=self.technology_mapping,
            ecoinvent_mapping=self.ecoinvent_mapping,
            iam_model=self.iam,
            topology_files=self.topology_files,
            on_missing_model="warn",
        )
        activity_mix = mapper.map_technologies(
            consumption_mix,
            check=self.check,
        )
        exchange_geography_map = mapper.build_exchange_geography_map(
            activity_mix
        )
        return premise_activity_mix_to_database_table(
            activity_mix,
            exchange_geography_map,
            countries=countries,
            inventory_resolution="annual",
            preserve_time=True,
        )

    def _store_year(self, year, results, activity_mix, table):
        if self.retain_hourly_results:
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

    def _consumption_profile_label(self):
        if isinstance(self.consumption_profile, str):
            return self.consumption_profile
        if self.consumption_profile.name is None:
            return "custom"
        return f"custom: {self.consumption_profile.name}"

    def _data_root(self):
        if self.data_dir is not None:
            return self.data_dir
        return Path(get_package_user_data_dir())

    def _tyndp_cache_dir(self):
        return self._data_root() / "tyndp" / "cache"

    def _selection_for_year(self, year):
        consumption_profile = self._consumption_profiles_by_year[year]
        if isinstance(consumption_profile, pd.Series):
            return None, _positive_profile_times(consumption_profile, year)

        if self.time_range is not None:
            timestamps = pd.to_datetime(self.time_range)
            if len(self.years) == 1:
                if any(timestamp.year != year for timestamp in timestamps):
                    raise ValueError(
                        f"time_range timestamps must belong to year {year}"
                    )
            else:
                timestamps = _rebase_time_range(
                    timestamps,
                    year,
                )
            if timestamps[-1] < timestamps[0]:
                raise ValueError(
                    "time_range end must not precede its start after "
                    f"rebasing to {year}"
                )
            return list(timestamps), None

        timestamps = pd.DatetimeIndex(pd.to_datetime(self.times))
        if len(self.years) == 1:
            if any(timestamp.year != year for timestamp in timestamps):
                raise ValueError(f"times must belong to year {year}")
        else:
            timestamps = _rebase_explicit_times(timestamps, year)
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


def _validate_time_selection(
    time_range,
    hour_range,
    times,
    *,
    custom_profile=False,
):
    if custom_profile:
        ignored = [
            name
            for name, value in (
                ("times", times),
                ("time_range", time_range),
                ("hour_range", hour_range),
            )
            if value is not None
        ]
        if ignored:
            warnings.warn(
                "A custom consumption_profile defines its timestamps. "
                "Ignoring: " + ", ".join(ignored) + ".",
                stacklevel=3,
            )
        return

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


def _resolve_consumption_profiles(consumption_profile, years):
    if isinstance(consumption_profile, str):
        return {year: consumption_profile for year in years}

    profile_years = pd.Index(consumption_profile.index.year).unique()
    if len(profile_years) == 1:
        return {
            year: _rebase_consumption_profile(consumption_profile, year)
            for year in years
        }

    configured_years = set(years)
    available_years = set(map(int, profile_years))
    missing_years = configured_years.difference(available_years)
    if missing_years:
        raise ValueError(
            "Custom multi-year consumption_profile is missing configured "
            "years: " + ", ".join(map(str, sorted(missing_years)))
        )
    extra_years = available_years.difference(configured_years)
    if extra_years:
        warnings.warn(
            "Custom consumption_profile contains years that are not "
            "configured and will be ignored: "
            + ", ".join(map(str, sorted(extra_years))),
            stacklevel=3,
        )
    return {
        year: consumption_profile.loc[
            consumption_profile.index.year == year
        ].copy()
        for year in years
    }


def _rebase_time_range(timestamps, year):
    rebased = []
    clipped_february_29 = 0
    for timestamp in timestamps:
        try:
            rebased.append(timestamp.replace(year=year))
        except ValueError:
            if timestamp.month != 2 or timestamp.day != 29:
                raise
            rebased.append(timestamp.replace(year=year, day=28))
            clipped_february_29 += 1

    if clipped_february_29:
        warnings.warn(
            f"Clipped {clipped_february_29} February 29 time_range "
            f"boundary timestamp(s) to February 28 for non-leap model "
            f"year {year}.",
            stacklevel=3,
        )
    return pd.DatetimeIndex(rebased)


def _rebase_explicit_times(timestamps, year):
    rebased = []
    dropped_february_29 = 0
    for timestamp in timestamps:
        try:
            rebased.append(timestamp.replace(year=year))
        except ValueError:
            if timestamp.month != 2 or timestamp.day != 29:
                raise
            dropped_february_29 += 1

    if dropped_february_29:
        warnings.warn(
            f"Dropped {dropped_february_29} explicit February 29 "
            f"timestamp(s) while rebasing to non-leap model year {year}.",
            stacklevel=3,
        )
    if not rebased:
        raise ValueError(
            f"No explicit times remain after rebasing to model year {year}"
        )
    return pd.DatetimeIndex(rebased)


def _rebase_consumption_profile(consumption_profile, year):
    reference_year = int(consumption_profile.index.year[0])
    if reference_year == year:
        return consumption_profile.copy()

    rebased_timestamps = []
    retained_positions = []
    dropped_february_29 = 0
    for position, timestamp in enumerate(consumption_profile.index):
        try:
            rebased = timestamp.replace(year=year)
        except ValueError:
            if timestamp.month == 2 and timestamp.day == 29:
                dropped_february_29 += 1
                continue
            raise
        retained_positions.append(position)
        rebased_timestamps.append(rebased)

    if dropped_february_29:
        warnings.warn(
            f"Dropped {dropped_february_29} February 29 custom-profile "
            f"timestamp(s) while rebasing from {reference_year} to "
            f"non-leap model year {year}.",
            stacklevel=4,
        )

    rebased_profile = consumption_profile.iloc[retained_positions].copy()
    rebased_profile.index = pd.DatetimeIndex(
        rebased_timestamps,
        name=consumption_profile.index.name,
    )
    return rebased_profile


def _positive_profile_times(consumption_profile, year):
    positive = consumption_profile[consumption_profile > 0]
    if positive.empty:
        raise ValueError(
            f"Custom consumption_profile has no positive weights for {year}"
        )
    return positive.index


def _validate_custom_profile_source_coverage(
    consumption_profile,
    available_times,
    *,
    source_name,
    year,
):
    if not isinstance(consumption_profile, pd.Series):
        return
    requested_times = _positive_profile_times(consumption_profile, year)
    missing_times = requested_times.difference(pd.DatetimeIndex(available_times))
    if missing_times.empty:
        return
    examples = ", ".join(map(str, missing_times[:5]))
    raise ValueError(
        f"{source_name} data for {year} do not contain positive-weight custom "
        "consumption_profile timestamps: " + examples
    )


def _adapt_profile_to_tyndp_times(consumption_profile, available_times, *, year):
    available_times = pd.DatetimeIndex(available_times)
    missing_times = consumption_profile.index.difference(available_times)
    if missing_times.empty:
        return consumption_profile

    december_31 = missing_times[
        (missing_times.month == 12) & (missing_times.day == 31)
    ]
    if not december_31.empty:
        warnings.warn(
            f"TYNDP data for {year} do not include December 31; dropped "
            f"{len(december_31)} custom-profile timestamp(s) from that day.",
            stacklevel=3,
        )
        consumption_profile = consumption_profile.drop(december_31)

    _validate_custom_profile_source_coverage(
        consumption_profile,
        available_times,
        source_name="TYNDP",
        year=year,
    )
    return consumption_profile


def _select_dataframe_times(dataframe, *, time_range, times):
    if time_range is not None:
        selected = dataframe.loc[time_range[0] : time_range[1]]
    else:
        selected = dataframe.loc[dataframe.index.isin(times)]
    if selected.empty:
        raise ValueError("The requested time selection contains no source data")
    return selected


def _validate_tyndp_country_coverage(
    Z_gross,
    *,
    countries,
    scenario,
    year,
    climate_year,
):
    """Raise before solving when a TYNDP scenario omits requested countries."""
    if not isinstance(Z_gross.columns, pd.MultiIndex):
        return
    country_levels = {"country from", "country to"}
    if not country_levels.issubset(Z_gross.columns.names):
        return

    available = set()
    for level in country_levels:
        available.update(
            str(country)
            for country in Z_gross.columns.get_level_values(level)
            if pd.notna(country)
        )
    missing = sorted(set(countries).difference(available))
    if not missing:
        return

    raise ValueError(
        f"TYNDP scenario {scenario!r} for {year} with climate year "
        f"{climate_year} does not contain requested countries: "
        + ", ".join(missing)
        + ". Remove them from countries or use a scenario that models them."
    )
