SHRECC CHANGELOG
================

## Unreleased

### Added

- Add `consumption_profile` to `NewDatabase`, with equal hourly weighting
  through `"flat"`, country-specific volume weighting through
  `"national_demand"`, and custom timestamp weights supplied as a pandas
  Series. `"flat"` remains the backwards-compatible default.
- Validate custom profiles for timestamp uniqueness, finite non-negative
  weights, source-data coverage, and positive totals. A profile contained
  within one calendar year is reusable as a template for every configured
  model year.
- Record the selected consumption-profile method as metadata on written
  Brightway foreground activities.
- Add reusable analysis helpers for resolving delivered electricity to
  generation technologies and comparing SHRECC mixes and LCIA scores with
  ecoinvent or premise backgrounds.
- Add `retain_hourly_results=False` as a memory-efficient database-building
  mode when canonical hourly datasets do not need to remain on the
  `NewDatabase` object.

### Changed

- Aggregate selected canonical consumption mixes with their profile weights
  before mapping activities. This is algebraically equivalent to aggregating
  mapped hourly results while avoiding large hourly activity arrays.
- Let a custom Series define the modeled timestamps. Explicit `times`,
  `time_range`, and `hour_range` arguments are ignored with a warning.
- Drop February 29 with a warning when a custom profile is rebased to a
  non-leap model year, and drop unavailable TYNDP December 31 timestamps with
  a warning instead of treating them as zero consumption.
- Rebase multi-year temporal selections safely across leap and non-leap years:
  clip February 29 range boundaries to February 28 while dropping invalid
  explicit February 29 timestamps, with warnings in both cases.
- Reuse the shared analysis helpers in the annual-validation and
  consumption-profile notebooks, and disable optional four-dimensional volume
  results there to avoid multi-gigabyte duplicate arrays.

## 0.1.0.dev2 - 2026-07-27

- Update stale links in README.md

## 0.1.0.dev1 - 2026-07-27

- Add example notebooks

## 0.1.0 - 2026-07-22

This release unifies historical SHRECC and prospective TYNDP/FIONA processing
behind one public pipeline while preserving the established numerical and
mapping behavior.

### Added

- Add `shrecc.NewDatabase` as the high-level interface for creating mapped
  electricity tables and writing Brightway databases.
- Support historical Energy Charts data and prospective ENTSO-E TYNDP scenarios
  through the same workflow, with automatic source selection by year.
- Add a shared, source-independent country-network solver using canonical
  `production_volume`, `trade_volume`, and optional measured
  `consumption_volume` inputs.
- Retain normalized `consumption_mix` alongside physical
  `consumption_mix_volume`, production, trade, and consumption volumes for
  inspection, reporting, and visualization.
- Add reduced interconnected-node solving, time-chunked execution, compressed
  result caches, manifests, and selective date loading.
- Add TYNDP scenario validation, acquisition, parsing, country aggregation, and
  premise/REMIND-EU activity mapping.
- Add `results(year)` and `table(year)` accessors so intermediate results can be
  inspected before Brightway is modified.
- Add a minimal getting-started notebook and an expanded historical and
  prospective analysis notebook.
- Add an annual European validation notebook comparing SHRECC technology mixes
  and climate-change results with ecoinvent and premise backgrounds.
- Record the model year in written foreground activity names and Brightway
  activity metadata.

### Changed

- Use the same graph algebra for Energy Charts and TYNDP instead of maintaining
  separate full- and reduced-matrix implementations.
- Separate source acquisition, solving, activity mapping, persistence, and
  Brightway writing into focused modules.
- Make `create()` prepare and validate database tables without modifying
  Brightway; database changes occur only through `write()`.
- Rename the high-level background database argument to `bg_db_name` and
  simplify temporal selection to `times`, `time_range`, and `hour_range`.
- Reuse country-specific ecoinvent shares for established one-to-many TYNDP
  technologies and map unique prospective activities directly.
- Prefer source-country activity geography when available, then fall back to the
  corresponding premise/IAM region while preserving the source technology.
- Use measured TYNDP demand for `consumption_volume` while retaining gross
  production plus imports for consumption-mix attribution.
- Harmonize temporal filtering and cutoff handling for hourly, daily, weekly,
  monthly, range-based, and explicit timestamp selections.
- Replace runtime TYNDP XLSX concordances with packaged CSV/JSON resources.
- Read TYNDP `.xlsb` workbooks with `calamine` and remove the `pyxlsb` runtime
  dependency.
- Move downloaded and generated data to local user-data caches and keep only
  required mapping resources inside the installed package.
- Map historical country-technology allocations with a chunked sparse operator
  and restrict activity mapping to the requested consumer countries.

### Fixed

- Resume Energy Charts acquisition when cached data is missing requested
  countries, including Netherlands and Cyprus.
- Handle HTTP 429 responses using `Retry-After` and stop treating unavailable
  country timeouts as indefinitely retryable requests.
- Prefer measured demand over an inferred balance when TYNDP exports would
  otherwise produce negative consumption volumes.
- Preserve unresolved export origins instead of silently changing their
  technology attribution.
- Add country aliases for TYNDP `UK`/`NIE` and ecoinvent `GB` geography.
- Keep country-specific commercial photovoltaic inventories at country level.
- When `strict=False`, fall back to another geography for the same activity
  rather than substituting a different technology or dropping the exchange.
- Repair the Sphinx AutoAPI dependency combination so API documentation renders
  correctly on Windows.

### Compatibility and packaging

- Preserve established imports through compatibility facades in `download.py`,
  `activity_mapping.py`, `treatment.py`, and `treatment_fiona.py`.
- Isolate legacy reference implementations for regression comparison without
  making the new pipeline depend on them.
- Separate runtime, testing, development, premise, notebook, and documentation
  dependencies into optional installation groups.
- Retain the NED.nl-enhanced ecoinvent concordance and its editable source table
  for a future Netherlands fallback adapter.
- Expand automated coverage for the shared solver, source adapters, mapping,
  caching, pipeline orchestration, compatibility imports, and database fallback.

## 0.0.5 - 2026-02-04

+ fix issue #30 - add production exchanges.
+ fix issue #39 - allow `times` in filter_by_times to be of type DataIndex

## 0.0.4 - 2026-01-29

+ fix issue # 23 - add functions to parse the mix from ned.l
+ fix issue # 21 - make the package compatible with python 3.13
+ fix issue # 25 - network activities well taken into account for ei > 3.9, up to 3.12
+ fix issue # 28 - update naming convention for some CH located activities
+ new - data processing shows timestamps to know how long it takes to process

## [0.0.3] - 2025-09-23

+ remove dev and docs dependencies from the main package
