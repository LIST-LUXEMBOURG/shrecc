# Refactoring summary

This document records the architectural refactoring performed after commit
`2433247` (`last working push before rewriting the vanilla-shrecc data
treatment`). The numerical intent of SHRECC was preserved while the historical
Energy Charts and prospective TYNDP paths were brought onto one pipeline.

## Public workflow

- Added `shrecc.NewDatabase` as the common orchestration API.
- Made `create()` acquire, solve, map, filter, and apply cutoff without changing
  Brightway state.
- Kept `write()` as the explicit Brightway mutation boundary.
- Added per-year `results()` and `table()` accessors so normalized mixes,
  physical volumes, and mapped exchanges remain inspectable.
- Added automatic source selection: supported prospective scenario years use
  TYNDP; other years use Energy Charts. Explicit source selection remains
  available.

## Canonical data model and solver

- Defined source-independent `production_volume`, `trade_volume`, and optional
  measured `consumption_volume` inputs.
- Standardized outputs as `consumption_mix`, `consumption_mix_volume`, and the
  retained physical input volumes.
- Replaced the historical full-matrix path with the same reduced country-network
  algebra used for TYNDP.
- Solve only interconnected country nodes, then attach one-way generation
  technologies without enlarging the linear system.
- Added chunked solving and preserved explicit handling of zero-consumption and
  unresolved-origin cases.
- Use measured TYNDP demand for `consumption_volume` while retaining gross supply
  for consumption-mix attribution.

## Source adapters and persistence

- Consolidated Energy Charts acquisition, cleaning, retry behavior, and canonical
  input construction in `energy_charts.py`.
- Consolidated TYNDP acquisition, scenario validation, workbook parsing, node
  aggregation, and canonical input construction in `tyndp.py`.
- Added resumable Energy Charts acquisition for incomplete country caches,
  including rate-limit handling.
- Added versioned, compressed, time-chunked canonical caches with manifests and
  selective date loading in `result_store.py`.
- Kept large downloaded and generated files in the user-data directory instead
  of the installed package.

## Mapping and database writing

- Moved shared temporal filtering, ecoinvent allocation, and table conversion to
  `mapping.py`.
- Reused country-specific ecoinvent shares for established TYNDP technologies.
- Map one-to-one prospective activities directly, preferring the source-country
  geography when that activity exists and otherwise using its premise/IAM
  region.
- Preserve technology identity during Brightway lookup; when a requested
  geography is unavailable and `strict=False`, location fallback searches for
  the same activity rather than substituting another technology.
- Added explicit support for country-specific photovoltaic inventories and
  country aliases such as TYNDP `UK` to ecoinvent `GB`.
- Separated reusable cutoff/table preparation from Brightway activity lookup and
  database writing.

## Compatibility and repository structure

- Turned `download.py`, `activity_mapping.py`, `treatment.py`, and
  `treatment_fiona.py` into compatibility facades for established imports.
- Isolated reference implementations in `_legacy_treatment.py`,
  `_legacy_database.py`, and `_legacy_tyndp.py`; the canonical pipeline does not
  depend on them.
- Archived the former source-specific example notebooks and replaced them with
  a minimal `notebooks/1_shrecc_get_started.ipynb` workflow and the expanded
  `notebooks/2_shrecc_analysis.ipynb` walkthrough.
- Removed obsolete spreadsheet mappings and generated report artifacts from the
  tracked/package-data surface.
- Retained the NED.nl-enhanced ecoinvent concordance as packaged mapping data for
  a future Netherlands fallback adapter.
- Moved the editable electricity-source concordance to `mapping_sources/`,
  separating mapping maintenance inputs from installed runtime resources.
- Declared package data recursively and split runtime, testing, notebook,
  premise, development, and documentation dependencies into appropriate groups.

## Verification added

The refactoring added focused tests for the shared solver, source adapters,
canonical result cache, harmonized mapping, high-level pipeline, compatibility
imports, retry/resume behavior, and Brightway geography fallback. A regression
test compares the reduced solver with the legacy full-matrix result.

At the end of this pass, the suite contains 173 passing tests with 81% aggregate
coverage. Both the wheel and source distribution pass `twine check`; the wheel
contains only Python modules and the required CSV/JSON package resources.

## Deliberate remaining compatibility

The legacy modules remain temporarily to protect downstream imports and to make
numerical comparison possible. They should be removed only in a documented
breaking release after users have migrated to `NewDatabase` and the canonical
module names.
