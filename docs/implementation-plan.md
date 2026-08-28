# Implementation Plan

This file is the compact handoff for ongoing SHRECC development. Read it with
`docs/architecture.md` and `docs/content/data.md`; do not reconstruct repository
history from old notebooks or legacy modules unless a regression requires it.

## North star

Keep the beginner workflow obvious:

```python
electricity = NewDatabase(...)
electricity.create()
electricity.write()
```

`create()` prepares inspectable in-memory results, `write()` writes reusable
annual or monthly Brightway inventories, and `lcia()` assesses hourly mixes
without writing thousands of hourly activities. New features must preserve
this separation and should not add public arguments unless necessary.

## Current implementation

- `NewDatabase` is the public orchestrator for both historical Energy Charts
  and prospective TYNDP data. `source="auto"` selects the source by model year.
- Both sources produce the canonical variables `production_volume`,
  `trade_volume`, `consumption_volume`, and `consumption_mix`; optional
  `consumption_mix_volume` is retained only when requested.
- The shared solver resolves only interconnected country nodes. Technologies
  that do not exchange electricity are projected through the solved network
  afterward.
- `consumption_profile` supports `"flat"`, `"national_demand"`, and a custom
  timestamp-indexed pandas Series. Annual and monthly inventory aggregation
  applies these weights within each output period and then normalizes.
- Written `inventory_resolution` is `"annual"` or `"monthly"`. Hourly mixes
  remain calculation data and are never written as Brightway activities.
- Generated activities use lowercase names such as
  `electricity, consumption mix, 2040`, store geography only in `location`,
  use reference product `electricity`, and unit `kilowatt hour`.
- `NewDatabase.lcia()` defaults to installed EF v3.1 methods. The `linear`
  engine scores unique background inputs and multiplies hourly coefficients by
  those scores. `multilca` remains a validation engine.
- `LCIAResults.hourly(year)` returns `intensity`, `consumption_weight`, and
  `weighted_contribution`. `monthly()` and `annual()` return profile-weighted
  intensities. `impact_categories` exposes labels suitable for xarray
  selection.

## Mapping decisions

- Existing ecoinvent technology families use country-specific shares from
  `el_map_all_norm.csv`.
- TYNDP technologies with multiple compatible activities use those shares and
  retain the source-country geography.
- A unique compatible activity uses the source country when present there;
  otherwise it uses the corresponding premise IAM region.
- Preserve source-technology identity before geography. If an Italian offshore
  activity is absent, use the same offshore technology at an accepted fallback
  geography instead of an unrelated Italian technology.
- Historical mapping gaps fall back to a country-specific high-voltage
  production mix and are reported by `mapping_report()`. They must not fall
  back to an infrastructure-bearing market activity.
- Mapping CSV files are package data generated from licensed background data.
  Their provenance should be reproducible, but derived ecoinvent data must not
  be distributed unless licensing permits it.

## Performance decisions

- Canonical solved source results are stored in compressed time chunks and
  loaded selectively.
- Linear LCIA maps one consumer country at a time to bound memory use.
- Background activity scores are persisted locally in
  `lcia_source_scores_v1` under SHRECC's configured data directory. The key
  includes the Brightway project, background/dependency metadata, and complete
  LCIA method data. These licensed derived scores are local only and ignored by
  Git.
- Controlled cache benchmark: scoring 20 electricity activities for one EF
  v3.1 method took 7.916 seconds; loading their persistent cache took 0.0059
  seconds.
- Brightway database writing remains dominated by an unconditional SQLite
  vacuum in `bw2data 4.7`. A measured foreground write took about 135 seconds,
  of which about 133 seconds was vacuuming; an equivalent private no-vacuum
  write took about 1.6 seconds. SHRECC should use the public writer and minimize
  write calls until Brightway exposes a supported `vacuum=False` option.
- Do not optimize pandas, mapping reloads, or matrix operations without a
  profile demonstrating a material gain.

## Recent branch state

`develop` includes merge request `!21` / issue 48. Current work is on
`feature/lcia-usability`, with one auditable commit per concern:

1. `5f8a475` - development principles in `AGENTS.md`.
2. `bcfed86` - clearer `LCIAResults` metadata and discoverability.
3. `4a6a1eb` - user-facing output decision guide.
4. `d7e20bc` - optional hourly LCIA example in the beginner notebook.
5. `e8bbed0` - persistent local background LCIA source-score cache.

Focused LCIA and pipeline verification after the cache change: 39 tests passed.
The preceding issue-48 affected suite passed 133 tests, and its GitLab pipeline
passed.

## Next implementation step

Add profile reweighting without another Brightway calculation. Keep the
hourly `intensity` invariant and replace only temporal weights and weighted
contributions.

Before coding, resolve the smallest honest API for `"national_demand"`:

- `LCIAResults` currently retains only the profile weights selected by
  `NewDatabase`.
- Reweighting to `"flat"` or an explicit Series can be done from the result
  object alone.
- Reweighting to `"national_demand"` requires retaining the canonical country
  demand alongside the assessment or delegating reweighting through
  `NewDatabase`.

Prefer extending `LCIAResults` if the needed demand data can be retained
without meaningful memory overhead. Do not duplicate hourly intensities and do
not add another `NewDatabase` constructor argument. Test multi-year timestamp
alignment, country-specific demand, custom Series coverage, zero/negative
weights, and preservation of the original result.

## Deferred work

- Regionalize grid infrastructure and SF6 exchanges using values extracted
  from each selected background database.
- Model delivered-electricity losses from the ecoinvent voltage chain. Do not
  treat supply-demand balance residuals as measured grid losses.
- Keep extraction as focused functions reusing `BackgroundActivityIndex`.
  Introduce a background extractor class only if repeated state and arguments
  demonstrably justify it.
- Consider writing several consumption-profile activities in one foreground
  database call to avoid repeated Brightway vacuums.
- Request or contribute a supported Brightway option to defer vacuuming.
- Continue issue 47 by auditing `NewDatabase` arguments and removing options
  whose behavior can be inferred safely.
