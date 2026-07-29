# SHRECC data

SHRECC keeps small, version-controlled mapping resources inside the Python
package and stores downloaded or generated time series in a user-data cache.
This distinction lets installed packages remain reproducible without bundling
large scenario workbooks or solved hourly arrays.

## Packaged mapping data

The default resources are available through the `shrecc.data` package:

```text
shrecc/data/
|-- el_map_all_norm.csv
|-- el_map_all_norm_w_ned.csv
|-- generation_units_by_country.csv
|-- techs_agg.json
`-- tyndp/
    |-- remind-eu-topology.json
    |-- tyndp_activities.csv
    |-- tyndp_connections.csv
    `-- tyndp_countries.csv
```

- `el_map_all_norm.csv` contains country-specific ecoinvent activity shares for
  established electricity technologies.
- `el_map_all_norm_w_ned.csv` retains the alternative NED.nl production labels
  prepared for filling gaps in Netherlands Energy Charts data. It is not selected
  automatically by the current Energy Charts adapter.
- `generation_units_by_country.csv` records the technologies represented in
  each historical country mix.
- `techs_agg.json` harmonizes Energy Charts production and trade labels.
- `tyndp_activities.csv` maps TYNDP technologies to compatible premise or
  ecoinvent electricity activities.
- `tyndp_connections.csv` and `tyndp_countries.csv` map TYNDP nodes and links to
  country codes.
- `remind-eu-topology.json` supplies the default REMIND-EU regional topology.

Custom paths can be passed to `NewDatabase` through `technology_mapping`,
`country_mapping`, `ecoinvent_mapping`, and `topology_files`.

The editable `mapping_sources/electricity_sources.xlsx` workbook is retained in
the repository for maintaining the Energy Charts and NED.nl concordances, but it
is not installed as package data.

## Downloaded and generated data

By default, `NewDatabase` stores source data beneath the platform-specific
`appdirs.user_data_dir("shrecc")` directory. Supplying `data_dir` selects a
different cache root.

Historical runs cache Energy Charts API responses and canonical solved results.
Prospective runs cache downloaded TYNDP archives, extracted workbooks, and parsed
production and trade tables. These files are local data and are not package
resources.

Canonical solved results are written as compressed time chunks with a manifest.
Only chunks intersecting the requested date or timestamp selection are loaded.
The resulting xarray Dataset can contain:

- `production_volume`
- `trade_volume`
- `consumption_volume`
- `consumption_mix`
- `consumption_mix_volume`

The volume variables retain physical quantities for analysis and visualization;
`consumption_mix` is normalized for mapping to life-cycle inventory activities.

## Inventory resolution

`NewDatabase(inventory_resolution=...)` controls the temporal resolution of
the written foreground activities:

- `"annual"` (the default) writes one activity per consumer country and model
  year. `"yearly"` is accepted as an alias.
- `"monthly"` writes one activity per consumer country and selected calendar
  month.
- `"hourly"` writes one activity per consumer country and selected timestamp.

Annual and monthly mixes apply `consumption_profile` independently within each
inventory period before normalization. Hourly mixes are already resolved at
their final temporal resolution, so profiles are ignored with a warning.
Hourly timestamps must therefore be selected with `times`, `time_range`, and
optionally `hour_range`.

Written activity names end in `YYYY`, `YYYY-MM`, or `YYYY-MM-DD HH:MM`,
respectively. The resolution, represented period, and applicable consumption
profile are also recorded in each Brightway activity's `comment` field for
display in ActivityBrowser documentation.
