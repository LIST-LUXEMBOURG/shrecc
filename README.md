# SHRECC

Simple Hourly Resolution Electricity Consumption Calculation

## Description

SHRECC creates time-resolved electricity-consumption databases for Brightway.
It uses measured Energy Charts data for historical years and ENTSO-E TYNDP
scenario data for prospective years, while exposing the same workflow for both.

## Features

- Hourly production, trade, consumption-volume, and consumption-mix results.
- One reduced country-network solver shared by Energy Charts and TYNDP data.
- Historical ecoinvent allocation and prospective premise/IAM mapping.
- Annual or monthly Brightway inventories from range-based or explicit
  timestamp selections, plus hourly LCIA without hourly foreground activities.
- Inspectable intermediate results before any Brightway database is changed.
- Resumable source acquisition and compressed, time-chunked result caches.
- Brightway 2 and 2.5 database writing through `NewDatabase`.
- Multi-year runs with a separate, year-labelled foreground database for each
  requested year.

## Documentation

The full documentation is hosted at [Read the Docs page for shrecc](https://shrecc.readthedocs.io/en/latest/)

The internal data flow and module responsibilities are summarized in
[docs/architecture.md](https://git.list.lu/shrecc_project/SHRECC/-/blob/main/docs/architecture.md).

## Installation

`shrecc` can be installed from pypi or from source.

### From pypi

The package is published at [pypi.org/projects/shrecc](https://pypi.org/project/shrecc).
You can install it with pip (or any other pypi compatible util like `uv` or `poetry` as follows:

```
pip install shrecc
```

Install optional premise geography support or the notebook environment with:

```
pip install "shrecc[premise,notebooks]"
```

### From source 

To install shrecc from source, clone the code and then install the package and if necessary the dependencies manually.


## Usage

The repository contains a minimal
[getting-started notebook](https://git.list.lu/shrecc_project/shrecc/-/blob/develop/notebooks/1_shrecc_get_started.ipynb)
and a more detailed
[analysis notebook](https://git.list.lu/shrecc_project/SHRECC/-/blob/develop/notebooks/2_shrecc_analysis.ipynb).
Both are also available in the documentation (only for main releases) at
[read the docs](https://shrecc.readthedocs.io/en/latest/content/notebooks/).

The harmonized historical/prospective workflow is available through
`NewDatabase`:

```python
from shrecc import NewDatabase

electricity = NewDatabase(
    scenario="DE",
    years=[2035, 2040, 2050],
    climate_year=2009,
    bg_db_name={
        2035: "premise-remind-eu-2035",
        2040: "premise-remind-eu-2040",
        2050: "premise-remind-eu-2050",
    },
    my_db_name="shrecc_tyndp_DE_june_noon",
    countries=["ES", "FR", "DE", "IT", "PT", "BE", "NL", "LU", "AT", "CH"],
    time_range=["2040-06-01 00:00:00", "2040-06-30 23:00:00"],
    hour_range=[10, 14],
    project_name="SHRECCei311",
    source="auto",
    inventory_resolution="annual",
)

electricity.create()
electricity.write()
```

For a multi-year run, the month/day/time selection is reused for each year and
`write()` creates one foreground database per year. A year suffix is added to
`my_db_name` automatically unless explicit year-specific names are supplied.
Foreground activity names and metadata also record their modeled period.

`inventory_resolution` controls how many foreground activities are written:
`"annual"` creates one activity per country and year, `"monthly"` creates one
per country and selected calendar month. Both resolutions apply
`consumption_profile` within each period. Hourly inventories are not written
as Brightway activities.

Hourly LCIA is calculated directly from the retained hourly mixes. By default,
SHRECC selects every installed EF v3.1 impact category, scores each unique
background input once, and combines those scores with the hourly coefficients:

```python
lcia_results = electricity.lcia()
hourly_intensity = lcia_results.hourly(2040)["intensity"]
annual_intensity = lcia_results.annual()
```

Use `electricity.lcia(methods=..., engine="multilca")` to validate the
optimized calculation with composite Brightway MultiLCA functional units.
Neither engine writes hourly foreground activities.

Intermediate canonical results and annual/monthly mapped tables remain
available on the class instance through `results(year)` and `table(year)`. Use
`mapping_report(year)` to inspect any historical technologies assigned to a
country-specific high-voltage fallback activity.


## Contributing

Please take a look at the [DEVELOPPING.md](https://git.list.lu/shrecc_project/shrecc/-/blob/main/DEVELOPPING.md) file for details on how to contribute code to the repository.

## License

Copyright © 2025 Luxembourg Institute of Science and Technology
Licensed under the MIT License.

## Authors

* Sabina Bednářová (<sabina.bednarova@list.lu>)
* Thomas Gibon (<thomas.gibon@list.lu>)
