# SHRECC data

shrecc requires 2 types of data:

+ The time series of the electricity mixes
+ Mapping data

## Mapping data

The package is configured to load the mapping data from within the package itself.
The data is under the `shrecc.data` subdirectory.

```
shrecc
├── data
│   ├── el_map_all_norm.csv
│   ├── generation_units_by_country.csv
│   └── techs_agg.json
├── database.py
├── download.py
├── __init__.py
└── treatment.py

```

When calling the {py:func}`shrecc.database.filt_cutoff` function, if no value is passed to the `mapping_root` argument, the data is taken from the package, but the user can supply a path to a directory that contains a subdirectory `data` that has the necessary files (`el_map_all_norm.csv`, etc.)


### el_map_all_norm.csv

This CSV file contains a mapping of ENTSO-E technology categories to their corresponding ecoinvent classifications.

### generation_units_by_country.csv

This csv file is a binary matrix indicating which electricity generation technologies are present in each country, used to support technology mapping and data filtering.

### techs_agg.json

This JSON file provides a mapping of various electricity generation and trade technology labels to consistent names.

## Time series

This data is pre-calculated, and stored in the [shrecc_data](https://git.list.lu/shrecc_project/shrecc_data) repository.

The user must download this data to the `shrecc` appdirs user directory.
This can be done with the {py:func}`shrecc.download.download_shrecc_data`, and it will automatically place the data in the right place.
The user can also manually download the data.

The data can also be generated on the fly. Please look at the example notebook for how to do this.
