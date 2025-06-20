# SHRECC

Simple Hourly Resolution Electricity Consumption Calculation

## Description

SHRECC package is a python package directly compatible with Brightway2 or Brightway2.5 to create time-aware electricity databases. For any given year and countries (check availability on https://api.energy-charts.info/), download and prepare data for low-voltage electricity consumption.

## Features

- **High-resolution electricity mixes** – Generates electricity life cycle inventories (LCIs) with **hourly** resolution, enhancing accuracy for life cycle assessment (LCA).
- **Brightway2/2.5 compatibility** – Seamlessly integrates with Brightway, allowing direct use in existing LCA models.
- **Dynamic temporal representation** – Users can select electricity mixes by **hour, month, or season**, addressing fluctuations in renewable energy generation and consumption.
- **Automated data retrieval** – Pulls electricity production, trade, and consumption data from the **Energy Charts API**, ensuring up-to-date datasets.
- **Ecoinvent matching** – Aligns with **ecoinvent classifications**, converting from ENTSO-E datasets.
- **User-controlled updates** – Enables **one-time or recurring** updates, allowing continuous tracking of electricity mix evolution over time.
- **Optimized impact assessments** – Helps reduce uncertainty and improve **decision-making for electricity-intensive technologies** by considering real-time electricity mix variations.

## Documentation

The full documentation is hosted at [Read the Docs page for shrecc](https://shrecc.readthedocs.io/en/latest/)

## Installation

`shrecc` can be installed from pypi or from source.

### From pypi

The package is published at [pypi.org/projects/shrecc](https://pypi.org/project/shrecc).
You can install it with pip (or any other pypi compatible util like `uv` or `poetry` as follows:

```
pip install shrecc
```

### From source 

To install shrecc from source, clone the code and then install the package and if necessary the dependencies manually.


## Usage

You can find usage examples in the Jupyter notebook in this repo: [notebooks/example.ipynb](notebooks/example.ipynb)
_and_ in the documentation at [read the docs](https://shrecc.readthedocs.io/en/latest/content/notebooks/notebooks.html).


## Contributing

Please take a look at the [DEVELOPPING.md](DEVELOPPING.md) file for details on how to contribute code to the repository.

## License

Copyright © 2024 Luxembourg Institute of Science and Technology
Licensed under the MIT License.

## Authors

* Sabina Bednářová (<sabina.bednarova@list.lu>)
* Thomas Gibon (<thomas.gibon@list.lu>)
