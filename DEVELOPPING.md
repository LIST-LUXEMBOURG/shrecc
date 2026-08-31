# Developing SHRECC

## Versionning

The project uses [setuptools_scm](https://pypi.org/project/setuptools-scm/) to handle versioning, and it uses [semantic versioning](https://semver.org/).

### Version "bumping"

+ To increase the version, just add a git tag to the `main` branch.
  This can be done either directly from the CLI console, or through the web interface of gitlab.
  Only the commits pushed to the main branch with a "tag" get succesfully published to [pypi.org](https://pypi.org/project/shrecc/).
+ In general, prefer keeping a ".devX" tag on the "develop branch" and make it become a full semantic versionning compliant tag once a merge from develop to main is tone.
  For example, the current tag in main could be "0.1.0", and the tag on the develop branch of a commit after the latest one in main could be "0.2.0.dev1".
  This is useful to push "development" versions to pypi.

### Code linting and formatting with pre-commit

This repository has a working [pre-commit configuration](.pre-commit-config.yaml).
Please make sure you add [pre-commit](https://pre-commit.com/) to your environment when developping (tests, features, debugging, etc.) against this repository.

## Clone the source

For the code: 

```
git clone https://git.list.lu/shrecc_project/shrecc
```

For the data: 
```
https://git.list.lu/shrecc_project/shrecc_data
```

### Dependencies

Create and activate a Python 3.10 or newer virtual environment, then install
the package and the maintained development extras from the repository root:

```bash
git clone https://git.list.lu/shrecc_project/shrecc
cd shrecc
python -m pip install -e ".[dev,notebooks,premise]"
```

`pyproject.toml` is the source of truth for runtime and optional dependencies.
Install only the extras needed for narrower work, for example `.[dev]` for
tests and linting or `.[docs]` for documentation.

## Documentation

The documentation is automatically built using `sphinx` uppon pushing to the `main` branch of the repository, and published at [shrecc.readthedocs.io](https://shrecc.readthedocs.io/en/latest/)

The documentation of the package is under the [docs](docs) directory.
The directory contains some configuration files, and the index to the documentation.

+ Under this directory, there is a subdirectory [contents](docs/contents) where the actual documentation should go.
+ the [notebooks](docs/contents/notebooks/) directory can have full jupyter notebooks that get added to the documentation.

### Building the Documentation

Install the package with its documentation dependencies:

```bash
python -m pip install -e ".[docs]"
```

Then run the build command:

```bash
sphinx-autobuild docs docs/_build/html -a -j auto --ignore 'docs/content/api*' --open-browser
```

