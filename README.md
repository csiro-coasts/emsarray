# emsarray

[![Binder](https://mybinder.org/badge_logo.svg)][emsarray-binder]
[![Documentation Status](https://readthedocs.org/projects/emsarray/badge/?version=latest)](https://emsarray.readthedocs.io/en/latest/?badge=latest)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/emsarray.svg)](https://anaconda.org/conda-forge/emsarray)

The `emsarray` package provides a common interface
for working with the many model formats used at CSIRO.
It enhances [`xarray`][xarray] Datasets
and provides a set of common operations for manipulating datasets.

To use, open the dataset using the `emsarray.open_dataset()` function
and use the `dataset.ems` attribute:

```python
import emsarray
import json

dataset = emsarray.tutorial.open_dataset('gbr4')
with open("geometry.geojson", "w") as f:
	json.dump(dataset.ems.make_geojson_geometry(), f)
```

Some methods take a DataArray as a parameter:

```python
# Plot the sea surface temperature for time = 0
temp = dataset['temp'].isel(time=0, k=-1)
dataset.ems.plot(temp)
```

![Plot of sea surface temperature from the GBR4 example file](docs/_static/images/gbr4_temp.png)

## Examples

Examples of using `emsarray` are available in the [emsarray-notebooks][emsarray-notebooks] repository.
You can [explore these notebooks online][emsarray-binder] with Binder.


## Developing

To get set up for development, make a virtual environment and install the dependencies:

```shell
$ python3 -m venv
$ source venv/bin/activate
$ pip install --upgrade pip>=21.3
$ pip install -e . -r continuous-integration/requirements.txt
```

## Tests

To run the tests, install and run `tox`:

```shell
$ python3 -m venv
$ source venv/bin/activate
$ pip install --upgrade pip>=21.3 tox
$ tox
```

## Documentation

To build the documentation, install the development requirements as above and invoke Sphinx:

```shell
$ make -C docs/ html
```

While updating or adding to the documentation,
run the `live` target to automatically rebuild the docs whenever anything changes.
This will serve the documentation via a [`livereload`][livereload] server.

```shell
$ make -C docs/ live
```

You can the view the docs at <http://localhost:5500>

[emsarray-notebooks]: https://github.com/csiro-coasts/emsarray-notebooks
[emsarray-binder]: https://mybinder.org/v2/gh/csiro-coasts/emsarray-notebooks/HEAD
[xarray]: https://xarray.pydata.org/
[livereload]: https://livereload.readthedocs.io/en/latest/
