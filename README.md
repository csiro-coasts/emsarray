# emsarray

The `emsarray` package provides a common interface
for working with the many model formats used at CSIRO.
It enhances [`xarray`][xarray] Datasets
and provides a set of common operations for manipulating datasets.

To use, open the dataset using the `emsarray.open_dataset()` function
and use the `dataset.ems` attribute:

```python
import emsarray
import json

dataset = emsarray.open_dataset('./tests/datasets/shoc_standard.nc')
with open("geometry.geojson", "w") as f:
	json.dump(dataset.ems.make_geojson_geometry(), f)
```

Some methods take a DataArray as a parameter:

```python
eta = ds.eta.isel(record=1)

# Get all sea surface height values
values = ds.ems.make_linear(eta)

# Plot the sea surface height for record=1
ds.ems.plot(eta)
```

## Developing

To get set up for development, make a virtual environment and install the dependencies:

```shell
$ python3 -m venv
$ source venv/bin/activate
$ pip install --upgrade pip>=21.3
$ pip install -e . -r requirements.txt
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

[xarray]: https://xarray.pydata.org/
[livereload]: https://livereload.readthedocs.io/en/latest/
