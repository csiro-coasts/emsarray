import numpy.testing
import pytest
import xarray

from emsarray import utils


def dataset_with_lon_lat(
    lon_values: numpy.ndarray | None = None,
    lat_values: numpy.ndarray | None = None,
):
    if lon_values is None:
        lon_values = numpy.arange(5)
    if lat_values is None:
        lat_values = numpy.arange(7)
    return xarray.Dataset(
        coords={
            'lon': xarray.DataArray(data=lon_values, dims='x'),
            'lat': xarray.DataArray(data=lat_values, dims='y'),
        }
    )


def test_estimate_bounds_1d():
    dataset = dataset_with_lon_lat(
        lon_values=numpy.array([0, 1, 3, 7, 10], dtype=float)
    )

    # Take a copy so we can check that the function does not modify the dataset
    original_dataset = dataset.copy()

    dataset_with_bounds = utils.estimate_bounds_1d(dataset, 'lon')

    # Check that the original dataset was not modified
    xarray.testing.assert_identical(original_dataset, dataset)
    del dataset

    # Check that the 'lon' variable was updated correctly
    assert 'bounds' in dataset_with_bounds['lon'].attrs
    bounds_name = dataset_with_bounds['lon'].attrs['bounds']
    assert bounds_name == 'lon_bounds'
    assert bounds_name in dataset_with_bounds

    # Test that the bounds variable is correct
    bounds = dataset_with_bounds[bounds_name]
    assert bounds.dims == ('x', 'Two')
    assert bounds.sizes['Two'] == 2
    numpy.testing.assert_array_equal(bounds.values, [
        [0.0, 0.5],
        [0.5, 2.0],
        [2.0, 5.0],
        [5.0, 8.5],
        [8.5, 10.],
    ])

    # Check that the 'lat' variable was not modified
    assert dataset_with_bounds['lat'].attrs == {}


def test_estimate_bounds_1d_does_not_clobber():
    dataset = dataset_with_lon_lat()

    # The presence of this attribute is sufficient to raise an error
    dataset['lon'].attrs['bounds'] = 'lon_bounds'

    with pytest.raises(ValueError, match="Coordinate 'lon' already has a 'bounds' attribute"):
        utils.estimate_bounds_1d(dataset, 'lon')


def test_estimate_bounds_1d_raises_on_2d():
    xx, yy = numpy.meshgrid(
        numpy.arange(5),
        numpy.arange(7),
    )
    dataset = xarray.Dataset(
        coords={
            'lon': (('j', 'i'), xx),
            'lat': (('j', 'i'), yy),
        }
    )

    with pytest.raises(ValueError, match="Coordinate 'lon' has 2 dimensions .*, expecting one dimension"):
        utils.estimate_bounds_1d(dataset, 'lon')


def test_estimate_bounds_1d_bounds_dimension():
    dataset = dataset_with_lon_lat()
    dataset = utils.estimate_bounds_1d(dataset, 'lon', bounds_dimension='lon_bounds')
    assert dataset['lon_bounds'].dims == ('x', 'lon_bounds')


def test_estimate_bounds_1d_bounds_dimension_reused():
    dataset = dataset_with_lon_lat()
    dataset = utils.estimate_bounds_1d(dataset, 'lon')
    dataset = utils.estimate_bounds_1d(dataset, 'lat')
    assert dataset['lon_bounds'].dims == ('x', 'Two')
    assert dataset['lat_bounds'].dims == ('y', 'Two')


def test_estimate_bounds_1d_bounds_dimension_clash():
    dataset = dataset_with_lon_lat(lon_values=[1, 2, 3, 4])

    with pytest.raises(ValueError, match="Dataset already has a conflicting dimension 'x' of size 4"):
        utils.estimate_bounds_1d(dataset, 'lon', bounds_dimension='x')


def test_estimate_bounds_1d_bounds_name():
    dataset = dataset_with_lon_lat()
    bounds_name = 'lonbnds'
    dataset = utils.estimate_bounds_1d(dataset, 'lon', bounds_name=bounds_name)
    assert dataset['lon'].attrs['bounds'] == bounds_name
    assert bounds_name in dataset


def test_estimate_bounds_1d_bounds_name_auto():
    dataset = dataset_with_lon_lat()

    # Add a new variable that will conflict with the automatic bounds name
    auto_name = utils.estimate_bounds_1d(dataset, 'lon')['lon'].attrs['bounds']
    dataset[auto_name] = xarray.DataArray(
        data=[0, 1, 2],
        dims='Three',
    )

    # Make some bounds
    dataset = utils.estimate_bounds_1d(dataset, 'lon')

    # Check that the bounds variable got a non-conflicting name
    bounds_name = dataset['lon'].attrs['bounds']
    assert bounds_name != auto_name
    assert bounds_name in dataset

    # Check that the conflicting variable hasn't been modified
    assert dataset[auto_name].dims == ('Three',)


def test_estimate_bounds_1d_bounds_name_clash():
    dataset = dataset_with_lon_lat()

    with pytest.raises(ValueError, match="Dataset already has a variable named 'lat'"):
        utils.estimate_bounds_1d(dataset, 'lon', bounds_name='lat')
