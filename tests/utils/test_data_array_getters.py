import numpy
import pandas
import pytest
import xarray
import xarray.testing

from emsarray.utils import data_array_to_name, name_to_data_array


def _make_dataset():
    grid = xarray.DataArray(
        data=numpy.arange(5 * 7).reshape(5, 7),
        dims=['x', 'y'],
    )
    time = xarray.DataArray(
        data=pandas.date_range("2024-07-08", periods=3),
        dims=['t'],
    )
    dataset = xarray.Dataset({'grid': grid, 'time': time})
    return dataset


def test_name_to_data_array_name():
    dataset = _make_dataset()
    xarray.testing.assert_equal(
        name_to_data_array(dataset, 'grid'),
        dataset['grid'])
    xarray.testing.assert_equal(
        name_to_data_array(dataset, 'time'),
        dataset['time'])


def test_name_to_data_array_missing_name():
    dataset = _make_dataset()
    name = 'missing'
    with pytest.raises(ValueError, match=f"Data array {name!r} is not in the dataset"):
        name_to_data_array(dataset, name),


def test_name_to_data_array_matching_data_array():
    dataset = _make_dataset()
    xarray.testing.assert_equal(
        name_to_data_array(dataset, dataset['grid']),
        dataset['grid'])
    xarray.testing.assert_equal(
        name_to_data_array(dataset, dataset['time']),
        dataset['time'])


def test_name_to_data_array_mismatched_dimensions():
    dataset = _make_dataset()
    # These dimensions are swapped so incorrect
    data_array = xarray.DataArray(data=numpy.arange(5 * 7).reshape(5, 7), dims=['y', 'x'])
    with pytest.raises(ValueError, match="Dimension mismatch between dataset and data array"):
        name_to_data_array(dataset, data_array),


def test_data_array_to_name_name():
    dataset = _make_dataset()
    assert data_array_to_name(dataset, 'grid') == 'grid'
    assert data_array_to_name(dataset, 'time') == 'time'


def test_data_array_to_name_no_name():
    dataset = _make_dataset()
    name = 'missing'
    with pytest.raises(ValueError, match=f"Data array {name!r} is not in the dataset"):
        data_array_to_name(dataset, name)


def test_data_array_to_name_matching_data_array():
    dataset = _make_dataset()
    assert data_array_to_name(dataset, dataset['grid']) == 'grid'
    assert data_array_to_name(dataset, dataset['time']) == 'time'


def test_data_array_to_name_unnamed_data_array():
    dataset = _make_dataset()
    data_array = xarray.DataArray(data=numpy.arange(5 * 7).reshape(5, 7), dims=['x', 'y'])
    with pytest.raises(ValueError, match="Data array has no name"):
        data_array_to_name(dataset, data_array)


def test_data_array_to_name_missing_data_array():
    dataset = _make_dataset()
    name = 'missing'
    data_array = xarray.DataArray(data=numpy.arange(5 * 7).reshape(5, 7), dims=['x', 'y'], name=name)
    with pytest.raises(ValueError, match=f"Dataset does not have a data array named {name!r}"):
        data_array_to_name(dataset, data_array)


def test_data_array_to_name_mismatched_dimensions():
    dataset = _make_dataset()
    # These dimensions are swapped so incorrect
    name = 'grid'
    data_array = xarray.DataArray(data=numpy.arange(5 * 7).reshape(5, 7), dims=['y', 'x'], name=name)
    with pytest.raises(ValueError, match="Dimension mismatch between dataset and data array"):
        data_array_to_name(dataset, data_array)
