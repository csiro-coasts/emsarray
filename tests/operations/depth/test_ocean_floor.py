import numpy
import pandas
import pytest
import xarray
from numpy.testing import assert_equal

import emsarray
from emsarray.operations.depth import ocean_floor


def test_ocean_floor():
    # Values will be a 3D cube of values, with a slice along the x-axis like
    #     y
    #   44444
    #   3333.
    # d 222..
    #   11...
    #   0....
    values = numpy.full((5, 5, 5, 5), fill_value=numpy.nan)
    for i in range(5):
        values[:, i, :i + 1, :i + 1] = i

    temp = xarray.DataArray(
        data=values,
        dims=['t', 'z', 'y', 'x'],
    )
    dataset = xarray.Dataset(
        data_vars={"temp": temp},
        coords={
            'time': (['t'], pandas.date_range('2022-02-08', periods=5)),
            'lon': (['x'], -numpy.arange(5)),
            'lat': (['y'], numpy.arange(5)),
            'depth': (['z'], 4.25 - numpy.arange(5), {'positive': 'down'}),
        }
    )

    floor_dataset = ocean_floor(dataset, ['depth'], non_spatial_variables=['time'])

    assert floor_dataset.sizes == {
        't': 5,
        'x': 5,
        'y': 5,
    }
    assert set(floor_dataset.coords.keys()) == {'time', 'lon', 'lat'}
    # We should see values for the deepest layer that has a value there
    expected_values = [
        [0, 1, 2, 3, 4],
        [1, 1, 2, 3, 4],
        [2, 2, 2, 3, 4],
        [3, 3, 3, 3, 4],
        [4, 4, 4, 4, 4],
    ]
    assert_equal(
        floor_dataset['temp'].values,
        numpy.array([expected_values] * 5))


@pytest.mark.tutorial
@pytest.mark.parametrize('name', [
    'bran2020',
])
def test_ocean_floor_from_files(name):
    # Make sure that the operation runs without throwing an error,
    # and that the dataset returned has the expected dimensions.
    # It does not look at the actual data returned.
    dataset = emsarray.tutorial.open_dataset(name)

    floored = dataset.ems.ocean_floor()

    depth_dimensions = {
        coordinate.dims[0]
        for coordinate in dataset.ems.depth_coordinates
    }
    original_dimensions = set(dataset.dims)
    floored_dimensions = set(floored.dims)
    # The original dataset should have all the depth dimensions
    assert original_dimensions > depth_dimensions
    # The new dataset should not have any depth dimensions
    assert floored_dimensions == original_dimensions - depth_dimensions

    # The floored dataset should use the same convention type
    assert type(floored.ems) is type(dataset.ems)  # noqa: E721
