import datetime
import logging
import pathlib

import netCDF4
import numpy
import numpy.testing
import pandas
import pytest
import xarray
import xarray.testing

from emsarray import utils
from tests.utils import filter_warning

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    ('existing', 'new'),
    [
        ('days since 1990-01-01T00:00:00+10:00', 'days since 1990-01-01 00:00:00 +10:00'),
        ('days since 1990-1-1 0:00:00 +10', 'days since 1990-01-01 00:00:00 +10:00'),
        ('hours since 2021-11-16T12:00:00+11:00', 'hours since 2021-11-16 12:00:00 +11:00'),
    ],
)
def test_format_time_units_for_ems(existing, new):
    assert new == utils.format_time_units_for_ems(existing)


def test_fix_time_units_for_ems(tmp_path: pathlib.Path):
    dataset_path = tmp_path / "dataset.nc"
    # Make a minimal dataset, with a record dimension and time data
    xr_dataset = xarray.Dataset(
        data_vars={
            't': xarray.DataArray(
                data=[datetime.datetime(2021, 11, 16, hour) for hour in range(24)],
                dims=["record"],
                attrs={'coordinate': 'time', 'long_name': 'Time'},
            ),
        },
    )
    # Set a nice, EMS compatible time unit string
    xr_dataset.data_vars['t'].encoding.update({
        'units': 'hours since 2021-11-01 00:00:00 +10:00',
        'calendar': 'proleptic_gregorian',
    })
    xr_dataset.encoding['unlimited_dims'] = {'record'}
    xr_dataset.to_netcdf(dataset_path)

    with netCDF4.Dataset(dataset_path, "r") as nc_dataset:
        # xarray will have reformatted the time units to ISO8601 format,
        # with a 'T' separator and no space before the timezone
        assert nc_dataset.variables['t'].getncattr('units') == 'hours since 2021-11-01T00:00:00+10:00'

    utils.fix_time_units_for_ems(dataset_path, "t")

    with netCDF4.Dataset(dataset_path, "r+") as nc_dataset:
        # The time units should now be in an EMS-compatible format
        assert nc_dataset.variables['t'].getncattr('units') == 'hours since 2021-11-01 00:00:00 +10:00'


def test_disable_default_fill_value(tmp_path: pathlib.Path):
    int_var = xarray.DataArray(
        data=numpy.arange(35, dtype=int).reshape(5, 7),
        dims=['j', 'i'],
        attrs={"Hello": "World"},
    )

    float_var = xarray.DataArray(
        data=numpy.arange(35, dtype=numpy.float64).reshape(5, 7),
        dims=['j', 'i'])

    f_data = numpy.where(
        numpy.tri(5, 7, dtype=bool),
        numpy.arange(35, dtype=numpy.float64).reshape(5, 7),
        numpy.nan)
    float_with_fill_value_var = xarray.DataArray(data=f_data, dims=['j', 'i'])
    float_with_fill_value_var.encoding["_FillValue"] = numpy.nan

    td_data = numpy.where(
        numpy.tri(5, 7, dtype=bool),
        numpy.arange(35).reshape(5, 7) * numpy.timedelta64(1, 'D').astype('timedelta64[ns]'),
        numpy.timedelta64('nat', 'ns'))
    logger.info('%r', td_data)
    timedelta_with_missing_value_var = xarray.DataArray(
        data=td_data, dims=['j', 'i'])
    timedelta_with_missing_value_var.encoding['missing_value'] = numpy.float64('1e35')
    timedelta_with_missing_value_var.encoding['units'] = 'days'

    dataset = xarray.Dataset(data_vars={
        "int_var": int_var,
        "float_var": float_var,
        "float_with_fill_value_var": float_with_fill_value_var,
        "timedelta_with_missing_value_var": timedelta_with_missing_value_var,
    })

    # Save to a netCDF4 and then prove that it is bad
    # This emits warnings in current versions of xarray / numpy.
    # See https://github.com/pydata/xarray/issues/7942
    with filter_warning(
        'default', category=RuntimeWarning,
        message='invalid value encountered in cast',
        module=r'xarray\.coding\.times',
        record=True,
    ) as ws:
        dataset.to_netcdf(tmp_path / "bad.nc")
        assert len(ws) == 1

    with netCDF4.Dataset(tmp_path / "bad.nc", "r") as nc_dataset:
        # This one shouldn't be here because it is an integer datatype. xarray
        # does the right thing already in this case.
        assert '_FillValue' not in nc_dataset.variables["int_var"].ncattrs()
        # This one shouldn't be here as we didnt set it, and the array is full!
        # This is the problem we are trying to solve
        assert numpy.isnan(nc_dataset.variables["float_var"].getncattr("_FillValue"))
        # This one is quite alright, we did explicitly set it after all
        assert numpy.isnan(nc_dataset.variables["float_with_fill_value_var"].getncattr("_FillValue"))
        # This one is incorrect, a `missing_value` attribute has already been set
        assert numpy.isnan(nc_dataset.variables["timedelta_with_missing_value_var"].getncattr("_FillValue"))

    utils.disable_default_fill_value(dataset)

    # See https://github.com/pydata/xarray/issues/7942
    with filter_warning(
        'default', category=RuntimeWarning,
        message='invalid value encountered in cast',
        module=r'xarray\.coding\.times',
        record=True,
    ) as ws:
        dataset.to_netcdf(tmp_path / "good.nc")
        assert len(ws) == 1

    with netCDF4.Dataset(tmp_path / "good.nc", "r") as nc_dataset:
        # This one should still be unset
        assert '_FillValue' not in nc_dataset.variables["int_var"].ncattrs()
        # This one should now be unset
        assert '_FillValue' not in nc_dataset.variables["float_var"].ncattrs()
        # Make sure this didn't get clobbered
        assert numpy.isnan(nc_dataset.variables["float_with_fill_value_var"].getncattr("_FillValue"))
        # This one should now be unset
        nc_timedelta = nc_dataset.variables["timedelta_with_missing_value_var"]
        assert '_FillValue' not in nc_timedelta.ncattrs()
        assert nc_timedelta.getncattr('missing_value') == numpy.float64('1e35')


def test_dataset_like():
    sample_foo = xarray.DataArray(
        data=numpy.arange(10, dtype=numpy.int32),
        coords=[numpy.arange(10) * 2], dims=['x'])
    sample_foo.attrs = {'units': 'meters'}
    sample_foo.encoding = {'_FillValue': 10}
    sample_bar = xarray.DataArray(
        data=numpy.arange(20, dtype=numpy.double),
        coords=[list("abcdefghijklmnopqrst")], dims=['y'])
    sample_bar.attrs = {'units': 'seconds', 'long_name': 'Seconds since lunch'}
    sample_bar.encoding = {'_FillValue': numpy.nan}
    sample_dataset = xarray.Dataset({'foo': sample_foo, 'bar': sample_bar})
    sample_dataset.attrs = {"hello": "world", "from": "sample"}
    sample_dataset.encoding = {"reticulation": "spline"}

    new_foo = xarray.DataArray(
        data=numpy.arange(5, dtype=numpy.int32), coords=[numpy.arange(5) * 3], dims=['x'])
    new_bar = xarray.DataArray(
        data=numpy.arange(10, dtype=numpy.double), coords=[list("ABCDEFGHIJ")], dims=['y'])
    new_bar.attrs = {'long_name': 'Seconds since breakfast'}
    new_bar.encoding = {'_FillValue': -9999.}
    new_dataset = xarray.Dataset({'bar': new_bar, 'foo': new_foo})
    new_dataset.attrs = {"from": "new"}
    new_dataset.encoding = {"reticulation": "quartic"}

    # Variables have an order in the dataset. The order in the new dataset is wrong
    assert list(sample_dataset.data_vars.keys()) == ['foo', 'bar']
    assert list(new_dataset.data_vars.keys()) == ['bar', 'foo']
    assert list(sample_dataset.dims.keys()) == ['x', 'y']
    assert list(new_dataset.dims.keys()) == ['y', 'x']

    # Make a new dataset like the input dataset
    like_dataset = utils.dataset_like(sample_dataset, new_dataset)

    # The data variables and dimensions should now be identical
    assert list(like_dataset.data_vars.keys()) == ['foo', 'bar']
    assert list(like_dataset.dims.keys()) == ['x', 'y']
    # The attributes and encodings should be merged
    assert like_dataset.attrs == {"hello": "world", "from": "new"}
    assert like_dataset.encoding == {"reticulation": "quartic"}

    # The data should remain the same.
    # The attributes and encoding should be merged.
    like_foo = like_dataset.data_vars['foo']
    numpy.testing.assert_equal(like_foo.values, new_foo.values)
    assert like_foo.attrs == {'units': 'meters'}
    assert like_foo.encoding['_FillValue'] == 10
    like_bar = like_dataset.data_vars['bar']
    numpy.testing.assert_equal(like_bar.values, new_bar.values)
    assert like_bar.attrs == {'units': 'seconds', 'long_name': 'Seconds since breakfast'}
    assert like_bar.encoding['_FillValue'] == -9999.


def test_extract_vars():
    time_size, depth_size = 4, 10
    lon_size, lat_size = 10, 15
    lon_grid, lat_grid = numpy.arange(lon_size + 1), numpy.arange(lat_size + 1)

    dataset = xarray.Dataset(
        data_vars={
            'lon_bounds': (["lon", 2], numpy.stack([lon_grid[:-1], lon_grid[1:]], axis=-1)),
            'lat_bounds': (["lat", 2], numpy.stack([lat_grid[:-1], lat_grid[1:]], axis=-1)),
            'botz': (["lat", "lon"], 50 + 25 * numpy.random.random_sample((lat_size, lon_size))),
            'eta': (
                ["time", "lat", "lon"],
                numpy.random.random_sample((time_size, lat_size, lon_size))),
            'temp': (
                ["time", "depth", "lat", "lon"],
                15 + 3 * numpy.random.random_sample((time_size, depth_size, lat_size, lon_size))),
            'salt': (
                ["time", "depth", "lat", "lon"],
                34 + numpy.random.random_sample((time_size, depth_size, lat_size, lon_size))),
        },
        coords={
            'time': (["time"], pandas.date_range("2021-12-21", periods=time_size)),
            'depth': (["depth"], numpy.arange(depth_size), {"positive": "down"}),
            'lon': (["lon"], (lon_grid[1:] + lon_grid[:-1]) / 2, {"bounds": "lon_bounds"}),
            'lat': (["lat"], (lat_grid[1:] + lat_grid[:-1]) / 2, {"bounds": "lat_bounds"}),
        },
    )

    eta_botz_and_bounds = utils.extract_vars(dataset, ["eta", "botz"])
    # Bounds variables for the coordinates are included
    assert set(eta_botz_and_bounds.data_vars.keys()) == {'lon_bounds', 'lat_bounds', 'eta', 'botz'}
    # Note that 'depth' is included in the coords, even though no variables use it
    assert set(eta_botz_and_bounds.coords.keys()) == {'time', 'depth', 'lon', 'lat'}

    salt_temp = utils.extract_vars(dataset, ["salt", "temp"], keep_bounds=False)
    # No more bounds variables
    assert set(salt_temp.data_vars.keys()) == {'salt', 'temp'}
    # All the coords are still kept
    assert set(salt_temp.coords.keys()) == {'time', 'depth', 'lon', 'lat'}

    with pytest.raises(ValueError):
        utils.extract_vars(dataset, ['eta', 'cloud'])

    just_eta = utils.extract_vars(dataset, ['eta', 'cloud'], keep_bounds=False, errors='ignore')
    assert set(just_eta.data_vars.keys()) == {'eta'}


def test_check_data_array_dimensions_match_complete():
    """
    check_data_array_dimensions_match with all dimensions present and matching
    """
    time_size = 4
    lon_size, lat_size = 10, 15

    dataset = xarray.Dataset(
        data_vars={
            'botz': (["lat", "lon"], 50 + 25 * numpy.random.random_sample((lat_size, lon_size))),
            'eta': (
                ["time", "lat", "lon"],
                numpy.random.random_sample((time_size, lat_size, lon_size)),
            ),
        },
        coords={
            'time': (["time"], pandas.date_range("2021-12-21", periods=time_size)),
            'lon': (["lon"], numpy.arange(lon_size) + 0.5),
            'lat': (["lat"], numpy.arange(lat_size) + 0.5),
        }
    )

    # This data array has the same time/lat/lon dimension size as the dataset
    # above, but is otherwise unrelated. The data can still be plotted using
    # the spatial data of the dataset, assuming the dimensions match.
    surface_temp = xarray.DataArray(
        data=15 + 3 * numpy.random.random_sample((time_size, lat_size, lon_size)),
        dims=['time', 'lat', 'lon'],
    )

    utils.check_data_array_dimensions_match(dataset, surface_temp)


def test_check_data_array_dimensions_match_subset():
    """
    check_data_array_dimensions_match with a subset of dimensions present and matching
    """
    time_size, depth_size = 4, 3
    lon_size, lat_size = 10, 15

    dataset = xarray.Dataset(
        data_vars={
            'botz': (["lat", "lon"], 50 + 25 * numpy.random.random_sample((lat_size, lon_size))),
            'temp': (
                ["time", "depth", "lat", "lon"],
                numpy.random.random_sample((time_size, depth_size, lat_size, lon_size)),
            ),
        },
        coords={
            'time': (["time"], pandas.date_range("2021-12-21", periods=time_size)),
            'depth': (["depth"], numpy.arange(depth_size), {'positive': 'down'}),
            'lon': (["lon"], numpy.arange(lon_size) + 0.5),
            'lat': (["lat"], numpy.arange(lat_size) + 0.5),
        }
    )

    # By slicing off one layer and dropping the depth dimension, the data array
    # dimensions should still match as a subset of the dataset dimensions.
    surface_temp = dataset.data_vars['temp'].isel(depth=0, drop=True)

    utils.check_data_array_dimensions_match(dataset, surface_temp)


def test_check_data_array_dimensions_match_size_mismatch():
    """
    check_data_array_dimensions_match with some dimensions having different sizes
    """
    time_size, depth_size = 4, 3
    lon_size, lat_size = 10, 15

    dataset = xarray.Dataset(
        data_vars={
            'botz': (["lat", "lon"], 50 + 25 * numpy.random.random_sample((lat_size, lon_size))),
            'temp': (
                ["time", "depth", "lat", "lon"],
                numpy.random.random_sample((time_size, depth_size, lat_size, lon_size)),
            ),
        },
        coords={
            'time': (["time"], pandas.date_range("2021-12-21", periods=time_size)),
            'depth': (["depth"], numpy.arange(depth_size), {'positive': 'down'}),
            'lon': (["lon"], numpy.arange(lon_size) + 0.5),
            'lat': (["lat"], numpy.arange(lat_size) + 0.5),
        }
    )

    # Slicing the temperature to get a subset of the lat/lon grid will cause
    # the dimensions to differ
    surface_temp = dataset.data_vars['temp'].isel(
        {'lon': numpy.s_[2:-2], 'lat': numpy.s_[3:-3], 'depth': 0},
        drop=True)

    with pytest.raises(ValueError):
        utils.check_data_array_dimensions_match(dataset, surface_temp)

    # If you subset the dataset as well, this should work fine though!
    dataset_subset = dataset.isel({'lon': numpy.s_[2:-2], 'lat': numpy.s_[3:-3]})
    surface_temp_subset = dataset_subset.data_vars['temp'].isel({'depth': 0}, drop=True)
    utils.check_data_array_dimensions_match(dataset_subset, surface_temp_subset)


def test_check_data_array_dimensions_match_unknown_dimension():
    """
    check_data_array_dimensions_match with an unknown dimension
    """
    time_size, depth_size = 4, 3
    lon_size, lat_size = 10, 15

    dataset = xarray.Dataset(
        data_vars={
            'botz': (["lat", "lon"], 50 + 25 * numpy.random.random_sample((lat_size, lon_size))),
            'eta': (
                ["time", "lat", "lon"],
                numpy.random.random_sample((time_size, lat_size, lon_size)),
            ),
        },
        coords={
            'time': (["time"], pandas.date_range("2021-12-21", periods=time_size)),
            'lon': (["lon"], numpy.arange(lon_size) + 0.5),
            'lat': (["lat"], numpy.arange(lat_size) + 0.5),
        }
    )

    # Slicing the temperature to get a subset of the lat/lon grid will cause
    # the dimensions to differ
    temp = xarray.DataArray(
        data=15 + 3 * numpy.random.random_sample((time_size, depth_size, lat_size, lon_size)),
        dims=['time', 'depth', 'lat', 'lon'],
    )

    with pytest.raises(ValueError):
        utils.check_data_array_dimensions_match(dataset, temp)

    # If you remove the extra dimension, this should be fine
    surface_temp = temp.isel({'depth': 0}, drop=True)
    utils.check_data_array_dimensions_match(dataset, surface_temp)


def test_move_dimensions_to_end():
    data_array = xarray.DataArray(
        data=numpy.random.random((2, 7, 5, 3)),
        dims=['t', 'x', 'y', 'z'],
    )

    # This should result in a dimension order of ('t', 'z', 'y', 'z')
    transposed = utils.move_dimensions_to_end(data_array, ['y', 'x'])

    # Check that the dimensions are in the correct order, with the correct shape
    assert transposed.dims == ('t', 'z', 'y', 'x')
    assert transposed.shape == (2, 3, 5, 7)

    # Check that the values were rearranged as expected
    numpy.testing.assert_equal(
        transposed.isel(t=1, z=2).values,
        numpy.transpose(data_array.isel(t=1, z=2).values))

    # This should be a no-op, as those dimensions are already at the end
    xarray.testing.assert_equal(
        data_array,
        utils.move_dimensions_to_end(data_array, ['y', 'z']))


def test_move_dimensions_to_end_missing():
    data_array = xarray.DataArray(
        data=numpy.random.random((2, 7, 5, 3)),
        dims=['t', 'x', 'y', 'z'],
    )

    with pytest.raises(ValueError) as exc:
        utils.move_dimensions_to_end(data_array, ['x', 'foo', 'bar'])

    message = "DataArray does not contain dimensions ['bar', 'foo']"
    assert str(exc.value) == message


def test_linearize_dimensions_exact_dimensions():
    data_array = xarray.DataArray(
        data=numpy.random.random((3, 5)),
        dims=['y', 'x'],
    )
    expected = xarray.DataArray(
        data=data_array.values.ravel(),
        dims=['index'],
    )
    linearized = utils.ravel_dimensions(data_array, ['y', 'x'])
    xarray.testing.assert_equal(linearized, expected)


def test_linearize_dimensions_extra_dimensions():
    data_array = xarray.DataArray(
        data=numpy.random.random((2, 7, 5, 3)),
        dims=['t', 'x', 'y', 'z'],
        coords={
            't': pandas.date_range("2022-03-02", periods=2),
            'x': numpy.arange(7),
            'y': numpy.arange(5),
            'z': [0.25, 0.5, 1.5],
        }
    )
    expected = xarray.DataArray(
        data=numpy.reshape(numpy.transpose(data_array.values, (0, 3, 2, 1)), (2, 3, -1)),
        dims=['t', 'z', 'index'],
        coords={
            't': data_array.coords['t'],
            'z': data_array.coords['z'],
        },
    )
    linearized = utils.ravel_dimensions(data_array, ['y', 'x'])
    xarray.testing.assert_equal(linearized, expected)


def test_linearize_dimensions_custom_name():
    data_array = xarray.DataArray(
        data=numpy.random.random((2, 7, 5, 3)),
        dims=['t', 'x', 'y', 'z'],
    )
    expected = xarray.DataArray(
        data=numpy.reshape(numpy.transpose(data_array.values, (0, 3, 2, 1)), (2, 3, -1)),
        dims=['t', 'z', 'i'],
    )
    linearized = utils.ravel_dimensions(data_array, ['y', 'x'], linear_dimension='i')
    xarray.testing.assert_equal(linearized, expected)


def test_linearize_dimensions_auto_name_conflict():
    data_array = xarray.DataArray(
        data=numpy.random.random((2, 7, 5, 3)),
        dims=['index', 'index_0', 'index_1', 'index_2'],
    )
    expected = xarray.DataArray(
        data=numpy.reshape(numpy.transpose(data_array.values, (0, 1, 3, 2)), (2, 7, -1)),
        dims=['index', 'index_0', 'index_1'],
    )
    linearized = utils.ravel_dimensions(data_array, ['index_2', 'index_1'])
    xarray.testing.assert_equal(linearized, expected)


def test_wind_dimension():
    data_array = xarray.DataArray(
        data=numpy.arange(6 * 5 * 4).reshape(6, -1),
        dims=['time', 'index'],
    )
    expected = xarray.DataArray(
        data=numpy.arange(6 * 5 * 4).reshape(6, 5, 4),
        dims=['time', 'y', 'x'],
    )
    wound = utils.wind_dimension(data_array, ['y', 'x'], [5, 4])
    xarray.testing.assert_equal(wound, expected)


def test_wind_dimension_middle():
    data_array = xarray.DataArray(
        data=numpy.arange(6 * 5 * 4 * 3).reshape(6, -1, 3),
        dims=['time', 'index', 'colour'],
    )
    expected = xarray.DataArray(
        data=numpy.arange(6 * 5 * 4 * 3).reshape(6, 5, 4, 3),
        dims=['time', 'y', 'x', 'colour'],
    )
    wound = utils.wind_dimension(data_array, ['y', 'x'], [5, 4])
    xarray.testing.assert_equal(wound, expected)


def test_wind_dimension_renamed():
    data_array = xarray.DataArray(
        data=numpy.arange(6 * 5 * 4).reshape(6, -1),
        dims=['time', 'ix'],
    )
    expected = xarray.DataArray(
        data=numpy.arange(6 * 5 * 4).reshape(6, 5, 4),
        dims=['time', 'y', 'x'],
    )
    wound = utils.wind_dimension(data_array, ['y', 'x'], [5, 4], linear_dimension='ix')
    xarray.testing.assert_equal(wound, expected)
