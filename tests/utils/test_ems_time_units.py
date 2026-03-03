import datetime
import pathlib

import netCDF4
import pytest
import xarray

from emsarray import utils


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
