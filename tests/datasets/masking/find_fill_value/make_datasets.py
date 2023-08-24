#!/usr/bin/env python3

"""
Make some datasets for testing fill values.
Because of how xarray preprocesses variables to apply masks,
it is easier to construct these datasets using the plain netCDF4 library,
save the datasets to disk, and then load them using xarray.
This guarantees that the behaviour in the tests will replicate real-world use.

Running this script will overwrite any datasets already constructed in this directory.
This operation should result in byte-for-byte identical datasets each time it is run.
However each netCDF4 dataset will encode the versions of the
netCDF4, hdf5 and other relevant libraries used to construct the dataset.
If the versions have changed, the script will create new files that git thinks have changed.
"""

import pathlib

import netCDF4
import numpy

here = pathlib.Path(__file__).parent


def make_float_with_fill_value(
    output_path: pathlib.Path = here / "float_with_fill_value.nc"
) -> None:
    ds = netCDF4.Dataset(output_path, "w", "NETCDF4")
    ds.createDimension("x", 2)
    ds.createDimension("y", 2)

    fill_value = numpy.float64(-999.0)
    var = ds.createVariable("var", "f8", ["y", "x"], fill_value=fill_value)
    var[:] = [[1.0, 2.0], [fill_value, 4.0]]

    ds.close()


def make_float_with_fill_value_and_offset(
    output_path: pathlib.Path = here / "float_with_fill_value_and_offset.nc",
) -> None:
    ds = netCDF4.Dataset(output_path, "w", "NETCDF4")
    ds.createDimension("x", 2)
    ds.createDimension("y", 2)

    fill_value = numpy.float32(-999.0)
    var = ds.createVariable("var", "f4", ["y", "x"], fill_value=fill_value)
    var.add_offset = 10.
    var.scale_factor = 2.
    var[:] = numpy.ma.masked_array(numpy.arange(4) + 1, [0, 0, 1, 0]).reshape((2, 2))

    ds.close()


def make_timedelta_with_missing_value(
    output_path: pathlib.Path = here / "timedelta_with_missing_value.nc",
) -> None:
    ds = netCDF4.Dataset(output_path, "w", "NETCDF4")
    ds.createDimension("x", 2)
    ds.createDimension("y", 2)

    missing_value = numpy.float32(1.e+35)
    var = ds.createVariable("var", "f4", ["y", "x"], fill_value=False)
    var.missing_value = missing_value
    var.units = "days"
    var[:] = numpy.arange(4).reshape((2, 2))
    var[1, 1] = missing_value

    ds.close()


def make_int_with_fill_value_and_offset(
    output_path: pathlib.Path = here / "int_with_fill_value_and_offset.nc",
) -> None:
    ds = netCDF4.Dataset(output_path, "w", "NETCDF4")
    ds.createDimension("x", 2)
    ds.createDimension("y", 2)

    fill_value = numpy.int8(-1)
    var = ds.createVariable("var", "i1", ["y", "x"], fill_value=fill_value)
    var.add_offset = -10.
    var.scale_factor = .5
    var[:] = numpy.ma.masked_array(numpy.arange(4) + 1, [0, 0, 1, 0]).reshape((2, 2))

    ds.close()


if __name__ == '__main__':
    make_float_with_fill_value()
    make_float_with_fill_value_and_offset()
    make_timedelta_with_missing_value()
    make_int_with_fill_value_and_offset()
