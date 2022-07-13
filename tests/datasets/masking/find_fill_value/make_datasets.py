#!/usr/bin/env python3

import pathlib

import netCDF4
import numpy as np

here = pathlib.Path(__file__).parent


def make_float_with_fill_value() -> None:
    ds = netCDF4.Dataset(here / "float_with_fill_value.nc", "w", "NETCDF4")
    ds.createDimension("x", 2)
    ds.createDimension("y", 2)

    fill_value = np.float64(-999.0)
    var = ds.createVariable("var", "f8", ["y", "x"], fill_value=fill_value)
    var[:] = [[1.0, 2.0], [fill_value, 4.0]]

    ds.close()


def make_float_with_fill_value_and_offset() -> None:
    ds = netCDF4.Dataset(here / "float_with_fill_value_and_offset.nc", "w", "NETCDF4")
    ds.createDimension("x", 2)
    ds.createDimension("y", 2)

    fill_value = np.float32(-999.0)
    var = ds.createVariable("var", "f4", ["y", "x"], fill_value=fill_value)
    var.add_offset = 10.
    var.scale_factor = 2.
    var[:] = np.ma.masked_array(np.arange(4) + 1, [0, 0, 1, 0]).reshape((2, 2))

    ds.close()


def make_int_with_fill_value_and_offset() -> None:
    ds = netCDF4.Dataset(here / "int_with_fill_value_and_offset.nc", "w", "NETCDF4")
    ds.createDimension("x", 2)
    ds.createDimension("y", 2)

    fill_value = np.int8(-1)
    var = ds.createVariable("var", "i1", ["y", "x"], fill_value=fill_value)
    var.add_offset = -10.
    var.scale_factor = .5
    var[:] = np.ma.masked_array(np.arange(4) + 1, [0, 0, 1, 0]).reshape((2, 2))

    ds.close()


if __name__ == '__main__':
    make_float_with_fill_value()
    make_float_with_fill_value_and_offset()
    make_int_with_fill_value_and_offset()
