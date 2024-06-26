"""
These tests don't have much to do with emsarray directly, but relate to bugs in
xarray found by people while using emsarray. emsarray may add workarounds to
these xarray issues if we choose, so tracking the resolution of the bugs is
useful.
"""
import pathlib
import re

import numpy
import pytest
import xarray

from tests.utils import only_versions


def make_dataset():
    one_day = numpy.timedelta64(1, 'D').astype('timedelta64[ns]')
    nat = numpy.timedelta64('nat')

    data = numpy.arange(5) * one_day
    data[2] = nat
    data[4] = nat

    period = xarray.DataArray(data=data, dims=['x'])
    period.encoding.update({
        "units": "days",
        "_FillValue": numpy.int16(-1),
        "dtype": numpy.dtype('int16'),
    })
    dataset = xarray.Dataset(data_vars={'period': period})

    return dataset


@only_versions('xarray < 2023.09.0')
def test_xarray_timedelta_units_unsafe_cast(
    tmp_path: pathlib.Path,
):
    """
    When saving a dataset to disk, xarray.coding.times.cast_to_int_if_safe
    will check if it is possible to encode a timedelta64 using integer values
    by casting the values and checking for equality.
    Recent versions of numpy will emit warnings
    when casting a data array with dtype timedelta64 to int
    if it contains NaT (not a time) values.
    xarray fixed this but introduced a new issue instead.

    See also
    ========
    https://github.com/pydata/xarray/issues/7942
    https://github.com/pydata/xarray/issues/9134

    """
    dataset = make_dataset()
    with pytest.warns(RuntimeWarning, match='invalid value encountered in cast'):
        dataset.to_netcdf(tmp_path / "period.nc")


@only_versions('xarray >= 2024.02.0')
def test_xarray_timedelta_units_cast_runtimewarning(
    tmp_path: pathlib.Path,
):
    """
    When saving a dataset to disk, xarray.coding.times._cast_to_dtype_if_safe
    will check if it is possible to encode a timedelta64 using integer values
    by casting the values and checking for equality.
    Recent versions of xarray will raise an OverflowError
    when casting a data array with dtype timedelta64 to int16
    if it contains NaT (not a time) values.

    See also
    ========
    https://github.com/pydata/xarray/issues/9134
    """
    dataset = make_dataset()
    message = "Not possible to cast encoded times from dtype('int64') to dtype('int16') without overflow."
    with pytest.raises(OverflowError, match=re.escape(message)):
        dataset.to_netcdf(tmp_path / "period.nc")
