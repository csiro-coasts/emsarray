from __future__ import annotations

import pathlib

import netCDF4
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_equal

from emsarray import masking
from emsarray.utils import to_netcdf_with_fixes
from tests.utils import filter_warning, mask_from_strings


def assert_raw_values(
    dataset_path: pathlib.Path,
    variable_name: str,
    values: np.ndarray,
) -> None:
    __tracebackhide__ = True
    with netCDF4.Dataset(dataset_path, 'r') as ds:
        ds.set_auto_maskandscale(False)
        var = ds.variables[variable_name]
        assert_dtype_equal(var[:], values)


def assert_dtype_equal(actual, desired):
    __tracebackhide__ = True
    assert_equal(actual, desired)
    assert actual.dtype == desired.dtype


def test_find_fill_value_masked_float(datasets):
    dataset_path = datasets / 'masking/find_fill_value/float_with_fill_value.nc'

    assert_raw_values(
        dataset_path, 'var',
        np.array([[1.0, 2.0], [-999., 4.0]], dtype=np.float64))

    # When opened with mask_and_scale=True (the default) xarray uses np.nan
    # to indicate masked values.
    with xr.open_dataarray(dataset_path, mask_and_scale=True) as data_array:
        assert_dtype_equal(
            data_array.values,
            np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float64))
        assert masking.find_fill_value(data_array) is np.nan

    # When opened with mask_and_scale=False, xarray does nothing with masks.
    # The raw _FillValue should be returned.
    with xr.open_dataarray(dataset_path, mask_and_scale=False) as data_array:
        assert_dtype_equal(
            data_array.values,
            np.array([[1.0, 2.0], [-999., 4.]], dtype=np.float64))
        assert_dtype_equal(masking.find_fill_value(data_array), np.float64(-999.))


def test_find_fill_value_masked_and_scaled_float(datasets):
    dataset_path = datasets / 'masking/find_fill_value/float_with_fill_value_and_offset.nc'

    assert_raw_values(
        dataset_path, 'var',
        np.array([[-4.5, -4.0], [-999., -3.0]], dtype=np.float32))

    # When opened with mask_and_scale=True (the default) xarray uses np.nan
    # to indicate masked values.
    with xr.open_dataarray(dataset_path, mask_and_scale=True) as data_array:
        assert_dtype_equal(
            data_array.values,
            np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32))
        assert masking.find_fill_value(data_array) is np.nan

    # When opened with mask_and_scale=False, xarray does nothing with masks.
    # The raw _FillValue should be returned.
    with xr.open_dataarray(dataset_path, mask_and_scale=False) as data_array:
        assert_dtype_equal(
            data_array.values,
            np.array([[-4.5, -4.0], [-999., -3.0]], dtype=np.float32))
        assert_dtype_equal(masking.find_fill_value(data_array), np.float32(-999.))


def test_find_fill_value_masked_and_scaled_int(datasets):
    dataset_path = datasets / 'masking/find_fill_value/int_with_fill_value_and_offset.nc'

    assert_raw_values(
        dataset_path, 'var',
        np.array([[22, 24], [-1, 28]], dtype=np.int8))

    # When opened with mask_and_scale=True (the default) xarray uses np.nan
    # to indicate masked values.
    with xr.open_dataarray(dataset_path, mask_and_scale=True) as data_array:
        assert_dtype_equal(
            data_array.values,
            np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32))
        assert masking.find_fill_value(data_array) is np.nan

    # When opened with mask_and_scale=False, xarray does nothing with masks.
    # The raw _FillValue should be returned.
    with xr.open_dataarray(dataset_path, mask_and_scale=False) as data_array:
        assert_dtype_equal(
            data_array.values,
            np.array([[22, 24], [-1, 28]], dtype=np.int8))
        assert_dtype_equal(masking.find_fill_value(data_array), np.int8(-1))


def test_find_fill_value_timedelta_with_missing_value(
    datasets: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    dataset_path = datasets / 'masking/find_fill_value/timedelta_with_missing_value.nc'

    missing_value = np.float32(1.e35)
    assert_raw_values(
        dataset_path, 'var',
        np.array([[0, 1], [2, missing_value]], dtype=np.float32))

    with xr.open_dataset(dataset_path) as dataset:
        data_array = dataset['var']
        assert dataset['var'].dtype == np.dtype('timedelta64[ns]')
        fill_value = masking.find_fill_value(data_array)
        assert np.isnat(fill_value)

        # See https://github.com/pydata/xarray/issues/7942
        with filter_warning(
            'ignore', category=RuntimeWarning,
            message='invalid value encountered in cast',
            module=r'xarray\.coding\.times',
        ):
            # Write this out for easier debugging purposes
            to_netcdf_with_fixes(dataset, tmp_path / 'dataset.nc')


def test_calculate_mask_bounds():
    mask = xr.Dataset(
        data_vars={
            '1d': xr.DataArray(
                # This has bounds {"1x": [2:5]}
                data=np.array([False, False, True, False, True, False]),
                dims=['1x'],
            ),
            '2d': xr.DataArray(
                # This has bounds {"2y": [1:4], "2x": [1:5]}
                data=mask_from_strings([
                    "000000",
                    "000010",
                    "000110",
                    "001100",
                    "000000",
                    "000000",
                ]),
                dims=['2y', '2x'],
            ),
            '3d': xr.DataArray(
                # This has bounds {"3z": [0:1], "3y": [0:2], "3x": [1:2]}
                data=np.stack([
                    mask_from_strings(["010", "010", "000"]),
                    mask_from_strings(["000", "000", "000"]),
                ]),
                dims=['3z', '3y', '3x'],
            ),
        },
    )

    assert masking.calculate_grid_mask_bounds(mask) == {
        '1x': np.s_[2:5],
        '2y': np.s_[1:4],
        '2x': np.s_[2:5],
        '3z': np.s_[0:1],
        '3y': np.s_[0:2],
        '3x': np.s_[1:2],
    }


def test_calculate_mask_bounds_empty():
    mask = xr.Dataset(
        data_vars={
            '2d': xr.DataArray(
                # This mask has no True items, and should raise an error
                data=np.zeros((5, 5), dtype=bool),
                dims=['2x', '2y'],
            )
        },
    )

    with pytest.raises(ValueError, match="Mask '2d' is completely empty!"):
        masking.calculate_grid_mask_bounds(mask)


@pytest.mark.parametrize(
    ('padding', 'original', 'expected'),
    [
        (
            (True, True),
            mask_from_strings(["1"]),
            mask_from_strings(["11", "11"]),
        ),
        (
            (False, True),
            mask_from_strings(["1"]),
            mask_from_strings(["11"]),
        ),
        (
            (True, False),
            mask_from_strings(["1"]),
            mask_from_strings(["1", "1"]),
        ),
        (
            (False, False),
            mask_from_strings(["1"]),
            mask_from_strings(["1"]),
        ),
        (
            (False, True),
            mask_from_strings(["00100", "01010", "10001"]),
            mask_from_strings(["001100", "011110", "110011"]),
        ),
        (
            (True, True),
            mask_from_strings(["00100", "01010", "10001"]),
            mask_from_strings(["001100", "011110", "111111", "110011"]),
        ),
        (
            # Lets be fancy and try a one-dimensional case
            (True,),
            np.array([True, False, False, False, True, False]),
            np.array([True, True, False, False, True, True, False]),
        ),
        (
            # Even fancier - three-dimensional case
            (True, True, True),
            np.stack([
                mask_from_strings(["1000", "0000", "0000", "0000"]),
                mask_from_strings(["0000", "0100", "0000", "0000"]),
                mask_from_strings(["0000", "0000", "0010", "0000"]),
                mask_from_strings(["0000", "0000", "0000", "0001"]),
            ]),
            np.stack([
                mask_from_strings(["11000", "11000", "00000", "00000", "00000"]),
                mask_from_strings(["11000", "11100", "01100", "00000", "00000"]),
                mask_from_strings(["00000", "01100", "01110", "00110", "00000"]),
                mask_from_strings(["00000", "00000", "00110", "00111", "00011"]),
                mask_from_strings(["00000", "00000", "00000", "00011", "00011"]),
            ]),
        ),
    ],
)
def test_smear_mask(padding, original, expected):
    smeared = masking.smear_mask(original, padding)
    assert_equal(smeared, expected)


@pytest.mark.parametrize(
    ('original', 'expected'),
    [
        # 1D arrays
        (np.array([True]), np.array([True])),
        (np.array([True, False, False]), np.array([True, True, False])),
        # 2D arrays
        (mask_from_strings(["0"]), mask_from_strings(["0"])),
        (mask_from_strings(["1"]), mask_from_strings(["1"])),
        (mask_from_strings(["10", "00"]), mask_from_strings(["11", "11"])),
        (mask_from_strings(["000", "010", "000"]), mask_from_strings(["111", "111", "111"])),
        (
            mask_from_strings(["10000", "00000", "00010", "00001"]),
            mask_from_strings(["11000", "11111", "00111", "00111"]),
        ),
        # 3D arrays
        (
            np.stack([
                mask_from_strings(["1000", "0000", "0000"]),
                mask_from_strings(["0000", "0100", "0000"]),
                mask_from_strings(["0000", "0000", "0000"]),
                mask_from_strings(["1000", "0000", "0000"]),
            ]),
            np.stack([
                mask_from_strings(["1110", "1110", "1110"]),
                mask_from_strings(["1110", "1110", "1110"]),
                mask_from_strings(["1110", "1110", "1110"]),
                mask_from_strings(["1100", "1100", "0000"]),
            ]),
        ),
    ],
)
def test_blur_mask(original, expected):
    blurred = masking.blur_mask(original)
    assert_equal(blurred, expected)
