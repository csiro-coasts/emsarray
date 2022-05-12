from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_equal

from emsarray import masking
from tests.utils import mask_from_strings


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
