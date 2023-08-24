from __future__ import annotations

import pathlib

import netCDF4
import numpy as np
import pandas as pd
import pytest
import shapely.geometry
import xarray
from numpy.testing import assert_equal

import emsarray
from emsarray.conventions.arakawa_c import (
    ArakawaCGridKind, c_mask_from_centres
)
from tests.utils import DiagonalShocGrid, ShocLayerGenerator, mask_from_strings


def test_standard_mask_from_centres():
    centres = mask_from_strings([
        "00000",
        "01111",
        "01110",
        "01100",
        "01000",
        "00000",
    ])
    mask = c_mask_from_centres(centres, {
        ArakawaCGridKind.face: ('j_centre', 'i_centre'),
        ArakawaCGridKind.back: ('j_back', 'i_back'),
        ArakawaCGridKind.left: ('j_left', 'i_left'),
        ArakawaCGridKind.node: ('j_node', 'i_node'),
    })

    # Check the dimensions are all the correct sizes.
    # The centre should be the same as the input array.
    # Left should be one bigger in the i dimension, back in j.
    # Grid should be one bigger in both j and i dimensions.
    assert mask.dims == {
        'j_centre': 6, 'i_centre': 5,
        'j_node': 7, 'i_node': 6,
        'j_left': 6, 'i_left': 6,
        'j_back': 7, 'i_back': 5,
    }
    # There should be four masks here
    assert set(mask.data_vars.keys()) == {'face_mask', 'left_mask', 'back_mask', 'node_mask'}

    # Check that the data in each mask
    assert_equal(mask.data_vars['face_mask'].values, centres)
    assert_equal(mask.data_vars['left_mask'].values, mask_from_strings([
        "000000",
        "011111",
        "011110",
        "011100",
        "011000",
        "000000",
    ]))
    assert_equal(mask.data_vars['back_mask'].values, mask_from_strings([
        "00000",
        "01111",
        "01111",
        "01110",
        "01100",
        "01000",
        "00000",
    ]))
    assert_equal(mask.data_vars['node_mask'].values, mask_from_strings([
        "000000",
        "011111",
        "011111",
        "011110",
        "011100",
        "011000",
        "000000",
    ]))


def test_mask_dataset(tmp_path: pathlib.Path):
    # There are a number of things that need testing here:
    #
    # * The dataset is trimmed to the bounds of the mask
    # * Attributes and encodings of variables and the dataset are preserved
    # * Data arrays that can be masked are masked, those that can't be are preserved
    # * Different coordinate sets are masked using the appropriate mask

    # The input dataset is a 6x5 grid, and the mask defines a 4x3 area in the
    # centre with the (2,2), (3,1), and (3,2) indices not included. The input
    # dataset is nominally a SHOC standard dataset.
    centres = mask_from_strings(["00000", "01110", "01110", "01100", "01000", "00000"])
    mask = c_mask_from_centres(centres, {
        ArakawaCGridKind.face: ('j_centre', 'i_centre'),
        ArakawaCGridKind.back: ('j_back', 'i_back'),
        ArakawaCGridKind.left: ('j_left', 'i_left'),
        ArakawaCGridKind.node: ('j_node', 'i_node'),
    })

    records, k_size, j_size, i_size = 8, 7, 6, 5
    grid = DiagonalShocGrid(j=j_size, i=i_size)
    layers = ShocLayerGenerator(k=k_size)

    # int data without a fill value. This should be sliced but not masked.
    flag1 = xarray.DataArray(
        data=np.arange(k_size * j_size * i_size, dtype=np.short).reshape(k_size, j_size, i_size),
        dims=['k_centre', 'j_centre', 'i_centre'],
    )
    # int data with a fill value. This should be sliced and masked.
    flag2 = xarray.DataArray(
        data=np.arange(k_size * j_size * i_size, dtype=np.short).reshape(k_size, j_size, i_size) + 25,
        dims=['k_centre', 'j_centre', 'i_centre'],
    )
    # This belongs in attrs. If you open a dataset with an int variable that
    # has a _FillValue, it will be converted to a float variable. This
    # behaviour is disabled if you set `mask_and_scale=False` when opening the
    # dataset, but this leaves `_FillValue` in attrs not encoding
    flag2.encoding['_FillValue'] = np.short(-999)

    # Some variables are only defined on one axis. These should be sliced to
    # the mask bounds, but not masked
    only_j = xarray.DataArray(
        data=np.arange(j_size).astype(np.short),
        dims=["j_centre"],
    )
    only_i = xarray.DataArray(
        data=np.arange(i_size).astype(np.short),
        dims=["i_centre"],
    )

    t = xarray.DataArray(
        data=list(pd.date_range("2021-11-11", periods=records)),
        dims=["record"],
        attrs={
            "long_name": "Time",
            "standard_name": "time",
            "coordinate_type": "time",
        },
    )
    t.encoding["units"] = "days since 1990-01-01 00:00:00 +10"

    botz_missing_value = np.float32(-99.)
    botz = xarray.DataArray(
        data=np.random.random((j_size, i_size)).astype(np.float32) * 10 + 50,
        dims=["j_centre", "i_centre"],
        attrs={
            "units": "metre",
            "long_name": "Depth of sea-bed",
            "standard_name": "depth",
            "positive": "down",
            "outside": "9999",
        }
    )
    botz.encoding['missing_value'] = botz_missing_value

    eta = xarray.DataArray(
        data=np.random.normal(0, 0.2, (records, j_size, i_size)),
        dims=["record", "j_centre", "i_centre"],
        attrs={
            "units": "metre",
            "long_name": "Surface elevation",
            "standard_name": "sea_surface_height_above_geoid",
        }
    )
    temp = xarray.DataArray(
        data=np.random.normal(12, 0.5, (records, k_size, j_size, i_size)),
        dims=["record", "k_centre", "j_centre", "i_centre"],
        attrs={
            "units": "degrees C",
            "long_name": "Temperature",
        },
    )

    dataset = xarray.Dataset(
        data_vars={
            **layers.standard_vars,
            **grid.standard_vars,
            "botz": botz,
            "eta": eta,
            "flag1": flag1,
            "flag2": flag2,
            "t": t,
            "temp": temp,
            "only_j": only_j,
            "only_i": only_i,
        },
        # This coord is not used in any data array but should be preserverd
        coords={"some_coord": (["some_coord"], np.arange(30, dtype=int), {'reticulating': 'splines'})},
        attrs={
            "title": "Example SHOC dataset",
            "ems_version": "v1.2.3 fake",
            "Conventions": "CMR/Timeseries/SHOC",
            "nce1": j_size,
            "nce2": i_size,
            "nfe1": j_size + 1,
            "nfe2": i_size + 1,
            "gridtype": "NUMERICAL",
        },
    )
    dataset.encoding["unlimited_dims"] = {"record"}

    # Clip it!
    clipped = dataset.ems.apply_clip_mask(mask, tmp_path)
    # Test that the clipped dataset can be saved to disk
    clipped.ems.to_netcdf(tmp_path / "clipped.nc")
    del clipped

    # Open this with plain netCDF4 so we can introspect the dataset without
    # xarray reinterpreting our data
    with netCDF4.Dataset(tmp_path / "clipped.nc", "r") as nc_dataset:
        # Dataset attributes should have been preserved
        assert nc_dataset.getncattr('title') == "Example SHOC dataset"
        assert nc_dataset.getncattr('Conventions') == "CMR/Timeseries/SHOC"

        # Some dimensions were not trimmed, and should be the same size
        record = nc_dataset.dimensions['record']
        assert record.size == records
        assert record.isunlimited(), "`record` dimension should still be unlimited"
        assert nc_dataset.dimensions['k_centre'].size == k_size
        assert nc_dataset.dimensions['k_grid'].size == k_size + 1
        assert nc_dataset.dimensions['some_coord'].size == 30

        # These should have been trimmed
        assert nc_dataset.dimensions['j_centre'].size == 4
        assert nc_dataset.dimensions['i_centre'].size == 3
        assert nc_dataset.dimensions['j_node'].size == 5
        assert nc_dataset.dimensions['i_node'].size == 4
        assert nc_dataset.dimensions['j_left'].size == 4
        assert nc_dataset.dimensions['i_left'].size == 4
        assert nc_dataset.dimensions['j_back'].size == 5
        assert nc_dataset.dimensions['i_back'].size == 3

        # Variable attributes should have been preserved
        temp = nc_dataset.variables['temp']
        # This variable should not have a _FillValue, even though we are now
        # using nan as a fill value. Modifying attributes is complicated. Some
        # variables have `missing_value`, for example, and setting
        # a `_FillValue` on these would cause Problems.
        assert set(temp.ncattrs()) == {'units', 'long_name'}
        assert temp.getncattr('units') == 'degrees C'
        assert temp.getncattr('long_name') == 'Temperature'
        # The corner should be masked with nan
        assert np.isnan(temp[0, 0, 2, 2])

        # This variable had a missing_value instead of _FillValue, which should
        # have been used
        botz = nc_dataset.variables['botz']
        assert set(botz.ncattrs()) == {
            'units', 'long_name', 'standard_name',
            'positive', 'outside', 'missing_value',
        }
        assert botz.getncattr('missing_value') == botz_missing_value
        # The corner should be masked with nan
        assert botz[2, 2] is np.ma.masked

        # The flag1 variable should have been trimmed, but not masked
        nc_flag1 = nc_dataset.variables['flag1']
        assert nc_flag1.shape == (k_size, 4, 3)
        assert_equal(nc_flag1[:], flag1.values[:, 1:5, 1:4])
        assert '_FillValue' not in nc_flag1.ncattrs()

        # The flag2 variable should have been trimmed and also masked
        nc_flag2 = nc_dataset.variables['flag2']
        assert nc_flag2.shape == (k_size, 4, 3)
        flag2_mask = np.stack([np.array([
            [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]
        ])] * k_size).astype(bool)
        expected: np.ndarray = np.ma.masked_array(
            flag2.values[:, 1:5, 1:4].copy(),
            mask=flag2_mask,
        )
        assert_equal(nc_flag2[:], expected)
        assert nc_flag2.getncattr('_FillValue') == np.short(-999)

        # The only_j and only_i variables should have been trimmed, but not
        # masked
        nc_only_j = nc_dataset.variables['only_j']
        assert nc_only_j.shape == (4,)
        assert_equal(nc_only_j[:], np.arange(1, 5).astype(int))
        nc_only_i = nc_dataset.variables['only_i']
        assert nc_only_i.shape == (3,)
        assert_equal(nc_only_i[:], np.arange(1, 4).astype(int))

        # This coordinate should have passed through unchanged
        nc_some_coord = nc_dataset.variables['some_coord']
        assert nc_some_coord.shape == (30,)
        assert_equal(nc_some_coord[:], np.arange(30, dtype=int))
        assert nc_some_coord.getncattr('reticulating') == 'splines'


@pytest.mark.tutorial
def test_clip_unmasked_dataset(tmp_path: pathlib.Path):
    """
    Test clipping the bran2020 dataset. The `temp` variable in this dataset is
    stored as a int with a _FillValue, add_offset, and scale_factor.
    If opened with mask_and_scale=False, clipping must use _FillValue directly.

    See Also
    --------
    https://github.com/csiro-coasts/emsarray/pull/4
    """
    dataset = emsarray.tutorial.open_dataset('bran2020')
    dataset = dataset.isel({'st_ocean': 0}, drop=True)

    tasmania_clip = shapely.geometry.Polygon([
        (141.459, -40.780),
        (142.954, -39.198),
        (149.106, -39.095),
        (150.864, -41.376),
        (149.809, -44.621),
        (144.843, -45.706),
        (141.723, -43.389),
        (141.372, -40.913),
    ])

    tassie = dataset.ems.clip(tasmania_clip, tmp_path)
    tassie.ems.to_netcdf(tmp_path / "tassie.nc")
    assert np.isnan(tassie['temp'].values[0, 0, 0])
    tassie.close()

    tassie = emsarray.open_dataset(tmp_path / "tassie.nc")
    assert np.isnan(tassie['temp'].values[0, 0, 0])
