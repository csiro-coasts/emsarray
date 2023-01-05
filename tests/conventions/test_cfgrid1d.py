from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.figure import Figure
from numpy.testing import assert_allclose, assert_equal
from shapely.geometry import Polygon, box

from emsarray.conventions import get_dataset_convention
from emsarray.conventions.grid import CFGrid1D, CFGridKind, CFGridTopology
from emsarray.operations import geometry
from tests.utils import mask_from_strings


def make_dataset(
    *,
    width: int,
    height: int,
    depth: int = 5,
    time_size: int = 4,
) -> xr.Dataset:
    longitude_name = 'lon'
    latitude_name = 'lat'
    depth_name = 'depth'
    time_name = 'time'

    lon = xr.DataArray(
        data=np.arange(width) * 0.1,
        dims=[longitude_name],
        name=longitude_name,
        attrs={
            'standard_name': 'longitude',
            'long_name': "Longitude of cell centres",
            'coordinate_type': 'longitude',
            'units': 'degrees_east',
            'projection': 'geographic',
        },
    )
    lat = xr.DataArray(
        data=np.arange(height) * 0.1,
        dims=[latitude_name],
        name=latitude_name,
        attrs={
            'standard_name': 'latitude',
            'long_name': "Latitude of cell centres",
            'coordinate_type': 'latitude',
            'units': 'degrees_north',
            'projection': 'geographic',
        },
    )
    depth_var = xr.DataArray(
        data=(-1 * np.arange(0, depth))[::-1],
        dims=[depth_name],
        name=depth_name,
        attrs={
            'standard_name': 'depth',
            'long_name': 'Layer depth',
            'coordinate_type': 'depth',
        },
    )

    time = xr.DataArray(
        # Note: Using pd.date_range() directly here will lead to strange
        # behaviours, where the `time` dimension becomes a data variable with
        # a datetime64 dtype. Using a list of datetimes instead seems to avoid
        # this, resulting in time simply being a dimension.
        data=list(pd.date_range("2021-11-11", periods=time_size)),
        dims=[time_name],
        name=time_name,
        attrs={
            "long_name": "Time",
            "standard_name": "time",
            "coordinate_type": "time",
        },
    )
    # Note: xarray will reformat this in to 1990-01-01T00:00:00+10:00, which
    # EMS fails to parse. There is no way around this using xarray natively,
    # you have to adjust it with nctool after saving it.
    time.encoding["units"] = "days since 1990-01-01 00:00:00 +10"

    botz = xr.DataArray(
        data=np.random.random((height, width)) * 10 + 50,
        dims=[latitude_name, longitude_name],
        name="botz",
        attrs={
            "units": "metre",
            "long_name": "Z coordinate at sea-bed at cell centre",
            "standard_name": "depth",
            "positive": "down",
            "outside": "9999",
        }
    )
    eta = xr.DataArray(
        data=np.random.normal(0, 0.2, (time_size, height, width)),
        dims=[time_name, latitude_name, longitude_name],
        name="eta",
        attrs={
                "units": "metre",
                "long_name": "Surface elevation",
                "standard_name": "sea_surface_height_above_geoid",
            }
    )
    temp = xr.DataArray(
        data=np.random.normal(12, 0.5, (time_size, depth, height, width)),
        dims=[time_name, depth_name, latitude_name, longitude_name],
        name="temp",
        attrs={
            "units": "degrees C",
            "long_name": "Temperature",
        },
    )

    dataset = xr.Dataset(
        data_vars={var.name: var for var in [
            time, lat, lon, depth_var,
            botz, eta, temp
        ]},
        attrs={
            'title': "COMPAS defalt version",
            'paramhead': "Example COMPAS grid",
            'paramfile': "in.prm",
            'version': "v1.0 rev(1234)",
            'Conventions': "UGRID-1.0",
            'start_index': 0,
        },
    )
    dataset.encoding['unlimited_dims'] = {'time'}
    return dataset


def test_make_dataset():
    dataset = make_dataset(width=11, height=7, depth=5)

    # Check that this is recognised as a UGrid dataset
    assert get_dataset_convention(dataset) is CFGrid1D

    # Check that the correct convention class is used
    assert isinstance(dataset.ems, CFGrid1D)

    # Check the coordinate generation worked.
    lats = dataset.variables["lat"]
    lons = dataset.variables["lon"]
    assert_allclose(lats.values, np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    assert_allclose(lons.values, np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))


def test_varnames():
    dataset = make_dataset(width=11, height=7, depth=5)
    assert dataset.ems.get_depth_name() == 'depth'
    assert dataset.ems.get_time_name() == 'time'


def test_polygons():
    dataset = make_dataset(width=3, height=4)
    polygons = dataset.ems.polygons

    # Should be one item for every face
    assert len(polygons) == 3 * 4

    # There should be no empty polygons
    assert all(poly is not None for poly in polygons)
    assert all(dataset.ems.mask)

    # Check the coordinates for the generated polygons.
    assert polygons[0].equals_exact(box(-0.05, -0.05, 0.05, 0.05), 1e-6)
    assert polygons[5].equals_exact(box(.15, .05, .25, .15), 1e-6)


def test_selector_for_index():
    dataset = make_dataset(width=11, height=7, depth=5)
    convention: CFGrid1D = dataset.ems

    index = (3, 4)
    selector = {'lat': 3, 'lon': 4}
    assert selector == convention.selector_for_index(index)


def test_make_geojson_geometry():
    dataset = make_dataset(width=3, height=3)
    out = json.dumps(geometry.to_geojson(dataset))
    assert isinstance(out, str)


def test_ravel():
    dataset = make_dataset(width=3, height=5)
    convention = CFGrid1D(dataset)
    for index in range(3 * 5):
        y, x = divmod(index, 3)
        assert convention.ravel_index((y, x)) == index
        assert convention.unravel_index(index) == (y, x)


def test_grid_kinds():
    dataset = make_dataset(width=3, height=3)
    convention: CFGrid1D = dataset.ems

    assert convention.grid_kinds == frozenset({CFGridKind.face})
    assert convention.default_grid_kind == CFGridKind.face


def test_grid_kind_and_size():
    dataset = make_dataset(width=3, height=5)
    convention = CFGrid1D(dataset)
    grid_kind, size = convention.get_grid_kind_and_size(dataset.data_vars['temp'])
    assert grid_kind is CFGridKind.face
    assert size == 3 * 5


def test_drop_geometry(datasets: pathlib.Path):
    dataset = xr.open_dataset(datasets / 'cfgrid1d.nc')

    dropped = dataset.ems.drop_geometry()
    assert dropped.dims.keys() == {'lon', 'lat'}

    topology = dataset.ems.topology
    assert topology.longitude_name in dataset.variables
    assert topology.longitude_name in dataset.variables
    assert topology.longitude_name not in dropped.variables
    assert topology.longitude_name not in dropped.variables


def test_values():
    dataset = make_dataset(width=3, height=5)
    eta = dataset.data_vars["eta"].isel(time=0)
    values = dataset.ems.make_linear(eta)

    # There should be one value per cell polygon
    assert len(values) == len(dataset.ems.polygons)

    # The values should be in a specific order
    assert_equal(values, eta.values.ravel())


@pytest.mark.matplotlib
def test_plot_on_figure():
    # Not much to test here, mostly that it doesn't throw an error
    dataset = make_dataset(width=3, height=5)
    surface_temp = dataset.data_vars["temp"].isel(depth=-1, time=0)

    figure = Figure()
    dataset.ems.plot_on_figure(figure, surface_temp)

    assert len(figure.axes) == 2


def test_make_clip_mask():
    dataset = make_dataset(width=10, height=11)
    convention = CFGrid1D(dataset)
    topology = convention.topology

    # The dataset will have cells with centres from 0-.5 longitude, 0-.7 latitude
    clip_geometry = Polygon([
        (0.18, .30), (.40, .30), (.60, .51), (.60, .64), (.18, .64), (.18, .30),
    ])

    mask = convention.make_clip_mask(clip_geometry)
    expected_cells = mask_from_strings([
        "0000000000",
        "0000000000",
        "0000000000",
        "0011100000",
        "0011110000",
        "0011111000",
        "0011111000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
    ])
    assert_equal(mask.data_vars['cell_mask'].values, expected_cells)

    assert mask.attrs == {'type': 'CFGrid mask'}
    assert mask.dims == {
        topology.longitude_name: topology.longitude.size,
        topology.latitude_name: topology.latitude.size,
    }
    assert list(mask.data_vars.keys()) == ['cell_mask']

    # Test adding a buffer also
    mask = convention.make_clip_mask(clip_geometry, buffer=1)
    expected_cells = mask_from_strings([
        "0000000000",
        "0000000000",
        "0111110000",
        "0111111000",
        "0111111100",
        "0111111100",
        "0111111100",
        "0111111100",
        "0000000000",
        "0000000000",
        "0000000000",
    ])
    assert_equal(mask.data_vars['cell_mask'].values, expected_cells)


def test_apply_clip_mask(tmp_path):
    dataset = make_dataset(width=10, height=11)
    convention = CFGrid1D(dataset)

    # Clip it!
    clip_geometry = Polygon([
        (0.18, .30), (.40, .30), (.60, .51), (.60, .64), (.18, .64), (.18, .30),
    ])
    mask = convention.make_clip_mask(clip_geometry)
    clipped = dataset.ems.apply_clip_mask(mask, tmp_path)

    assert isinstance(clipped.ems, CFGrid1D)

    # Check that the variable and dimension keys were preserved
    assert set(dataset.data_vars.keys()) == set(clipped.data_vars.keys())
    assert set(dataset.coords.keys()) == set(clipped.coords.keys())
    assert set(dataset.dims.keys()) == set(clipped.dims.keys())

    # Check that the new topology seems reasonable
    assert clipped.ems.topology.longitude.size == 5
    assert clipped.ems.topology.latitude.size == 4

    # Check that the data were preserved, beyond being clipped
    def clip_values(values: np.ndarray) -> np.ndarray:
        values = values[..., 3:7, 2:7].copy()
        values[..., 0, -2:] = np.nan
        values[..., 1, -1:] = np.nan
        return values

    assert_equal(clipped.data_vars['botz'].values, clip_values(dataset.data_vars['botz'].values))
    assert_equal(clipped.data_vars['eta'].values, clip_values(dataset.data_vars['eta'].values))
    assert_equal(clipped.data_vars['temp'].values, clip_values(dataset.data_vars['temp'].values))

    # Check that the new geometry matches the relevant polygons in the old geometry
    assert len(clipped.ems.polygons) == 5 * 4
    original_polys = np.concatenate([
        dataset.ems.polygons[(i * 10 + 2):(i * 10 + 7)]
        for i in range(3, 7)
    ], axis=None)
    assert len(clipped.ems.polygons) == len(original_polys)
    for original_poly, clipped_poly in zip(original_polys, clipped.ems.polygons):
        assert original_poly.equals_exact(clipped_poly, 1e-6)


def test_topology():
    dataset = make_dataset(width=10, height=11)
    topology = dataset.ems.topology
    assert isinstance(topology, CFGridTopology)

    assert topology.shape == (11, 10)

    assert topology.longitude_name == 'lon'
    assert topology.latitude_name == 'lat'
    assert topology.longitude.size == 10
    assert topology.latitude.size == 11

    assert_equal(topology.longitude.values, dataset.coords['lon'].values)
    assert_equal(topology.latitude.values, dataset.coords['lat'].values)

    longitude_bounds = topology.longitude_bounds
    assert longitude_bounds.dims == (topology.longitude_name, 'bounds')
    assert_allclose(
        longitude_bounds.values,
        0.1 * np.array([[i - 0.5, i + 0.5] for i in range(10)]))

    latitude_bounds = topology.latitude_bounds
    assert latitude_bounds.dims == (topology.latitude_name, 'bounds')
    assert_allclose(
        latitude_bounds.values,
        0.1 * np.array([[i - 0.5, i + 0.5] for i in range(11)]))
