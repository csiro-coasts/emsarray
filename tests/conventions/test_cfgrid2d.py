"""
Test the CFGrid2D convention implementation

The SHOC simple convention is a specific implementation of CFGrid2D.
Instead of writing two identical test suites,
the SHOC simple convention is used to test both.
"""
from __future__ import annotations

import itertools
import json
import pathlib
from typing import Type

import numpy as np
import pandas as pd
import pytest
import xarray
from matplotlib.figure import Figure
from numpy.testing import assert_allclose
from shapely.geometry import Polygon
from shapely.testing import assert_geometries_equal

from emsarray.conventions import get_dataset_convention
from emsarray.conventions.grid import CFGrid2DTopology, CFGridKind
from emsarray.conventions.shoc import ShocSimple
from emsarray.operations import geometry
from tests.utils import (
    DiagonalShocGrid, ShocGridGenerator, ShocLayerGenerator,
    assert_property_not_cached
)


def make_dataset(
    *,
    k_size: int = 5,
    j_size: int,
    i_size: int,
    time_size: int = 4,
    grid_type: Type[ShocGridGenerator] = DiagonalShocGrid,
    corner_size: int = 0,
    include_bounds: bool = False,
) -> xarray.Dataset:
    """
    Make a dummy SHOC simple dataset of a particular size.
    It will have a sheared grid of points located near (0, 0),
    with increasing j moving north east, and increasing i moving south east.

    For a grid with j_size=5, i_size=9, the centre longitude-latitude will be:

    * index (0, 0): (0.1, 0.9),
    * index (0, 1): (0.2, 0.8),
    * index (0, 8): (0.9, 0.1)
    * index (1, 0): (0.2, 0.10)
    * index (1, 8): (0.10, 0.2)
    * index (4, 0): (0.5, 0.13)

    Three data variables will be made: botz, eta, and temp.
    Each will be filled with random data.
    There will be a one-cell border around the edge of the dataset where all
    data variables will be nans.

    The (i=0, j=j_size) corner will not have any coordinates in a 4x4 box.
    The (i=i_size, j=j_size) corner will have coordinates,
    but data variables will be masked off
    """
    coordinate_centre_mask = np.full((j_size, i_size), True)
    # Cut a chunk out of the corner where the coordinates will not be defined.
    if corner_size > 0:
        coordinate_centre_mask[-(corner_size):, :+(corner_size)] = False

    wet_centre_mask = np.full((j_size, i_size), True)
    if corner_size > 0:
        wet_centre_mask[-corner_size:, :+corner_size] = False
        wet_centre_mask[-corner_size:, -corner_size:] = False
    wet_centre_mask[:+1, :] = False
    wet_centre_mask[-1:, :] = False
    wet_centre_mask[:, :+1] = False
    wet_centre_mask[:, -1:] = False
    wet_mask = xarray.DataArray(data=wet_centre_mask, dims=["j", "i"])

    grid = grid_type(
        j=j_size, i=i_size,
        face_mask=coordinate_centre_mask,
        include_bounds=include_bounds)
    layers = ShocLayerGenerator(k=k_size)

    time = xarray.DataArray(
        data=pd.date_range("2021-11-11", periods=time_size),
        dims=["time"],
        attrs={
            "long_name": "Time",
            "standard_name": "time",
            "coordinate_type": "time",
        },
    )

    botz = xarray.DataArray(
        data=np.random.random((j_size, i_size)) * 10 + 50,
        dims=["j", "i"],
        attrs={
            "units": "metre",
            "long_name": "Depth of sea-bed",
            "standard_name": "depth",
            "positive": "down",
            "outside": "9999",
            "missing_value": -99.,
        }
    ).where(wet_mask)
    botz.values[1, 1] = -99.

    eta = xarray.DataArray(
        data=np.random.normal(0, 0.2, (time_size, j_size, i_size)),
        dims=["time", "j", "i"],
        attrs={
            "units": "metre",
            "long_name": "Surface elevation",
            "standard_name": "sea_surface_height_above_geoid",
        }
    ).where(wet_mask)
    temp = xarray.DataArray(
        data=np.random.normal(12, 0.5, (time_size, k_size, j_size, i_size)),
        dims=["time", "k", "j", "i"],
        attrs={
            "units": "degrees C",
            "long_name": "Temperature",
        },
    ).where(wet_mask)

    return xarray.Dataset(
        data_vars={"botz": botz, "eta": eta, "temp": temp, **grid.simple_vars},
        coords={**layers.simple_coords, **grid.simple_coords, "time": time},
        attrs={
            "title": "Example SHOC dataset",
            "ems_version": "v1.2.3 fake",
            "Conventions": "CF-1.0",
        },
    )


def test_make_dataset():
    dataset = make_dataset(j_size=5, i_size=9)

    # Check that this is recognised as a ShocSimple dataset
    assert get_dataset_convention(dataset) is ShocSimple

    # Check that the correct convention is used
    assert isinstance(dataset.ems, ShocSimple)

    # Check the coordinate generation worked.
    longitudes = dataset.coords["longitude"]
    latitudes = dataset.coords["latitude"]
    assert longitudes[0, 0] == pytest.approx(0.1)
    assert longitudes[0, 1] == pytest.approx(0.2)
    assert longitudes[0, 8] == pytest.approx(0.9)
    assert longitudes[1, 0] == pytest.approx(0.2)
    assert latitudes[0, 0] == pytest.approx(0.9)
    assert latitudes[0, 1] == pytest.approx(0.8)
    assert latitudes[0, 8] == pytest.approx(0.1)
    assert latitudes[1, 0] == pytest.approx(1.0)


def test_varnames():
    dataset = make_dataset(j_size=10, i_size=10)
    assert dataset.ems.get_depth_name() == 'zc'
    assert dataset.ems.get_all_depth_names() == ['zc']
    assert dataset.ems.get_time_name() == 'time'


@pytest.mark.parametrize(
    ['name', 'attrs'],
    [
        ('lat', {'units': 'degrees_north'}),
        ('l1', {'units': 'degree_north'}),
        ('latitude', {'units': 'degree_N'}),
        ('y', {'units': 'degrees_N'}),
        ('Latitude', {'units': 'degreeN'}),
        ('lats', {'units': 'degreesN'}),
        ('latitude', {'standard_name': 'latitude'}),
        ('y', {'axis': 'Y'}),
    ],
)
def test_latitude_detection(name: str, attrs: dict):
    dataset = xarray.Dataset({
        name: xarray.DataArray([[0, 1], [2, 3]], dims=['j', 'i'], attrs=attrs),
        'dummy': xarray.DataArray([3, 4, 5], dims=['other']),
    })
    topology = CFGrid2DTopology(dataset)
    assert topology.latitude_name == name


@pytest.mark.parametrize(
    ['name', 'attrs'],
    [
        ('lon', {'units': 'degrees_east'}),
        ('l2', {'units': 'degree_east'}),
        ('longitude', {'units': 'degree_E'}),
        ('x', {'units': 'degrees_E'}),
        ('Longitude', {'units': 'degreeE'}),
        ('lons', {'units': 'degreesE'}),
        ('longitude', {'standard_name': 'longitude'}),
        ('x', {'axis': 'X'}),
    ],
)
def test_longitude_detection(name: str, attrs: dict):
    dataset = xarray.Dataset({
        name: xarray.DataArray([[0, 1], [2, 3]], dims=['j', 'i'], attrs=attrs),
        'dummy': xarray.DataArray([3, 4, 5], dims=['other']),
    })
    topology = CFGrid2DTopology(dataset)
    assert topology.longitude_name == name


def test_manual_coordinate_names():
    dataset = xarray.Dataset({
        'n': xarray.DataArray([[0, 1], [2, 3]], dims=['j', 'i']),
        'e': xarray.DataArray([[4, 5], [6, 7]], dims=['j', 'i']),
    })
    topology = CFGrid2DTopology(dataset)
    with pytest.raises(ValueError):
        topology.latitude_name

    topology = CFGrid2DTopology(dataset, latitude='n', longitude='e')
    assert topology.latitude_name == 'n'
    assert topology.longitude_name == 'e'
    xarray.testing.assert_equal(topology.latitude, dataset['n'])
    xarray.testing.assert_equal(topology.longitude, dataset['e'])


def test_polygons_no_bounds():
    dataset = make_dataset(
        j_size=10, i_size=20,
        include_bounds=False, corner_size=3)
    polygons = dataset.ems.polygons
    print(type(dataset.ems))

    # Should be one item for every cell in the shape
    assert len(polygons) == 10 * 20

    assert_geometries_equal(
        polygons[0],
        Polygon([[0.1, 2], [0.15, 1.95], [0.2, 2], [0.15, 2.05]]))

    assert_geometries_equal(
        polygons[21],
        Polygon([[0.2, 2], [0.3, 1.9], [0.4, 2], [0.3, 2.1]]))

    assert polygons[20 * 9 + 0] is None
    assert polygons[20 * 9 + 1] is None
    assert polygons[20 * 9 + 2] is None
    assert polygons[20 * 9 + 3] is not None


def test_polygons_with_bounds():
    dataset = make_dataset(
        j_size=10, i_size=20,
        include_bounds=True, corner_size=3)
    polygons = dataset.ems.polygons

    # Should be one item for every cell in the shape
    assert len(polygons) == 10 * 20

    assert_geometries_equal(
        polygons[0],
        Polygon([[0, 2], [0.1, 1.9], [0.2, 2], [0.1, 2.1]]))

    assert_geometries_equal(
        polygons[21],
        Polygon([[0.2, 2], [0.3, 1.9], [0.4, 2], [0.3, 2.1]]))

    assert polygons[20 * 9 + 0] is None
    assert polygons[20 * 9 + 1] is None
    assert polygons[20 * 9 + 2] is None
    assert polygons[20 * 9 + 3] is not None


def test_holes():
    dataset = make_dataset(j_size=10, i_size=20, corner_size=5)
    only_polygons = dataset.ems.polygons[dataset.ems.mask]

    # The grid is 10*20 minus a 5x5 box removed from a corner
    assert len(only_polygons) == 10 * 20 - (5 * 5)

    # Check the coordinates for the generated polygons. The vertices for
    # a polygon are calculated by averaging the surrounding cell centres.
    # The edges and corners are truncated, as they do not have all of their surrounding cells.

    # The corner polygon has no neighbours on two sides, so is a quarter of the size
    poly = only_polygons[0]
    assert poly.equals_exact(Polygon([(0.1, 2.0), (0.15, 1.95), (0.2, 2.0), (0.15, 2.05), (0.1, 2.0)]), 1e-6)

    # An edge polygon has no neighbour on one side, so is half of the size
    poly = only_polygons[2]
    assert poly.equals_exact(Polygon([(0.25, 1.85), (0.35, 1.75), (0.4, 1.8), (0.3, 1.9), (0.25, 1.85)]), 1e-6)

    # A polygon surrounded by all neighbours is full sized
    poly = only_polygons[22]
    assert poly.equals_exact(Polygon([(0.3, 1.9), (0.4, 1.8), (0.5, 1.9), (0.4, 2.0), (0.3, 1.9)]), 1e-6)


def test_bounds():
    dataset = make_dataset(
        j_size=10, i_size=20, corner_size=5, include_bounds=True)
    # The corner cuts out some of the upper bounds
    assert_allclose(dataset.ems.bounds, (0, 0, 3, 2.5))
    assert_property_not_cached(dataset.ems, 'geometry')


def test_face_centres():
    # SHOC simple face centres are taken directly from the coordinates,
    # not calculated from polygon centres.
    dataset = make_dataset(j_size=10, i_size=20, corner_size=3)
    convention: ShocSimple = dataset.ems

    face_centres = convention.face_centres
    lons = dataset.variables['longitude'].values
    lats = dataset.variables['latitude'].values
    for j in range(dataset.dims['j']):
        for i in range(dataset.dims['i']):
            lon = lons[j, i]
            lat = lats[j, i]
            linear_index = convention.ravel_index((j, i))
            np.testing.assert_equal(face_centres[linear_index], [lon, lat])


def test_selector_for_index():
    dataset = make_dataset(j_size=5, i_size=7)
    convention: ShocSimple = dataset.ems

    # Shoc simple only has a single face grid
    index = (3, 4)
    selector = {'j': 3, 'i': 4}
    assert selector == convention.selector_for_index(index)


def test_make_geojson_geometry():
    dataset = make_dataset(j_size=10, i_size=10, corner_size=3)
    out = json.dumps(geometry.to_geojson(dataset))
    assert isinstance(out, str)


def test_ravel():
    dataset = make_dataset(j_size=5, i_size=7)
    convention: ShocSimple = dataset.ems

    for linear_index, (j, i) in enumerate(itertools.product(range(5), range(7))):
        index = (j, i)
        assert convention.ravel_index(index) == linear_index
        assert convention.unravel_index(linear_index) == index
        assert convention.unravel_index(linear_index, CFGridKind.face) == index


def test_grid_kinds():
    dataset = make_dataset(j_size=3, i_size=5)
    convention: ShocSimple = dataset.ems

    assert convention.grid_kinds == frozenset({CFGridKind.face})
    assert convention.default_grid_kind == CFGridKind.face


def test_grid_kind_and_size():
    dataset = make_dataset(j_size=5, i_size=7)
    convention: ShocSimple = dataset.ems

    grid_kind, size = convention.get_grid_kind_and_size(dataset.data_vars['temp'])
    assert grid_kind is CFGridKind.face
    assert size == 5 * 7


def test_drop_geometry(datasets: pathlib.Path):
    dataset = xarray.open_dataset(datasets / 'cfgrid2d.nc')

    dropped = dataset.ems.drop_geometry()
    assert dropped.dims.keys() == {'i', 'j'}

    topology = dataset.ems.topology
    assert topology.longitude_name in dataset.variables
    assert topology.longitude_name in dataset.variables
    assert topology.longitude_name not in dropped.variables
    assert topology.longitude_name not in dropped.variables


def test_values():
    dataset = make_dataset(j_size=10, i_size=20, corner_size=5)
    eta = dataset.data_vars["eta"].isel(time=0)
    values = dataset.ems.make_linear(eta)

    # There should be one value per cell polygon
    assert len(values) == len(dataset.ems.polygons)

    # The values should be in a specific order
    assert np.allclose(values, eta.values.ravel(), equal_nan=True)


@pytest.mark.matplotlib
def test_plot_on_figure():
    # Not much to test here, mostly that it doesn't throw an error
    dataset = make_dataset(j_size=10, i_size=20)
    surface_temp = dataset.data_vars["temp"].isel(k=-1, time=0)

    figure = Figure()
    dataset.ems.plot_on_figure(figure, surface_temp)

    assert len(figure.axes) == 2
