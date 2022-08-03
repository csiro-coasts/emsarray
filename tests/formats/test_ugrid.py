from __future__ import annotations

import json
from typing import Tuple

import geojson
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.figure import Figure
from numpy.testing import assert_allclose, assert_equal
from shapely.geometry import Polygon

from emsarray.formats import get_file_format
from emsarray.formats.ugrid import (
    Mesh2DTopology, NoEdgeDimensionException, UGrid, UGridKind, buffer_faces,
    mask_from_face_indices
)


def make_faces(width: int, height, fill_value: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangle_nodes = sum(range(width + 2))
    square_rows = height
    square_columns = width
    square_nodes = (square_rows + 1) * (square_columns + 1)
    total_nodes = triangle_nodes + square_nodes - (width + 1)

    triangle_faces = width ** 2
    square_faces = square_rows * square_columns
    total_faces = triangle_faces + square_faces

    triangle_edges = 3 * sum(range(width + 1))
    square_edges = 2 * square_rows * square_columns + square_rows + square_columns
    total_edges = triangle_edges + square_edges - width

    face_node = np.ma.masked_array(
        np.full((total_faces, 4), fill_value, dtype=np.int32),
        mask=True, fill_value=fill_value)
    edge_node = np.zeros((total_edges, 2), dtype=np.int32)

    for row in range(1, width + 1):
        # Rows are 1-indexed. Row 1 has one triangle, row 2 has 3 triangles,
        # row n has 2n-1 triangles. All previous rows have a total of (n-1)**2
        # triangles.
        face_index = (row - 1) ** 2

        # Total edges in previous rows
        edge_index = (row * (row - 1)) * 3 // 2

        # The node at the top of the first upwards triangle
        top_left_node = sum(range(row))
        up_tri_shape = np.array([0, row, row + 1]) + top_left_node
        down_tri_shape = np.array([1, 0, row + 1]) + top_left_node

        # There are n upwards triangles on the nth row
        for up_tri in range(row):
            nodes = up_tri_shape + up_tri
            face_node[face_index + up_tri * 2, :3] = nodes
            edge_node[edge_index + up_tri * 3 + 0] = nodes[[0, 1]]
            edge_node[edge_index + up_tri * 3 + 1] = nodes[[1, 2]]
            edge_node[edge_index + up_tri * 3 + 2] = nodes[[2, 0]]

        # There are (n-1) downwards triangles on the nth row
        for down_tri in range(row - 1):
            face_node[face_index + down_tri * 2 + 1, :3] = down_tri_shape + down_tri

    square_node_shape = np.array([0, square_columns + 1, square_columns + 2, 1])
    vertical_edge_shape = np.array([0, square_columns + 1])
    horizontal_edge_shape = np.array([square_columns + 1, square_columns + 2])
    for row in range(square_rows):
        node_row_offset = triangle_nodes - square_columns - 1 + (row * (square_columns + 1))

        face_row_index = triangle_faces + row * square_columns
        edge_row_index = triangle_edges + (square_columns * 2 + 1) * row

        for column in range(square_columns):
            face_node[face_row_index + column] = node_row_offset + column + square_node_shape
            edge_node[edge_row_index + column * 2 + 0] = \
                vertical_edge_shape + node_row_offset + column
            edge_node[edge_row_index + column * 2 + 1] = \
                horizontal_edge_shape + node_row_offset + column
        edge_node[edge_row_index + square_columns * 2] = vertical_edge_shape + node_row_offset + square_columns

    coords = np.full((total_nodes, 2), dtype=np.double, fill_value=np.nan)
    layer_height = np.sin(np.pi / 3)
    for layer in range(0, width + 1):
        # Each layer n has n points evenly distributed around the central axis
        offset = (width - layer) / 2
        points_till_now = sum(range(layer + 1))
        for point in range(layer + 1):
            coords[points_till_now + point, :] = [point + offset, (width - layer) * layer_height]

    for row in range(square_rows):
        angle = -((row + 1) * np.pi) / (4 * square_rows)
        for column in range(square_columns + 1):
            coords[triangle_nodes + row * (square_columns + 1) + column] = [
                np.cos(angle) * (column + square_rows) - square_rows,
                np.sin(angle) * (column + square_rows),
            ]

    return face_node, edge_node, coords


def make_dataset(
    *,
    width: int,
    height: int = 10,
    depth_size: int = 5,
    time_size: int = 4,
    make_edges: bool = True,
    make_face_coordinates: bool = False,
) -> xr.Dataset:
    fill_value = 999999
    face_node_values, edge_node_values, coordinate_values = make_faces(width, height, fill_value=fill_value)

    max_node_dimension = 'nMaxMesh2_face_nodes'
    node_dimension = 'nMesh2_node'
    edge_dimension = 'nMesh2_edge'
    face_dimension = 'nMesh2_face'
    time_dimension = 'record'
    depth_dimension = 'Mesh2_layers'

    cell_size = face_node_values.shape[0]

    node_x = xr.DataArray(
        data=coordinate_values[:, 0],
        dims=[node_dimension],
        name="Mesh2_node_x",
        attrs={
            'standard_name': 'longitude',
            'long_name': "Longitude of 2D mesh nodes",
            'coordinate_type': 'longitude',
            'units': 'degrees_east',
            'projection': 'geographic',
        },
    )
    node_y = xr.DataArray(
        data=coordinate_values[:, 1],
        dims=[node_dimension],
        name="Mesh2_node_y",
        attrs={
            'standard_name': 'latitude',
            'long_name': "Latitude of 2D mesh nodes",
            'coordinate_type': 'latitude',
            'units': 'degrees_north',
            'projection': 'geographic',
        },
    )
    face_node_connectivity = xr.DataArray(
        data=face_node_values,
        dims=[face_dimension, max_node_dimension],
        name="Mesh2_face_nodes",
        attrs={
            "cf_role": "face_node_connectivity",
            "long_name": "Maps every face to its corner nodes.",
            "start_index": 0,
        },
    )

    mesh = xr.DataArray(
        name='Mesh2',
        attrs={
            'cf_role': 'mesh_topology',
            'long_name': "Topology data of 2D unstructured mesh",
            'topology_dimension': 2,
            'node_coordinates': f'{node_x.name} {node_y.name}',
            'face_node_connectivity': face_node_connectivity.name,
            'face_dimension': face_dimension,
        },
    )

    t = xr.DataArray(
        # Note: Using pd.date_range() directly here will lead to strange
        # behaviours, where the `record` dimension becomes a data variable with
        # a datetime64 dtype. Using a list of datetimes instead seems to avoid
        # this, resulting in record simply being a dimension.
        data=list(pd.date_range("2021-11-11", periods=time_size)),
        dims=[time_dimension],
        name='t',
        attrs={
            "long_name": "Time",
            "standard_name": "time",
            "coordinate_type": "time",
        },
    )
    # Note: xarray will reformat this in to 1990-01-01T00:00:00+10:00, which
    # EMS fails to parse. There is no way around this using xarray natively,
    # you have to adjust it with nctool after saving it.
    t.encoding["units"] = "days since 1990-01-01 00:00:00 +10"

    botz = xr.DataArray(
        data=np.random.random(cell_size) * 10 + 50,
        dims=[face_dimension],
        name="Mesh2_depth",
        attrs={
            "units": "metre",
            "long_name": "Z coordinate at sea-bed at cell centre",
            "standard_name": "depth",
            "positive": "down",
            "outside": "9999",
        }
    )
    eta = xr.DataArray(
        data=np.random.normal(0, 0.2, (time_size, cell_size)),
        dims=[time_dimension, face_dimension],
        name="eta",
        attrs={
                "units": "metre",
                "long_name": "Surface elevation",
                "standard_name": "sea_surface_height_above_geoid",
            }
    )
    temp = xr.DataArray(
        data=np.random.normal(12, 0.5, (time_size, depth_size, cell_size)),
        dims=[time_dimension, depth_dimension, face_dimension],
        name="temp",
        attrs={
            "units": "degrees C",
            "long_name": "Temperature",
        },
    )

    dataset = xr.Dataset(
        data_vars={var.name: var for var in [
            mesh, face_node_connectivity, node_x, node_y,
            t, botz, eta, temp
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
    dataset.encoding['unlimited_dims'] = {'record'}
    if make_edges:
        edge_node = xr.DataArray(
            data=edge_node_values,
            dims=[edge_dimension, 'Two'],
            name='Mesh2_edge_nodes',
            attrs={"cf_role": "edge_node_connectivity"},
        )
        edge_size = len(edge_node_values)
        mesh = dataset.data_vars['Mesh2']
        mesh.attrs.update({
            'edge_dimension': edge_dimension,
            'edge_node_connectivity': edge_node.name,
        })

        u1 = xr.DataArray(
            data=np.random.normal(0, 2, (time_size, depth_size, edge_size)),
            dims=[time_dimension, depth_dimension, edge_dimension],
            name='u1',
            attrs={
                'units': 'metre second-1',
                'long_name': "normal component of current at edge",
            },
        )
        dataset.update({value.name: value for value in [edge_node, u1]})

    if make_face_coordinates:
        face_x = xr.DataArray(
            data=[
                np.average(coordinate_values[face_nodes.compressed(), 0])
                for face_nodes in face_node_values
            ],
            dims=[face_dimension],
            name="face_x",
            attrs={
                "long_name": "Characteristic longitude value of a face",
                "start_index": 0,
            },
        )

        face_y = xr.DataArray(
            data=[
                np.average(coordinate_values[face_nodes.compressed(), 1])
                for face_nodes in face_node_values
            ],
            dims=[face_dimension],
            name="face_y",
            attrs={
                "long_name": "Characteristic longitude value of a face",
                "start_index": 0,
            },
        )
        mesh = dataset.data_vars['Mesh2']
        mesh.attrs.update({
            'face_coordinates': f'{face_x.name} {face_y.name}',
        })

        dataset.update({value.name: value for value in [face_x, face_y]})

    return dataset


def test_make_dataset():
    dataset = make_dataset(width=3, depth_size=5)

    # Check that this is recognised as a UGrid dataset
    assert get_file_format(dataset) is UGrid

    # Check that the correct format helper is made
    assert isinstance(dataset.ems, UGrid)

    # Check the coordinate generation worked.
    node_x = dataset.variables["Mesh2_node_x"]
    node_y = dataset.variables["Mesh2_node_y"]
    assert_allclose(node_x[0:3], [1.5, 1, 2])
    assert_allclose(node_x[6:10], [0, 1, 2, 3])
    assert_allclose(node_y[0:1], 3 * np.sin(np.pi / 3))
    assert_allclose(node_y[1:3], 2 * np.sin(np.pi / 3))
    assert_allclose(node_y[6:10], 0)

    # Check the mesh generation worked
    face_node = dataset.variables["Mesh2_face_nodes"]
    assert_equal(face_node.values[0], [0., 1., 2., np.nan])
    assert_equal(face_node.values[1], [1., 3., 4., np.nan])
    assert_equal(face_node.values[2], [2., 1., 4., np.nan])
    assert_equal(face_node.values[3], [2., 4., 5., np.nan])
    assert_equal(face_node.values[9], [6., 10., 11., 7.])
    assert_equal(face_node.values[10], [7., 11., 12., 8.])
    assert_equal(face_node.values[12], [10., 14., 15., 11.])


def test_varnames():
    dataset = make_dataset(width=3)
    assert dataset.ems.get_depth_name() == 'Mesh2_layers'
    assert dataset.ems.get_all_depth_names() == ['Mesh2_layers']
    assert dataset.ems.get_time_name() == 't'


def test_polygons():
    dataset = make_dataset(width=3)
    polygons = dataset.ems.polygons

    # Should be one item for every face
    assert len(polygons) == dataset.dims['nMesh2_face']

    # There should be no empty polygons
    assert all(poly is not None for poly in polygons)
    assert all(dataset.ems.mask)

    topology = dataset.ems.topology

    # Check the coordinates for the generated polygons.
    height = np.sin(np.pi / 3)
    triangle = polygons[0]
    assert triangle.equals_exact(Polygon([(1.5, height * 3), (1, height * 2), (2, height * 2), (1.5, height * 3)]), 1e-6)
    square = polygons[-1]
    node_x = topology.node_x.values
    node_y = topology.node_y.values
    assert square.equals_exact(Polygon([
        (node_x[-6], node_y[-6]), (node_x[-2], node_y[-2]), (node_x[-1], node_y[-1]), (node_x[-5], node_y[-5])
    ]), 1e-6)


def test_face_centres_from_variables():
    dataset = make_dataset(width=3, make_face_coordinates=True)
    helper: UGrid = dataset.ems

    face_centres = helper.face_centres
    lons = dataset['face_x'].values
    lats = dataset['face_y'].values
    for face in range(dataset.dims['nMesh2_face']):
        lon = lons[face]
        lat = lats[face]
        linear_index = helper.ravel_index((UGridKind.face, face))
        np.testing.assert_equal(face_centres[linear_index], [lon, lat])


def test_face_centres_from_centroids():
    dataset = make_dataset(width=3, make_face_coordinates=False)
    helper: UGrid = dataset.ems

    face_centres = helper.face_centres
    for face in range(dataset.dims['nMesh2_face']):
        linear_index = helper.ravel_index((UGridKind.face, face))
        polygon = helper.polygons[linear_index]
        lon, lat = polygon.centroid.coords[0]
        np.testing.assert_equal(face_centres[linear_index], [lon, lat])


@pytest.mark.parametrize(
    ['index', 'selector'],
    (
        [(UGridKind.face, 3), {'nMesh2_face': 3}],
        [(UGridKind.edge, 4), {'nMesh2_edge': 4}],
        [(UGridKind.node, 5), {'nMesh2_node': 5}],
    ),
)
def test_selector_for_index(index, selector):
    dataset = make_dataset(width=4, make_edges=True)
    helper: UGrid = dataset.ems
    assert selector == helper.selector_for_index(index)


def test_make_geojson_geometry():
    dataset = make_dataset(width=3)
    feature_collection = dataset.ems.make_geojson_geometry()
    assert len(feature_collection.features) == len(dataset.ems.polygons)
    out = json.dumps(feature_collection)
    assert isinstance(out, str)


def test_ravel():
    dataset = make_dataset(width=3)
    helper: UGrid = dataset.ems
    for linear_index in range(dataset.dims['nMesh2_face']):
        index = (UGridKind.face, linear_index)
        assert helper.ravel_index(index) == linear_index
        assert helper.unravel_index(linear_index) == index

    for linear_index in range(dataset.ems.topology.edge_count):
        index = (UGridKind.edge, linear_index)
        assert helper.ravel_index(index) == linear_index
        assert helper.unravel_index(linear_index, UGridKind.edge) == index

    for linear_index in range(dataset.ems.topology.node_count):
        index = (UGridKind.node, linear_index)
        assert helper.ravel_index(index) == linear_index
        assert helper.unravel_index(linear_index, UGridKind.node) == index


def test_grid_kinds_with_edges():
    dataset = make_dataset(width=3, make_edges=True)
    helper: UGrid = dataset.ems

    assert helper.grid_kinds == frozenset({
        UGridKind.face,
        UGridKind.edge,
        UGridKind.node,
    })

    assert helper.default_grid_kind == UGridKind.face


def test_grid_kinds_without_edges():
    dataset = make_dataset(width=3, make_edges=False)
    helper: UGrid = dataset.ems

    assert helper.grid_kinds == frozenset({
        UGridKind.face,
        UGridKind.node,
    })

    assert helper.default_grid_kind == UGridKind.face


def test_grid_kind_and_size():
    dataset = make_dataset(width=3, make_edges=True)
    helper: UGrid = dataset.ems

    grid_kind, size = helper.get_grid_kind_and_size(dataset.data_vars['temp'])
    assert grid_kind is UGridKind.face
    assert size == helper.topology.face_count

    grid_kind, size = helper.get_grid_kind_and_size(dataset.data_vars['u1'])
    assert grid_kind is UGridKind.edge
    assert size == helper.topology.edge_count


def test_values():
    dataset = make_dataset(width=3)
    eta = dataset.data_vars["eta"].isel(record=0)
    values = dataset.ems.make_linear(eta)

    # There should be one value per face
    assert len(values) == dataset.ems.topology.face_count

    # The values should be in a specific order
    assert_equal(values, eta.values)


@pytest.mark.matplotlib
def test_plot_on_figure():
    # Not much to test here, mostly that it doesn't throw an error
    dataset = make_dataset(width=3)
    surface_temp = dataset.data_vars["temp"].isel(Mesh2_layers=-1, record=0)

    figure = Figure()
    dataset.ems.plot_on_figure(figure, surface_temp)

    assert len(figure.axes) == 2


@pytest.mark.parametrize(
    ["face_indices", "expected"],
    # It really helps to draw a picture and number the faces to understand
    # what is going on here.
    # There are 25 triangles in a stack, and 50 squares in a grid.
    # The faces are numbered 0-74, row-by-row, left-to-right.
    [
        # Empty input : empty output
        ([], []),
        # One triangle : The surrounding triangles
        ([6], [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]),
        # One triangle : The surrounding triangles and squares
        ([20], [11, 12, 13, 18, 19, 20, 21, 22, 26, 27, 28]),
        # Two neighbouring triangles : All the neighbours
        ([20, 21], [11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 26, 27, 28]),
        # The first triangle doesn't have many neighbours
        ([0], [0, 1, 2, 3]),
        # The last square doesn't either
        ([74], [68, 69, 73, 74]),
        # One of the squares in the middle : All nine surrounding squares
        ([32], [26, 27, 28, 31, 32, 33, 36, 37, 38]),
        # The last row of triangles : The last two rows of triangles plus the
        # first row of squares
        (np.arange(16, 25), np.arange(9, 30)),
        # All the squares : All the squares plus the last row of triangles
        (np.arange(25, 75), np.arange(16, 75)),
        # All faces : All faces
        (np.arange(75), np.arange(75))
    ],
)
def test_buffer_faces(face_indices, expected):
    dataset = make_dataset(width=5)
    topology = Mesh2DTopology(dataset)
    assert_equal(buffer_faces(np.array(face_indices, dtype=int), topology), expected)


def test_mask_from_face_indices_without_edges():
    dataset = make_dataset(width=5, make_edges=False)
    topology = Mesh2DTopology(dataset)

    face_indices = [20, 21, 22, 23, 24, 27, 28, 29]
    node_indices = [12, 13, 14, 17, 18, 19, 20, 23, 24, 25, 26]

    mask = mask_from_face_indices(np.array(face_indices), topology)
    assert mask.dims == {
        'old_node_index': topology.node_count,
        'old_face_index': topology.face_count,
    }

    expected_face = np.full(topology.face_count, fill_value=np.nan)
    expected_face[face_indices] = np.arange(len(face_indices))
    assert_equal(expected_face, mask.data_vars['new_face_index'].values)

    expected_node = np.full(topology.node_count, fill_value=np.nan)
    expected_node[node_indices] = np.arange(len(node_indices))
    assert_equal(expected_node, mask.data_vars['new_node_index'].values)


def test_mask_from_face_indices_with_edges():
    dataset = make_dataset(width=5, make_edges=True)
    topology = Mesh2DTopology(dataset)

    face_indices = [20, 21, 22, 23, 24, 27, 28, 29]
    edge_indices = [25, 28, 36, 37, 38, 39, 40, 41, 42, 43, 44, 49, 50, 51, 52, 53, 54, 55]
    node_indices = [12, 13, 14, 17, 18, 19, 20, 23, 24, 25, 26]

    mask = mask_from_face_indices(np.array(face_indices), topology)
    assert mask.dims == {
        'old_node_index': topology.node_count,
        'old_edge_index': topology.edge_count,
        'old_face_index': topology.face_count,
    }

    expected_face = np.full(topology.face_count, fill_value=np.nan)
    expected_face[face_indices] = np.arange(len(face_indices))
    assert_equal(expected_face, mask.data_vars['new_face_index'].values)

    expected_edge = np.full(topology.edge_count, fill_value=np.nan)
    expected_edge[edge_indices] = np.arange(len(edge_indices))
    assert_equal(expected_edge, mask.data_vars['new_edge_index'].values)

    expected_node = np.full(topology.node_count, fill_value=np.nan)
    expected_node[node_indices] = np.arange(len(node_indices))
    assert_equal(expected_node, mask.data_vars['new_node_index'].values)


def test_apply_clip_mask(tmp_path):
    dataset = make_dataset(width=5)
    topology = Mesh2DTopology(dataset)

    # Sketch this out and number the faces, edges, and nodes, if you want to verify
    face_indices = [20, 21, 22, 23, 24, 27, 28, 29]
    edge_indices = [25, 28, 36, 37, 38, 39, 40, 41, 42, 43, 44, 49, 50, 51, 52, 53, 54, 55]
    node_indices = [12, 13, 14, 17, 18, 19, 20, 23, 24, 25, 26]

    # Clip it!
    mask = mask_from_face_indices(np.array(face_indices), topology)
    clipped = dataset.ems.apply_clip_mask(mask, tmp_path)

    assert isinstance(clipped.ems, UGrid)

    # Check that the variable and dimension keys were preserved
    assert set(dataset.variables.keys()) == set(clipped.variables.keys())
    assert set(dataset.dims.keys()) == set(clipped.dims.keys())

    # Check that the new topology seems reasonable
    assert clipped.ems.topology.face_count == len(face_indices)
    assert clipped.ems.topology.edge_count == len(edge_indices)
    assert clipped.ems.topology.node_count == len(node_indices)

    # Check that the data were preserved, beyond being clipped
    assert_equal(clipped.data_vars['Mesh2_depth'].values, dataset.data_vars['Mesh2_depth'].values[face_indices])
    assert_equal(clipped.data_vars['eta'].values, dataset.data_vars['eta'].values[:, face_indices])
    assert_equal(clipped.data_vars['temp'].values, dataset.data_vars['temp'].values[:, :, face_indices])
    assert_equal(clipped.data_vars['u1'].values, dataset.data_vars['u1'].values[:, :, edge_indices])

    # Check that the new geometry matches the relevant polygons in the old geometry
    assert len(clipped.ems.polygons) == len(face_indices)
    original_polys = [dataset.ems.polygons[index] for index in face_indices]
    for original_poly, clipped_poly in zip(original_polys, clipped.ems.polygons):
        assert original_poly.equals_exact(clipped_poly, 1e-6)


def test_make_and_apply_clip_mask(tmp_path):
    dataset = make_dataset(width=5)
    dataset.ems.to_netcdf(tmp_path / "original.nc")
    with open(tmp_path / 'original.geojson', 'w') as f:
        json.dump(dataset.ems.make_geojson_geometry(), f)

    polygon = Polygon([[3.4, 1], [3.4, -1], [6, -1], [6, 1], [3.4, 1]])
    with open(tmp_path / 'clip.geojson', 'w') as f:
        json.dump(geojson.Feature(geometry=polygon), f)

    topology = dataset.ems.topology

    # Make a clip mask
    clip_mask = dataset.ems.make_clip_mask(polygon, buffer=1)
    clip_mask.to_netcdf(tmp_path / "clip.nc")
    assert clip_mask.dims == {
        'old_face_index': topology.face_count,
        'old_edge_index': topology.edge_count,
        'old_node_index': topology.node_count,
    }

    # Intersecting the clip polygon and the dataset geometry should have
    # selected some faces. Open the geojson files that are generated in qgis
    # and inspect the index attributes. This list is built from that.
    face_indices = [6, 7, 8, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 27, 28, 29, 32, 33, 34]
    fill_value = topology.sensible_fill_value
    new_face_indices = np.ma.masked_array(
        np.full(topology.face_count, fill_value, dtype=np.float64), mask=True)
    new_face_indices[face_indices] = np.arange(len(face_indices))
    assert_equal(clip_mask.data_vars['new_face_index'].values, new_face_indices)

    # Apply the clip mask
    work_dir = tmp_path / 'work_dir'
    work_dir.mkdir()
    clipped = dataset.ems.apply_clip_mask(clip_mask, work_dir)
    clipped.ems.to_netcdf(tmp_path / "clipped.nc")
    with open(tmp_path / 'clipped.geojson', 'w') as f:
        json.dump(clipped.ems.make_geojson_geometry(), f)


def test_derive_connectivity():
    """
    Faces, edges, and nodes are numbered top-to-bottom, left-to-right,
    low-to-high. It helps to sketch out a small grid and number it.
    """
    dataset = make_dataset(width=2, height=2, make_edges=False)
    topology = Mesh2DTopology(dataset)

    # This dataset should start with no extra connectivity variables
    assert not topology.has_valid_edge_node_connectivity
    assert not topology.has_valid_edge_face_connectivity
    assert not topology.has_valid_face_edge_connectivity
    assert not topology.has_valid_face_face_connectivity

    fv = np.ma.masked

    with pytest.raises(NoEdgeDimensionException):
        edge_node = topology.edge_node_array
        # WHoops ok lets fix that...
    dataset.variables['Mesh2'].attrs['edge_dimension'] = 'nMesh2_edge'

    # The actual order of these edges, and the order of the nodes in the pair,
    # is irrelevant to us. They just need to all exist
    edge_node = topology.edge_node_array
    expected_edge_node = [
        # Triangle edges
        (0, 1), (1, 2), (2, 0),
        (1, 3), (3, 4), (4, 1), (2, 4), (4, 5), (5, 2),
        # Square edges
        {3, 6}, (6, 7), (4, 7), (7, 8), (5, 8),
        (6, 9), (9, 10), (7, 10), (10, 11), (8, 11),
    ]
    expected_edge_node = sorted(tuple(sorted(pair)) for pair in expected_edge_node)
    actual_edge_node = sorted(tuple(sorted(pair)) for pair in edge_node.tolist())
    assert expected_edge_node == actual_edge_node

    # Again the actual order here is irrelevant, as long as the edge maps to
    # the correct pair of faces. An edge index names a node pair, so using
    # a node pair as an index gives a stable method of addressing edges.
    edge_face = topology.edge_face_array
    expected_edge_face = {
        (0, 1): (0, fv), (0, 2): (0, fv), (1, 2): (0, 2),
        (1, 3): (1, fv), (3, 4): (1, 4), (1, 4): (1, 2),
        (2, 4): (2, 3), (4, 5): (3, 5), (2, 5): (3, fv),
        (3, 6): (4, fv), (6, 7): (4, 6), (4, 7): (4, 5), (7, 8): (5, 7), (5, 8): (5, fv),
        (6, 9): (6, fv), (9, 10): (6, fv), (7, 10): (6, 7), (10, 11): (7, fv), (8, 11): (7, fv),
    }
    actual_edge_face = {
        tuple(sorted(edge_node[edge_index])): tuple(sorted(face_indices))
        for edge_index, face_indices in enumerate(edge_face)
    }
    assert expected_edge_face == actual_edge_face

    # face_edge is a little tedious. It helps if we build up a mapping of
    # node_pair: edge_index
    face_edge = topology.face_edge_array
    edge_indices = {
        tuple(sorted(node_indices)): edge_index
        for edge_index, node_indices in enumerate(edge_node.tolist())
    }
    expected_face_edge = [
        sorted(edge_indices[pair] for pair in row)
        for row in [
            [(0, 1), (1, 2), (0, 2)],
            [(1, 3), (3, 4), (1, 4)],
            [(1, 2), (1, 4), (2, 4)],
            [(2, 4), (2, 5), (4, 5)],
            [(3, 4), (3, 6), (4, 7), (6, 7)],
            [(4, 5), (4, 7), (5, 8), (7, 8)],
            [(6, 7), (6, 9), (7, 10), (9, 10)],
            [(7, 8), (7, 10), (8, 11), (10, 11)],
        ]
    ]
    actual_face_edge = [sorted(row.compressed()) for row in face_edge]
    assert expected_face_edge == actual_face_edge

    # This one is fairly straight forward at least. This lists which faces
    # border a particular face
    face_face = topology.face_face_array
    expected_face_face = [
        [2],
        [2, 4],
        [0, 1, 3],
        [2, 5],
        [1, 5, 6],
        [3, 4, 7],
        [4, 7],
        [5, 6],
    ]
    actual_face_face = [
        sorted(face_indices.compressed())
        for face_indices in face_face
    ]
    assert expected_face_face == actual_face_face

    assert topology.edge_count == 19
