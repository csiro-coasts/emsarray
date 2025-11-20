import re

import numpy
import pytest
import xarray
from shapely import Polygon

from emsarray.exceptions import InvalidGeometryWarning
from emsarray.conventions.ugrid import UGridKind


def make_ugrid_from_polygons(polygons: list[Polygon]) -> xarray.Dataset:
    unique_vertices = sorted(set(
        vertex
        for polygon in polygons
        for vertex in polygon.exterior.coords
    ))
    node_indexes = {v: i for i, v in enumerate(unique_vertices)}
    coordinate_values = numpy.array(unique_vertices)
    max_nodes = max([len(polygon.exterior.coords) for polygon in polygons])

    face_node_values = numpy.full((len(polygons), max_nodes), fill_value=numpy.nan)
    for face_index, polygon in enumerate(polygons):
        face_node_values[face_index, 0:len(polygon.exterior.coords)] = [
            node_indexes[v] for v in polygon.exterior.coords]

    node_dimension = 'node'
    face_dimension = 'face'
    max_node_dimension = 'max_node'

    node_x = xarray.DataArray(
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
    node_y = xarray.DataArray(
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
    face_node_connectivity = xarray.DataArray(
        data=face_node_values,
        dims=[face_dimension, max_node_dimension],
        name="Mesh2_face_nodes",
        attrs={
            "cf_role": "face_node_connectivity",
            "long_name": "Maps every face to its corner nodes.",
            "start_index": 0,
        },
    )

    mesh = xarray.DataArray(
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

    dataset = xarray.Dataset(
        data_vars={var.name: var for var in [
            mesh, face_node_connectivity, node_x, node_y,
        ]},
        attrs={
            'Conventions': "UGRID-1.0",
            'start_index': 0,
        },
    )

    return dataset


def make_ugrid_with_bad_polygons() -> xarray.Dataset:
    # A unit square at the origin
    valid_polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    # Bowtie shaped invalid polygon that meets at a point
    bowtie_1 = Polygon([(10, 0), (10, 2), (11, 1), (12, 2), (12, 0), (11, 1), (10, 0)])
    # Alternate bowtie polygon that self intersects
    bowtie_2 = Polygon([(20, 0), (21, 1), (20, 1), (21, 0), (20, 0)])

    assert valid_polygon.is_valid
    assert not bowtie_1.is_valid
    assert not bowtie_2.is_valid

    return make_ugrid_from_polygons([valid_polygon, bowtie_1, bowtie_2])


def test_bad_polygons():
    dataset = make_ugrid_with_bad_polygons()
    grid = dataset.ems.grids['face']
    with pytest.warns(InvalidGeometryWarning, match=re.escape('Dropping invalid UGridKind.face geometry at indices [1 2]')):
        grid.geometry

    assert grid.geometry[0] is not None
    assert grid.geometry[1] is None
    assert grid.geometry[2] is None

    for i, polygon in enumerate(dataset.ems._make_geometry(UGridKind.face)):
        assert polygon is not None
        if polygon.is_valid:
            assert grid.geometry[i] is not None
        else:
            assert grid.geometry[i] is None
