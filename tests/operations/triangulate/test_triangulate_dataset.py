from collections import defaultdict
from functools import reduce
from typing import List, Tuple

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Polygon

import emsarray
from emsarray.operations.triangulate import (
    _triangulate_polygon, triangulate_dataset
)


def test_triangulate_dataset_cfgrid1d(datasets):
    dataset = emsarray.open_dataset(datasets / 'cfgrid1d.nc')
    topology = dataset.ems.topology
    dataset.ems.polygons
    vertices, triangles, cell_indices = triangulate_dataset(dataset)

    # A vertex at the intersection of every cell
    assert len(vertices) == (topology.latitude.size + 1) * (topology.longitude.size + 1)

    # Two triangles per cell.
    assert len(triangles) == 2 * topology.latitude.size * topology.longitude.size

    assert min(cell_indices) == 0
    assert max(cell_indices) == topology.latitude.size * topology.longitude.size - 1

    check_triangulation(dataset, vertices, triangles, cell_indices)


def test_triangulate_dataset_cfgrid2d(datasets):
    dataset = emsarray.open_dataset(datasets / "cfgrid2d.nc")
    vertices, triangles, cell_indices = triangulate_dataset(dataset)
    topology = dataset.ems.topology

    # There is a hole in one corner, taking out 6 vertices from the expected count
    assert len(vertices) == np.prod(np.array((1, 1)) + topology.shape) - 6

    # Two triangles per polygon, 6 polygons are missing in the corner
    assert len(triangles) == 2 * (topology.size - 6)

    # Shoc cells are quadrilaterals, so they each have two triangles
    only_polygons = dataset.ems.polygons[dataset.ems.mask]
    assert len(triangles) == 2 * len(only_polygons)

    check_triangulation(dataset, vertices, triangles, cell_indices)


def test_triangulate_dataset_shoc_standard(datasets):
    dataset = emsarray.open_dataset(datasets / 'shoc_standard.nc')
    vertices, triangles, cell_indices = triangulate_dataset(dataset)

    # There is no good way of calculating the number of vertices, as the
    # geometry is quite complicated in shoc datasets with wet cells

    # Shoc cells are quadrilaterals, so they each have two triangles
    only_polygons = dataset.ems.polygons[dataset.ems.mask]
    assert len(triangles) == 2 * len(only_polygons)

    check_triangulation(dataset, vertices, triangles, cell_indices)


def test_triangulate_dataset_ugrid(datasets):
    dataset = emsarray.open_dataset(datasets / "ugrid_mesh2d.nc")
    topology = dataset.ems.topology
    vertices, triangles, cell_indices = triangulate_dataset(dataset)

    # The vertices should be identical to the ugrid nodes
    assert len(vertices) == topology.node_count

    # A polygon with n vertices triangulates in to (n-2) triangles.
    # A polygon is a closed line, so the line has (n+1) coordinates.
    # Therefore, the number of triangles per polygon is len(coords) - 3.
    expected_triangle_count = sum(
        len(polygon.exterior.coords) - 3
        for polygon in dataset.ems.polygons
    )
    assert len(triangles) == expected_triangle_count

    check_triangulation(dataset, vertices, triangles, cell_indices)


def check_triangulation(
    dataset: xr.Dataset,
    vertices: List[Tuple[float, float]],
    triangles: List[Tuple[int, int, int]],
    cell_indices: List[int],
):
    """
    Check the triangulation of a dataset by reconstructing all polygons.
    These checks are independent of the specific format type.
    """
    # Check that the cell indices are within bounds.
    assert len(cell_indices) == len(triangles)
    assert min(cell_indices) >= 0
    assert max(cell_indices) <= len(dataset.ems.polygons)

    # For each cell in the dataset, reconstruct its polygon from the triangles
    # and check that it matches
    cell_triangles = defaultdict(list)
    for triangle, cell_index in zip(triangles, cell_indices):
        cell_triangles[cell_index].append(triangle)

    for index, polygon in enumerate(dataset.ems.polygons):
        if polygon is None:
            assert index not in cell_triangles
            continue

        # Find all relevant triangles...
        assert index in cell_triangles
        triangles = cell_triangles[index]

        # Turn them in to polygons...
        polygons = [
            Polygon([vertices[i] for i in triangle])
            for triangle in triangles
        ]
        # Union the those together in to one large polygon
        union = reduce(lambda a, b: a.union(b), polygons)
        # Check it matches
        assert polygon.equals(union)


def test_triangulate_polygon():
    # These coordinates are carefully chosen to produce a polygon that:
    # * is not convex
    # * has three non-sequential vertices in a row
    # * has four non-sequential vertices in a row
    coords = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 2), (1, 1), (1, 3), (0, 1.5)]
    for offset in range(len(coords)):
        polygon = Polygon(coords[offset:] + coords[:offset + 1])
        assert polygon.is_valid
        assert polygon.is_simple

        triangles = _triangulate_polygon(polygon)
        assert len(triangles) == len(coords) - 2

        union = reduce(lambda a, b: a.union(b), triangles)
        assert union.equals(polygon)


@pytest.mark.parametrize("poly", [
    # Polygon with a hole
    Polygon(
        [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)],
        [[(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)]],
    ),
    # Polygon that intersects itself
    Polygon([(0, 0), (1, 0), (0, 1), (1, 1), (0, 0)]),
])
def test_triangulate_polygon_non_simple(poly):
    with pytest.raises(ValueError):
        _triangulate_polygon(poly)
