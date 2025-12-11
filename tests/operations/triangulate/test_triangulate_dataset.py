from collections import defaultdict

import numpy
import shapely
import xarray
from shapely.geometry import Polygon

import emsarray
from emsarray.conventions.arakawa_c import ArakawaCGridKind
from emsarray.conventions.grid import CFGridKind
from emsarray.conventions.ugrid import UGridKind
from emsarray.operations import triangulate


def test_triangulate_dataset_cfgrid1d(datasets):
    dataset = emsarray.open_dataset(datasets / 'cfgrid1d.nc')
    topology = dataset.ems.topology
    triangulation = dataset.ems.make_triangulation()

    # A vertex at the intersection of every cell
    assert len(triangulation.vertices) == (topology.latitude.size + 1) * (topology.longitude.size + 1)

    # Two triangles per cell.
    assert len(triangulation.triangles) == 2 * topology.latitude.size * topology.longitude.size

    assert min(triangulation.face_indexes) == 0
    assert max(triangulation.face_indexes) == topology.latitude.size * topology.longitude.size - 1

    assert triangulation.face_grid_kind is CFGridKind.face
    assert triangulation.vertex_grid_kind is None

    check_triangulation(dataset, triangulation)


def test_triangulate_dataset_cfgrid2d(datasets):
    dataset = emsarray.open_dataset(datasets / "cfgrid2d.nc")
    face_grid = dataset.ems.grids['face']
    triangulation = dataset.ems.make_triangulation()
    topology = dataset.ems.topology

    # There is a hole in one corner, taking out 6 vertices from the expected count
    assert len(triangulation.vertices) == numpy.prod(numpy.array((1, 1)) + topology.shape) - 6

    # Two triangles per polygon, 6 polygons are missing in the corner
    assert len(triangulation.triangles) == 2 * (topology.size - 6)

    # Shoc cells are quadrilaterals, so they each have two triangles
    only_polygons = face_grid.geometry[face_grid.mask]
    assert len(triangulation.triangles) == 2 * len(only_polygons)

    assert triangulation.face_grid_kind is CFGridKind.face
    assert triangulation.vertex_grid_kind is None

    check_triangulation(dataset, triangulation)


def test_triangulate_dataset_shoc_standard(datasets):
    dataset = emsarray.open_dataset(datasets / 'shoc_standard.nc')
    face_grid = dataset.ems.grids['face']
    triangulation = dataset.ems.make_triangulation()

    # There is no good way of calculating the number of vertices, as the
    # geometry is quite complicated in shoc datasets with wet cells

    # Shoc cells are quadrilaterals, so they each have two triangles
    only_polygons = face_grid.geometry[face_grid.mask]
    assert len(triangulation.triangles) == 2 * len(only_polygons)

    assert triangulation.face_grid_kind is ArakawaCGridKind.face
    assert triangulation.vertex_grid_kind is ArakawaCGridKind.node

    check_triangulation(dataset, triangulation)


def test_triangulate_dataset_ugrid(datasets):
    dataset = emsarray.open_dataset(datasets / "ugrid_mesh2d.nc")
    face_grid = dataset.ems.grids['face']
    topology = dataset.ems.topology
    triangulation = dataset.ems.make_triangulation()

    # The vertices should be identical to the ugrid nodes
    assert len(triangulation.vertices) == topology.node_count

    # A polygon with n vertices triangulates in to (n-2) triangles.
    # A polygon is a closed line, so the line has (n+1) coordinates.
    # Therefore, the number of triangles per polygon is len(coords) - 3.
    expected_triangle_count = sum(
        len(polygon.exterior.coords) - 3
        for polygon in face_grid.geometry
    )
    assert len(triangulation.triangles) == expected_triangle_count

    assert triangulation.face_grid_kind is UGridKind.face
    assert triangulation.vertex_grid_kind is UGridKind.node

    check_triangulation(dataset, triangulation)


def check_triangulation(
    dataset: xarray.Dataset,
    triangulation: triangulate.Triangulation,
):
    """
    Check the triangulation of a dataset by reconstructing all polygons.
    These checks are independent of the specific convention.
    """
    vertices = triangulation.vertices
    triangles = triangulation.triangles
    face_indexes = triangulation.face_indexes
    face_grid = dataset.ems.grids[dataset.ems.default_grid_kind]
    # Check that the cell indexes are within bounds.
    assert len(face_indexes) == len(triangles)
    assert min(face_indexes) >= 0
    assert max(face_indexes) <= face_grid.size

    # For each cell in the dataset, reconstruct its polygon from the triangles
    # and check that it matches
    # face_triangle_indexes is a mapping of {face_index: triangles},
    # where triangles is a list of [v1, v2, v3] triangle vertices
    face_triangle_indexes = defaultdict(list)
    for triangle, face_index in zip(triangles, face_indexes):
        face_triangle_indexes[face_index].append(triangle)

    for index, polygon in enumerate(face_grid.geometry):
        if polygon is None:
            assert index not in face_triangle_indexes
            continue

        # Find all relevant triangles...
        assert index in face_triangle_indexes
        face_triangles = face_triangle_indexes[index]

        # Turn them in to polygons...
        reconstructed_polygon = shapely.unary_union(shapely.polygons(vertices[face_triangles]))
        # Check it matches
        assert polygon.equals(reconstructed_polygon)


def test_triangulate_convex_polygon():
    # These coordinates are carefully chosen to produce a polygon that:
    # * is not convex
    # * has three non-sequential vertices in a row
    # * has four non-sequential vertices in a row
    coords = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 2), (1, 1), (1, 3), (0, 1.5)]
    for offset in range(len(coords)):
        polygon = Polygon(coords[offset:] + coords[:offset + 1])
        indexes = list(range(len(coords)))
        indexes = indexes[offset:] + indexes[:offset]
        assert polygon.is_valid
        assert polygon.is_simple

        triangles = triangulate._triangulate_concave_polygon(polygon, indexes)
        print(triangles)
        assert len(triangles) == len(coords) - 2

        triangle_coords = numpy.array([
            [coords[index] for index in tri]
            for tri in triangles
        ])
        union = shapely.union_all(shapely.polygons(triangle_coords))
        assert union.equals(polygon)
