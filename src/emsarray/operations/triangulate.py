"""
Operations for making a triangular mesh out of the polygons of a dataset.

Examples
--------

Using holoviews_. Try this in an IPython notebook for nice visualisations:

.. code-block:: python

    import emsarray
    import holoviews as hv
    from holoviews import opts
    hv.extension('bokeh')

    # Triangulate the dataset
    dataset = emsarray.tuorial.open_dataset("austen")
    triangulation = dataset.ems.make_triangulation(dataset)

    # This takes a while to render
    mesh = hv.TriMesh((triangulation.triangles, triangulation.vertices))
    mesh

Using trimesh_. This should pop up a new window to display the output:

.. code-block:: python

    import emsarray
    import numpy
    import trimesh

    dataset = emsarray.tutorial.open_dataset("gbr4")
    triangulation = dataset.ems.make_triangulation(dataset)
    # Trimesh expects 3D vertices.
    vertices = numpy.c_[triangulation.vertices, numpy.zeros(len(triangulation.vertices))]
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangulation.triangles)
    mesh.invert()  # Trimesh grids expect the opposite winding order

    depth = 1 - (dataset.data_vars["Mesh2_depth"].values / -200)
    depth_colour = numpy.c_[depth, depth, depth, numpy.ones_like(depth)] * 255
    mesh.visual.face_colors = depth_colour[cell_indexes]
    mesh.show()

.. _holoviews: https://holoviews.org/reference/elements/bokeh/TriMesh.html
.. _trimesh: https://trimsh.org
"""
import dataclasses
from typing import cast

import numpy
import shapely
import xarray
from shapely.geometry import LineString, MultiPoint, Polygon

from emsarray import conventions

type Vertex = tuple[float, float]
type VertexTriangle = tuple[Vertex, Vertex, Vertex]
type IndexTriangle = tuple[int, int, int]


@dataclasses.dataclass(kw_only=True)
class Triangulation[GridKind]:
    vertices: numpy.ndarray
    triangles: numpy.ndarray
    face_indexes: numpy.ndarray

    face_grid_kind: GridKind
    vertex_grid_kind: GridKind | None = None


def triangulate_dataset(
    dataset: xarray.Dataset,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Triangulate the polygon cells of a dataset in a naive way.
    Users should prefer to use :meth:`.Convention.triangulate()`.

    Parameters
    ----------
    dataset
        The dataset to triangulate

    Returns
    -------
    tuple of vertices, triangles, and `cell_indexes`
        A tuple of three numpy arrays is returned,
        containing vertices, triangles, and cell indexes respectively.

        `vertices` is a numpy array of shape (V, 2)
        where V is the number of unique vertices in the dataset.
        The vertex coordinates are in (x, y) or (lon, lat) order.

        `triangles` is a numpy array of shape (T, 3)
        where T is the number of triangles in the dataset.
        Each triangle is a set of three vertex indexes.

        `cell_indexes` is a numpy list of length T.
        Each entry indicates which polygon from the dataset a triangle is a part of.
    """
    convention = cast(conventions.Convention, dataset.ems)
    triangulation = convention.make_triangulation()
    return (triangulation.vertices, triangulation.triangles, triangulation.face_indexes)


def find_unique_vertices(
    polygons: numpy.ndarray,
) -> numpy.ndarray:
    """
    Find all the unique coordinates in a collection of polygons.

    Parameters
    ----------
    polygons : numpy.ndarray of shapely.Polygon
        A numpy array of polygons.

    Returns
    -------
    numpy.ndarray of shapely.Point
        An array of all of the unique vertex coordinates in all the polygons,
        in some arbitrary order.
    """
    # Find all the unique coordinates and assign them each a unique index
    # TODO Benchmark this method of finding the unique coord pairs.
    all_coords = shapely.get_coordinates(polygons)
    unique_coords = set(tuple(coord) for coord in all_coords)
    return cast(numpy.ndarray, shapely.points(list(unique_coords)))


def polygons_to_vertex_indexes(
    polygons: numpy.ndarray,
    vertices: numpy.ndarray,
) -> numpy.ndarray:
    """
    For a set of polygons and a set of all vertices,
    find the index into the vertex array of each vertex of the polygons.

    Parameters
    ----------
    polygons : numpy.ndarray of shapely.Polygon
        A numpy array of polygons.
    vertices : numpy.ndarray of shapely.Point
        A numpy array of all vertices of these polygons.

    Returns
    -------
    numpy.ndarray
        A numpy.ndarray of shape (# polygons, max node count),
        where `# polygons` is the length of the polygons array,
        and `max node count` is the node count of the largest polygon.
        For each vertext of each polygon, this specifies the index of that
        vertex coordinate in the vertices array.
    """
    max_node_count = shapely.get_num_coordinates(polygons).max()
    polygon_vertex_indexes = numpy.full(shape=(len(polygons), max_node_count), fill_value=-1)
    node_indexes = {p.coords[0]: i for i, p in enumerate(vertices)}

    for polygon_index in range(len(polygons)):
        polygon = polygons[polygon_index]
        if polygon is None:
            continue

        num_coords = shapely.get_num_coordinates(polygon) - 1
        polygon_vertex_indexes[polygon_index, :num_coords] = [
            node_indexes[coord]
            for coord in polygon.exterior.coords[:-1]
        ]

    return polygon_vertex_indexes


def triangulate(
    vertices: numpy.ndarray,
    polygons: numpy.ndarray,
    polygon_vertex_indexes: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Triangulate a set of polygons.

    Parameters
    ----------
    vertices : numpy.ndarray of shapely.Point
        All the unique vertices of all the polygons
    polygons : numpy.ndarray of shapely.Polygon
        All the polygons to triangulate
    polygon_vertex_indexes : numpy.ndarray
        An array of arrays (or a two dimensional array)
        mapping vertices for each polygon
        to the correct index in the vertices array.

    Returns
    -------
    tuple of vertices, triangles, and `cell_indexes`
        A tuple of three numpy arrays is returned,
        containing vertices, triangles, and cell indexes respectively.

        `vertices` is a numpy array of shape (V, 2)
        where V is the number of unique vertices in the dataset.
        The vertex coordinates are in (x, y) or (lon, lat) order.

        `triangles` is a numpy array of shape (T, 3)
        where T is the number of triangles in the dataset.
        Each triangle is a set of three vertex indexes.

        `cell_indexes` is a numpy list of length T.
        Each entry is the index of the polygon in the polygons array the triangle is a part of.

    """

    polygon_length = shapely.get_num_coordinates(polygons)
    polygon_length[polygon_length != 0] -= 1

    # Count the total number of triangles.
    # A polygon with n sides can be decomposed in to n-2 triangles,
    # however polygon_length counts an extra vertex because of the closed rings.
    total_triangles = numpy.sum(polygon_length[numpy.nonzero(polygon_length)] - 2)

    # Pre-allocate some numpy arrays that will store the triangle data.
    # This is much faster than building up these data structures iteratively.
    polygon_indexes = numpy.empty(total_triangles, dtype=int)
    triangle_vertices = numpy.empty((total_triangles, 3), dtype=int)
    # This is the index of which face we will populate next in the above arrays
    current_triangle = 0

    def _add_triangles(polygon_index: int, vertex_triangles: numpy.ndarray) -> None:
        """
        Append some triangles to the polygon_indexes and triangle_vertices arrays.

        Parameters
        ----------
        polygon_index : int
            The polygon index of all the triangles
        vertex_triangles : numpy.ndarray
            The vertex indexes for the triangles of this face as an (n, 3) array,
            where n is the number of triangles in the face.
        """
        nonlocal current_triangle
        current_length = len(vertex_triangles)
        polygon_indexes[current_triangle:current_triangle + current_length] = polygon_index
        triangle_vertices[current_triangle:current_triangle + current_length] = vertex_triangles
        current_triangle += current_length

    # Find all concave polygons by comparing each polygon to its convex hull.
    # A convex polygon is its own convex hull,
    # while the convex hull of a concave polygon
    # will always have fewer vertices than the original polygon.
    # Comparing the number of vertices is a shortcut.
    convex_hulls = shapely.convex_hull(polygons)
    convex_hull_length = shapely.get_num_coordinates(convex_hulls)
    convex_hull_length[convex_hull_length != 0] -= 1
    polygon_is_concave = numpy.flatnonzero(convex_hull_length != polygon_length)

    # Categorize each polygon by length, skipping concave polygons.
    # We will handle them separately.
    polygon_length[polygon_is_concave] = 0
    unique_lengths = numpy.unique(polygon_length)

    # Triangulate polygons in batches of identical sizes.
    # This allows the coordinates to be manipulated efficiently.
    for unique_length in unique_lengths:
        if unique_length == 0:
            # Any `None` polygons will have a length of 0,
            # and any concave polygons have been set to 0.
            continue

        # Because these triangles are convex and because we only care about the
        # vertex indexes, we can maniuplate the vertex indexes to make a fan
        # triangulation easily.
        same_length_polygon_indexes = numpy.flatnonzero(polygon_length == unique_length)

        if unique_length == 3:
            for polygon_index in same_length_polygon_indexes:
                _add_triangles(
                    polygon_index,
                    polygon_vertex_indexes[polygon_index:polygon_index + 1, :3])
        else:
            vertex_triangles = _triangulate_polygons_by_length(
                polygon_vertex_indexes[same_length_polygon_indexes, :unique_length])
            for polygon_index, triangles in zip(same_length_polygon_indexes, vertex_triangles):
                _add_triangles(int(polygon_index), triangles)

    # Triangulate each concave polygon using a slower manual method.
    # Anecdotally concave polygons are very rare,
    # so using a slower method isn't an issue.
    for polygon_index in polygon_is_concave:
        polygon = polygons[polygon_index]
        num_coordinates = shapely.get_num_coordinates(polygon) - 1
        triangles = _triangulate_concave_polygon(
            polygon,
            polygon_vertex_indexes[polygon_index, :num_coordinates])
        _add_triangles(int(polygon_index), triangles)

    # Check that we have handled each triangle we expected.
    assert current_triangle == total_triangles

    not_empty_vertices = numpy.flatnonzero(vertices != None)  # noqa: E711
    vertex_coords = numpy.full(shape=(len(vertices), 2), fill_value=numpy.nan)
    vertex_coords[not_empty_vertices] = shapely.get_coordinates(vertices)

    return vertex_coords, triangle_vertices, polygon_indexes


def _triangulate_polygons_by_length(
    polygon_vertex_indexes: numpy.ndarray,
) -> numpy.ndarray:
    """
    Triangulate a list of convex polygons of equal length.

    Parameters
    ----------
    polygons : numpy.ndarray of shapely.Polygon
        The polygons to triangulate.
        These must all have the same number of vertices
        and must all be convex.

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (# polygons, # triangles, 3),
        where `# polygons` is the length of `polygons`
        and `# triangles` is the number of triangles each polygon is decomposed in to.
    """

    # for a single polygon_vertex_indexes
    # [
    #     [pvi[0], pvi[n], pvi[n+1]]
    #     for n in range(vertex_count)
    # ]
    vertex_count = polygon_vertex_indexes.shape[1]
    triangle_count = vertex_count - 2
    v0 = polygon_vertex_indexes[:, :1].repeat(triangle_count, axis=1)
    v1 = polygon_vertex_indexes[:, 1:-1]
    v2 = polygon_vertex_indexes[:, 2:]
    triangles = numpy.stack([v0, v1, v2], axis=2)
    return triangles


def _triangulate_concave_polygon(
    polygon: Polygon,
    polygon_vertex_indexes: numpy.ndarray,
) -> numpy.ndarray:
    """
    Triangulate a single convex polygon.

    Parameters
    ----------
    polygon : shapely.Polygon
        The polygon to triangulate.

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (# triangles, 3, 2),
        where `# triangles` is the number of triangles the polygon is decomposed in to.
    """
    # This is the 'ear clipping' method of polygon triangulation.
    # In any simple polygon, there is guaranteed to be at least two 'ears'
    # - three neighbouring vertices whos diagonal is inside the polygon.
    # An ear can be clipped off the polygon, giving one triangle and a new,
    # smaller polygon. Repeat till the polygon is a triangle.
    #
    # This algorithm is not the best if given particularly bad polygons,
    # but for small convex polygons it is approximately linear in time.
    # Most polygons will be either squares, convex quadrilaterals, or convex
    # polygons.

    # A triangle with n vertices will have n - 2 triangles.
    # Because the exterior is a closed loop, we need to subtract 3.
    triangle_count = len(polygon.exterior.coords) - 3
    triangles = numpy.empty((triangle_count, 3), dtype=int)
    triangle_index = 0

    # Note that shapely polygons with n vertices will be closed, and thus have
    # n+1 coordinates. We trim that superfluous coordinate off in the next line
    while len(polygon.exterior.coords) > 4:
        exterior = polygon.exterior
        coords = exterior.coords[:-1]
        # We try and make an ear from vertices (i, i+1, i+2)
        for i in range(len(coords) - 2):
            # If the diagonal between i and i+2 is within the larger polygon,
            # then this entire triangle is within the larger polygon
            vertices = [coords[i], coords[i + 2]]
            multipoint = MultiPoint(vertices)
            diagonal = LineString(vertices)
            if (
                diagonal.covered_by(polygon)
                # The intersection of the diagonal with the boundary should be two
                # points - the vertices in question. If three or more points are in
                # a line, the intersection will be something different. Removing
                # that ear will result in two disconnected polygons.
                and exterior.intersection(diagonal).equals(multipoint)
            ):
                triangles[triangle_index] = polygon_vertex_indexes[i:i + 3]
                triangle_index += 1
                polygon = Polygon(coords[:i + 1] + coords[i + 2:])
                polygon_vertex_indexes = numpy.delete(polygon_vertex_indexes, i + 1)
                break
        else:
            # According to the twos ear theorem, we should never reach this
            raise ValueError(
                f"Could not find interior diagonal for polygon! {polygon.wkt}")

    # The trimmed polygon is now a triangle. Add it to the list and we are done!
    triangles[triangle_index] = polygon_vertex_indexes
    assert (triangle_index + 1) == triangle_count

    return triangles
