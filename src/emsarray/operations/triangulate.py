"""
Operations for making a triangular mesh out of the polygons of a dataset.
"""
from typing import List, Tuple, cast

import xarray
from shapely.geometry import LineString, MultiPoint, Polygon

Vertex = Tuple[float, float]
Triangle = Tuple[int, int, int]


def triangulate_dataset(
    dataset: xarray.Dataset,
) -> Tuple[List[Vertex], List[Triangle], List[int]]:
    """
    Triangulate the polygon cells of a dataset

    A mesh can be constructed from this triangulation,
    for example using `Holoviews TriMesh <holoviews_>`_
    or `trimesh.Trimesh <trimesh_>`_.

    Parameters
    ----------
    dataset
        The dataset to triangulate

    Returns
    -------
    tuple of vertices, triangles, and cell indices.
        A tuple of three lists is returned,
        containing vertices, triangles, and cell indices respectively.

        Each vertex is a tuple of (x, y) or (lon, lat) coordinates.

        Each triangle is a tuple of three integers,
        indicating which vertices make up the triangle.

        The cell indices tie the triangles to the original cell polygon,
        allowing you to plot data on the triangle mesh.

    Examples
    --------

    Using holoviews_. Try this in an IPython notebook for nice visualisations:

    .. code-block:: python

        import emsarray
        import holoviews as hv
        from emsarray.operations import triangulate_dataset
        from holoviews import opts
        hv.extension('bokeh')

        # Triangulate the dataset
        dataset = emsarray.tuorial.open_dataset("austen")
        vertices, triangles, cell_indices = triangulate_dataset(dataset)

        # This takes a while to render
        mesh = hv.TriMesh((triangles, vertices))
        mesh

    Using trimesh_. This should pop up a new window to display the output:

    .. code-block:: python

        import emsarray
        import numpy
        import trimesh
        from emsarray.operations import triangulate_dataset

        dataset = emsarray.tutorial.open_dataset("gbr4")
        vertices, triangles, cell_indices = triangulate_dataset(dataset)
        # Trimesh expects 3D vertices.
        vertices = numpy.c_[vertices, numpy.zeros(len(vertices))]
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.invert()  # Trimesh grids expect the opposite winding order

        depth = 1 - (dataset.data_vars["Mesh2_depth"].values / -200)
        depth_colour = numpy.c_[depth, depth, depth, numpy.ones_like(depth)] * 255
        mesh.visual.face_colors = depth_colour[cell_indices]
        mesh.show()

    .. _holoviews: https://holoviews.org/reference/elements/bokeh/TriMesh.html
    .. _trimesh: https://trimsh.org
    """
    polygons = dataset.ems.polygons

    # Getting all the vertices is easy - extract them from the polygons.
    # By going through a set, this will deduplicate the vertices.
    # Back to a list and we have a stable order
    vertices: List[Vertex] = list({
        vertex
        for polygon in polygons
        if polygon is not None
        for vertex in polygon.exterior.coords
    })

    # This maps between a vertex tuple and its index.
    # Vertex positions are (probably) floats. For grid datasets, where cells
    # are implicitly defined by their centres, be careful to compute cell
    # vertices in consistent ways. Float equality is tricky!
    vertex_indices = {vertex: index for index, vertex in enumerate(vertices)}

    # Each cell polygon needs to be triangulated,
    # while also recording the convention native index of the cell,
    # so that we can later correlate cell data with the triangles.
    polygons_with_index = [
        (polygon, index)
        for index, polygon in enumerate(polygons)
        if polygon is not None]
    triangles_with_index = list(
        (tuple(vertex_indices[vertex] for vertex in triangle_coords), dataset_index)
        for polygon, dataset_index in polygons_with_index
        for triangle_coords in _triangulate_polygon(polygon)
    )
    triangles: List[Triangle] = [tri for tri, index in triangles_with_index]  # type: ignore
    indices = [index for tri, index in triangles_with_index]

    return (vertices, triangles, indices)


def _triangulate_polygon(polygon: Polygon) -> List[Tuple[Vertex, Vertex, Vertex]]:
    """
    Triangulate a polygon.

    .. note::

        This currently only supports simple polygons - polygons that do not
        intersect themselves and do not have holes.

    Examples
    --------

    .. code-block:: python

        >>> polygon = Polygon([(0, 0), (2, 0), (2, 2), (1, 3), (0, 2), (0, 0)])
        >>> for triangle in triangulate_polygon(polygon):
        ...     print(triangle.wkt)
        POLYGON ((0 0, 2 0, 2 2, 0 0))
        POLYGON ((0 0, 2 2, 1 3, 0 0))
        POLYGON ((0 0, 1 3, 0 2, 0 0))

    See Also
    --------
    :func:`triangulate_dataset`,
    `Polygon triangulation <https://en.wikipedia.org/wiki/Polygon_triangulation>`_
    """
    if not polygon.is_simple:
        raise ValueError("_triangulate_polygon only supports simple polygons")

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

    # Maintain a consistent winding order
    polygon = polygon.normalize()

    triangles: List[Tuple[Vertex, Vertex, Vertex]] = []
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
                triangles.append((coords[i], coords[i + 1], coords[i + 2]))
                polygon = Polygon(coords[:i + 1] + coords[i + 2:])
                break
        else:
            # According to the twos ear theorem, we should never reach this
            raise ValueError(
                f"Could not find interior diagonal for polygon! {polygon.wkt}")

    # The trimmed polygon is now a triangle. Add it to the list and we are done!

    triangles.append(cast(
        Tuple[Vertex, Vertex, Vertex],
        tuple(map(tuple, polygon.exterior.coords[:-1]))))
    return triangles
