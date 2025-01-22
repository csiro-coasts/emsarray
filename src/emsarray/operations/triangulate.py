"""
Operations for making a triangular mesh out of the polygons of a dataset.
"""
import numpy
import pandas
import shapely
import xarray
from shapely.geometry import LineString, MultiPoint, Polygon

Vertex = tuple[float, float]
VertexTriangle = tuple[Vertex, Vertex, Vertex]
IndexTriangle = tuple[int, int, int]


def triangulate_dataset(
    dataset: xarray.Dataset,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
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
    tuple of vertices, triangles, and `cell_indices`
        A tuple of three numpy arrays is returned,
        containing vertices, triangles, and cell indexes respectively.

        `vertices` is a numpy array of shape (V, 2)
        where V is the number of unique vertices in the dataset.
        The vertex coordinates are in (x, y) or (lon, lat) order.

        `triangles` is a numpy array of shape (T, 3)
        where T is the number of triangles in the dataset.
        Each triangle is a set of three vertex indices.

        `cell_indices` is a numpy list of length T.
        Each entry indicates which polygon from the dataset a triangle is a part of.



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
        vertices, triangles, cell_indexes = triangulate_dataset(dataset)

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
        vertices, triangles, cell_indexes = triangulate_dataset(dataset)
        # Trimesh expects 3D vertices.
        vertices = numpy.c_[vertices, numpy.zeros(len(vertices))]
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.invert()  # Trimesh grids expect the opposite winding order

        depth = 1 - (dataset.data_vars["Mesh2_depth"].values / -200)
        depth_colour = numpy.c_[depth, depth, depth, numpy.ones_like(depth)] * 255
        mesh.visual.face_colors = depth_colour[cell_indexes]
        mesh.show()

    .. _holoviews: https://holoviews.org/reference/elements/bokeh/TriMesh.html
    .. _trimesh: https://trimsh.org
    """
    polygons = dataset.ems.polygons

    # Find all the unique coordinates and assign them each a unique index
    all_coords = shapely.get_coordinates(polygons)
    vertex_index = pandas.MultiIndex.from_arrays(all_coords.T).drop_duplicates()
    vertex_series = pandas.Series(numpy.arange(len(vertex_index)), index=vertex_index)
    vertex_coords = numpy.array(vertex_index.to_list())

    polygon_length = shapely.get_num_coordinates(polygons)

    # Count the total number of triangles.
    # A polygon with n sides can be decomposed in to n-2 triangles,
    # however polygon_length counts an extra vertex because of the closed rings.
    total_triangles = numpy.sum(polygon_length[numpy.nonzero(polygon_length)] - 3)

    # Pre-allocate some numpy arrays that will store the triangle data.
    # This is much faster than building up these data structures iteratively.
    face_indices = numpy.empty(total_triangles, dtype=int)
    triangle_coords = numpy.empty((total_triangles, 3, 2), dtype=float)
    # This is the index of which face we will populate next in the above arrays
    current_face = 0

    def _add_triangles(face_index: int, vertex_triangles: numpy.ndarray) -> None:
        """
        Append some triangles to the face_indices and triangle_coords arrays.

        Parameters
        ----------
        face_index : int
            The face index of all the triangles
        vertex_triangles : numpy.ndarray
            The triangles for this face as an (n, 3, 2) array,
            where n is the number of triangles in the face.
        """
        nonlocal current_face
        current_length = len(vertex_triangles)
        face_indices[current_face:current_face + current_length] = face_index
        triangle_coords[current_face:current_face + current_length] = vertex_triangles
        current_face += current_length

    # Find all concave polygons by comparing each polygon to its convex hull.
    # A convex polygon is its own convex hull,
    # while the convex hull of a concave polygon
    # will always have fewer vertices than the original polygon.
    # Comparing the number of vertices is a shortcut.
    convex_hulls = shapely.convex_hull(polygons)
    convex_hull_length = shapely.get_num_coordinates(convex_hulls)
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

        same_length_face_indices = numpy.flatnonzero(polygon_length == unique_length)
        same_length_polygons = polygons[same_length_face_indices]
        vertex_triangles = _triangulate_polygons_by_length(same_length_polygons)

        for face_index, triangles in zip(same_length_face_indices, vertex_triangles):
            _add_triangles(int(face_index), triangles)

    # Triangulate each concave polygon using a slower manual method.
    # Anecdotally concave polygons are very rare,
    # so using a slower method isn't an issue.
    for face_index in polygon_is_concave:
        polygon = polygons[face_index]
        triangles = _triangulate_concave_polygon(polygon)
        _add_triangles(int(face_index), triangles)

    # Check that we have handled each triangle we expected.
    assert current_face == total_triangles

    # Make a DataFrame. By manually constructing Series the data in the
    # underlying numpy arrays will be used in place.
    face_triangle_df = pandas.DataFrame({
        'face_indices': pandas.Series(face_indices),
        'x0': pandas.Series(triangle_coords[:, 0, 0]),
        'y0': pandas.Series(triangle_coords[:, 0, 1]),
        'x1': pandas.Series(triangle_coords[:, 1, 0]),
        'y1': pandas.Series(triangle_coords[:, 1, 1]),
        'x2': pandas.Series(triangle_coords[:, 2, 0]),
        'y2': pandas.Series(triangle_coords[:, 2, 1]),
    }, copy=False)

    joined_df = face_triangle_df\
        .join(vertex_series.rename('v0'), on=['x0', 'y0'])\
        .join(vertex_series.rename('v1'), on=['x1', 'y1'])\
        .join(vertex_series.rename('v2'), on=['x2', 'y2'])

    faces = joined_df['face_indices'].to_numpy()
    triangles = joined_df[['v0', 'v1', 'v2']].to_numpy()

    return vertex_coords, triangles, faces


def _triangulate_polygons_by_length(polygons: numpy.ndarray) -> numpy.ndarray:
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
        A numpy array of shape (# polygons, # triangles, 3, 2),
        where `# polygons` is the length of `polygons`
        and `# triangles` is the number of triangles each polygon is decomposed in to.
    """
    vertex_count = len(polygons[0].exterior.coords) - 1

    # An array of shape (len(polygons), vertex_count, 2)
    coordinates = shapely.get_coordinates(shapely.get_exterior_ring(polygons))
    coordinates = coordinates.reshape((len(polygons), vertex_count + 1, 2))
    coordinates = coordinates[:, :-1, :]

    # Arrays of shape (len(polygons), vertex_count - 2, 2)
    v0 = numpy.repeat(
        coordinates[:, 0, :].reshape((-1, 1, 2)),
        repeats=vertex_count - 2,
        axis=1)
    v1 = coordinates[:, 1:-1]
    v2 = coordinates[:, 2:]

    # An array of shape (len(polygons), vertex_count - 2, 3, 2)
    triangles: numpy.ndarray = numpy.stack([v0, v1, v2], axis=2)
    return triangles


def _triangulate_concave_polygon(polygon: Polygon) -> numpy.ndarray:
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
    triangles = numpy.empty((triangle_count, 3, 2))
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
                triangles[triangle_index] = coords[i:i + 3]
                triangle_index += 1
                polygon = Polygon(coords[:i + 1] + coords[i + 2:])
                break
        else:
            # According to the twos ear theorem, we should never reach this
            raise ValueError(
                f"Could not find interior diagonal for polygon! {polygon.wkt}")

    # The trimmed polygon is now a triangle. Add it to the list and we are done!
    triangles[triangle_index] = polygon.exterior.coords[:-1]
    assert (triangle_index + 1) == triangle_count

    return triangles
