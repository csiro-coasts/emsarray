"""Useful functions for analysing or manipulating :class:`xarray.Dataset` instances.
These operations are not specific to ems datasets.

Most operations have aliases on :class:`.Format`,
which can be accessed via the :attr:`dataset.ems` accessor.

"""
import warnings
from collections import defaultdict
from typing import Dict, FrozenSet, Hashable, List, Optional, Tuple, cast

import numpy as np
import xarray as xr
from shapely.geometry import LineString, MultiPoint, Polygon

from emsarray import utils


def ocean_floor(
    dataset: xr.Dataset,
    depth_variables: List[Hashable],
    *,
    non_spatial_variables: Optional[List[Hashable]] = None,
) -> xr.Dataset:
    """Make a new :class:`xarray.Dataset` reduced along the given depth
    coordinates to only contain values along the ocean floor.

    Parameters
    ----------
    dataset
        The dataset to reduce.
    depth_variables
        The names of depth coordinate variables.
        For supported formats, use :meth:`.Format.get_all_depth_names()`.
    non_spatial_variables
        Optional.
        A list of the names of any non-spatial coordinate variables, such as time.
        The ocean floor is assumed to be static across non-spatial dimensions.
        For supported formats, use :meth:`.Format.get_time_name()`.

    Returns
    -------
    :class:`xarray.Dataset`
        A new dataset with values taken from the deepest data.

    Examples
    --------

    .. code-block:: python

        >>> dataset
        <xarray.Dataset>
        Dimensions:  (z: 5, y: 5, x: 5)
        Coordinates:
            lon      (x) int64 0 -1 -2 -3 -4
            lat      (y) int64 0 1 2 3 4
            depth    (z) float64 4.25 3.25 2.25 1.25 0.25
        Dimensions without coordinates: z, y, x
        Data variables:
            temp     (z, y, x) float64 0.0 nan nan nan nan nan ... 4.0 4.0 4.0 4.0 4.0
        >>> operations.ocean_floor(dataset, ['depth'])
        <xarray.Dataset>
        Dimensions:  (y: 5, x: 5)
        Coordinates:
            lon      (x) int64 0 -1 -2 -3 -4
            lat      (y) int64 0 1 2 3 4
        Dimensions without coordinates: y, x
        Data variables:
            temp     (y, x) float64 0.0 1.0 2.0 3.0 4.0 1.0 ... 4.0 4.0 4.0 4.0 4.0

    This operation is relatively efficient,
    but still involves masking every variable that includes a depth axis.
    Where possible, do any time and space slicing before calling this method,
    and drop any variables you are not interested in.

    .. code-block:: python

        >>> operations.ocean_floor(
        ...     big_dataset['temp'].isel(record=0).to_dataset(),
        ...     depth_variables=big_dataset.ems.get_all_depth_names())
        <xarray.Dataset>
        Dimensions:  (y: 5, x: 5)
        Coordinates:
            lon      (x) int64 0 -1 -2 -3 -4
            lat      (y) int64 0 1 2 3 4
        Dimensions without coordinates: y, x
        Data variables:
            temp     (y, x) float64 0.0 1.0 2.0 3.0 4.0 1.0 ... 4.0 4.0 4.0 4.0 4.0

    See also
    --------
    :meth:`.Format.ocean_floor`
    :meth:`.Format.get_all_depth_names`
    :func:`.normalize_depth_variables`
    :func:`.utils.extract_vars`
    """
    # Consider both curvilinear SHOC datasets and UGRID COMPASS datasets.
    #
    # SHOC datasets have four 'grids': faces, left edges, back edges, and vertices.
    # Each of these grids has a two dimensional (j, i) index.
    #
    # COMPASS datasets have up to three 'grids': faces, edges, and nodes.
    # Each of these grids has a one dimensional index.
    #
    # Data variables might be defined on any one of these grids,
    # might have non-spatial (e.g. time) dimensions,
    # and might have a depth axis.
    # We assume that a data variable has at most one depth axis,
    # and that for a combination of depth dimension and spatial dimensions,
    # the ocean floor is static.

    dataset = normalize_depth_variables(
        dataset, depth_variables,
        positive_down=True, deep_to_shallow=False)

    if non_spatial_variables is None:
        non_spatial_variables = []

    # The name of all the relevant _dimensions_, not _coordinates_
    depth_dimensions = utils.dimensions_from_coords(dataset, depth_variables)
    non_spatial_dimensions = utils.dimensions_from_coords(dataset, non_spatial_variables)

    for depth_dimension in sorted(depth_dimensions, key=hash):
        dimension_sets: Dict[FrozenSet[str], List[str]] = defaultdict(list)
        for name, variable in dataset.data_vars.items():
            if depth_dimension not in variable.dims:
                continue  # Skip data variables without this depth dimension

            spatial_dimensions = frozenset(map(str, variable.dims)).difference(
                {depth_dimension}, non_spatial_dimensions)
            if not spatial_dimensions:
                continue  # Skip data variables with no other spatial dimenions

            dimension_sets[spatial_dimensions].append(name)

        for spatial_dimensions, variable_names in dimension_sets.items():
            # We now have a set of spatial_dimenions,
            # and a list of data variable_names.
            # We assume that a specific combination of depth variable and
            # spatial dimensions has a static ocean floor.
            # We find the ocean floor for one of the data variables,
            # and then use that to mask out each data variable in turn.

            # Get an example data array and drop all the non-spatial dimensions.
            data_array = dataset.data_vars[variable_names[0]].isel(
                {name: 0 for name in non_spatial_dimensions},
                drop=True, missing_dims='ignore')
            # Then find the ocean floor indices.
            ocean_floor_indices = _find_ocean_floor_indices(
                data_array, depth_dimension)

            # Extract just the variables with these spatial coordinates
            dataset_subset = utils.extract_vars(dataset, variable_names)

            # Drop any coordinates for this depth variable.
            # For some reason .isel() call will play havok with them,
            # so best to drop them beforehand.
            dataset_subset = dataset_subset.drop_vars([
                name for name, coordinate in dataset_subset.coords.items()
                if coordinate.dims == (depth_dimension,)
            ])

            # Find the ocean floor using the ocean_floor_indices
            dataset_subset = dataset_subset.isel(
                {depth_dimension: ocean_floor_indices},
                drop=True, missing_dims='ignore')

            # Merge these floored variables back in to the dataset
            dataset = dataset_subset.merge(dataset, compat='override')

    # Finally, drop the depth dimensions.
    # This will clear up any residual variables that use the depth variables,
    # such as depth coordinate variables.
    # errors='ignore' because the depth dimensions may have already been dropped
    dataset = dataset.drop_dims(depth_dimensions, errors='ignore')

    return dataset


def _find_ocean_floor_indices(
    data_array: xr.DataArray,
    depth_dimension: Hashable,
) -> xr.DataArray:
    # This needs some explaining.
    # (any number * 0 + 1) is 1, while (nan * 0 + 1) is nan.
    # As layers under the ocean floor are nans,
    # this will give us a series of 1's for water layers, with nans below,
    # such as [1, 1, 1, nan, nan, nan].
    # `.cumsum()` will then add all the 1's up cumulatively,
    # giving something like `[1, 2, 3, nan, nan, nan]` for a column.
    # `.argmax()` will find the highest non-nan index, and we have our answer!
    #
    # Columns of all nans will have an argmax index of 0.
    # Item 0 in the column will be nan, resulting in nan in the output as desired.
    depth_indices = (data_array * 0 + 1).cumsum(depth_dimension)
    max_depth_indices = depth_indices.argmax(depth_dimension)
    return cast(xr.DataArray, max_depth_indices)


def normalize_depth_variables(
    dataset: xr.Dataset,
    depth_variables: List[Hashable],
    *,
    positive_down: bool = True,
    deep_to_shallow: bool = True,
) -> xr.Dataset:
    """
    Some datasets represent depth as a positive variable, some as negative.
    Some datasets sort depth layers from deepest to most shallow, others
    from shallow to deep. :func:`normalize_depth_variables` will return
    a new dataset with the depth variables normalized.

    The default behaviour is for positive values to indicate deeper depths
    (indicated via the variable attribute ``positive: "down"``),
    and for the layers to be ordered deep to shallow.
    This behaviour can be modified using the parameters
    ``positive_down`` and ``deep_to_shallow`` respectively.

    Parameters
    ----------
    dataset
        The dataset to normalize
    depth_variables
        The names of the depth coordinate variables.
        This should be the names of the variables, not the dimensions,
        for datasets where these differ.
    positive_down
        When true (the default), positive values will indicate depth below the
        surface. When false, negative values indicate depth below the surface.
    deep_to_shallow
        When true (the default), the layers are ordered such that deeper layers
        have lower indices.

    See also
    --------

    :meth:`.Format.normalize_depth_variables`
    :meth:`.Format.get_all_depth_names`
    """
    new_dataset = dataset.copy()
    for name in depth_variables:
        variable = dataset[name]
        if len(variable.dims) != 1:
            raise ValueError(
                f"Can't normalize multidimensional depth variable {name!r} "
                f"with dimensions {list(variable.dims)!r}"
            )
        dimension = variable.dims[0]

        new_variable = new_dataset[name]
        new_variable.attrs['positive'] = 'down' if positive_down else 'up'

        positive_attr = variable.attrs.get('positive')
        if positive_attr is None:
            # This is a _depth_ variable. If there are more values >0 than <0,
            # positive is probably down.
            total_values = len(variable.values)
            positive_values = len(variable.values[variable.values > 0])
            positive_attr = 'down' if positive_values > total_values / 2 else 'up'

            warnings.warn(
                f"Depth variable {name!r} had no 'positive' attribute, "
                f"guessing `positive: {positive_attr!r}`",
                stacklevel=2)

        # Reverse the polarity
        if (positive_attr == 'down') != positive_down:
            new_values = -1 * new_variable.values
            if name == dimension:
                new_dataset = new_dataset.assign_coords({name: new_values})
                new_dataset[name].attrs = new_variable.attrs
                new_dataset[name].encoding = new_variable.encoding
                new_variable = new_dataset[name]
            else:
                new_dataset = new_dataset.assign({
                    name: ([dimension], new_values, new_variable.attrs, new_variable.encoding)
                })
                new_variable = new_dataset[name]

        # Check if the existing data goes from deep to shallow, correcting for
        # the positive_down we just adjusted above. This assumes that depth
        # data are monotonic across all values. If this is not the case,
        # good luck.
        d1, d2 = new_variable.values[0:2]
        data_deep_to_shallow = (d1 > d2) == positive_down

        # Flip the order of the coordinate
        if data_deep_to_shallow != deep_to_shallow:
            new_dataset = new_dataset.isel({dimension: np.s_[::-1]})

    return new_dataset


Vertex = Tuple[float, float]
Triangle = Tuple[int, int, int]


def triangulate_dataset(
    dataset: xr.Dataset,
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
        import numpy as np
        import trimesh
        from emsarray.operations import triangulate_dataset

        dataset = emsarray.tutorial.open_dataset("gbr4")
        vertices, triangles, cell_indices = triangulate_dataset(dataset)
        # Trimesh expects 3D vertices.
        vertices = np.c_[vertices, np.zeros(len(vertices))]
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.invert()  # Trimesh grids expect the opposite winding order

        depth = 1 - (dataset.data_vars["Mesh2_depth"].values / -200)
        depth_colour = np.c_[depth, depth, depth, np.ones_like(depth)] * 255
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

    # Each cell polygon needs to be triangulated, while also recording the
    # format-native index of the cell, so that we can later correlate cell data
    # with the triangles.
    polygons_with_index = [
        (polygon, index)
        for index, polygon in enumerate(polygons)
        if polygon is not None]
    triangles_with_index = list(
        (tuple(vertex_indices[vertex] for vertex in triangle.exterior.coords[:-1]), dataset_index)
        for polygon, dataset_index in polygons_with_index
        for triangle in _triangulate_polygon(polygon)
    )
    triangles: List[Triangle] = [tri for tri, index in triangles_with_index]  # type: ignore
    indices = [index for tri, index in triangles_with_index]

    return (vertices, triangles, indices)


def _triangulate_polygon(polygon: Polygon) -> List[Polygon]:
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

    See also
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

    triangles = []
    # Note that shapely polygons with n vertices will be closed, and thus have
    # n+1 coordinates. We trim that superfluous coordinate off in the next line
    while len(polygon.exterior.coords) > 4:
        exterior = polygon.exterior
        coords = exterior.coords[:-1]
        # We try and make an ear from vertices (i, i+1, i+2)
        for i in range(len(coords) - 2):
            # If the diagonal between i and i+1 is within the larger polygon,
            # then this entire triangle is within the larger polygon
            vertices = [coords[i], coords[i + 2]]
            multipoint = MultiPoint(vertices)
            diagonal = LineString(vertices)
            # The intersection of the diagonal with the boundary should be two
            # points - the vertices in question. If three or more points are in
            # a line, the intersection will be something different. Removing
            # that ear will result in two disconnected polygons.
            intersection = exterior.intersection(diagonal)
            if diagonal.covered_by(polygon) and intersection.equals(multipoint):
                triangle = Polygon([coords[i], coords[i + 1], coords[i + 2]])
                triangles.append(triangle)
                polygon = Polygon(coords[:i + 1] + coords[i + 2:])
                break
        else:
            # According to the twos ear theorem, we should never reach this
            raise ValueError(
                f"Could not find interior diagonal for polygon! {polygon.wkt}")

    # The trimmed polygon is now a triangle. Add it to the list and we are done!
    triangles.append(polygon)
    return triangles
