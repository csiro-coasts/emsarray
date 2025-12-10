import copy
from typing import Any, Self, cast

import numpy
import shapely
import xarray
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection, TriMesh
from matplotlib.quiver import Quiver
from matplotlib.tri import Triangulation, TriContourSet

from emsarray import conventions


class GridArtist(Artist):
    """
    A matplotlib Artist subclass that knows what Grid it is associated with,
    and has a `set_data_array()` method.
    Users can call `GridArtist.set_data_array()` to update the data in a plot.
    This is useful when making animations, for example.
    """
    _grid: 'conventions.Grid'

    def set_grid(self, grid: 'conventions.Grid') -> None:
        if hasattr(self, '_grid'):
            raise ValueError("_grid can not be changed once set")
        self._grid = grid

    def get_grid(self) -> 'conventions.Grid':
        return self._grid

    def set_data_array(self, data_array: Any) -> None:
        """
        Update the data this artist is plotting.
        The data array must be defined on the same :meth:`grid <GridArtist.get_grid>`,
        and must not have any extra dimensions such as depth or time.
        """
        raise NotImplementedError("Subclasses must implement this")


def make_polygon_scalar_collection(
    axes: Axes,
    grid: 'conventions.Grid',
    data_array: xarray.DataArray | None = None,
    add_colorbar: bool | None = None,
    **kwargs: Any,
) -> 'PolygonScalarCollection':
    """
    Make a :class:`PolygonScalarCollection` for a :class:`~emsarray.conventions.Grid`
    and :class:`~xarray.DataArray` with some sensible defaults.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The axes to add the artist to
    grid : emsarray.conventions.Grid
        The grid containing the geometry to plot on.
    data_array : xarray.DataArray or None
        The data array to plot. Optional, will plot just the geometry if None.
    add_colorbar : bool or None, default None
        Whether to add a colorbar. Will add a colorbar by default if a data array is supplied.
    kwargs : Any
        Extra kwargs for styling the PolygonScalarCollection.
        See :class:`matplotlib.collections.PolyCollection` for valid options.

    Returns
    -------
    PolygonScalarCollection
    """
    kwargs = {
        'edgecolor': 'face',
        'cmap': 'viridis',
        'transform': grid.convention.data_crs,
        **kwargs,
    }

    collection = PolygonScalarCollection.from_grid(grid, data_array=data_array, **kwargs)
    axes.add_collection(collection)

    if add_colorbar is None:
        add_colorbar = data_array is not None

    if add_colorbar:
        if data_array is not None:
            units = data_array.attrs.get('units')
        axes.figure.colorbar(collection, ax=axes, location='right', label=units)

    return collection


class PolygonScalarCollection(PolyCollection, GridArtist):
    """
    A :class:`GridArtist` wrapper around a :class:`~matplotlib.collections.PolyCollection`.
    This artist can plot scalar variables on grids with polygonal geometry.
    """
    @classmethod
    def from_grid(
        cls,
        grid: 'conventions.Grid',
        data_array: xarray.DataArray | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Create a PolygonScalarCollection for a particular polygon grid of a dataset.

        Parameters
        ----------
        grid : emsarray.conventions.Grid
            The grid to make the polygon collection for
        data_array : xarray.DataArray
            A data array, defined on the grid, with data to plot.
            Optional, if not provided the polygons will be empty.

        Returns
        -------
        PolygonScalarCollection
        """
        if not issubclass(grid.geometry_type, shapely.Polygon):
            raise ValueError("Grid must have polygon geometry")

        if data_array is not None:
            values = cls.ravel_data_array(grid, data_array)
            kwargs['array'] = values

        return cls(
            verts=[
                numpy.asarray(polygon.exterior.coords)
                for polygon in grid.geometry[grid.mask]
            ],
            closed=False,
            grid=grid,
            **kwargs,
        )

    def set_data_array(self, data_array: xarray.DataArray | None) -> None:
        if data_array is None:
            self.set_array(None)
        else:
            self.set_array(self.ravel_data_array(self._grid, data_array))

    @staticmethod
    def ravel_data_array(grid: 'conventions.Grid', data_array: xarray.DataArray) -> numpy.ndarray:
        flattened = grid.ravel(data_array)
        if len(flattened.dims) > 1:
            unexpected_dimensions = set(data_array.dims) & set(flattened.dims)
            raise ValueError(
                "Data array has too many dimensions, "
                "did you forget to select a single time step or depth layer? "
                f"Unexpected dimensions: {unexpected_dimensions}")
        return cast(numpy.ndarray, flattened.values[grid.mask])


def make_polygon_contour(
    axes: Axes,
    grid: 'conventions.Grid',
    data_array: xarray.DataArray,
    add_colorbar: bool = True,
    **kwargs: Any,
) -> 'PolygonTriContourSet':
    """
    Make a :func:`~matplotlib.pyplot.tricontour` plot
    through the centres of a polygon grid.
    This is a wrapper around making a :class:`PolygonTriContourSet`.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The axes to add the artist to
    grid : emsarray.conventions.Grid
        The grid containing the geometry to plot on.
    data_array : xarray.DataArray or None
        The data array to plot.
    add_colorbar : bool, default True
        Whether to add a colorbar.
    kwargs : Any
        Extra kwargs for styling the PolygonTriContourSet.
        See :class:`matplotlib.tri.TriContourSet` for valid options.

    Returns
    -------
    PolygonTriContourSet
    """
    if 'transform' not in kwargs:
        kwargs['transform'] = grid.convention.data_crs
    artist = PolygonTriContourSet.from_grid(axes, grid, data_array, **kwargs)
    axes.add_artist(artist)

    if add_colorbar:
        axes.figure.colorbar(artist._tri_contour_set)

    return artist


class PolygonTriContourSet(GridArtist, Artist):
    axes: Axes
    triangulation: Triangulation
    _tri_contour_set: TriContourSet

    def __init__(
        self,
        axes: Axes,
        triangulation: Triangulation,
        data_array: xarray.DataArray,
        grid: 'conventions.Grid',
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.axes = axes
        self.set_grid(grid)
        self.triangulation = triangulation
        self.tri_contour_set_properties = kwargs
        self.set_data_array(data_array)

    def get_children(self) -> list[Artist]:
        return [self._tri_contour_set]

    @classmethod
    def from_grid(
        cls,
        axes: Axes,
        grid: 'conventions.Grid',
        data_array: xarray.DataArray,
        **kwargs: Any,
    ) -> Self:
        """
        Create a PolygonTriContourSet for a particular polygon grid of a dataset.

        Parameters
        ----------
        grid : emsarray.conventions.Grid
            The grid to make the tricontour for
        data_array : xarray.DataArray
            A data array, defined on the grid, with data to plot.
        kwargs : Any
            Extra kwargs to configure the artist.
            See :class:`matplotlib.tri.TriContourSet` for valid options.

        Returns
        -------
        PolygonTriContourSet
        """
        if not issubclass(grid.geometry_type, shapely.Polygon):
            raise ValueError("Grid must have polygon geometry")

        triangulation = cls.make_triangulation(grid)
        return cls(
            axes=axes,
            triangulation=triangulation,
            data_array=data_array,
            grid=grid,
            **kwargs)

    @staticmethod
    def make_triangulation(grid: 'conventions.Grid') -> Triangulation:
        convention = grid.convention
        # Compute the Delaunay triangulation of the face centres
        face_centres = grid.centroid
        coords = numpy.full(fill_value=numpy.nan, shape=(grid.size, 2))
        coords[face_centres != None] = shapely.get_coordinates(face_centres)  # noqa: E711
        triangulation = Triangulation(coords[:, 0], coords[:, 1])

        # Mask out any Triangles that are not contained within the dataset geometry.
        # These are either in concave areas of the geometry (e.g. an inlet or bay)
        # or cover holes in the geometry (e.g. islands).
        valid_tris = convention.geometry.contains(shapely.polygons([
            [[triangulation.x[i], triangulation.y[i]] for i in tri]
            for tri in triangulation.triangles
        ]))
        triangulation.set_mask(~valid_tris)

        return triangulation

    def set_data_array(self, data_array: xarray.DataArray) -> None:
        triangulation = copy.copy(self.triangulation)
        values = self.ravel_data_array(self._grid, data_array)
        # TriContourSet does not handle nans within the data.
        # These need to be masked out.
        # Unfortunately it is not possible to modify the triangulation mask
        # after the TriContourSet has been created
        # so we need to recreate it.
        # We reuse the original Triangulation (TriContourSet discards it, so we save a copy).
        # We also reuse the kwargs passed in initially.

        # The mask applies to triangles, but it is the vertices that have nans.
        # We need to find all nan vertices,
        # then find all triangles that use one of those vertices,
        # then mask out those triangles,
        # while also not clobbering the existing mask that removes triangles outside the geometry.
        invalid_indices = numpy.flatnonzero(~numpy.isfinite(values))
        invalid_tris = numpy.any(numpy.isin(triangulation.triangles, invalid_indices), axis=1)
        triangulation.set_mask(triangulation.mask | invalid_tris)

        # Remove the old TriContourSet, if set
        if hasattr(self, '_tri_contour_set'):
            self._tri_contour_set.remove()

        # Make a new TriContourSet
        self._tri_contour_set = TriContourSet(
            self.axes, triangulation, values, **self.tri_contour_set_properties)

    def __getattr__(self, name: str) -> Any:
        # There is no good way to duplicate a TriContourSet,
        # updating its triangulation while keeping its properties.
        # There is also no good way of extracting all the properties that have been set.
        # We record the initial properties passed in as kwargs,
        # but we wouldn't track any properties that are modified after creation.
        # Lets make some dynamic setter functions that can track this.
        # I am so sorry.
        if name.startswith('set_'):
            prop = name[4:]
            actual_setter = getattr(self._tri_contour_set, name)

            def setter(value: Any) -> None:
                self.tri_contour_set_properties[prop] = value
                actual_setter(value)

            return setter
        raise AttributeError(name)

    @staticmethod
    def ravel_data_array(grid: 'conventions.Grid', data_array: xarray.DataArray) -> numpy.ndarray:
        flattened = grid.ravel(data_array)
        if len(flattened.dims) > 1:
            unexpected_dimensions = set(data_array.dims) & set(flattened.dims)
            raise ValueError(
                "Data array has too many dimensions, "
                "did you forget to select a single time step or depth layer? "
                f"Unexpected dimensions: {unexpected_dimensions}")
        return cast(numpy.ndarray, flattened.values[grid.mask])


type UVDataArray = tuple[xarray.DataArray, xarray.DataArray]


def make_polygon_vector_quiver(
    axes: Axes,
    grid: 'conventions.Grid',
    data_array: UVDataArray | None = None,
    **kwargs: Any,
) -> 'PolygonVectorQuiver':
    """
    Make a :class:`PolygonVectorQuiver` for a :class:`~emsarray.conventions.Grid`
    and :class:`~xarray.DataArray` with some sensible defaults.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The axes to add the artist to
    grid : emsarray.conventions.Grid
        The grid containing the geometry to plot on.
    data_array : tuple of (xarray.DataArray, xarray.DataArray) or None
        The data arrays to plot. Optional, will make some zero length arrows if not set.
    kwargs : Any
        Extra kwargs for styling the PolygonVectorQuiver
        See :class:`matplotlib.collections.Quiver` for valid options.

    Returns
    -------
    PolygonVectorQuiver
    """
    if 'transform' not in kwargs:
        kwargs['transform'] = grid.convention.data_crs

    collection = PolygonVectorQuiver.from_grid(axes, grid, data_array, **kwargs)
    axes.add_collection(collection)
    return collection


class PolygonVectorQuiver(Quiver, GridArtist):
    @classmethod
    def from_grid(
        cls,
        axes: Axes,
        grid: 'conventions.Grid',
        data_array: UVDataArray | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Create a PolygonVectorQuiver for a particular polygon grid of a dataset.

        Parameters
        ----------
        grid : emsarray.conventions.Grid
            The grid to make the quiver for
        data_array : tuple of (xarray.DataArray, xarray.DataArray) or None
            A data arrays, defined on the grid, with data to plot.

        Returns
        -------
        PolygonVectorQuiver
        """
        if not issubclass(grid.geometry_type, shapely.Polygon):
            raise ValueError("Grid must have polygon geometry")

        coords = numpy.full(fill_value=numpy.nan, shape=(grid.size, 2))
        coords[grid.centroid != None] = shapely.get_coordinates(grid.centroid)  # noqa: E711

        # A Quiver needs some values when being initialized.
        # We don't always want to provide values to the quiver,
        # sometimes preferring to fill them in later,
        # so `u` and `v` are optional.
        # If they are not provided, we set default quiver values of `numpy.nan`.
        values: tuple[numpy.ndarray, numpy.ndarray] | tuple[float, float]
        values = numpy.nan, numpy.nan

        if data_array is not None:
            values = cls.ravel_data_array(grid, data_array)

        return cls(axes, coords[:, 0], coords[:, 1], grid=grid, *values, **kwargs)

    def set_data_array(self, data_array: UVDataArray | None) -> None:
        if data_array is None:
            return
        values = self.ravel_data_array(self._grid, data_array)
        self.set_UVC(*values)

    @staticmethod
    def ravel_data_array(
        grid: 'conventions.Grid',
        data_array: UVDataArray,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        u, v = data_array

        if u.dims != v.dims:
            raise ValueError(
                "Vector data array dimensions must be identical!\n"
                f"u dimensions: {tuple(u.dims)}\n"
                f"v dimensions: {tuple(v.dims)}"
            )

        u, v = grid.ravel(u), grid.ravel(v)

        if len(u.dims) > 1:
            raise ValueError(
                "Vector data arrays have too many dimensions - did you forget to "
                "select a single timestep or a single depth layer?")

        return u.values, v.values


def make_node_scalar_artist(
    axes: Axes,
    grid: 'conventions.Grid',
    data_array: xarray.DataArray | None = None,
    *,
    add_colorbar: bool | None = None,
    **kwargs: Any,
) -> 'NodeTriMesh':
    """
    Make a :class:`NodeTriMesh` for a :class:`~emsarray.conventions.Grid`
    and :class:`~xarray.DataArray` with some sensible defaults.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The axes to add the artist to
    grid : emsarray.conventions.Grid
        The grid containing the geometry to plot on.
    data_array : xarray.DataArray
        The data array to plot.
    add_colorbar : bool, default True
        Whether to add a color bar to the plot.
    kwargs : Any
        Extra kwargs for styling the NodeTriMesh
        See :class:`matplotlib.collections.TriMesh` for valid options.

    Returns
    -------
    NodeTriMesh
    """
    if 'transform' not in kwargs:
        kwargs['transform'] = grid.convention.data_crs

    trimesh = NodeTriMesh.from_grid(
        grid=grid,
        data_array=data_array,
        **kwargs,
    )
    axes.add_collection(trimesh)

    if add_colorbar is None:
        add_colorbar = data_array is not None

    if add_colorbar:
        if data_array is not None:
            units = data_array.attrs.get('units')
        axes.figure.colorbar(trimesh, ax=axes, location='right', label=units)

    return trimesh


class NodeTriMesh(TriMesh, GridArtist):
    """
    A :class:`.GridArtist` wrapper around :class:`~matplotlib.collections.TriMesh`
    that can plot on the vertices of a dataset triangulation.
    """
    @classmethod
    def from_grid(
        cls,
        grid: 'conventions.Grid',
        data_array: xarray.DataArray | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Create a NodeTriMesh for a particular node grid of a dataset.

        Parameters
        ----------
        grid : emsarray.conventions.Grid
            The grid to make the trimesh for
        data_array : xarray.DataArray
            A data array, defined on the grid, with data to plot.

        Returns
        -------
        NodeTriMesh
        """
        if not issubclass(grid.geometry_type, shapely.Point):
            raise ValueError("NodeTriMesh can only plot data on node geometries")

        triangulation = grid.convention.make_triangulation()
        if triangulation.vertex_grid_kind is not grid.grid_kind:
            raise ValueError(f"Expected dataset triangulation vertices to have the grid kind {grid.grid_kind}")

        mpl_triangulation = Triangulation(
            triangulation.vertices[:, 0], triangulation.vertices[:, 1], triangulation.triangles)

        if data_array is not None:
            values = cls.ravel_data_array(grid, data_array)
            kwargs['array'] = values

        return cls(
            mpl_triangulation,
            grid=grid,
            **kwargs,
        )

    def set_data_array(self, data_array: xarray.DataArray | None) -> None:
        if data_array is None:
            self.set_array(None)
        else:
            self.set_array(self.ravel_data_array(self._grid, data_array))

    @staticmethod
    def ravel_data_array(grid: 'conventions.Grid', data_array: xarray.DataArray) -> numpy.ndarray:
        flattened = grid.ravel(data_array)

        if len(flattened.dims) > 1:
            extra_dimensions = ", ".join(map(str, set(data_array.dims) & set(flattened.dims)))
            raise ValueError(
                "Node data array has too many dimensions - did you forget to "
                "select a single timestep or a single depth layer? "
                f"Extra dimensions: {extra_dimensions}.")

        return cast(numpy.ndarray, flattened.values)
