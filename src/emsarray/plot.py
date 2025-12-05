# required as this module imports optional
# dependencies that are used for type hints.
from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from typing import Any, Literal, Self, TypeVar, cast

import numpy
import shapely
import xarray

from emsarray import conventions, utils
from emsarray.exceptions import NoSuchCoordinateError
from emsarray.types import DataArrayOrName, Landmark

try:
    import cartopy.crs
    from cartopy.feature import GSHHSFeature
    from cartopy.mpl import gridliner
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib import animation, patheffects
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection, TriMesh
    from matplotlib.figure import Figure
    from matplotlib.quiver import Quiver
    from matplotlib.tri import TriContourSet, Triangulation
    CAN_PLOT = True
    IMPORT_EXCEPTION = None
except ImportError as exc:
    CAN_PLOT = False
    IMPORT_EXCEPTION = exc


__all___ = ['CAN_PLOT', 'plot_on_figure', 'polygons_to_collection']


_requires_plot = utils.requires_extra(extra='plot', import_error=IMPORT_EXCEPTION)


def add_coast(axes: GeoAxes, **kwargs: Any) -> None:
    """
    Add coastlines to an :class:`~matplotlib.axes.Axes`.
    Some default styles are applied:
    the land polygons are light grey and semi-transparent,
    and the coastlines are opaque dark grey.

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        The axes to add the coastline to
    **kwargs
        Passed to :meth:`Axes.add_feature() <matplotlib.axes.Axes.add_feature>`.
    """
    kwargs = {
        'facecolor': (0.7, 0.7, 0.7, 0.5),
        'edgecolor': 'darkgrey',
        'linewidth': 0.5,
        **kwargs,
    }
    coast = GSHHSFeature()
    axes.add_feature(coast, **kwargs)


def add_gridlines(axes: GeoAxes, **kwargs: Any) -> gridliner.Gridliner:
    """
    Add a :class:`~cartopy.mpl.gridliner.Gridliner` to the axes.

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        The axes to add the gridlines to.
    kwargs : Any
        Extra kwargs to pass to the gridliner

    Returns
    -------
    cartopy.mpl.gridliner.Gridliner
    """
    kwargs = {
        'draw_labels': ['left', 'bottom'],
        **kwargs,
    }
    return axes.gridlines(**kwargs)


def add_landmarks(
    axes: Axes,
    landmarks: Iterable[Landmark],
    color: str = 'black',
    outline_color: str = 'white',
    outline_width: int = 2,
) -> None:
    """
    Place some named landmarks on a plot.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The axes to add the landmarks to.
    landmarks : list of :data:`landmarks <emsarray.types.Landmark>`
        The landmarks to add. These are tuples of (name, point).
    color : str, default 'black'
        The color for the landmark marker and labels.
    outline_color : str, default 'white'
        The color for the outline.
        Both the marker and the labels are outlined.
    outline_width : ind, default 2
        The linewidth of the outline.

    Examples
    --------
    Draw a plot of a specific area with some landmarks:

    .. code-block:: python

        import emsarray.plot
        import shapely
        from matplotlib import pyplot

        dataset = emsarray.tutorial.open_dataset('gbr4')

        # set up the figure
        figure = pyplot.figure()
        axes = figure.add_subplot(projection=dataset.ems.data_crs)
        axes.set_title("Sea surface temperature around Mackay")
        axes.set_aspect('equal', adjustable='datalim')
        emsarray.plot.add_coast(axes, zorder=1)

        # Focus on the area of interest
        axes.set_extent((148.245710, 151.544167, -19.870197, -21.986412))

        # Plot the temperature
        temperature = dataset.ems.make_artist(
            axes,
            dataset['temp'].isel(time=0, k=-1),
            cmap='jet', edgecolor='face', zorder=0)

        # Name key locations
        emsarray.plot.add_landmarks(axes, [
            ('The Percy Group', shapely.Point(150.270579, -21.658269)),
            ('Whitsundays', shapely.Point(148.955319, -20.169076)),
            ('Mackay', shapely.Point(149.192671, -21.146719)),
        ])

        figure.show()
    """
    outline = patheffects.withStroke(
        linewidth=outline_width, foreground=outline_color)

    points = axes.scatter(
        [p.x for n, p in landmarks], [p.y for n, p in landmarks],
        c=color, edgecolors=outline_color, linewidths=outline_width / 2)
    points.set_path_effects([outline])

    for name, point in landmarks:
        text = axes.annotate(
            name, (point.x, point.y),
            textcoords='offset pixels', xytext=(10, -5))
        text.set_path_effects([outline])


def bounds_to_extent(bounds: tuple[float, float, float, float]) -> list[float]:
    """
    Convert a Shapely bounds tuple to a matplotlib extents.

    A Shapely bounds is a tuple of (min x, min y, max x, max y),
    while a Matplotlib extent is a list of (min x, max x, min y, max y).

    Example
    -------

    .. code-block:: python

        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
        from emsarray.plot import bounds_to_extent
        from shapely.geometry import Polygon

        polygon = Polygon([
            (148.30, -40.75), (146.47, -41.19), (144.67, -40.62),
            (146.05, -43.53), (146.87, -43.60), (148.30, -40.75),
        ])
        figure = plt.figure(figsize=(10, 8), dpi=100)
        axes = plt.subplot(projection=ccrs.PlateCarree())
        axes.set_extent(bounds_to_extent(polygon.buffer(0.1).bounds))
    """
    minx, miny, maxx, maxy = bounds
    return [minx, maxx, miny, maxy]


@_requires_plot
def polygons_to_collection(
    polygons: Iterable[shapely.Polygon],
    **kwargs: Any,
) -> PolyCollection:
    """
    Convert a list of :class:`shapely.Polygon` instances
    to a matplotlib :class:`~matplotlib.collections.PolyCollection`.

    Parameters
    ----------
    polygons : iterable of :class:`shapely.Polygon`
        The polygons for the poly collection
    **kwargs : Any
        Keyword arguments to pass to the PolyCollection constructor.

    Returns
    -------
    :class:`matplotlib.collections.PolyCollection`
        A PolyCollection made up of the polygons passed in.
    """
    return PolyCollection(
        verts=[
            numpy.asarray(polygon.exterior.coords)
            for polygon in polygons
        ],
        closed=False,
        **kwargs
    )


def make_plot_title(
    dataset: xarray.Dataset,
    data_array: xarray.DataArray,
) -> str | None:
    """
    Make a suitable plot title for a variable.
    This will attempt to find a name for the variable by looking through the attributes.
    If the variable has a time coordinate,
    and the time coordinate has a single value,
    the time step is appended after the title.
    """
    if 'long_name' in data_array.attrs:
        title = str(data_array.attrs['long_name'])
    elif 'standard_name' in data_array.attrs:
        title = str(data_array.attrs['standard_name'])
    elif data_array.name is not None:
        title = str(data_array.name)
    else:
        return None

    # Check if this variable has a time coordinate
    try:
        time_coordinate = dataset.ems.time_coordinate
    except NoSuchCoordinateError:
        return title
    if time_coordinate.name not in data_array.coords:
        return title
    # Fetch the coordinate from the data array itself,
    # in case someone did `data_array = dataset['temp'].isel(time=0)`
    time_coordinate = data_array.coords[time_coordinate.name]

    if len(time_coordinate.dims) == 0:
        time_value = time_coordinate.values
    elif time_coordinate.size == 1:
        time_value = time_coordinate.values[0]
    else:
        return title

    time_string = numpy.datetime_as_string(time_value, unit='auto')
    return f'{title}\n{time_string}'


@_requires_plot
def plot_on_figure(
    figure: Figure,
    convention: 'conventions.Convention',
    *variables: DataArrayOrName | tuple[DataArrayOrName, ...],
    title: str | None = None,
    projection: cartopy.crs.Projection | None = None,
    landmarks: Iterable[Landmark] | None = None,
    gridlines: bool = True,
    coast: bool = True,
) -> None:
    """
    Plot a :class:`~xarray.DataArray`
    on a :mod:`matplotlib` :class:`~matplotlib.figure.Figure`.

    Parameters
    ----------
    figure
        The :class:`~matplotlib.figure.Figure` instace to plot on.
    convention
        The :class:`~emsarray.conventions.Convention` instance of the dataset.
        This is used to build the polygons and vector quivers.
    scalar : :class:`xarray.DataArray`, optional
        The data to plot as an :class:`xarray.DataArray`.
        This will be passed to :meth:`.Convention.make_poly_collection`.
    vector : tuple of :class:`numpy.ndarray`, optional
        The *u* and *v* components of a vector field
        as a tuple of :class:`xarray.DataArray`.
        These will be passed to :meth:`.Convention.make_quiver`.
    title : str, optional
        The title of the plot. Optional.
    projection : :class:`~cartopy.crs.Projection`
        The projection to use when plotting.
        Optional, defaults to :class:`~cartopy.crs.PlateCarree`.
        This is different to the coordinate reference system for the data,
        which is defined in :attr:`.Convention.data_crs`.
    landmarks : list of :data:`landmarks <emsarray.types.Landmark>`, optional
        Landmarks to add to the plot. These are tuples of (name, point).
    gridlines : bool, default True
        Whether to add gridlines to the plot using :func:`add_gridlines()`.
    coast : bool, default True
        Whether to add coastlines to the plot using :func:`add_coast()`.
    """
    if projection is None:
        projection = cartopy.crs.PlateCarree()

    axes: GeoAxes = figure.add_subplot(projection=projection)
    axes.set_aspect(aspect='equal', adjustable='datalim')

    data_arrays = utils.name_to_data_array(convention.dataset, variables)

    if title is None:
        if len(variables) == 0:
            title = 'Geometry'

        if len(variables) == 1:
            data_array = data_arrays[0]
            if isinstance(data_array, xarray.DataArray):
                title = make_plot_title(convention.dataset, data_array)

    if title is not None:
        axes.set_title(title)

    for data_array in data_arrays:
        convention.make_artist(axes, data_array)

    if len(variables) == 0:
        convention.plot_geometry(axes)

    if landmarks:
        add_landmarks(axes, landmarks)

    if coast:
        add_coast(axes)
    if gridlines:
        add_gridlines(axes)

    axes.autoscale()


@_requires_plot
def animate_on_figure(
    figure: Figure,
    convention: 'conventions.Convention',
    coordinate: xarray.DataArray,
    *variables: DataArrayOrName | tuple[DataArrayOrName, ...],
    title: str | Callable[[Any], str] | None = None,
    projection: cartopy.crs.Projection | None = None,
    landmarks: Iterable[Landmark] | None = None,
    gridlines: bool = True,
    coast: bool = True,
    interval: int = 1000,
    repeat: bool | Literal['cycle', 'bounce'] = True,
) -> animation.FuncAnimation:
    """
    Plot a :class:`xarray.DataArray`
    on a :mod:`matplotlib` :class:`~matplotlib.figure.Figure`
    as a :class:`~matplotlib.animation.FuncAnimation`.

    Parameters
    ----------
    figure : :class:`matplotlib.figure.Figure`
        The :class:`~matplotlib.figure.Figure` instace to plot on.
    convention
        The :class:`~emsarray.conventions.Convention` instance of the dataset.
        This is used to build the polygons and vector quivers.
    coordinate : :class:`xarray.DataArray`
        The coordinate values to vary across frames in the animation.
    variables : :class:`xarray.DataArray` or tuple of :class:`xarray.DataArray`.
        The data to plot as an :class:`xarray.DataArray`.
        This will be passed to :meth:`.Convention.make_poly_collection`.
        It should have horizontal dimensions appropriate for this convention,
        and a dimension matching the ``coordinate`` parameter.
    title : str or callable, optional
        The title for each frame of animation.
        Optional, will default to the coordinate value for each frame.

        If this is a string, :meth:`str.format` will be called
        with the coordinate value for each frame.

        If this is a callable,
        it will be called with the coordinate value for each frame,
        and it should return a string for the title.
    projection : :class:`~cartopy.crs.Projection`
        The projection to use when plotting.
        Optional, defaults to :class:`~cartopy.crs.PlateCarree`.
        This is different to the coordinate reference system for the data,
        which is defined in :attr:`.Convention.data_crs`.
    landmarks : list of :data:`landmarks <emsarray.types.Landmark>`, optional
        Landmarks to add to the plot. These are tuples of (name, point).
    gridlines : bool, default True
        Whether to add gridlines to the plot using :func:`add_gridlines()`.
    coast : bool, default True
        Whether to add coastlines to the plot using :func:`add_coast()`.
    interval : int
        The interval between frames of animation
    repeat : {True, False, 'cycle', 'bounce'}
        Whether to repeat the animation or not.
        ``True`` and ``'cycle'`` will play the animation on loop forever.
        ``'bounce'`` will play the animation forward, then backward, then repeat.
        ``False`` will play the animation once then stop.

    Returns
    -------
    :class:`matplotlib.animation.Animation`
    """
    if projection is None:
        projection = cartopy.crs.PlateCarree()

    axes = figure.add_subplot(projection=projection)
    axes.set_aspect(aspect='equal', adjustable='datalim')
    axes.title.set_animated(True)

    data_arrays = utils.name_to_data_array(convention.dataset, variables)
    coordinate_dim = coordinate.dims[0]
    artists: list[GridArtist] = []
    for data_array in data_arrays:
        current_variable: xarray.DataArray | tuple[xarray.DataArray, ...]
        if isinstance(data_array, xarray.DataArray):
            current_variable = data_array.isel({coordinate_dim: 0})
        else:
            current_variable = tuple(v.isel({coordinate_dim: 0}) for v in data_array)

        artist = convention.make_artist(axes, current_variable, animated=True)
        artists.append(artist)

    # Draw a coast overlay
    if coast:
        add_coast(axes)
    if gridlines:
        add_gridlines(axes)
    if landmarks:
        add_landmarks(axes, landmarks)
    axes.autoscale()

    repeat_arg = True
    coordinate_indexes = numpy.arange(coordinate.size)
    if repeat is False:
        repeat_arg = False
    elif repeat == 'bounce':
        coordinate_indexes = numpy.concatenate(
            (coordinate_indexes, coordinate_indexes[1:-1][::-1]))

    # Set up the animation
    coordinate_callable: Callable[[Any], str]
    if title is None:
        coordinate_callable = str
    elif isinstance(title, str):
        coordinate_callable = title.format
    else:
        coordinate_callable = title
    axes.set_title(coordinate_callable(coordinate[0]))

    def animate(index: int) -> Iterable[Artist]:
        if index > 0:
            figure.set_layout_engine('none')

        changes: list[Artist] = []
        coordinate_value = coordinate.values[index]
        frame_title = coordinate_callable(coordinate_value)
        axes.title.set_text(frame_title)
        changes.append(axes.title)

        for data_array, artist in zip(data_arrays, artists):
            current_variable: xarray.DataArray | tuple[xarray.DataArray, ...]
            if isinstance(data_array, xarray.DataArray):
                current_variable = data_array.isel({coordinate_dim: index})
            else:
                current_variable = tuple(v.isel({coordinate_dim: index}) for v in data_array)
            artist.set_data_array(current_variable)
            changes.append(artist)

        return changes

    # Draw the figure to force everything to compute its size,
    # for vectors to be initialized, etc.
    figure.draw_without_rendering()

    # Set the first frame of data
    animate(0)

    # Make the animation
    # blit=True makes things much faster, but means that things outside of the axes can not be animated.
    # This includes the title.
    return animation.FuncAnimation(
        figure, animate, frames=coordinate_indexes,
        interval=interval, repeat=repeat_arg,
        init_func=lambda: animate(0))


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
) -> PolygonScalarCollection:
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
        'linewidth': 0.5,
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
) -> PolygonTriContourSet:
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
        coords = numpy.array([
            [point.x, point.y] if point is not None else [numpy.nan, numpy.nan]
            for point in face_centres
        ])
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
        # so we need to recreated it.
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
) -> PolygonVectorQuiver:
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

        face_centres = numpy.array([
            [p.x, p.y] if p is not None else [numpy.nan, numpy.nan]
            for p in grid.centroid
        ])

        x, y = numpy.transpose(face_centres)

        # A Quiver needs some values when being initialized.
        # We don't always want to provide values to the quiver,
        # sometimes preferring to fill them in later,
        # so `u` and `v` are optional.
        # If they are not provided, we set default quiver values of `numpy.nan`.
        values: tuple[numpy.ndarray, numpy.ndarray] | tuple[float, float]
        values = numpy.nan, numpy.nan

        if data_array is not None:
            values = cls.ravel_data_array(grid, data_array)

        return cls(axes, x, y, grid=grid, *values, **kwargs)

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
) -> NodeTriMesh:
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

        vertices, triangles, _face_indexes = grid.convention.triangulate()
        if len(vertices) != grid.size:
            raise ValueError("Expected dataset triangulation to have the same number of vertices as the grid size.")

        triangulation = Triangulation(vertices[:, 0], vertices[:, 1], triangles)

        if data_array is not None:
            values = cls.ravel_data_array(grid, data_array)
            kwargs['array'] = values

        return cls(
            triangulation,
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
