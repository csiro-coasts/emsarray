"""
This module contains :func:`plot_on_figure` and :func:`animate_on_figure`.
These functions will generate a simple plot of any supported variable.
These functions have limited customisation options
as they are intended as quick and simple ways of exploring a dataset.
Consult the :ref:`examples gallery <examples>`
for demonstrations on making more customised plots.
"""

from collections.abc import Callable, Iterable
from typing import Any, Literal

import cartopy.crs
import numpy
import xarray
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import animation
from matplotlib.artist import Artist
from matplotlib.figure import Figure

from emsarray import conventions, utils
from emsarray.types import DataArrayOrName, Landmark

from . import shortcuts
from .artists import GridArtist
from .utils import make_plot_title


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

    This method is a shortcut for quickly generating simple plots.
    It is not intended to be fully featured.
    See the :ref:`examples <examples>` for more comprehensive plotting examples.

    Parameters
    ----------
    figure
        The :class:`~matplotlib.figure.Figure` instace to plot on.
    convention
        The :class:`~emsarray.conventions.Convention` instance of the dataset.
        This is used to build the polygons and vector quivers.
    *variables : :class:`xarray.DataArray` or tuples of :class:`xarray.DataArray`
        Any number of dataset variables to plot.
        Scalar variables should be passed in directly,
        while vector pairs should be passed in as a tuple.
        These will be passed to :meth:`.Convention.make_artist`.
    scalar : :class:`xarray.DataArray`, optional
        The data to plot as an :class:`xarray.DataArray`.

        .. deprecated:: 1.0.0

            Pass in variables as positional arguments instead

    vector : tuple of :class:`numpy.ndarray`, optional
        The *u* and *v* components of a vector field
        as a tuple of :class:`xarray.DataArray`.

        .. deprecated:: 1.0.0

            Pass in variables as positional arguments instead

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

    data_arrays = [
        utils.names_to_data_arrays(convention.dataset, v)
        for v in variables
    ]

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
        shortcuts.add_landmarks(axes, landmarks)
    if coast:
        shortcuts.add_coast(axes)
    if gridlines:
        shortcuts.add_gridlines(axes)

    axes.autoscale()


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

    This method is a shortcut for quickly generating simple plots.
    It is not intended to be fully featured.
    See the :ref:`examples <examples>` for more comprehensive plotting examples.

    Parameters
    ----------
    figure : :class:`matplotlib.figure.Figure`
        The :class:`~matplotlib.figure.Figure` instace to plot on.
    convention
        The :class:`~emsarray.conventions.Convention` instance of the dataset.
        This is used to build the polygons and vector quivers.
    coordinate : :class:`xarray.DataArray`
        The coordinate values to vary across frames in the animation.
    *variables : :class:`xarray.DataArray` or tuples of :class:`xarray.DataArray`
        Any number of dataset variables to plot.
        Scalar variables should be passed in directly,
        while vector pairs should be passed in as a tuple.
        These will be passed to :meth:`.Convention.make_artist`.
    scalar : :class:`xarray.DataArray`, optional
        The data to plot as an :class:`xarray.DataArray`.

        .. deprecated:: 1.0.0

            Pass in variables as positional arguments instead

    vector : tuple of :class:`numpy.ndarray`, optional
        The *u* and *v* components of a vector field
        as a tuple of :class:`xarray.DataArray`.

        .. deprecated:: 1.0.0

            Pass in variables as positional arguments instead

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

    if landmarks:
        shortcuts.add_landmarks(axes, landmarks)
    if coast:
        shortcuts.add_coast(axes)
    if gridlines:
        shortcuts.add_gridlines(axes)

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
