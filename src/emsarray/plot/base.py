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

from emsarray import conventions
from emsarray.types import Landmark

from . import shortcuts


def plot_on_figure(
    figure: Figure,
    convention: 'conventions.Convention',
    *,
    scalar: xarray.DataArray | None = None,
    vector: tuple[xarray.DataArray, xarray.DataArray] | None = None,
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

    if scalar is None and vector is None:
        # Plot the polygon shapes for want of anything else to draw
        collection = convention.make_poly_collection()
        axes.add_collection(collection)
        if title is None:
            title = 'Geometry'

    if scalar is not None:
        # Plot a scalar variable on the polygons using a colour map
        collection = convention.make_poly_collection(
            scalar, edgecolor='face')
        axes.add_collection(collection)
        units = scalar.attrs.get('units')
        figure.colorbar(collection, ax=axes, location='right', label=units)

    if vector is not None:
        # Plot a vector variable using a quiver
        quiver = convention.make_quiver(axes, *vector)
        axes.add_collection(quiver)

    if title:
        axes.set_title(title)

    if landmarks:
        shortcuts.add_landmarks(axes, landmarks)
    if coast:
        shortcuts.add_coast(axes)
    if gridlines:
        shortcuts.add_gridlines(axes)

    axes.autoscale()

    # Work around for gridline positioning issues
    # https://github.com/SciTools/cartopy/issues/2245#issuecomment-1732313921
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.execute(figure)


def animate_on_figure(
    figure: Figure,
    convention: 'conventions.Convention',
    *,
    coordinate: xarray.DataArray,
    scalar: xarray.DataArray | None = None,
    vector: tuple[xarray.DataArray, xarray.DataArray] | None = None,
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
    scalar : :class:`xarray.DataArray`, optional
        The data to plot as an :class:`xarray.DataArray`.
        This will be passed to :meth:`.Convention.make_poly_collection`.
        It should have horizontal dimensions appropriate for this convention,
        and a dimension matching the ``coordinate`` parameter.
    vector : tuple of :class:`numpy.ndarray`, optional
        The *u* and *v* components of a vector field
        as a tuple of :class:`xarray.DataArray`.
        These will be passed to :meth:`.Convention.make_quiver`.
        These should have horizontal dimensions appropriate for this convention,
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

    collection = None
    if scalar is not None:
        # Plot a scalar variable on the polygons using a colour map
        scalar_values = convention.ravel(scalar).values[:, convention.mask]
        collection = convention.make_poly_collection(
            edgecolor='face',
            clim=(numpy.nanmin(scalar_values), numpy.nanmax(scalar_values)))
        axes.add_collection(collection)
        collection.set_animated(True)
        units = scalar.attrs.get('units')
        figure.colorbar(collection, ax=axes, location='right', label=units)

    quiver = None
    if vector is not None:
        # Plot a vector variable using a quiver
        vector_u_values, vector_v_values = (
            convention.ravel(vec).values
            for vec in vector)
        # Quivers must start with some data.
        # Vector arrows are scaled using this initial data.
        # Find the absolute maximum value in all directions for initial data
        # to make the autoscaling behave across all frames.
        coordinate_dim = coordinate.dims[0]
        initial_u, initial_v = (
            abs(vec).max(dim=str(coordinate_dim), skipna=True)
            for vec in vector)
        quiver = convention.make_quiver(axes, initial_u, initial_v)
        quiver.set_animated(True)
        axes.add_collection(quiver)

    if landmarks:
        shortcuts.add_landmarks(axes, landmarks)
    if coast:
        shortcuts.add_coast(axes)
    if gridlines:
        gridliner = shortcuts.add_gridlines(axes)

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

    def animate(index: int) -> Iterable[Artist]:
        changes: list[Artist] = []
        coordinate_value = coordinate.values[index]
        axes.title.set_text(coordinate_callable(coordinate_value))
        changes.append(axes.title)
        if gridlines:
            changes.extend(gridliner.xline_artists)
            changes.extend(gridliner.yline_artists)

        if collection is not None:
            collection.set_array(scalar_values[index])
            changes.append(collection)

        if quiver is not None:
            quiver.set_UVC(vector_u_values[index], vector_v_values[index])
            changes.append(quiver)

        return changes

    # Draw the figure to force everything to compute its size,
    # for vectors to be initializes, etc.
    figure.draw_without_rendering()

    # Set the first frame of data
    animate(0)

    # Make the animation
    return animation.FuncAnimation(
        figure, animate, frames=coordinate_indexes,
        interval=interval, repeat=repeat_arg, blit=True,
        init_func=lambda: animate(0))
