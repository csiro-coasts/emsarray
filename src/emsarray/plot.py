from __future__ import annotations

from typing import (
    TYPE_CHECKING, Any, Callable, Iterable, List, Literal, Optional, Tuple,
    Union
)

import numpy as np
import xarray as xr

from emsarray.utils import requires_extra

if TYPE_CHECKING:
    from .formats import Format

try:
    import cartopy.crs
    from cartopy.feature import GSHHSFeature
    from cartopy.mpl import gridliner
    from matplotlib import animation, patches
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.collections import PatchCollection
    from matplotlib.figure import Figure
    from shapely.geometry import Polygon
    CAN_PLOT = True
    IMPORT_EXCEPTION = None
except ImportError as exc:
    CAN_PLOT = False
    IMPORT_EXCEPTION = exc


__all___ = ['CAN_PLOT', 'plot_on_figure', 'polygon_to_patch']


_requires_plot = requires_extra(extra='plot', import_error=IMPORT_EXCEPTION)


def add_coast(axes: Axes, **kwargs: Any) -> None:
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
    axes.add_feature(coast,  **kwargs)


def add_gridlines(axes: Axes) -> gridliner.Gridliner:
    gridlines = axes.gridlines(draw_labels=True)
    gridlines.top_labels = False
    gridlines.right_labels = False
    return gridlines


def bounds_to_extent(bounds: Tuple[float, float, float, float]) -> List[float]:
    """
    Convert a Shapely bounds tuple to a matplotlib extents.

    A Shapely bounds is a tuple of (min x, min y, max x, max y),
    while a Matplotlib extent is a list of (min x, max x, min y, max y).

    Example
    -------

    .. code-block:: python

        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
        from emsarray.plot import bounds_to_extent, polygon_to_patch
        from shapely.geometry import Polygon

        polygon = Polygon([
            (148.30, -40.75), (146.47, -41.19), (144.67, -40.62),
            (146.05, -43.53), (146.87, -43.60), (148.30, -40.75),
        ])
        figure = plt.figure(figsize=(10, 8), dpi=100)
        axes = plt.subplot(projection=ccrs.PlateCarree())
        axes.set_extent(bounds_to_extent(polygon.buffer(0.1).bounds))
        axes.add_patch(polygon_to_patch(polygon))
        figure.show()
    """
    minx, miny, maxx, maxy = bounds
    return [minx, maxx, miny, maxy]


@_requires_plot
def polygon_to_patch(polygon: Polygon, **kwargs: Any) -> patches.Polygon:
    """
    Convert a :class:`shapely.geometry.Polygon <Polygon>` to a
    :class:`matplotlib.patches.Polygon`.
    """
    return patches.Polygon(np.transpose(polygon.exterior.xy), **kwargs)


@_requires_plot
def polygons_to_patch_collection(
    polygons: Iterable[Polygon],
    **kwargs: Any,
) -> PatchCollection:
    """
    Convert a list of Shapely :class:`Polygons <Polygon>`
    to a matplotlib :class:`~matplotlib.collections.PatchCollection`.

    Parameters
    ----------
    polygons : iterable of `Polygon`
        The polygons for the patch collection
    **kwargs : Any
        Keyword arguments to pass to the PatchCollection constructor.

    Returns
    -------
    :class:`matplotlib.collections.PatchCollection`
        The PatchCollection made up of the polygons passed in.
    """
    return PatchCollection(map(polygon_to_patch, polygons), **kwargs)


@_requires_plot
def plot_on_figure(
    figure: Figure,
    format: Format,
    *,
    scalar: Optional[xr.DataArray] = None,
    vector: Optional[Tuple[xr.DataArray, xr.DataArray]] = None,
    title: Optional[str] = None,
    projection: Optional[cartopy.crs.Projection] = None,
) -> None:
    """
    Plot a :class:`~xarray.DataArray`
    on a :mod:`matplotlib` :class:`~matplotlib.figure.Figure`.

    Parameters
    ----------
    figure
        The :class:`~matplotlib.figure.Figure` instace to plot on.
    format
        The :class:`~emsarray.formats.Format` instance of the dataset.
        This is used to build the polygons and vector quivers.
    scalar : :class:`xarray.DataArray`, optional
        The data to plot as an :class:`xarray.DataArray`.
        This will be passed to :meth:`.Format.make_patch_collection`.
    vector : tuple of :class:`numpy.ndarray`, optional
        The *u* and *v* components of a vector field
        as a tuple of :class:`xarray.DataArray`.
        These will be passed to :meth:`.Format.make_quiver`.
    title : str, optional
        The title of the plot. Optional.
    projection : :class:`~cartopy.crs.Projection`
        The projection to use when plotting.
        Optional, defaults to :class:`~cartopy.crs.PlateCarree`.
        This is different to the coordinate reference system for the data,
        which is defined in :attr:`.Format.data_crs`.
    """
    if projection is None:
        projection = cartopy.crs.PlateCarree()

    axes = figure.add_subplot(projection=projection)
    axes.set_aspect(aspect='equal', adjustable='datalim')

    if scalar is None and vector is None:
        # Plot the polygon shapes for want of anything else to draw
        patches = format.make_patch_collection()
        axes.add_collection(patches)
        if title is None:
            title = 'Geometry'

    if scalar is not None:
        # Plot a scalar variable on the polygons using a colour map
        patches = format.make_patch_collection(
            scalar, cmap='jet', edgecolor='face')
        axes.add_collection(patches)
        units = scalar.attrs.get('units')
        figure.colorbar(patches, ax=axes, location='right', label=units)

    if vector is not None:
        # Plot a vector variable using a quiver
        quiver = format.make_quiver(axes, *vector)
        axes.add_collection(quiver)

    if title:
        axes.set_title(title)

    add_coast(axes)
    add_gridlines(axes)
    axes.autoscale()


@_requires_plot
def animate_on_figure(
    figure: Figure,
    format: Format,
    *,
    coordinate: xr.DataArray,
    scalar: Optional[xr.DataArray] = None,
    vector: Optional[Tuple[xr.DataArray, xr.DataArray]] = None,
    title: Optional[Union[str, Callable[[Any], str]]] = None,
    projection: Optional[cartopy.crs.Projection] = None,
    interval: int = 1000,
    repeat: Union[bool, Literal['cycle', 'bounce']] = True,
) -> animation.FuncAnimation:
    """
    Plot a :class:`xarray.DataArray`
    on a :mod:`matplotlib` :class:`~matplotlib.figure.Figure`
    as a :class:`~matplotlib.animation.FuncAnimation`.

    Parameters
    ----------
    figure : :class:`matplotlib.figure.Figure`
        The :class:`~matplotlib.figure.Figure` instace to plot on.
    format
        The :class:`~emsarray.formats.Format` instance of the dataset.
        This is used to build the polygons and vector quivers.
    coordinate : :class:`xarray.DataArray`
        The coordinate values to vary across frames in the animation.
    scalar : :class:`xarray.DataArray`, optional
        The data to plot as an :class:`xarray.DataArray`.
        This will be passed to :meth:`.Format.make_patch_collection`.
        It should have horizontal dimensions appropriate for this format,
        and a dimension matching the ``coordinate`` parameter.
    vector : tuple of :class:`numpy.ndarray`, optional
        The *u* and *v* components of a vector field
        as a tuple of :class:`xarray.DataArray`.
        These will be passed to :meth:`.Format.make_quiver`.
        These should have horizontal dimensions appropriate for this format,
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
        which is defined in :attr:`.Format.data_crs`.
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

    patches = None
    if scalar is not None:
        # Plot a scalar variable on the polygons using a colour map
        scalar_values = format.make_linear(scalar).values[:, format.mask]
        patches = format.make_patch_collection(
            cmap='jet', edgecolor='face',
            clim=(np.nanmin(scalar_values), np.nanmax(scalar_values)))
        axes.add_collection(patches)
        patches.set_animated(True)
        units = scalar.attrs.get('units')
        figure.colorbar(patches, ax=axes, location='right', label=units)

    quiver = None
    if vector is not None:
        # Plot a vector variable using a quiver
        vector_u_values, vector_v_values = (
            format.make_linear(vec).values
            for vec in vector)
        # Quivers must start with some data.
        # Vector arrows are scaled using this initial data.
        # Find the absolute maximum value in all directions for initial data
        # to make the autoscaling behave across all frames.
        coordinate_dim = coordinate.dims[0]
        initial_u, initial_v = (
            abs(vec).max(dim=coordinate_dim, skipna=True)
            for vec in vector)
        quiver = format.make_quiver(axes, initial_u, initial_v)
        quiver.set_animated(True)
        axes.add_collection(quiver)

    # Draw a coast overlay
    add_coast(axes)
    gridlines = add_gridlines(axes)
    axes.autoscale()

    repeat_arg = True
    coordinate_indexes = np.arange(coordinate.size)
    if repeat is False:
        repeat_arg = False
    elif repeat == 'bounce':
        coordinate_indexes = np.concatenate(
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
        changes: List[Artist] = []
        coordinate_value = coordinate.values[index]
        axes.title.set_text(coordinate_callable(coordinate_value))
        changes.append(axes.title)
        changes.extend(gridlines.xline_artists)
        changes.extend(gridlines.yline_artists)

        if patches is not None:
            patches.set_array(scalar_values[index])
            changes.append(patches)

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
