"""
This module contains miscellaneous useful functions for making plots with matplotlib.
"""
from collections.abc import Iterable
from typing import Any

import numpy
import xarray
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon

from emsarray.exceptions import NoSuchCoordinateError


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


def polygons_to_collection(
    polygons: Iterable[Polygon],
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

    Parameters
    ----------
    dataset : xarray.Dataset
        A dataset.
    data_array : xarray.Dataset
        A data array from the dataset, or one of compatible shape.

    Returns
    -------
    title : str
        A suitable title for a plot of this data array.

    See also
    --------
    emsarray.utils.datetime_from_np_time
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
