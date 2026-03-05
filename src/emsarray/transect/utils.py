from typing import Any

import cfunits
import numpy
import shapely
import xarray
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter

from emsarray.exceptions import NoSuchCoordinateError
from emsarray.plot import make_plot_title
from emsarray.types import DataArrayOrName, Landmark
from emsarray.utils import name_to_data_array
from .base import Transect


def plot(
    dataset: xarray.Dataset,
    line: shapely.LineString,
    data_array: DataArrayOrName,
    *,
    figsize: tuple = (12, 3),
    title: str | None = None,
    bathymetry: xarray.DataArray | None = None,
    landmarks: list[Landmark] | None = None,
    **kwargs: Any,
) -> Figure:
    """
    Display a transect or cross section of a dataset along a path.

    This is convenience function that handles the most common use cases.
    For more control over the figure,
    see the comprehensive :ref:`transect example <sphx_glr_examples_plot-kgari-transect.py>`.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to transect.
    line : shapely.LineString
        The transect path to plot.
    data_array : DataArrayOrName
        A variable from the dataset to plot.
    figsize : tuple of int, int
        The size of the figure.
    title : str, optional
        The title of the plot.
        If `None` a title is pulled from the data array
        using :func:`~emsarray.plot.make_plot_title`.
    bathymetry : DataArrayOrName
        Used to draw an ocean floor polygon over the cross section.
        Only used if the data array to plot has a depth dimension.
    landmarks : list of Landmark
        Landmarks to add to the top axis of the plot.
        These can help viewers locate the transect in space.
    **kwargs
        Passed to :meth:`Transect.make_artist()`.
    """
    transect = Transect(dataset, line)

    figure = plt.figure(figsize=figsize, layout='constrained')
    data_array = name_to_data_array(dataset, data_array)
    transect_data = transect.extract(data_array)
    try:
        depth_coordinate = dataset.ems.get_depth_coordinate_for_data_array(data_array)
    except NoSuchCoordinateError:
        depth_coordinate = None

    axes = figure.add_subplot()
    artist = transect.make_artist(axes, data_array, **kwargs)

    if title is None:
        title = make_plot_title(dataset, data_array)
    if title:
        axes.set_title(title)

    setup_distance_axis(transect, axes)
    if depth_coordinate is not None:
        if len(transect.segments) > 0:
            depths_with_data = numpy.flatnonzero(numpy.isfinite(transect_data.values).any(axis=-1))
            depth_bounds = dataset[depth_coordinate.attrs['bounds']]
            ylim = (
                depth_bounds.values[depths_with_data[0], 0],
                depth_bounds.values[depths_with_data[-1], 1],
            )
        else:
            ylim = None
        setup_depth_axis(transect, axes, depth_coordinate=depth_coordinate, ylim=ylim)

        if bathymetry is not None:
            transect.make_ocean_floor_artist(axes, bathymetry)

    else:
        ylim = (numpy.nanmin(transect_data), numpy.nanmax(transect_data))
        axes.set_ylim(ylim)

    if landmarks is not None:
        top_axis = axes.secondary_xaxis('top')
        top_axis.set_ticks(
            [transect.distance_along_line(point) for label, point in landmarks],
            [label for label, point in landmarks],
        )

    plt.show()
    return figure


def setup_distance_axis(transect: Transect, axes: Axes) -> None:
    """
    Configure the x-axis of a :class:`~matplotlib.axes.Axes` for values along a transect.

    Parameters
    ==========
    transect : emsarray.transect.Transect
        The transect being plotted
    axes : matplotlib.axes.Axes
        The axes to configure
    """
    axis = axes.xaxis

    axes.set_xlim(transect.points[0].distance_metres, transect.points[-1].distance_metres)
    axis.set_label_text("Distance along transect")
    axis.set_major_formatter(EngFormatter(unit='m'))


def setup_depth_axis(
    transect: "Transect",
    axes: Axes,
    data_array: DataArrayOrName | None = None,
    depth_coordinate: DataArrayOrName | None = None,
    ylim: tuple[float, float] | bool = True,
    label: str | None | bool = True,
    units: str | None | bool = True,
) -> None:
    """
    Configure the y-axis of a :class:`~matplolib.axes.Axes` for values along a depth coordinate.

    Parameters
    ==========
    transect : emsarray.transect.Transect
        The transect being plotted
    axes : matplotlib.axes.Axes
        The axes to configure
    data_array : DataArrayOrName, optional
    depth_coordinate : DataArrayOrName, optional
        One of `data_array` or `depth_coordinate` must be provided.
        The y-axis is configured to show values along this depth coordinate.
        If data_array is provided, the depth coordinate for this data array is used.
    ylim : tuple of float, float, optional
        The ylim of the axes. If not provided the limit is calculated from the depth coordinate.
    label : str or None, optional
        The label for the y-axis.
        Optional, defaults to the `long_name` attribute of the depth coordinate.
        Set to `None` to disable the label.
    units : str or None, optional
        The units for the y-axis.
        Optional, defaults to the `units` attribute of the depth coordinate.
        Set to `None` to disable the units and formatting of tick labels.
    """
    if data_array is None and depth_coordinate is None:
        raise ValueError("Either data_array or depth_coordinate must be provided")
    if data_array is not None and depth_coordinate is not None:
        raise ValueError("Only one of data_array or depth_bounds must be provided")

    if data_array is not None:
        depth_coordinate = transect.convention.get_depth_coordinate_for_data_array(data_array)
    else:
        depth_coordinate = name_to_data_array(transect.dataset, depth_coordinate)

    axis = axes.yaxis

    if ylim is True:
        depth_bounds = transect.dataset[depth_coordinate.attrs['bounds']].values
        positive_down = depth_coordinate.attrs['positive'].lower() == 'down'
        depth_min, depth_max = numpy.nanmin(depth_bounds), numpy.nanmax(depth_bounds)

        if positive_down:
            axes.set_ylim(depth_max, depth_min)
        else:
            axes.set_ylim(depth_min, depth_max)
    elif ylim not in {False, None}:
        axes.set_ylim(ylim)

    if label is True:
        label = depth_coordinate.attrs.get('long_name')
    if label not in {False, None}:
        axis.set_label_text(label)

    if units is True:
        units = depth_coordinate.attrs.get('units')
    if units not in {False, None}:
        formatted_units = cfunits.Units(units).formatted()
        axis.set_major_formatter(EngFormatter(unit=formatted_units))
