"""
This module contains shortcut functions to make common plotting operations simple.
They are designed as quick shortcuts for making basic plots for exploring a dataset.
They aim for ease of use and simplicity over being fully featured.
"""
from collections.abc import Iterable
from typing import Any

from cartopy.feature import GSHHSFeature
from cartopy.mpl import gridliner
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import patheffects
from matplotlib.axes import Axes

from emsarray.types import Landmark


def add_coast(axes: GeoAxes, **kwargs: Any) -> None:
    """
    Add coastlines to an :class:`~matplotlib.axes.Axes`
    using :class:`cartopy.feature.GSHHSFeature`
    and :meth:`GeoAxes.add_feature() <cartopy.mpl.geoaxes.GeoAxes.add_feature>`.
    Some default styles are applied:
    the land polygons are light grey and semi-transparent,
    and the coastlines are opaque dark grey.

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        The axes to add the coastline to
    kwargs
        Passed to :meth:`GeoAxes.add_feature() <cartopy.mpl.geoaxes.GeoAxes.add_feature>`.
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
    Add a :class:`~cartopy.mpl.gridliner.Gridliner` to the axes
    including gridlines and with tick labels on bottom and left sides.

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        The axes to add the gridlines to.
    kwargs
        Passed to :meth:`GeoAxes.gridlines() <cartopy.mpl.geoaxes.GeoAxes.gridlines>`.

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
        temperature = dataset.ems.make_poly_collection(
            dataset['temp'].isel(time=0, k=-1),
            cmap='viridis', edgecolor='face', zorder=0)
        axes.add_collection(temperature)
        figure.colorbar(temperature, label='°C')

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
