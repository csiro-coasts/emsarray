"""
This module contains shortcut functions to make common plotting operations simple.
They are designed as quick shortcuts for making basic plots for exploring a dataset.
They aim for ease of use and simplicity over being fully featured.
"""
import copy
import logging
from collections.abc import Iterable
from functools import cache
from importlib.metadata import version
from typing import Any

import cartopy
from cartopy.feature import GSHHSFeature
from cartopy.mpl import gridliner
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import patheffects
from matplotlib.axes import Axes
from packaging.version import Version

from emsarray.types import Landmark

logger = logging.getLogger(__name__)

# The GSHHS download URL changed because of NOAA funding problems.
# Cartopy has updated the URL for the download but has not released a patched
# version yet. This is the last version with the old broken URL.
# We will monkeypatch the cartopy GSHHS downloader in this version and older.
#
# See also: https://github.com/SciTools/cartopy/pull/2659
CARTOPY_BAD_GSHHS_VERSION = Version('0.25.0')


@cache  # Abusing cache so this can only be called a single time
def monkeypatch_gshhs_downloader() -> None:
    """
    Monkeypatch the cartopy GSHHS downloader to update the URL.
    """
    logger.debug("monkeypatching gshhs download URL")
    cartopy_version = Version(version('cartopy'))
    if cartopy_version > CARTOPY_BAD_GSHHS_VERSION:
        logger.debug("bailing because version check")
        return

    # Make sure everything is where it should be
    downloader_key = ('shapefiles', 'gshhs')
    if downloader_key not in cartopy.config['downloaders']:
        logger.debug("bailing because downloader isn't present")
        return

    # Make sure we don't accidentally double patch someone elses monkeypatch
    current_downloader = cartopy.config['downloaders'][downloader_key]
    if not current_downloader.url({}).startswith('https://www.ngdc.noaa.gov/mgg/shorelines/data/'):
        logger.debug("bailing because download url doesn't match expected")
        return

    # Swap out the downloader for a copy with the correct URL
    new_downloader = copy.copy(current_downloader)
    new_downloader.url_template = 'https://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip'
    cartopy.config['downloaders'][downloader_key] = new_downloader

    logger.info("Monkeypatched cartopy GSHHS downloader url")


def add_coast(axes: GeoAxes, **kwargs: Any) -> None:
    """
    Add coastlines to an :class:`~matplotlib.axes.Axes`.
    Some default styles are applied:
    the land polygons are light grey and semi-transparent,
    and the coastlines are opaque dark grey.

    This uses the :class:`GSHHS coastline feature <cartopy.feature.GSHHSFeature>`
    which provides a reasonably accurate, reasonably detailed coast line
    for large scale models.
    Plots of smaller regions may find the resolution not suitable
    and may need to source a more detailed coastline shape from elsewhere.

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        The axes to add the coastline to
    kwargs
        Passed to :meth:`GeoAxes.add_feature() <cartopy.mpl.geoaxes.GeoAxes.add_feature>`.
    """
    monkeypatch_gshhs_downloader()

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
    For all available options consult the cartopy
    :class:`Gridliner documentation <cartopy.mpl.gridliner.Gridliner>`.

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
    .. minigallery:: ../examples/plot-set-extent.py
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
