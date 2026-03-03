import shapely
import xarray
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import pyplot as plt

from emsarray.types import Bounds, Pathish


def box(minx, miny, maxx, maxy) -> shapely.Polygon:
    """
    Make a box, with coordinates going counterclockwise
    starting at (minx miny).
    """
    return shapely.Polygon([
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
    ])


def plot_geometry(
    dataset: xarray.Dataset,
    out: Pathish,
    *,
    figsize: tuple[float, float] = (10, 10),
    extent: Bounds | None = None,
    title: str | None = None
) -> None:
    figure = plt.figure(layout='constrained', figsize=figsize)
    axes: GeoAxes = figure.add_subplot(projection=dataset.ems.data_crs)
    axes.set_aspect(aspect='equal', adjustable='datalim')
    axes.gridlines(draw_labels=['left', 'bottom'], linestyle='dashed')

    dataset.ems.plot_geometry(axes)
    grid = dataset.ems.default_grid
    x, y = grid.centroid_coordinates.T
    axes.scatter(x, y, c='red')

    if title is not None:
        axes.set_title(title)
    if extent is not None:
        axes.set_extent(extent)

    figure.savefig(out)
