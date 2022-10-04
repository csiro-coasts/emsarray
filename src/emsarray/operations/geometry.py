import json

import geojson
import xarray as xr

from emsarray.types import Pathish


def to_geojson(
    dataset: xr.Dataset,
) -> geojson.FeatureCollection:
    """Make a ``geojson.FeatureCollection`` out of the cells in this dataset,
    one feature per cell.

    Each feature will include the linear index and the native index
    of the corresponding cell in its properties.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to export as GeoJSON geometry.

    Returns
    -------
    ``geojson.FeatureCollection``
        The geometry of this dataset as a FeatureCollection.

    See also
    --------
    :func:`.write_geojson`
    """
    return geojson.FeatureCollection([
        geojson.Feature(geometry=polygon, properties={
            'linear_index': i,
            'index': dataset.ems.unravel_index(i),
        })
        for i, polygon in enumerate(dataset.ems.polygons)
        if polygon is not None
    ])


def write_geojson(
    dataset: xr.Dataset,
    path: Pathish,
) -> None:
    """
    Export the geometry of this dataset to a GeoJSON file
    as a ``geojson.FeatureCollection`` with one feature per cell.

    Each feature will include the linear index and the native index
    of the corresponding cell in its properties.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to export as GeoJSON geometry.
    path : str or pathlib.Path
        The path where the geometry should be written to.

    See also
    --------
    :func:`.to_geojson`
    """
    with open(path, 'w') as f:
        json.dump(to_geojson(dataset), f)

