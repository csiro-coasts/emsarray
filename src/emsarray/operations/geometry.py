import json
import os
import pathlib
from contextlib import contextmanager
from typing import IO, Any, Generator, Optional, TypeVar, Union

import geojson
import shapefile
import xarray as xr

from emsarray.types import Pathish

T = TypeVar('T')


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


@contextmanager
def _maybe_open(path_or_file: Union[Pathish, IO], mode: str) -> Generator[IO, None, None]:
    """
    Given either a path to a file or an open file handle,
    return an open file handle wrapped in a context manager.

    If given a path, the file is opened and returned as the context manager.
    The file is closed when the context manager closes.

    If given an open file handle, the handle is returned as-is.
    The file is not closed when the context manager closes.
    """
    if isinstance(path_or_file, IO):
        yield path_or_file
    else:
        with open(path_or_file, mode) as f:
            yield f


def write_shapefile(
    dataset: xr.Dataset,
    target: Optional[Pathish] = None,
    *,
    shp: Optional[Union[Pathish, IO]] = None,
    shx: Optional[Union[Pathish, IO]] = None,
    dbf: Optional[Union[Pathish, IO]] = None,
    prj: Optional[Union[Pathish, IO]] = None,
    **kwargs: Any,
) -> None:
    """
    Write the geometry of this dataset to a Shapefile.
    Each polygon is saved as an individual record
    with ``name``, ``linear_index``, and ``index`` fields.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to export the geometry for.
    target : str or pathlib.Path, optional
        Where to save the shapefile.
        The names for the .shp, .shx, .dbf, and .prj files are derived from this.
        Optional, you can instead provide arguments for each individual file.
    shp, shx, dbf, prj : str or pathlib.Path or opened file handle
        Where to write the individual shapefile components to.
        Optional, you can instead provide the base path as ``target``.
    **kwargs
        An extra keyword arguments are passed on to the shapefile.Writer instance.
    """
    target = str(target) if isinstance(target, pathlib.Path) else target
    with shapefile.Writer(target, shp=shp, shx=shx, dbf=dbf, **kwargs) as writer:
        writer.field('name', 'C')
        writer.field('linear_index', 'N')
        writer.field('index', 'C')
        for i, polygon in enumerate(dataset.ems.polygons):
            if polygon is None:
                continue
            writer.record(
                name=f'polygon{i}',
                linear_index=i,
                index=json.dumps(dataset.ems.unravel_index(i)),
            )
            writer.shape(polygon.__geo_interface__)

        # Write the projection file also, if we can find a filename for it...
        if prj is None:
            if target is not None:
                prj = os.path.splitext(target)[0] + '.prj'
        if prj is not None:
            with _maybe_open(prj, 'w') as prj_file:
                prj_file.write(dataset.ems.data_crs.to_wkt())
