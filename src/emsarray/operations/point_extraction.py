"""
Subset a dataset at a set of points.

:func:`.extract_dataframe` takes a pandas :class:`~pandas.DataFrame`,
subsets the dataset at the point specified in each row,
and merges the dataset with the dataframe.
The points extracted will form the coordinates for the new dataset.

:func:`.extract_points` takes a list of Shapely :class:`Points <Point>`,
subsets the dataset at these points,
and returns a new dataset with out any associated geometry.
This is useful if you want to add your own metadata to the subset dataset.

If any of the supplied points does not intersect the dataset geometry,
a :exc:`.NonIntersectingPoints` exception is raised.
This will include the indices of the points that do not intersect.

:ref:`emsarray extract-points` is a command line interface to :func:`.extract_dataframe`.
"""
import dataclasses
from typing import Hashable, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from emsarray.conventions import Convention


@dataclasses.dataclass
class NonIntersectingPoints(ValueError):
    """
    Raised when a point to extract does not intersect the dataset geometry.
    """

    #: The indices of the points that do not intersect
    indices: np.ndarray

    #: The non-intersecting points
    points: List[Point]

    def __post_init__(self) -> None:
        super().__init__(f"{self.points[0].wkt} does not intersect the dataset geometry")


def _dataframe_to_dataset(
    dataframe: pd.DataFrame,
    *,
    dimension_name: Hashable,
) -> xr.Dataset:
    """
    Convert a pandas DataFrame to an xarray Dataset.
    pandas adds an 'index' coordinate that numbers the 'index' dimension.
    We don't need the coordinate, and the dimension needs to be renamed.
    """
    index_name = dataframe.index.name or 'index'
    dataset = dataframe.to_xarray()
    dataset = dataset.drop_vars(index_name)
    if dimension_name != index_name:
        dataset = dataset.rename_dims({index_name: dimension_name})
    return dataset


def extract_points(
    dataset: xr.Dataset,
    points: List[Point],
    *,
    point_dimension: Hashable = 'point',
) -> xr.Dataset:
    """
    Drop all data except for cells that intersect the given points.
    Return a new dataset with a new dimension named ``point_dimension``,
    with the same size as the nubmer of ``points``,
    containing only data at those points.

    The returned dataset has no coordinate information.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to extract point data from.
    points : list of :class:`Point`
        The points to select.
    point_dimension : Hashable, optional
        The name of the new dimension to index points along.
        Defaults to ``"point"``.

    Returns
    -------
    xarray.Dataset
        A subset of the input dataset that only contains data at the given points.
        The dataset will only contain the values, without any coordinate information.

    See also
    --------
    :func:`extract_dataframe`
    """
    convention: Convention = dataset.ems

    # Find the indexer for each given point
    indexes = np.array([convention.get_index_for_point(point) for point in points])

    # TODO It would be nicer if out-of-bounds points were represented in the
    # output by masked values, rather than raising an error.
    out_of_bounds = np.flatnonzero(np.equal(indexes, None))  # type: ignore
    if len(out_of_bounds):
        raise NonIntersectingPoints(
            indices=out_of_bounds,
            points=[points[i] for i in out_of_bounds])

    # Make a DataFrame out of all point indexers
    selector_df = pd.DataFrame([
        convention.selector_for_index(index.index)
        for index in indexes
        if index is not None])

    # Subset the dataset to the points
    selector_ds = _dataframe_to_dataset(selector_df, dimension_name=point_dimension)
    return convention.drop_geometry().isel(selector_ds)


def extract_dataframe(
    dataset: xr.Dataset,
    dataframe: pd.DataFrame,
    coordinate_columns: Tuple[str, str],
    *,
    point_dimension: Hashable = 'point',
) -> xr.Dataset:
    """
    Extract the points listed in a pandas :class:`~pandas.DataFrame`,
    and merge the remaining columns in to the :class:`~xarray.Dataset`.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to extract point data from
    dataframe : pandas.DataFrame
        A dataframe with longitude and latitude columns,
        and possibly other columns.
    coordinate_columns : tuple of str, str
        The names of the longitude and latitude columns in the dataframe.
    point_dimension : Hashable, optional
        The name of the new dimension to create in the dataset.
        Optional, defaults to "point".

    Returns
    -------
    xarray.Dataset
        A new dataset that only contains data at the given points,
        plus any new columns present in the dataframe.

    Example
    -------

    .. code-block:: python

        import emsarray
        import pandas as pd
        from emsarray.operations import point_extraction

        ds = emsarray.tutorial.open_dataset('gbr4')
        df = pd.DataFrame({
            'lon': [152.807, 152.670, 153.543],
            'lat': [-24.9595, -24.589, -25.488],
            'name': ['a', 'b', 'c'],
        })
        point_data = point_extraction.extract_dataframe(
            ds, df, ['lon', 'lat'])
        point_data

    .. code-block:: text

        <xarray.Dataset>
        Dimensions:  (k: 47, point: 3, time: 1)
        Coordinates:
            zc       (k) float32 ...
          * time     (time) datetime64[ns] 2022-05-11T14:00:00
            lon      (point) float64 152.8 152.7 153.5
            lat      (point) float64 -24.96 -24.59 -25.49
        Dimensions without coordinates: k, point
        Data variables:
            botz     (point) float32 ...
            eta      (time, point) float32 ...
            salt     (time, k, point) float32 ...
            temp     (time, k, point) float32 ...
            name     (point) object 'a' 'b' 'c'
        Attributes: (12/14)
            ...
    """
    lon_coord, lat_coord = coordinate_columns

    # Extract the points from the dataset
    points = [
        Point(row[lon_coord], row[lat_coord])
        for i, row in dataframe.iterrows()]
    point_dataset = extract_points(dataset, points, point_dimension=point_dimension)

    # Merge in the dataframe
    point_dataset = point_dataset.merge(_dataframe_to_dataset(
        dataframe, dimension_name=point_dimension))
    point_dataset = point_dataset.set_coords(coordinate_columns)

    # Add CF attributes to the new coordinate variables
    point_dataset[lon_coord].attrs.update({
        "long_name": "longitude",
        "units": "degrees_east",
        "standard_name": "longitude",
    })
    point_dataset[lat_coord].attrs.update({
        "long_name": "latitude",
        "units": "degrees_north",
        "standard_name": "latitude",
    })

    return point_dataset
