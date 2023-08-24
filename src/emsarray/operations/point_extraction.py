"""
Subset a dataset at a set of points.

:func:`.extract_dataframe` takes a pandas :class:`~pandas.DataFrame`,
subsets the dataset at the point specified in each row,
and merges the dataset with the dataframe.
The points extracted will form the coordinates for the new dataset.

:func:`.extract_points` takes a list of :class:`Points <shapely.Point>`,
subsets the dataset at these points,
and returns a new dataset with out any associated geometry.
This is useful if you want to add your own metadata to the subset dataset.

:ref:`emsarray extract-points` is a command line interface to :func:`.extract_dataframe`.
"""
import dataclasses
from typing import Any, Hashable, List, Literal, Tuple

import numpy
import pandas
import shapely
import xarray
import xarray.core.dtypes as xrdtypes

from emsarray.conventions import Convention


@dataclasses.dataclass
class NonIntersectingPoints(ValueError):
    """
    Raised when a point to extract does not intersect the dataset geometry.
    """

    #: The indices of the points that do not intersect
    indices: numpy.ndarray

    #: The non-intersecting points
    points: List[shapely.Point]

    def __post_init__(self) -> None:
        super().__init__(f"{self.points[0].wkt} does not intersect the dataset geometry")


def _dataframe_to_dataset(
    dataframe: pandas.DataFrame,
    *,
    dimension_name: Hashable,
) -> xarray.Dataset:
    """Convert a pandas DataFrame to an xarray Dataset."""
    dataframe = dataframe.copy()
    dataframe.index.name = dimension_name
    dataset = dataframe.to_xarray()
    return dataset


def extract_points(
    dataset: xarray.Dataset,
    points: List[shapely.Point],
    *,
    point_dimension: Hashable = 'point',
    missing_points: Literal['error', 'drop'] = 'error',
) -> xarray.Dataset:
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
    points : list of :class:`shapely.Point`
        The points to select.
    point_dimension : Hashable, optional
        The name of the new dimension to index points along.
        Defaults to ``"point"``.
    errors : {'raise', 'drop'}, default 'raise'
        How to handle points which do not intersect the dataset.

        - If 'raise', a :exc:`NonIntersectingPoints` is raised.
        - If 'drop', the points are dropped from the returned dataset.

    Returns
    -------
    xarray.Dataset
        A subset of the input dataset that only contains data at the given points.
        The dataset will only contain the values, without any geometry coordinates.
        The `point_dimension` dimension will have a coordinate with the same name
        whose values match the indices of the `points` array.
        This is useful when `errors` is 'drop' to find out which points were dropped.

    See Also
    --------
    :func:`extract_dataframe`
    """
    convention: Convention = dataset.ems

    # Find the indexer for each given point
    indexes = numpy.array([convention.get_index_for_point(point) for point in points])

    if missing_points == 'error':
        out_of_bounds = numpy.flatnonzero(numpy.equal(indexes, None))  # type: ignore
        if len(out_of_bounds):
            raise NonIntersectingPoints(
                indices=out_of_bounds,
                points=[points[i] for i in out_of_bounds])

    # Make a DataFrame out of all point indexers
    selector_df = pandas.DataFrame([
        convention.selector_for_index(index.index)
        for index in indexes
        if index is not None])
    point_indexes = [i for i, index in enumerate(indexes) if index is not None]

    # Subset the dataset to the points
    point_ds = convention.drop_geometry()
    selector_ds = _dataframe_to_dataset(selector_df, dimension_name=point_dimension)
    point_ds = point_ds.isel(selector_ds)
    point_ds = point_ds.assign_coords({
        point_dimension: ([point_dimension], point_indexes),
    })
    return point_ds


def extract_dataframe(
    dataset: xarray.Dataset,
    dataframe: pandas.DataFrame,
    coordinate_columns: Tuple[str, str],
    *,
    point_dimension: Hashable = 'point',
    missing_points: Literal['error', 'drop', 'fill'] = 'error',
    fill_value: Any = xrdtypes.NA,
) -> xarray.Dataset:
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
    missing_points : {'error', 'drop', 'fill'}, default 'error'
        How to handle points that do not intersect the dataset geometry:

        - 'error' will raise a :class:`.NonIntersectingPoints` exception.
        - 'drop' will drop those points from the dataset.
        - 'fill' will include those points but all data variables
          will be filled with an appropriate fill value
          such as :data:`numpy.nan` for float values.
    fill_value
        Passed to :meth:`xarray.Dataset.merge` when `missing_points` is 'fill'.
        See the documentation for that method for all options.
        Defaults to a sensible fill value for each variables dtype.

    Returns
    -------
    xarray.Dataset
        A new dataset that only contains data at the given points,
        plus any new columns present in the dataframe.
        The `point_dimension` dimension will have a coordinate with the same name
        whose values match the row numbers of the dataframe.
        This is useful when `missing_points` is "drop" to find out which points were dropped.

    Example
    -------

    .. code-block:: python

        import emsarray
        import pandas
        from emsarray.operations import point_extraction

        ds = emsarray.tutorial.open_dataset('gbr4')
        df = pandas.DataFrame({
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
          * point    (point) int64 0 1 2
            lon      (point) float64 152.8 152.7 153.5
            lat      (point) float64 -24.96 -24.59 -25.49
        Dimensions without coordinates: k
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
    points = shapely.points(numpy.c_[dataframe[lon_coord], dataframe[lat_coord]])

    point_dataset = extract_points(
        dataset, points, point_dimension=point_dimension,
        missing_points='error' if missing_points == 'error' else 'drop')
    coord_dataset = _dataframe_to_dataset(dataframe, dimension_name=point_dimension)

    # Merge in the dataframe
    join: Literal['outer', 'inner'] = 'outer' if missing_points == 'fill' else 'inner'
    point_dataset = point_dataset.merge(coord_dataset, join=join, fill_value=fill_value)
    point_dataset = point_dataset.set_coords(coordinate_columns)

    # Add CF attributes to the new coordinate variables
    point_dataset[lon_coord].attrs.update({
        "long_name": "Longitude",
        "units": "degrees_east",
        "standard_name": "longitude",
    })
    point_dataset[lat_coord].attrs.update({
        "long_name": "Latitude",
        "units": "degrees_north",
        "standard_name": "latitude",
    })

    return point_dataset
