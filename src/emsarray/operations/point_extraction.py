import dataclasses
from typing import Hashable, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from emsarray.formats import Format


@dataclasses.dataclass
class NonIntersectingPoints(ValueError):
    """
    Raised when a point to extract does not intersect the dataset geometry.
    """
    indices: np.ndarray
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
    helper: Format = dataset.ems

    # Find the indexer for each given point
    indexes = np.array([helper.get_index_for_point(point) for point in points])

    # TODO It would be nicer if out-of-bounds points were represented in the
    # output by masked values, rather than raising an error.
    out_of_bounds = np.flatnonzero(np.equal(indexes, None))  # type: ignore
    if len(out_of_bounds):
        raise NonIntersectingPoints(
            indices=out_of_bounds,
            points=[points[i] for i in out_of_bounds])

    # Make a DataFrame out of all point indexers
    selector_df = pd.DataFrame([
        helper.selector_for_index(index.index)
        for index in indexes
        if index is not None])

    # Subset the dataset to the points
    selector_ds = _dataframe_to_dataset(selector_df, dimension_name=point_dimension)
    return helper.drop_geometry().isel(selector_ds)


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
    points = [
        Point(row[coordinate_columns[0]], row[coordinate_columns[1]])
        for i, row in dataframe.iterrows()]
    point_dataset = extract_points(dataset, points, point_dimension=point_dimension)

    point_dataset = point_dataset.merge(_dataframe_to_dataset(
        dataframe, dimension_name=point_dimension))
    point_dataset = point_dataset.set_coords(coordinate_columns)

    return point_dataset
