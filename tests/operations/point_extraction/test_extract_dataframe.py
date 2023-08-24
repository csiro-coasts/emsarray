import pathlib

import numpy
import pandas as pd
import pytest
import xarray
from numpy.testing import assert_equal
from shapely.geometry import Point

from emsarray.operations import point_extraction


def test_extract_dataframe(
    datasets: pathlib.Path,
) -> None:
    num_points = 10
    names = [f'{chr(97 + i)}{i + 1}' for i in range(num_points)]
    rs = numpy.random.sample(num_points) * 3
    θs = numpy.random.sample(num_points) * 2 * numpy.pi
    xs = numpy.cos(θs) * rs
    ys = numpy.sin(θs) * rs
    points_df = pd.DataFrame({'name': names, 'lon': xs, 'lat': ys})

    in_dataset = xarray.open_dataset(datasets / 'ugrid_mesh2d.nc')
    point_dataset = point_extraction.extract_dataframe(
        in_dataset, points_df, coordinate_columns=('lon', 'lat'))

    # There should be a new dimension named 'points'
    # with the same size as the number of rows in the CSV
    assert 'point' in point_dataset.dims
    assert point_dataset.dims['point'] == num_points

    # All the columns from the CSV should have been merged in
    assert_equal(points_df['name'], point_dataset['name'].values)
    assert_equal(points_df['lon'], point_dataset['lon'].values)
    assert_equal(points_df['lat'], point_dataset['lat'].values)

    # The new point coordinate variables should have the relevant CF attributes
    assert point_dataset['lon'].attrs == {
        "long_name": "Longitude",
        "units": "degrees_east",
        "standard_name": "longitude",
    }
    assert point_dataset['lat'].attrs == {
        "long_name": "Latitude",
        "units": "degrees_north",
        "standard_name": "latitude",
    }

    # The values should be extracted from the dataset, one per point
    assert 'values' in point_dataset.data_vars
    values = point_dataset.data_vars['values']
    assert values.dims == ('point',)
    print(values.values)
    assert_equal(values.values, [
        in_dataset.ems.select_point(Point(row['lon'], row['lat']))['values'].values
        for i, row in points_df.iterrows()
    ])


def test_extract_dataframe_point_dimension(
    datasets: pathlib.Path,
) -> None:
    points_df = pd.DataFrame({
        'name': ['a', 'b', 'c', 'd'],
        'lon': [0, 1, 2, 3],
        'lat': [0, 0, 0, 0],
    })
    in_dataset = xarray.open_dataset(datasets / 'ugrid_mesh2d.nc')
    point_dataset = point_extraction.extract_dataframe(
        in_dataset, points_df, ('lon', 'lat'), point_dimension='foo')
    assert point_dataset.dims['foo'] == 4
    assert point_dataset['foo'].dims == ('foo',)
    assert_equal(point_dataset['foo'].values, [0, 1, 2, 3])
    assert point_dataset['values'].dims == ('foo',)


def test_extract_points_missing_point_error(
    datasets: pathlib.Path,
) -> None:
    points_df = pd.DataFrame({
        'name': ['a', 'b', 'c', 'd'],
        'lon': [0, 10, 1, 20],
        'lat': [0, 0, 0, 0],
    })
    in_dataset = xarray.open_dataset(datasets / 'ugrid_mesh2d.nc')
    with pytest.raises(point_extraction.NonIntersectingPoints) as exc_info:
        point_extraction.extract_dataframe(in_dataset, points_df, ('lon', 'lat'))
    exc: point_extraction.NonIntersectingPoints = exc_info.value
    assert_equal(exc.indices, [1, 3])


def test_extract_points_missing_point_drop(
    datasets: pathlib.Path,
) -> None:
    points_df = pd.DataFrame({
        'name': ['a', 'b', 'c', 'd'],
        'lon': [0, 10, 1, 20],
        'lat': [0, 0, 0, 0],
    })
    in_dataset = xarray.open_dataset(datasets / 'ugrid_mesh2d.nc')
    point_dataset = point_extraction.extract_dataframe(
        in_dataset, points_df, ('lon', 'lat'), missing_points='drop')
    assert point_dataset.dims['point'] == 2
    assert 'values' in point_dataset.data_vars
    assert_equal(point_dataset['values'].values, [
        in_dataset.ems.select_point(Point(0, 0))['values'].values,
        in_dataset.ems.select_point(Point(1, 0))['values'].values,
    ])
    assert_equal(point_dataset['point'].values, [0, 2])
    assert_equal(point_dataset['name'].values, ['a', 'c'])


def test_extract_points_missing_point_fill(
    datasets: pathlib.Path,
) -> None:
    points_df = pd.DataFrame({
        'name': ['a', 'b', 'c', 'd'],
        'lon': [0, 10, 1, 20],
        'lat': [0, 0, 0, 0],
    })
    in_dataset = xarray.open_dataset(datasets / 'ugrid_mesh2d.nc')
    point_dataset = point_extraction.extract_dataframe(
        in_dataset, points_df, ('lon', 'lat'), missing_points='fill')
    assert point_dataset.dims['point'] == 4
    assert 'values' in point_dataset.data_vars
    assert_equal(point_dataset['values'].values, [
        in_dataset.ems.select_point(Point(0, 0))['values'].values,
        numpy.nan,
        in_dataset.ems.select_point(Point(1, 0))['values'].values,
        numpy.nan,
    ])
    assert_equal(point_dataset['point'].values, [0, 1, 2, 3])
    assert_equal(point_dataset['name'].values, ['a', 'b', 'c', 'd'])
