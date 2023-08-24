import pathlib

import numpy
import pytest
import xarray
from numpy.testing import assert_equal
from shapely.geometry import Point

from emsarray.operations import point_extraction


def test_extract_points(
    datasets: pathlib.Path,
) -> None:
    num_points = 10
    rs = numpy.random.sample(num_points) * 3
    θs = numpy.random.sample(num_points) * 2 * numpy.pi
    coords = numpy.c_[numpy.cos(θs) * rs, numpy.sin(θs) * rs]
    points = [Point(x, y) for x, y in coords]

    in_dataset = xarray.open_dataset(datasets / 'ugrid_mesh2d.nc')
    point_dataset = point_extraction.extract_points(in_dataset, points)

    assert 'point' in point_dataset.dims
    assert point_dataset.dims['point'] == num_points

    assert point_dataset.variables.keys() == {'values', 'point'}

    values = point_dataset.data_vars['values']
    assert values.dims == ('point',)
    assert_equal(values.values, [
        in_dataset.ems.select_point(point)['values'].values
        for point in points
    ])

    point_coord = point_dataset.coords['point']
    assert point_coord.dims == ('point',)
    assert_equal(point_coord.values, numpy.arange(num_points))


def test_extract_points_point_dimension(
    datasets: pathlib.Path,
) -> None:
    points = [Point(i, 0) for i in range(4)]
    in_dataset = xarray.open_dataset(datasets / 'ugrid_mesh2d.nc')
    point_dataset = point_extraction.extract_points(in_dataset, points, point_dimension='foo')
    assert point_dataset.dims['foo'] == 4
    assert point_dataset['foo'].dims == ('foo',)
    assert_equal(point_dataset['foo'].values, [0, 1, 2, 3])


def test_extract_points_missing_point_error(
    datasets: pathlib.Path,
) -> None:
    points = [Point(i, 0) for i in range(8)]
    in_dataset = xarray.open_dataset(datasets / 'ugrid_mesh2d.nc')
    with pytest.raises(point_extraction.NonIntersectingPoints) as exc_info:
        point_extraction.extract_points(in_dataset, points)
    exc: point_extraction.NonIntersectingPoints = exc_info.value
    assert_equal(exc.indices, [4, 5, 6, 7])


def test_extract_points_missing_point_drop(
    datasets: pathlib.Path,
) -> None:
    points = [Point(i, 0) for i in range(8)]
    in_dataset = xarray.open_dataset(datasets / 'ugrid_mesh2d.nc')
    point_dataset = point_extraction.extract_points(in_dataset, points, missing_points='drop')
    assert point_dataset.dims['point'] == 4
    assert_equal(point_dataset['point'].values, [0, 1, 2, 3])
