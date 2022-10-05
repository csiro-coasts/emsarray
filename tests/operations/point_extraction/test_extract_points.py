import pathlib

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_equal
from shapely.geometry import Point

from emsarray.operations import point_extraction


def test_extract_points(
    datasets: pathlib.Path,
) -> None:
    num_points = 10
    rs = np.random.sample(num_points) * 3
    θs = np.random.sample(num_points) * 2 * np.pi
    coords = np.c_[np.cos(θs) * rs, np.sin(θs) * rs]
    points = [Point(x, y) for x, y in coords]

    in_dataset = xr.open_dataset(datasets / 'ugrid_mesh2d.nc')
    point_dataset = point_extraction.extract_points(in_dataset, points)

    assert 'point' in point_dataset.dims
    assert point_dataset.dims['point'] == num_points

    assert point_dataset.variables.keys() == {'values'}
    values = point_dataset.data_vars['values']
    assert values.dims == ('point',)
    print(values.values)
    assert_equal(values.values, [
        in_dataset.ems.select_point(point)['values'].values
        for point in points
    ])


def test_extract_points_out_of_bounds(
    datasets: pathlib.Path,
) -> None:
    points = [Point(i, 0) for i in range(8)]
    in_dataset = xr.open_dataset(datasets / 'ugrid_mesh2d.nc')
    with pytest.raises(point_extraction.NonIntersectingPoints) as exc_info:
        point_extraction.extract_points(in_dataset, points)
    exc: point_extraction.NonIntersectingPoints = exc_info.value
    assert_equal(exc.indices, [4, 5, 6, 7])
