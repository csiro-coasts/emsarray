import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray
from numpy.testing import assert_allclose, assert_equal
from shapely.geometry import Point

from emsarray.cli import CommandException, main


def test_extract_points(
    datasets: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    in_path = datasets / 'ugrid_mesh2d.nc'
    csv_path = tmp_path / 'points.csv'
    out_path = tmp_path / 'out.nc'

    num_points = 10
    names = [f'{chr(97 + i)}{i + 1}' for i in range(num_points)]
    rs = np.random.sample(num_points) * 3
    θs = np.random.sample(num_points) * 2 * np.pi
    xs = np.cos(θs) * rs
    ys = np.sin(θs) * rs
    points_df = pd.DataFrame({'name': names, 'lon': xs, 'lat': ys})
    points_df.to_csv(csv_path, index=False)

    main(['extract-points', str(in_path), str(csv_path), str(out_path)])

    assert out_path.exists()

    in_dataset = xarray.open_dataset(in_path)
    point_dataset = xarray.open_dataset(out_path)

    assert 'point' in point_dataset.dims
    assert point_dataset.dims['point'] == num_points
    assert_equal(points_df['name'], point_dataset['name'].values)
    assert_allclose(points_df['lon'], point_dataset['lon'].values)
    assert_allclose(points_df['lat'], point_dataset['lat'].values)

    assert point_dataset['lon'].attrs['standard_name'] == 'longitude'
    assert point_dataset['lon'].attrs['units'] == 'degrees_east'
    assert point_dataset['lat'].attrs['standard_name'] == 'latitude'
    assert point_dataset['lat'].attrs['units'] == 'degrees_north'

    assert 'values' in point_dataset.data_vars
    values = point_dataset.data_vars['values']
    assert values.dims == ('point',)
    print(values.values)
    assert_equal(values.values, [
        in_dataset.ems.select_point(Point(row['lon'], row['lat']))['values'].values
        for i, row in points_df.iterrows()
    ])


def test_extract_points_out_of_bounds(
    datasets: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    in_path = datasets / 'ugrid_mesh2d.nc'
    csv_path = tmp_path / 'points.csv'
    out_path = tmp_path / 'out.nc'

    points_df = pd.DataFrame({'lon': np.arange(10), 'lat': np.zeros(10)})
    points_df.to_csv(csv_path, index=False)
    print(points_df.iloc[[4, 5, 6, 7, 8, 9]])

    with pytest.raises(CommandException) as exc_info:
        main(['extract-points', str(in_path), str(csv_path), str(out_path)], handle_errors=False)
    exc: CommandException = exc_info.value
    assert 'Error extracting points:' in str(exc)
    assert '(total rows: 6)' in str(exc)
