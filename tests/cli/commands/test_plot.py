import pathlib
from unittest import mock

import pytest
import xarray

from emsarray import Convention
from emsarray.cli import main


@pytest.mark.matplotlib
def test_plot_geometry(
    datasets: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = datasets / 'ugrid_mesh2d.nc'

    with mock.patch.object(Convention, 'plot') as plot_mock:
        main(['plot', str(dataset_path)])

    plot_mock.assert_called_once_with(None)


@pytest.mark.matplotlib
def test_plot_values(
    datasets: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = datasets / 'ugrid_mesh2d.nc'

    with mock.patch.object(Convention, 'plot') as plot_mock:
        main(['plot', str(dataset_path), 'values'])

    args, kwargs = plot_mock.call_args
    values = args[0]
    plot_mock.assert_called_once_with(values)
    assert isinstance(values, xarray.DataArray)
    assert values.name == 'values'
    assert values.dims == ('face',)
