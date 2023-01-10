import pathlib
from unittest import mock

import pytest

from emsarray.cli import main
from emsarray.cli.commands import export_geometry


@pytest.mark.parametrize(
    ['expected_format', 'output_name'],
    [
        ('geojson', 'output.json'),
        ('geojson', 'output.geojson'),
        ('shapefile', 'output.shp'),
        ('wkt', 'output.wkt'),
        ('wkb', 'output.wkb'),
    ],
)
def test_export_geometry_autodetect(
    datasets: pathlib.Path,
    tmp_path: pathlib.Path,
    expected_format: str,
    output_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writers = {
        name: mock.MagicMock(spec=writer, wraps=writer)
        for name, writer in export_geometry.format_writers.items()
    }
    monkeypatch.setattr(export_geometry, 'format_writers', writers)

    in_path = datasets / 'ugrid_mesh2d.nc'
    out_path = tmp_path / output_name

    main(['export-geometry', str(in_path), str(out_path)])

    writers[expected_format].assert_called_once()
    assert out_path.exists()


@pytest.mark.parametrize(
    'format',
    ['geojson', 'shapefile', 'wkt', 'wkb'],
)
def test_export_geometry_format(
    datasets: pathlib.Path,
    tmp_path: pathlib.Path,
    format: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writers = {
        name: mock.MagicMock(spec=writer, wraps=writer)
        for name, writer in export_geometry.format_writers.items()
    }
    monkeypatch.setattr(export_geometry, 'format_writers', writers)

    in_path = datasets / 'ugrid_mesh2d.nc'
    out_path = tmp_path / 'out.blob'

    main(['export-geometry', str(in_path), str(out_path), '--format', format])

    writers[format].assert_called_once()

    # Shapefiles are particular
    if format == 'shapefile':
        out_paths = [tmp_path / f'out.{ext}' for ext in ['dbf', 'prj', 'shp', 'shx']]
        for out_path in out_paths:
            assert out_path.exists()
    else:
        assert out_path.exists()
