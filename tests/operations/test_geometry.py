import json
import pathlib
from typing import Any

import geojson
import numpy
import pandas
import shapefile
import shapely
import shapely.geometry
import xarray
from shapely.testing import assert_geometries_equal

from emsarray.operations import geometry


def _polygons(dataset: xarray.Dataset) -> pandas.DataFrame:
    rows = [
        (linear_index, polygon)
        for linear_index, polygon in enumerate(dataset.ems.polygons)
        if polygon is not None]
    return pandas.DataFrame(rows, columns=['linear_index', 'polygon'])


def _json_roundtrip(value: Any) -> Any:
    return json.loads(json.dumps(value))


def test_write_geojson(datasets: pathlib.Path, tmp_path: pathlib.Path):
    dataset = xarray.open_dataset(datasets / 'cfgrid2d.nc')
    out_path = tmp_path / 'out.geojson'

    geometry.write_geojson(dataset, out_path)
    assert out_path.exists()

    with open(out_path, 'rb') as f:
        saved_geometry = geojson.load(f)
    assert saved_geometry['type'] == 'FeatureCollection'
    assert len(saved_geometry['features']) == numpy.count_nonzero(dataset.ems.polygons)

    polygons = _polygons(dataset)
    assert len(polygons) == len(saved_geometry['features'])
    polygons = polygons.assign(feature=saved_geometry['features'])
    for row in polygons.itertuples():
        actual_polygon = shapely.geometry.shape(row.feature['geometry'])
        assert_geometries_equal(row.polygon, actual_polygon, tolerance=1e-6)
        assert row.linear_index == row.feature['properties']['linear_index']
        assert _json_roundtrip(dataset.ems.unravel_index(row.linear_index)) \
            == row.feature['properties']['index']

    saved_geometry = shapely.from_geojson(out_path.read_bytes())
    expected_geometry = shapely.GeometryCollection([
        p for p in dataset.ems.polygons if p is not None])
    assert_geometries_equal(saved_geometry, expected_geometry, tolerance=1e-6)


def test_write_shapefile(datasets: pathlib.Path, tmp_path: pathlib.Path):
    dataset = xarray.open_dataset(datasets / 'cfgrid2d.nc')
    out_path = tmp_path / 'out.shp'

    geometry.write_shapefile(dataset, out_path)
    out_files = [tmp_path / f'out.{ext}' for ext in ['dbf', 'prj', 'shp', 'shx']]
    for out_file in out_files:
        assert out_file.exists()

    with shapefile.Reader(out_path) as shp:
        assert shp.shapeType == shapefile.POLYGON
        polygons = _polygons(dataset)
        assert len(shp) == len(polygons)
        polygons = polygons.assign(shape_record=shp.shapeRecords())
        for row in polygons.itertuples():
            assert row.shape_record.record[1] == row.linear_index
            assert json.loads(row.shape_record.record[2]) \
                == _json_roundtrip(dataset.ems.unravel_index(row.linear_index))
            actual_polygon = shapely.geometry.shape(row.shape_record.__geo_interface__['geometry']),
            assert_geometries_equal(actual_polygon, row.polygon)


def test_write_wkt(datasets: pathlib.Path, tmp_path: pathlib.Path):
    dataset = xarray.open_dataset(datasets / 'cfgrid2d.nc')
    out_path = tmp_path / 'out.wkt'

    geometry.write_wkt(dataset, out_path)
    assert out_path.exists()

    saved_geometry = shapely.from_wkt(out_path.read_text())
    assert isinstance(saved_geometry, shapely.MultiPolygon)
    expected_geometry = shapely.MultiPolygon([
        p for p in dataset.ems.polygons if p is not None])
    assert_geometries_equal(saved_geometry, expected_geometry, tolerance=1e-6)


def test_write_wkb(datasets: pathlib.Path, tmp_path: pathlib.Path):
    dataset = xarray.open_dataset(datasets / 'cfgrid2d.nc')
    out_path = tmp_path / 'out.wkb'

    geometry.write_wkb(dataset, out_path)
    assert out_path.exists()

    saved_geometry = shapely.from_wkb(out_path.read_bytes())
    assert isinstance(saved_geometry, shapely.MultiPolygon)
    expected_geometry = shapely.MultiPolygon([
        p for p in dataset.ems.polygons if p is not None])
    assert_geometries_equal(saved_geometry, expected_geometry, tolerance=1e-7)
