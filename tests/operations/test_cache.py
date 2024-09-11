from __future__ import annotations

import hashlib
import pathlib

import emsarray
import emsarray.operations.cache

empty_sha1_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

attr_hash_lon = "5a717d9e800ca6b326fcfbebef696f8deede3861"
attr_hash_lat = "506ddefdfeab4c5ec921f3518c4e905908b5a4ed"

cache_key_hash_cf1d = "07b5deaa33d086c66730d6e2708ae03600226064"
cache_key_hash_cf2d = "0798cf2c2e5f875d7d9df5735f99df494773d377"
cache_key_hash_shoc_standard = "c2f32a69a6e206acb99cf2646573bb02f483c533"
cache_key_hash_ugrid_mesh2d = "48aabb57fd8e162cc2900b7a1afb193ade903e5b"
cache_key_hash_ugrid_mesh2d_one_indexed = "b72bc9106145ac6232eee30a5b2b84ce568fe552"


def test_hash_attributes(datasets: pathlib.Path):
    dataset = emsarray.open_dataset(datasets / 'cfgrid1d.nc')

    m_lon = hashlib.sha1()

    assert m_lon.hexdigest() == empty_sha1_hash

    emsarray.operations.cache.hash_attributes(m_lon, dataset.variables['lon'].attrs)

    result_hash_lon = m_lon.hexdigest()

    assert result_hash_lon is not None

    assert result_hash_lon != empty_sha1_hash

    assert result_hash_lon == attr_hash_lon


def test_cache_key_cfgrid1d(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'cfgrid1d.nc')

    result_cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf)

    assert result_cache_key_cf is not None

    assert result_cache_key_cf == cache_key_hash_cf1d


def test_cache_key_cfgrid2d(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'cfgrid2d.nc')

    result_cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf)

    assert result_cache_key_cf is not None

    assert result_cache_key_cf == cache_key_hash_cf2d


def test_cache_key_shoc_standard(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'shoc_standard.nc')

    result_cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf)

    assert result_cache_key_cf is not None

    assert result_cache_key_cf == cache_key_hash_shoc_standard


def test_cache_key_ugrid_mesh2d(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'ugrid_mesh2d.nc')

    result_cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf)

    assert result_cache_key_cf is not None

    assert result_cache_key_cf == cache_key_hash_ugrid_mesh2d


def test_cache_key_ugrid_mesh2d_one_indexed(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'ugrid_mesh2d_one_indexed.nc')

    result_cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf)

    assert result_cache_key_cf is not None

    assert result_cache_key_cf == cache_key_hash_ugrid_mesh2d_one_indexed


def test_cache_key_is_deterministic(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'cfgrid1d.nc')

    result_cache_key_cf_1 = emsarray.operations.cache.make_cache_key(dataset_cf)
    result_cache_key_cf_2 = emsarray.operations.cache.make_cache_key(dataset_cf)

    assert result_cache_key_cf_1 == result_cache_key_cf_2

    assert result_cache_key_cf_1 == cache_key_hash_cf1d


def test_hash_attributes_is_deterministic(datasets: pathlib.Path):
    dataset = emsarray.open_dataset(datasets / 'cfgrid1d.nc')

    m_lon_1 = hashlib.sha1()
    m_lon_2 = hashlib.sha1()

    emsarray.operations.cache.hash_attributes(m_lon_1, dataset.variables['lon'].attrs)
    emsarray.operations.cache.hash_attributes(m_lon_2, dataset.variables['lon'].attrs)

    result_hash_lon_1 = m_lon_1.hexdigest()
    result_hash_lon_2 = m_lon_2.hexdigest()

    assert result_hash_lon_1 == result_hash_lon_2

    assert result_hash_lon_1 == attr_hash_lon


def test_cache_key_gives_unique_keys(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'cfgrid1d.nc')
    dataset_ugrid = emsarray.open_dataset(datasets / 'ugrid_mesh2d.nc')

    result_cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf)
    result_cache_key_ugrid = emsarray.operations.cache.make_cache_key(dataset_ugrid)

    assert result_cache_key_cf is not None
    assert result_cache_key_ugrid is not None

    assert result_cache_key_cf != result_cache_key_ugrid

    assert result_cache_key_cf == cache_key_hash_cf1d
    assert result_cache_key_ugrid == cache_key_hash_ugrid_mesh2d


def test_hash_attributes_gives_unique_keys(datasets: pathlib.Path):
    dataset = emsarray.open_dataset(datasets / 'cfgrid1d.nc')

    m_lon = hashlib.sha1()
    m_lat = hashlib.sha1()

    assert m_lon.hexdigest() == empty_sha1_hash
    assert m_lat.hexdigest() == empty_sha1_hash

    emsarray.operations.cache.hash_attributes(m_lon, dataset.variables['lon'].attrs)
    emsarray.operations.cache.hash_attributes(m_lat, dataset.variables['lat'].attrs)

    result_hash_lon = m_lon.hexdigest()
    result_hash_lat = m_lat.hexdigest()

    assert result_hash_lon is not None
    assert result_hash_lat is not None

    assert result_hash_lon != result_hash_lat

    assert result_hash_lon == attr_hash_lon
    assert result_hash_lat == attr_hash_lat
