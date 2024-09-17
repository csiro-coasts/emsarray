from __future__ import annotations

import hashlib
import pathlib

import emsarray
import emsarray.operations.cache

# Sha1
empty_sha1_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"
attr_hash_lon = "4151c4ac3e0e913dd2c5520c0b2c33e6f20eb004"
attr_hash_lat = "2cb433979fc2d9c3884eea8569dd6a44406950f3"
cache_key_hash_cf1d_sha1 = "e8642ca2ef921d066be127f518bcb65c84cba650"

# Blake2b
cache_key_hash_cf1d = "affbdeee76e1a7f0530a2847b567e5b16d3d6152f753e5cbbbf68b099e32eff6"
cache_key_hash_cf2d = "79ff732a7ce0c6461032a8b30022057640977a99333ec08e632dd00e4ec2c278"
cache_key_hash_shoc_standard = "f94917756d084db26b032feef3124b6a61af9e8bc4521600462d7f7f94a3e367"
cache_key_hash_ugrid_mesh2d = "b00260280514beba8332a3dda4ec133eda03354d5a55d13828045e2527041399"
cache_key_hash_ugrid_mesh2d_one_indexed = "2bbdcb0aa3d55d7fbf4bc962d387898a37cff36100e1a2b60624a908ebb352e9"


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


def test_cache_key_cfgrid1d_sha1(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'cfgrid1d.nc')

    result_cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf, hashlib.sha1())

    assert result_cache_key_cf is not None

    assert result_cache_key_cf == cache_key_hash_cf1d_sha1
