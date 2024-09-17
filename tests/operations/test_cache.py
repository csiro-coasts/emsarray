from __future__ import annotations

import hashlib
import pathlib

import emsarray
import emsarray.operations.cache

# Sha1
empty_sha1_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"
attr_hash_lon = "86508f0e34ae9311b4dd4643e5afbf275eaf0052"
str_hash = 'd79c41b2fb9cbfa7e81c951747609057691cb686'
int_hash = '604b8984b112bf57de911f48a75ec769dd63411f'
attr_hash_lat = "08d7bc76818a474e80847c26ed78d588374bfbc4"
cache_key_hash_cf1d_sha1 = "414050a852498ee4b57154ead45b048756004cb5"

# Blake2b
cache_key_hash_cf1d = "bf5d43b5a378ebfbd4c1204d6e088d8445b9d2fea75a8d8cf2d610f05a72f282"
cache_key_hash_cf2d = "93b2546b28b544792db65b4709a760ca3145a22329072a24d59f714ff5694c30"
cache_key_hash_shoc_standard = "d88fe0470c7e38c38b04e9e5028f3d671e855cf31ec242af8bd656cccbe373bf"
cache_key_hash_ugrid_mesh2d = "e541bf3bbe4a3ed3a64ce48efb493bef0271ff45431036168be08376a89cea57"
cache_key_hash_ugrid_mesh2d_one_indexed = "f0ebdfdff229659f81037d4ed90aa629433b0a1843ae6d8d9f0c7c732f1a9754"


def test_hash_attributes(datasets: pathlib.Path):
    dataset = emsarray.open_dataset(datasets / 'cfgrid1d.nc')

    m_lon = hashlib.sha1()

    assert m_lon.hexdigest() == empty_sha1_hash

    emsarray.operations.cache.hash_attributes(m_lon, dataset.variables['lon'].attrs)

    result_hash_lon = m_lon.hexdigest()

    assert result_hash_lon is not None

    assert result_hash_lon != empty_sha1_hash

    assert result_hash_lon == attr_hash_lon


def test_hash_string():
    hash = hashlib.sha1()

    assert hash.hexdigest() == empty_sha1_hash

    emsarray.operations.cache.hash_string(hash, "1234")

    result_hash = hash.hexdigest()

    assert result_hash is not None

    assert result_hash != empty_sha1_hash

    assert result_hash == str_hash


def test_hash_int():
    hash = hashlib.sha1()

    assert hash.hexdigest() == empty_sha1_hash

    emsarray.operations.cache.hash_int(hash, 1234)

    result_hash = hash.hexdigest()

    assert result_hash is not None

    assert result_hash != empty_sha1_hash

    assert result_hash == int_hash


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
