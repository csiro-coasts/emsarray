from __future__ import annotations

import hashlib
import pathlib

import emsarray
import emsarray.operations.cache

# Sha1
empty_sha1_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"
attr_hash_lon = "5a717d9e800ca6b326fcfbebef696f8deede3861"
attr_hash_lat = "506ddefdfeab4c5ec921f3518c4e905908b5a4ed"
cache_key_hash_cf1d_sha1 = "07b5deaa33d086c66730d6e2708ae03600226064"

# Blake2b
cache_key_hash_cf1d = "0cd2f2e230e1e2867347d7e5a8ac12de1dff7c89a30761fc7239d2468dc5a3a9013b3710b074bdb81dd2651fb44fb734638b803bb6e4ad0b611c6b3f94311656"
cache_key_hash_cf2d = "04c28b946c000446f22331b8decea4452a5601f0629b818e100c0e318b390b8b3a671de20fc0c94ce17abbb8bb81de08bc7949bd0e6b64adf88c4d8d1a9d911f"
cache_key_hash_shoc_standard = "738283c80dc5955db4622d5652ff7c913c523567e514d615c82e4fbd53be075d00b49c15678d16d209c4e08dc431f2f57b5f929db4d34432ccee977e1c360c74"
cache_key_hash_ugrid_mesh2d = "3451aa234b0dcbd8c0c39e856d5cd6a3bfdf2bf02d3f0787b3d5e0499fc6819fbed8b87ca791ab0daea483975bafca9b856cb21652b5e7c5cd488486e3879292"
cache_key_hash_ugrid_mesh2d_one_indexed = "3ef7e126218f84320ca2d4eb7ded448477fc42ee179dba5eba3c81b5d9dce4a3ba8431645675ff86d71a1092c81e05f1de02dd4dccac3d26b6c1ea4a3c93c42d"


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

    result_cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf, hashlib.sha1)

    assert result_cache_key_cf is not None

    assert result_cache_key_cf == cache_key_hash_cf1d_sha1
