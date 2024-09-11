from __future__ import annotations

import hashlib
import pathlib

import emsarray
import emsarray.operations.cache

attr_hash_lon = "5a717d9e800ca6b326fcfbebef696f8deede3861"
attr_hash_lat = "506ddefdfeab4c5ec921f3518c4e905908b5a4ed"

cache_key_hash_cf = "07b5deaa33d086c66730d6e2708ae03600226064"
cache_key_hash_ugrid = "48aabb57fd8e162cc2900b7a1afb193ade903e5b"


def test_hash_attributes(datasets: pathlib.Path):
    dataset = emsarray.open_dataset(datasets / 'cfgrid1d.nc')

    m_lon = hashlib.sha1()
    m_lat = hashlib.sha1()

    emsarray.operations.cache.hash_attributes(m_lon, dataset.variables['lon'].attrs)
    emsarray.operations.cache.hash_attributes(m_lat, dataset.variables['lat'].attrs)

    resultant_hash_lon = m_lon.hexdigest()
    resultant_hash_lat = m_lat.hexdigest()

    assert resultant_hash_lon is not None
    assert resultant_hash_lat is not None

    assert resultant_hash_lon != resultant_hash_lat

    assert resultant_hash_lon == attr_hash_lon
    assert resultant_hash_lat == attr_hash_lat


def test_cache_key(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'cfgrid1d.nc')
    dataset_ugrid = emsarray.open_dataset(datasets / 'ugrid_mesh2d.nc')

    cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf)
    cache_key_ugrid = emsarray.operations.cache.make_cache_key(dataset_ugrid)

    assert cache_key_cf is not None
    assert cache_key_ugrid is not None

    assert cache_key_cf == cache_key_hash_cf
    assert cache_key_ugrid == cache_key_hash_ugrid
